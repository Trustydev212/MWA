"""Semantic Arbiter — LLM-backed conflict resolver.

This is the *interesting* layer of MWA.  :class:`RuleBasedResolver` handles
the cheap obvious cases (idempotent writes, dominant confidence, causal
acknowledgement) and escalates everything else.  :class:`SemanticArbiter`
picks up the escalations, asks an LLM to score both sides on four
criteria, and returns a structured :class:`Resolution`.

Why keep rule-based and semantic separate?
------------------------------------------
1. **Cost.**  Every LLM call burns tokens.  If 80% of conflicts can be
   decided by a rule in microseconds, we want that 80% to skip the LLM.
2. **Measurability.**  With the split we can log "rule fired" vs
   "escalated to LLM" and tune the rule thresholds against real data.
3. **Testability.**  Rules are deterministic and offline.  The LLM path
   is exercised via :class:`~mwa.llm.providers.FakeProvider` — no
   network, no keys, no flakiness.
4. **Graceful degradation.**  If the LLM provider is down, the runtime
   can fall back to rule-only mode and keep flowing (escalations queue
   up for later human review).

Scoring model
-------------
The LLM is asked to score each proposal on four criteria (all in
``[0, 1]``):

- **structural_importance** — how many downstream nodes depend on this?
- **causal_depth** — how deeply derived is this state from upstream?
- **downstream_impact** — how much rework if this side wins?
- **confidence** — how certain was the agent when it wrote the value?

Weights come from ``HarnessMap.conflict_resolution.scoring_weights``
(the schema validates they sum to 1.0).  The Arbiter itself doesn't
compute weighted totals — it just returns the raw scores plus a
winner pick and an ``overall_confidence``.  Runtime code consumes the
:class:`Resolution`.

Confidence gating
-----------------
The LLM's ``overall_confidence`` is compared against
``auto_resolve_threshold`` (default 0.85).  Above threshold →
apply/keep directly.  Below → the Resolution is marked ESCALATE so a
human (or another higher-tier resolver) makes the final call.  This
mirrors the pattern README describes: *"auto-resolve when the Arbiter
is very confident, queue for review otherwise."*
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from mwa.arbiter.resolution import Resolution, ResolutionDecision
from mwa.llm.base import ChatOptions, Message, MessageRole
from mwa.types import Conflict

if TYPE_CHECKING:
    from mwa.harness import HarnessMap
    from mwa.llm.base import LLMProvider


# ---------------------------------------------------------------------------
# Structured output schema — what the LLM must return
# ---------------------------------------------------------------------------


class ArbiterScores(BaseModel):
    """Per-criterion scores (all bounded to ``[0, 1]``).

    These are the raw numbers the Arbiter extracts from the LLM call.
    Weighted totals are computed downstream if callers want them.
    """

    model_config = ConfigDict(frozen=True)

    structural_importance: float = Field(ge=0.0, le=1.0)
    causal_depth: float = Field(ge=0.0, le=1.0)
    downstream_impact: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class ArbiterDecision(BaseModel):
    """What the LLM must return for each conflict.

    Shape is deliberately minimal: just enough for the runtime to act on
    plus enough metadata for an audit log.  ``update_plan`` is a hint
    from the LLM about which downstream nodes need re-computation — the
    runtime uses it only as a suggestion, not a binding contract.
    """

    model_config = ConfigDict(frozen=True)

    winner: Literal["existing", "proposed"]
    reason: str
    proposed_scores: ArbiterScores
    existing_scores: ArbiterScores
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the arbiter is in its own decision.",
    )
    update_plan: list[str] = Field(
        default_factory=list,
        description="Ordered list of downstream nodes that may need re-computation if the winner's value is applied.",
    )


# ---------------------------------------------------------------------------
# SemanticArbiter
# ---------------------------------------------------------------------------


class SemanticArbiter:
    """LLM-backed conflict resolver.

    Parameters
    ----------
    provider:
        Any :class:`~mwa.llm.base.LLMProvider` — this is where the
        structured output call goes.  In production this is typically
        an ``LLMRouter`` wrapping a primary + fallback chain; in tests
        it's a :class:`~mwa.llm.providers.FakeProvider`.
    harness:
        The static :class:`~mwa.harness.HarnessMap`.  Used to build the
        subgraph context passed into the LLM prompt, and to surface
        impact metadata for the conflicted node.
    auto_resolve_threshold:
        Confidence floor for an automatic decision.  Below this, the
        resulting Resolution is marked ``ESCALATE`` regardless of who
        the LLM picked — the runtime is expected to queue it for human
        review.  Default 0.85 matches the example harness map.
    subgraph_depth:
        How many hops of downstream context to include in the prompt.
        Too shallow and the LLM misses impact; too deep and the prompt
        balloons.  Default 2 is enough for realistic harness maps.
    max_output_tokens:
        Upper bound on the LLM's reply length.  ``ArbiterDecision`` is
        a nested schema with 9 required fields plus an update_plan
        array, so we default to 2048 (well above the ~400 token typical
        serialised output) to keep weak-model routing from truncating
        mid-JSON.  Tightening this risks truncated responses → schema
        parse failures.
    """

    def __init__(
        self,
        provider: LLMProvider,
        harness: HarnessMap,
        *,
        auto_resolve_threshold: float = 0.85,
        subgraph_depth: int = 2,
        max_output_tokens: int = 2048,
    ) -> None:
        if not 0.0 <= auto_resolve_threshold <= 1.0:
            raise ValueError("auto_resolve_threshold must be in [0, 1]")
        if subgraph_depth < 0:
            raise ValueError("subgraph_depth must be >= 0")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")

        self._provider = provider
        self._harness = harness
        self._auto_resolve_threshold = auto_resolve_threshold
        self._subgraph_depth = subgraph_depth
        self._max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, conflict: Conflict) -> Resolution:
        """Ask the LLM to score the conflict and return a Resolution.

        The LLM call is a structured-output call, so the result is
        schema-validated before we even look at it.  If the provider
        raises (rate limit, auth, schema mismatch), the exception
        propagates — callers with a retry policy wrap this.
        """
        messages = self._build_prompt(conflict)
        # temperature=0 for determinism, max_tokens set high so ArbiterDecision
        # never gets truncated on a slow/verbose model.
        options = ChatOptions(temperature=0.0, max_tokens=self._max_output_tokens)
        decision: ArbiterDecision = await self._provider.structured(
            messages, ArbiterDecision, options=options
        )
        return self._decision_to_resolution(decision)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, conflict: Conflict) -> list[Message]:
        """Assemble the system + user messages for the LLM call.

        The system message is the *constitution* — it explains the
        scoring criteria, the expected output shape, and the rules.
        The user message is the *payload* — the specific conflict plus
        the subgraph context.  Keeping them separate lets the LLM
        cache the system prompt across calls if the provider supports
        prompt caching.
        """
        node_name = conflict.node
        node_schema = self._harness.get(node_name)

        # Downstream context — these are the nodes that will need to
        # react if the winner's value propagates.
        downstream = self._harness.traverse_downstream(
            node_name, depth=self._subgraph_depth
        )
        # Build a minimal subgraph that includes the conflicted node
        # plus its downstream.  This is the "only show the LLM what
        # matters" trick — keeps the prompt small and cheap.
        subgraph_nodes = [node_name, *downstream]
        subgraph = self._harness.get_subgraph(subgraph_nodes)

        weights = self._harness.conflict_resolution.scoring_weights

        system = (
            "You are the Semantic Arbiter of a Multi World Agent (MWA) runtime.\n"
            "Two agents have proposed conflicting values for the same node in a "
            "shared World Model.  Your job is to score both proposals on four "
            "criteria and pick a winner.\n"
            "\n"
            "Scoring criteria (each in [0, 1]):\n"
            "  1. structural_importance — how many downstream nodes depend on "
            "this node?  A root with 20 descendants scores ~1.0; a leaf scores ~0.0.\n"
            "  2. causal_depth — how deeply derived is this state from upstream "
            "reasoning?  A fresh user input scores ~0.0; a 5-step inference ~1.0.\n"
            "  3. downstream_impact — how much rework happens downstream if THIS "
            "side wins?  More rework = higher score (more at stake).\n"
            "  4. confidence — how certain was the agent when it wrote this value?\n"
            "\n"
            f"Weights from the harness map (for reference, you do not compute "
            f"the weighted sum yourself):\n"
            f"  structural_importance = {weights.structural_importance}\n"
            f"  causal_depth          = {weights.causal_depth}\n"
            f"  downstream_impact     = {weights.downstream_impact}\n"
            f"  confidence            = {weights.confidence}\n"
            "\n"
            "Output rules (all mandatory):\n"
            "  - winner must be exactly 'existing' or 'proposed' (lowercase string)\n"
            "  - reason must be one short paragraph explaining the decision\n"
            "  - both proposed_scores and existing_scores must be present\n"
            "  - overall_confidence in [0, 1] reflects how certain YOU are\n"
            "  - update_plan is an ordered list of downstream node names that "
            "will need re-computation if the winner's value propagates\n"
            "\n"
            "If the conflict is genuinely ambiguous, say so by setting "
            "overall_confidence low (< 0.85) — the runtime will escalate to "
            "a human instead of applying a bad decision."
        )

        user = (
            f"CONFLICT ON NODE: {node_name!r}\n"
            f"  impact:      {node_schema.impact.value}\n"
            f"  description: {node_schema.description or '(none)'}\n"
            "\n"
            f"EXISTING FACT (agent {conflict.existing.agent_id!r}):\n"
            f"  value:      {conflict.existing.value!r}\n"
            f"  confidence: {conflict.existing.confidence}\n"
            f"  written_at: {conflict.existing.timestamp.isoformat()}\n"
            "\n"
            f"PROPOSED FACT (agent {conflict.proposed.agent_id!r}):\n"
            f"  value:      {conflict.proposed.value!r}\n"
            f"  confidence: {conflict.proposed.confidence}\n"
            "\n"
            "HARNESS SUBGRAPH (conflicted node + downstream reachable within "
            f"{self._subgraph_depth} hop(s)):\n"
            f"{self._format_subgraph(subgraph)}\n"
            "\n"
            f"DIRECT DOWNSTREAM ORDER: {downstream}\n"
            "\n"
            "Score both sides and pick a winner."
        )

        return [
            Message(role=MessageRole.SYSTEM, content=system),
            Message(role=MessageRole.USER, content=user),
        ]

    @staticmethod
    def _format_subgraph(subgraph: dict[str, list[str]]) -> str:
        """Render a subgraph as a readable ASCII listing."""
        if not subgraph:
            return "  (empty — node has no downstream edges)"
        lines: list[str] = []
        for node in sorted(subgraph):
            children = subgraph[node]
            if children:
                lines.append(f"  {node} → {', '.join(sorted(children))}")
            else:
                lines.append(f"  {node}  (leaf)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Decision → Resolution mapping
    # ------------------------------------------------------------------

    def _decision_to_resolution(self, decision: ArbiterDecision) -> Resolution:
        """Turn the LLM's structured decision into a runtime Resolution.

        Mapping rules:
        - ``overall_confidence < threshold`` → ``ESCALATE`` regardless of
          which side the LLM picked.  We never auto-apply a low-confidence
          decision.
        - Otherwise → ``APPLY_PROPOSED`` or ``KEEP_EXISTING`` based on winner.
        """
        scoring = {
            "proposed.structural_importance": decision.proposed_scores.structural_importance,
            "proposed.causal_depth": decision.proposed_scores.causal_depth,
            "proposed.downstream_impact": decision.proposed_scores.downstream_impact,
            "proposed.confidence": decision.proposed_scores.confidence,
            "existing.structural_importance": decision.existing_scores.structural_importance,
            "existing.causal_depth": decision.existing_scores.causal_depth,
            "existing.downstream_impact": decision.existing_scores.downstream_impact,
            "existing.confidence": decision.existing_scores.confidence,
            "overall_confidence": decision.overall_confidence,
        }

        if decision.overall_confidence < self._auto_resolve_threshold:
            return Resolution(
                decision=ResolutionDecision.ESCALATE,
                reason=(
                    f"LLM arbiter picked {decision.winner!r} but overall_confidence "
                    f"({decision.overall_confidence:.2f}) is below the auto-resolve "
                    f"threshold ({self._auto_resolve_threshold:.2f}).  "
                    f"Escalating for review.  LLM reasoning: {decision.reason}"
                ),
                rule_applied=None,
                confidence=decision.overall_confidence,
                scoring=scoring,
            )

        if decision.winner == "proposed":
            return Resolution(
                decision=ResolutionDecision.APPLY_PROPOSED,
                reason=decision.reason,
                rule_applied=None,
                confidence=decision.overall_confidence,
                scoring=scoring,
            )
        return Resolution(
            decision=ResolutionDecision.KEEP_EXISTING,
            reason=decision.reason,
            rule_applied=None,
            confidence=decision.overall_confidence,
            scoring=scoring,
        )
