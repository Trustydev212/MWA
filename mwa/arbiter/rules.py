"""Rule-based conflict resolver — the cheap fast path.

The point of having rules at all is **measurement**.  In a real workload
we want to know: of every conflict that hits the runtime, what fraction
is so obvious that a deterministic rule can decide it correctly?  If
that number is 80%, paying for an LLM call on every conflict is wasteful.
If it's 5%, rules are theatre and we should just always call the LLM.

We start with three rules, evaluated in order; the first that fires wins:

1. **Idempotent write** — if the proposed value equals the existing
   value, this is a touch/refresh, not a conflict.  Apply.
2. **Dominant confidence** — if one side's confidence is significantly
   higher than the other (default delta 0.3), the high side wins.
3. **(reserved) Causal recency** — if the proposed write was derived
   from the existing fact (i.e. the proposer already saw it and chose
   to overwrite), accept the new write.  Implemented via
   ``causal_parents``.

Anything else **escalates**.  No silent last-write-wins fallback —
that's exactly the kind of "looks fine until production" failure mode
the whole project is trying to avoid.
"""

from __future__ import annotations

from mwa.arbiter.resolution import Resolution, ResolutionDecision
from mwa.types import Conflict


class RuleBasedResolver:
    """Deterministic, LLM-free conflict resolver.

    Parameters
    ----------
    confidence_dominance_delta:
        Minimum confidence gap that lets one side win without escalation.
        Default 0.3 — chosen as "clearly more confident, not just noise".
        We will tune this once we have real data.
    """

    def __init__(self, *, confidence_dominance_delta: float = 0.3) -> None:
        if not 0.0 <= confidence_dominance_delta <= 1.0:
            raise ValueError("confidence_dominance_delta must be in [0, 1]")
        self._delta = confidence_dominance_delta

    def resolve(self, conflict: Conflict) -> Resolution:
        """Apply rules in order; return the first decision that fires."""

        # Rule 1: idempotent write — same value is not really a conflict.
        # The conflict detector should already filter these out, but the
        # resolver is defensive in case a different detector slips one
        # through.
        if conflict.existing.value == conflict.proposed.value:
            return Resolution(
                decision=ResolutionDecision.APPLY_PROPOSED,
                reason="Proposed value is identical to existing value (idempotent write).",
                rule_applied="idempotent_write",
            )

        # Rule 2 (a): the proposer explicitly cited the existing episode
        # as a causal parent.  That means they read the current state and
        # chose to overwrite it on purpose — accept the new write.
        if conflict.existing.id in conflict.proposed.causal_parents:
            return Resolution(
                decision=ResolutionDecision.APPLY_PROPOSED,
                reason=(
                    f"Proposer ({conflict.proposed.agent_id}) read the existing "
                    f"fact and explicitly chose to overwrite it."
                ),
                rule_applied="causal_acknowledged",
            )

        # Rule 3: dominant confidence — only fires when the gap is large.
        delta = conflict.proposed.confidence - conflict.existing.confidence
        if delta >= self._delta:
            return Resolution(
                decision=ResolutionDecision.APPLY_PROPOSED,
                reason=(
                    f"Proposed confidence ({conflict.proposed.confidence:.2f}) "
                    f"dominates existing ({conflict.existing.confidence:.2f}) "
                    f"by {delta:.2f} ≥ {self._delta:.2f}."
                ),
                rule_applied="dominant_confidence",
            )
        if delta <= -self._delta:
            return Resolution(
                decision=ResolutionDecision.KEEP_EXISTING,
                reason=(
                    f"Existing confidence ({conflict.existing.confidence:.2f}) "
                    f"dominates proposed ({conflict.proposed.confidence:.2f}) "
                    f"by {-delta:.2f} ≥ {self._delta:.2f}."
                ),
                rule_applied="dominant_confidence",
            )

        # No rule fires — escalate to a higher-tier resolver (Semantic Arbiter).
        return Resolution(
            decision=ResolutionDecision.ESCALATE,
            reason=(
                f"Confidence gap ({delta:+.2f}) is below dominance threshold "
                f"(±{self._delta:.2f}); rule-based resolution declined."
            ),
            rule_applied=None,
        )
