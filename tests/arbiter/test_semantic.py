"""Unit tests for :class:`~mwa.arbiter.SemanticArbiter`.

Every test here uses :class:`~mwa.llm.providers.FakeProvider` — no network,
no API keys, deterministic output.  The Fake provider is queued with
JSON that matches :class:`ArbiterDecision`, and we assert on the
resulting :class:`Resolution` plus the prompt that was sent.

Scope of these tests:

1. **Decision mapping** — winner=proposed/existing with high confidence
   → APPLY_PROPOSED / KEEP_EXISTING.
2. **Confidence gating** — low overall_confidence → ESCALATE regardless
   of the LLM's winner pick.
3. **Prompt construction** — the user message must contain the node
   name, both agent values, the subgraph, and the downstream list; the
   system message must list the scoring criteria and the weights from
   the harness map.
4. **Scoring propagation** — every per-criterion score from the LLM
   must land in ``Resolution.scoring``.
5. **Validation errors** — constructor rejects out-of-range thresholds
   and negative subgraph depths.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from mwa.arbiter import ArbiterDecision, Resolution, ResolutionDecision, SemanticArbiter
from mwa.harness import HarnessMap
from mwa.llm.providers import FakeProvider
from mwa.types import Conflict, Episode, WriteProposal

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_harness() -> HarnessMap:
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "campaign_goal": {
                    "impact": "critical",
                    "affects": ["tone", "duration"],
                    "order": 1,
                    "description": "Top-level campaign objective",
                },
                "tone": {
                    "impact": "high",
                    "affects": ["script_language", "visual_style"],
                    "order": 2,
                    "description": "Overall video tone",
                },
                "duration": {
                    "impact": "high",
                    "affects": ["scene_count"],
                    "order": 2,
                },
                "script_language": {
                    "impact": "medium",
                    "affects": [],
                    "order": 3,
                },
                "visual_style": {
                    "impact": "medium",
                    "affects": [],
                    "order": 3,
                },
                "scene_count": {
                    "impact": "medium",
                    "affects": [],
                    "order": 3,
                },
            },
        }
    )


def _build_conflict(
    *,
    node: str = "tone",
    existing_value: Any = "serious",
    proposed_value: Any = "playful",
    existing_conf: float = 0.7,
    proposed_conf: float = 0.75,
) -> Conflict:
    existing = Episode(
        agent_id="alice",
        node=node,
        value=existing_value,
        confidence=existing_conf,
    )
    proposed = WriteProposal(
        agent_id="bob",
        node=node,
        value=proposed_value,
        confidence=proposed_conf,
    )
    return Conflict(node=node, existing=existing, proposed=proposed)


def _canned_decision(
    *,
    winner: str = "proposed",
    overall_confidence: float = 0.9,
    reason: str = "Proposed value is a better fit for the campaign goal.",
    update_plan: list[str] | None = None,
    proposed_scores: dict[str, float] | None = None,
    existing_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a valid JSON dict for ArbiterDecision.

    Centralised so every test uses the same baseline and only overrides
    what it cares about.
    """
    default_scores = {
        "structural_importance": 0.8,
        "causal_depth": 0.6,
        "downstream_impact": 0.7,
        "confidence": 0.9,
    }
    return {
        "winner": winner,
        "reason": reason,
        "proposed_scores": proposed_scores or default_scores,
        "existing_scores": existing_scores
        or {
            "structural_importance": 0.4,
            "causal_depth": 0.5,
            "downstream_impact": 0.3,
            "confidence": 0.6,
        },
        "overall_confidence": overall_confidence,
        "update_plan": update_plan or ["script_language", "visual_style"],
    }


@pytest.fixture
def harness() -> HarnessMap:
    return _build_harness()


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(name="fake-arbiter", model="arbiter-v1")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_threshold_out_of_range_raises(harness: HarnessMap, provider: FakeProvider) -> None:
    with pytest.raises(ValueError, match="auto_resolve_threshold"):
        SemanticArbiter(provider, harness, auto_resolve_threshold=1.5)
    with pytest.raises(ValueError, match="auto_resolve_threshold"):
        SemanticArbiter(provider, harness, auto_resolve_threshold=-0.1)


def test_negative_subgraph_depth_raises(harness: HarnessMap, provider: FakeProvider) -> None:
    with pytest.raises(ValueError, match="subgraph_depth"):
        SemanticArbiter(provider, harness, subgraph_depth=-1)


# ---------------------------------------------------------------------------
# Decision mapping — proposed wins
# ---------------------------------------------------------------------------


async def test_proposed_wins_high_confidence_applies(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    provider.enqueue_json(_canned_decision(winner="proposed", overall_confidence=0.92))
    arbiter = SemanticArbiter(provider, harness, auto_resolve_threshold=0.85)

    result: Resolution = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.APPLY_PROPOSED
    assert result.rule_applied is None  # semantic, not rule-based
    assert result.confidence == 0.92
    assert "Proposed" in result.reason
    # Every per-criterion score landed in the resolution.scoring dict.
    assert result.scoring is not None
    assert result.scoring["proposed.structural_importance"] == 0.8
    assert result.scoring["existing.confidence"] == 0.6
    assert result.scoring["overall_confidence"] == 0.92


async def test_existing_wins_high_confidence_keeps(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    provider.enqueue_json(
        _canned_decision(
            winner="existing",
            overall_confidence=0.9,
            reason="Existing state is more structurally important.",
        )
    )
    arbiter = SemanticArbiter(provider, harness)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.KEEP_EXISTING
    assert "structurally important" in result.reason
    assert result.confidence == 0.9


# ---------------------------------------------------------------------------
# Confidence gating — low overall_confidence escalates
# ---------------------------------------------------------------------------


async def test_low_confidence_escalates_even_when_proposed_wins(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    """Below the auto-resolve threshold, the Arbiter MUST escalate —
    applying a low-confidence decision is exactly the failure mode
    the semantic layer exists to prevent."""
    provider.enqueue_json(
        _canned_decision(winner="proposed", overall_confidence=0.6)
    )
    arbiter = SemanticArbiter(provider, harness, auto_resolve_threshold=0.85)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.ESCALATE
    assert "below the auto-resolve threshold" in result.reason
    assert "0.60" in result.reason  # confidence is rendered in reason
    # scoring should still be populated — audit trail needs it.
    assert result.scoring is not None
    assert result.scoring["overall_confidence"] == 0.6


async def test_low_confidence_escalates_even_when_existing_wins(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    provider.enqueue_json(
        _canned_decision(winner="existing", overall_confidence=0.5)
    )
    arbiter = SemanticArbiter(provider, harness, auto_resolve_threshold=0.85)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.ESCALATE


async def test_threshold_boundary_applies_at_exact_equal(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    """confidence == threshold should APPLY, not escalate (>= semantics)."""
    provider.enqueue_json(
        _canned_decision(winner="proposed", overall_confidence=0.85)
    )
    arbiter = SemanticArbiter(provider, harness, auto_resolve_threshold=0.85)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.APPLY_PROPOSED


async def test_custom_threshold_respected(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    provider.enqueue_json(
        _canned_decision(winner="proposed", overall_confidence=0.7)
    )
    # Strict arbiter — 0.7 is not enough.
    arbiter = SemanticArbiter(provider, harness, auto_resolve_threshold=0.9)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.ESCALATE


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


async def test_prompt_includes_node_values_and_subgraph(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    provider.enqueue_json(_canned_decision())
    arbiter = SemanticArbiter(provider, harness)

    await arbiter.resolve(_build_conflict(node="tone"))

    # FakeProvider records every call — inspect the messages actually sent.
    assert len(provider.calls) == 1
    messages, _opts = provider.calls[0]
    roles = [m.role.value for m in messages]
    assert roles[0] == "system"
    assert roles[1] == "user"

    system = messages[0].content
    user = messages[1].content

    # System message must explain the criteria and weights
    assert "structural_importance" in system
    assert "causal_depth" in system
    assert "downstream_impact" in system
    assert "winner" in system
    assert "existing" in system
    assert "proposed" in system
    # Weights from harness should be referenced
    assert "0.35" in system  # structural_importance weight
    assert "0.15" in system  # confidence weight

    # User message must include the conflict specifics
    assert "'tone'" in user or "tone" in user
    assert "alice" in user
    assert "bob" in user
    assert "'serious'" in user
    assert "'playful'" in user
    # Subgraph should mention downstream nodes (2 hops from tone)
    assert "script_language" in user
    assert "visual_style" in user


async def test_prompt_uses_subgraph_depth(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    """Depth=1 should only include direct downstream, not 2-hop."""
    provider.enqueue_json(_canned_decision())
    arbiter = SemanticArbiter(provider, harness, subgraph_depth=1)

    # campaign_goal → [tone, duration]; depth=1 should not reach scene_count
    await arbiter.resolve(_build_conflict(node="campaign_goal"))

    user = provider.calls[0][0][1].content
    assert "tone" in user
    assert "duration" in user
    # scene_count is reached only at depth >=2 (duration → scene_count)
    assert "scene_count" not in user


async def test_prompt_handles_leaf_node_gracefully(
    harness: HarnessMap, provider: FakeProvider
) -> None:
    """Leaf nodes have no downstream — prompt must still build without error."""
    provider.enqueue_json(_canned_decision())
    arbiter = SemanticArbiter(provider, harness)

    await arbiter.resolve(_build_conflict(node="scene_count"))

    user = provider.calls[0][0][1].content
    assert "scene_count" in user
    # Empty downstream list should be rendered (not crash)
    assert "DIRECT DOWNSTREAM ORDER" in user


# ---------------------------------------------------------------------------
# Provider integration — structured() is called with the right schema
# ---------------------------------------------------------------------------


async def test_uses_structured_call_with_arbiter_decision_schema(
    harness: HarnessMap,
) -> None:
    """Verify the provider's structured() is what gets called, not chat().

    We build a provider that records which API was used by tracking calls
    and by relying on the fact that structured() goes through chat()
    internally on FakeProvider but parses JSON.  The key assertion is
    that the returned Resolution mirrors the JSON we queued.
    """
    provider = FakeProvider()
    provider.enqueue_json(
        _canned_decision(
            winner="proposed",
            overall_confidence=0.91,
            reason="Marker string xyz-42",
        )
    )
    arbiter = SemanticArbiter(provider, harness)

    result = await arbiter.resolve(_build_conflict())

    assert result.decision is ResolutionDecision.APPLY_PROPOSED
    assert result.reason == "Marker string xyz-42"
    assert result.confidence == 0.91


# ---------------------------------------------------------------------------
# ArbiterDecision schema edge cases
# ---------------------------------------------------------------------------


def test_arbiter_decision_winner_must_be_literal() -> None:
    """winner is a Literal['existing', 'proposed'] — typos rejected."""
    with pytest.raises(ValidationError):
        ArbiterDecision.model_validate(
            {
                "winner": "alice",  # not a literal value
                "reason": "x",
                "proposed_scores": {
                    "structural_importance": 0.5,
                    "causal_depth": 0.5,
                    "downstream_impact": 0.5,
                    "confidence": 0.5,
                },
                "existing_scores": {
                    "structural_importance": 0.5,
                    "causal_depth": 0.5,
                    "downstream_impact": 0.5,
                    "confidence": 0.5,
                },
                "overall_confidence": 0.9,
            }
        )


def test_arbiter_decision_scores_clamped_to_unit_interval() -> None:
    """Scores out of [0, 1] must be rejected at parse time."""
    bad = {
        "winner": "proposed",
        "reason": "x",
        "proposed_scores": {
            "structural_importance": 1.2,  # out of range
            "causal_depth": 0.5,
            "downstream_impact": 0.5,
            "confidence": 0.5,
        },
        "existing_scores": {
            "structural_importance": 0.5,
            "causal_depth": 0.5,
            "downstream_impact": 0.5,
            "confidence": 0.5,
        },
        "overall_confidence": 0.9,
    }
    with pytest.raises(ValidationError):
        ArbiterDecision.model_validate(bad)


def test_arbiter_decision_round_trips_json() -> None:
    """Round-trip via JSON to catch any serialization bugs."""
    payload = _canned_decision()
    decision = ArbiterDecision.model_validate(payload)
    rebuilt = ArbiterDecision.model_validate(json.loads(decision.model_dump_json()))
    assert rebuilt.winner == decision.winner
    assert rebuilt.overall_confidence == decision.overall_confidence
    assert rebuilt.update_plan == decision.update_plan
