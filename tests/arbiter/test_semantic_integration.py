"""End-to-end: full write pipeline with SemanticArbiter handling escalations.

This is the M4 sibling of ``test_integration.py``.  The previous file
stops at ``ESCALATE`` — this one plugs the Semantic Arbiter into that
slot and shows the complete flow:

    propose → detector → rule-based resolver
        │                    │
        ├── clean            ├── idempotent / dominant / causal_ack → apply/reject
        └── conflict         └── ESCALATE
                                 │
                                 └── SemanticArbiter.resolve()
                                         ├── high confidence → apply/reject
                                         └── low confidence → human queue

We use :class:`~mwa.llm.providers.FakeProvider` for the LLM — zero
network, deterministic output — so the test is as cheap as the
rule-based integration test.
"""

from __future__ import annotations

from typing import Any

import pytest

from mwa.arbiter import (
    ConflictDetector,
    Resolution,
    ResolutionDecision,
    RuleBasedResolver,
    SemanticArbiter,
)
from mwa.harness import HarnessMap
from mwa.llm.providers import FakeProvider
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness() -> HarnessMap:
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "tone": {
                    "impact": "high",
                    "affects": ["script_language"],
                    "order": 1,
                    "description": "Overall tone",
                },
                "script_language": {
                    "impact": "medium",
                    "affects": [],
                    "order": 2,
                },
            },
            "hard_constraints": [],
        }
    )


@pytest.fixture
def world(harness: HarnessMap) -> InMemoryWorldModel:
    return InMemoryWorldModel(harness=harness)


@pytest.fixture
def detector(world: InMemoryWorldModel) -> ConflictDetector:
    return ConflictDetector(world)


@pytest.fixture
def rule_resolver() -> RuleBasedResolver:
    return RuleBasedResolver()


@pytest.fixture
def fake_llm() -> FakeProvider:
    return FakeProvider(name="fake-arbiter", model="arbiter-test")


@pytest.fixture
def semantic_arbiter(fake_llm: FakeProvider, harness: HarnessMap) -> SemanticArbiter:
    return SemanticArbiter(fake_llm, harness, auto_resolve_threshold=0.85)


# ---------------------------------------------------------------------------
# Helper — full write pipeline (rule → semantic)
# ---------------------------------------------------------------------------


def _canned_decision(
    *,
    winner: str = "proposed",
    overall_confidence: float = 0.9,
    reason: str = "LLM picked a side.",
) -> dict[str, Any]:
    scores = {
        "structural_importance": 0.7,
        "causal_depth": 0.5,
        "downstream_impact": 0.6,
        "confidence": 0.8,
    }
    return {
        "winner": winner,
        "reason": reason,
        "proposed_scores": scores,
        "existing_scores": scores,
        "overall_confidence": overall_confidence,
        "update_plan": [],
    }


async def _propose(
    *,
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    proposal: WriteProposal,
) -> tuple[str, Resolution | None]:
    """Full pipeline: detect → rule resolve → semantic on escalation."""
    conflict = await detector.check(proposal)
    if conflict is None:
        await world.apply(proposal)
        return ("applied", None)

    resolution = rule_resolver.resolve(conflict)
    if resolution.decision is ResolutionDecision.ESCALATE:
        # Hand off to the LLM arbiter.
        resolution = await semantic_arbiter.resolve(conflict)

    if resolution.decision is ResolutionDecision.APPLY_PROPOSED:
        await world.apply(proposal)
        return ("applied", resolution)
    if resolution.decision is ResolutionDecision.KEEP_EXISTING:
        await world.reject(proposal, reason=resolution.reason)
        return ("rejected", resolution)
    # Still ESCALATE after both layers — queue for human.
    return ("escalated_to_human", resolution)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def test_rule_resolves_without_calling_llm(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    fake_llm: FakeProvider,
) -> None:
    """Dominant-confidence case: rule-based handles it, LLM never runs.

    This is the whole point of keeping the two layers separate — we
    don't want to burn tokens on conflicts that have an obvious
    winner.  If this test starts calling the LLM, we've regressed.
    """
    await world.apply(
        WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.95)
    )
    # Bob's proposal has much lower confidence → dominant_confidence rule fires.
    outcome, resolution = await _propose(
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.3),
    )

    assert outcome == "rejected"
    assert resolution is not None
    assert resolution.rule_applied == "dominant_confidence"
    # The fake LLM was never called — no responses were dequeued.
    assert len(fake_llm.calls) == 0


async def test_rule_escalates_semantic_apply_proposed(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    fake_llm: FakeProvider,
) -> None:
    """Close confidence gap: rule escalates → semantic picks proposed."""
    fake_llm.enqueue_json(
        _canned_decision(
            winner="proposed",
            overall_confidence=0.9,
            reason="Proposed value aligns better with campaign goal.",
        )
    )
    await world.apply(
        WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.7)
    )
    outcome, resolution = await _propose(
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.75),
    )

    assert outcome == "applied"
    assert resolution is not None
    assert resolution.rule_applied is None  # Came from LLM, not a rule
    assert resolution.confidence == 0.9
    # Fake LLM was called exactly once.
    assert len(fake_llm.calls) == 1
    # And the world state now holds Bob's value.
    fact = await world.read("tone")
    assert fact is not None
    assert fact.episode.value == "playful"
    assert fact.episode.agent_id == "bob"


async def test_rule_escalates_semantic_keep_existing(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    fake_llm: FakeProvider,
) -> None:
    """Close confidence gap: rule escalates → semantic keeps existing."""
    fake_llm.enqueue_json(
        _canned_decision(
            winner="existing",
            overall_confidence=0.88,
            reason="Existing tone is more structurally important.",
        )
    )
    await world.apply(
        WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.7)
    )
    outcome, _ = await _propose(
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.75),
    )

    assert outcome == "rejected"
    # World still holds Alice's original value.
    fact = await world.read("tone")
    assert fact is not None
    assert fact.episode.value == "serious"


async def test_rule_escalates_semantic_also_escalates_to_human(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    fake_llm: FakeProvider,
) -> None:
    """LLM returns low overall_confidence → escalate to human.

    The runtime must NOT apply or reject a low-confidence LLM decision.
    World stays unchanged; caller sees a distinct `escalated_to_human`
    outcome they can route to a review queue.
    """
    fake_llm.enqueue_json(
        _canned_decision(
            winner="proposed",
            overall_confidence=0.5,  # well below 0.85 threshold
            reason="Genuinely ambiguous — need human input.",
        )
    )
    await world.apply(
        WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.7)
    )
    starting_version = world.version

    outcome, resolution = await _propose(
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.75),
    )

    assert outcome == "escalated_to_human"
    assert resolution is not None
    assert resolution.decision is ResolutionDecision.ESCALATE
    assert "below the auto-resolve threshold" in resolution.reason
    # World is unchanged — escalation MUST NOT silently mutate.
    assert world.version == starting_version
    fact = await world.read("tone")
    assert fact is not None
    assert fact.episode.value == "serious"


async def test_audit_scoring_survives_round_trip(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    rule_resolver: RuleBasedResolver,
    semantic_arbiter: SemanticArbiter,
    fake_llm: FakeProvider,
) -> None:
    """Semantic Resolution.scoring must land all 8 per-criterion scores
    plus overall_confidence so the audit log has complete data."""
    fake_llm.enqueue_json(
        {
            "winner": "proposed",
            "reason": "audit",
            "proposed_scores": {
                "structural_importance": 0.11,
                "causal_depth": 0.22,
                "downstream_impact": 0.33,
                "confidence": 0.44,
            },
            "existing_scores": {
                "structural_importance": 0.55,
                "causal_depth": 0.66,
                "downstream_impact": 0.77,
                "confidence": 0.88,
            },
            "overall_confidence": 0.99,
            "update_plan": [],
        }
    )
    await world.apply(
        WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.7)
    )
    _, resolution = await _propose(
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.75),
    )
    assert resolution is not None
    assert resolution.scoring is not None
    s = resolution.scoring
    assert s["proposed.structural_importance"] == 0.11
    assert s["proposed.causal_depth"] == 0.22
    assert s["proposed.downstream_impact"] == 0.33
    assert s["proposed.confidence"] == 0.44
    assert s["existing.structural_importance"] == 0.55
    assert s["existing.causal_depth"] == 0.66
    assert s["existing.downstream_impact"] == 0.77
    assert s["existing.confidence"] == 0.88
    assert s["overall_confidence"] == 0.99
