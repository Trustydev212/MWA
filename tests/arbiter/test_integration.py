"""End-to-end: write protocol with conflict detection + rule-based resolution.

This is the smallest "real" pipeline MWA can run before the LLM Arbiter
exists.  If this test ever breaks, it means the contracts between the
World Model, Detector, and Resolver have drifted apart and need to be
re-aligned before any higher-level work proceeds.
"""

from __future__ import annotations

import pytest

from mwa.arbiter import ConflictDetector, ResolutionDecision, RuleBasedResolver
from mwa.harness import HarnessMap
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel


@pytest.fixture
def harness() -> HarnessMap:
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "tone": {"impact": "high", "affects": ["script_language"], "order": 1},
                "script_language": {"impact": "medium", "affects": [], "order": 2},
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
def resolver() -> RuleBasedResolver:
    return RuleBasedResolver()


async def _propose(
    *,
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
    proposal: WriteProposal,
) -> tuple[str, str]:
    """The minimal write protocol: detect → resolve → apply or reject.

    Returns ``(outcome, reason)`` so callers can make assertions about
    *what happened*, not just *what's left in storage*.
    """
    conflict = await detector.check(proposal)
    if conflict is None:
        await world.apply(proposal)
        return ("applied", "no conflict")

    resolution = resolver.resolve(conflict)
    if resolution.decision == ResolutionDecision.APPLY_PROPOSED:
        await world.apply(proposal)
        return ("applied_after_resolution", resolution.reason)
    if resolution.decision == ResolutionDecision.KEEP_EXISTING:
        await world.reject(proposal, reason=resolution.reason)
        return ("rejected", resolution.reason)
    # ESCALATE — for M3 we surface this as a distinct state.  M4 will
    # plug in the Semantic Arbiter here.
    return ("escalated", resolution.reason)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def test_two_agents_clean_writes_to_different_nodes(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
) -> None:
    out1, _ = await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="alice", node="tone", value="playful"),
    )
    out2, _ = await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="bob", node="script_language", value="conversational"),
    )
    assert out1 == "applied"
    assert out2 == "applied"
    assert world.version == 2


async def test_high_confidence_proposal_overrides_low_confidence_existing(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
) -> None:
    await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.4),
    )
    out, reason = await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.95),
    )

    assert out == "applied_after_resolution"
    assert "dominates" in reason
    fact = await world.read("tone")
    assert fact is not None and fact.episode.value == "playful"
    assert fact.episode.agent_id == "bob"


async def test_low_confidence_proposal_is_rejected(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
) -> None:
    await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.95),
    )
    out, _ = await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.3),
    )

    assert out == "rejected"
    fact = await world.read("tone")
    assert fact is not None and fact.episode.value == "serious"

    # And the rejected attempt is in audit history
    full = await world.history("tone", include_rejected=True)
    rejected = [f for f in full if f.is_rejected]
    assert len(rejected) == 1
    assert rejected[0].episode.value == "playful"


async def test_close_confidence_escalates_does_not_apply(
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
) -> None:
    await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="alice", node="tone", value="serious", confidence=0.7),
    )
    out, _ = await _propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="bob", node="tone", value="playful", confidence=0.75),
    )

    assert out == "escalated"
    # World is unchanged — escalation must not silently apply
    fact = await world.read("tone")
    assert fact is not None and fact.episode.value == "serious"
    assert world.version == 1
