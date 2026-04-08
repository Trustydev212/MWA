"""Tests for InMemoryWorldModel — storage semantics & temporal behaviour."""

from __future__ import annotations

import asyncio

import pytest

from mwa.errors import HardConstraintViolation, WorldModelError
from mwa.harness import HarnessMap
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel, WorldModelProtocol


@pytest.fixture
def world() -> InMemoryWorldModel:
    return InMemoryWorldModel()


@pytest.fixture
def constrained_world() -> InMemoryWorldModel:
    """A world bound to a tiny harness map with one hard constraint."""
    hm = HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "tone": {"impact": "high", "affects": [], "order": 1},
                "visual_style": {"impact": "medium", "affects": [], "order": 2},
            },
            "hard_constraints": ["visual_style cartoon không compatible với tone corporate"],
        }
    )
    return InMemoryWorldModel(harness=hm)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_in_memory_satisfies_protocol(world: InMemoryWorldModel) -> None:
    """Structural typing check — if this fails the API drifted."""
    assert isinstance(world, WorldModelProtocol)


def test_initial_version_is_zero(world: InMemoryWorldModel) -> None:
    assert world.version == 0


# ---------------------------------------------------------------------------
# Read on empty world
# ---------------------------------------------------------------------------


async def test_read_unknown_node_returns_none(world: InMemoryWorldModel) -> None:
    assert await world.read("nope") is None
    assert await world.history("nope") == []


# ---------------------------------------------------------------------------
# Apply / version / history
# ---------------------------------------------------------------------------


async def test_apply_creates_current_fact(world: InMemoryWorldModel) -> None:
    proposal = WriteProposal(agent_id="a", node="tone", value="happy", confidence=0.9)
    episode = await world.apply(proposal)

    assert episode.value == "happy"
    assert world.version == 1

    fact = await world.read("tone")
    assert fact is not None
    assert fact.episode.id == episode.id
    assert fact.is_current is True
    assert fact.is_rejected is False
    assert fact.valid_to is None


async def test_apply_supersedes_previous_fact(world: InMemoryWorldModel) -> None:
    await world.apply(WriteProposal(agent_id="a", node="tone", value="happy"))
    await world.apply(WriteProposal(agent_id="b", node="tone", value="serious"))

    fact = await world.read("tone")
    assert fact is not None
    assert fact.episode.value == "serious"
    assert world.version == 2

    history = await world.history("tone")
    assert len(history) == 2
    # Old fact must have a valid_to stamp
    assert history[0].episode.value == "happy"
    assert history[0].valid_to is not None
    assert history[0].is_current is False
    # New fact is current
    assert history[1].is_current is True
    assert history[1].valid_to is None


async def test_history_chronological(world: InMemoryWorldModel) -> None:
    for value in ("a", "b", "c", "d"):
        await world.apply(WriteProposal(agent_id="x", node="n", value=value))
    history = await world.history("n")
    assert [f.episode.value for f in history] == ["a", "b", "c", "d"]


async def test_episodes_since_replays_only_newer(world: InMemoryWorldModel) -> None:
    e1 = await world.apply(WriteProposal(agent_id="a", node="x", value=1))
    snapshot_version = world.version
    e2 = await world.apply(WriteProposal(agent_id="a", node="y", value=2))
    e3 = await world.apply(WriteProposal(agent_id="a", node="z", value=3))

    new_episodes = await world.episodes_since(snapshot_version)
    new_ids = {ep.id for ep in new_episodes}
    assert e1.id not in new_ids
    assert e2.id in new_ids
    assert e3.id in new_ids


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


async def test_detect_conflict_returns_none_on_empty(world: InMemoryWorldModel) -> None:
    proposal = WriteProposal(agent_id="a", node="tone", value="happy")
    assert await world.detect_conflict(proposal) is None


async def test_detect_conflict_returns_none_on_same_value(world: InMemoryWorldModel) -> None:
    """Idempotent writes are not conflicts."""
    await world.apply(WriteProposal(agent_id="a", node="tone", value="happy"))
    assert (
        await world.detect_conflict(WriteProposal(agent_id="b", node="tone", value="happy")) is None
    )


async def test_detect_conflict_returns_object_on_contradiction(
    world: InMemoryWorldModel,
) -> None:
    await world.apply(WriteProposal(agent_id="a", node="tone", value="happy"))
    proposal = WriteProposal(agent_id="b", node="tone", value="serious", confidence=0.9)
    conflict = await world.detect_conflict(proposal)
    assert conflict is not None
    assert conflict.node == "tone"
    assert conflict.existing.value == "happy"
    assert conflict.proposed.value == "serious"


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


async def test_reject_does_not_change_current_fact(world: InMemoryWorldModel) -> None:
    await world.apply(WriteProposal(agent_id="a", node="tone", value="happy"))
    starting_version = world.version

    rejected = await world.reject(
        WriteProposal(agent_id="b", node="tone", value="sad"),
        reason="lost confidence battle",
    )

    fact = await world.read("tone")
    assert fact is not None and fact.episode.value == "happy"
    assert world.version == starting_version  # rejection does not bump version

    # Rejected episode is hidden by default but visible with include_rejected
    assert all(not f.is_rejected for f in await world.history("tone"))
    full = await world.history("tone", include_rejected=True)
    assert any(f.episode.id == rejected.id for f in full)
    rejected_fact = next(f for f in full if f.episode.id == rejected.id)
    assert rejected_fact.is_rejected is True
    assert rejected_fact.rejection_reason == "lost confidence battle"


async def test_reject_requires_reason(world: InMemoryWorldModel) -> None:
    with pytest.raises(WorldModelError, match="reason"):
        await world.reject(WriteProposal(agent_id="b", node="tone", value="sad"), reason="")


# ---------------------------------------------------------------------------
# Hard constraint enforcement
# ---------------------------------------------------------------------------


async def test_hard_constraint_violation_blocks_apply(
    constrained_world: InMemoryWorldModel,
) -> None:
    await constrained_world.apply(WriteProposal(agent_id="a", node="tone", value="corporate"))
    with pytest.raises(HardConstraintViolation, match="incompatible"):
        await constrained_world.apply(
            WriteProposal(agent_id="b", node="visual_style", value="cartoon")
        )

    # State is unchanged after the rejected attempt
    assert constrained_world.version == 1
    visual = await constrained_world.read("visual_style")
    assert visual is None


async def test_hard_constraint_allows_compatible_state(
    constrained_world: InMemoryWorldModel,
) -> None:
    await constrained_world.apply(WriteProposal(agent_id="a", node="tone", value="playful"))
    await constrained_world.apply(WriteProposal(agent_id="b", node="visual_style", value="cartoon"))
    assert constrained_world.version == 2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_applies_all_persist(world: InMemoryWorldModel) -> None:
    """Many parallel applies should all land — no lost updates.

    Each applies to a different node so there's no logical conflict.
    """

    async def writer(i: int) -> None:
        await world.apply(WriteProposal(agent_id=f"a{i}", node=f"n{i}", value=i))

    await asyncio.gather(*[writer(i) for i in range(50)])
    assert world.version == 50
    for i in range(50):
        fact = await world.read(f"n{i}")
        assert fact is not None
        assert fact.episode.value == i


async def test_concurrent_applies_to_same_node_serialise(
    world: InMemoryWorldModel,
) -> None:
    """50 racing writers to the same node — exactly one ends up current
    and the version count is exact (no double-increments / lost writes)."""

    async def writer(i: int) -> None:
        await world.apply(WriteProposal(agent_id=f"a{i}", node="x", value=i))

    await asyncio.gather(*[writer(i) for i in range(50)])
    assert world.version == 50
    history = await world.history("x")
    assert len(history) == 50
    # Exactly one current
    currents = [f for f in history if f.is_current]
    assert len(currents) == 1
