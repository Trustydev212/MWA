"""Conflict detector tests."""

from __future__ import annotations

import pytest

from mwa.arbiter import ConflictDetector
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel


@pytest.fixture
def world() -> InMemoryWorldModel:
    return InMemoryWorldModel()


@pytest.fixture
def detector(world: InMemoryWorldModel) -> ConflictDetector:
    return ConflictDetector(world)


async def test_check_returns_none_on_clean_proposal(
    world: InMemoryWorldModel, detector: ConflictDetector
) -> None:
    p = WriteProposal(agent_id="a", node="x", value=1)
    assert await detector.check(p) is None


async def test_check_returns_conflict_on_contradiction(
    world: InMemoryWorldModel, detector: ConflictDetector
) -> None:
    await world.apply(WriteProposal(agent_id="a", node="x", value=1))
    p = WriteProposal(agent_id="b", node="x", value=2)
    conflict = await detector.check(p)
    assert conflict is not None
    assert conflict.existing.value == 1
    assert conflict.proposed.value == 2


async def test_partition_splits_clean_and_conflicting(
    world: InMemoryWorldModel, detector: ConflictDetector
) -> None:
    await world.apply(WriteProposal(agent_id="a", node="conflicted", value="old"))

    proposals = [
        WriteProposal(agent_id="b", node="fresh1", value="hello"),
        WriteProposal(agent_id="b", node="conflicted", value="new"),
        WriteProposal(agent_id="b", node="fresh2", value="world"),
    ]
    clean, conflicts = await detector.partition(proposals)

    assert len(clean) == 2
    assert {p.node for p in clean} == {"fresh1", "fresh2"}
    assert len(conflicts) == 1
    assert conflicts[0].node == "conflicted"


async def test_partition_empty_input(detector: ConflictDetector) -> None:
    clean, conflicts = await detector.partition([])
    assert clean == []
    assert conflicts == []
