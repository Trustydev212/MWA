"""Smoke tests for the project foundation.

These exist mostly to catch packaging mistakes — if the project can't even
import, every other test will fail with the same error and the failure mode
becomes hard to read.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import mwa
from mwa.errors import (
    ConflictError,
    HardConstraintViolation,
    HarnessMapError,
    LLMProviderError,
    MWAError,
    WorldModelError,
)
from mwa.types import Conflict, Episode, Impact, WriteProposal, WriteResult


def test_package_has_version() -> None:
    assert isinstance(mwa.__version__, str)
    assert mwa.__version__.count(".") >= 1


def test_error_hierarchy() -> None:
    """Every MWA error must be catchable as ``MWAError``."""
    for cls in (
        HarnessMapError,
        HardConstraintViolation,
        ConflictError,
        LLMProviderError,
        WorldModelError,
    ):
        assert issubclass(cls, MWAError), f"{cls.__name__} must subclass MWAError"


def test_hard_constraint_violation_carries_context() -> None:
    err = HardConstraintViolation(
        "duration cannot be both 15s and 60s",
        constraint="duration_unique",
        node="duration",
        value=60,
    )
    assert err.constraint == "duration_unique"
    assert err.node == "duration"
    assert err.value == 60
    assert isinstance(err, MWAError)


def test_impact_weights_are_monotonic() -> None:
    """Higher impact must always score higher than lower impact.

    The Arbiter relies on this — if it ever stops being true the conflict
    resolution math becomes nonsense.
    """
    weights = [Impact.LOW.weight, Impact.MEDIUM.weight, Impact.HIGH.weight, Impact.CRITICAL.weight]
    assert weights == sorted(weights)
    assert all(0 <= w <= 1 for w in weights)


def test_episode_is_immutable() -> None:
    """Episodes must never be mutated — that's the whole point of temporal facts."""
    ep = Episode(agent_id="a", node="tone", value="happy")
    with pytest.raises(ValidationError):
        ep.value = "sad"  # type: ignore[misc]


def test_episode_auto_id_and_timestamp() -> None:
    ep = Episode(agent_id="a", node="tone", value="happy")
    assert ep.id.startswith("ep_")
    assert ep.timestamp.tzinfo is not None, "timestamps must be timezone-aware"
    assert 0.0 <= ep.confidence <= 1.0


def test_episode_confidence_is_validated() -> None:
    with pytest.raises(ValidationError):
        Episode(agent_id="a", node="tone", value="happy", confidence=1.5)
    with pytest.raises(ValidationError):
        Episode(agent_id="a", node="tone", value="happy", confidence=-0.1)


def test_conflict_carries_both_sides() -> None:
    existing = Episode(agent_id="a", node="tone", value="serious")
    proposed = WriteProposal(agent_id="b", node="tone", value="playful", confidence=0.9)
    conflict = Conflict(node="tone", existing=existing, proposed=proposed)
    assert conflict.existing.agent_id == "a"
    assert conflict.proposed.agent_id == "b"
    assert conflict.existing.value != conflict.proposed.value


def test_write_result_states() -> None:
    """Make sure the documented result statuses round-trip cleanly."""
    for status in ("applied", "rejected_constraint", "conflict_resolved"):
        result = WriteResult(status=status)
        assert result.status == status
