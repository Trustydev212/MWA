"""Shared dataclasses used across MWA layers.

These types are deliberately small and storage-agnostic.  They describe
*what* is being written/read/contradicted, not *how* it is persisted.

Design notes
------------
- We use ``pydantic.BaseModel`` (not stdlib dataclass) so callers get free
  runtime validation when constructing values from JSON / dict input — every
  layer in MWA needs to defend against malformed agent payloads.
- Timestamps are timezone-aware UTC.  ``datetime.utcnow()`` is naive and was
  deprecated in Python 3.12; we use ``datetime.now(UTC)`` everywhere.
- ``confidence`` is bounded to ``[0, 1]`` and validated at construction time
  to keep the Arbiter scoring math safe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp.

    Centralised so tests can monkey-patch a single symbol if they need
    deterministic timestamps.
    """
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Impact(StrEnum):
    """Structural importance of a node in the Harness Map.

    The Arbiter uses this as one input when scoring conflicts.  ``critical``
    nodes (e.g. ``campaign_goal``) win against ``low`` nodes by default.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        """Numeric weight in ``[0, 1]`` used by Arbiter scoring."""
        return {
            Impact.CRITICAL: 1.0,
            Impact.HIGH: 0.75,
            Impact.MEDIUM: 0.5,
            Impact.LOW: 0.25,
        }[self]


class Episode(BaseModel):
    """A single write event by an agent against a single node.

    Episodes are the atomic unit the World Model persists.  An episode is
    *never* mutated after creation — invalidating an old fact creates a new
    episode rather than rewriting the old one.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: _new_id("ep"))
    agent_id: str
    node: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    timestamp: datetime = Field(default_factory=_utcnow)
    causal_parents: tuple[str, ...] = Field(default_factory=tuple)
    """IDs of episodes this write was derived from (lets us trace causal chains)."""


class WriteProposal(BaseModel):
    """A pending write that has not yet been applied to the World Model."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    node: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    causal_parents: tuple[str, ...] = Field(default_factory=tuple)


class Fact(BaseModel):
    """Temporal view of an :class:`Episode` inside a :class:`WorldModel`.

    An ``Episode`` is the immutable record of *what was written*.  A ``Fact``
    is the world model's *view* of that episode at a moment in time — it
    knows when the episode became valid, whether it is still valid, and (if
    superseded) when it stopped being valid.

    The split exists so ``Episode`` can stay frozen and replayable while
    ``Fact`` carries the lifecycle metadata that the storage layer alone
    knows.  ``Fact`` is also frozen — to "update" a fact you build a new one.
    """

    model_config = ConfigDict(frozen=True)

    episode: Episode
    valid_from: datetime
    valid_to: datetime | None = None
    """``None`` means the fact is still valid right now."""

    is_current: bool = False
    """``True`` iff this is the value the World Model would return for a
    plain ``read(node)``.  At most one fact per node may be current."""

    is_rejected: bool = False
    """``True`` iff the underlying write proposal lost a conflict and was
    never made current.  Rejected facts are kept for audit but never
    returned by plain ``read(node)``."""

    rejection_reason: str | None = None


class Conflict(BaseModel):
    """A contradiction detected by the Conflict Detector.

    Carries everything the Arbiter needs to make a decision so the Arbiter
    layer never has to query the World Model itself.
    """

    model_config = ConfigDict(frozen=True)

    node: str
    existing: Episode
    proposed: WriteProposal
    detected_at: datetime = Field(default_factory=_utcnow)


class WriteResult(BaseModel):
    """Outcome of a ``world.write(...)`` call."""

    model_config = ConfigDict(frozen=True)

    status: str  # "applied" | "rejected_constraint" | "conflict_resolved"
    episode: Episode | None = None
    reason: str | None = None
