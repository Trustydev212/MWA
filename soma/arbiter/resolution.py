"""Resolution types — the contract between resolvers and the runtime.

A :class:`Resolution` is the *answer* to a :class:`Conflict`.  Resolvers
return one of three decisions:

- ``APPLY_PROPOSED`` — the new write wins, supersede the existing fact
- ``KEEP_EXISTING`` — the existing fact wins, reject the new write
- ``ESCALATE`` — neither side dominates, send this to a higher-tier
  resolver (the LLM-based Semantic Arbiter, or eventually a human)

The runtime is responsible for *applying* the resolution against the
World Model.  Resolvers stay pure: they look at the conflict, return a
decision, never touch storage.  This makes them trivially testable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ResolutionDecision(StrEnum):
    """What the runtime should do with a conflict."""

    APPLY_PROPOSED = "apply_proposed"
    KEEP_EXISTING = "keep_existing"
    ESCALATE = "escalate"


class Resolution(BaseModel):
    """The output of a resolver for a single conflict.

    Carries enough metadata to write a useful audit log entry — every
    field except ``decision`` and ``reason`` is optional but encouraged.
    """

    model_config = ConfigDict(frozen=True)

    decision: ResolutionDecision
    reason: str
    """Human-readable explanation.  Will end up in audit logs and (for
    escalations) in the LLM prompt to the next-tier resolver."""

    rule_applied: str | None = None
    """Name of the rule that produced this decision (e.g.
    ``"dominant_confidence"``).  ``None`` for LLM resolutions."""

    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    """How sure the resolver is about its decision.  Rule-based decisions
    are usually 1.0; LLM decisions can be lower and trigger escalation."""

    scoring: dict[str, float] | None = None
    """Reserved for the Semantic Arbiter — per-criterion scores."""
