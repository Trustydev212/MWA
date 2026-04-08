"""Rule-based resolver tests — every rule plus the escalation fallback."""

from __future__ import annotations

import pytest

from mwa.arbiter import Resolution, ResolutionDecision, RuleBasedResolver
from mwa.types import Conflict, Episode, WriteProposal


def make_conflict(
    *,
    existing_value: object,
    proposed_value: object,
    existing_conf: float = 0.5,
    proposed_conf: float = 0.5,
    causal_parents: tuple[str, ...] = (),
) -> Conflict:
    existing = Episode(
        agent_id="alice",
        node="tone",
        value=existing_value,
        confidence=existing_conf,
    )
    proposed = WriteProposal(
        agent_id="bob",
        node="tone",
        value=proposed_value,
        confidence=proposed_conf,
        causal_parents=causal_parents,
    )
    return Conflict(node="tone", existing=existing, proposed=proposed)


@pytest.fixture
def resolver() -> RuleBasedResolver:
    return RuleBasedResolver()


def test_init_validates_delta_range() -> None:
    with pytest.raises(ValueError):
        RuleBasedResolver(confidence_dominance_delta=1.5)
    with pytest.raises(ValueError):
        RuleBasedResolver(confidence_dominance_delta=-0.1)


# ---------------------------------------------------------------------------
# Rule 1 — idempotent
# ---------------------------------------------------------------------------


def test_idempotent_write_applies(resolver: RuleBasedResolver) -> None:
    """Same value on both sides — accept (it's a touch)."""
    res = resolver.resolve(make_conflict(existing_value="happy", proposed_value="happy"))
    assert res.decision == ResolutionDecision.APPLY_PROPOSED
    assert res.rule_applied == "idempotent_write"


# ---------------------------------------------------------------------------
# Rule 2 — causal acknowledgement
# ---------------------------------------------------------------------------


def test_causal_acknowledged_lets_proposer_overwrite(
    resolver: RuleBasedResolver,
) -> None:
    """If the proposer cited the existing episode as a causal parent,
    they intentionally overwrote it — accept even with low confidence."""
    existing = Episode(agent_id="a", node="tone", value="happy", confidence=0.9)
    proposed = WriteProposal(
        agent_id="b",
        node="tone",
        value="serious",
        confidence=0.5,  # lower than existing!
        causal_parents=(existing.id,),
    )
    conflict = Conflict(node="tone", existing=existing, proposed=proposed)
    res = resolver.resolve(conflict)
    assert res.decision == ResolutionDecision.APPLY_PROPOSED
    assert res.rule_applied == "causal_acknowledged"


# ---------------------------------------------------------------------------
# Rule 3 — dominant confidence
# ---------------------------------------------------------------------------


def test_dominant_proposed_confidence_wins(resolver: RuleBasedResolver) -> None:
    res = resolver.resolve(
        make_conflict(
            existing_value="happy",
            proposed_value="serious",
            existing_conf=0.4,
            proposed_conf=0.95,
        )
    )
    assert res.decision == ResolutionDecision.APPLY_PROPOSED
    assert res.rule_applied == "dominant_confidence"


def test_dominant_existing_confidence_keeps(resolver: RuleBasedResolver) -> None:
    res = resolver.resolve(
        make_conflict(
            existing_value="happy",
            proposed_value="serious",
            existing_conf=0.95,
            proposed_conf=0.4,
        )
    )
    assert res.decision == ResolutionDecision.KEEP_EXISTING
    assert res.rule_applied == "dominant_confidence"


def test_dominance_threshold_is_strict(resolver: RuleBasedResolver) -> None:
    """Default delta = 0.3.  A gap of exactly 0.3 should still fire (≥)."""
    res = resolver.resolve(
        make_conflict(
            existing_value="happy",
            proposed_value="sad",
            existing_conf=0.5,
            proposed_conf=0.8,  # gap = 0.3
        )
    )
    assert res.decision == ResolutionDecision.APPLY_PROPOSED


# ---------------------------------------------------------------------------
# Escalation fallback
# ---------------------------------------------------------------------------


def test_close_confidence_escalates(resolver: RuleBasedResolver) -> None:
    """Tiny confidence gap should NOT silently last-write-wins."""
    res = resolver.resolve(
        make_conflict(
            existing_value="happy",
            proposed_value="sad",
            existing_conf=0.7,
            proposed_conf=0.75,
        )
    )
    assert res.decision == ResolutionDecision.ESCALATE
    assert res.rule_applied is None
    assert "below dominance threshold" in res.reason


def test_resolution_is_immutable(resolver: RuleBasedResolver) -> None:
    """Resolutions should be frozen — they're audit records, not state."""
    res = resolver.resolve(make_conflict(existing_value="a", proposed_value="b"))
    with pytest.raises(Exception):  # noqa: B017 - Pydantic ValidationError
        res.decision = ResolutionDecision.APPLY_PROPOSED  # type: ignore[misc]


def test_custom_delta_changes_escalation_boundary() -> None:
    strict = RuleBasedResolver(confidence_dominance_delta=0.05)
    res = strict.resolve(
        make_conflict(
            existing_value="happy",
            proposed_value="sad",
            existing_conf=0.7,
            proposed_conf=0.8,
        )
    )
    assert res.decision == ResolutionDecision.APPLY_PROPOSED


# ---------------------------------------------------------------------------
# Resolution model
# ---------------------------------------------------------------------------


def test_resolution_decision_enum_values() -> None:
    assert ResolutionDecision.APPLY_PROPOSED.value == "apply_proposed"
    assert ResolutionDecision.KEEP_EXISTING.value == "keep_existing"
    assert ResolutionDecision.ESCALATE.value == "escalate"


def test_resolution_carries_audit_metadata() -> None:
    r = Resolution(
        decision=ResolutionDecision.APPLY_PROPOSED,
        reason="test",
        rule_applied="dominant_confidence",
        confidence=0.9,
    )
    assert r.confidence == 0.9
    assert r.rule_applied == "dominant_confidence"
