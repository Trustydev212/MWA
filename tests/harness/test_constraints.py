"""Hard-constraint DSL parser & evaluator tests."""

from __future__ import annotations

from mwa.harness import HarnessMap
from mwa.harness.constraints import (
    HardConstraintEvaluator,
    parse_constraint,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_value_mutex_vietnamese() -> None:
    parsed = parse_constraint("duration không thể đồng thời là 15s và 60s")
    assert parsed.kind == "value_mutex"
    assert parsed.node_a == "duration"
    assert "15s" in parsed.values
    assert "60s" in parsed.values


def test_parse_value_mutex_english() -> None:
    parsed = parse_constraint("tone cannot be both formal and casual")
    assert parsed.kind == "value_mutex"
    assert parsed.node_a == "tone"
    assert set(parsed.values) == {"formal", "casual"}


def test_parse_pair_mutex_vietnamese() -> None:
    parsed = parse_constraint("visual_style cartoon không compatible với tone corporate")
    assert parsed.kind == "pair_mutex"
    assert parsed.node_a == "visual_style"
    assert parsed.value_a == "cartoon"
    assert parsed.node_b == "tone"
    assert parsed.value_b == "corporate"


def test_parse_pair_mutex_english() -> None:
    parsed = parse_constraint("style cartoon incompatible with tone corporate")
    assert parsed.kind == "pair_mutex"
    assert parsed.value_a == "cartoon"
    assert parsed.value_b == "corporate"


def test_parse_freeform_passthrough() -> None:
    parsed = parse_constraint("scene_count must be between 3 and 10")
    assert parsed.kind == "freeform"
    assert parsed.raw == "scene_count must be between 3 and 10"


def test_parse_coerces_int_values() -> None:
    parsed = parse_constraint("duration cannot be both 15 and 60")
    assert 15 in parsed.values
    assert 60 in parsed.values


# ---------------------------------------------------------------------------
# Evaluator — pair mutex
# ---------------------------------------------------------------------------


def test_pair_mutex_violation_detected() -> None:
    ev = HardConstraintEvaluator(["visual_style cartoon không compatible với tone corporate"])
    violations = ev.evaluate({"visual_style": "cartoon", "tone": "corporate"})
    assert len(violations) == 1
    assert violations[0].node == "visual_style"
    assert "incompatible" in violations[0].reason


def test_pair_mutex_no_violation_when_only_one_side_matches() -> None:
    ev = HardConstraintEvaluator(["visual_style cartoon không compatible với tone corporate"])
    assert ev.evaluate({"visual_style": "cartoon", "tone": "playful"}) == []
    assert ev.evaluate({"visual_style": "minimal", "tone": "corporate"}) == []


def test_pair_mutex_silent_when_partial_state() -> None:
    """If we don't even know the value of one side, we can't violate."""
    ev = HardConstraintEvaluator(["visual_style cartoon không compatible với tone corporate"])
    assert ev.evaluate({"visual_style": "cartoon"}) == []


# ---------------------------------------------------------------------------
# Evaluator — value mutex (multi-valued nodes)
# ---------------------------------------------------------------------------


def test_value_mutex_on_list_node_detected() -> None:
    """A multi-valued node (e.g. tags) holding both forbidden values."""
    ev = HardConstraintEvaluator(["tone cannot be both formal and casual"])
    violations = ev.evaluate({"tone": ["formal", "casual"]})
    assert len(violations) == 1


def test_value_mutex_on_scalar_node_is_satisfied() -> None:
    """A scalar node can never violate a mutex by definition."""
    ev = HardConstraintEvaluator(["tone cannot be both formal and casual"])
    assert ev.evaluate({"tone": "formal"}) == []
    assert ev.evaluate({"tone": "casual"}) == []


def test_freeform_constraint_never_violates() -> None:
    """Freeform constraints are documentation; they never block writes."""
    ev = HardConstraintEvaluator(["scene_count must be between 3 and 10"])
    assert ev.evaluate({"scene_count": 99}) == []


def test_evaluator_returns_all_violations_not_just_first() -> None:
    ev = HardConstraintEvaluator(
        [
            "visual_style cartoon không compatible với tone corporate",
            "tone cannot be both formal and casual",
        ]
    )
    violations = ev.evaluate(
        {
            "visual_style": "cartoon",
            "tone": ["formal", "casual", "corporate"],
        }
    )
    # Both constraints fire
    assert len(violations) >= 2


# ---------------------------------------------------------------------------
# Integration with HarnessMap.validate_state
# ---------------------------------------------------------------------------


def test_harness_map_validate_state_round_trips() -> None:
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
    assert hm.validate_state({"visual_style": "minimal", "tone": "corporate"}) == []
    bad = hm.validate_state({"visual_style": "cartoon", "tone": "corporate"})
    assert len(bad) == 1
