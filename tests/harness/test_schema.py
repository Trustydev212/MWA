"""Schema-level validation tests for harness.map.json."""

from __future__ import annotations

import pytest

from mwa.errors import HarnessMapError
from mwa.harness import HarnessMap
from mwa.harness.schema import HarnessSchema, ScoringWeights


def _minimal_dict() -> dict:
    return {
        "version": "1.0",
        "domain": "test",
        "nodes": {
            "root": {"impact": "critical", "affects": ["leaf"], "order": 1},
            "leaf": {"impact": "low", "affects": [], "order": 2},
        },
    }


def test_minimal_schema_loads() -> None:
    schema = HarnessSchema.model_validate(_minimal_dict())
    assert schema.version == "1.0"
    assert schema.domain == "test"
    assert set(schema.nodes) == {"root", "leaf"}
    # defaults
    assert schema.hard_constraints == []
    assert schema.conflict_resolution.default_strategy == "arbiter"


def test_unknown_node_in_affects_is_rejected() -> None:
    bad = _minimal_dict()
    bad["nodes"]["root"]["affects"] = ["nonexistent"]
    with pytest.raises(HarnessMapError, match="unknown"):
        HarnessMap.from_dict(bad)


def test_self_loop_in_affects_is_rejected() -> None:
    bad = _minimal_dict()
    bad["nodes"]["root"]["affects"] = ["root"]
    with pytest.raises(HarnessMapError, match="self-loop"):
        HarnessMap.from_dict(bad)


def test_extra_field_in_node_is_rejected() -> None:
    bad = _minimal_dict()
    bad["nodes"]["root"]["afects"] = ["leaf"]  # typo
    with pytest.raises(HarnessMapError):
        HarnessMap.from_dict(bad)


def test_scoring_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        ScoringWeights(
            structural_importance=0.5,
            causal_depth=0.5,
            downstream_impact=0.5,
            confidence=0.5,
        )


def test_default_scoring_weights_sum_to_one() -> None:
    w = ScoringWeights()
    total = w.structural_importance + w.causal_depth + w.downstream_impact + w.confidence
    assert abs(total - 1.0) < 1e-9


def test_malformed_json_raises_harness_error() -> None:
    with pytest.raises(HarnessMapError, match="malformed"):
        HarnessMap.from_json("{not valid json")


def test_load_missing_file_raises_harness_error(tmp_path) -> None:
    with pytest.raises(HarnessMapError, match="Cannot read"):
        HarnessMap.load(tmp_path / "nope.json")
