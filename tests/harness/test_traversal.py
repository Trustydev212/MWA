"""Traversal & topological order tests for HarnessMap."""

from __future__ import annotations

from pathlib import Path

import pytest

from soma.errors import HarnessMapError
from soma.harness import HarnessMap


@pytest.fixture
def diamond_map() -> HarnessMap:
    """A diamond-shaped graph that catches duplicate-visit bugs::

           A
          / \
         B   C
          \\ /
           D
    """
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "A": {"impact": "critical", "affects": ["B", "C"], "order": 1},
                "B": {"impact": "high", "affects": ["D"], "order": 2},
                "C": {"impact": "high", "affects": ["D"], "order": 2},
                "D": {"impact": "medium", "affects": [], "order": 3},
            },
        }
    )


@pytest.fixture
def video_map() -> HarnessMap:
    """The example map from README."""
    path = Path(__file__).resolve().parents[2] / "harness_maps" / "video_production.json"
    return HarnessMap.load(path)


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------


def test_contains_and_len(diamond_map: HarnessMap) -> None:
    assert "A" in diamond_map
    assert "Z" not in diamond_map
    assert len(diamond_map) == 4


def test_get_unknown_node_raises(diamond_map: HarnessMap) -> None:
    with pytest.raises(HarnessMapError, match="Unknown node"):
        diamond_map.get("Z")


def test_affects_and_affected_by(diamond_map: HarnessMap) -> None:
    assert sorted(diamond_map.affects("A")) == ["B", "C"]
    assert diamond_map.affects("D") == []
    assert sorted(diamond_map.affected_by("D")) == ["B", "C"]
    assert diamond_map.affected_by("A") == []


# ---------------------------------------------------------------------------
# Downstream / upstream
# ---------------------------------------------------------------------------


def test_downstream_visits_each_node_once(diamond_map: HarnessMap) -> None:
    """Diamond shape — D must NOT be visited twice."""
    result = diamond_map.traverse_downstream("A")
    assert result.count("D") == 1
    assert set(result) == {"B", "C", "D"}
    # BFS order: B,C must come before D
    assert result.index("B") < result.index("D")
    assert result.index("C") < result.index("D")


def test_downstream_excludes_start_node(diamond_map: HarnessMap) -> None:
    assert "A" not in diamond_map.traverse_downstream("A")


def test_downstream_depth_zero_returns_empty(diamond_map: HarnessMap) -> None:
    assert diamond_map.traverse_downstream("A", depth=0) == []


def test_downstream_depth_one_matches_direct_affects(diamond_map: HarnessMap) -> None:
    assert sorted(diamond_map.traverse_downstream("A", depth=1)) == ["B", "C"]


def test_downstream_unknown_node_raises(diamond_map: HarnessMap) -> None:
    with pytest.raises(HarnessMapError):
        diamond_map.traverse_downstream("Z")


def test_upstream_collects_all_ancestors(diamond_map: HarnessMap) -> None:
    assert set(diamond_map.traverse_upstream("D")) == {"A", "B", "C"}
    assert diamond_map.traverse_upstream("A") == []


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------


def test_topological_order_diamond(diamond_map: HarnessMap) -> None:
    order = diamond_map.topological_order()
    assert len(order) == 4
    # A before B, C; B & C before D
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_topological_order_is_stable(diamond_map: HarnessMap) -> None:
    """Two different HarnessMap instances of the same map should produce
    the same topological order — otherwise tests downstream become flaky."""
    a = diamond_map.topological_order()
    b = diamond_map.topological_order()
    assert a == b


def test_cycle_detection() -> None:
    bad = {
        "version": "1.0",
        "domain": "test",
        "nodes": {
            "A": {"impact": "critical", "affects": ["B"], "order": 1},
            "B": {"impact": "high", "affects": ["C"], "order": 2},
            "C": {"impact": "medium", "affects": ["A"], "order": 3},  # cycle
        },
    }
    with pytest.raises(HarnessMapError, match="cycle"):
        HarnessMap.from_dict(bad)


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------


def test_subgraph_drops_outside_edges(diamond_map: HarnessMap) -> None:
    sub = diamond_map.get_subgraph(["A", "B"])
    assert set(sub) == {"A", "B"}
    # A->C is dropped because C not in subgraph; A->B kept.
    assert sub["A"] == ["B"]
    assert sub["B"] == []  # B->D dropped


def test_subgraph_unknown_node_raises(diamond_map: HarnessMap) -> None:
    with pytest.raises(HarnessMapError):
        diamond_map.get_subgraph(["A", "Z"])


# ---------------------------------------------------------------------------
# Real-world example map
# ---------------------------------------------------------------------------


def test_video_map_loads_and_is_acyclic(video_map: HarnessMap) -> None:
    assert video_map.domain == "video_ad_production"
    order = video_map.topological_order()
    assert order[0] == "campaign_goal", "campaign_goal must be the root"
    assert len(order) == len(video_map)


def test_video_map_campaign_goal_propagates_widely(video_map: HarnessMap) -> None:
    affected = video_map.traverse_downstream("campaign_goal")
    # Should reach leaf nodes through several hops
    assert "color_palette" in affected
    assert "voiceover_tone" in affected
    assert "scene_count" in affected
    # Every other node must be downstream of the root
    assert set(affected) == set(video_map) - {"campaign_goal"}
