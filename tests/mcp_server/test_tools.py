"""Unit tests for the MCP server's pure tool handlers.

These tests call the handler functions directly against a fresh
:class:`AgentRuntime`.  No MCP package import required, no stdio, no
subprocess.  The MCP protocol wiring is a separate concern tested by
:mod:`tests.mcp_server.test_server`.
"""

from __future__ import annotations

import pytest

from mwa.harness import HarnessMap
from mwa.llm.providers import FakeProvider
from mwa.mcp_server.tools import (
    TOOL_SCHEMAS,
    ToolError,
    tool_list_nodes,
    tool_read_history,
    tool_read_world,
    tool_seed_world,
    tool_submit_write,
)
from mwa.sdk import AgentRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness() -> HarnessMap:
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "root": {
                    "impact": "critical",
                    "affects": ["mid"],
                    "order": 1,
                    "description": "root node",
                },
                "mid": {
                    "impact": "high",
                    "affects": ["leaf"],
                    "order": 2,
                    "description": "",
                },
                "leaf": {
                    "impact": "medium",
                    "affects": [],
                    "order": 3,
                },
            },
        }
    )


@pytest.fixture
def runtime(harness: HarnessMap) -> AgentRuntime:
    return AgentRuntime(harness=harness, arbiter_llm=FakeProvider())


# ---------------------------------------------------------------------------
# Schema exposure
# ---------------------------------------------------------------------------


def test_every_handler_has_a_schema() -> None:
    """Sanity guard — if someone adds a handler but forgets its schema,
    or renames one without the other, the keys drift.  This catches it."""
    from mwa.mcp_server.server import HANDLERS

    assert set(HANDLERS.keys()) == set(TOOL_SCHEMAS.keys())


def test_schemas_are_valid_json_schema_shape() -> None:
    """Structural: every schema must be an object-type JSON Schema."""
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["type"] == "object", f"{name}: wrong top type"
        assert "properties" in schema, f"{name}: missing properties"
        assert schema.get("additionalProperties") is False, (
            f"{name}: must forbid extras"
        )


# ---------------------------------------------------------------------------
# mwa_list_nodes
# ---------------------------------------------------------------------------


async def test_list_nodes_returns_all_nodes_in_topo_order(
    runtime: AgentRuntime,
) -> None:
    result = await tool_list_nodes(runtime, {})
    names = [n["name"] for n in result["nodes"]]
    assert names == ["root", "mid", "leaf"]  # topological


async def test_list_nodes_carries_impact_and_description(
    runtime: AgentRuntime,
) -> None:
    result = await tool_list_nodes(runtime, {})
    root = result["nodes"][0]
    assert root["name"] == "root"
    assert root["impact"] == "critical"
    assert root["affects"] == ["mid"]
    assert root["description"] == "root node"


# ---------------------------------------------------------------------------
# mwa_read_world
# ---------------------------------------------------------------------------


async def test_read_world_returns_null_for_unset_node(
    runtime: AgentRuntime,
) -> None:
    result = await tool_read_world(runtime, {"node": "root"})
    assert result == {"node": "root", "fact": None}


async def test_read_world_returns_fact_after_seed(
    runtime: AgentRuntime,
) -> None:
    await runtime.seed_world("root", "hello")
    result = await tool_read_world(runtime, {"node": "root"})
    assert result["fact"] is not None
    assert result["fact"]["value"] == "hello"
    assert result["fact"]["agent_id"] == "user"
    assert result["fact"]["is_current"] is True


async def test_read_world_missing_node_argument_raises(
    runtime: AgentRuntime,
) -> None:
    with pytest.raises(ToolError, match="missing required argument: 'node'"):
        await tool_read_world(runtime, {})


async def test_read_world_unknown_node_raises(runtime: AgentRuntime) -> None:
    with pytest.raises(ToolError, match="unknown node"):
        await tool_read_world(runtime, {"node": "nonexistent"})


# ---------------------------------------------------------------------------
# mwa_read_history
# ---------------------------------------------------------------------------


async def test_read_history_empty_before_writes(runtime: AgentRuntime) -> None:
    result = await tool_read_history(runtime, {"node": "root"})
    assert result["episodes"] == []


async def test_read_history_returns_all_applied_episodes(
    runtime: AgentRuntime,
) -> None:
    await runtime.seed_world("root", "first")
    await runtime.submit_write(
        agent_id="alice", node="root", value="first", confidence=0.9
    )
    result = await tool_read_history(runtime, {"node": "root"})
    assert len(result["episodes"]) == 2


async def test_read_history_include_rejected_flag(
    runtime: AgentRuntime,
) -> None:
    # Seed with high confidence, then try to overwrite with low — rejected.
    await runtime.seed_world("root", "first", confidence=0.95)
    await runtime.submit_write(
        agent_id="alice", node="root", value="second", confidence=0.3
    )
    without_rejected = await tool_read_history(runtime, {"node": "root"})
    assert len(without_rejected["episodes"]) == 1
    with_rejected = await tool_read_history(
        runtime, {"node": "root", "include_rejected": True}
    )
    assert len(with_rejected["episodes"]) == 2
    rejected = [e for e in with_rejected["episodes"] if e["is_rejected"]]
    assert len(rejected) == 1
    assert rejected[0]["value"] == "second"


# ---------------------------------------------------------------------------
# mwa_seed_world
# ---------------------------------------------------------------------------


async def test_seed_world_default_agent_and_confidence(
    runtime: AgentRuntime,
) -> None:
    result = await tool_seed_world(runtime, {"node": "root", "value": 42})
    assert result["status"] == "applied"
    assert result["resolution"] is None

    fact = await runtime.world.read("root")
    assert fact is not None
    assert fact.episode.agent_id == "user"
    assert fact.episode.confidence == 1.0
    assert fact.episode.value == 42


async def test_seed_world_custom_agent_and_confidence(
    runtime: AgentRuntime,
) -> None:
    await tool_seed_world(
        runtime,
        {"node": "root", "value": "x", "agent_id": "alice", "confidence": 0.7},
    )
    fact = await runtime.world.read("root")
    assert fact is not None
    assert fact.episode.agent_id == "alice"
    assert fact.episode.confidence == 0.7


async def test_seed_world_confidence_out_of_range_raises(
    runtime: AgentRuntime,
) -> None:
    with pytest.raises(ToolError, match="confidence must be in"):
        await tool_seed_world(
            runtime, {"node": "root", "value": "x", "confidence": 1.5}
        )


async def test_seed_world_missing_value_raises(runtime: AgentRuntime) -> None:
    with pytest.raises(ToolError, match="missing required argument: 'value'"):
        await tool_seed_world(runtime, {"node": "root"})


# ---------------------------------------------------------------------------
# mwa_submit_write
# ---------------------------------------------------------------------------


async def test_submit_write_applied_on_clean_node(
    runtime: AgentRuntime,
) -> None:
    result = await tool_submit_write(
        runtime,
        {"agent_id": "alice", "node": "root", "value": "hi", "confidence": 0.9},
    )
    assert result["status"] == "applied"
    assert result["resolution"] is None


async def test_submit_write_rejected_on_dominant_confidence_loss(
    runtime: AgentRuntime,
) -> None:
    await runtime.seed_world("root", "first", confidence=0.95)
    result = await tool_submit_write(
        runtime,
        {"agent_id": "bob", "node": "root", "value": "second", "confidence": 0.3},
    )
    assert result["status"] == "rejected"
    assert result["resolution"] is not None
    assert result["resolution"]["decision"] == "keep_existing"
    assert result["resolution"]["rule_applied"] == "dominant_confidence"


async def test_submit_write_missing_agent_id_raises(
    runtime: AgentRuntime,
) -> None:
    with pytest.raises(ToolError, match="missing required argument: 'agent_id'"):
        await tool_submit_write(
            runtime, {"node": "root", "value": "x"}
        )


async def test_submit_write_unknown_node_raises(runtime: AgentRuntime) -> None:
    with pytest.raises(ToolError, match="unknown node"):
        await tool_submit_write(
            runtime,
            {"agent_id": "alice", "node": "nonexistent", "value": "x"},
        )
