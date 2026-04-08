"""Pure tool handlers for the MWA MCP server.

Each handler is an async function with a simple contract:

- Takes an :class:`~mwa.sdk.AgentRuntime` and a plain ``dict`` of
  arguments (the shape MCP callers send over the wire).
- Returns a JSON-serialisable ``dict`` (the shape MCP callers receive).
- Raises :class:`ToolError` on any caller-visible failure with a
  human-readable message — the server wrapping layer turns these
  into standard MCP error responses.

Why this file has **no** ``mcp`` package import
------------------------------------------------
1. Tests call these handlers directly against a fresh runtime with a
   :class:`FakeProvider`.  Pure functions + in-process state = fast,
   deterministic, zero-infra tests.
2. The ``mcp`` Python SDK is an optional extra.  If we imported it
   here, ``import mwa.mcp_server`` would break on a minimal install.
   Keeping the SDK import inside the server module preserves the
   lazy-loading discipline we already use for LLM providers.

Schema definitions
------------------
:data:`TOOL_SCHEMAS` holds the JSON Schema for each tool's
``inputSchema`` — the server module reads these and passes them
through to MCP's ``Tool`` objects at registration time.  Keeping the
schemas here (next to the handlers) means a typo in a field name
fails loudly at module-load rather than at runtime call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mwa.sdk import AgentRuntime


class ToolError(Exception):
    """Caller-visible tool failure.

    Raised by handlers when an argument is missing, malformed, or
    when the underlying runtime operation fails in a way the caller
    should know about (e.g. trying to read an unknown node).  The
    server wrapper turns this into a standard MCP error response.
    """


# ---------------------------------------------------------------------------
# JSON Schema definitions for every tool's inputSchema
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "mwa_list_nodes": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "mwa_read_world": {
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
                "description": "Name of the node to read.",
            },
        },
        "required": ["node"],
        "additionalProperties": False,
    },
    "mwa_read_history": {
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
                "description": "Name of the node whose history to fetch.",
            },
            "include_rejected": {
                "type": "boolean",
                "description": (
                    "If true, include writes that lost a conflict "
                    "resolution (default false)."
                ),
                "default": False,
            },
        },
        "required": ["node"],
        "additionalProperties": False,
    },
    "mwa_seed_world": {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
            "value": {
                "description": "Any JSON-serialisable value.",
            },
            "agent_id": {
                "type": "string",
                "description": "Identity of the writer. Defaults to 'user'.",
                "default": "user",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 1.0,
            },
        },
        "required": ["node", "value"],
        "additionalProperties": False,
    },
    "mwa_submit_write": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Name of the writing agent.",
            },
            "node": {"type": "string"},
            "value": {
                "description": "Any JSON-serialisable value.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.8,
            },
        },
        "required": ["agent_id", "node", "value"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require(arguments: dict[str, Any], key: str) -> Any:
    """Extract a required argument or raise :class:`ToolError`."""
    if key not in arguments:
        raise ToolError(f"missing required argument: {key!r}")
    return arguments[key]


def _serialize_fact(fact: Any) -> dict[str, Any]:
    """Render an :class:`~mwa.types.Fact` as a plain dict.

    We avoid ``.model_dump()`` here because Fact contains a nested
    Episode with ``datetime`` fields — JSON-serialisation needs those
    formatted as ISO strings explicitly.
    """
    episode = fact.episode
    return {
        "node": episode.node,
        "value": episode.value,
        "agent_id": episode.agent_id,
        "confidence": episode.confidence,
        "timestamp": episode.timestamp.isoformat(),
        "episode_id": episode.id,
        "causal_parents": list(episode.causal_parents),
        "valid_from": fact.valid_from.isoformat(),
        "valid_to": fact.valid_to.isoformat() if fact.valid_to is not None else None,
        "is_current": fact.is_current,
        "is_rejected": fact.is_rejected,
        "rejection_reason": fact.rejection_reason,
    }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def tool_list_nodes(
    runtime: AgentRuntime, arguments: dict[str, Any]
) -> dict[str, Any]:
    """List every node declared in the runtime's harness map.

    Returns ``{"nodes": [{"name": str, "impact": str, ...}, ...]}``.
    The LLM caller uses this to discover what's writable before
    attempting :func:`tool_submit_write`.
    """
    nodes = []
    for name in runtime.harness.topological_order():
        schema = runtime.harness.get(name)
        nodes.append(
            {
                "name": name,
                "impact": schema.impact.value,
                "affects": list(schema.affects),
                "order": schema.order,
                "description": schema.description,
            }
        )
    return {"nodes": nodes}


async def tool_read_world(
    runtime: AgentRuntime, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Return the current fact for a node, or ``null`` if unset."""
    node = _require(arguments, "node")
    if node not in runtime.harness:
        raise ToolError(
            f"unknown node {node!r}. Call mwa_list_nodes to see valid names."
        )
    fact = await runtime.world.read(node)
    if fact is None:
        return {"node": node, "fact": None}
    return {"node": node, "fact": _serialize_fact(fact)}


async def tool_read_history(
    runtime: AgentRuntime, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Return the full episode history for a node."""
    node = _require(arguments, "node")
    include_rejected = bool(arguments.get("include_rejected", False))
    if node not in runtime.harness:
        raise ToolError(
            f"unknown node {node!r}. Call mwa_list_nodes to see valid names."
        )
    history = await runtime.world.history(node, include_rejected=include_rejected)
    return {
        "node": node,
        "include_rejected": include_rejected,
        "episodes": [_serialize_fact(f) for f in history],
    }


async def tool_seed_world(
    runtime: AgentRuntime, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Seed a node from outside any registered agent.

    Thin wrapper around :meth:`AgentRuntime.seed_world`.  Returns the
    resulting ``WriteOutcome`` plus any Resolution produced.
    """
    node = _require(arguments, "node")
    if node not in runtime.harness:
        raise ToolError(
            f"unknown node {node!r}. Call mwa_list_nodes to see valid names."
        )
    value = _require(arguments, "value")
    agent_id = str(arguments.get("agent_id", "user"))
    confidence = float(arguments.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ToolError(f"confidence must be in [0, 1], got {confidence}")

    status, resolution = await runtime.seed_world(
        node,
        value,
        agent_id=agent_id,
        confidence=confidence,
    )
    return {
        "status": status,
        "resolution": (
            None
            if resolution is None
            else {
                "decision": resolution.decision.value,
                "reason": resolution.reason,
                "rule_applied": resolution.rule_applied,
                "confidence": resolution.confidence,
                "scoring": resolution.scoring,
            }
        ),
    }


async def tool_submit_write(
    runtime: AgentRuntime, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Write a new value from a named agent through the full pipeline.

    Runs conflict detection + rule-based resolution + semantic
    arbitration (if needed).  Returns the outcome plus any resolution
    metadata.  Does NOT drive the dispatcher — callers that want
    reactions should additionally invoke ``run_until_idle`` via a
    separate tool call (future milestone).
    """
    agent_id = _require(arguments, "agent_id")
    node = _require(arguments, "node")
    value = _require(arguments, "value")
    confidence = float(arguments.get("confidence", 0.8))
    if not 0.0 <= confidence <= 1.0:
        raise ToolError(f"confidence must be in [0, 1], got {confidence}")
    if node not in runtime.harness:
        raise ToolError(
            f"unknown node {node!r}. Call mwa_list_nodes to see valid names."
        )

    status, resolution = await runtime.submit_write(
        agent_id=str(agent_id),
        node=node,
        value=value,
        confidence=confidence,
    )
    return {
        "status": status,
        "resolution": (
            None
            if resolution is None
            else {
                "decision": resolution.decision.value,
                "reason": resolution.reason,
                "rule_applied": resolution.rule_applied,
                "confidence": resolution.confidence,
                "scoring": resolution.scoring,
            }
        ),
    }
