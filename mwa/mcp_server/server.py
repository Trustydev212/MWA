"""MCP protocol wiring for the MWA MCP server.

This module is the **thin** adapter that turns JSON-RPC MCP tool
calls into :mod:`mwa.mcp_server.tools` invocations.  All the
interesting logic lives in ``tools.py``; this file is just glue.

``mcp`` package is lazy-imported inside :func:`build_server` so
``import mwa.mcp_server.server`` doesn't fail on installs that
don't have the optional ``[mcp]`` extra.  Tests never need to go
through this layer — they call the pure handlers in ``tools.py``
directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mwa.errors import MWAError
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

if TYPE_CHECKING:
    from mcp.server import Server


HANDLERS = {
    "mwa_list_nodes": tool_list_nodes,
    "mwa_read_world": tool_read_world,
    "mwa_read_history": tool_read_history,
    "mwa_seed_world": tool_seed_world,
    "mwa_submit_write": tool_submit_write,
}

TOOL_DESCRIPTIONS = {
    "mwa_list_nodes": (
        "List every node in the MWA harness map with its impact, "
        "downstream edges, and description.  Call this first to "
        "discover what's writable."
    ),
    "mwa_read_world": (
        "Read the current fact for a given node.  Returns null if "
        "the node hasn't been written yet."
    ),
    "mwa_read_history": (
        "Return the full episode history for a node, optionally "
        "including rejected writes (for audit)."
    ),
    "mwa_seed_world": (
        "Seed a node from outside any registered agent (default "
        "agent_id='user', confidence=1.0)."
    ),
    "mwa_submit_write": (
        "Write a new value as a named agent.  Runs the full conflict "
        "pipeline (detect → rule → semantic arbiter).  Returns the "
        "outcome plus any resolution metadata."
    ),
}


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------


def load_runtime_from_env() -> AgentRuntime:
    """Build an :class:`AgentRuntime` from environment variables.

    Expects:
    - ``MWA_HARNESS_PATH`` — absolute or relative path to a JSON
      harness map.  Required.

    The arbiter LLM defaults to a :class:`FakeProvider` because the
    MCP server runs as a stdio subprocess under the host client — it
    has no way to prompt the host's LLM back out-of-band.  If a
    conflict triggers arbitration and the fake provider has no queued
    response, the tool call fails loudly (caller sees the error in
    their MCP client).  A future milestone will let callers register
    a real arbiter by sending a tool call that ingests their own
    provider config.
    """
    harness_path = os.environ.get("MWA_HARNESS_PATH", "").strip()
    if not harness_path:
        raise MWAError(
            "MWA_HARNESS_PATH is not set.  The MCP server needs to know "
            "which harness map to load — point it at a JSON file, e.g.\n"
            "    MWA_HARNESS_PATH=./harness_maps/openclaw_agent_builder.json\n"
            "or set it in your MCP host's server config block."
        )
    path = Path(harness_path)
    if not path.exists():
        raise MWAError(f"MWA_HARNESS_PATH points to a missing file: {path}")

    harness = HarnessMap.load(path)
    arbiter_llm = FakeProvider(name="mcp-stub-arbiter", model="stub")
    return AgentRuntime(harness=harness, arbiter_llm=arbiter_llm)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def build_server(runtime: AgentRuntime) -> Server:
    """Construct the MCP :class:`Server` bound to a runtime.

    Lazy-imports the ``mcp`` package so this function only fails if
    the user actually tries to run the server without having the
    ``[mcp]`` extra installed — module import stays cheap.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - tested via install check
        raise MWAError(
            "Running the MCP server requires the `mcp` package. "
            "Install with: `uv pip install mwa[mcp]`"
        ) from exc

    app = Server("mwa")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                inputSchema=TOOL_SCHEMAS[name],
            )
            for name in HANDLERS
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = HANDLERS.get(name)
        if handler is None:
            raise ToolError(f"unknown tool: {name!r}")
        result = await handler(runtime, arguments)
        # MCP returns content blocks; we wrap the JSON result in one
        # text block so the caller sees structured output directly.
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return app


async def run_stdio_server() -> None:
    """Block on stdio transport with a runtime built from env vars.

    Used by :mod:`mwa.mcp_server.__main__`.  Separated into its own
    function for ease of testing (tests can call
    :func:`build_server` without touching stdio).
    """
    from mcp.server.stdio import stdio_server

    runtime = load_runtime_from_env()
    app = build_server(runtime)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )
