"""MCP server — expose an MWA world over Model Context Protocol.

The MCP server lets any MCP-compatible host (Claude Code, Cursor,
Codex, custom clients, …) talk to an MWA runtime as a first-class
tool provider.  An LLM running inside the host can seed the world,
read facts, write proposals, and inspect audit history — all through
MCP's standard JSON-RPC surface.

Install with::

    uv pip install mwa[mcp]

Run with::

    MWA_HARNESS_PATH=./harness_maps/openclaw_agent_builder.json \\
        python -m mwa.mcp_server

Or register the entry point with your MCP host by adding this to
``~/.config/claude/mcp_servers.json`` (example)::

    {
        "mcp_servers": {
            "mwa": {
                "command": "mwa-mcp",
                "env": {
                    "MWA_HARNESS_PATH": "/absolute/path/to/harness.json"
                }
            }
        }
    }

Module layout
-------------
- :mod:`mwa.mcp_server.tools` — pure Python handler functions that
  take a runtime plus arguments and return a plain JSON-serialisable
  dict.  **No mcp-package dependency** — these are importable and
  testable without installing the MCP SDK.
- :mod:`mwa.mcp_server.server` — MCP protocol wiring.  Lazy-imports
  the ``mcp`` package; only needed when actually running a server.
- :mod:`mwa.mcp_server.__main__` — CLI entry point
  (``python -m mwa.mcp_server``).

Testability
-----------
All the interesting logic lives in :mod:`~mwa.mcp_server.tools`.  Tests
call those functions directly against a fresh :class:`AgentRuntime`
with a :class:`FakeProvider` — no MCP process, no stdio, no network.
The protocol wiring in :mod:`server` is thin glue and is exercised
separately.
"""

from mwa.mcp_server.tools import (
    TOOL_SCHEMAS,
    ToolError,
    tool_list_nodes,
    tool_read_history,
    tool_read_world,
    tool_seed_world,
    tool_submit_write,
)

__all__ = [
    "TOOL_SCHEMAS",
    "ToolError",
    "tool_list_nodes",
    "tool_read_history",
    "tool_read_world",
    "tool_seed_world",
    "tool_submit_write",
]
