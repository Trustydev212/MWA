"""CLI entry point — ``python -m mwa.mcp_server``.

Also reachable via the ``mwa-mcp`` console script once the package
is installed (see ``pyproject.toml`` ``[project.scripts]`` block).

The server block-reads stdio until the MCP host closes the stream,
so this script doesn't return on its own — Ctrl-C or host
disconnection stop it.
"""

from __future__ import annotations

import asyncio
import sys

from mwa.errors import MWAError


def main() -> int:
    """Run the MCP stdio server.  Returns a POSIX exit code."""
    # Lazy import so ``--help`` and import-time errors fire before
    # we touch the mcp SDK.
    from mwa.mcp_server.server import run_stdio_server

    try:
        asyncio.run(run_stdio_server())
    except MWAError as exc:
        print(f"mwa-mcp: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
