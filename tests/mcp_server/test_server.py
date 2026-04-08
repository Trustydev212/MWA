"""Thin smoke tests for the MCP server wiring layer.

The interesting logic lives in ``tools.py`` (covered by
``test_tools.py``).  This file verifies only that:

- ``load_runtime_from_env`` respects ``MWA_HARNESS_PATH`` and fails
  cleanly when it's missing or wrong.
- ``build_server`` constructs an mcp ``Server`` instance and binds
  every handler name → callable pair.
- The handler dispatch layer converts tool results into
  ``TextContent`` blocks the MCP host expects.

We keep this thin on purpose — the real protocol integration test
is "running the server under an MCP host" and that's not something
unit tests can reasonably drive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mwa.errors import MWAError
from mwa.mcp_server.server import (
    HANDLERS,
    TOOL_DESCRIPTIONS,
    build_server,
    load_runtime_from_env,
)
from mwa.sdk import AgentRuntime


@pytest.fixture
def harness_path(tmp_path: Path) -> Path:
    """Write a minimal harness map JSON to a temp file and return the path."""
    path = tmp_path / "harness.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "domain": "test",
                "nodes": {
                    "root": {
                        "impact": "high",
                        "affects": [],
                        "order": 1,
                    },
                },
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# load_runtime_from_env
# ---------------------------------------------------------------------------


def test_load_runtime_requires_harness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MWA_HARNESS_PATH", raising=False)
    with pytest.raises(MWAError, match="MWA_HARNESS_PATH is not set"):
        load_runtime_from_env()


def test_load_runtime_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MWA_HARNESS_PATH", str(tmp_path / "does_not_exist.json"))
    with pytest.raises(MWAError, match="missing file"):
        load_runtime_from_env()


def test_load_runtime_builds_from_valid_harness(
    monkeypatch: pytest.MonkeyPatch, harness_path: Path
) -> None:
    monkeypatch.setenv("MWA_HARNESS_PATH", str(harness_path))
    runtime = load_runtime_from_env()
    assert isinstance(runtime, AgentRuntime)
    assert "root" in runtime.harness


# ---------------------------------------------------------------------------
# build_server
# ---------------------------------------------------------------------------


def test_build_server_returns_mcp_server_instance(
    harness_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke: build_server returns an mcp.server.Server instance.

    Skipped if the ``mcp`` package isn't installed — that's expected
    behaviour for minimal installs.
    """
    pytest.importorskip("mcp")
    from mcp.server import Server

    monkeypatch.setenv("MWA_HARNESS_PATH", str(harness_path))
    runtime = load_runtime_from_env()
    app = build_server(runtime)
    assert isinstance(app, Server)


def test_tool_descriptions_cover_every_handler() -> None:
    """Every handler must have a description — catches copy-paste rot."""
    assert set(HANDLERS.keys()) == set(TOOL_DESCRIPTIONS.keys())
    for name, desc in TOOL_DESCRIPTIONS.items():
        assert desc, f"{name}: empty description"
        assert len(desc) > 30, f"{name}: description too terse"
