"""Demo-specific helpers — provider factories for the OpenClaw team.

After the M6 ``WorldAgent`` SDK landed, the heavy lifting moved into
:class:`mwa.sdk.AgentRuntime` and this file shrank to just the pieces
that are genuinely demo-local:

1. Where the harness map lives on disk.
2. How to build an ``AgentRuntime`` from that harness plus an arbiter
   LLM (thin wrapper around ``AgentRuntime.from_harness_file``).
3. How to construct a provider per agent in offline or live mode.

Everything else — the write pipeline, the conflict resolution, the
event dispatcher — now belongs to ``mwa.sdk``.
"""

from __future__ import annotations

import os
from pathlib import Path

from mwa.llm import LLMRouter, RetryPolicy
from mwa.llm.base import LLMProvider
from mwa.llm.providers import FakeProvider, OpenAIProvider
from mwa.sdk import AgentRuntime

HARNESS_PATH = (
    Path(__file__).resolve().parents[2]
    / "harness_maps"
    / "openclaw_agent_builder.json"
)


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------


def build_runtime(arbiter_llm: LLMProvider) -> AgentRuntime:
    """Thin wrapper so the demo never has to know the harness path.

    Keeps the ``AgentRuntime`` construction in one place — if the SDK
    constructor signature changes, only this function updates.
    """
    return AgentRuntime.from_harness_file(
        HARNESS_PATH,
        arbiter_llm=arbiter_llm,
        auto_resolve_threshold=0.85,
    )


# ---------------------------------------------------------------------------
# Provider factory (offline + live)
# ---------------------------------------------------------------------------


def _live_config() -> dict[str, str] | None:
    """Return live env config if every MWA_LIVE_* var is set, else None."""
    keys = {
        "api_key": "MWA_LIVE_API_KEY",
        "base_url": "MWA_LIVE_BASE_URL",
        "model": "MWA_LIVE_MODEL",
    }
    values: dict[str, str] = {}
    for k, env_name in keys.items():
        val = os.environ.get(env_name, "").strip()
        if not val:
            return None
        values[k] = val
    return values


def make_live_provider() -> LLMProvider:
    """Real OpenAI-compatible provider wrapped in RetryPolicy.

    Mirrors the production pattern from the live test suite — the
    futrixapi 504 session taught us that Cloudflare flakes are routine
    on gateway routes, so every real provider gets router-wrapped.
    """
    cfg = _live_config()
    if cfg is None:
        raise RuntimeError(
            "make_live_provider() called but MWA_LIVE_API_KEY / _BASE_URL / "
            "_MODEL are not set.  Use make_fake_provider(...) for offline mode."
        )
    raw = OpenAIProvider(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        structured_strategy="auto",
    )
    return LLMRouter(
        primary=raw,
        retry=RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
        ),
    )


def make_fake_provider(name: str, queued_json: list[dict[str, object]]) -> FakeProvider:
    """Offline provider pre-loaded with deterministic decisions.

    One instance per agent so concurrent handler execution can't
    shuffle responses.  Each dict is serialised and returned by the
    next ``.structured()`` call.
    """
    provider = FakeProvider(name=name, model=f"{name}-fake")
    for payload in queued_json:
        provider.enqueue_json(payload)
    return provider


def is_live_mode() -> bool:
    """True iff every MWA_LIVE_* env var is set."""
    return _live_config() is not None
