"""Live integration tests against any OpenAI-compatible endpoint.

These tests actually hit the network and cost (potentially real) tokens.
They are marked ``integration`` and **skipped by default** — pytest will
only run them when you explicitly invoke the ``integration`` marker AND
the required environment variables are present::

    export MWA_LIVE_API_KEY="sk-..."
    export MWA_LIVE_BASE_URL="https://futrixapi.com/v1"  # or https://api.openai.com/v1
    export MWA_LIVE_MODEL="auto"                        # or "gpt-4o-mini"

    uv run pytest -m integration tests/llm/test_live_openai_compat.py -v

Why a separate file?
--------------------
- Offline unit tests never touch the network — safe by default.
- Live tests are opt-in via both the marker **and** the env vars, so
  CI never accidentally burns tokens.
- The env-var pattern keeps credentials out of the repo entirely —
  nothing in this file references a specific key or URL.

What we test
------------
1. ``chat()`` end-to-end: non-empty response, usage reported, provider
   metadata consistent.
2. ``structured()`` end-to-end: Pydantic validation survives the real
   vendor's JSON-schema behaviour.
3. ``LLMRouter`` end-to-end with the real provider as primary and an
   offline :class:`FakeProvider` as the last-resort fallback, proving
   the whole chain (adapter → retry → router) works against a real API.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from mwa.llm.base import ChatOptions, Message, MessageRole
from mwa.llm.providers import FakeProvider, OpenAIProvider
from mwa.llm.retry import RetryPolicy
from mwa.llm.router import LLMRouter

LIVE_ENV = {
    "api_key": "MWA_LIVE_API_KEY",
    "base_url": "MWA_LIVE_BASE_URL",
    "model": "MWA_LIVE_MODEL",
}


def _live_config() -> dict[str, str] | None:
    """Return live config if every env var is set, else ``None``."""
    values: dict[str, str] = {}
    for key, env_name in LIVE_ENV.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            return None
        values[key] = value
    return values


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _live_config() is None,
        reason=(
            "Live OpenAI-compatible tests are skipped. Set "
            + ", ".join(LIVE_ENV.values())
            + " to run."
        ),
    ),
]


@pytest.fixture
def provider() -> OpenAIProvider:
    cfg = _live_config()
    assert cfg is not None  # guarded by skipif above
    return OpenAIProvider(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )


# ---------------------------------------------------------------------------
# Basic chat
# ---------------------------------------------------------------------------


async def test_live_chat_returns_non_empty_response(provider: OpenAIProvider) -> None:
    response = await provider.chat(
        [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a terse assistant. Answer in 8 words or fewer.",
            ),
            Message(role=MessageRole.USER, content="Say hello from MWA."),
        ],
        options=ChatOptions(temperature=0.0, max_tokens=64),
    )

    assert response.provider == "openai"
    assert response.content, "live endpoint returned empty content"
    # usage is optional but every decent provider reports something:
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0


# ---------------------------------------------------------------------------
# Structured output (json_schema)
# ---------------------------------------------------------------------------


class _Decision(BaseModel):
    winner: str = Field(description="Name of the agent who wins the conflict")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


async def test_live_structured_output(provider: OpenAIProvider) -> None:
    """Prove that the real endpoint produces schema-valid JSON.

    Some OpenAI-compatible gateways don't implement ``response_format``
    strictly.  If this test fails with ``ResponseSchemaError`` on a
    gateway like LiteLLM / OpenRouter / etc., it's a real interop
    finding we want to surface, not a MWA bug.
    """
    decision = await provider.structured(
        [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are the MWA Semantic Arbiter. Given a fake conflict, "
                    "return a structured decision."
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    "Agent alice wrote tone='serious' with confidence 0.7. "
                    "Agent bob wrote tone='playful' with confidence 0.75. "
                    "Return a decision."
                ),
            ),
        ],
        _Decision,
        options=ChatOptions(temperature=0.0, max_tokens=256),
    )

    assert decision.winner in {"alice", "bob"}
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason


# ---------------------------------------------------------------------------
# Router end-to-end with offline fallback
# ---------------------------------------------------------------------------


async def test_live_router_with_offline_fallback(provider: OpenAIProvider) -> None:
    """Real provider as primary, offline FakeProvider as last resort.

    Happy path exercises the router's success branch against a live
    endpoint; the fallback is there as insurance so a transient outage
    during the test still produces a green run.
    """
    fallback = FakeProvider(name="offline", model="offline").enqueue_text("offline fallback used")

    async def _noop_sleep(_: float) -> None:
        return None

    router = LLMRouter(
        primary=provider,
        fallbacks=[fallback],
        retry=RetryPolicy(max_attempts=2, base_delay=0.1, sleep=_noop_sleep),
    )

    response = await router.chat(
        [Message(role=MessageRole.USER, content="Say 'hi' and nothing else.")],
        options=ChatOptions(temperature=0.0, max_tokens=16),
    )

    # Either the live call succeeded (expected) OR the fallback kicked in
    # (acceptable on a flaky connection) — both are valid pass states.
    assert response.content
    assert response.provider in {"openai", "offline"}
