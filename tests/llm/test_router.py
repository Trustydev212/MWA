"""Tests for LLMRouter — fallback chain, retry integration, budget."""

from __future__ import annotations

from decimal import Decimal

import pytest

from mwa.errors import LLMProviderError
from mwa.llm import (
    LLMRouter,
    Message,
    MessageRole,
    PermanentProviderError,
    PricingTable,
    RateLimitError,
    RetryPolicy,
    TransientProviderError,
)
from mwa.llm.providers import FakeProvider


async def _noop_sleep(_: float) -> None:
    return None


def _retry() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        base_delay=0.01,
        max_delay=0.1,
        sleep=_noop_sleep,
    )


def _user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


# ---------------------------------------------------------------------------
# Primary-only success
# ---------------------------------------------------------------------------


async def test_primary_success_no_fallback_used() -> None:
    primary = FakeProvider(name="primary").enqueue_text("primary response")
    fallback = FakeProvider(name="fallback").enqueue_text("fallback response")

    router = LLMRouter(primary=primary, fallbacks=[fallback], retry=_retry())
    response = await router.chat([_user("hi")])

    assert response.content == "primary response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


# ---------------------------------------------------------------------------
# Fallback after retry exhausted
# ---------------------------------------------------------------------------


async def test_fallback_used_after_primary_retries_exhausted() -> None:
    primary = (
        FakeProvider(name="primary")
        .enqueue_error(TransientProviderError("5xx 1"))
        .enqueue_error(TransientProviderError("5xx 2"))
    )
    fallback = FakeProvider(name="fallback").enqueue_text("rescue")

    router = LLMRouter(primary=primary, fallbacks=[fallback], retry=_retry())
    response = await router.chat([_user("hi")])

    assert response.content == "rescue"
    assert len(primary.calls) == 2  # retried once
    assert len(fallback.calls) == 1


async def test_rate_limit_triggers_retry_then_fallback() -> None:
    primary = (
        FakeProvider(name="primary")
        .enqueue_error(RateLimitError("429"))
        .enqueue_error(RateLimitError("429"))
    )
    fallback = FakeProvider(name="fallback").enqueue_text("ok")

    router = LLMRouter(primary=primary, fallbacks=[fallback], retry=_retry())
    response = await router.chat([_user("hi")])

    assert response.content == "ok"


# ---------------------------------------------------------------------------
# Permanent errors still try fallback
# ---------------------------------------------------------------------------


async def test_permanent_error_on_primary_still_tries_fallback() -> None:
    """A bad API key on Anthropic shouldn't stop us trying OpenAI."""
    primary = FakeProvider(name="primary").enqueue_error(PermanentProviderError("invalid api key"))
    fallback = FakeProvider(name="fallback").enqueue_text("fallback ok")

    router = LLMRouter(primary=primary, fallbacks=[fallback], retry=_retry())
    response = await router.chat([_user("hi")])

    assert response.content == "fallback ok"
    assert len(primary.calls) == 1  # NOT retried (permanent)
    assert len(fallback.calls) == 1


# ---------------------------------------------------------------------------
# All providers fail
# ---------------------------------------------------------------------------


async def test_all_providers_fail_raises_last_error() -> None:
    primary = (
        FakeProvider(name="primary")
        .enqueue_error(TransientProviderError("p1"))
        .enqueue_error(TransientProviderError("p2"))
    )
    fallback = (
        FakeProvider(name="fallback")
        .enqueue_error(TransientProviderError("f1"))
        .enqueue_error(TransientProviderError("f2"))
    )

    router = LLMRouter(primary=primary, fallbacks=[fallback], retry=_retry())
    with pytest.raises(TransientProviderError, match="f2"):
        await router.chat([_user("hi")])


# ---------------------------------------------------------------------------
# Budget gate
# ---------------------------------------------------------------------------


async def test_budget_skips_expensive_providers() -> None:
    """Primary is expensive, fallback is free — router picks fallback."""
    table = PricingTable(entries={})
    table.set("expensive", "*", input_per_million=1000, output_per_million=2000)
    table.set("cheap", "*", input_per_million=0, output_per_million=0)

    primary = FakeProvider(name="expensive").enqueue_text("expensive")
    fallback = FakeProvider(name="cheap").enqueue_text("cheap")

    router = LLMRouter(
        primary=primary,
        fallbacks=[fallback],
        retry=_retry(),
        budget_per_call_usd=Decimal("0.01"),
        pricing=table,
    )
    response = await router.chat([_user("a long enough message to estimate")])

    assert response.content == "cheap"
    assert len(primary.calls) == 0  # skipped by budget


async def test_budget_within_limit_uses_primary() -> None:
    table = PricingTable(entries={})
    table.set("cheap", "*", input_per_million=0, output_per_million=0)

    primary = FakeProvider(name="cheap").enqueue_text("ok")
    router = LLMRouter(
        primary=primary,
        retry=_retry(),
        budget_per_call_usd=Decimal("10.00"),
        pricing=table,
    )
    response = await router.chat([_user("hi")])
    assert response.content == "ok"


async def test_all_providers_budget_blocked_raises() -> None:
    table = PricingTable(entries={})
    table.set("expensive", "*", input_per_million=1_000_000, output_per_million=1_000_000)

    primary = FakeProvider(name="expensive")
    router = LLMRouter(
        primary=primary,
        retry=_retry(),
        budget_per_call_usd=Decimal("0.001"),
        pricing=table,
    )
    with pytest.raises(LLMProviderError, match="Budget-blocked"):
        await router.chat([_user("hi")])


# ---------------------------------------------------------------------------
# Router satisfies LLMProvider shape
# ---------------------------------------------------------------------------


def test_router_exposes_provider_metadata() -> None:
    primary = FakeProvider(name="primary", model="model-a")
    fallback = FakeProvider(name="fallback", model="model-b")
    router = LLMRouter(primary=primary, fallbacks=[fallback])

    assert router.name == "router"
    assert "model-a" in router.model
    assert "model-b" in router.model
    assert router.providers == (primary, fallback)
    # Count tokens delegates to primary
    assert router.count_tokens("hello world") >= 1
