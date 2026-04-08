"""Tests for RetryPolicy — no real sleeps, deterministic failure counts."""

from __future__ import annotations

import pytest

from mwa.errors import LLMProviderError
from mwa.llm import (
    PermanentProviderError,
    RateLimitError,
    RetryPolicy,
    TransientProviderError,
)


async def _noop_sleep(_: float) -> None:
    return None


def _policy(**kwargs: object) -> RetryPolicy:
    defaults: dict[str, object] = {
        "max_attempts": 3,
        "base_delay": 0.1,
        "max_delay": 1.0,
        "sleep": _noop_sleep,
    }
    defaults.update(kwargs)
    return RetryPolicy(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_invalid_base_delay() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=0)


def test_invalid_exponential_base() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(exponential_base=1.0)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_returns_first_success() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    policy = _policy()
    assert await policy.run(op) == "ok"
    assert calls == 1


# ---------------------------------------------------------------------------
# Retry on transient
# ---------------------------------------------------------------------------


async def test_retries_rate_limit_then_succeeds() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError("slow down")
        return "ok"

    policy = _policy(max_attempts=5)
    assert await policy.run(op) == "ok"
    assert calls == 3


async def test_retries_transient_error() -> None:
    calls = 0

    async def op() -> None:
        nonlocal calls
        calls += 1
        raise TransientProviderError("flaky")

    policy = _policy(max_attempts=3)
    with pytest.raises(TransientProviderError):
        await policy.run(op)
    assert calls == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# Do NOT retry on permanent
# ---------------------------------------------------------------------------


async def test_does_not_retry_permanent_error() -> None:
    calls = 0

    async def op() -> None:
        nonlocal calls
        calls += 1
        raise PermanentProviderError("bad api key")

    policy = _policy(max_attempts=5)
    with pytest.raises(PermanentProviderError):
        await policy.run(op)
    assert calls == 1


async def test_does_not_retry_unrelated_error() -> None:
    calls = 0

    async def op() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("not an LLM error")

    policy = _policy()
    with pytest.raises(ValueError):
        await policy.run(op)
    assert calls == 1


async def test_custom_retry_on() -> None:
    """Callers can override which exceptions trigger a retry."""
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise LLMProviderError("custom transient")
        return "ok"

    policy = _policy(retry_on=(LLMProviderError,))
    assert await policy.run(op) == "ok"
    assert calls == 2


# ---------------------------------------------------------------------------
# Delay sequencing
# ---------------------------------------------------------------------------


async def test_sleeps_are_called_between_attempts() -> None:
    sleeps: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def op() -> None:
        raise TransientProviderError("boom")

    policy = RetryPolicy(
        max_attempts=4,
        base_delay=0.1,
        max_delay=1.0,
        jitter=False,
        sleep=recording_sleep,
    )
    with pytest.raises(TransientProviderError):
        await policy.run(op)

    # 4 attempts = 3 sleeps between them
    assert len(sleeps) == 3
    # Strictly increasing (exponential, no jitter)
    assert sleeps == sorted(sleeps)
