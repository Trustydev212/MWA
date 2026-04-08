"""Exponential-backoff retry with jitter.

Narrowly-scoped: only retries the exception types we've marked as
"transient" (rate limits, 5xx, timeouts).  Everything else escapes
immediately.  Retrying permanent errors (auth failures, malformed
requests) just wastes budget without fixing anything.

We use *decorrelated jitter* rather than full jitter — it tends to
recover faster under thundering-herd conditions.  Reference:
https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from mwa.llm.base import RateLimitError, TransientProviderError

T = TypeVar("T")


class RetryPolicy:
    """Retry async callables on transient failures.

    Parameters
    ----------
    max_attempts:
        Total attempts including the first.  ``max_attempts=3`` means
        "try once, retry up to 2 times".
    base_delay:
        Seconds to wait before the first retry.
    max_delay:
        Upper bound on a single sleep.
    exponential_base:
        Multiplier applied between attempts.  ``2.0`` → 0.5, 1.0, 2.0, 4.0 ...
    jitter:
        If ``True``, apply decorrelated jitter to each sleep.
    retry_on:
        Exception types that trigger a retry.  Subclasses are included.
    sleep:
        Injection point for tests — tests pass a no-op ``async def sleep(_): ...``
        so the suite doesn't actually wait.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple[type[BaseException], ...] = (
            RateLimitError,
            TransientProviderError,
        ),
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_delay <= 0 or max_delay <= 0:
            raise ValueError("base_delay and max_delay must be > 0")
        if exponential_base <= 1.0:
            raise ValueError("exponential_base must be > 1.0")

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on
        self._sleep = sleep or asyncio.sleep

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run ``fn`` with retry.

        On the final attempt the original exception is re-raised so
        callers see the real failure, not a wrapper.
        """
        attempt = 0
        last_delay = self.base_delay
        while True:
            attempt += 1
            try:
                return await fn()
            except self.retry_on:
                if attempt >= self.max_attempts:
                    raise
                last_delay = self._next_delay(attempt, last_delay)
                await self._sleep(last_delay)

    def _next_delay(self, attempt: int, last_delay: float) -> float:
        """Decorrelated jitter: next = random(base, last * base).

        Falls back to plain exponential if ``jitter=False``.
        """
        ceiling = min(self.max_delay, self.base_delay * (self.exponential_base**attempt))
        if not self.jitter:
            return ceiling
        low = self.base_delay
        high = min(self.max_delay, last_delay * self.exponential_base)
        high = max(high, low)  # guard when base_delay ≈ max_delay
        return random.uniform(low, high)
