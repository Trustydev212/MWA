"""In-memory fake provider — the workhorse of MWA's LLM test suite.

:class:`FakeProvider` implements :class:`~mwa.llm.base.LLMProvider`
without ever making a network call.  Tests configure it by stacking
canned responses or an error queue; the provider pops one entry per
call.  That gives deterministic, fast, offline tests for every layer
above the provider interface — the Semantic Arbiter, the Router, the
Agent SDK — without any of them ever touching a real LLM.

Two consumers are expected:

1. **MWA's own tests** (``tests/llm/``, ``tests/arbiter/``, ...).
2. **User projects** that want to unit-test their agents without
   burning tokens.  That's why this lives under ``mwa.llm.providers``
   instead of ``tests/`` — it's part of the public API.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from mwa.llm.base import (
    ChatChunk,
    ChatOptions,
    ChatResponse,
    Message,
    ResponseSchemaError,
    Usage,
)

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class FakeProvider:
    """Queue-driven fake implementing :class:`~mwa.llm.base.LLMProvider`.

    Parameters
    ----------
    name:
        Provider identifier returned by :attr:`name`.  Default ``"fake"``.
    model:
        Model identifier returned by :attr:`model`.  Default ``"fake-1"``.
    """

    def __init__(self, *, name: str = "fake", model: str = "fake-1") -> None:
        self._name = name
        self._model = model
        self._responses: deque[ChatResponse | BaseException] = deque()
        self._calls: list[tuple[tuple[Message, ...], ChatOptions]] = []

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        self._calls.append((tuple(messages), options or ChatOptions()))
        # Yield to the event loop so tests that interleave calls are
        # realistic (and so we can exercise concurrency code paths).
        await asyncio.sleep(0)
        return self._pop_next()

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        response = await self.chat(messages, options=options)
        # Emit the whole response in one chunk — good enough for tests
        # that just want to verify streaming plumbing.
        yield ChatChunk(delta=response.content, finish_reason=response.finish_reason)

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        response = await self.chat(messages, options=options)
        try:
            data = json.loads(response.content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ResponseSchemaError(
                f"FakeProvider returned content that did not match {schema.__name__}: {exc}"
            ) from exc

    def count_tokens(self, text: str) -> int:
        # Simple heuristic — 4 chars ≈ 1 token.  Tests that care about
        # exact counts should use ``enqueue_usage`` and assert on usage.
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Test helpers (public — that's the point)
    # ------------------------------------------------------------------

    def enqueue_text(
        self,
        content: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> FakeProvider:
        """Queue a plain text response.

        Returns ``self`` so calls can be chained::

            fake.enqueue_text("hi").enqueue_text("bye")
        """
        self._responses.append(
            ChatResponse(
                content=content,
                model=self._model,
                provider=self._name,
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                finish_reason=finish_reason,
            )
        )
        return self

    def enqueue_json(
        self,
        data: object,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> FakeProvider:
        """Queue a JSON response — convenience for structured-output tests."""
        return self.enqueue_text(
            json.dumps(data),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def enqueue_error(self, error: BaseException) -> FakeProvider:
        """Queue an error to be raised on the next call."""
        self._responses.append(error)
        return self

    @property
    def calls(self) -> list[tuple[tuple[Message, ...], ChatOptions]]:
        """Every ``(messages, options)`` pair received by this fake."""
        return list(self._calls)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pop_next(self) -> ChatResponse:
        if not self._responses:
            raise RuntimeError(
                "FakeProvider was called but no responses are queued. "
                "Did you forget enqueue_text() / enqueue_json() in the test?"
            )
        item = self._responses.popleft()
        if isinstance(item, BaseException):
            raise item
        return item
