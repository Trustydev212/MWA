"""OpenAI adapter.

Implements :class:`~soma.llm.base.LLMProvider` on top of the official
``openai`` async SDK.  Lazy-imports the SDK so the module stays
importable without ``openai`` installed.

Install with::

    uv pip install soma[openai]

Structured output uses OpenAI's ``response_format`` with ``json_schema``
mode — the strictest option available, which guarantees the response
either matches the schema or the API call errors out.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from soma.llm.base import (
    ChatChunk,
    ChatOptions,
    ChatResponse,
    Message,
    MessageRole,
    PermanentProviderError,
    RateLimitError,
    ResponseSchemaError,
    TransientProviderError,
    Usage,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class OpenAIProvider:
    """Adapter for OpenAI chat models (GPT-4/4o/5, o-series, ...)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model

        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise PermanentProviderError(
                    "OpenAIProvider requires the `openai` package. "
                    "Install with: `uv pip install soma[openai]`"
                ) from exc
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        try:
            # The openai SDK types messages as strict TypedDicts; SOMA's
            # neutral Message format intentionally stays simpler, so we
            # silence the messages arg-type check.  Wire format is still
            # exactly what the SDK expects.
            raw = await self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,  # type: ignore[arg-type]
                temperature=opts.temperature,
                max_tokens=opts.max_tokens,
                top_p=opts.top_p,
                stop=list(opts.stop) or None,
            )
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

        return self._build_response(raw)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        try:
            # stream=True picks the AsyncStream overload; messages silenced
            # for the same reason as chat().
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,  # type: ignore[arg-type]
                temperature=opts.temperature,
                max_tokens=opts.max_tokens,
                top_p=opts.top_p,
                stop=list(opts.stop) or None,
                stream=True,
            )
            async for event in stream:  # type: ignore[union-attr]
                choice = event.choices[0] if event.choices else None
                if choice is None:
                    continue
                delta_text = getattr(choice.delta, "content", None) or ""
                if delta_text:
                    yield ChatChunk(delta=delta_text)
                if choice.finish_reason:
                    yield ChatChunk(delta="", finish_reason=choice.finish_reason)
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

        try:
            # response_format is a hand-built dict rather than the SDK's
            # strict TypedDict; messages silenced for the same reason as
            # chat().  Together this confuses overload resolution so we
            # silence the whole call-overload selection here.
            raw = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self._model,
                messages=api_messages,
                temperature=opts.temperature,
                max_tokens=opts.max_tokens,
                response_format=response_format,
            )
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

        content = self._extract_text(raw)
        try:
            return schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ResponseSchemaError(
                f"OpenAI returned content that did not validate against {schema.__name__}: {exc}"
            ) from exc

    def count_tokens(self, text: str) -> int:
        # A real implementation would use tiktoken.  We keep the
        # heuristic to avoid pulling in tiktoken unconditionally; users
        # who need exact counts can install tiktoken themselves and
        # override this.
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """SOMA → OpenAI chat message format."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role.value
            if msg.role is MessageRole.TOOL:
                # OpenAI tool messages need a tool_call_id; we don't
                # track one at the SOMA level yet so emit as a function
                # message with the name.
                out.append({"role": "tool", "content": msg.content, "name": msg.name or "tool"})
            else:
                out.append({"role": role, "content": msg.content})
        return out

    def _build_response(self, raw: Any) -> ChatResponse:
        content = self._extract_text(raw)
        usage_obj = getattr(raw, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        )
        choices = getattr(raw, "choices", [])
        finish_reason = choices[0].finish_reason if choices else None
        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            usage=usage,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_text(raw: Any) -> str:
        choices = getattr(raw, "choices", [])
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", "") or ""

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        name = type(exc).__name__
        msg = str(exc)
        if "RateLimit" in name or "429" in msg:
            return RateLimitError(msg)
        if "APIConnection" in name or "Timeout" in name or "InternalServer" in name:
            return TransientProviderError(msg)
        if "Authentication" in name or "PermissionDenied" in name or "NotFound" in name:
            return PermanentProviderError(msg)
        return TransientProviderError(msg)
