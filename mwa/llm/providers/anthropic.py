"""Anthropic Claude adapter.

Implements :class:`~mwa.llm.base.LLMProvider` on top of the official
``anthropic`` Python SDK.  The SDK is imported **lazily** inside
``__init__`` so this module stays importable even when ``anthropic``
isn't installed — important for testing and for users who only use
other providers.

Install with::

    uv pip install mwa[anthropic]

Structured output uses Anthropic's tool_use mechanism (forcing a single
tool call that matches the schema).  That's more reliable than prompt
engineering JSON and gives us proper error messages on mismatches.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from mwa.llm.base import (
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
    from anthropic import AsyncAnthropic

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class AnthropicProvider:
    """Adapter for Anthropic's Claude models."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
        max_tokens_default: int = 4096,
    ) -> None:
        self._model = model
        self._max_tokens_default = max_tokens_default

        if client is not None:
            # Dependency injection — mainly for tests that pass a mock.
            self._client = client
        else:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise PermanentProviderError(
                    "AnthropicProvider requires the `anthropic` package. "
                    "Install with: `uv pip install mwa[anthropic]`"
                ) from exc
            self._client = AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        system, api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        try:
            raw = await self._client.messages.create(
                model=self._model,
                messages=api_messages,
                system=system or "",
                max_tokens=opts.max_tokens or self._max_tokens_default,
                temperature=opts.temperature,
                stop_sequences=list(opts.stop) or None,
            )
        except Exception as exc:  # pragma: no cover - exercised via mock
            raise self._translate_error(exc) from exc

        return self._build_response(raw)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        system, api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        try:
            async with self._client.messages.stream(
                model=self._model,
                messages=api_messages,
                system=system or "",
                max_tokens=opts.max_tokens or self._max_tokens_default,
                temperature=opts.temperature,
                stop_sequences=list(opts.stop) or None,
            ) as stream:
                async for delta in stream.text_stream:
                    yield ChatChunk(delta=delta)
                final = await stream.get_final_message()
                yield ChatChunk(delta="", finish_reason=final.stop_reason)
        except Exception as exc:  # pragma: no cover - exercised via mock
            raise self._translate_error(exc) from exc

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        """Use tool_use to force the model to return ``schema``.

        We register the schema as a single tool and force the model to
        call it via ``tool_choice={"type": "tool", "name": ...}``.
        """
        system, api_messages = self._translate_messages(messages)
        opts = options or ChatOptions()

        tool_name = schema.__name__
        tool = {
            "name": tool_name,
            "description": f"Return a well-formed {tool_name}.",
            "input_schema": schema.model_json_schema(),
        }

        try:
            raw = await self._client.messages.create(
                model=self._model,
                messages=api_messages,
                system=system or "",
                max_tokens=opts.max_tokens or self._max_tokens_default,
                temperature=opts.temperature,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

        # Find the tool_use block and validate its input against the schema.
        for block in getattr(raw, "content", []):
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == tool_name
            ):
                try:
                    return schema.model_validate(block.input)
                except ValidationError as exc:
                    raise ResponseSchemaError(
                        f"Anthropic returned tool_use for {tool_name} but the "
                        f"arguments did not validate: {exc}"
                    ) from exc
        raise ResponseSchemaError(f"Anthropic did not return a tool_use block for {tool_name}.")

    def count_tokens(self, text: str) -> int:
        # Anthropic's SDK has a token counter but calling it is async
        # and costs an API roundtrip on newer SDK versions.  Fall back
        # to the standard heuristic here; call sites that need exact
        # counts should use the SDK directly.
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_messages(
        messages: Sequence[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split MWA messages into (system, api_messages).

        Anthropic puts the system prompt in a separate field, not as a
        message.  MWA's neutral format uses a ``system`` role so we
        extract those here and concatenate them in order.
        """
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is MessageRole.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role is MessageRole.TOOL:
                # Tool results become user messages with a tool_result block
                # — but since MWA's base tool messages are text-only we
                # pass them through as user messages for now.  Rich tool
                # message support is a later milestone.
                api_messages.append({"role": "user", "content": msg.content})
            else:
                api_messages.append({"role": msg.role.value, "content": msg.content})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, api_messages

    def _build_response(self, raw: Any) -> ChatResponse:
        text_parts: list[str] = []
        for block in getattr(raw, "content", []):
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        content = "".join(text_parts)

        usage_obj = getattr(raw, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        )

        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            usage=usage,
            finish_reason=getattr(raw, "stop_reason", None),
        )

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        """Map anthropic SDK exceptions to MWA's error hierarchy.

        We only import the anthropic exception classes if they are
        already imported (they must be, since the SDK already raised).
        """
        name = type(exc).__name__
        msg = str(exc)
        if "RateLimit" in name or "429" in msg:
            return RateLimitError(msg)
        if "APIConnection" in name or "APITimeout" in name or "InternalServer" in name:
            return TransientProviderError(msg)
        if "Authentication" in name or "PermissionDenied" in name or "NotFound" in name:
            return PermanentProviderError(msg)
        # Fallback: treat unknown errors as transient — retry may help.
        return TransientProviderError(msg)
