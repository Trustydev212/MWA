"""OpenAI adapter.

Implements :class:`~mwa.llm.base.LLMProvider` on top of the official
``openai`` async SDK.  Lazy-imports the SDK so the module stays
importable without ``openai`` installed.

Install with::

    uv pip install mwa[openai]

Structured output
-----------------
OpenAI's own API supports ``response_format={"type": "json_schema",
"strict": true}`` which server-side constrains decoding to a schema.
That's the strongest guarantee available, and it's the default path.

Third-party OpenAI-compatible gateways vary wildly:

- **Strict support** (OpenAI proper, some LiteLLM deployments):
  ``json_schema`` mode is respected and decoding is constrained.
- **Weak support** (futrixapi, OpenRouter free tier, …): the flag is
  accepted but silently ignored — the model produces JSON, but not
  matching the schema.  :class:`ResponseSchemaError` follows.
- **No support** at all: the API errors out on the flag.

:attr:`OpenAIProvider` handles all three via the ``structured_strategy``
constructor parameter:

- ``"json_schema"`` — always use json_schema strict.  Best for OpenAI.
- ``"json_object"`` — always use ``response_format={"type": "json_object"}``
  plus a schema hint injected as a system message (same approach
  :class:`~mwa.llm.providers.OllamaProvider` uses).  More portable.
- ``"auto"`` (default) — try json_schema first; on
  :class:`ResponseSchemaError` fall back to json_object and remember
  that decision for the rest of the instance's lifetime.  One extra
  round-trip on the first failure, zero on every call after.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar

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
    from openai import AsyncOpenAI

T_Schema = TypeVar("T_Schema", bound=BaseModel)

StructuredStrategy = Literal["json_schema", "json_object", "auto"]


class OpenAIProvider:
    """Adapter for OpenAI chat models (GPT-4/4o/5, o-series, ...)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        structured_strategy: StructuredStrategy = "auto",
    ) -> None:
        self._model = model
        self._structured_strategy: StructuredStrategy = structured_strategy
        # When strategy is "auto" we start by trying json_schema.  If it
        # fails once, we latch to json_object and never retry the strict
        # path on this instance — third-party gateways don't fix
        # themselves mid-session.
        self._auto_fallback_latched = False

        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise PermanentProviderError(
                    "OpenAIProvider requires the `openai` package. "
                    "Install with: `uv pip install mwa[openai]`"
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
            raw = await self._client.chat.completions.create(
                **self._build_request_kwargs(api_messages, opts)
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
            stream = await self._client.chat.completions.create(
                **self._build_request_kwargs(api_messages, opts, stream=True)
            )
            async for event in stream:
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
        strategy = self._structured_strategy

        if strategy == "json_schema":
            return await self._structured_via_json_schema(messages, schema, options)
        if strategy == "json_object":
            return await self._structured_via_json_object(messages, schema, options)

        # "auto" — try json_schema first unless we already learned it's broken.
        if self._auto_fallback_latched:
            return await self._structured_via_json_object(messages, schema, options)
        try:
            return await self._structured_via_json_schema(messages, schema, options)
        except ResponseSchemaError:
            # Gateway accepted json_schema but didn't honour it.  Latch the
            # fallback decision and retry this call with the portable path.
            self._auto_fallback_latched = True
            return await self._structured_via_json_object(messages, schema, options)

    async def _structured_via_json_schema(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        options: ChatOptions | None,
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
            raw = await self._client.chat.completions.create(
                **self._build_request_kwargs(
                    api_messages, opts, response_format=response_format
                )
            )
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

        content = self._extract_text(raw)
        try:
            return schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ResponseSchemaError(
                f"OpenAI json_schema mode returned content that did not "
                f"validate against {schema.__name__}: {exc}"
            ) from exc

    async def _structured_via_json_object(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        options: ChatOptions | None,
    ) -> T_Schema:
        """Portable structured output: prompt injection + json_object mode.

        Works on gateways that ignore json_schema strict but respect
        response_format=json_object (forces the model to emit a single JSON
        object).  We inject the schema as a system message so the model
        knows what shape to produce.
        """
        schema_hint = (
            "Respond with a single JSON object that matches this JSON schema. "
            "Use exactly the field names listed. Do not include explanations, "
            "markdown fences, or extra fields.\n\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        augmented: list[Message] = [
            *messages,
            Message(role=MessageRole.SYSTEM, content=schema_hint),
        ]
        api_messages = self._translate_messages(augmented)
        opts = options or ChatOptions()

        try:
            raw = await self._client.chat.completions.create(
                **self._build_request_kwargs(
                    api_messages, opts, response_format={"type": "json_object"}
                )
            )
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

        content = self._extract_text(raw)
        try:
            return schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ResponseSchemaError(
                f"OpenAI json_object mode returned content that did not "
                f"validate against {schema.__name__}: {exc}. "
                f"Raw content: {content[:200]}..."
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

    def _build_request_kwargs(
        self,
        api_messages: list[dict[str, Any]],
        opts: ChatOptions,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the kwargs dict passed to ``chat.completions.create``.

        Optional fields are only added when the caller actually set them.
        OpenAI's own API is lenient and accepts ``null`` for every optional
        field, but third-party OpenAI-compatible gateways (futrixapi,
        LiteLLM, OpenRouter, ...) vary: some reject ``{"top_p": null}``
        with a 400.  The only portable thing to do is omit fields we don't
        care about instead of sending explicit nulls.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": opts.temperature,
        }
        if opts.max_tokens is not None:
            kwargs["max_tokens"] = opts.max_tokens
        if opts.top_p is not None:
            kwargs["top_p"] = opts.top_p
        if opts.stop:
            kwargs["stop"] = list(opts.stop)
        kwargs.update(extra)
        return kwargs

    @staticmethod
    def _translate_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """MWA → OpenAI chat message format."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role.value
            if msg.role is MessageRole.TOOL:
                # OpenAI tool messages need a tool_call_id; we don't
                # track one at the MWA level yet so emit as a function
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
        # BadRequest / 400 = malformed payload, retrying is pointless.
        # Authentication / NotFound / PermissionDenied = same story.
        if (
            "BadRequest" in name
            or "400" in msg
            or "Authentication" in name
            or "PermissionDenied" in name
            or "NotFound" in name
        ):
            return PermanentProviderError(msg)
        return TransientProviderError(msg)
