"""Ollama adapter — local / self-hosted models.

Talks to an Ollama server over HTTP (the default is
``http://localhost:11434``).  Uses ``httpx`` directly rather than the
``ollama`` Python package because Ollama's REST API is stable and
small, and because many OpenAI-compatible local servers expose the
same endpoints — using plain httpx lets this adapter work for a broader
set of local tools (Llama.cpp, LM Studio, LocalAI, etc. in Ollama mode).

Install with::

    uv pip install soma[ollama]

Structured output uses Ollama's ``format=json`` mode plus a schema
injection in the system prompt.  Ollama doesn't natively enforce
schemas yet, so we validate aggressively on the Python side.
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
    ResponseSchemaError,
    TransientProviderError,
    Usage,
)

if TYPE_CHECKING:
    import httpx

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class OllamaProvider:
    """Adapter for a local / self-hosted Ollama server."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        if client is not None:
            self._client = client
        else:
            try:
                import httpx
            except ImportError as exc:
                raise PermanentProviderError(
                    "OllamaProvider requires the `httpx` package. "
                    "Install with: `uv pip install soma[ollama]`"
                ) from exc
            self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        payload = self._payload(messages, options)
        raw = await self._post("/api/chat", payload)
        return self._build_response(raw)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        payload = self._payload(messages, options)
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("done"):
                        yield ChatChunk(delta="", finish_reason="stop")
                        return
                    msg = event.get("message") or {}
                    delta = msg.get("content") or ""
                    if delta:
                        yield ChatChunk(delta=delta)
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        schema_hint = (
            f"Respond with a single JSON object that matches this JSON schema. "
            f"Do not include any explanation or markdown fences.\n\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        augmented = [*messages, Message(role=MessageRole.SYSTEM, content=schema_hint)]
        payload = self._payload(augmented, options)
        payload["format"] = "json"

        raw = await self._post("/api/chat", payload)
        content = (raw.get("message") or {}).get("content") or ""

        try:
            return schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ResponseSchemaError(
                f"Ollama returned content that did not validate against "
                f"{schema.__name__}: {exc}. Raw content: {content[:200]}..."
            ) from exc

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # HTTP internals
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except Exception as exc:  # pragma: no cover
            raise self._translate_error(exc) from exc

    def _payload(
        self,
        messages: Sequence[Message],
        options: ChatOptions | None,
    ) -> dict[str, Any]:
        opts = options or ChatOptions()
        ollama_options: dict[str, Any] = {"temperature": opts.temperature}
        if opts.top_p is not None:
            ollama_options["top_p"] = opts.top_p
        if opts.max_tokens is not None:
            ollama_options["num_predict"] = opts.max_tokens
        if opts.stop:
            ollama_options["stop"] = list(opts.stop)

        return {
            "model": self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": ollama_options,
        }

    def _build_response(self, raw: dict[str, Any]) -> ChatResponse:
        msg = raw.get("message") or {}
        content = msg.get("content") or ""
        # Ollama reports eval_count / prompt_eval_count in nanoseconds metrics
        # blob; the token counts are under those keys.
        usage = Usage(
            input_tokens=raw.get("prompt_eval_count") or 0,
            output_tokens=raw.get("eval_count") or 0,
        )
        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            usage=usage,
            finish_reason="stop" if raw.get("done") else None,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        name = type(exc).__name__
        msg = str(exc)
        # httpx.HTTPStatusError carries the response
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            from soma.llm.base import RateLimitError

            return RateLimitError(msg)
        if status is not None and 500 <= status < 600:
            return TransientProviderError(msg)
        if status is not None and 400 <= status < 500:
            return PermanentProviderError(msg)
        if "Timeout" in name or "Connect" in name:
            return TransientProviderError(msg)
        return TransientProviderError(msg)
