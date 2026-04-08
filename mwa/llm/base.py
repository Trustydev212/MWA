"""Provider-agnostic types + Protocol.

Everything in this module is **pure** — no SDK imports, no I/O, no
side-effects.  That keeps ``mwa.llm.base`` importable even when none
of the actual provider SDKs are installed, which is essential for
testing and for composing adapters without pulling in every vendor
SDK simultaneously.

Design notes
------------
- Message format is **neutral** (role + content) — not Anthropic's
  shape, not OpenAI's shape.  Each provider adapter translates to
  its SDK's format on the way out.  If we used one vendor's format
  we'd forever be translating in an arbitrary direction.
- ``Usage`` is the minimum viable cost unit: input tokens + output
  tokens.  Cache hits, thinking tokens, etc. are vendor-specific and
  go into ``ChatResponse.raw`` for callers that care.
- Errors form a deliberate 3-level hierarchy so retry policies can
  be concise (``retry_on=(RateLimitError, TransientProviderError)``)
  without enumerating every vendor's exception classes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mwa.errors import LLMProviderError

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class RateLimitError(LLMProviderError):
    """HTTP 429 / provider-specific rate limit.  Safe to retry with backoff."""


class TransientProviderError(LLMProviderError):
    """5xx / network / timeout.  Safe to retry with backoff."""


class PermanentProviderError(LLMProviderError):
    """4xx (not 429) / authentication / invalid request.  Do NOT retry."""


class ResponseSchemaError(LLMProviderError):
    """Provider returned data that didn't match the requested schema.

    Not automatically retried — the retry would be deterministic and waste
    budget.  Callers that want best-of-N should catch this explicitly.
    """


# ---------------------------------------------------------------------------
# Message / Usage / Options
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """One message in a chat conversation — provider-neutral shape."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str
    name: str | None = None
    """Only used for ``tool`` messages, to identify which tool responded."""


class Usage(BaseModel):
    """Tokens consumed by a single provider call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ChatOptions(BaseModel):
    """Per-call generation parameters — only the settings every provider
    understands.  Provider-specific knobs live in each adapter's
    constructor (e.g. ``top_k``, ``reasoning_effort``, etc.)."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int | None = Field(ge=1, default=None)
    top_p: float | None = Field(ge=0.0, le=1.0, default=None)
    stop: tuple[str, ...] = Field(default_factory=tuple)


class ChatResponse(BaseModel):
    """The full response from a ``chat()`` call."""

    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
    """Provider-specific extras (thinking blocks, tool calls, cache ids...)."""


class ChatChunk(BaseModel):
    """One token / delta from a streaming response."""

    model_config = ConfigDict(frozen=True)

    delta: str
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

T_Schema = TypeVar("T_Schema", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Every provider adapter implements this Protocol.

    Callers can rely on these five methods + two properties being
    available regardless of the backing vendor.  Adapters are free to
    add more methods (e.g. Anthropic-specific cache-aware helpers) but
    MWA core only ever touches what's defined here.
    """

    @property
    def name(self) -> str:
        """Stable identifier for the provider family, e.g. ``"anthropic"``.

        Used by cost tables, logs, and :class:`LLMRouter` diagnostics.
        """
        ...

    @property
    def model(self) -> str:
        """Specific model identifier, e.g. ``"claude-opus-4-6"``."""
        ...

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        """Single-shot chat call.  Returns the complete response."""
        ...

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streaming chat call.  Yields :class:`ChatChunk` as tokens arrive."""
        ...

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        """Force the model to return an instance of ``schema``.

        Each adapter picks the best mechanism its backend offers
        (Anthropic: tool_use; OpenAI: response_format json_schema;
        Gemini: response_schema; Ollama: format=json + prompt) and
        runs the result through Pydantic validation before returning.

        Raises :class:`ResponseSchemaError` on validation failure.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Best-effort token count for ``text``.

        Adapters that have real tokenizers (e.g. ``tiktoken`` for
        OpenAI) should override.  The default falls back to a
        4-chars-per-token heuristic, which is close enough for
        budgeting but not for billing.
        """
        ...
