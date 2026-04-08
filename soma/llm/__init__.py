"""Provider-agnostic LLM interface.

This package is the only place in SOMA that knows about LLM SDKs.  Every
other layer — the Semantic Arbiter, the WorldAgent SDK, the runtime —
talks to an :class:`LLMProvider` and never imports ``anthropic``,
``openai``, ``google.genai``, or anything provider-specific.

Why that matters
----------------
1. **No vendor lock-in.**  Swapping Claude for GPT-4o or a local Llama
   is a one-line config change, not a refactor.
2. **Testability.**  The whole stack can be tested with
   :class:`soma.llm.providers.FakeProvider` — no API keys, no network,
   deterministic output.
3. **Cost control.**  A single :class:`LLMRouter` wraps primary +
   fallbacks and enforces per-call budgets, retries, and cost-aware
   routing in *one* place instead of every caller reinventing it.

Layers
------

- :mod:`soma.llm.base` — types + Protocol.  Pure, no I/O, no SDK deps.
- :mod:`soma.llm.retry` — :class:`RetryPolicy` with exponential backoff + jitter.
- :mod:`soma.llm.cost` — pricing table and per-call cost calculation.
- :mod:`soma.llm.router` — primary/fallback orchestration.
- :mod:`soma.llm.providers` — concrete adapters; each lazy-imports its SDK.
"""

from soma.llm.base import (
    ChatChunk,
    ChatOptions,
    ChatResponse,
    LLMProvider,
    Message,
    MessageRole,
    PermanentProviderError,
    RateLimitError,
    ResponseSchemaError,
    TransientProviderError,
    Usage,
)
from soma.llm.cost import PricingEntry, PricingTable, calculate_cost
from soma.llm.retry import RetryPolicy
from soma.llm.router import LLMRouter

__all__ = [
    "ChatChunk",
    "ChatOptions",
    "ChatResponse",
    "LLMProvider",
    "LLMRouter",
    "Message",
    "MessageRole",
    "PermanentProviderError",
    "PricingEntry",
    "PricingTable",
    "RateLimitError",
    "ResponseSchemaError",
    "RetryPolicy",
    "TransientProviderError",
    "Usage",
    "calculate_cost",
]
