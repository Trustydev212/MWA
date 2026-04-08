"""LLMRouter — fallback chain + retry + per-call budget.

The Router is the *only* place in SOMA that knows how to handle a
failing LLM call gracefully.  Every other layer just asks a "provider"
to chat and gets a response back; the Router is what makes that
provider resilient in the face of rate limits, outages, and cost
overruns.

Call sequence for :meth:`chat`:

1. Iterate providers in order (primary → fallbacks).
2. For each provider:
   a. Check the per-call budget (if configured).  Skip if over.
   b. Run the call through the :class:`RetryPolicy`.
   c. On success, return the :class:`ChatResponse`.
   d. On a retried-out exception, fall through to the next provider.
3. If no provider succeeded, re-raise the last exception we saw.

We deliberately do **not** implement load balancing, canary routing,
or any form of adaptive weighting here.  Those are legitimate needs
but each one comes with subtle policy choices (sticky sessions?
warmup windows? cost-weighted vs latency-weighted?).  Shipping them
prematurely would paint us into a corner.  The current Router
handles the 95% case — "primary died, fall back to secondary" — and
leaves room for a smarter router layer on top later.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TypeVar

from pydantic import BaseModel

from soma.errors import LLMProviderError
from soma.llm.base import (
    ChatOptions,
    ChatResponse,
    LLMProvider,
    Message,
    Usage,
)
from soma.llm.cost import PricingTable, calculate_cost
from soma.llm.retry import RetryPolicy

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class LLMRouter:
    """Orchestrates primary + fallback providers with retry and budget.

    The Router itself satisfies :class:`LLMProvider` structurally (it
    has ``chat`` / ``stream`` / ``structured`` / ``count_tokens``), so
    you can drop it anywhere a provider is expected — including inside
    another Router.

    Parameters
    ----------
    primary:
        The first provider to try.
    fallbacks:
        Ordered list of providers to try if ``primary`` fails.
    retry:
        Retry policy applied to *each* provider in turn.  Defaults to
        :class:`RetryPolicy()` with its own defaults.
    budget_per_call_usd:
        Optional ceiling (USD) for the estimated cost of a single call.
        A provider is skipped if its pre-call estimate exceeds this.
        Estimate is ``input_tokens_in_prompt + max_tokens_requested``.
    pricing:
        Optional :class:`PricingTable` override.  Defaults to the
        process-global table.
    """

    def __init__(
        self,
        *,
        primary: LLMProvider,
        fallbacks: Sequence[LLMProvider] = (),
        retry: RetryPolicy | None = None,
        budget_per_call_usd: Decimal | None = None,
        pricing: PricingTable | None = None,
    ) -> None:
        self._providers: tuple[LLMProvider, ...] = (primary, *fallbacks)
        self._retry = retry or RetryPolicy()
        self._budget = budget_per_call_usd
        self._pricing = pricing

    # ------------------------------------------------------------------
    # LLMProvider-ish surface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "router"

    @property
    def model(self) -> str:
        return ",".join(p.model for p in self._providers)

    @property
    def providers(self) -> tuple[LLMProvider, ...]:
        return self._providers

    def count_tokens(self, text: str) -> int:
        return self._providers[0].count_tokens(text)

    # ------------------------------------------------------------------
    # chat / structured
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        last_error: LLMProviderError | None = None
        skipped_for_budget: list[str] = []

        for provider in self._providers:
            if not self._within_budget(provider, messages, options):
                skipped_for_budget.append(provider.name)
                continue

            try:

                async def _call(p: LLMProvider = provider) -> ChatResponse:
                    return await p.chat(messages, options=options)

                return await self._retry.run(_call)
            except LLMProviderError as exc:
                # Both transient (retry exhausted) and permanent errors
                # fall through to the next provider.  Rationale: a
                # permanent error on Anthropic (e.g. invalid API key)
                # doesn't mean OpenAI will fail the same way — that's
                # the whole point of having a fallback chain.
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise LLMProviderError(
            "All providers skipped by budget gate before any call was made. "
            f"Budget-blocked: {skipped_for_budget}. "
            "Consider raising budget_per_call_usd or passing max_tokens."
        )

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T_Schema],
        *,
        options: ChatOptions | None = None,
    ) -> T_Schema:
        last_error: LLMProviderError | None = None
        skipped_for_budget: list[str] = []

        for provider in self._providers:
            if not self._within_budget(provider, messages, options):
                skipped_for_budget.append(provider.name)
                continue

            try:

                async def _call(p: LLMProvider = provider) -> T_Schema:
                    return await p.structured(messages, schema, options=options)

                return await self._retry.run(_call)
            except LLMProviderError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise LLMProviderError(
            "All providers skipped by budget gate before any call was made. "
            f"Budget-blocked: {skipped_for_budget}."
        )

    # ------------------------------------------------------------------
    # Budget gate
    # ------------------------------------------------------------------

    def _within_budget(
        self,
        provider: LLMProvider,
        messages: Sequence[Message],
        options: ChatOptions | None,
    ) -> bool:
        if self._budget is None:
            return True
        estimate = self._estimate_cost(provider, messages, options)
        return estimate <= self._budget

    def _estimate_cost(
        self,
        provider: LLMProvider,
        messages: Sequence[Message],
        options: ChatOptions | None,
    ) -> Decimal:
        """Pre-call cost estimate.

        Input tokens = sum over messages.  Output tokens = the
        ``max_tokens`` asked for, or a conservative 1024 guess.  We
        intentionally *over*-estimate (assume output hits the cap)
        so the budget gate is pessimistic.
        """
        input_tokens = sum(provider.count_tokens(f"{m.role.value}:{m.content}") for m in messages)
        output_tokens = options.max_tokens if options and options.max_tokens else 1024
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        return calculate_cost(provider.name, provider.model, usage, table=self._pricing)
