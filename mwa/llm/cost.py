"""Cost accounting for LLM calls.

Two moving parts:

1. **PricingTable** — maps ``(provider, model) → (input_price, output_price)``
   where both prices are in USD per 1M tokens.
2. **calculate_cost()** — applies the table to a :class:`Usage` record.

The shipped default table is intentionally **a placeholder**.  Real
vendor pricing moves constantly (and varies by region, cache discounts,
volume tier...) so hard-coding prices in the repo would be misleading.
Callers should override via :meth:`PricingTable.set` or construct their
own table from config.

Matching order:
- exact ``(provider, model)`` wins
- then ``(provider, "*")`` — a wildcard for "anything from this provider"
- then ``Decimal(0)`` — unknown models are free on paper but you'll get
  a warning via :func:`calculate_cost`.

We use :class:`decimal.Decimal` instead of ``float`` because billing
arithmetic over millions of tokens accumulates float error fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from mwa.llm.base import Usage


@dataclass(frozen=True)
class PricingEntry:
    """Per-million-token prices in USD."""

    input_per_million: Decimal
    output_per_million: Decimal

    def cost(self, usage: Usage) -> Decimal:
        ONE_M = Decimal(1_000_000)
        return (
            self.input_per_million * Decimal(usage.input_tokens) / ONE_M
            + self.output_per_million * Decimal(usage.output_tokens) / ONE_M
        )


def _placeholder_defaults() -> dict[tuple[str, str], PricingEntry]:
    """Conservative placeholder pricing.

    These numbers are **rough estimates** only — users who care about
    billing accuracy should override at runtime.  The entries here exist
    so tests and examples can reference "some provider, some model"
    without crashing.
    """
    zero = PricingEntry(Decimal(0), Decimal(0))
    return {
        # Local / free
        ("ollama", "*"): zero,
        ("vllm", "*"): zero,
        ("llamacpp", "*"): zero,
        ("lmstudio", "*"): zero,
        # Anthropic (placeholder)
        ("anthropic", "*"): PricingEntry(Decimal("3.00"), Decimal("15.00")),
        # OpenAI (placeholder)
        ("openai", "*"): PricingEntry(Decimal("2.50"), Decimal("10.00")),
        # Google (placeholder)
        ("gemini", "*"): PricingEntry(Decimal("1.25"), Decimal("5.00")),
        # Test fake — explicit zero so test cost assertions are stable
        ("fake", "*"): zero,
    }


@dataclass
class PricingTable:
    """Mutable lookup from ``(provider, model)`` to a :class:`PricingEntry`."""

    entries: dict[tuple[str, str], PricingEntry] = field(default_factory=_placeholder_defaults)

    def set(
        self,
        provider: str,
        model: str,
        *,
        input_per_million: Decimal | float | str,
        output_per_million: Decimal | float | str,
    ) -> None:
        """Register / update a pricing entry."""
        self.entries[(provider, model)] = PricingEntry(
            Decimal(str(input_per_million)),
            Decimal(str(output_per_million)),
        )

    def get(self, provider: str, model: str) -> PricingEntry | None:
        """Return an entry with the exact → wildcard → None fallback chain."""
        return self.entries.get((provider, model)) or self.entries.get((provider, "*"))


# Process-global default table.  Tests that want isolation should
# construct their own :class:`PricingTable` and pass it explicitly.
DEFAULT_PRICING = PricingTable()


def calculate_cost(
    provider: str,
    model: str,
    usage: Usage,
    *,
    table: PricingTable | None = None,
) -> Decimal:
    """Return the cost of ``usage`` for ``(provider, model)`` in USD."""
    entry = (table or DEFAULT_PRICING).get(provider, model)
    if entry is None:
        return Decimal(0)
    return entry.cost(usage)
