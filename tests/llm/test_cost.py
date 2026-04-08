"""Tests for the pricing / cost calculation layer."""

from __future__ import annotations

from decimal import Decimal

from mwa.llm import PricingEntry, PricingTable, Usage, calculate_cost


def test_pricing_entry_zero_usage_is_zero() -> None:
    entry = PricingEntry(Decimal("3.00"), Decimal("15.00"))
    assert entry.cost(Usage()) == Decimal(0)


def test_pricing_entry_computes_proportional_cost() -> None:
    entry = PricingEntry(Decimal("3.00"), Decimal("15.00"))
    # 1M input + 1M output = 3 + 15 = 18 USD
    assert entry.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == Decimal("18.00")
    # 500k input only = 1.50 USD
    assert entry.cost(Usage(input_tokens=500_000)) == Decimal("1.50")


def test_table_exact_match_wins_over_wildcard() -> None:
    table = PricingTable(entries={})
    table.set("foo", "*", input_per_million=1, output_per_million=2)
    table.set("foo", "special", input_per_million=10, output_per_million=20)

    assert table.get("foo", "special") == PricingEntry(Decimal("10"), Decimal("20"))
    assert table.get("foo", "other") == PricingEntry(Decimal("1"), Decimal("2"))


def test_table_returns_none_for_unknown_provider() -> None:
    table = PricingTable(entries={})
    assert table.get("unknown", "model") is None


def test_calculate_cost_uses_default_table() -> None:
    """Default table has an entry for ``fake`` (zero) — should not crash."""
    assert calculate_cost("fake", "fake-1", Usage(input_tokens=100)) == Decimal(0)


def test_calculate_cost_custom_table() -> None:
    table = PricingTable(entries={})
    table.set("x", "y", input_per_million="4.0", output_per_million="8.0")
    usage = Usage(input_tokens=500_000, output_tokens=250_000)
    # 4 * 0.5 + 8 * 0.25 = 2 + 2 = 4
    assert calculate_cost("x", "y", usage, table=table) == Decimal("4.00")


def test_calculate_cost_unknown_model_returns_zero() -> None:
    assert calculate_cost("nonexistent", "model", Usage(input_tokens=1000)) == Decimal(0)


def test_placeholder_defaults_include_local_providers_as_free() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_cost("ollama", "llama3", usage) == Decimal(0)
    assert calculate_cost("vllm", "mistral", usage) == Decimal(0)
