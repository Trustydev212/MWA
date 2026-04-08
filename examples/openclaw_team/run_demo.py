"""End-to-end OpenClaw multi-agent builder demo.

Run with::

    uv run python -m examples.openclaw_team.run_demo

or, from the repo root::

    PYTHONPATH=. python examples/openclaw_team/run_demo.py

What the demo does
------------------
A user says "Build me a research agent that reads arXiv papers and
extracts claims."  Four specialised OpenClaw sub-agents then
collaborate via a shared MWA world model to design that agent:

1. **Architect-Agent** — picks architecture + sub-agent count +
   orchestration pattern.
2. **Security-Agent** — picks tool permissions + memory strategy.
3. **Provider-Selector-Agent** — picks LLM backend + cost / latency
   budgets.
4. **Deployer-Agent** — picks deployment target + observability +
   error handling.

Agents never talk to each other directly — every decision lands in
the shared :class:`InMemoryWorldModel` and downstream agents read
their inputs from there.  This is the whole MWA value prop in
miniature.

Two stages
----------
**Stage A — happy path.**  The four agents run (Architect first, then
Security + Provider-Selector + Deployer concurrently).  All decisions
land cleanly.  Final state printed.

**Stage B — conflict + arbitration.**  We seed a *contradictory*
Architect write on ``memory_strategy`` (Architect changed its mind
and now wants a ``long_term_persistent`` store) with similar
confidence to Security's existing ``short_term_conversation``.  The
rule-based resolver can't decide so it escalates to the Semantic
Arbiter, which picks a winner and the world settles.

Offline / live mode
-------------------
By default everything runs with :class:`FakeProvider` — no network,
no API keys, deterministic output.  To run against a real OpenAI-
compatible endpoint (futrixapi, OpenAI, OpenRouter, …) set::

    MWA_LIVE_API_KEY=sk-...
    MWA_LIVE_BASE_URL=https://futrixapi.com/v1
    MWA_LIVE_MODEL=auto

The demo auto-detects those env vars and switches to the real
provider, retry-wrapped in an :class:`LLMRouter` the same way
production code does.
"""

from __future__ import annotations

import asyncio

from examples.openclaw_team import fake_responses as fx
from examples.openclaw_team.agents import (
    run_architect,
    run_deployer,
    run_provider_selector,
    run_security,
)
from examples.openclaw_team.runtime import (
    Runtime,
    build_runtime,
    is_live_mode,
    make_fake_provider,
    make_live_provider,
    write_to_world,
)
from mwa.llm.base import LLMProvider
from mwa.types import WriteProposal

USER_INTENT = (
    "Build a research agent that reads arXiv papers and extracts "
    "claims with confidence scores."
)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _banner(text: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {text}")
    print("═" * 68)


def _section(text: str) -> None:
    print(f"\n── {text}")


async def _print_world_state(runtime: Runtime, nodes: list[str]) -> None:
    print("\n  Final world state:")
    max_len = max(len(n) for n in nodes)
    for node in nodes:
        fact = await runtime.world.read(node)
        if fact is None:
            print(f"    {node:<{max_len}}  (unset)")
            continue
        value = fact.episode.value
        agent = fact.episode.agent_id
        conf = fact.episode.confidence
        print(f"    {node:<{max_len}}  {value!r:<26} by {agent:<24} conf={conf}")
    print(f"  World version: {runtime.world.version}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _build_providers() -> dict[str, LLMProvider]:
    """One provider per agent plus one for the arbiter.

    README promises each agent can use a different LLM provider — this
    is where that promise lives in the demo.  In live mode everyone
    shares the same router, but the *shape* of the wiring stays the
    same so a future user can easily swap individual providers.
    """
    if is_live_mode():
        shared = make_live_provider()
        return {
            "architect": shared,
            "security": shared,
            "provider_selector": shared,
            "deployer": shared,
            "arbiter": shared,
        }

    # Offline mode: per-agent FakeProvider pre-loaded with one
    # happy-path response each.  The `security` provider gets a
    # second queued response for the Stage B conflict round.
    return {
        "architect": make_fake_provider("architect", [fx.ARCHITECT_HAPPY]),
        "security": make_fake_provider(
            "security", [fx.SECURITY_HAPPY, fx.SECURITY_CONFLICT]
        ),
        "provider_selector": make_fake_provider(
            "provider_selector", [fx.PROVIDER_SELECTOR_HAPPY]
        ),
        "deployer": make_fake_provider("deployer", [fx.DEPLOYER_HAPPY]),
        "arbiter": make_fake_provider(
            "arbiter", [fx.ARBITER_DECIDES_SECURITY_WINS]
        ),
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


async def _stage_a_happy_path(
    runtime: Runtime, providers: dict[str, LLMProvider]
) -> None:
    _banner("Stage A — Happy path: four agents build an agent")
    print(f"  User intent: {USER_INTENT!r}")

    # Seed the user_intent — every other decision is downstream of this.
    await write_to_world(
        runtime,
        WriteProposal(
            agent_id="user",
            node="user_intent",
            value=USER_INTENT,
            confidence=1.0,
        ),
    )

    _section("Architect-Agent (sequential — everyone depends on it)")
    arch = await run_architect(runtime, providers["architect"])
    print(f"  → agent_architecture    = {arch.agent_architecture}")
    print(f"  → sub_agent_count       = {arch.sub_agent_count}")
    print(f"  → orchestration_pattern = {arch.orchestration_pattern}")
    print(f"  → confidence            = {arch.confidence}")
    print(f"  → reason                : {arch.reason}")

    _section("Security + Provider-Selector (concurrent — both read only from Architect)")
    sec, sel = await asyncio.gather(
        run_security(runtime, providers["security"]),
        run_provider_selector(runtime, providers["provider_selector"]),
    )
    print(f"  Security      → permissions={sec.tool_permissions}, memory={sec.memory_strategy}")
    print(
        f"  ProviderSel   → llm={sel.llm_provider}, cost=${sel.cost_budget_usd}, "
        f"latency={sel.latency_budget_ms}ms"
    )

    _section("Deployer-Agent (sequential — needs Security's memory + permissions)")
    dep = await run_deployer(runtime, providers["deployer"])
    print(
        f"  Deployer      → target={dep.deployment_target}, observability={dep.observability}, "
        f"error={dep.error_handling}"
    )

    await _print_world_state(
        runtime,
        [
            "user_intent",
            "agent_architecture",
            "sub_agent_count",
            "orchestration_pattern",
            "tool_permissions",
            "memory_strategy",
            "llm_provider",
            "cost_budget",
            "latency_budget",
            "deployment_target",
            "observability",
            "error_handling",
        ],
    )


async def _stage_b_conflict(
    runtime: Runtime, providers: dict[str, LLMProvider]
) -> None:
    _banner("Stage B — Conflict: Architect changes mind on memory_strategy")
    print(
        "  Architect-Agent now argues the research agent needs "
        "'long_term_persistent' memory\n"
        "  to cache paper abstracts across sessions.  Security already "
        "wrote 'short_term_conversation'."
    )

    # Architect writes a contradicting memory_strategy with confidence
    # close to Security's — forces the rule-based resolver to escalate.
    arch_outcome, arch_res = await write_to_world(
        runtime,
        WriteProposal(
            agent_id="architect_agent",
            node="memory_strategy",
            value="long_term_persistent",
            confidence=0.84,  # very close to Security's 0.82
        ),
    )
    print(f"\n  Architect's contradicting write → {arch_outcome.upper()}")
    if arch_res is not None:
        print(f"    resolver: {arch_res.rule_applied or 'semantic_arbiter'}")
        print(f"    reason  : {arch_res.reason}")

    final = await runtime.world.read("memory_strategy")
    assert final is not None
    print(f"\n  Final memory_strategy  = {final.episode.value!r}")
    print(f"  Winning agent          = {final.episode.agent_id}")
    print(f"  World version          = {runtime.world.version}")

    # Audit trail — rejected writes are not gone, just hidden.
    history = await runtime.world.history("memory_strategy", include_rejected=True)
    print(f"\n  Audit trail for memory_strategy ({len(history)} episodes):")
    for fact in history:
        marker = "✓" if fact.is_current else ("✗" if fact.is_rejected else "·")
        print(
            f"    {marker} {fact.episode.value!r:<26} "
            f"by {fact.episode.agent_id:<24} conf={fact.episode.confidence} "
            f"rejected={fact.is_rejected}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    mode = "LIVE" if is_live_mode() else "OFFLINE (FakeProvider)"
    print(f"OpenClaw team demo — mode: {mode}")

    providers = _build_providers()
    runtime = build_runtime(providers["arbiter"])

    await _stage_a_happy_path(runtime, providers)
    await _stage_b_conflict(runtime, providers)

    _banner("Demo complete")
    print(
        "  All four agents wrote their decisions to the shared World Model.\n"
        "  The Semantic Arbiter resolved the memory_strategy conflict.\n"
        "  Audit log retained every write — applied and rejected."
    )


if __name__ == "__main__":
    asyncio.run(main())
