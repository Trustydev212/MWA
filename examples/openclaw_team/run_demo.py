"""End-to-end OpenClaw multi-agent builder demo — M6 SDK edition.

Run with::

    uv run python -m examples.openclaw_team.run_demo

or, from the repo root::

    PYTHONPATH=. python examples/openclaw_team/run_demo.py

How the SDK version differs from the pre-M6 version
----------------------------------------------------
The pre-SDK version manually called ``run_architect`` → ``gather(...)``
→ ``run_deployer`` in a fixed order.  The SDK version is **reactive**:
you seed ``user_intent`` and the dispatcher cascades every handler
automatically in topology order.

The bulk of ``main()`` is now:

1. Build four :class:`~mwa.sdk.WorldAgent` instances (one per builder).
2. Seed ``user_intent``.
3. Call ``runtime.run_until_idle()`` — dispatcher handles the rest.

Pretty-printing adds the visual narrative, but it's optional glue.
"""

from __future__ import annotations

import asyncio

from examples.openclaw_team import fake_responses as fx
from examples.openclaw_team.agents import (
    build_architect,
    build_deployer,
    build_provider_selector,
    build_security,
)
from examples.openclaw_team.runtime import (
    build_runtime,
    is_live_mode,
    make_fake_provider,
    make_live_provider,
)
from mwa.llm.base import LLMProvider
from mwa.sdk import AgentRuntime

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


async def _print_world_state(runtime: AgentRuntime, nodes: list[str]) -> None:
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

    In live mode everyone shares the same retry-wrapped router — it's
    the same gateway behind all agents, and the demo is too small to
    justify separate API keys.  The shape still supports per-agent
    providers (README promise) if you want it.

    In offline mode every agent gets its own FakeProvider with canned
    responses, so concurrent dispatch can't shuffle queues.
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
    runtime: AgentRuntime,
    providers: dict[str, LLMProvider],
) -> None:
    _banner("Stage A — Reactive happy path: seed + run_until_idle()")
    print(f"  User intent: {USER_INTENT!r}")
    print(
        "\n  The four builder agents are registered on different nodes.\n"
        "  When user_intent lands, the dispatcher fires handlers in\n"
        "  topology order — no manual orchestration from the demo."
    )

    # Build agents.  Each constructor self-registers with the runtime
    # and attaches its ``@agent.on("...")`` handler.
    build_architect(runtime, providers["architect"])
    build_security(runtime, providers["security"])
    build_provider_selector(runtime, providers["provider_selector"])
    build_deployer(runtime, providers["deployer"])

    _section("Seeding user_intent + draining event queue")
    await runtime.seed_world("user_intent", USER_INTENT)
    invocations = await runtime.run_until_idle()
    print(f"  → dispatcher fired {invocations} handler invocations")

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


async def _stage_b_conflict(runtime: AgentRuntime) -> None:
    _banner("Stage B — Conflict: Architect changes mind on memory_strategy")
    print(
        "  Architect-Agent now argues the research agent needs "
        "'long_term_persistent' memory\n"
        "  to cache paper abstracts across sessions.  Security already "
        "wrote 'short_term_conversation'."
    )

    # Architect writes a contradicting memory_strategy with confidence
    # close to Security's — forces the rule-based resolver to escalate.
    outcome, resolution = await runtime.submit_write(
        agent_id="architect_agent",
        node="memory_strategy",
        value="long_term_persistent",
        confidence=0.84,  # very close to Security's 0.82
    )
    print(f"\n  Architect's contradicting write → {outcome.upper()}")
    if resolution is not None:
        print(f"    resolver: {resolution.rule_applied or 'semantic_arbiter'}")
        print(f"    reason  : {resolution.reason}")

    # Drain any downstream reactions that may have been enqueued by
    # the write (none, for a rejected write, but we still call it so
    # the demo keeps the "always drain after a write" discipline).
    await runtime.run_until_idle()

    final = await runtime.world.read("memory_strategy")
    assert final is not None
    print(f"\n  Final memory_strategy  = {final.episode.value!r}")
    print(f"  Winning agent          = {final.episode.agent_id}")
    print(f"  World version          = {runtime.world.version}")

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
    print(f"OpenClaw team demo (M6 SDK edition) — mode: {mode}")

    providers = _build_providers()
    runtime = build_runtime(providers["arbiter"])

    await _stage_a_happy_path(runtime, providers)
    await _stage_b_conflict(runtime)

    _banner("Demo complete")
    print(
        "  All four agents wrote their decisions to the shared World Model.\n"
        "  The Semantic Arbiter resolved the memory_strategy conflict.\n"
        "  Audit log retained every write — applied and rejected."
    )


if __name__ == "__main__":
    asyncio.run(main())
