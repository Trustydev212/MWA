"""Smoke test for the OpenClaw team demo — M6 SDK edition.

Purpose: guarantee the demo keeps running end-to-end as the codebase
evolves.  It's a smoke test, not a unit test — the goal is to catch
*"I broke the demo by refactoring the SDK"* failures, not to verify
every decision the LLM makes.

We reuse the demo's own ``build_*`` factories and fake responses so
the demo script and the smoke test can never drift apart.  Every
assertion is about *structural* invariants (which nodes are set,
which agent wrote them, how many episodes are in the audit log) —
never about specific content that might reasonably change.
"""

from __future__ import annotations

import pytest

from examples.openclaw_team import fake_responses as fx
from examples.openclaw_team.agents import (
    build_architect,
    build_deployer,
    build_provider_selector,
    build_security,
)
from examples.openclaw_team.runtime import build_runtime, make_fake_provider
from mwa.arbiter import ResolutionDecision
from mwa.llm.providers import FakeProvider
from mwa.sdk import AgentRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def providers() -> dict[str, FakeProvider]:
    """Per-agent fake providers pre-loaded with happy-path responses."""
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


@pytest.fixture
def runtime(providers: dict[str, FakeProvider]) -> AgentRuntime:
    """Runtime with all four builder agents registered."""
    rt = build_runtime(providers["arbiter"])
    build_architect(rt, providers["architect"])
    build_security(rt, providers["security"])
    build_provider_selector(rt, providers["provider_selector"])
    build_deployer(rt, providers["deployer"])
    return rt


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------


async def test_full_demo_happy_path_and_conflict(
    runtime: AgentRuntime,
) -> None:
    """Drive every stage of the demo by hand, then assert final state.

    This mirrors :func:`examples.openclaw_team.run_demo.main` but with
    quiet execution and hard asserts instead of pretty printing.
    """
    # ------------------------------------------------------------------
    # Stage A — reactive happy path
    # ------------------------------------------------------------------
    await runtime.seed_world(
        "user_intent", "Build a research agent.", confidence=1.0
    )
    invocations = await runtime.run_until_idle()

    # Dispatcher fired exactly 4 handlers:
    #   user_intent  → architect
    #   agent_architecture → security + provider_selector (2 agents)
    #   memory_strategy → deployer
    assert invocations == 4

    # All 12 nodes must be set after stage A.
    expected_nodes = [
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
    ]
    for node in expected_nodes:
        fact = await runtime.world.read(node)
        assert fact is not None, f"missing expected node: {node}"

    # 12 clean writes + 1 user seed = 12 version bumps.  (Seed counts too.)
    assert runtime.world.version == 12

    # ------------------------------------------------------------------
    # Stage B — conflict on memory_strategy → SemanticArbiter decides
    # ------------------------------------------------------------------
    outcome, resolution = await runtime.submit_write(
        agent_id="architect_agent",
        node="memory_strategy",
        value="long_term_persistent",
        confidence=0.84,  # close to Security's 0.82 — forces escalation
    )

    # Canned arbiter decided existing (Security) wins → architect's
    # write must be rejected.
    assert outcome == "rejected"
    assert resolution is not None
    assert resolution.decision is ResolutionDecision.KEEP_EXISTING
    # Went through the semantic arbiter, not a rule.
    assert resolution.rule_applied is None
    # Scoring dict from the arbiter survived into the Resolution.
    assert resolution.scoring is not None
    assert "overall_confidence" in resolution.scoring

    # World state unchanged — memory_strategy still Security's value.
    final = await runtime.world.read("memory_strategy")
    assert final is not None
    assert final.episode.value == "short_term_conversation"
    assert final.episode.agent_id == "security_agent"

    # Version MUST NOT increment on rejection.
    assert runtime.world.version == 12

    # Audit log has 2 entries for memory_strategy — 1 applied, 1 rejected.
    history = await runtime.world.history(
        "memory_strategy", include_rejected=True
    )
    assert len(history) == 2
    applied = [f for f in history if not f.is_rejected]
    rejected = [f for f in history if f.is_rejected]
    assert len(applied) == 1
    assert len(rejected) == 1
    assert applied[0].episode.agent_id == "security_agent"
    assert rejected[0].episode.agent_id == "architect_agent"
    assert rejected[0].episode.value == "long_term_persistent"

    # Draining after the rejected write must be a no-op — the write
    # didn't apply, so no new events were enqueued.
    post_invocations = await runtime.run_until_idle()
    assert post_invocations == 0


async def test_architect_runs_before_downstream(
    runtime: AgentRuntime,
) -> None:
    """Regression guard: after user_intent is seeded, the architect's
    three fields must land before any other agent writes anything."""
    await runtime.seed_world("user_intent", "x")
    await runtime.run_until_idle()

    for node in ("agent_architecture", "sub_agent_count", "orchestration_pattern"):
        fact = await runtime.world.read(node)
        assert fact is not None, f"architect did not write {node}"
        assert fact.episode.agent_id == "architect_agent"


async def test_dispatcher_idempotent_second_drain(
    runtime: AgentRuntime,
) -> None:
    """Draining the queue a second time with no new events is a no-op."""
    await runtime.seed_world("user_intent", "x")
    first = await runtime.run_until_idle()
    second = await runtime.run_until_idle()
    assert first > 0
    assert second == 0
    assert runtime.pending_events == 0
