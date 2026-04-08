"""Smoke test for the OpenClaw team demo.

Purpose: guarantee the demo keeps running end-to-end as the codebase
evolves.  It's a smoke test, not a unit test — the goal is to catch
*"I broke the demo by refactoring the SDK"* failures, not to verify
every decision the LLM makes.

We use the module's own :mod:`fake_responses` so the demo script and
the smoke test can never drift apart.  Every assertion is about
*structural* invariants (which nodes are set, which agent wrote them,
how many episodes are in the audit log) — never about specific
content that might reasonably change between fake-response updates.
"""

from __future__ import annotations

import pytest

from examples.openclaw_team import fake_responses as fx
from examples.openclaw_team.agents import (
    run_architect,
    run_deployer,
    run_provider_selector,
    run_security,
)
from examples.openclaw_team.runtime import (
    build_runtime,
    make_fake_provider,
    write_to_world,
)
from mwa.arbiter import ResolutionDecision
from mwa.llm.providers import FakeProvider
from mwa.types import WriteProposal

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


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


async def test_full_demo_happy_path_and_conflict(
    providers: dict[str, FakeProvider],
) -> None:
    """Drive every stage of the demo by hand, then assert final state.

    This mirrors :func:`examples.openclaw_team.run_demo.main` but with
    quiet execution and hard asserts instead of pretty printing.
    """
    runtime = build_runtime(providers["arbiter"])

    # ------------------------------------------------------------------
    # Stage A — happy path
    # ------------------------------------------------------------------
    await write_to_world(
        runtime,
        WriteProposal(
            agent_id="user",
            node="user_intent",
            value="Build a research agent.",
            confidence=1.0,
        ),
    )

    arch = await run_architect(runtime, providers["architect"])
    assert arch.agent_architecture == "multi_agent"
    assert arch.sub_agent_count == 3

    sec = await run_security(runtime, providers["security"])
    assert sec.tool_permissions == "read_only"
    assert sec.memory_strategy == "short_term_conversation"

    sel = await run_provider_selector(runtime, providers["provider_selector"])
    assert sel.llm_provider == "anthropic_claude"

    dep = await run_deployer(runtime, providers["deployer"])
    assert dep.deployment_target == "cloud"

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

    # World version == number of applied writes (12 clean writes so far).
    assert runtime.world.version == 12

    # ------------------------------------------------------------------
    # Stage B — conflict on memory_strategy → SemanticArbiter decides
    # ------------------------------------------------------------------
    outcome, resolution = await write_to_world(
        runtime,
        WriteProposal(
            agent_id="architect_agent",
            node="memory_strategy",
            value="long_term_persistent",
            confidence=0.84,  # close to Security's 0.82 — forces escalation
        ),
    )

    # Canned arbiter decided existing (Security) wins → architect's
    # write must be rejected.
    assert outcome == "rejected"
    assert resolution is not None
    assert resolution.decision is ResolutionDecision.KEEP_EXISTING
    # Not a rule-based decision — must have gone through the semantic
    # arbiter (rule_applied is None for LLM paths).
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


async def test_architect_writes_everything_before_downstream(
    providers: dict[str, FakeProvider],
) -> None:
    """Regression guard: architect's 3 fields must all be readable
    before Security / Provider-Selector / Deployer run.

    If someone breaks the schema or the write order, downstream
    agents fail with `_read_required` raising.  This test pins the
    contract explicitly.
    """
    runtime = build_runtime(providers["arbiter"])
    await write_to_world(
        runtime,
        WriteProposal(
            agent_id="user",
            node="user_intent",
            value="x",
            confidence=1.0,
        ),
    )

    await run_architect(runtime, providers["architect"])

    for node in ("agent_architecture", "sub_agent_count", "orchestration_pattern"):
        fact = await runtime.world.read(node)
        assert fact is not None, f"architect did not write {node}"
        assert fact.episode.agent_id == "architect_agent"
