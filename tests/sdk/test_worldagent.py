"""Unit tests for the M6 :class:`WorldAgent` SDK layer.

Every test uses :class:`FakeProvider` so there's no network, no keys,
and deterministic output.  The goal is to pin the *contract* the SDK
promises — the dispatcher behaviour, the write pipeline integration,
the decorator registration rules — not to re-test what lower layers
already cover.
"""

from __future__ import annotations

import pytest

from mwa.arbiter import ResolutionDecision
from mwa.harness import HarnessMap
from mwa.llm.providers import FakeProvider
from mwa.sdk import AgentContext, AgentRuntime, WorldAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness() -> HarnessMap:
    """Tiny three-node harness: root → mid → leaf."""
    return HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "test",
            "nodes": {
                "root": {"impact": "critical", "affects": ["mid"], "order": 1},
                "mid": {"impact": "high", "affects": ["leaf"], "order": 2},
                "leaf": {"impact": "medium", "affects": [], "order": 3},
            },
            "hard_constraints": [],
        }
    )


@pytest.fixture
def arbiter_llm() -> FakeProvider:
    """Arbiter LLM is only invoked on escalation; tests queue when needed."""
    return FakeProvider(name="arbiter-fake", model="arbiter-test")


@pytest.fixture
def runtime(harness: HarnessMap, arbiter_llm: FakeProvider) -> AgentRuntime:
    return AgentRuntime(harness=harness, arbiter_llm=arbiter_llm)


# ---------------------------------------------------------------------------
# Agent registration + handler decoration
# ---------------------------------------------------------------------------


def test_agent_auto_registers_with_runtime(runtime: AgentRuntime) -> None:
    """Constructing a WorldAgent joins the runtime — no explicit register call."""
    agent = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    assert "a" in runtime.agents
    assert runtime.agents["a"] is agent


def test_duplicate_agent_name_rejected(runtime: AgentRuntime) -> None:
    WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    with pytest.raises(ValueError, match="Duplicate agent name"):
        WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)


def test_empty_name_rejected(runtime: AgentRuntime) -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        WorldAgent(name="", llm=FakeProvider(), runtime=runtime)


def test_on_decorator_validates_node_exists(runtime: AgentRuntime) -> None:
    """Typo in node name must fail at registration, not dispatch."""
    agent = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    with pytest.raises(ValueError, match="unknown node"):

        @agent.on("typo_node")  # not in harness map
        async def bad(ctx: AgentContext) -> None:
            pass


def test_duplicate_handler_on_same_agent_rejected(runtime: AgentRuntime) -> None:
    agent = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)

    @agent.on("root")
    async def first(ctx: AgentContext) -> None:
        pass

    with pytest.raises(ValueError, match="already has a handler"):

        @agent.on("root")
        async def second(ctx: AgentContext) -> None:
            pass


def test_handlers_property_returns_copy(runtime: AgentRuntime) -> None:
    agent = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)

    @agent.on("root")
    async def h(ctx: AgentContext) -> None:
        pass

    snapshot = agent.handlers
    snapshot["mid"] = h  # type: ignore[assignment]
    # Real handlers dict must not be mutated.
    assert "mid" not in agent.handlers


# ---------------------------------------------------------------------------
# Seed + dispatch flow
# ---------------------------------------------------------------------------


async def test_seed_world_enqueues_event_and_handler_fires(
    runtime: AgentRuntime,
) -> None:
    agent = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    calls: list[tuple[str, object]] = []

    @agent.on("root")
    async def h(ctx: AgentContext) -> None:
        calls.append((ctx.trigger_node, ctx.trigger_value))

    await runtime.seed_world("root", "x")
    assert runtime.pending_events == 1

    invocations = await runtime.run_until_idle()
    assert invocations == 1
    assert calls == [("root", "x")]
    assert runtime.pending_events == 0


async def test_seed_with_no_handler_is_noop_drain(runtime: AgentRuntime) -> None:
    """seed_world enqueues even when nobody reacts — drain completes cleanly."""
    await runtime.seed_world("root", "x")
    invocations = await runtime.run_until_idle()
    assert invocations == 0


async def test_handler_writes_enqueue_downstream_event(
    runtime: AgentRuntime,
) -> None:
    """Agent A reacts to root, writes mid.  Agent B reacts to mid."""
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    b = WorldAgent(name="b", llm=FakeProvider(), runtime=runtime)
    b_calls: list[object] = []

    @a.on("root")
    async def on_root(ctx: AgentContext) -> None:
        await ctx.world.write("mid", "derived_from_root", confidence=0.9)

    @b.on("mid")
    async def on_mid(ctx: AgentContext) -> None:
        b_calls.append(ctx.trigger_value)

    await runtime.seed_world("root", "x")
    invocations = await runtime.run_until_idle()

    # Two invocations: a.on_root + b.on_mid
    assert invocations == 2
    assert b_calls == ["derived_from_root"]

    mid = await runtime.world.read("mid")
    assert mid is not None
    assert mid.episode.agent_id == "a"
    assert mid.episode.value == "derived_from_root"


async def test_multiple_agents_react_to_same_node(runtime: AgentRuntime) -> None:
    """Two agents both register on the same node → both fire."""
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    b = WorldAgent(name="b", llm=FakeProvider(), runtime=runtime)
    fired: list[str] = []

    @a.on("root")
    async def ah(ctx: AgentContext) -> None:
        fired.append("a")

    @b.on("root")
    async def bh(ctx: AgentContext) -> None:
        fired.append("b")

    await runtime.seed_world("root", "x")
    await runtime.run_until_idle()
    assert set(fired) == {"a", "b"}


async def test_agent_does_not_react_to_own_writes(runtime: AgentRuntime) -> None:
    """Skipping self-triggers prevents accidental infinite loops.

    Agent A handles root, and inside its handler writes root again.
    The handler must NOT fire a second time for its own write.
    """
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    fire_count = {"n": 0}

    @a.on("root")
    async def h(ctx: AgentContext) -> None:
        fire_count["n"] += 1
        # Only bounce once — otherwise we'd loop forever if the skip
        # wasn't working.  High confidence so the rule-based resolver
        # fires dominance, not escalation (no arbiter LLM queued).
        if fire_count["n"] == 1:
            await ctx.world.write("root", "bounced", confidence=0.95)

    # Seed at low confidence so the 0.95 write cleanly dominates.
    await runtime.seed_world("root", "original", confidence=0.5)
    await runtime.run_until_idle()
    # Fired once for the seed, not a second time for its own write.
    assert fire_count["n"] == 1


async def test_dispatcher_raises_on_runaway_loop() -> None:
    """Self-reaction skip is per-agent.  A → B → A cycle will loop —
    the runtime catches this via ``max_iterations``.

    Uses idempotent writes (same value every time) so the conflict
    detector passes through without needing the semantic arbiter to
    have queued responses."""
    harness = HarnessMap.from_dict(
        {
            "version": "1.0",
            "domain": "t",
            "nodes": {
                "x": {"impact": "high", "affects": ["y"], "order": 1},
                "y": {"impact": "high", "affects": [], "order": 2},
            },
        }
    )
    runtime = AgentRuntime(harness=harness, arbiter_llm=FakeProvider())
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    b = WorldAgent(name="b", llm=FakeProvider(), runtime=runtime)

    @a.on("x")
    async def a_on_x(ctx: AgentContext) -> None:
        # Same value as the seed → no conflict → applies idempotently
        # → enqueues a new event for y.
        await ctx.world.write("y", "const_y", confidence=0.9)

    @b.on("y")
    async def b_on_y(ctx: AgentContext) -> None:
        # Same value as the seed → idempotent apply → enqueues x again,
        # which re-triggers A, which enqueues y, …
        await ctx.world.write("x", "const_x", confidence=0.9)

    await runtime.seed_world("x", "const_x")
    with pytest.raises(RuntimeError, match="exceeded max_iterations"):
        await runtime.run_until_idle(max_iterations=10)


# ---------------------------------------------------------------------------
# Write pipeline integration (rule-based + semantic)
# ---------------------------------------------------------------------------


async def test_rejected_write_does_not_enqueue_event(
    runtime: AgentRuntime,
) -> None:
    """A losing write should NOT trigger downstream reactions."""
    # Agent "a" exists purely to provide an origin for the rejected write
    # below; its own handlers are empty.  We still construct it so the
    # runtime has the agent_id on file.
    WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    b = WorldAgent(name="b", llm=FakeProvider(), runtime=runtime)
    b_calls: list[object] = []

    @b.on("root")
    async def bh(ctx: AgentContext) -> None:
        b_calls.append(ctx.trigger_value)

    # Seed with high confidence.
    await runtime.seed_world("root", "first", confidence=0.95)
    await runtime.run_until_idle()
    assert b_calls == ["first"]

    # Agent A tries to overwrite with low confidence — rule-based
    # dominance rejects it.  No new event for B.
    outcome, resolution = await runtime.submit_write(
        agent_id="a", node="root", value="second", confidence=0.3
    )
    assert outcome == "rejected"
    assert resolution is not None
    assert resolution.decision is ResolutionDecision.KEEP_EXISTING

    await runtime.run_until_idle()
    # B still only saw the first value.
    assert b_calls == ["first"]


async def test_context_write_uses_agent_id(runtime: AgentRuntime) -> None:
    """Writes via ``ctx.world.write`` must be stamped with the owning agent."""
    a = WorldAgent(name="alice", llm=FakeProvider(), runtime=runtime)

    @a.on("root")
    async def h(ctx: AgentContext) -> None:
        await ctx.world.write("mid", "x", confidence=0.9)

    await runtime.seed_world("root", "trigger")
    await runtime.run_until_idle()

    mid = await runtime.world.read("mid")
    assert mid is not None
    assert mid.episode.agent_id == "alice"


async def test_context_read_returns_current_fact(runtime: AgentRuntime) -> None:
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    seen: list[object] = []

    @a.on("mid")
    async def h(ctx: AgentContext) -> None:
        fact = await ctx.world.read("root")
        seen.append(fact.episode.value if fact else None)

    await runtime.seed_world("root", "hello")
    await runtime.seed_world("mid", "trigger")
    await runtime.run_until_idle()
    assert seen == ["hello"]


async def test_context_exposes_agent_id_and_llm(runtime: AgentRuntime) -> None:
    my_llm = FakeProvider(name="my-model", model="x")
    a = WorldAgent(name="alice", llm=my_llm, runtime=runtime)
    captured: dict[str, object] = {}

    @a.on("root")
    async def h(ctx: AgentContext) -> None:
        captured["agent_id"] = ctx.agent_id
        captured["llm_name"] = ctx.llm.name
        captured["trigger"] = ctx.trigger_node

    await runtime.seed_world("root", "x")
    await runtime.run_until_idle()

    assert captured["agent_id"] == "alice"
    assert captured["llm_name"] == "my-model"
    assert captured["trigger"] == "root"


async def test_harness_accessible_through_context(runtime: AgentRuntime) -> None:
    a = WorldAgent(name="a", llm=FakeProvider(), runtime=runtime)
    seen_downstream: list[list[str]] = []

    @a.on("root")
    async def h(ctx: AgentContext) -> None:
        seen_downstream.append(ctx.world.harness.traverse_downstream("root"))

    await runtime.seed_world("root", "x")
    await runtime.run_until_idle()
    assert seen_downstream == [["mid", "leaf"]]


# ---------------------------------------------------------------------------
# seed_world default confidence
# ---------------------------------------------------------------------------


async def test_seed_world_default_agent_is_user(runtime: AgentRuntime) -> None:
    await runtime.seed_world("root", "x")
    fact = await runtime.world.read("root")
    assert fact is not None
    assert fact.episode.agent_id == "user"
    assert fact.episode.confidence == 1.0
