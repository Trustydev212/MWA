"""The four OpenClaw builder agents — M6 WorldAgent SDK version.

Each agent is now a :class:`~mwa.sdk.WorldAgent` with one or more
``@agent.on("node")`` handlers.  Compared to the pre-SDK version of
this file:

- No ``_read_required`` helper — handlers read directly from
  ``ctx.world``.
- No ``_write_field`` helper — ``ctx.world.write`` already stamps
  the agent id and runs the full conflict pipeline.
- No explicit orchestration — the runtime dispatcher fires handlers
  automatically as upstream nodes change.
- No :class:`Runtime` dataclass parameter — each agent holds a back-
  reference to its :class:`AgentRuntime` via the SDK.

Lines of code per agent dropped from ~60 to ~30.  The pattern that
used to live across ``agents.py`` + ``runtime.py`` glue is now
behind the SDK.

The prompts themselves are unchanged from the pre-SDK version — this
refactor is strictly about shape, not content.
"""

from __future__ import annotations

from examples.openclaw_team.schemas import (
    ArchitectDecision,
    DeployerDecision,
    ProviderSelectorDecision,
    SecurityDecision,
)
from mwa.llm.base import ChatOptions, LLMProvider, Message, MessageRole
from mwa.sdk import AgentContext, AgentRuntime, WorldAgent

AGENT_CHAT_OPTS = ChatOptions(temperature=0.0, max_tokens=1024)


# ---------------------------------------------------------------------------
# Architect
# ---------------------------------------------------------------------------


def build_architect(runtime: AgentRuntime, llm: LLMProvider) -> WorldAgent:
    """Construct Architect-Agent and register its handler."""
    agent = WorldAgent(name="architect_agent", llm=llm, runtime=runtime)

    @agent.on("user_intent")
    async def design(ctx: AgentContext) -> None:
        intent = ctx.trigger_value
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are the Architect-Agent inside OpenClaw, a meta-agent "
                    "builder.  Your sole job is to pick the structural layout for "
                    "the agent the user wants built — nothing else.  Return an "
                    "ArchitectDecision object.\n"
                    "\n"
                    "Heuristics:\n"
                    "  - A simple research/QA task: single_agent, 1 sub-agent.\n"
                    "  - A task with distinct roles (plan+execute, code+test): multi_agent.\n"
                    "  - A task with dispatch + fan-out: hierarchical.\n"
                    "  - Default orchestration to sequential unless the task has "
                    "genuinely independent sub-tasks."
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=f"user_intent: {intent!r}\n\nReturn your ArchitectDecision.",
            ),
        ]
        decision = await ctx.llm.structured(
            messages, ArchitectDecision, options=AGENT_CHAT_OPTS
        )

        await ctx.world.write(
            "agent_architecture",
            decision.agent_architecture,
            confidence=decision.confidence,
        )
        await ctx.world.write(
            "sub_agent_count",
            decision.sub_agent_count,
            confidence=decision.confidence,
        )
        await ctx.world.write(
            "orchestration_pattern",
            decision.orchestration_pattern,
            confidence=decision.confidence,
        )

    return agent


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def build_security(runtime: AgentRuntime, llm: LLMProvider) -> WorldAgent:
    agent = WorldAgent(name="security_agent", llm=llm, runtime=runtime)

    @agent.on("agent_architecture")
    async def audit(ctx: AgentContext) -> None:
        # Read peer context that the handler needs.
        intent_fact = await ctx.world.read("user_intent")
        assert intent_fact is not None

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are the Security-Agent inside OpenClaw.  Your job is to "
                    "pick tool_permissions and memory_strategy for the agent "
                    "being built, with defence-in-depth as the default.\n"
                    "\n"
                    "Heuristics:\n"
                    "  - 'read_only' unless the intent explicitly needs to write.\n"
                    "  - 'stateless' for pure transforms; 'short_term_conversation' "
                    "for chat-like; 'long_term_persistent' only when the user "
                    "explicitly needs cross-session memory.\n"
                    "  - multi_agent / hierarchical architectures often need at "
                    "least 'short_term_conversation' to share context."
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"user_intent: {intent_fact.episode.value!r}\n"
                    f"agent_architecture: {ctx.trigger_value!r}\n\n"
                    "Return your SecurityDecision."
                ),
            ),
        ]
        decision = await ctx.llm.structured(
            messages, SecurityDecision, options=AGENT_CHAT_OPTS
        )

        await ctx.world.write(
            "tool_permissions",
            decision.tool_permissions,
            confidence=decision.confidence,
        )
        await ctx.world.write(
            "memory_strategy",
            decision.memory_strategy,
            confidence=decision.confidence,
        )

    return agent


# ---------------------------------------------------------------------------
# Provider Selector
# ---------------------------------------------------------------------------


def build_provider_selector(
    runtime: AgentRuntime, llm: LLMProvider
) -> WorldAgent:
    agent = WorldAgent(name="provider_selector_agent", llm=llm, runtime=runtime)

    @agent.on("agent_architecture")
    async def pick(ctx: AgentContext) -> None:
        intent_fact = await ctx.world.read("user_intent")
        sub_count_fact = await ctx.world.read("sub_agent_count")
        assert intent_fact is not None
        assert sub_count_fact is not None

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are the Provider-Selector-Agent inside OpenClaw.  Your "
                    "job is to pick the LLM provider, cost budget, and latency "
                    "budget for the agent being built.\n"
                    "\n"
                    "Heuristics:\n"
                    "  - anthropic_claude: best reasoning, higher cost.\n"
                    "  - openai_gpt4o: strong all-round, moderate cost.\n"
                    "  - google_gemini: cheap + long context.\n"
                    "  - local_llama: free, offline, lower quality.\n"
                    "  - More sub-agents → tighter per-agent budget to keep total "
                    "sane.\n"
                    "  - Latency budgets scale with sub-agent count for sequential "
                    "orchestration."
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"user_intent: {intent_fact.episode.value!r}\n"
                    f"agent_architecture: {ctx.trigger_value!r}\n"
                    f"sub_agent_count: {sub_count_fact.episode.value}\n\n"
                    "Return your ProviderSelectorDecision."
                ),
            ),
        ]
        decision = await ctx.llm.structured(
            messages, ProviderSelectorDecision, options=AGENT_CHAT_OPTS
        )

        await ctx.world.write(
            "llm_provider", decision.llm_provider, confidence=decision.confidence
        )
        await ctx.world.write(
            "cost_budget",
            decision.cost_budget_usd,
            confidence=decision.confidence,
        )
        await ctx.world.write(
            "latency_budget",
            decision.latency_budget_ms,
            confidence=decision.confidence,
        )

    return agent


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------


def build_deployer(runtime: AgentRuntime, llm: LLMProvider) -> WorldAgent:
    """Deployer reacts to memory_strategy — the last Security-Agent write.

    By the time memory_strategy lands, every other Security output
    (tool_permissions) is already in the world, and so is the Architect's
    agent_architecture.  Reacting to memory_strategy gives us a single
    reliable trigger for "everything the deployer needs is ready".
    """
    agent = WorldAgent(name="deployer_agent", llm=llm, runtime=runtime)

    @agent.on("memory_strategy")
    async def deploy(ctx: AgentContext) -> None:
        intent_fact = await ctx.world.read("user_intent")
        arch_fact = await ctx.world.read("agent_architecture")
        perms_fact = await ctx.world.read("tool_permissions")
        assert intent_fact is not None
        assert arch_fact is not None
        assert perms_fact is not None

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are the Deployment-Agent inside OpenClaw.  Your job is "
                    "to pick deployment_target, observability, and error_handling "
                    "for the agent being built.\n"
                    "\n"
                    "Heuristics:\n"
                    "  - Respect the harness constraint: memory_strategy = "
                    "'long_term_persistent' is INCOMPATIBLE with deployment_target "
                    "= 'edge'.  Pick 'local' or 'cloud' in that case.\n"
                    "  - Respect the harness constraint: tool_permissions = "
                    "'admin' is INCOMPATIBLE with error_handling = 'retry'.  Pick "
                    "'fail_fast' or 'human_escalation' when admin permissions are "
                    "in play.\n"
                    "  - Choose 'traces' for multi_agent systems — you'll need "
                    "them to debug cross-agent flows."
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"user_intent: {intent_fact.episode.value!r}\n"
                    f"agent_architecture: {arch_fact.episode.value!r}\n"
                    f"memory_strategy: {ctx.trigger_value!r}\n"
                    f"tool_permissions: {perms_fact.episode.value!r}\n\n"
                    "Return your DeployerDecision."
                ),
            ),
        ]
        decision = await ctx.llm.structured(
            messages, DeployerDecision, options=AGENT_CHAT_OPTS
        )

        await ctx.world.write(
            "deployment_target",
            decision.deployment_target,
            confidence=decision.confidence,
        )
        await ctx.world.write(
            "observability", decision.observability, confidence=decision.confidence
        )
        await ctx.world.write(
            "error_handling",
            decision.error_handling,
            confidence=decision.confidence,
        )

    return agent
