"""The four OpenClaw builder agents.

Each agent is an ``async def`` that:

1. Reads whatever it needs from the shared world.
2. Assembles a system + user prompt.
3. Calls ``provider.structured(<its decision schema>)``.
4. Writes every field of the returned decision to the world via
   :func:`~examples.openclaw_team.runtime.write_to_world`.

They don't inherit from a shared base because M6's ``WorldAgent`` will
eventually supply that abstraction.  For now, plain functions keep the
demo small and the pattern obvious.

The prompts are deliberately terse — production prompts would be
longer and include few-shot examples.  The goal here is to *show the
shape*, not maximise quality.
"""

from __future__ import annotations

from typing import Any

from examples.openclaw_team.runtime import Runtime, WriteOutcome, write_to_world
from examples.openclaw_team.schemas import (
    ArchitectDecision,
    DeployerDecision,
    ProviderSelectorDecision,
    SecurityDecision,
)
from mwa.llm.base import ChatOptions, LLMProvider, Message, MessageRole
from mwa.types import WriteProposal

AGENT_CHAT_OPTS = ChatOptions(temperature=0.0, max_tokens=1024)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


async def _read_required(runtime: Runtime, node: str) -> Any:
    """Read a node from the world, failing loudly if it's not set yet.

    Every downstream agent depends on some upstream facts existing.
    A missing fact is a wiring bug, not a "gracefully continue" case,
    so we raise rather than substitute a default.
    """
    fact = await runtime.world.read(node)
    if fact is None:
        raise RuntimeError(
            f"Agent dependency missing: node {node!r} has no current value. "
            f"Upstream agent probably failed or hasn't run yet."
        )
    return fact.episode.value


async def _write_field(
    runtime: Runtime,
    *,
    agent_id: str,
    node: str,
    value: Any,
    confidence: float,
) -> WriteOutcome:
    """Push one field of an agent decision through the full write pipeline."""
    outcome, _ = await write_to_world(
        runtime,
        WriteProposal(
            agent_id=agent_id,
            node=node,
            value=value,
            confidence=confidence,
        ),
    )
    return outcome


# ---------------------------------------------------------------------------
# Architect
# ---------------------------------------------------------------------------


async def run_architect(runtime: Runtime, provider: LLMProvider) -> ArchitectDecision:
    """Architect reads user_intent and designs the structural layout."""
    user_intent = await _read_required(runtime, "user_intent")

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
            content=f"user_intent: {user_intent!r}\n\nReturn your ArchitectDecision.",
        ),
    ]
    decision = await provider.structured(messages, ArchitectDecision, options=AGENT_CHAT_OPTS)

    await _write_field(
        runtime,
        agent_id="architect_agent",
        node="agent_architecture",
        value=decision.agent_architecture,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="architect_agent",
        node="sub_agent_count",
        value=decision.sub_agent_count,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="architect_agent",
        node="orchestration_pattern",
        value=decision.orchestration_pattern,
        confidence=decision.confidence,
    )
    return decision


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


async def run_security(runtime: Runtime, provider: LLMProvider) -> SecurityDecision:
    """Security reads user_intent + agent_architecture and picks guardrails."""
    user_intent = await _read_required(runtime, "user_intent")
    architecture = await _read_required(runtime, "agent_architecture")

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
                f"user_intent: {user_intent!r}\n"
                f"agent_architecture: {architecture!r}\n\n"
                "Return your SecurityDecision."
            ),
        ),
    ]
    decision = await provider.structured(messages, SecurityDecision, options=AGENT_CHAT_OPTS)

    await _write_field(
        runtime,
        agent_id="security_agent",
        node="tool_permissions",
        value=decision.tool_permissions,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="security_agent",
        node="memory_strategy",
        value=decision.memory_strategy,
        confidence=decision.confidence,
    )
    return decision


# ---------------------------------------------------------------------------
# Provider Selector
# ---------------------------------------------------------------------------


async def run_provider_selector(
    runtime: Runtime, provider: LLMProvider
) -> ProviderSelectorDecision:
    """Provider-Selector picks the LLM backend for the built agent."""
    user_intent = await _read_required(runtime, "user_intent")
    architecture = await _read_required(runtime, "agent_architecture")
    sub_count = await _read_required(runtime, "sub_agent_count")

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
                f"user_intent: {user_intent!r}\n"
                f"agent_architecture: {architecture!r}\n"
                f"sub_agent_count: {sub_count}\n\n"
                "Return your ProviderSelectorDecision."
            ),
        ),
    ]
    decision = await provider.structured(
        messages, ProviderSelectorDecision, options=AGENT_CHAT_OPTS
    )

    await _write_field(
        runtime,
        agent_id="provider_selector_agent",
        node="llm_provider",
        value=decision.llm_provider,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="provider_selector_agent",
        node="cost_budget",
        value=decision.cost_budget_usd,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="provider_selector_agent",
        node="latency_budget",
        value=decision.latency_budget_ms,
        confidence=decision.confidence,
    )
    return decision


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------


async def run_deployer(runtime: Runtime, provider: LLMProvider) -> DeployerDecision:
    """Deployer picks the runtime surface and operational behaviour."""
    user_intent = await _read_required(runtime, "user_intent")
    architecture = await _read_required(runtime, "agent_architecture")
    memory = await _read_required(runtime, "memory_strategy")
    permissions = await _read_required(runtime, "tool_permissions")

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
                f"user_intent: {user_intent!r}\n"
                f"agent_architecture: {architecture!r}\n"
                f"memory_strategy: {memory!r}\n"
                f"tool_permissions: {permissions!r}\n\n"
                "Return your DeployerDecision."
            ),
        ),
    ]
    decision = await provider.structured(messages, DeployerDecision, options=AGENT_CHAT_OPTS)

    await _write_field(
        runtime,
        agent_id="deployer_agent",
        node="deployment_target",
        value=decision.deployment_target,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="deployer_agent",
        node="observability",
        value=decision.observability,
        confidence=decision.confidence,
    )
    await _write_field(
        runtime,
        agent_id="deployer_agent",
        node="error_handling",
        value=decision.error_handling,
        confidence=decision.confidence,
    )
    return decision
