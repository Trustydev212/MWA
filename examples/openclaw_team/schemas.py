"""Per-agent structured-output schemas.

Each OpenClaw builder agent returns a small Pydantic model that maps
1-to-1 onto the nodes it owns in the harness map.  Keeping the schemas
tight means:

1. The LLM gets a small, explicit contract — much easier for weak
   routed models to follow than a single fat "give me everything" schema.
2. The demo can show per-agent token cost cleanly.
3. Future M6 ``WorldAgent`` SDK can auto-extract the "write these nodes"
   mapping from the schema field names.

Confidence fields are required on every decision — they feed directly
into the Conflict Detector's dominance rules.  A weak decision (low
confidence) gets beaten by a strong one without ever touching the LLM
arbiter.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArchitectDecision(BaseModel):
    """What the Architect-Agent decides about structural layout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_architecture: Literal["single_agent", "multi_agent", "hierarchical"] = Field(
        description=(
            "Overall shape of the agent being built. 'single_agent' is one LLM "
            "call per request; 'multi_agent' is N peers collaborating via shared "
            "state; 'hierarchical' is a supervisor dispatching to workers."
        ),
    )
    sub_agent_count: int = Field(
        ge=1,
        le=20,
        description="How many sub-agents the built system will spawn.",
    )
    orchestration_pattern: Literal["sequential", "parallel", "event_driven"] = Field(
        description=(
            "How sub-agents are scheduled. 'sequential' is a fixed pipeline; "
            "'parallel' is concurrent via asyncio.gather; 'event_driven' is "
            "pub/sub on the world model."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your overall certainty in this architecture plan.",
    )
    reason: str = Field(
        description="Short explanation — what in the user intent drove this choice."
    )


class SecurityDecision(BaseModel):
    """What the Security-Agent decides about permissions and memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_permissions: Literal["read_only", "read_write", "admin"] = Field(
        description=(
            "Scope of side effects the built agent can cause. Pick the least "
            "powerful tier that still satisfies user_intent — defense in depth."
        ),
    )
    memory_strategy: Literal[
        "stateless", "short_term_conversation", "long_term_persistent"
    ] = Field(
        description=(
            "How the agent retains state across turns. 'stateless' forgets "
            "after every call; 'short_term_conversation' keeps the current "
            "session; 'long_term_persistent' writes to a durable store."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ProviderSelectorDecision(BaseModel):
    """What the Provider-Selector-Agent decides about the LLM backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_provider: Literal[
        "anthropic_claude",
        "openai_gpt4o",
        "google_gemini",
        "local_llama",
    ] = Field(
        description=(
            "Which LLM backs every inference call in the built agent. "
            "Balance capability, cost, latency, and privacy."
        ),
    )
    cost_budget_usd: float = Field(
        ge=0.0,
        le=100.0,
        description="Maximum USD per single agent run.",
    )
    latency_budget_ms: int = Field(
        ge=100,
        le=60_000,
        description="Maximum wall-clock ms per single agent run.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class DeployerDecision(BaseModel):
    """What the Deployment-Agent decides about runtime surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deployment_target: Literal["local", "cloud", "edge"] = Field(
        description=(
            "Where the built agent runs. 'local' is the user's machine; "
            "'cloud' is a managed runtime; 'edge' is a constrained device "
            "(phone, IoT, embedded)."
        ),
    )
    observability: Literal["logs", "metrics", "traces", "none"] = Field(
        description=(
            "Which signals the runtime emits. 'traces' is the richest, "
            "'none' is lightest."
        ),
    )
    error_handling: Literal["retry", "fallback", "fail_fast", "human_escalation"] = Field(
        description=(
            "What happens on internal failure. 'retry' re-runs with backoff; "
            "'fallback' tries a secondary path; 'fail_fast' surfaces the "
            "error immediately; 'human_escalation' queues for review."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
