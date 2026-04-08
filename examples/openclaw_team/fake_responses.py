"""Canned JSON responses for the offline demo path.

Every entry here is what an LLM would plausibly return for the
scenario ``user_intent = "Build a research agent that reads arXiv
papers and extracts claims with confidence scores"``.

Two groups:

- **Happy path** (``*_HAPPY``) — the decisions the four builder agents
  produce first.  These align on the memory_strategy node (security
  wants ``short_term_conversation``), which means there's no conflict.

- **Conflict path** (``*_CONFLICT``) — a variant set where Architect
  is explicitly told to write a *long_term_persistent* memory strategy
  **directly** (pre-seeded before Security runs), creating a genuine
  conflict with Security's ``short_term_conversation``.  This is the
  scenario that exercises the Semantic Arbiter end-to-end.

The demo runs the happy path first, then seeds the conflict and re-runs
just the Security agent so the Semantic Arbiter fires.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Happy path — first 4 agent calls
# ---------------------------------------------------------------------------

ARCHITECT_HAPPY = {
    "agent_architecture": "multi_agent",
    "sub_agent_count": 3,
    "orchestration_pattern": "sequential",
    "confidence": 0.88,
    "reason": (
        "Research + claim-extraction has three natural stages (fetch, read, "
        "extract) that benefit from specialised sub-agents, and the output "
        "of each feeds the next so sequential orchestration is enough."
    ),
}

SECURITY_HAPPY = {
    "tool_permissions": "read_only",
    "memory_strategy": "short_term_conversation",
    "confidence": 0.82,
    "reason": (
        "Reading arXiv papers only needs HTTP GET; no write tools required. "
        "Short-term conversation memory is enough to carry paper context "
        "across the fetch → read → extract pipeline without a persistent store."
    ),
}

PROVIDER_SELECTOR_HAPPY = {
    "llm_provider": "anthropic_claude",
    "cost_budget_usd": 0.75,
    "latency_budget_ms": 30_000,
    "confidence": 0.8,
    "reason": (
        "Claim extraction is a reasoning-heavy task where Claude's strength "
        "on long-context summarisation and structured output justifies the "
        "cost.  0.75 USD per run leaves headroom for 3 sub-agent calls; 30s "
        "latency budget matches a sequential pipeline."
    ),
}

DEPLOYER_HAPPY = {
    "deployment_target": "cloud",
    "observability": "traces",
    "error_handling": "fallback",
    "confidence": 0.78,
    "reason": (
        "Cloud deployment since research papers live behind rate-limited "
        "APIs best accessed from a stable egress.  Traces are essential to "
        "debug the multi-agent pipeline.  Fallback on a smaller model keeps "
        "the agent usable when the primary hits rate limits."
    ),
}


# ---------------------------------------------------------------------------
# Conflict path — Security is asked to re-decide after Architect has
# already written a stronger opinion on memory_strategy.
# ---------------------------------------------------------------------------

SECURITY_CONFLICT = {
    "tool_permissions": "read_only",
    "memory_strategy": "short_term_conversation",
    # Similar confidence to what Architect wrote — forces rule_based
    # to escalate to SemanticArbiter.
    "confidence": 0.83,
    "reason": (
        "Short-term conversation memory is enough for a pipeline that "
        "processes one paper at a time; a persistent store would be over-"
        "engineered for the current user intent."
    ),
}


# The Semantic Arbiter's own structured output.  In Stage B the
# existing fact is Security's ``short_term_conversation`` and the
# proposal is Architect's contradicting ``long_term_persistent``.
#
# We pick ``winner="existing"`` here — i.e. Security's value stays
# put and Architect's new write is rejected.  This is the more
# interesting demo outcome: the guardrail-focused agent (Security)
# overrules the structural-focused agent (Architect) because the
# extra persistent-store complexity isn't justified by the user's
# intent.  Overall confidence is well above the 0.85 auto-resolve
# threshold so the runtime applies the decision immediately.
ARBITER_DECIDES_SECURITY_WINS = {
    "winner": "existing",
    "reason": (
        "Security-Agent's 'short_term_conversation' better matches the "
        "architecture (sequential multi-agent) and avoids committing to a "
        "persistent store the user intent doesn't justify.  Architect's "
        "long-term choice would add deployment complexity (need to pick "
        "a cloud target, lose the 'edge' option) without measurable "
        "benefit for a per-paper extraction pipeline."
    ),
    "proposed_scores": {
        "structural_importance": 0.65,
        "causal_depth": 0.45,
        "downstream_impact": 0.7,
        "confidence": 0.84,
    },
    "existing_scores": {
        "structural_importance": 0.7,
        "causal_depth": 0.55,
        "downstream_impact": 0.6,
        "confidence": 0.82,
    },
    "overall_confidence": 0.9,
    "update_plan": [],
}
