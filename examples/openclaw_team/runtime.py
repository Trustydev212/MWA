"""Shared runtime helpers for the OpenClaw team demo.

Everything in this module is **demo glue** — the kind of boilerplate
that lives inside :class:`~mwa.sdk.WorldAgent` once M6 lands.  For now
we wire it by hand so the demo is self-contained and the pattern is
visible.

Three concerns live here:

1. **Runtime assembly** — one ``World + HarnessMap + Detector +
   RuleBasedResolver + SemanticArbiter`` bundle used by every agent.
   All four agents read from and write to the same instance; that is
   the whole point of MWA.

2. **Provider factory** — either returns a retry-wrapped real
   ``OpenAIProvider`` (when ``MWA_LIVE_*`` env vars are set) or a
   pre-queued :class:`FakeProvider` (for the zero-network default
   path).  A separate provider per agent, so per-agent model choice
   remains possible.

3. **``write_to_world``** — the full "detect → rule → semantic →
   apply/reject" flow for a single :class:`WriteProposal`.  Every
   agent calls this instead of ``world.apply`` directly so conflicts
   get routed through the whole stack the same way production code
   will.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mwa.arbiter import (
    ConflictDetector,
    Resolution,
    ResolutionDecision,
    RuleBasedResolver,
    SemanticArbiter,
)
from mwa.harness import HarnessMap
from mwa.llm import LLMRouter, RetryPolicy
from mwa.llm.base import LLMProvider
from mwa.llm.providers import FakeProvider, OpenAIProvider
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel

HARNESS_PATH = (
    Path(__file__).resolve().parents[2]
    / "harness_maps"
    / "openclaw_agent_builder.json"
)


WriteOutcome = Literal["applied", "rejected", "escalated_to_human"]


@dataclass
class Runtime:
    """Single shared MWA runtime used by every agent in the demo.

    We keep it as a tiny dataclass rather than a full class because it
    really is just a bag of already-wired collaborators — the value is
    in the wiring, not in any methods.
    """

    harness: HarnessMap
    world: InMemoryWorldModel
    detector: ConflictDetector
    rule_resolver: RuleBasedResolver
    semantic_arbiter: SemanticArbiter


def build_runtime(arbiter_provider: LLMProvider) -> Runtime:
    """Assemble the shared runtime.

    ``arbiter_provider`` is the LLM that backs the Semantic Arbiter —
    typically the strongest model available, because arbitration is
    where reasoning quality matters most.  Individual agents may use
    cheaper providers.
    """
    harness = HarnessMap.load(HARNESS_PATH)
    world = InMemoryWorldModel(harness=harness)
    detector = ConflictDetector(world)
    rule_resolver = RuleBasedResolver()
    semantic_arbiter = SemanticArbiter(
        arbiter_provider,
        harness,
        auto_resolve_threshold=0.85,
    )
    return Runtime(
        harness=harness,
        world=world,
        detector=detector,
        rule_resolver=rule_resolver,
        semantic_arbiter=semantic_arbiter,
    )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _live_config() -> dict[str, str] | None:
    """Return live env config if every MWA_LIVE_* var is set, else None."""
    keys = {
        "api_key": "MWA_LIVE_API_KEY",
        "base_url": "MWA_LIVE_BASE_URL",
        "model": "MWA_LIVE_MODEL",
    }
    values: dict[str, str] = {}
    for k, env_name in keys.items():
        val = os.environ.get(env_name, "").strip()
        if not val:
            return None
        values[k] = val
    return values


def make_live_provider() -> LLMProvider:
    """Real OpenAI-compatible provider wrapped in RetryPolicy.

    Mirrors the production pattern from the live test suite — we learnt
    from the futrixapi 504 session that Cloudflare flakes are routine
    on gateway routes, so every real provider gets router-wrapped.
    """
    cfg = _live_config()
    if cfg is None:
        raise RuntimeError(
            "make_live_provider() called but MWA_LIVE_API_KEY / _BASE_URL / "
            "_MODEL are not set.  Use make_fake_provider(...) for offline mode."
        )
    raw = OpenAIProvider(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        structured_strategy="auto",
    )
    return LLMRouter(
        primary=raw,
        retry=RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
        ),
    )


def make_fake_provider(name: str, queued_json: list[dict[str, object]]) -> FakeProvider:
    """Offline provider pre-loaded with deterministic decisions.

    Each queued dict is serialised and returned on the next
    ``.structured()`` call.  Exposed per-agent so every builder has its
    own queue and concurrent execution can't shuffle responses.
    """
    provider = FakeProvider(name=name, model=f"{name}-fake")
    for payload in queued_json:
        provider.enqueue_json(payload)
    return provider


def is_live_mode() -> bool:
    """True iff every MWA_LIVE_* env var is set.

    Callers use this to pick between ``make_live_provider()`` and
    ``make_fake_provider(...)``.
    """
    return _live_config() is not None


# ---------------------------------------------------------------------------
# Write pipeline
# ---------------------------------------------------------------------------


async def write_to_world(
    runtime: Runtime,
    proposal: WriteProposal,
) -> tuple[WriteOutcome, Resolution | None]:
    """Run the full M3+M4 write pipeline for a single proposal.

    Returns ``(outcome, resolution)`` where:

    - ``outcome == "applied"`` — the value is now the current fact
    - ``outcome == "rejected"`` — the loser is in the audit log
    - ``outcome == "escalated_to_human"`` — neither layer was confident;
      the caller is expected to queue this for review

    ``resolution`` is ``None`` for clean writes (no conflict), otherwise
    the :class:`Resolution` produced by whichever resolver handled it.

    This is the same shape the future M6 ``WorldAgent.world.write()``
    will have — centralising it here means every agent in the demo
    follows the same flow.
    """
    conflict = await runtime.detector.check(proposal)
    if conflict is None:
        await runtime.world.apply(proposal)
        return ("applied", None)

    resolution = runtime.rule_resolver.resolve(conflict)
    if resolution.decision is ResolutionDecision.ESCALATE:
        resolution = await runtime.semantic_arbiter.resolve(conflict)

    if resolution.decision is ResolutionDecision.APPLY_PROPOSED:
        await runtime.world.apply(proposal)
        return ("applied", resolution)
    if resolution.decision is ResolutionDecision.KEEP_EXISTING:
        await runtime.world.reject(proposal, reason=resolution.reason)
        return ("rejected", resolution)
    # Both layers gave up — runtime should queue for human review.
    return ("escalated_to_human", resolution)
