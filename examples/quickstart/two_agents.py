"""Quickstart: building an AI agent with OpenClaw, the MWA way.

Scenario
--------
**OpenClaw** is a meta-agent builder — a tool that takes a human user's
intent (*"build me a research agent that reads arxiv papers and extracts
claims"*) and spawns a coordinated team of sub-agents that design, code,
test, and deploy the requested agent.

Today's quickstart simulates the earliest phase of that workflow: two
OpenClaw sub-agents arguing over fundamental decisions about the agent
being built.

- **Architect-Agent** (``architect_agent``) owns the *structural* decisions:
  which architecture, how many sub-agents, what orchestration pattern.
- **Security-Agent** (``security_agent``) owns the *guardrail* decisions:
  tool permissions, memory strategy, deployment surface.

Both agents share one World Model — the architect writes
``agent_architecture``, the security agent writes ``tool_permissions``
and ``memory_strategy``.  They don't talk to each other directly; they
*observe the same world* via the MWA runtime.

Five steps below walk through the whole M3 protocol:

1. Architect picks ``multi_agent`` architecture — clean write, applied.
2. Security picks ``read_only`` permissions — clean, different node.
3. Architect tries to downgrade to ``single_agent`` with low confidence
   → rejected by ``dominant_confidence`` rule.
4. Security picks ``short_term_conversation`` memory — clean write.
5. Architect vs. Security disagree on ``orchestration_pattern`` with
   very close confidence → escalated (no LLM wired up in this demo —
   the next milestone plugs :class:`~mwa.arbiter.SemanticArbiter` in here).

Run with::

    uv run python examples/quickstart/two_agents.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mwa.arbiter import ConflictDetector, ResolutionDecision, RuleBasedResolver
from mwa.harness import HarnessMap
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel

HARNESS_PATH = (
    Path(__file__).resolve().parents[2] / "harness_maps" / "openclaw_agent_builder.json"
)


async def propose(
    *,
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
    proposal: WriteProposal,
) -> None:
    """Run one write through the detect → resolve → apply loop and
    pretty-print the outcome."""
    label = f"{proposal.agent_id} → {proposal.node}={proposal.value!r} (conf={proposal.confidence})"

    conflict = await detector.check(proposal)
    if conflict is None:
        await world.apply(proposal)
        print(f"  ✓ APPLIED      {label}")
        return

    resolution = resolver.resolve(conflict)
    if resolution.decision == ResolutionDecision.APPLY_PROPOSED:
        await world.apply(proposal)
        print(f"  ✓ RESOLVED→NEW {label}")
        print(f"      reason: {resolution.reason}")
    elif resolution.decision == ResolutionDecision.KEEP_EXISTING:
        await world.reject(proposal, reason=resolution.reason)
        print(f"  ✗ REJECTED     {label}")
        print(f"      reason: {resolution.reason}")
    else:
        # ESCALATE — M4 plugs SemanticArbiter in here.  For this demo
        # we just surface the escalation so the output is honest.
        print(f"  ? ESCALATED    {label}")
        print(f"      reason: {resolution.reason}")


async def main() -> None:
    print(f"Loading OpenClaw harness map from {HARNESS_PATH.name}")
    harness = HarnessMap.load(HARNESS_PATH)
    print(f"  domain:           {harness.domain}")
    print(f"  nodes:            {len(harness)}")
    print(f"  hard constraints: {len(harness.hard_constraints)}")
    print()

    world = InMemoryWorldModel(harness=harness)
    detector = ConflictDetector(world)
    resolver = RuleBasedResolver()

    print("Step 1 — Architect-Agent picks a multi-agent architecture")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="architect_agent",
            node="agent_architecture",
            value="multi_agent",
            confidence=0.9,
        ),
    )

    print("\nStep 2 — Security-Agent picks read_only tool permissions (different node)")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="security_agent",
            node="tool_permissions",
            value="read_only",
            confidence=0.85,
        ),
    )

    print("\nStep 3 — Architect-Agent tries to downgrade to single_agent with low confidence")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="architect_agent",
            node="agent_architecture",
            value="single_agent",
            confidence=0.3,
        ),
    )

    print("\nStep 4 — Security-Agent picks a short_term_conversation memory strategy")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="security_agent",
            node="memory_strategy",
            value="short_term_conversation",
            confidence=0.7,
        ),
    )

    print("\nStep 5 — Architect vs. Security disagree on orchestration_pattern (close confidence)")
    print("         the gap is too tight for the rule-based resolver → ESCALATED")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="security_agent",
            node="orchestration_pattern",
            value="sequential",
            confidence=0.7,
        ),
    )
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="architect_agent",
            node="orchestration_pattern",
            value="parallel",
            confidence=0.72,
        ),
    )

    print("\nFinal world state")
    print("─" * 60)
    for node in (
        "agent_architecture",
        "tool_permissions",
        "memory_strategy",
        "orchestration_pattern",
    ):
        fact = await world.read(node)
        if fact is None:
            print(f"  {node:24s} (unset)")
        else:
            print(
                f"  {node:24s} {fact.episode.value!r:26s} "
                f"by {fact.episode.agent_id}  conf={fact.episode.confidence}"
            )

    print(f"\nTotal world version: {world.version}")
    print("Audit history for `agent_architecture`:")
    for fact in await world.history("agent_architecture", include_rejected=True):
        marker = "✓" if fact.is_current else ("✗" if fact.is_rejected else "·")
        print(
            f"  {marker} {fact.episode.value!r:16s} by {fact.episode.agent_id}  "
            f"conf={fact.episode.confidence}  rejected={fact.is_rejected}"
        )


if __name__ == "__main__":
    asyncio.run(main())
