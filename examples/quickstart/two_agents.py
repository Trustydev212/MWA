"""Quickstart: two agents writing into one shared world.

Run with:

    uv run python examples/quickstart/two_agents.py

This is the smallest non-trivial demonstration of SOMA's value
proposition.  Two "agents" (just async functions here — no LLM, no
network) write into a shared World Model.  Some writes go in cleanly,
some create conflicts that the rule-based resolver handles, and one is
deliberately ambiguous so it escalates to a human.

What you should see in the output
---------------------------------
1. Alice writes ``tone=playful`` (high confidence) — applied
2. Bob writes ``script_language=conversational`` — applied (different node)
3. Bob proposes ``tone=serious`` (low confidence) — REJECTED, Alice wins
4. Alice writes ``visual_style=cartoon`` — applied
5. Bob proposes ``visual_style=corporate`` (close confidence) — ESCALATED
6. Final world state is consistent with the Harness Map
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from soma.arbiter import ConflictDetector, ResolutionDecision, RuleBasedResolver
from soma.harness import HarnessMap
from soma.types import WriteProposal
from soma.world import InMemoryWorldModel

HARNESS_PATH = Path(__file__).resolve().parents[2] / "harness_maps" / "video_production.json"


async def propose(
    *,
    world: InMemoryWorldModel,
    detector: ConflictDetector,
    resolver: RuleBasedResolver,
    proposal: WriteProposal,
) -> None:
    """Run one write through the M3 protocol and pretty-print the outcome."""
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
        # M4 will plug the LLM Semantic Arbiter in here.
        print(f"  ? ESCALATED    {label}")
        print(f"      reason: {resolution.reason}")


async def main() -> None:
    print(f"Loading harness map from {HARNESS_PATH.name}")
    harness = HarnessMap.load(HARNESS_PATH)
    print(f"  domain: {harness.domain}")
    print(f"  nodes:  {len(harness)}")
    print(f"  hard constraints: {len(harness.hard_constraints)}")
    print()

    world = InMemoryWorldModel(harness=harness)
    detector = ConflictDetector(world)
    resolver = RuleBasedResolver()

    print("Step 1 — Alice sets the tone (clean write)")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="alice", node="tone", value="playful", confidence=0.9),
    )

    print("\nStep 2 — Bob picks a script language (different node, no conflict)")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="bob",
            node="script_language",
            value="conversational",
            confidence=0.85,
        ),
    )

    print("\nStep 3 — Bob tries to overwrite Alice's tone with low confidence")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(agent_id="bob", node="tone", value="serious", confidence=0.3),
    )

    print("\nStep 4 — Alice picks visual_style=cartoon")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="alice", node="visual_style", value="cartoon", confidence=0.7
        ),
    )

    print("\nStep 5 — Bob proposes visual_style=corporate with close confidence")
    print("        (the close gap should ESCALATE to a higher-tier resolver)")
    await propose(
        world=world,
        detector=detector,
        resolver=resolver,
        proposal=WriteProposal(
            agent_id="bob", node="visual_style", value="corporate", confidence=0.72
        ),
    )

    print("\nFinal world state")
    print("─" * 50)
    for node in ("tone", "script_language", "visual_style"):
        fact = await world.read(node)
        if fact is None:
            print(f"  {node:20s} (unset)")
        else:
            print(
                f"  {node:20s} {fact.episode.value!r:20s} "
                f"by {fact.episode.agent_id}  conf={fact.episode.confidence}"
            )

    print(f"\nTotal world version: {world.version}")
    print("Audit history for `tone`:")
    for fact in await world.history("tone", include_rejected=True):
        marker = "✓" if fact.is_current else ("✗" if fact.is_rejected else "·")
        print(
            f"  {marker} {fact.episode.value!r:12s} by {fact.episode.agent_id}  "
            f"conf={fact.episode.confidence}  rejected={fact.is_rejected}"
        )


if __name__ == "__main__":
    asyncio.run(main())
