"""The :class:`AgentRuntime` — shared world + dispatcher for a team of agents.

One runtime hosts:

- one :class:`~mwa.world.InMemoryWorldModel` (the shared source of truth)
- one :class:`~mwa.arbiter.ConflictDetector` + :class:`RuleBasedResolver`
  + :class:`SemanticArbiter`
- one :class:`~mwa.harness.HarnessMap`
- N :class:`~mwa.sdk.WorldAgent` instances
- one FIFO event queue that drives reactive dispatch

When a write lands (via :meth:`submit_write` or :meth:`seed_world`),
the runtime:

1. Runs the proposal through the conflict pipeline
   (``detect → rule_resolver → semantic_arbiter → apply/reject``).
2. If the write was applied, enqueues a reaction event on the internal
   queue.
3. :meth:`run_until_idle` then drains the queue by invoking every
   registered handler for each event's node, in registration order.

The dispatcher is deliberately **sequential** for M6.  One handler at
a time, one event at a time.  That's enough for the demo and small
production workloads, and keeps the ordering semantics dead simple.
Parallel dispatch is a later optimisation — the event queue shape
supports it, we just don't turn it on yet.

Events that don't apply (rejection / escalation) do NOT enqueue
reactions — there's nothing new for downstream handlers to react to.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mwa.arbiter import (
    ConflictDetector,
    Resolution,
    ResolutionDecision,
    RuleBasedResolver,
    SemanticArbiter,
)
from mwa.harness import HarnessMap
from mwa.sdk.context import AgentContext, AgentWorldView
from mwa.types import WriteProposal
from mwa.world import InMemoryWorldModel

if TYPE_CHECKING:
    from mwa.llm.base import LLMProvider
    from mwa.sdk.agent import WorldAgent


WriteStatus = Literal["applied", "rejected", "escalated_to_human"]
WriteOutcome = tuple[WriteStatus, Resolution | None]


@dataclass(frozen=True)
class _Event:
    """One entry on the runtime's reaction queue."""

    node: str
    value: Any
    version: int  # world version at the time of the write
    origin_agent_id: str


class AgentRuntime:
    """Shared world + dispatcher for a group of :class:`WorldAgent` s.

    Parameters
    ----------
    harness:
        The loaded :class:`HarnessMap`.
    world:
        Pre-built :class:`InMemoryWorldModel`.  Usually omitted and
        built internally from ``harness`` — inject a pre-built world
        only when a test needs to seed state before any agents run.
    arbiter_llm:
        :class:`LLMProvider` that backs the :class:`SemanticArbiter`.
        Typically the strongest available model, since arbitration is
        where reasoning quality matters most.
    auto_resolve_threshold:
        Passed through to :class:`SemanticArbiter`.  Default 0.85.
    """

    def __init__(
        self,
        *,
        harness: HarnessMap,
        arbiter_llm: LLMProvider,
        world: InMemoryWorldModel | None = None,
        auto_resolve_threshold: float = 0.85,
    ) -> None:
        self._harness = harness
        self._world = world if world is not None else InMemoryWorldModel(harness=harness)
        self._detector = ConflictDetector(self._world)
        self._rule_resolver = RuleBasedResolver()
        self._semantic_arbiter = SemanticArbiter(
            arbiter_llm,
            harness,
            auto_resolve_threshold=auto_resolve_threshold,
        )

        self._agents: dict[str, WorldAgent] = {}
        self._event_queue: deque[_Event] = deque()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_harness_file(
        cls,
        path: str | Path,
        *,
        arbiter_llm: LLMProvider,
        auto_resolve_threshold: float = 0.85,
    ) -> AgentRuntime:
        """Convenience: load a harness map from JSON and build a runtime."""
        harness = HarnessMap.load(path)
        return cls(
            harness=harness,
            arbiter_llm=arbiter_llm,
            auto_resolve_threshold=auto_resolve_threshold,
        )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def harness(self) -> HarnessMap:
        return self._harness

    @property
    def world(self) -> InMemoryWorldModel:
        """Direct handle on the shared world.

        Exposed mostly for test inspection and for the demo's
        pretty-printing step.  Production handlers should go through
        ``ctx.world`` instead so their writes are correctly stamped.
        """
        return self._world

    @property
    def agents(self) -> dict[str, WorldAgent]:
        """Read-only view of ``{name → WorldAgent}``."""
        return dict(self._agents)

    @property
    def pending_events(self) -> int:
        """Size of the reaction queue — useful for assertions in tests."""
        return len(self._event_queue)

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register(self, agent: WorldAgent) -> None:
        """Join an agent to the runtime.

        Called automatically by :class:`WorldAgent.__init__`.  Raises
        :class:`ValueError` on duplicate names because silently
        overwriting an agent is almost always a bug.
        """
        if agent.name in self._agents:
            raise ValueError(
                f"Duplicate agent name {agent.name!r}.  Agent names must be "
                f"unique within a runtime."
            )
        self._agents[agent.name] = agent

    # ------------------------------------------------------------------
    # Write pipeline
    # ------------------------------------------------------------------

    async def submit_write(
        self,
        *,
        agent_id: str,
        node: str,
        value: Any,
        confidence: float = 0.8,
        causal_parents: tuple[str, ...] = (),
    ) -> WriteOutcome:
        """Run a write through the full conflict-resolution pipeline.

        This is the *one* place in the SDK where world mutations flow
        through.  Both :meth:`seed_world` and
        :meth:`AgentWorldView.write` call this, so every write — from
        a user seed, from an agent's reaction, or from a test — takes
        the same path.
        """
        proposal = WriteProposal(
            agent_id=agent_id,
            node=node,
            value=value,
            confidence=confidence,
            causal_parents=causal_parents,
        )

        conflict = await self._detector.check(proposal)
        if conflict is None:
            await self._world.apply(proposal)
            self._enqueue_event(
                node=node,
                value=value,
                origin_agent_id=agent_id,
                version=self._world.version,
            )
            return ("applied", None)

        resolution = self._rule_resolver.resolve(conflict)
        if resolution.decision is ResolutionDecision.ESCALATE:
            resolution = await self._semantic_arbiter.resolve(conflict)

        if resolution.decision is ResolutionDecision.APPLY_PROPOSED:
            await self._world.apply(proposal)
            self._enqueue_event(
                node=node,
                value=value,
                origin_agent_id=agent_id,
                version=self._world.version,
            )
            return ("applied", resolution)

        if resolution.decision is ResolutionDecision.KEEP_EXISTING:
            await self._world.reject(proposal, reason=resolution.reason)
            return ("rejected", resolution)

        # Still ESCALATE after both layers — the caller queues for human.
        return ("escalated_to_human", resolution)

    async def seed_world(
        self,
        node: str,
        value: Any,
        *,
        agent_id: str = "user",
        confidence: float = 1.0,
    ) -> WriteOutcome:
        """Bootstrap a node from outside any agent.

        Equivalent to a reaction from a fictitious "user" agent.  The
        default confidence of 1.0 mirrors the fact that user inputs
        are the ground truth that everything else derives from.
        """
        return await self.submit_write(
            agent_id=agent_id,
            node=node,
            value=value,
            confidence=confidence,
        )

    def _enqueue_event(
        self,
        *,
        node: str,
        value: Any,
        origin_agent_id: str,
        version: int,
    ) -> None:
        self._event_queue.append(
            _Event(
                node=node,
                value=value,
                version=version,
                origin_agent_id=origin_agent_id,
            )
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def run_until_idle(self, *, max_iterations: int = 10_000) -> int:
        """Drain the event queue by invoking all registered handlers.

        Returns the total number of handler invocations that fired,
        so callers can assert on dispatcher behaviour in tests.

        Dispatch algorithm
        ------------------
        1. Pop the next event from the FIFO queue.
        2. For every agent, check if it has a handler for
           ``event.node``.  Iterate agents in registration order for
           deterministic output.
        3. Invoke the handler with a fresh :class:`AgentContext`.
           Handler writes call :meth:`submit_write` which may enqueue
           new events — those land at the BACK of the queue, so
           dispatch stays deterministic (FIFO, registration-order
           within an event).
        4. Repeat until the queue is empty.
        5. Guard against runaway loops via ``max_iterations`` — in
           practice with an acyclic harness map this should never fire.

        Skips handlers on the same agent that triggered the event
        unless explicitly configured otherwise — an agent reacting to
        its own write is almost always a bug (infinite loop).
        """
        invocations = 0
        iterations = 0
        while self._event_queue:
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError(
                    f"AgentRuntime dispatcher exceeded max_iterations "
                    f"({max_iterations}).  This usually means a handler "
                    f"cycle — agent A writes node X, handler on X writes "
                    f"node Y, handler on Y writes X again, …  Check your "
                    f"handler logic for runaway loops."
                )

            event = self._event_queue.popleft()
            for agent_name, agent in self._agents.items():
                if agent_name == event.origin_agent_id:
                    # Skip self-reactions by default; they're almost
                    # always a feedback-loop bug.
                    continue
                handler = agent.handlers.get(event.node)
                if handler is None:
                    continue
                ctx = AgentContext(
                    agent_id=agent_name,
                    llm=agent.llm,
                    world=AgentWorldView(self, agent_name),
                    trigger_node=event.node,
                    trigger_value=event.value,
                )
                await handler(ctx)
                invocations += 1

        return invocations
