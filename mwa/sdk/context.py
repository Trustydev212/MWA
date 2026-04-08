"""Per-invocation context passed into every :class:`WorldAgent` handler.

A fresh :class:`AgentContext` is built by the :class:`AgentRuntime`
dispatcher each time a handler fires.  It bundles everything the
handler could plausibly need into a single parameter, so handlers
stay tight::

    @agent.on("user_intent")
    async def design(ctx: AgentContext) -> None:
        intent = ctx.trigger_value            # what changed
        existing = await ctx.world.read("agent_architecture")
        decision = await ctx.llm.structured(messages, Schema)
        await ctx.world.write(
            "agent_architecture",
            decision.agent_architecture,
            confidence=decision.confidence,
        )

Design notes
------------
- We intentionally don't expose the raw ``InMemoryWorldModel`` on
  ``ctx``.  Handlers see only an :class:`AgentWorldView` wrapper that
  (a) stamps every ``write`` with the owning agent's id, and
  (b) routes the call through the full conflict-detection pipeline
  (``detect → rule_resolver → semantic_arbiter → apply/reject``).
  This is the exact flow the demo's ``write_to_world`` helper runs —
  now centralised under the SDK so individual agents never touch
  ``world.apply`` directly.
- :class:`AgentContext` is frozen (via ``slots + __init__``) so
  handlers can't sneakily mutate it.  Extra per-invocation state
  should live on the :class:`WorldAgent` instance, not on ``ctx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mwa.llm.base import LLMProvider

if TYPE_CHECKING:
    from mwa.harness import HarnessMap
    from mwa.sdk.runtime import AgentRuntime, WriteOutcome
    from mwa.types import Fact


class AgentWorldView:
    """Agent-scoped read/write facade over the shared world.

    Exposed on :attr:`AgentContext.world`.  Every ``write`` goes through
    the runtime's full conflict-resolution pipeline and is stamped with
    the owning agent's id — handlers never pass an ``agent_id``.
    """

    __slots__ = ("_agent_id", "_runtime")

    def __init__(self, runtime: AgentRuntime, agent_id: str) -> None:
        self._runtime = runtime
        self._agent_id = agent_id

    async def read(self, node: str) -> Fact | None:
        """Return the currently-valid fact for ``node``, or ``None``.

        Passthrough to the underlying ``InMemoryWorldModel.read``.
        Reads never trigger reactions — only writes do.
        """
        return await self._runtime.world.read(node)

    async def write(
        self,
        node: str,
        value: Any,
        *,
        confidence: float = 0.8,
        causal_parents: tuple[str, ...] = (),
    ) -> WriteOutcome:
        """Push a new value through the full write pipeline.

        Returns the :class:`WriteOutcome` tuple the runtime produces —
        ``(status, resolution_or_none)``.  ``status`` is one of
        ``"applied"``, ``"rejected"``, or ``"escalated_to_human"``.
        Applied writes automatically enqueue a reaction event in the
        runtime dispatcher so downstream handlers fire on the next
        ``run_until_idle`` iteration.
        """
        return await self._runtime.submit_write(
            agent_id=self._agent_id,
            node=node,
            value=value,
            confidence=confidence,
            causal_parents=causal_parents,
        )

    @property
    def harness(self) -> HarnessMap:
        """Read-only handle on the runtime's :class:`HarnessMap`.

        Handlers that need to look at the dependency graph (e.g. to
        decide what downstream nodes they'll need to update) use this
        without having to plumb harness through their own closure.
        """
        return self._runtime.harness


class AgentContext:
    """The per-handler-call context object.

    Every field is populated by :class:`AgentRuntime` when it dispatches
    an event to a handler.  Handlers receive it as their only argument.
    """

    __slots__ = (
        "agent_id",
        "llm",
        "trigger_node",
        "trigger_value",
        "world",
    )

    def __init__(
        self,
        *,
        agent_id: str,
        llm: LLMProvider,
        world: AgentWorldView,
        trigger_node: str,
        trigger_value: Any,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm
        self.world = world
        self.trigger_node = trigger_node
        self.trigger_value = trigger_value
