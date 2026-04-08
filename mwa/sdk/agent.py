"""The :class:`WorldAgent` — one named participant in an MWA runtime.

A :class:`WorldAgent` is intentionally minimal:

- It owns its **name** (used as ``agent_id`` on every write it makes).
- It owns its **LLM provider** (so different agents can use different
  models in the same runtime).
- It owns its **handler registry** (``{node_name → async callable}``),
  populated via the :meth:`on` decorator.
- It holds a back-reference to the :class:`~mwa.sdk.AgentRuntime` so
  handlers can access the shared world, and so the runtime can
  enumerate handlers during dispatch.

Everything else — conflict resolution, arbitration, event queue — lives
on the runtime.  That separation matters for the multi-agent case:
N agents all share ONE world, ONE arbiter, ONE event queue, but each
has its own LLM + handler set.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from mwa.llm.base import LLMProvider

if TYPE_CHECKING:
    from mwa.sdk.context import AgentContext
    from mwa.sdk.runtime import AgentRuntime


Handler = Callable[["AgentContext"], Awaitable[object]]
"""An agent handler — an async callable that takes a single
:class:`AgentContext` and returns anything (the return value is
discarded, handlers communicate back through ``ctx.world.write``)."""


class WorldAgent:
    """One named agent in an :class:`AgentRuntime`.

    Parameters
    ----------
    name:
        Stable identifier used as ``agent_id`` on every write this
        agent performs.  Must be unique within a runtime.
    llm:
        The :class:`~mwa.llm.base.LLMProvider` this agent uses for its
        structured calls.  Different agents can use different
        providers in the same runtime — that's the whole point of
        splitting the LLM per-agent.
    runtime:
        The :class:`AgentRuntime` the agent joins.  Registration is
        done automatically in ``__init__`` — callers don't need to
        call ``runtime.register(agent)`` separately.
    """

    def __init__(
        self,
        *,
        name: str,
        llm: LLMProvider,
        runtime: AgentRuntime,
    ) -> None:
        if not name:
            raise ValueError("WorldAgent name must be non-empty")

        self._name = name
        self._llm = llm
        self._runtime = runtime
        self._handlers: dict[str, Handler] = {}

        # Register ourselves with the runtime so the dispatcher can
        # find us.  Runtime.register is idempotent per-name and raises
        # on duplicate names.
        runtime.register(self)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    @property
    def handlers(self) -> dict[str, Handler]:
        """Read-only view of ``{node_name → handler}``.

        A copy rather than the live dict so callers can't mutate the
        registry accidentally.  The runtime dispatcher uses this to
        find which handler to fire on a given event.
        """
        return dict(self._handlers)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def on(self, node: str) -> Callable[[Handler], Handler]:
        """Decorator registering ``handler`` as the reaction for ``node``.

        Usage::

            @agent.on("user_intent")
            async def design(ctx: AgentContext) -> None:
                decision = await ctx.llm.structured(...)
                await ctx.world.write("agent_architecture", ...)

        Rules:

        - ``node`` must be a real node in the runtime's harness map.
          We validate at registration time, not at dispatch time, so
          typos fail loudly at module-load.
        - One handler per ``(agent, node)`` pair.  Re-registering the
          same pair raises :class:`ValueError` — it's almost always a
          bug, and silently overriding makes debugging nightmarish.
        - Multiple agents CAN react to the same node — that's the
          multi-agent pattern in action.  The runtime fires every
          registered handler when the node changes.
        """
        if node not in self._runtime.harness:
            raise ValueError(
                f"Cannot register handler on unknown node {node!r}. "
                f"Harness map has: {sorted(self._runtime.harness.nodes)}"
            )

        def decorator(fn: Handler) -> Handler:
            if node in self._handlers:
                raise ValueError(
                    f"Agent {self._name!r} already has a handler for node "
                    f"{node!r}.  Re-registration is almost certainly a bug — "
                    f"if you really want to replace it, call "
                    f"agent._handlers.pop({node!r}) first (private API, use "
                    f"sparingly)."
                )
            self._handlers[node] = fn
            return fn

        return decorator
