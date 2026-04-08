"""WorldAgent SDK — ergonomic layer on top of the MWA runtime.

Everything under this package is **convenience wrapping** around the
lower layers (``mwa.world``, ``mwa.arbiter``, ``mwa.llm``).  No new
semantics — the SDK is there to cut boilerplate from ~100 lines per
agent to ~20 and to make the "reactive agent" pattern look like what
README promises:

.. code-block:: python

    from mwa.sdk import AgentRuntime, WorldAgent

    runtime = AgentRuntime.from_harness_file(
        "./openclaw.json",
        arbiter_llm=claude_provider,
    )

    architect = WorldAgent(
        name="architect_agent",
        llm=gpt4o_provider,
        runtime=runtime,
    )

    @architect.on("user_intent")
    async def design(ctx):
        decision = await ctx.llm.structured(messages, ArchitectDecision)
        await ctx.world.write("agent_architecture", decision.agent_architecture,
                              confidence=decision.confidence)
        return decision

    await runtime.seed_world("user_intent", "Build a research agent...")
    await runtime.run_until_idle()

Three moving parts:

- :class:`AgentRuntime` — owns the shared
  :class:`~mwa.world.InMemoryWorldModel`, the
  :class:`~mwa.arbiter.ConflictDetector`, the
  :class:`~mwa.arbiter.RuleBasedResolver`, and the
  :class:`~mwa.arbiter.SemanticArbiter`.  Also holds the event queue
  that drives reactive dispatch.
- :class:`WorldAgent` — one named agent with its own LLM provider.
  Handlers are registered via the :meth:`~WorldAgent.on` decorator.
- :class:`AgentContext` — the per-invocation handle passed into every
  handler; exposes ``ctx.world`` (agent-scoped read/write), ``ctx.llm``,
  ``ctx.trigger_node``, ``ctx.trigger_value``, ``ctx.agent_id``.
"""

from mwa.sdk.agent import WorldAgent
from mwa.sdk.context import AgentContext, AgentWorldView
from mwa.sdk.runtime import AgentRuntime

__all__ = [
    "AgentContext",
    "AgentRuntime",
    "AgentWorldView",
    "WorldAgent",
]
