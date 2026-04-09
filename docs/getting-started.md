# Getting Started with MWA

> **MWA** — Multi World Agent. Agents don't talk to each other. They observe the same world.

This doc walks you from `pip install` to a running multi-agent system in ~10 minutes.

## 1. Install

```bash
# Minimum install — just the library
pip install multi-world-agent

# With your LLM provider of choice
pip install "multi-world-agent[anthropic]"   # Claude
pip install "multi-world-agent[openai]"      # GPT-4/4o, o-series, or OpenAI-compatible gateways
pip install "multi-world-agent[ollama]"      # local Llama via Ollama

# With the MCP server (plug into Claude Code, Cursor, Codex)
pip install "multi-world-agent[mcp]"

# Everything
pip install "multi-world-agent[all]"

# Dev dependencies
pip install "multi-world-agent[dev]"
```

> **Distribution name vs module name:** the PyPI package is
> `multi-world-agent` (the short `mwa` was already reserved on PyPI)
> but the Python module stays short: `import mwa`.  Same pattern as
> `pip install scikit-learn` / `import sklearn`.  The extras shell-
> quote because some shells treat `[...]` as a glob pattern.

Python 3.11+ required.

## 2. The mental model

MWA is structured around 4 layers, bottom-up:

```
┌──────────────────────────────────────────────────┐
│  Your agents  (WorldAgent instances)             │
└──────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  mwa.sdk — AgentRuntime + dispatcher             │
│    • shared world, shared arbiter                │
│    • FIFO event queue + @agent.on() decorator    │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  mwa.arbiter — ConflictDetector,                 │
│                RuleBasedResolver,                │
│                SemanticArbiter (LLM)             │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  mwa.world — InMemoryWorldModel                  │
│    • append-only, temporal facts                 │
│    • hard-constraint enforcement                 │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  mwa.harness — HarnessMap (the dependency graph) │
└──────────────────────────────────────────────────┘
```

Typical user only touches `mwa.sdk`. Lower layers exist for advanced users who want to override individual components.

## 3. Your first agent

Create a harness map — this is the "constitution" of your world, declaring what nodes exist and how they depend on each other:

```json
// my_harness.json
{
  "version": "1.0",
  "domain": "sentiment_pipeline",
  "nodes": {
    "raw_text":     {"impact": "critical", "affects": ["sentiment", "topics"], "order": 1},
    "sentiment":    {"impact": "high",     "affects": ["summary"],             "order": 2},
    "topics":       {"impact": "high",     "affects": ["summary"],             "order": 2},
    "summary":      {"impact": "medium",   "affects": [],                      "order": 3}
  },
  "hard_constraints": []
}
```

Write a minimal agent:

```python
# sentiment_agent.py
import asyncio
from pydantic import BaseModel

from mwa.sdk import AgentRuntime, WorldAgent, AgentContext
from mwa.llm.providers import AnthropicProvider
from mwa.llm.base import Message, MessageRole


class SentimentDecision(BaseModel):
    sentiment: str            # "positive" | "neutral" | "negative"
    confidence: float         # 0..1


async def main():
    # 1. Build the runtime (shared world + arbiter).  The arbiter LLM
    #    only fires on conflicts — a fake stub is fine for a pipeline
    #    where conflicts are unlikely.
    runtime = AgentRuntime.from_harness_file(
        "./my_harness.json",
        arbiter_llm=AnthropicProvider(model="claude-opus-4-6"),
    )

    # 2. Create an agent with its own LLM.
    sentiment_agent = WorldAgent(
        name="sentiment_agent",
        llm=AnthropicProvider(model="claude-haiku-4-5"),  # cheaper for routine work
        runtime=runtime,
    )

    # 3. Register a handler that fires when raw_text changes.
    @sentiment_agent.on("raw_text")
    async def classify(ctx: AgentContext) -> None:
        text = ctx.trigger_value
        decision = await ctx.llm.structured(
            [
                Message(
                    role=MessageRole.SYSTEM,
                    content="Classify the sentiment of the user's text.",
                ),
                Message(role=MessageRole.USER, content=text),
            ],
            SentimentDecision,
        )
        await ctx.world.write(
            "sentiment",
            decision.sentiment,
            confidence=decision.confidence,
        )

    # 4. Seed the world and drain the dispatcher.
    await runtime.seed_world(
        "raw_text",
        "I absolutely loved this movie, it made me cry.",
    )
    invocations = await runtime.run_until_idle()
    print(f"Dispatcher fired {invocations} handler invocations")

    sentiment = await runtime.world.read("sentiment")
    print(f"Sentiment: {sentiment.episode.value} "
          f"(confidence {sentiment.episode.confidence})")


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python sentiment_agent.py
```

Output:

```
Dispatcher fired 1 handler invocations
Sentiment: positive (confidence 0.94)
```

## 4. Multiple agents sharing a world

The whole point of MWA is **many agents, one world**. Add a topics agent that reacts to the same `raw_text`:

```python
class TopicsDecision(BaseModel):
    topics: list[str]
    confidence: float


topics_agent = WorldAgent(
    name="topics_agent",
    llm=AnthropicProvider(model="claude-haiku-4-5"),
    runtime=runtime,   # same runtime → same world
)


@topics_agent.on("raw_text")
async def extract(ctx: AgentContext) -> None:
    decision = await ctx.llm.structured(
        [
            Message(
                role=MessageRole.SYSTEM,
                content="Extract 1-3 main topics from the user's text.",
            ),
            Message(role=MessageRole.USER, content=ctx.trigger_value),
        ],
        TopicsDecision,
    )
    await ctx.world.write("topics", decision.topics, confidence=decision.confidence)
```

When you seed `raw_text`, **both** agents fire concurrently at the SDK level (sequentially in the current dispatcher — parallel dispatch lands in a later milestone).

## 5. Conflict resolution

Two agents can disagree on the same node — MWA automatically routes the disagreement through:

1. **`ConflictDetector`** — `O(1)` value comparison against the current fact.
2. **`RuleBasedResolver`** — cheap deterministic rules (idempotent write, causal acknowledgement, dominant confidence). Handles ~80% of real conflicts without an LLM call.
3. **`SemanticArbiter`** — LLM-backed scoring on structural importance, causal depth, downstream impact, and confidence. Fires only when rules escalate.
4. **Auto-resolve threshold** — if the arbiter's `overall_confidence` falls below a threshold (default 0.85), the runtime marks the write `escalated_to_human` and does NOT apply — a reviewer makes the final call.

Nothing silently falls back to last-write-wins. That's the whole reason MWA exists.

## 6. Running the MCP server

Once you've got a harness map, expose the runtime to any MCP host (Claude Code, Cursor, Codex, …):

```bash
pip install "multi-world-agent[mcp]"
MWA_HARNESS_PATH=./my_harness.json mwa-mcp
```

Or register it with Claude Code by adding to `~/.config/claude/mcp_servers.json`:

```json
{
  "mcp_servers": {
    "mwa": {
      "command": "mwa-mcp",
      "env": {
        "MWA_HARNESS_PATH": "/absolute/path/to/my_harness.json"
      }
    }
  }
}
```

The server exposes five tools to the LLM:

| Tool | What it does |
|---|---|
| `mwa_list_nodes` | Discover what nodes exist in the harness map |
| `mwa_read_world` | Read the current fact for a node |
| `mwa_read_history` | Read the full episode history (with optional rejected) |
| `mwa_seed_world` | Seed a node from "outside" any agent |
| `mwa_submit_write` | Write as a named agent, running the full conflict pipeline |

Claude Code can now treat your MWA runtime as a first-class tool provider.

## 7. The OpenClaw example

A full 4-agent demo lives in `examples/openclaw_team/`. Four agents (`architect`, `security`, `provider_selector`, `deployer`) cooperatively design a new AI agent from a user intent. Run it:

```bash
# Offline — deterministic, uses FakeProvider
python -m examples.openclaw_team.run_demo

# Live — plugs into a real OpenAI-compatible endpoint
export MWA_LIVE_API_KEY=sk-...
export MWA_LIVE_BASE_URL=https://futrixapi.com/v1
export MWA_LIVE_MODEL=auto
python -m examples.openclaw_team.run_demo
```

The demo runs two stages:

1. **Happy path** — seed `user_intent`, dispatcher fires all 4 agents in topology order, 12 nodes populate cleanly.
2. **Conflict** — Architect changes mind on `memory_strategy`, rule-based resolver escalates (tight confidence gap), `SemanticArbiter` picks a winner, losing write lands in the audit log.

Read `examples/openclaw_team/README.md` for the full walkthrough.

## 8. Next steps

- **`ARCHITECTURE.md`** — deep dive into every layer, including the production patterns (retry chains, multi-provider routing, targeted push).
- **`RESEARCH.md`** — build journal with design decisions, trade-offs, and lessons learned from live interop debugging (futrixapi, etc).
- **`examples/`** — runnable examples.
- **`mwa.sdk` source** — the SDK is ~700 lines total and heavily commented. Read it.

## 9. FAQ

### Why is the default arbiter LLM a stub?

Because running the arbiter requires a real provider, and forcing every user to wire one up before their first `seed_world` call is a bad onboarding experience. The default stub fails cleanly with a clear error if a conflict actually escalates — at that point you plug in the provider you want.

### Do I need Neo4j / Graphiti?

Not for the alpha. `InMemoryWorldModel` is all you get today and it's enough for prototypes, demos, and in-process production use. Graph DB backends are a later milestone.

### Can agents running in different processes share one world?

Not yet. The M5 `Transport` layer (WebSocket hub + targeted push) is the path there. For now, everything runs in one process.

### How do I test my agents offline?

`FakeProvider` is part of the public API — pre-queue JSON responses and pass it as the agent's LLM. See `tests/sdk/test_worldagent.py` and `tests/examples/test_openclaw_team_demo.py` for worked examples.

### Where do I report bugs / request features?

https://github.com/Trustydev212/MWA/issues
