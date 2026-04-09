# Changelog

All notable changes to **MWA — Multi World Agent** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet — next on deck is M5 (Transport / WebSocket) and M8
(Graphiti backend).  Neither is blocking any current use case;
order depends on which one a real user asks for first._

## [0.1.1] — 2026-04-09

### Changed

- **PyPI distribution name**: `mwa` → `multi-world-agent`.  The short
  three-letter `mwa` name was already reserved on PyPI by a different
  account, so every attempt to publish ``0.1.0`` returned a
  ``403 The user 'Phamtrusty' isn't allowed to upload to project 'mwa'``.
  Renaming the distribution unblocks the first public release.
- **Python module name is unchanged**: ``import mwa`` still works —
  only the PyPI distribution name is longer.  Users run
  ``pip install multi-world-agent`` but then ``from mwa.sdk import
  WorldAgent``.  Same pattern as ``pip install scikit-learn`` /
  ``import sklearn``.
- README, `docs/getting-started.md`, `docs/release.md`, and the
  PyPI/Python badge links all updated to the new distribution name.

### Technical note

The ``mwa`` name on PyPI remains reserved — if a future maintainer
is able to claim it through PyPI's abandoned-project process
(<https://pypi.org/help/#project-name>), we can add ``mwa`` back as
a secondary distribution alongside ``multi-world-agent`` rather than
renaming again.  Until then, ``multi-world-agent`` is the canonical
PyPI name.

## [0.1.0] — 2026-04-08

First public alpha.  Ships milestones M0 through M7 plus the
OpenClaw team demo, verified end-to-end against a live
OpenAI-compatible gateway.

### Added

#### Core layers (M0 – M3)

- **`mwa.types`** — Pydantic v2 frozen data models: `Episode`, `Fact`,
  `Conflict`, `WriteProposal`, `Impact`, `WriteResult`.  Timezone-aware
  UTC timestamps throughout.
- **`mwa.errors`** — exception hierarchy rooted at `MWAError`.
- **`mwa.harness`** — `HarnessMap` loader with BFS/DFS traversal,
  Kahn topological order + cycle detection at load time, subgraph
  extraction for prompt building, and a tiny bilingual (Vietnamese
  + English) hard-constraint DSL (`value_mutex`, `pair_mutex`,
  `freeform`).
- **`mwa.world`** — `WorldModelProtocol` + `InMemoryWorldModel`.
  Append-only temporal storage, `asyncio.Lock`-serialised writes,
  hard-constraint enforcement inside `apply()`.
- **`mwa.arbiter.detector`** — `ConflictDetector` with `check()` and
  batch `partition()` helpers.
- **`mwa.arbiter.rules`** — `RuleBasedResolver` with three
  deterministic rules (idempotent write, causal acknowledged,
  dominant confidence) and an explicit escalate-on-ambiguity
  default.  No silent last-write-wins fallback.

#### LLM provider layer (M2)

- **`mwa.llm.base`** — `LLMProvider` Protocol, neutral
  `Message`/`ChatOptions`/`ChatResponse` types, three-level error
  hierarchy (`RateLimitError`, `TransientProviderError`,
  `PermanentProviderError`).
- **`mwa.llm.retry`** — `RetryPolicy` with exponential backoff +
  decorrelated jitter.
- **`mwa.llm.cost`** — `PricingTable` with `Decimal` arithmetic for
  token cost accounting.
- **`mwa.llm.router`** — `LLMRouter` with primary/fallback chain,
  per-call budget gate, and retry integration.
- **`mwa.llm.providers`** — `FakeProvider` (public, for tests),
  `AnthropicProvider` (tool_use structured output),
  `OpenAIProvider` (with `structured_strategy` auto-fallback from
  `json_schema` → `json_object`, markdown-fence stripping, forceful
  schema hinting — battle-tested against futrixapi), and
  `OllamaProvider` (httpx-based, local `format=json`).
- Lazy SDK imports: importing `mwa.llm.providers` never pulls vendor
  SDKs unless they're actually installed.

#### Semantic Arbiter (M4)

- **`mwa.arbiter.SemanticArbiter`** — LLM-backed conflict resolver
  with subgraph-scoped prompt building.
- **`mwa.arbiter.ArbiterDecision`** — Pydantic structured output
  schema (9 fields, 2 nested score models, `Literal["existing",
  "proposed"]` winner).
- Confidence-gated auto-apply with escalation to human-review when
  `overall_confidence < auto_resolve_threshold` (default 0.85).
- **`HarnessMap.conflict_resolution`** property for SDK callers.

#### WorldAgent SDK (M6)

- **`mwa.sdk.WorldAgent`** — named agent with its own LLM provider
  and `@agent.on("node")` handler decorator.
- **`mwa.sdk.AgentRuntime`** — shared world + arbiter + FIFO event
  queue + reactive dispatcher.  Sequential dispatch in 0.1 (parallel
  is a later optimisation).
- **`mwa.sdk.AgentContext`** / **`AgentWorldView`** — per-invocation
  handle with agent-scoped `ctx.world.write()` that stamps
  `agent_id` and runs the full conflict pipeline automatically.
- Dispatcher safeguards: self-reaction skip (agents don't react to
  their own writes), `max_iterations` guard catches cross-agent
  `A → B → A` cycles.

#### MCP server (M7)

- **`mwa.mcp_server`** package with layered architecture:
  - `tools.py` — pure async handlers, no `mcp` SDK import, testable
    offline.
  - `server.py` — MCP protocol wiring, lazy-imports the `mcp`
    package.
  - `__main__.py` — CLI entry point (`python -m mwa.mcp_server`).
- **`mwa-mcp`** console script installed with `pip install mwa[mcp]`.
- Five exposed tools: `mwa_list_nodes`, `mwa_read_world`,
  `mwa_read_history`, `mwa_seed_world`, `mwa_submit_write`.

#### OpenClaw example

- **`examples/openclaw_team/`** — four builder agents (architect,
  security, provider_selector, deployer) cooperatively design an
  AI agent from a natural-language user intent.  Runs offline with
  FakeProvider or live against any `MWA_LIVE_*` endpoint.
- **`harness_maps/openclaw_agent_builder.json`** — 14-node harness
  map representing the "design an AI agent" domain.

#### Distribution (M7)

- PEP 561 `py.typed` marker — downstream type checkers see our
  annotations.
- `pyproject.toml` ships harness maps + examples + docs in the sdist.
- New optional extras: `[mcp]`, `[all]`.
- `mwa-mcp` console script entry point.
- Python 3.11+ required.

#### Documentation

- **`README.md`** — vision, architecture, OpenClaw example,
  multi-provider LLM table.
- **`ARCHITECTURE.md`** — per-layer deep dive with failure modes.
- **`RESEARCH.md`** — build journal with design decisions and
  three documented futrixapi interop findings.
- **`docs/getting-started.md`** — install-to-running in 10 minutes.
- **`docs/release.md`** — release checklist for maintainers.
- **`examples/openclaw_team/README.md`** — demo narrative.

#### CI

- **`.github/workflows/live-llm-test.yml`** — manual-dispatch
  integration suite against any OpenAI-compatible endpoint.  Uses
  `MWA_LIVE_*` repo secrets.  Opt-in Node.js 24 runtime.
- **`.github/workflows/release.yml`** — tag-driven PyPI release with
  full test + build + smoke-install + `twine check` gates.

### Changed

- Package renamed `soma` → `mwa`.  Single canonical name across
  repo, package, and docs.
- `LLMProvider.stream` Protocol signature fixed (`async def` →
  `def`) so async-generator implementations type-check correctly
  under `mypy --strict`.

### Security

- `.github/workflows/live-llm-test.yml` masks secrets in logs via
  GitHub Actions default.
- `LICENSE` file (MIT) shipped in sdist.

### Infrastructure

- 223 tests passing (6 live tests skipped without secrets).
- `mypy --strict` clean on 33 source files.
- `ruff check` clean.
- Python 3.11 on Linux CI runners.
- Uses `uv` for dependency installation in CI workflows.
