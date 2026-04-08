# OpenClaw Team Demo

Bốn agent chuyên biệt cùng build một agent mới cho user, phối hợp qua
**World Model dùng chung** — không nói chuyện trực tiếp với nhau.

Đây là manifest demo của MWA: *"Agents don't talk to each other. They
observe the same world."*

## Chạy thế nào

```bash
# Mặc định: offline, dùng FakeProvider, zero network, deterministic
uv run python -m examples.openclaw_team.run_demo
```

```bash
# Live mode: dùng LLM thật qua OpenAI-compatible endpoint
export MWA_LIVE_API_KEY=sk-...
export MWA_LIVE_BASE_URL=https://futrixapi.com/v1
export MWA_LIVE_MODEL=auto
uv run python -m examples.openclaw_team.run_demo
```

Trong live mode, provider thực được wrap trong `LLMRouter + RetryPolicy`
— cùng pattern mà `tests/arbiter/test_semantic_live.py` dùng để survive
gateway flakes (Cloudflare 504 etc).

## Kịch bản

User nói:

> "Build a research agent that reads arXiv papers and extracts claims
> with confidence scores."

OpenClaw Team gồm 4 agent sub-agent:

| Agent | Reads | Writes | Vai trò |
|---|---|---|---|
| **Architect-Agent** | `user_intent` | `agent_architecture`, `sub_agent_count`, `orchestration_pattern` | Thiết kế kiến trúc |
| **Security-Agent** | `user_intent`, `agent_architecture` | `tool_permissions`, `memory_strategy` | Guardrails |
| **Provider-Selector-Agent** | `user_intent`, `agent_architecture`, `sub_agent_count` | `llm_provider`, `cost_budget`, `latency_budget` | LLM backend chọn lọc |
| **Deployer-Agent** | `agent_architecture`, `memory_strategy`, `tool_permissions` | `deployment_target`, `observability`, `error_handling` | Runtime surface |

Cả 4 agent cùng view + ghi vào một `InMemoryWorldModel` duy nhất. Không
ai import ai, không ai call ai — chỉ read/write World Model.

## Flow

### Stage A — Happy path

```
     ┌──────────────────┐
     │  user_intent     │ ← user seeds
     └────────┬─────────┘
              │
        ┌─────▼──────┐
        │ Architect  │ (sequential — everyone depends on it)
        └─────┬──────┘
              │ writes agent_architecture, sub_agent_count,
              │        orchestration_pattern
      ┌───────┴──────────┐
      │                  │
┌─────▼──────┐   ┌───────▼────────┐
│  Security  │   │ ProviderSelect │ (concurrent via asyncio.gather)
└─────┬──────┘   └───────┬────────┘
      │ writes perms,    │ writes llm, cost, latency
      │        memory    │
      └───────┬──────────┘
              │
         ┌────▼─────┐
         │ Deployer │ (sequential — needs Security's memory + perms)
         └──────────┘
         writes target, observability, error_handling
```

Kết thúc Stage A, World Model chứa đầy đủ spec của agent mới:

```
user_intent            'Build a research agent...'
agent_architecture     'multi_agent'
sub_agent_count        3
orchestration_pattern  'sequential'
tool_permissions       'read_only'
memory_strategy        'short_term_conversation'
llm_provider           'anthropic_claude'
cost_budget            0.75
latency_budget         30000
deployment_target      'cloud'
observability          'traces'
error_handling         'fallback'
```

### Stage B — Conflict + Semantic Arbiter

Architect đổi ý: research agent nên có **`long_term_persistent`** memory
để cache abstracts across sessions. Ghi mới với confidence 0.84 — gần
bằng Security's 0.82. Rule-based resolver không đủ confident → escalate
đến **Semantic Arbiter**:

```
Architect: memory_strategy = 'long_term_persistent' (conf 0.84)
                   │
                   ▼
          ConflictDetector → conflict!
                   │
                   ▼
       RuleBasedResolver → close gap → ESCALATE
                   │
                   ▼
         SemanticArbiter (LLM call)
                   │
                   ▼
   Decision: winner='existing' (Security's value stays)
   Reason: "Short-term better matches the per-paper pipeline;
           persistent store adds complexity without benefit."
                   │
                   ▼
    Architect's write → REJECTED, in audit log
    World state unchanged
```

Point demo: **rejected write không mất** — nó ở trong audit log với
`rejected=True`. Caller có thể trace lại mọi decision history.

## Cấu trúc code

```
examples/openclaw_team/
├── README.md          ← file này
├── __init__.py
├── run_demo.py        ← entry point + staged orchestration
├── runtime.py         ← build_runtime, write_to_world, make_*_provider
├── agents.py          ← 4 agent async functions
├── schemas.py         ← Pydantic structured-output schemas
└── fake_responses.py  ← Canned JSON cho offline mode
```

### Lưu ý thiết kế

- **Mỗi agent là 1 hàm async đơn giản**, không phải class kế thừa. Đây
  là intentional — M6 (`WorldAgent` SDK) sẽ add abstraction layer. Đến
  lúc đó demo sẽ được refactor để show "cùng thứ với SDK thì ngắn hơn".

- **Mỗi agent có schema output riêng** — tight, specific. Không có "god
  schema" chung cho 4 agent. Lý do: LLM weaker dễ follow schema nhỏ hơn
  (lesson learned từ M4 interop debugging với futrixapi).

- **`write_to_world()` helper trong `runtime.py`** chạy full pipeline
  `detect → rule → semantic → apply/reject`. Đây là exact shape mà
  `WorldAgent.world.write()` sẽ có sau khi M6 ship — centralising ngay
  đây là để khi refactor sang SDK, chỉ cần extract helper này thành
  method.

- **Fake vs Live provider**: `is_live_mode()` check env vars. Offline
  mode dùng per-agent `FakeProvider` với queued JSON responses —
  deterministic. Live mode dùng `LLMRouter` wrapping real
  `OpenAIProvider` — retry-protected, production-shaped.

## Smoke test

```bash
pytest tests/examples/test_openclaw_team_demo.py -v
```

Test chạy full demo với FakeProvider và verify:
- World state cuối cùng chứa đủ 12 node
- Stage B resolution đúng là `KEEP_EXISTING`
- Audit log có đúng 2 episodes cho `memory_strategy` (1 applied, 1 rejected)

## Dự kiến trong M6

Sau khi `WorldAgent` SDK ship, file `agents.py` sẽ rút ngắn khoảng
60%. Pattern sẽ thành:

```python
# hypothetical M6 shape
from mwa import WorldAgent

architect = WorldAgent(
    name="architect_agent",
    llm=LLMProvider.anthropic(model="claude-opus-4-6"),
)

@architect.on_world_update("user_intent")
async def design(intent: str, context) -> ArchitectDecision:
    return await architect.llm.structured(
        messages=[...],
        schema=ArchitectDecision,
    )

# Auto-wire decision fields → world writes, auto-handle conflict flow.
architect.start()
```

Demo này là **input** cho M6 design — mọi pain point gặp phải khi
viết `agents.py` sẽ drive SDK shape.
