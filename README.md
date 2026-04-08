# MWA — Multi World Agent

> **"Agents don't talk to each other. They observe the same world."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)]()
[![Stack: Python + Neo4j + WebSocket](https://img.shields.io/badge/Stack-Python%20%2B%20Neo4j%20%2B%20WebSocket-blue)]()

---

## Tại sao MWA tồn tại

Tất cả multi-agent systems hiện tại đều dùng **message-passing** làm substrate — agents gửi text cho nhau, parse lại, rồi tiếp tục. Đây là kiến trúc sai về mặt cơ bản:

```
Hiện tại:   Agent A → "tôi đã làm X, kết quả Y" → Agent B parse text
MWA:        Agent A writes world.X = Y  →  Agent B observes world.X ngay lập tức
```

MWA giải quyết bài toán: **làm sao để nhiều agents chia sẻ một "thực tại" chung đủ nhất quán để phối hợp thực sự** — không phải qua pipeline tuần tự, không phải qua message chain.

---

## Tên & Ý nghĩa

**Multi World Agent** — không phải "multi-agent trong một world" mà là kiến trúc cho phép agents **cùng nhìn vào, cùng tác động lên, và cùng hiểu một World Model duy nhất**, bất kể agents đó thuộc domain nào, chạy trên stack nào, hay phục vụ mục đích gì.

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                    MWA RUNTIME                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LAYER 1: HARNESS MAP                               │   │
│  │  Bản đồ dependency của World — ai ảnh hưởng ai      │   │
│  │  harness.map.json  ←  define một lần, dùng mãi     │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │  LAYER 2: WORLD MODEL (Graph DB)                    │   │
│  │  Neo4j / FalkorDB / Memgraph                        │   │
│  │  - Nodes: entities của world                        │   │
│  │  - Edges: relationships + dependency                │   │
│  │  - Temporal: mỗi fact có validity window            │   │
│  │  - Versioned: mọi thay đổi đều có trace             │   │
│  └──────┬───────────────────────────────────┬──────────┘   │
│         │ write                             │ read          │
│  ┌──────▼──────────┐               ┌────────▼─────────┐    │
│  │ CONFLICT        │               │  GRAPH RAG       │    │
│  │ DETECTOR        │               │  Traverse graph  │    │
│  │                 │               │  không scan full │    │
│  │ Phát hiện khi   │               │  O(depth)        │    │
│  │ 2+ agents write │               │  không O(n)      │    │
│  │ contradicting   │               └────────┬─────────┘    │
│  │ values          │                        │              │
│  └──────┬──────────┘                        │              │
│         │ conflict detected                 │              │
│  ┌──────▼──────────────────────────────┐    │              │
│  │  LAYER 3: SEMANTIC ARBITER (LLM)   │    │              │
│  │                                    │    │              │
│  │  Input:                            │    │              │
│  │    - State A vs State B            │    │              │
│  │    - Harness Map (subgraph only)   │    │              │
│  │    - Causal history                │    │              │
│  │    - Hard constraints              │    │              │
│  │                                    │    │              │
│  │  Scoring:                          │    │              │
│  │    1. Structural importance        │    │              │
│  │    2. Causal depth                 │    │              │
│  │    3. Downstream impact            │    │              │
│  │    4. Confidence                   │    │              │
│  │                                    │    │              │
│  │  Output:                           │    │              │
│  │    Winner + Reason + Update Plan   │    │              │
│  └──────┬──────────────────────────────┘    │              │
│         │ resolution                        │              │
│  ┌──────▼──────────────────────────────┐    │              │
│  │  LAYER 4: TARGETED WEBSOCKET PUSH  │    │              │
│  │                                    │    │              │
│  │  Không broadcast tất cả            │    │              │
│  │  → Traverse graph downstream       │    │              │
│  │  → Chỉ push agents cần biết        │    │              │
│  │  → Latency: O(depth), không O(n)   │    │              │
│  └──────────────────────────────────────    │              │
│                                             │              │
│  ┌──────────────────────────────────────────┘              │
│  │         AGENTS (bất kỳ)                                 │
│  │                                                         │
│  │  Agent A          Agent B          Agent N              │
│  │  (script)         (visual)         (caption)            │
│  │     │                │                │                 │
│  │     └────────────────┴────────────────┘                 │
│  │              connect to MWA Runtime                     │
│  │              qua SDK / MCP / REST                       │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

---

## 5 Thành phần cốt lõi

### 1. Harness Map (`harness.map.json`)
Bản đồ dependency của World Model — define một lần, tất cả agents dùng chung.

```json
{
  "version": "1.0",
  "domain": "video_ad_production",
  "nodes": {
    "campaign_goal": {
      "impact": "critical",
      "affects": ["tone", "duration", "cta", "target_audience"],
      "order": 1,
      "description": "Mục tiêu chính của campaign"
    },
    "tone": {
      "impact": "high",
      "affects": ["script_language", "visual_style", "music_mood"],
      "order": 2,
      "description": "Tone tổng thể của video"
    },
    "duration": {
      "impact": "high",
      "affects": ["scene_count", "script_length", "cut_pace"],
      "order": 2,
      "description": "Độ dài video tính bằng giây"
    },
    "visual_style": {
      "impact": "medium",
      "affects": ["color_palette", "font_choice", "animation_type"],
      "order": 3
    },
    "script_language": {
      "impact": "medium",
      "affects": ["caption_style", "voiceover_tone"],
      "order": 3
    }
  },
  "hard_constraints": [
    "duration không thể đồng thời là 15s và 60s",
    "tone không thể đồng thời là formal và casual",
    "visual_style cartoon không compatible với tone corporate"
  ],
  "conflict_resolution": {
    "default_strategy": "arbiter",
    "auto_resolve_threshold": 0.85,
    "scoring_weights": {
      "structural_importance": 0.35,
      "causal_depth": 0.25,
      "downstream_impact": 0.25,
      "confidence": 0.15
    }
  }
}
```

---

### 2. World Model (Graph DB)

Dùng **Graphiti** làm core engine (open source, temporal, hybrid retrieval):

```
World Model = Temporal Knowledge Graph
  - Nodes:  entities (campaign_goal, tone, script, visual...)
  - Edges:  relationships + causal dependencies
  - Facts:  mỗi fact có valid_at, invalid_at
  - Episodes: mỗi agent write = một episode có provenance
```

Khi Agent A viết `tone = "vui vẻ"`:
- Fact cũ `tone = "nghiêm túc"` bị invalidate (không xóa)
- Fact mới được tạo với timestamp
- Causal edge từ Agent A được ghi
- Downstream nodes được traverse để identify ai cần biết

---

### 3. Semantic Arbiter

LLM layer được gọi khi Conflict Detector phát hiện contradiction:

```python
class SemanticArbiter:
    def resolve(self, conflict: Conflict) -> Resolution:
        # Nhận subgraph — không phải toàn bộ DB
        subgraph = harness_map.get_subgraph(conflict.nodes)
        
        prompt = f"""
        Conflict detected in World Model:
        
        Agent A proposes: {conflict.state_a}
        Agent B proposes: {conflict.state_b}
        
        Subgraph context:
        {subgraph.to_structured()}
        
        Causal history:
        {conflict.causal_history}
        
        Hard constraints:
        {harness_map.hard_constraints}
        
        Score each proposal on:
        1. Structural importance (0-1): how many downstream nodes affected?
        2. Causal depth (0-1): how deeply derived is this state?
        3. Downstream impact (0-1): how much rework if this wins?
        4. Confidence (0-1): how certain is the agent?
        
        Return JSON: winner, scores, reason, update_plan
        """
        
        return llm.call(prompt, structured_output=Resolution)
```

**Loser Protocol:**
- Loser đọc winner's state + reason
- Loser clean state của mình để align với winner
- Loser không làm mất tất cả work — chỉ adjust phần conflict
- Update plan được execute theo ordered batch, không phải từng item

---

### 4. Targeted WebSocket Push

Thay vì broadcast tất cả, push theo graph topology:

```python
class TargetedPush:
    def on_world_update(self, updated_node: str, new_value: Any):
        # Traverse downstream theo Harness Map
        affected_nodes = harness_map.traverse_downstream(updated_node)
        
        # Tìm agents đang work trên affected nodes
        interested_agents = agent_registry.get_agents_for(affected_nodes)
        
        # Push targeted — không broadcast
        for agent in interested_agents:
            websocket.push(agent.id, {
                "type": "world_update",
                "node": updated_node,
                "value": new_value,
                "affected_nodes": affected_nodes,
                "version": world_model.current_version
            })
```

**Kết quả:**
- Không agent nào nhận update không liên quan
- Latency: O(depth của graph), không O(số agents)
- Mỗi agent chỉ react với thứ thực sự ảnh hưởng mình

---

### 5. Agent SDK

```python
from mwa import WorldAgent, connect

# Kết nối vào MWA Runtime
agent = WorldAgent(
    name="script_writer",
    harness_map="./harness.map.json",
    llm_api_key="..."
)

# Đọc từ World Model (Graph RAG — không scan full DB)
tone = agent.world.read("tone")
duration = agent.world.read("duration")

# Viết vào World Model
agent.world.write("script_language", "conversational", confidence=0.9)

# React khi world thay đổi
@agent.on_world_update("tone")
def handle_tone_change(new_value, reason):
    # Cập nhật script language theo tone mới
    agent.world.write("script_language", adapt_to_tone(new_value))

# Start — agent chạy concurrent với các agents khác
agent.start()
```

---

## Vấn đề MWA giải quyết

### Hiện tại

| Vấn đề | Pipeline truyền thống | MWA |
|--------|----------------------|-----|
| Latency | O(n agents) tích lũy | O(depth) — graph traverse |
| Quality decay | 0.95^100 = 0.6% | Arbiter giữ coherence |
| Conflict | Last-write-wins hoặc crash | Semantic scoring + winner protocol |
| Debug | Trace 100 messages | Graph timeline + causal history |
| Shared state | Không tồn tại | World Model là source of truth |
| Bidirectional deps | Không thể | Concurrent với targeted push |

### Tương lai

```
1. AI-first software development
   Agent teams build software
   với shared codebase understanding
   → Không còn context loss giữa agents

2. Enterprise AI workflows
   100+ agents coordinate
   trên same business world model
   → Không còn "AI silos" trong org

3. Autonomous systems
   Robotics, IoT, simulation
   → Multiple AI controllers
      share one physical world model

4. Research acceleration
   Agent research teams
   share findings realtime
   build on each other's work
   → Science faster

5. Agent World Protocol (AWP)
   Như HTTP cho web
   Như MCP cho tools
   → Standard cho shared agent reality
```

---

## Phân biệt với existing solutions

```
LangChain / LangGraph:
  ✓ Orchestration, workflow
  ✗ Shared semantic state
  ✗ Conflict resolution
  ✗ Realtime bidirectional

AutoGen / CrewAI:
  ✓ Agent roles, communication
  ✗ Shared world model
  ✗ Graph-aware RAG
  ✗ Semantic Arbiter

Graphiti / Zep:
  ✓ Temporal knowledge graph
  ✓ Real-time ingestion
  ✗ Multi-agent concurrent writes
  ✗ Harness Map
  ✗ Semantic Arbiter
  ✗ Targeted WebSocket

MWA (this project):
  ✓ Tất cả ở trên
  ✓ Harness Map = dependency structure
  ✓ Semantic Arbiter = LLM conflict resolution
  ✓ Targeted WebSocket = graph-topology-aware push
  ✓ Universal — không lock vào domain
  ✓ Chi phí thấp — build trên existing stack
```

---

## Tech Stack

```
Core Graph Engine:    Graphiti (fork & extend)
Graph Database:       Neo4j (default) / FalkorDB (faster) / Memgraph (WebSocket native)
Realtime:             WebSocket (native Python / Socket.io)
Arbiter LLM:          Claude API (Anthropic) / OpenAI / bất kỳ LLM nào
Language:             Python 3.11+
Package Manager:      uv
Testing:              pytest + pytest-asyncio
Observability:        OpenTelemetry
Distribution:         PyPI package + MCP server + Docker Compose
```

---

## Repository Structure

```
mwa/
├── README.md                    ← file này
├── ARCHITECTURE.md              ← deep dive kiến trúc
├── RESEARCH.md                  ← research notes & papers
│
├── mwa/                         ← core package
│   ├── __init__.py
│   ├── world_model/
│   │   ├── graph.py             ← World Model (wraps Graphiti)
│   │   ├── temporal.py          ← Temporal fact management
│   │   └── versioning.py        ← Version + snapshot
│   │
│   ├── harness/
│   │   ├── map.py               ← Harness Map loader + traversal
│   │   ├── constraints.py       ← Hard constraint validation
│   │   └── schema.py            ← harness.map.json schema
│   │
│   ├── arbiter/
│   │   ├── detector.py          ← Conflict Detector
│   │   ├── arbiter.py           ← Semantic Arbiter LLM
│   │   ├── scoring.py           ← Scoring functions
│   │   └── resolution.py        ← Winner/Loser protocol
│   │
│   ├── transport/
│   │   ├── websocket.py         ← WebSocket hub
│   │   ├── push.py              ← Targeted push logic
│   │   └── registry.py          ← Agent registry
│   │
│   ├── rag/
│   │   ├── graph_rag.py         ← Graph-aware RAG
│   │   └── traversal.py         ← BFS/DFS traversal
│   │
│   └── sdk/
│       ├── agent.py             ← WorldAgent class
│       └── decorators.py        ← @on_world_update etc
│
├── mcp_server/                  ← MCP server cho Claude/Cursor/Codex
│   ├── server.py
│   └── tools.py
│
├── examples/
│   ├── quickstart/              ← 2 agents, minimal setup
│   ├── neuvell_studio/          ← Video ad production use case
│   └── code_team/               ← 3-agent coding team
│
├── harness_maps/                ← Example harness maps
│   ├── video_production.json
│   ├── software_development.json
│   └── research_pipeline.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker/
│   ├── docker-compose.yml       ← Neo4j + MWA + example agents
│   └── Dockerfile
│
└── docs/
    ├── getting-started.md
    ├── harness-map-guide.md
    ├── arbiter-guide.md
    └── api-reference.md
```

---

## Quick Start (Dev Guide)

### 1. Setup môi trường

```bash
# Clone repo
git clone https://github.com/your-org/mwa.git
cd mwa

# Install dependencies (dùng uv)
pip install uv
uv sync

# Hoặc pip
pip install -e ".[dev]"
```

### 2. Start infrastructure

```bash
# Start Neo4j + MWA server
docker-compose up -d

# Verify
curl http://localhost:7474  # Neo4j browser
curl http://localhost:8765  # MWA WebSocket server
```

### 3. Define Harness Map

```bash
cp harness_maps/video_production.json ./my_harness.map.json
# Edit để phù hợp với domain của bạn
```

### 4. Build agents đầu tiên

```python
# agent_a.py — Script Writer
from mwa import WorldAgent

agent = WorldAgent(
    name="script_writer",
    harness_map="./my_harness.map.json",
    mwa_url="ws://localhost:8765",
    llm_api_key="your_anthropic_key"
)

@agent.on_world_update("campaign_goal")
async def on_goal_change(new_value, context):
    script = await generate_script(new_value, context)
    await agent.world.write("script_content", script, confidence=0.85)

await agent.start()
```

```python
# agent_b.py — Visual Planner
from mwa import WorldAgent

agent = WorldAgent(
    name="visual_planner",
    harness_map="./my_harness.map.json",
    mwa_url="ws://localhost:8765",
    llm_api_key="your_anthropic_key"
)

@agent.on_world_update("tone")
async def on_tone_change(new_value, context):
    visual_style = await plan_visuals(new_value, context)
    await agent.world.write("visual_style", visual_style, confidence=0.9)

await agent.start()
```

```bash
# Chạy cả hai agents đồng thời
python agent_a.py &
python agent_b.py &
```

### 5. Observe world state

```python
from mwa import WorldInspector

inspector = WorldInspector(mwa_url="ws://localhost:8765")

# Xem toàn bộ world state hiện tại
state = await inspector.get_current_state()

# Xem conflict history
conflicts = await inspector.get_conflict_log()

# Xem causal chain của một node
chain = await inspector.get_causal_chain("visual_style")
```

---

## Development Roadmap

```
Phase 1 — Core (Tuần 1-4)
  [ ] Fork Graphiti, add MWA layers
  [ ] Harness Map loader + traversal
  [ ] Basic Conflict Detector
  [ ] Simple Arbiter (rule-based trước)
  [ ] WebSocket hub cơ bản
  [ ] WorldAgent SDK minimal
  [ ] Quickstart example chạy được

Phase 2 — Semantic (Tuần 5-8)
  [ ] Semantic Arbiter với LLM scoring
  [ ] Winner/Loser protocol đầy đủ
  [ ] Targeted WebSocket push theo graph
  [ ] Graph RAG traversal
  [ ] Hard constraint validation
  [ ] Conflict history + audit log

Phase 3 — Production (Tuần 9-12)
  [ ] MCP server
  [ ] Docker Compose template
  [ ] PyPI package
  [ ] Performance benchmarks
  [ ] Neuvell Studio integration
  [ ] Documentation đầy đủ

Phase 4 — Open Source (Tháng 4+)
  [ ] Public release
  [ ] Community harness maps
  [ ] Plugin system cho custom Arbiters
  [ ] Agent World Protocol (AWP) spec draft
```

---

## Research Background

Kiến trúc MWA được xây dựng dựa trên:

- **Blackboard Architecture** (Hayes-Roth, 1985) — foundation của shared world model
- **Graphiti / Zep** (2025) — temporal knowledge graph engine
- **CRDT** (Shapiro et al., 2011) — inspiration cho conflict-free state management
- **Shared Cognitive Substrates** (ICLR 2026 Workshop) — theoretical framework
- **Multi-Agent Memory** (Architecture 2.0, 2026) — memory hierarchy for agents
- **CodeCRDT** (2025) — CRDT applied to multi-agent code generation

**Khoảng trắng MWA lấp đầy:**
Không có project nào kết hợp: Temporal Graph DB + Harness Map + Semantic Arbiter LLM + Targeted WebSocket + Multi-agent concurrent write protocol thành một coherent system.

---

## Contributing

```bash
# Setup dev environment
uv sync --all-extras

# Run tests
pytest tests/

# Run linter
ruff check mwa/

# Format
ruff format mwa/
```

PR welcome. Focus areas:
- Harness Map templates cho các domains khác nhau
- Alternative Arbiter strategies (rule-based, hybrid)
- Graph DB backend adapters
- Performance benchmarks

---

## License

MIT — free to use, modify, distribute.

---

## Contact & Status

**Status:** Active research prototype — đang build Phase 1

**Tên đầy đủ:** Multi World Agent Runtime (MWA)  
**Tagline:** *"Agents don't talk to each other. They observe the same world."*

---

*Built on the shoulders of Graphiti, Neo4j, and years of distributed systems research.*  
*The missing layer between AI agents and true coordination.*
