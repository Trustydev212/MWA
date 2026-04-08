# ARCHITECTURE.md — Multi World Agent (MWA)

> Deep dive vào kiến trúc kỹ thuật. Đọc README.md trước.

---

## Tổng quan kiến trúc

MWA là một **runtime layer** ngồi giữa LLM agents và infrastructure. Agents không cần biết về nhau — họ chỉ cần biết về World Model.

```
┌─────────────────────────────────────────────────────────┐
│  AGENTS (external — bất kỳ stack nào)                   │
│  Agent A    Agent B    Agent C    Agent N                │
└──────┬──────────┬──────────┬──────────┬─────────────────┘
       │          │          │          │   MWA SDK / MCP / REST
┌──────▼──────────▼──────────▼──────────▼─────────────────┐
│  MWA RUNTIME                                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Transport Layer (WebSocket Hub)                │   │
│  │  - Agent registry                               │   │
│  │  - Connection management                        │   │
│  │  - Targeted push engine                         │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │  Coordination Layer                             │   │
│  │  - Conflict Detector                            │   │
│  │  - Semantic Arbiter                             │   │
│  │  - Resolution Protocol                         │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │  World Model Layer                              │   │
│  │  - Temporal Knowledge Graph (Graphiti core)     │   │
│  │  - Graph RAG (traversal-based)                  │   │
│  │  - Version management                           │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │  Harness Map Layer                              │   │
│  │  - Dependency graph                             │   │
│  │  - Hard constraints                             │   │
│  │  - Scoring weights                              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────┐
│  GRAPH DATABASE                                         │
│  Neo4j / FalkorDB / Memgraph                            │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Harness Map

### Vai trò
Harness Map là **bản đồ tĩnh** của World — define một lần bởi architect, không thay đổi trong runtime. Đây là "constitution" của World Model.

### Cấu trúc dữ liệu

```python
@dataclass
class HarnessNode:
    name: str
    impact: Literal["critical", "high", "medium", "low"]
    affects: List[str]           # downstream nodes
    affected_by: List[str]       # upstream nodes (computed)
    order: int                   # topological order
    description: str
    
@dataclass  
class HarnessMap:
    version: str
    domain: str
    nodes: Dict[str, HarnessNode]
    hard_constraints: List[str]
    conflict_resolution: ConflictConfig
    
    def traverse_downstream(self, node: str, depth: int = -1) -> List[str]:
        """BFS từ node xuống, trả về all affected nodes"""
        
    def traverse_upstream(self, node: str) -> List[str]:
        """Ngược lại — ai ảnh hưởng đến node này"""
        
    def get_subgraph(self, nodes: List[str]) -> SubGraph:
        """Lấy minimal subgraph chứa nodes và connections"""
        
    def get_topological_order(self) -> List[str]:
        """Thứ tự update an toàn"""
        
    def validate_constraint(self, state: Dict) -> List[Violation]:
        """Check hard constraints trước khi Arbiter chạy"""
```

### Tại sao quan trọng

Không có Harness Map, khi một node thay đổi:
- Không biết phải update gì tiếp
- Phải scan toàn bộ DB
- Arbiter nhận toàn bộ context → chậm, tốn token, kém chính xác

Với Harness Map:
- Traverse xuôi → biết chính xác downstream
- Arbiter nhận subgraph nhỏ → nhanh, rẻ, chính xác
- Update theo topological order → không có circular dependency

---

## Layer 2: World Model

### Nền tảng: Graphiti

MWA fork và extend Graphiti vì:
- Temporal: mỗi fact có `valid_at`, `invalid_at`
- Non-lossy: không xóa, chỉ invalidate
- Hybrid retrieval: semantic + keyword + graph traversal
- Production-proven: Zep dùng trong production

### Temporal Model

```
World = G(N, E, φ) where:
  N = nodes (entities trong world)
  E = edges (relationships + facts)
  φ: E → N × N (incidence function)

Mỗi Edge E có:
  - value: Any
  - valid_at: datetime
  - invalid_at: datetime | None  (None = still valid)
  - agent_id: str               (ai tạo ra fact này)
  - episode_id: str             (bối cảnh khi tạo)
  - confidence: float           (0.0 - 1.0)
  - version: int                (monotonically increasing)
```

### Write Protocol

```python
async def world_write(
    agent_id: str,
    node: str, 
    value: Any,
    confidence: float = 0.8
) -> WriteResult:
    
    # 1. Validate hard constraints TRƯỚC
    violation = harness_map.validate_constraint({node: value})
    if violation:
        return WriteResult.HARD_CONSTRAINT_VIOLATION(violation)
    
    # 2. Check conflict với existing valid facts
    existing = await graph.get_current_fact(node)
    if existing and existing.value != value:
        conflict = Conflict(
            node=node,
            existing=existing,
            proposed=WriteProposal(agent_id, value, confidence)
        )
        
        # 3. Gọi Arbiter nếu cần
        if conflict.severity > AUTO_RESOLVE_THRESHOLD:
            resolution = await arbiter.resolve(conflict)
            return await apply_resolution(resolution)
        else:
            # Auto-resolve theo simple rules
            return await auto_resolve(conflict)
    
    # 4. No conflict — write directly
    episode = Episode(
        agent_id=agent_id,
        node=node,
        value=value,
        confidence=confidence,
        timestamp=now()
    )
    await graph.write_episode(episode)
    
    # 5. Trigger targeted push
    await push_engine.notify_downstream(node, value, harness_map)
    
    return WriteResult.SUCCESS
```

### Read Protocol (Graph RAG)

```python
async def world_read(
    agent_id: str,
    node: str,
    include_upstream: bool = False,
    depth: int = 1
) -> ReadResult:
    
    # Traverse graph — không query toàn bộ DB
    if include_upstream:
        context_nodes = harness_map.traverse_upstream(node)
    else:
        context_nodes = [node]
    
    # Load minimal subgraph
    subgraph = await graph.get_subgraph(context_nodes, depth)
    
    # Return structured context — không phải raw text
    return ReadResult(
        node=node,
        current_value=subgraph.get_current(node),
        context=subgraph,
        version=subgraph.version
    )
```

---

## Layer 3: Conflict Detection & Semantic Arbiter

### Conflict Types

```python
class ConflictType(Enum):
    HARD_CONSTRAINT = "hard_constraint"    # Reject ngay, không Arbiter
    VALUE_CONTRADICTION = "value_contra"  # A says X, B says Y
    ORDERING_CONFLICT = "ordering"        # Thứ tự updates không hợp lệ
    CAUSAL_VIOLATION = "causal"           # B dùng assumption A đã invalid
```

### Arbiter Scoring

```python
class ArbiterScore:
    structural_importance: float  # Bao nhiêu downstream nodes affected?
    causal_depth: float           # Derived từ bao nhiêu steps?
    downstream_impact: float      # Bao nhiêu work phải redo nếu thắng?
    confidence: float             # Agent confidence khi write?
    
    @property
    def total(self) -> float:
        w = harness_map.scoring_weights
        return (
            self.structural_importance * w.structural +
            self.causal_depth * w.causal +
            self.downstream_impact * w.downstream +
            self.confidence * w.confidence
        )
```

### Arbiter Prompt Architecture

```
System: "Bạn là Semantic Arbiter của World Model.
         Nhiệm vụ: resolve conflict giữa agents
         bằng cách phân tích structural impact,
         không phải chọn ai "đúng" theo nghĩa thông thường."

Context (subgraph — nhỏ, targeted):
  - Harness Map subgraph liên quan
  - Current world state của affected nodes
  - Causal history (ai write gì, khi nào)
  - Hard constraints

Conflict:
  - Agent A: {node} = {value_a} (confidence: {conf_a})
  - Agent B: {node} = {value_b} (confidence: {conf_b})

Task:
  Score cả hai proposals theo 4 criteria.
  Return: winner, scores, reason, update_plan

Format: JSON (structured output)
```

### Winner/Loser Protocol

```python
async def apply_resolution(resolution: Resolution) -> WriteResult:
    winner = resolution.winner
    loser = resolution.loser
    
    # 1. Apply winner's state
    await graph.write_episode(winner.proposal)
    
    # 2. Notify loser
    await transport.notify_agent(loser.agent_id, {
        "type": "conflict_lost",
        "node": resolution.node,
        "winner_value": winner.value,
        "reason": resolution.reason,
        "your_score": resolution.loser_score,
        "winner_score": resolution.winner_score
    })
    
    # 3. Loser cleanup protocol
    # Loser đọc winner state, adjust downstream của mình
    loser_downstream = harness_map.traverse_downstream_for_agent(
        loser.agent_id
    )
    await transport.notify_agent(loser.agent_id, {
        "type": "cleanup_required",
        "nodes_to_review": loser_downstream,
        "winner_context": resolution.winner_context
    })
    
    # 4. Push update đến interested agents
    await push_engine.notify_downstream(
        resolution.node,
        winner.value,
        exclude=[loser.agent_id]  # Loser đã được notify riêng
    )
```

---

## Layer 4: Transport (WebSocket + Targeted Push)

### Topology-aware Push

```python
class TargetedPushEngine:
    def __init__(self, harness_map: HarnessMap, registry: AgentRegistry):
        self.harness_map = harness_map
        self.registry = registry
    
    async def notify_downstream(
        self, 
        updated_node: str,
        new_value: Any,
        exclude: List[str] = []
    ):
        # Traverse downstream theo Harness Map
        affected = self.harness_map.traverse_downstream(updated_node)
        
        # Tìm agents đang "subscribe" các nodes đó
        interested = self.registry.get_agents_for_nodes(affected)
        
        # Filter exclusions (e.g., loser đã nhận riêng)
        targets = [a for a in interested if a.id not in exclude]
        
        # Push targeted — concurrent
        await asyncio.gather(*[
            self._push_to_agent(agent, {
                "type": "world_update",
                "source_node": updated_node,
                "new_value": new_value,
                "your_affected_nodes": [
                    n for n in affected 
                    if n in agent.subscribed_nodes
                ],
                "world_version": self.world.current_version
            })
            for agent in targets
        ])
```

### Version Handling

```python
# Agent bắt đầu inference với version V
# Trong khi inference, world update lên V+3
# Agent xong inference, check version

async def on_inference_complete(agent_id, result, started_version):
    current_version = await world.get_version()
    
    if current_version > started_version:
        # World đã thay đổi trong khi agent suy nghĩ
        changed_nodes = await world.get_changes_since(started_version)
        
        # Chỉ retry nếu changed nodes overlap với result
        if result.affects_nodes & changed_nodes:
            # Discard và retry với state mới
            await agent.retry_with_current_state()
        else:
            # Changes không liên quan — apply kết quả
            await world.write_with_version_check(result, started_version)
    else:
        # No conflict with timing
        await world.write(result)
```

---

## Performance Characteristics

### Latency Model

```
Sequential Pipeline (current):
  Total = Σ(inference_i) for i in 1..n
  n=10 agents × 2s = 20s minimum

MWA Parallel:
  Total = max(inference_i) + coordination_overhead
  n=10 agents, longest=2s + 200ms overhead = 2.2s
  
  Speedup: 20s → 2.2s = ~9x faster
  (For independent subtasks)

MWA with Conflict:
  Total = max(inference_i) + arbiter_time + cleanup
  = 2s + 1.5s + 0.5s = 4s
  
  Still: 20s → 4s = 5x faster
```

### Read/Write Complexity

```
Traditional RAG:     O(n) — scan toàn bộ vector DB
Graph RAG (MWA):     O(d) — d = depth của traversal
  d thường = 2-4 hops
  n có thể = 100,000+ nodes
  
Speedup: ~1000x trên large DBs

Traditional broadcast: O(agents) — push tất cả
Targeted push (MWA):   O(affected_agents)
  affected thường = 20-30% total agents
  
Speedup: ~3-5x trên large agent pools
```

---

## Integration Guide

### Với LangChain / LangGraph

```python
from langchain.agents import AgentExecutor
from mwa.integrations.langchain import MWATool

# Add MWA tools vào existing LangChain agent
mwa_tools = MWATool.from_harness_map("./harness.map.json")
agent = AgentExecutor(tools=[...existing_tools, *mwa_tools])
```

### Với Claude Code / Cursor (qua MCP)

```json
// .claude/mcp_servers.json
{
  "mwa": {
    "url": "http://localhost:8766/mcp",
    "description": "Multi World Agent - shared world model"
  }
}
```

### Với bất kỳ HTTP client nào

```bash
# Read a node
GET /api/world/{node}

# Write a node  
POST /api/world/{node}
{"value": "...", "confidence": 0.9, "agent_id": "my_agent"}

# Get conflict log
GET /api/conflicts?limit=10

# Get causal chain
GET /api/causal/{node}
```

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---------|-----------|
| Arbiter sai | Confidence threshold + human escalation |
| Arbiter chậm | Cache common patterns + rule-based fast path |
| DB down | Agent local cache + sync khi reconnect |
| WebSocket drop | Reconnect + replay missed events từ version |
| Infinite loser loop | Max retry limit + human-in-the-loop trigger |
| Causal cycle | Harness Map validation at load time |
| Graph too large | Subgraph sampling + relevance pruning |

---

## What MWA Does NOT Solve

Cần minh bạch về limitations:

```
1. Perfect temporal sync
   Agents vẫn có inference latency
   World thay đổi trong khi agent "suy nghĩ"
   → Partial fix: version checking + retry

2. True shared cognition
   Agents vẫn có separate context windows
   Họ "observe" world, không "experience" nó cùng nhau
   → Fundamental limitation của LLM architecture

3. Zero-latency propagation
   WebSocket push < 10ms, LLM inference = 2-10s
   → Speed mismatch không thể eliminate
   → MWA minimizes impact, không eliminate
   
4. Guaranteed convergence
   Arbiter có thể sai trên edge cases
   → Mitigation: human escalation + audit log
```

Những gì MWA **thực sự giải quyết** là đưa multi-agent coordination từ ~0% structural consistency lên ~85-90% — đủ cho production use cases.

---

*Kiến trúc này là living document — sẽ cập nhật khi prototype phát triển.*
