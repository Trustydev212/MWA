# RESEARCH.md — Build Journal & Architectural Decisions

> Living document. Mỗi milestone build xong → ghi lại lý do quyết định, các
> trade-off đã cân nhắc, và những điều cần nghiên cứu thêm. Mục tiêu là khi
> nhìn lại trong 6 tháng vẫn hiểu *tại sao* code trông thế này.

---

## Naming: SOMA vs MWA

Repo có **hai tên** đang lưu hành:

- **MWA** (Multi World Agent) — tên gốc trong README/ARCHITECTURE.md
- **SOMA** (Shared Observer Multi Agent) — tên repo + commit init

**Quyết định:** code package = `soma`, docs sẽ rename "MWA" → "SOMA" ở
milestone tiếp theo. Lý do: SOMA mô tả chính xác hơn cơ chế (agents là
*observers* của shared world, không phải agents *trong* world). MWA dễ
nhầm với "multi-agent inside one world simulation" (game / robotics).

**Ý nghĩa thêm:** "soma" trong sinh học = thân tế bào, nơi mọi tín hiệu
hội tụ trước khi quyết định hành động. World Model = soma của agent network.

---

## Milestone 0 — Project Foundation

### Decision: pydantic v2 thay vì stdlib dataclass

**Lý do:** mọi layer trong SOMA đều phải defend chống malformed input từ
agents (qua REST/WebSocket/MCP). Pydantic cho free runtime validation +
structured error messages. Stdlib dataclass chỉ cho compile-time type hints.

**Trade-off:** thêm 1 dependency lớn (~5MB). Acceptable vì project đã chắc
chắn dùng pydantic ở SDK layer rồi.

### Decision: `Episode` là frozen model

**Lý do:** temporal knowledge graph philosophy là *không bao giờ mutate*
fact cũ. Khi tone đổi từ "serious" sang "playful", ta tạo episode mới và
mark episode cũ là invalidated, KHÔNG sửa episode cũ. Frozen model enforce
điều này ở runtime.

### Decision: timestamps timezone-aware UTC

**Lý do:** `datetime.utcnow()` deprecated từ Python 3.12. Naive datetimes
gây bug khi compare cross-timezone. Centralise qua `_utcnow()` để test có
thể monkey-patch một symbol duy nhất.

### Decision: mỗi LLM provider là 1 optional extra

**Lý do:** không ai dùng *cả* anthropic + openai + gemini SDK trong cùng
một project. User chỉ install những provider họ cần:
```bash
uv pip install soma[anthropic]              # chỉ Claude
uv pip install soma[anthropic,ollama]       # Claude + local
uv pip install soma[all-llm]                # tất cả
```

### Lessons learned

1. **Lint trước commit cứu rất nhiều thời gian.** Ruff catch `Impact(str, Enum)`
   nên dùng `StrEnum` (Python 3.11+ feature) — nếu không lint sẽ phải refactor
   sau khi viết tests.

2. **Mypy strict mode** bắt được `Returning Any` ở `_holds()` mặc dù logic
   đúng — dạy mình rằng `==` operator có thể trả về `Any` (nếu LHS overrides
   `__eq__` mà không annotate). Wrap `bool(...)` để safe.

---

## Milestone 1 — Harness Map

### Decision: tách Schema (data) khỏi Map (logic)

`HarnessSchema` = pure Pydantic model, không có method nào trừ validators.
`HarnessMap` = wrapper class chứa toàn bộ traversal/cache/algorithms.

**Lý do:**
- Schema layer cheap → tests chỉ care về validation thì không phải build cả map
- Logic layer có thể có expensive precomputation (topological order, reverse
  edges) mà không bị recompute khi schema được tái dùng
- Có thể swap Pydantic ra dataclass sau này mà không phá traversal API

### Decision: precompute reverse edges (`affected_by`) tại load time

**Trade-off:** dùng O(E) memory để có O(1) lookup `affected_by(node)` thay
vì O(N) scan mỗi lần. Vì Harness Map immutable nên precompute là free.

### Decision: topological order qua Kahn's algorithm + cycle detection ở load time

**Lý do:** nếu cycle xuất hiện thì application phải fail ở **startup**, không
phải lần đầu tiên một agent gọi `traverse_downstream()` trong production.

Kahn's algorithm tự nhiên fail khi còn nodes có in-degree > 0 sau khi process
hết → free cycle detection. Tarjan SCC mạnh hơn nhưng overkill cho use case này.

### Decision: tie-break topological order bằng `(order, name)`

**Lý do:** Python dict ordering ổn định trong cùng một process nhưng không
guarantee cross-Python-version (đặc biệt nếu có ai serialize/deserialize).
Stable order = deterministic tests = ít flake.

### Decision: Hard Constraint DSL v0 thay vì DSL phức tạp

Có 4 lựa chọn đã cân nhắc:

| Approach | Pros | Cons |
|----------|------|------|
| **Free-form text** | dễ viết | không programmatic check |
| **JSON schema** | strict | quá verbose, không đọc được tự nhiên |
| **Python expression eval** | maximum flexibility | sandbox security nightmare |
| **Tiny DSL (chosen)** | đọc tự nhiên + parse được | giới hạn expressiveness |

DSL v0 chỉ recognise 3 forms (value_mutex, pair_mutex, freeform). Mọi thứ
khác là "freeform" passthrough — KHÔNG silently ignore, chỉ là không có
programmatic check. Khi có ~10 real harness maps ta sẽ biết DSL cần grow
theo hướng nào (expression-based? prolog-like?).

### Decision: bilingual keywords (Việt + Anh)

**Lý do:** README example dùng tiếng Việt ("không thể đồng thời"), nhưng
SOMA muốn open-source international. Parser support cả hai từ ngày đầu.

**Trade-off:** parser code dài hơn. Acceptable vì DSL nhỏ.

### Decision: `_holds()` support cả scalar và list-valued nodes

**Lý do:** trong production, một node có thể là `tags: ["formal", "serious"]`
chứ không chỉ scalar. Constraint pair_mutex phải fire khi value xuất hiện
TRONG list, không chỉ khi `==`.

### Lessons learned

1. **Parser regex/string-split quá fragile cho Vietnamese.** Dấu cách + dấu
   thanh + filler words ("là", "thì") gây ra `("là 15s", "60s")` thay vì
   `("15s", "60s")`. Phải có `_FILLER_WORDS` lexicon. Nếu DSL grow hơn nữa
   sẽ cần real parser (lark/parsimonious).

2. **Diamond graph là test case quan trọng nhất** cho BFS traversal:
   ```
       A
      / \
     B   C
      \ /
       D
   ```
   Nếu D bị visit 2 lần thì kết quả sẽ có duplicates. Test này catch được
   bug ngay từ lần đầu viết `_bfs()`.

3. **Pydantic v2 `extra="forbid"` rất quan trọng** — nếu không, typo như
   `"afects"` (one f) thay vì `"affects"` sẽ silently tạo node với 0 edges
   và không ai biết. Strict-by-default là default đúng.

4. **Order tests theo độ generality:** schema → traversal → constraints →
   integration. Nếu schema fail thì 30 test khác cũng fail và bạn không biết
   bug ở đâu.

---

## Open Research Questions

Những thứ chưa biết câu trả lời, cần research khi build các milestone sau:

### LLM Provider Layer (Milestone 2)
- **Structured output strategy:** Anthropic dùng tool_use, OpenAI dùng JSON
  mode hoặc function calling, Gemini dùng response_schema. Có 3 option:
  1. Common-denominator (chỉ JSON string + parse) — đơn giản nhưng yếu
  2. Provider-specific best mechanism (chosen) — phức tạp implement
  3. Force everyone qua một format (vd. instructor library)
- **Token counting:** mỗi provider có tokenizer riêng. Có cần đếm chính xác
  hay estimate là đủ cho cost tracking?
- **Retry semantics:** rate limit (429), 5xx, timeout, schema mismatch — mỗi
  cái cần policy khác. Exponential backoff vs token bucket?

### World Model (Milestone 3)
- **In-memory backend interface phải compatible với Graphiti API?**
  Nếu yes thì sau swap dễ. Nếu no thì in-memory layer đơn giản hơn nhưng
  swap sẽ phá API.
- **Bi-temporal vs uni-temporal:** Graphiti có cả `valid_at` và `recorded_at`
  (ai biết, khi nào). Cần cả hai hay chỉ `valid_at`?
- **Episode batching:** khi 1 agent write 10 nodes liên tiếp, có nên gom
  thành 1 batch episode để giảm push noise?

### Semantic Arbiter (Milestone 4)
- **Rule-based vs LLM:** README nói "rule-based trước, LLM sau". Câu hỏi
  thực tế: rule-based có thể giải quyết bao nhiêu % conflict trước khi
  cần LLM? Cần đo trên real harness maps.
- **Arbiter prompt engineering:** structured output format nào cho 4 scoring
  criteria? Chain-of-thought có cần thiết không hay tốn token vô ích?
- **Conflict severity threshold:** khi nào auto-resolve, khi nào escalate
  human? Hard-coded threshold hay learned từ feedback?

### Targeted Push (Milestone 6)
- **Backpressure:** nếu một agent slow, push queue có thể grow vô hạn. Drop
  oldest? Coalesce updates? Error sau N retries?
- **Reconnect protocol:** sau drop, agent cần biết "tôi đã miss những version
  nào". Replay từ version số là đủ hay cần snapshot + delta?

### Production / Open Source (Milestone 7+)
- **Multi-tenancy:** một SOMA runtime serve nhiều World Models đồng thời?
  Hay một runtime per world?
- **Authentication:** agents identify như thế nào? API key per agent?
- **Audit log:** mỗi episode + arbitration cần persistent log để debug.
  Format: structured logs (JSON) hay timeline DB riêng?

---

## Reading list (papers & projects to study)

- [ ] **Graphiti** source code — đặc biệt phần temporal fact invalidation
- [ ] **Zep** production deployment patterns
- [ ] **CRDT papers** (Shapiro et al. 2011) — convergence guarantees có thể
      apply cho conflict-free fields trong World Model
- [ ] **Blackboard architecture** (Hayes-Roth 1985) — origin của shared
      world model concept
- [ ] **CodeCRDT** (2025) — multi-agent code generation với CRDT
- [ ] **MCP spec** — kiểm tra cách MCP serialize structured data, có thể
      reuse cho SOMA's WorldInspector API
