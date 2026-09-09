# Memnest Memory MCP Server

[![PyPI version](https://badge.fury.io/py/memnest-mcp.svg)](https://pypi.org/project/memnest-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Persistent graph memory for AI agents using [LadybugDB](https://ladybugdb.com/) — an embedded graph database with native vector search and full-text search.

Give your AI agent memory that persists across sessions, deduplicates automatically, and models knowledge as a graph with typed relationships.

## Why Memnest?

- **Graph memory** — memories linked via Topic nodes and relationships (RELATED_TO, SUPERSEDES, EXPLAINS) with Cypher queries
- **Three-layer auto-dedup** — exact hash + semantic similarity + LLM-driven consolidation
- **Workspace namespacing** — memories scoped per project; `global_search` opt-out
- **HNSW vector search** — fast cosine similarity over FastEmbed embeddings
- **Topic auto-linking** — tags become graph nodes, enabling traversal queries
- **Embedded** — no Docker, no server process, single database directory
- **Zero config** — sensible defaults, just install and run
- **Importance & access tracking** — memories ranked by relevance and usage

## Benchmarks

Memnest scores **82.9%** on the [LOCOMO benchmark](https://snap-research.github.io/locomo/) — the standard evaluation for long-term conversational memory (ACL 2024).

| Category | Score |
|----------|-------|
| Single-hop | 84.4% |
| Multi-hop | 76.9% |
| Open-domain | 85.7% |
| Temporal | 86.5% |
| Adversarial | 76.6% |
| **Overall** | **82.9%** |

Evaluated with Claude Sonnet 4.5 as the answer agent and Haiku 4.5 as the judge, using the industry-standard LLM-as-a-Judge methodology. All 5 LOCOMO categories included.

Re-measured on 0.24.1 (same protocol, 199 questions): **84.4%** and **85.4%**
across two runs of the default `legacy` fusion. Note the run-to-run noise —
two runs of the *identical* configuration flipped 16 individual questions and
differed by 1.0 point, so treat sub-2-point differences on this benchmark as
inconclusive.

### Fusion modes

`MEMORY_FUSION=rrf` (reciprocal rank fusion) exists because summing raw cosine
with max-normalized BM25 adds incomparable scales. It fixes three measured
scoring artifacts — see the [0.22.0 notes](#0220) — but it did **not** improve
answers, so `legacy` remains the default:

| | `legacy` | `rrf` |
|---|---|---|
| LOCOMO overall | **84.4% / 85.4%** | 82.4% |
| Gold-evidence recall @20 (no LLM) | 58.2% | **66.3%** |
| Gold-evidence recall @5 (no LLM) | 46.9% | 46.9% |
| Gold-evidence MRR @20 (no LLM) | **0.354** | 0.346 |
| Top-1 score on *unanswerable* questions | 0.70 | 0.93 |
| Score spread across top 4 | ~0.17 | ~0.01 |

The retrieval-only numbers are deterministic and show the real trade: `rrf`
surfaces considerably more gold evidence inside the top 20 (+8.1 points
recall) but ranks it slightly lower (−0.008 MRR). Because the answer agent
already reads the top 20, the extra recall didn't convert into better answers,
and an independent A/B on a 38-fact corpus lost two answers outright to
top-rank precision.

Two further costs, both measured:

- **Scores stop discriminating.** `rrf` inflates absolute scores (top-1 rises
  0.70 → 0.93) and compresses their spread to ~0.01 across the top 4, versus
  ~0.17 under `legacy`, because rank 1 contributes 1.0 per channel however
  weak the match is. Exact ties between adjacent results are normal. In
  practice you cannot threshold on an `rrf` score, and ordering inside the
  band is decided by the tiebreak rather than by relevance.
- **No absolute quality signal.** Neither mode separates answerable from
  unanswerable questions by score, so this isn't a lost refusal signal — but
  an `rrf` score carries no information about how good the match actually is.

Use `rrf` when you want maximum recall in a window you will read entirely and
stability under corpus edits. Keep `legacy` when you want scores that mean
something, which is why it is the default.

### Architecture advantages

- **Zero LLM calls in the server** — intelligence lives in the agent, not the memory layer
- **Local embeddings** — no API key needed (`bge-small-en-v1.5`, 384-dim)
- **Single embedded database** — no Docker, no PostgreSQL, no separate vector DB
- **Hybrid search** — Vector (HNSW) + Full-text (BM25) + Graph (PageRank + Louvain communities)
- **Minimal retrieval surface** — the benchmark agent above scored 82.9% using only `memory_search`, `memory_get` and a `calculator` for date arithmetic. Retrieval quality comes from the server, not from agent-side orchestration.

## Quick Start

```bash
# Run directly with uvx (no install needed)
uvx memnest-mcp
```

Or install and run:

```bash
pip install memnest-mcp
memnest-mcp
```

### Configure a project (Kiro)

From your project root, one command writes the workspace-level MCP config
(with the memory scope pinned to the project) plus the recall/persist/dream
agent hooks:

```bash
memnest-mcp config kiro            # configure the current directory
memnest-mcp config kiro --check    # verify only
memnest-mcp config kiro --no-hooks # MCP server config only
```

This writes `.kiro/settings/mcp.json` (server + pinned workspace),
`.kiro/hooks/memnest-recall.json` and `memnest-persist.json` (automatic recall
and persistence), and `.kiro/steering/memnest-dream.md` — a manual steering
file you invoke with `/memnest-dream` to consolidate memory.

The config is always workspace-level (`<project>/.kiro/`), so each project
gets its own correctly-scoped memory database at `<project>/.memnest/`.
Reconnect MCP servers in Kiro afterwards. The Kiro Power (below) remains
optional on top for keyword activation and skills.

## MCP Configuration

Add to your MCP client config (Kiro, Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "memnest": {
      "command": "uvx",
      "args": ["memnest-mcp@latest"],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

That's it — zero config required. All settings have sensible defaults.

## Tools

| Tool | What it does |
|------|-------------|
| `memory_store` | Store a memory (single or batch) with auto-dedup, auto-link to Topic nodes |
| `memory_search` | Hybrid semantic + keyword search, ranked by relevance |
| `memory_update` | Update content, importance, or tags (single or batch) |
| `memory_delete` | Delete one or more memories and their relationships |
| `memory_get` | Read one memory in full — untruncated content plus its edges |
| `memory_list` | Enumerate memories by recency / category / topic / importance (no ranking, pages to any depth) |
| `memory_relate` | Create RELATED_TO / SUPERSEDES / EXPLAINS relationships (single or batch, idempotent) |
| `memory_unrelate` | Remove a relationship — one type or all types between a pair |
| `memory_query` | Run any Cypher query — traversals, writes, extension calls (INSTALL/LOAD), table scans |
| `memory_schema` | Inspect live DB schema: tables, columns, indexes, extensions |
| `memory_topics` | List all topics (tags) with memory counts |
| `memory_stats` | Database statistics: counts, categories, topics, top memories, runtime health |
| `memory_dream` | Periodic consolidation — auto-prune stale, auto-merge trivial duplicates, surface clusters for review |
| `memory_reindex` | Rebuild both search indexes (vector HNSW and full-text BM25) |
| `memory_export` | Write all memories and edges to a portable JSON file |
| `memory_import` | Restore an export — ids remapped, edges rewired, dedup applied |
| `memory_set_workspace` | Pin the workspace scope and database location |
| `memory_graph_html` | Generate an interactive HTML visualization of the graph |
| `memory_traverse` | *Deprecated* — use `memory_query(read_only=True)` |

## Graph Data Model

```
(:Memory)  — content, embedding, category, tags, importance, access_count, timestamps
(:Topic)   — auto-created from tags

(:Memory)-[:ABOUT]->(:Topic)          # memory is about a topic
(:Memory)-[:RELATED_TO]->(:Memory)    # memories are related
(:Memory)-[:SUPERSEDES]->(:Memory)    # newer memory replaces older
```

### Example: Store and Search

```python
# Store a memory (via MCP tool call)
memory_store(
    content="User prefers Python over Node.js for backend tools",
    category="preference",
    tags=["python", "nodejs", "backend"],
    importance=4
)

# Search memories
memory_search(query="what language does the user prefer")

# Traverse the graph
memory_query(
    cypher_query="MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: 'python'}) RETURN m.content"
)
```

### Example: Graph Relationships

```python
# Link related memories
memory_relate(from_id=5, to_id=3, relationship="RELATED_TO")

# Mark a decision as superseded
memory_relate(from_id=8, to_id=2, relationship="SUPERSEDES")

# Find all memories about a topic
memory_query(
    cypher_query="MATCH (m:Memory)-[:ABOUT]->(t:Topic) RETURN t.name, COUNT(m) ORDER BY COUNT(m) DESC"
)
```

## Three-Layer Deduplication

Every `memory_store` call runs through three dedup layers:

1. **Exact hash** — SHA256 of normalized content. Identical content is rejected, importance bumped.
2. **Semantic similarity** — If cosine similarity > 0.92 with an existing memory, merges into it (keeps longer content, merges tags, bumps importance).
3. **Consolidation** — Periodic via `memory_dream`. Auto-prunes stale low-importance memories, auto-merges trivial duplicates (similarity ≥ 0.95), surfaces clusters for LLM-driven review.

## Categories

| Category | Use for |
|----------|---------|
| `learning` | Technical knowledge, facts, how things work |
| `preference` | User preferences and choices |
| `decision` | Architecture decisions, tool choices |
| `pattern` | Recurring workflows, conventions |
| `general` | Everything else (default) |

## Configuration

All settings are optional — defaults work out of the box.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MEMORY_DB_PATH` | `.memnest/memory.lbug` (in cwd) | LadybugDB database path. Use `:memory:` for ephemeral testing |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic similarity threshold for auto-dedup |
| `MEMORY_MERGE_TAG_OVERLAP` | `0.5` | Minimum tag Jaccard overlap before two similar memories may merge |
| `MEMORY_MERGE_VALUE_GATE` | `1` | Refuse to merge near-identical memories whose values disagree (`500ms` vs `900ms`). Set `0` to restore pure-similarity merging (unsafe) |
| `MEMORY_CONFLICT_THRESHOLD` | `0.85` | Similarity at which two results are flagged `near_duplicate` |
| `MEMORY_CONFLICT_VALUE_FLOOR` | `0.5` | Similarity floor for `value_disagreement` flagging — same subject, different value, however differently worded |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model for embeddings |
| `MEMORY_EMBEDDING_DIM` | `384` | Embedding dimension (must match model) |
| `MEMORY_WORKSPACE` | `cwd` | Workspace identifier for memory namespacing |
| `MEMORY_RESPONSE_FORMAT` | `toon` if installed, else `json` | Response serialization. `toon` is more token-efficient for LLM context |
| `MEMORY_SEARCH_LIMIT` | `10` | Max results from `memory_search` |
| `MEMORY_LIST_LIMIT` | `20` | Default page size for `memory_list` |
| `MEMORY_MAX_CONTENT` | `500` | Content truncation length in search/list results |
| `MEMORY_LATENCY_WARN_MS` | `200` | Log a warning when an op exceeds this (ms) |
| `MEMORY_DREAM_MIN_OPS` | `10` | Min ops since last dream before next runs |
| `MEMORY_DREAM_MIN_HOURS` | `24` | Min hours since last dream before next runs |
| `MEMORY_DREAM_MIN_MEMORIES` | `20` | Min total memories before dream is allowed (skipped otherwise) |
| `MEMORY_DREAM_PRUNE_DAYS` | `30` | Auto-prune memories older than N days (with low importance) |
| `MEMORY_DREAM_PRUNE_MAX_IMP` | `2` | Auto-prune only memories at or below this importance |
| `MEMORY_DREAM_TRIVIAL_THRESHOLD` | `0.95` | Cosine similarity ≥ this is auto-merged in dream |
| `MEMORY_DREAM_CLUSTER_LOW` | `0.88` | Cluster-review window: `[low, trivial)` is surfaced for agent review |
| `MEMORY_CONSOLIDATE_CLUSTERS` | `10` | Max clusters returned per `memory_dream` run |
| `MEMORY_CONSOLIDATE_SCAN` | `1000` | Memories examined per dream run — a rotating **window**, not a horizon. Above this size the window advances each run, so the whole corpus is covered over `ceil(corpus / window)` runs at unchanged per-run cost. `memory_dream` reports `scan_coverage` |
| `MEMORY_ALLOW_DESTRUCTIVE` | `false` | Allow DELETE/DROP/TRUNCATE/REMOVE/SET/COPY through `memory_query`. **Off by default for safety.** Prefer `memory_update`, `memory_delete`, `memory_unrelate` |
| `MEMORY_SEARCH_CANDIDATES` | `100` | Rows each search channel retrieves before fusion. Independent of `top_k`. Does not affect index-health coverage: above this size the census switches to a dedicated id-only probe |
| `MEMORY_FUSION` | `legacy` | Channel fusion: `legacy` (raw cosine + max-normalized FTS), `normalized` (min-max vector), or `rrf` (reciprocal rank fusion — only each channel's *ordering* enters the score, so channel scales can't interact and scores stay stable when memories are added or deleted). `rrf` stays opt-in: it measured **below** `legacy` on LOCOMO (see [Fusion modes](#fusion-modes)) |
| `MEMORY_RRF_K` | `60` | Rank-decay constant for `rrf` mode. Channel value is `(K+1)/(K+rank)`: 1.0 at rank 1, ~0.87 at rank 10 |
| `MEMORY_MAX_STORE_CHARS` | `20000` | Content longer than this is truncated on store |
| `MEMORY_MAX_BATCH` | `500` | Max items per batch call |
| `MEMORY_GRAPH_MAX_NODES` | `2000` | Max nodes `memory_graph_html` will render before refusing |
| `MEMORY_EMBED_TIMEOUT_S` | `30` | Soft timeout for embedding model load (warm-up only) |

### In-Memory Mode (Testing)

```json
"env": { "MEMORY_DB_PATH": ":memory:" }
```

All data is ephemeral — lost on restart. Useful for testing.

## Kiro Power

This repo includes a ready-to-use [Kiro Power](./power/memnest/) in the `power/memnest/` directory, packaged in the [Agent Plugins](https://agent-plugins.org/) v1.0.0 format with:

- Plugin manifest with activation keywords (`power/memnest/plugin.json`)
- Pre-configured MCP server (`power/memnest/mcp.json`)
- Two Kiro agent hooks for automatic recall and persistence (`power/memnest/dev.kiro/hooks/`, v1 hook schema — IDE/CLI only; on Kiro Web the agent follows the same workflow from the getting-started skill)
  - **memnest-recall** (`UserPromptSubmit`) — searches memory before responding to each prompt
  - **memnest-persist** (`Stop`) — stores important info when the agent finishes
  - consolidation runs on demand via `memory_dream` (see the getting-started skill)
- Agent Skills with the setup guide and Cypher query examples (`power/memnest/skills/`)

**Install in Kiro:** Add Custom Power → `https://github.com/arunkumars-mf/memnest-mcp/tree/main/power/memnest`

## Architecture

```
AI Agent (Kiro, Claude, etc.)
    │
    ├─ memory_store ──→ embed content → dedup check → insert node → link topics
    ├─ memory_search ─→ embed query → HNSW vector search → tag boost → rank
    ├─ memory_query ──→ execute Cypher → return graph results
    │
    └─ LadybugDB (embedded, single directory)
        ├─ Memory nodes (content + FLOAT[384] embeddings)
        ├─ Topic nodes (auto-linked from tags)
        ├─ HNSW vector index (cosine similarity)
        └─ Graph relationships (ABOUT, RELATED_TO, SUPERSEDES, EXPLAINS)
```

## Requirements

- Python 3.10+
- Dependencies installed automatically: `real-ladybug`, `fastembed`, `mcp`
- ~130MB disk for the embedding model (downloaded on first run)

## TOON Format (Optional)

Memnest supports [TOON](https://toonformat.dev/) (Token-Oriented Object Notation) as a response format, reducing token usage by 30–60% compared to JSON. This is useful when memory results are fed back into LLM context.

TOON is **optional** — the server falls back to compact JSON automatically if the package isn't installed. To enable it:

```bash
pip install "memnest-mcp[toon]"
```

Or with uvx (requires the `--prerelease=allow` flag since `toon-format` is currently in beta):

```bash
uvx --prerelease=allow --with "toon-format==0.9.0b1" memnest-mcp@latest
```

To switch formats at runtime, set the environment variable:

```bash
MEMORY_RESPONSE_FORMAT=toon   # compact, token-efficient (default when installed)
MEMORY_RESPONSE_FORMAT=json   # standard JSON (default when toon is not installed)
```

The official Python implementation of TOON is [toon-format/toon-python](https://github.com/toon-format/toon-python), currently at v0.9.0-beta.1. Once it reaches a stable 1.0 release, the `--prerelease=allow` flag will no longer be necessary.

## Contributing

Issues and PRs welcome. See [LICENSE](LICENSE) for terms.

## License

[MIT](LICENSE)

## Changelog

### 0.27.0

- **Result ordering no longer depends on the order memories were stored.** Two fixes to the same defect class: the `rrf` rank transform broke channel-value ties by memory id (and ids encode insertion order, so identical BM25 scores produced arbitrary ranks that propagated into different fused scores), and the final sort broke score ties by dict order. Channel ranks now use competition ranking — equal values get equal rank — and final ties break on importance, then recency, then id. A 4-memory fixture that reordered its own results purely by store order now doesn't.
- This was costing `rrf` measurable quality: gold-evidence recall@20 rises 64.3% → 66.3% and recall@5 45.4% → 46.9% (now equal to `legacy`). `legacy` is unaffected — it does no rank transform, and anchors are bit-identical.
- Documents the `rrf` score-compression cost: spread across the top 4 is ~0.01 versus ~0.17 under `legacy`, so `rrf` scores cannot be thresholded.

### 0.26.2

- Verifies a real restore into a separate on-disk database, not just an in-memory one: embeddings recomputed, index fully reachable, restored memories findable, supersession still resolving to current.

### 0.26.0

Closes two silent coverage losses that appeared at ordinary corpus sizes, not extreme ones.

- **Index-health coverage no longer lapses above the candidate pool.** The per-query census compared ranked vector hits against the corpus, which only works while the pool (100) covers it — so the detector for the worst bug class protected a shrinking slice as a workspace grew (2% at 5,000 memories). Above the pool it now runs a dedicated id-only probe at `k=corpus`: measured 19.7 ms against a 191.8 ms search, ~10% overhead, 100% coverage at every size. `explain_meta.census_mode` reports which path ran.
- **Dream's scan cap is a rotating window, not a horizon.** It always examined the newest `MEMORY_CONSOLIDATE_SCAN` by `updated_at`, so once a workspace passed the cap everything older was never considered for merge or prune again. The window now advances each run and wraps, covering any corpus over successive runs at unchanged per-run cost. `memory_dream` reports `scan_coverage`.
- **`memory_stats` reports `db_scope`** — whether the one-database-per-workspace invariant that makes lock-free operation safe actually holds, judged from which workspaces own memories in the file rather than from configuration. Warns when a shared `MEMORY_DB_PATH` has put unrelated projects in one graph.
- Soak harness gains `--sessions N`: N short sessions as separate processes, each verifying what the previous one left behind. This is the realistic stress pattern for one connection per workspace, and it is where the delete-churn bug actually manifested.

### 0.25.0

- Dream review clusters now carry only pairs that still need a decision. Different-subject and disjoint-scope pairs are separate permanently and are dropped; same-subject value conflicts are unresolved, so they stay and are labelled `gate: "value_conflict"` with the non-destructive resolution named.

### 0.24.1

- `memory_delete` reaps orphaned `Topic` nodes (a long-lived database had accumulated 273 orphans against 24 live topics); `memory_dream` does the same for its own prune/merge deletions and reports `topics_reaped`.
- The post-delete index census is exception-isolated — diagnostics can never fail a committed write.

### 0.24.0

Root-caused the recurring silent loss of vector-search coverage: LadybugDB 0.15.3 delete maintenance progressively orphans *surviving* HNSW nodes when transient batches are inserted then deleted. Reproduced standalone ([repro](./docs/upstream/ladybug-hnsw-delete-churn-unreachable.md)), monotonic, in-process, persists across restarts. Onset is non-monotonic in burst size and seed-dependent, which is why several earlier experiments wrongly cleared it.

- `memory_delete` now censuses index reachability and rebuilds on shortfall, so a session's deletes can't hand the next session a degraded index.
- `memory_stats` reports `vector_index.status: "degraded"` when the census contradicts the cached probe verdict.

### 0.23.0

- One shared implementation of the delete+recreate path used by store-dedup, update and dream merge (three copies had drifted; dream was zeroing `access_count`). Store and update preserve the count; dream merge sums the merged members'.

### 0.22.0

- New `MEMORY_FUSION=rrf` (opt-in). Fixes three artifacts of summing incomparable channel scales: survivor keyword scores rescaling 2.24× when the top hit was deleted, a +0.166 score jump that inverted a ranking, and a plateau of identical scores when the vector channel was dead. See [Fusion modes](#fusion-modes) for why it is not the default.

### 0.19.0

Surface hardening from a full tool-by-tool review.

- **Security**: the `memory_query` destructive-query guard was bypassable. It matched the substring `"DELETE "` — with a literal trailing space — so `MATCH (m:Memory)\nDETACH\nDELETE\nm;` reported success and deleted every memory with `MEMORY_ALLOW_DESTRUCTIVE=false`. Queries are now classified after stripping comments and string literals, matching keywords on word boundaries.
- **Breaking**: `read_only=True` now rejects *any* mutation. It previously permitted `CREATE`/`MERGE`/`SET`, so an overwrite succeeded under a flag named read-only. `MEMORY_ALLOW_DESTRUCTIVE` now also covers `SET`, `REMOVE` and `COPY` — an overwrite destroys the previous value as surely as a delete.
- **New `memory_unrelate`**: edges could be created but never removed, and `memory_query`'s DELETE is blocked by default, so a mistaken `SUPERSEDES` was permanent. This is also the supported way to break a circular `SUPERSEDES` chain that `memory_dream` reports.
- **`memory_get` now returns edges** (`include_edges=True` by default) plus `superseded` / `superseded_by`. Answering "what does this replace?" no longer requires Cypher.
- **Ranking fix**: the search candidate pool was `top_k * 3`, so the page size decided which memories were scored at all — on a 25-memory corpus, `top_k=10` surfaced two memories that outranked every result `top_k=5` returned. The pool is now fixed (`MEMORY_SEARCH_CANDIDATES`, default 100) and independent of `top_k`. Scores are unchanged; only coverage improves.
- **Pagination**: `memory_search(offset=...)` with `offset` / `has_more` in the response. Rank 11+ was previously unreachable.
- **New `memory_export` / `memory_import`**: JSON backup and restore including edges, with id remapping so an import can merge into an existing database. `memory_relate` is now idempotent (`status: "exists"`), so re-importing no longer doubles every edge.
- Input validation across every tool: two-sided clamping (`preview_chars=-5` used to slice content from the wrong end; `top_k=0` returned a `degraded` flag blaming the embedding model), content and batch size caps, and honest statuses (`memory_delete` reported `deleted` when every id was missing; `memory_list(min_importance='high')` raised a raw `ValueError`).
- `memory_get` and `memory_list` are no longer labelled compatibility aliases — each does something no other tool does. `memory_traverse` is marked deprecated.

### 0.3.0

- Default database is now per-workspace: `.memnest/memory.lbug` in the current directory. No more cross-workspace lock conflicts.
- Set `MEMORY_DB_PATH` to use a custom location (e.g. `~/.memnest/memory.lbug` for global shared memory).
- Hybrid search: Vector (HNSW) + Full-text (BM25) + Graph scoring with PageRank, Louvain community detection, and K-Core decomposition.
- LOCOMO benchmark: 82.9% overall score.

### 0.2.0

Compatibility-preserving redesign with improved safety defaults.

- New tools: `memory_query` (general Cypher), `memory_schema`, `memory_topics`, `memory_dream`, `memory_graph_html`. Batch mode added to `memory_store`, `memory_update`, `memory_relate`, `memory_delete`.
- **Breaking**: `MEMORY_ALLOW_DESTRUCTIVE` now defaults to `false`. Set it to `true` if you previously relied on `memory_query` deleting nodes.
- **Breaking**: tag storage migrated from comma-joined strings to JSON arrays. Old rows are still readable; rewriting (e.g. via `memory_update`) upgrades them to JSON.
- `memory_get`, `memory_list`, `memory_traverse` from 0.1.x are retained as compatibility aliases. (As of 0.19.0 `memory_get` and `memory_list` are first-class again; only `memory_traverse` remains deprecated.)
- TOON serialization is now the default response format when `toon-format` is installed; set `MEMORY_RESPONSE_FORMAT=json` to opt out.
- `memory_relate` validates that both endpoints exist before returning `created` (used to silently no-op on typo'd IDs).
- `memory_graph_html` is now XSS-safe (HTML-escaped tooltips, DOM `textContent` for the detail panel), refuses to render >`MEMORY_GRAPH_MAX_NODES`, and rotates snapshots.
- Workspace filter pushed inside the vector index `WITH` clause so search recall isn't starved across workspaces.
- Dream consolidation: dedupes parallel edges across merges, isolates clusters by workspace, persists state via atomic sidecar JSON.
