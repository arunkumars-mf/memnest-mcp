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

### Architecture advantages

- **Zero LLM calls in the server** — intelligence lives in the agent, not the memory layer
- **Local embeddings** — no API key needed (`bge-small-en-v1.5`, 384-dim)
- **Single embedded database** — no Docker, no PostgreSQL, no separate vector DB
- **Hybrid search** — Vector (HNSW) + Full-text (BM25) + Graph (PageRank + Louvain communities)
- **3 agent tools** — `memory_search`, `memory_get`, `calculator`. Simple interface, powerful retrieval.

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
| `memory_relate` | Create RELATED_TO / SUPERSEDES / EXPLAINS relationships (single or batch) |
| `memory_query` | Run any Cypher query — traversals, writes, extension calls (INSTALL/LOAD), table scans |
| `memory_schema` | Inspect live DB schema: tables, columns, indexes, extensions |
| `memory_topics` | List all topics (tags) with memory counts |
| `memory_stats` | Database statistics: counts, categories, topics, top memories |
| `memory_dream` | Periodic consolidation — auto-prune stale, auto-merge trivial duplicates, surface clusters for review |
| `memory_graph_html` | Generate an interactive HTML visualization of the graph |
| `memory_get` | (compat) Get full content of a memory by ID |
| `memory_list` | (compat) List memories filtered by recency, category, topic, or importance |
| `memory_traverse` | (compat) Read-only Cypher — alias for `memory_query(read_only=True)` |

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
| `MEMORY_DB_PATH` | `~/.memnest/memory.lbug` | LadybugDB database path. Use `:memory:` for ephemeral testing |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic similarity threshold for auto-dedup |
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
| `MEMORY_CONSOLIDATE_SCAN` | `1000` | Max memories scanned per dream phase |
| `MEMORY_ALLOW_DESTRUCTIVE` | `false` | Allow DELETE/DROP/TRUNCATE through `memory_query`. **Off by default for safety**. Opt in with `true` |
| `MEMORY_GRAPH_MAX_NODES` | `2000` | Max nodes `memory_graph_html` will render before refusing |
| `MEMORY_EMBED_TIMEOUT_S` | `30` | Soft timeout for embedding model load (warm-up only) |

### In-Memory Mode (Testing)

```json
"env": { "MEMORY_DB_PATH": ":memory:" }
```

All data is ephemeral — lost on restart. Useful for testing.

## Kiro Power

This server is also available as a [Kiro Power](https://github.com/arunkumars-mf/memnest-power) with:
- Pre-configured MCP server
- Three hooks for automatic memory persistence and recall
- Steering files with setup guide and Cypher query examples

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

## Contributing

Issues and PRs welcome. See [LICENSE](LICENSE) for terms.

## License

[MIT](LICENSE)

## Changelog

### 0.3.0

- Default database directory is `~/.memnest/`.
- Set `MEMORY_DB_PATH` to use a custom location.
- Hybrid search: Vector (HNSW) + Full-text (BM25) + Graph scoring with PageRank, Louvain community detection, and K-Core decomposition.
- LOCOMO benchmark: 82.9% overall score.

### 0.2.0

Compatibility-preserving redesign with improved safety defaults.

- New tools: `memory_query` (general Cypher), `memory_schema`, `memory_topics`, `memory_dream`, `memory_graph_html`. Batch mode added to `memory_store`, `memory_update`, `memory_relate`, `memory_delete`.
- **Breaking**: `MEMORY_ALLOW_DESTRUCTIVE` now defaults to `false`. Set it to `true` if you previously relied on `memory_query` deleting nodes.
- **Breaking**: tag storage migrated from comma-joined strings to JSON arrays. Old rows are still readable; rewriting (e.g. via `memory_update`) upgrades them to JSON.
- `memory_get`, `memory_list`, `memory_traverse` from 0.1.x are retained as compatibility aliases. They will be removed in 0.3.0.
- TOON serialization is now the default response format when `toon-format` is installed; set `MEMORY_RESPONSE_FORMAT=json` to opt out.
- `memory_relate` validates that both endpoints exist before returning `created` (used to silently no-op on typo'd IDs).
- `memory_graph_html` is now XSS-safe (HTML-escaped tooltips, DOM `textContent` for the detail panel), refuses to render >`MEMORY_GRAPH_MAX_NODES`, and rotates snapshots.
- Workspace filter pushed inside the vector index `WITH` clause so search recall isn't starved across workspaces.
- Dream consolidation: dedupes parallel edges across merges, isolates clusters by workspace, persists state via atomic sidecar JSON.
