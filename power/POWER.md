# Memnest Memory

Persistent graph memory for AI agents — powered by LadybugDB with hybrid vector + full-text + graph search.

## Description

Memnest gives your AI agent long-term memory that persists across sessions, deduplicates automatically, and models knowledge as a graph with typed relationships. It scores **82.9%** on the LOCOMO benchmark (ACL 2024).

**Zero LLM calls in the server** — all intelligence lives in the agent via hooks.

## Keywords

memory, agent memory, persistent memory, graph memory, knowledge graph, vector search, mcp, recall, context

## MCP Servers

- **memnest** — Memnest Memory MCP Server

## Hooks

- **persist-memory** — Automatically stores important information from the conversation when the agent stops
- **recall-memory** — Searches memory for relevant context before responding to each prompt
- **auto-dream** — Triggers memory consolidation (prune stale, merge duplicates) on user command

## Steering Files

- `getting-started.md` — Setup guide, tool reference, and usage patterns
- `graph-queries.md` — Cypher query examples for graph traversal and analysis

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DB_PATH` | `~/.memnest/memory.lbug` | Database file path. Use `:memory:` for testing |
| `MEMORY_WORKSPACE` | Current directory | Scope memories per project |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic similarity threshold for dedup |
| `MEMORY_RESPONSE_FORMAT` | `toon` | Response format (`toon` or `json`) |
| `MEMORY_SEARCH_LIMIT` | `10` | Max search results |

### Requirements

- Python 3.10+
- ~130MB disk for embedding model (auto-downloaded on first run)
- No Docker, no external services

## How It Works

### Memory Lifecycle

1. **Store** — Agent stores observations, decisions, preferences via `memory_store`
2. **Recall** — Before responding, agent searches memory via `memory_search`
3. **Dream** — Periodically consolidate: prune stale, merge duplicates, detect contradictions

### Hybrid Search Pipeline

When you call `memory_search`, Memnest fuses four scoring signals:

- **Vector** (40%) — HNSW cosine similarity over BGE-small embeddings
- **Full-text** (30%) — BM25 with English stemmer
- **Graph** (15%) — PageRank centrality + Louvain community expansion
- **Recency** (10%) — Exponential decay with 30-day half-life
- **Importance** (5%) — User-assigned 1-5 score

### Graph Data Model

```
(:Memory) — content, embedding, category, tags, importance, timestamps
(:Topic)  — auto-created from tags

(:Memory)-[:ABOUT]->(:Topic)
(:Memory)-[:RELATED_TO]->(:Memory)
(:Memory)-[:SUPERSEDES]->(:Memory)
(:Memory)-[:EXPLAINS]->(:Memory)
```

### Three-Layer Deduplication

1. **Exact hash** — SHA256 of normalized content rejects duplicates
2. **Semantic** — Cosine similarity > 0.92 triggers merge
3. **Dream** — Periodic consolidation auto-merges at ≥ 0.95, surfaces clusters at 0.88-0.95

## Core Tools (Recommended for Agent Use)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store memories with auto-dedup and topic linking |
| `memory_search` | Hybrid semantic + keyword + graph search |
| `memory_update` | Update content, importance, or tags |
| `memory_delete` | Delete memories and their relationships |
| `memory_relate` | Create typed relationships between memories |
| `memory_dream` | Consolidation: prune, merge, surface clusters |

## Advanced Tools (Available but rarely needed)

| Tool | Purpose |
|------|---------|
| `memory_query` | Raw Cypher queries for complex graph traversal |
| `memory_schema` | Inspect database schema |
| `memory_topics` | List all topics with memory counts |
| `memory_stats` | Database statistics and health |
| `memory_graph_html` | Interactive HTML visualization |
| `memory_get` | Get full memory by ID |
| `memory_list` | List memories with filters |
| `memory_traverse` | Read-only Cypher alias |
