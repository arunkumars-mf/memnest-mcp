---
name: "getting-started"
description: "Use Memnest persistent memory — recall context before responding, store learnings after tasks, verify workspace scoping. Use when working with agent memory, recalling past context, or storing preferences, decisions, and learnings."
license: "MIT"
compatibility: "Requires the memory-mcp server from this plugin (uvx, Python 3.10+). Automatic recall/persist hooks are Kiro IDE-only; other clients follow the workflow manually."
metadata:
  author: "arunkse"
  version: "1.0.0"
---

# Memnest Memory — Getting Started

## Overview

You have access to persistent long-term memory via the Memnest MCP server. Memory persists across sessions and is scoped to the current workspace by default. The system uses hybrid search (vector + full-text + graph algorithms) scoring 82.9% on the LOCOMO benchmark. Zero LLM calls in the server — all intelligence lives in the agent.

## Prerequisites Checklist

- [ ] Python 3.10+ available (`uvx` runs the server automatically)
- [ ] ~130MB disk for the embedding model (auto-downloaded on first run)
- [ ] No Docker, no external services required

## Core Workflow

1. **Recall** — Before responding, search memory for relevant past context
2. **Respond** — Use recalled context naturally in your response
3. **Persist** — After the conversation, store new important information
4. **Consolidate** — Periodically run `memory_dream` to prune, merge, and recompute graph scores

In the Kiro IDE, the bundled hooks (`dev.kiro/hooks/`) trigger recall and persist automatically. On clients without hook support (Kiro Web, CLI, other Agent Plugins hosts), follow this workflow manually — recall at the start of a task, persist at the end.

## Step-by-Step Guide

### Step 1: Verify workspace scoping (start of session)

Call `memory_stats()` and check the `workspace` field plus `runtime.workspace_source`. The workspace is auto-detected: explicit `MEMORY_WORKSPACE` env var, then the MCP client's workspace root (roots/list), then the server's cwd. The database lives at `<workspace>/.memnest/memory.lbug`.

If `workspace` is `''` or doesn't match the current project (auto-detection can fail on clients that launch servers from `/` without roots support), pin it yourself:

```
memory_set_workspace(path="/absolute/path/to/current/project")
```

This re-homes the database to that project's `.memnest/` directory. Memories stored before the switch stay in the previous database file.

### Step 2: Recall relevant context

`memory_search` is the primary retrieval tool:

```
memory_search(query="user's tech stack preferences", top_k=5, preview_chars=300)
memory_search(query="python backend", tags=["architecture", "backend"], top_k=5)
```

Hybrid scoring pipeline:
- **Vector** (40%) — HNSW cosine similarity over BGE-small embeddings
- **Full-text** (30%) — BM25 with English stemmer
- **Graph** (15%) — PageRank centrality + K-Core density + Louvain community expansion
- **Recency** (10%) — Exponential decay, 30-day half-life
- **Importance** (5%) — User-assigned 1-5 normalized

Tips:
- Use natural language queries — the vector component handles semantic matching
- Add `tags=["python", "backend"]` to disambiguate overloaded terms (e.g. "workspace" could mean Brazil, Kiro, or ATX)
- Set `global_search=True` to search across all workspaces
- Adjust `preview_chars` (default 200) if you need more context per result
- `top_k` caps at 10; use 5 for focused retrieval, 10 when exploring
- Use `memory_get(memory_id=42)` for full untruncated content after search returns previews
- Use `memory_topics(limit=20, min_count=2)` to discover existing tag filters

**If the response contains a `degraded` field**, semantic search is dead and
results are keyword-only (usually the embedding model failed to load). Tell
the user rather than silently accepting worse recall — check
`memory_stats().runtime.embeddings` for the cause.

### Step 3: Store new information

Single mode:

```
memory_store(
    content="User prefers Python for backend, TypeScript for frontend",
    category="preference",
    tags=["python", "typescript", "tech-stack"],
    importance=4
)
```

Batch mode (faster — single embedding call for all items):

```
memory_store(items=[
    {"content": "CDK uses NpmPrettyMuch for deps", "category": "learning", "tags": ["cdk", "npm", "brazil"], "importance": 3},
    {"content": "User prefers Sonnet for complex tasks, Haiku for fast ones", "category": "preference", "tags": ["llm", "model-selection"], "importance": 4},
    {"content": "Auth module lives in src/auth/, uses Cognito", "category": "learning", "tags": ["auth", "cognito", "architecture"], "importance": 3}
])
```

Categories:
- `learning` — Facts, how things work, technical knowledge
- `preference` — User choices and style preferences
- `decision` — Architecture decisions with rationale (high importance)
- `pattern` — Recurring workflows, conventions
- `general` — Everything else (default)

Importance: 1=trivial, 2=low, 3=neutral (default), 4=important, 5=critical

### Step 4: Link related memories (this is the highest-value step)

**Write edges, or you are using a graph database as a flat list.** Similarity
search alone cannot tell that fact B replaces fact A, so without a
`SUPERSEDES` edge an outdated decision keeps resurfacing next to its own
correction and the agent cannot tell which one is current. Edges are what
memnest offers over plain vector search.

Create an edge whenever one of these is true:
- You corrected or updated something already stored → `SUPERSEDES`
- A bug, incident, or gotcha stems from a stored decision → `RELATED_TO`
- One memory is the rationale for another → `EXPLAINS`

```
memory_relate(from_id=10, to_id=5, relationship="RELATED_TO", confidence=0.9)
memory_relate(from_id=10, to_id=3, relationship="SUPERSEDES")  # 10 replaces 3
memory_relate(from_id=10, to_id=7, relationship="EXPLAINS")    # 10 explains 7
```

Then retrieval can be deterministic instead of similarity guesswork — this
returns only the current version of a policy, ignoring superseded ones:

```
memory_query(cypher_query="MATCH (m:Memory) WHERE m.content CONTAINS 'retry policy' "
                          "AND NOT EXISTS { MATCH (x:Memory)-[:SUPERSEDES]->(m) } "
                          "RETURN m.id, m.content", read_only=True)
```

Relationship types:
- `RELATED_TO` — General association. Accepts `confidence` (0-1) and `provenance` (EXTRACTED|INFERRED|AMBIGUOUS)
- `SUPERSEDES` — Newer memory corrects/replaces older. Creates a correction chain.
- `EXPLAINS` — One memory explains/elaborates another

Batch mode:

```
memory_relate(relations=[
    {"from_id": 10, "to_id": 5, "relationship": "RELATED_TO", "confidence": 0.8},
    {"from_id": 10, "to_id": 3, "relationship": "SUPERSEDES"}
])
```

### Step 5: Update or delete

```
memory_update(memory_id=42, content="Updated preference", importance=5, tags=["new-tag"])
memory_update(updates=[{"memory_id": 42, "importance": 5}, {"memory_id": 43, "tags": ["deprecated"], "importance": 1}])
memory_delete(memory_id=42)
```

Updates preserve relationships across content changes.

### Step 6: Consolidate periodically

```
memory_dream(dry_run=True)   # Preview what would happen
memory_dream(force=True)     # Run immediately (bypasses cooldown)
```

What it does:
1. Recomputes graph algorithms (PageRank, Louvain communities, K-Core) — improves search ranking
2. Auto-prunes stale memories (>30 days old, importance ≤ 2)
3. Auto-merges trivial duplicates (similarity ≥ 0.95)
4. Surfaces clusters (similarity 0.88-0.95) for agent review
5. Detects SCC contradictions (circular SUPERSEDES chains)

Auto-triggers: 10+ operations AND 24h since last run. Use `force=True` to override.

## What to Store

Do store:
- User preferences and choices (importance 4-5)
- Technical decisions with rationale (importance 4-5)
- Bug root causes and fixes (importance 3-4)
- Project architecture and conventions (importance 3-4)
- Package-specific gotchas and workflows (importance 3)
- Recurring patterns the user follows (importance 3)

Don't store:
- Routine greetings and acknowledgments
- Ephemeral details (today's weather, current time)
- Information already in project files (code, configs)
- Content that's obviously duplicate (system auto-deduplicates, but be reasonable)

## Tags Best Practices

Tags become Topic nodes in the graph, enabling traversal and disambiguation:
- Use lowercase, specific terms: `["python", "fastapi", "error-handling"]`
- Include domain context: `["auth", "cognito", "backend"]`
- Add project identifiers when relevant: `["atx", "mainframe", "migration"]`
- Use `memory_topics()` to see existing tags before inventing new ones

## Deduplication (Automatic)

Three layers — you don't need to check manually:
1. **Exact hash** — SHA256 normalized content rejects identical duplicates
2. **Semantic** — Cosine similarity > 0.92 auto-merges (keeps longer content, merges tags)
3. **Dream** — Periodic consolidation handles near-duplicates at 0.95+

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DB_PATH` | `<workspace>/.memnest/memory.lbug` | Database file path. Workspace is auto-detected from the MCP client (roots/list), falling back to cwd, then `~/.memnest/`. Use `:memory:` for testing |
| `MEMORY_WORKSPACE` | Auto-detected | Scope memories per project. Auto-detected from the MCP client's workspace root; set explicitly to override |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic similarity threshold for dedup |
| `MEMORY_RESPONSE_FORMAT` | `toon` | Response format (`toon` or `json`) |
| `MEMORY_SEARCH_LIMIT` | `10` | Max search results |
| `MEMORY_FUSION` | `legacy` | Vector-channel scaling: `legacy` (raw cosine) or `normalized` (min-max, matching the FTS channel). `normalized` is more robust on very small memory sets |

## Troubleshooting

### Memory database is locked

LadybugDB allows a single read-write process per database file. Another memnest server (usually another IDE window on the same project) is holding it. Close the other session, or give this one its own database via `memory_set_workspace` / `MEMORY_DB_PATH`.

### Search returns results from other projects, or workspace is ''

Auto-detection failed (client launched the server from `/` without roots support). Check `memory_stats().runtime.workspace_source`, then pin with `memory_set_workspace(path=...)`.

### Search results include a `degraded` field

Semantic search is unavailable, so results are keyword-only and recall is
noticeably worse. Call `memory_stats()` and check `runtime.embeddings`: a
non-zero `missing` count means some memories were stored while the embedding
model was down (they are invisible to vector search — re-store them), and
`healthy: false` with `missing: 0` means the model is failing at query time.
Server logs record the cause at ERROR level.

### First call is slow

`uvx` downloads the package and the ~130MB embedding model on first run. Subsequent starts are fast.

## Best Practices

- Verify workspace scoping once at session start (Step 1) before storing anything
- Prefer batch mode for multi-item stores — single embedding call
- Record corrections with `SUPERSEDES` instead of deleting old memories — the chain preserves history
- For graph traversal, aggregation, or relationship-based filtering beyond search, see the [graph-queries skill](../graph-queries/SKILL.md)
