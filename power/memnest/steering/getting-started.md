---
inclusion: auto
---

# Memnest Memory — Getting Started

## Overview

You have access to persistent long-term memory via the Memnest MCP server. Memory persists across sessions and is scoped to the current workspace by default. The system uses hybrid search (vector + full-text + graph algorithms) scoring 82.9% on the LOCOMO benchmark.

## Core Workflow

1. **Recall** — Before responding, search memory for relevant past context
2. **Respond** — Use recalled context naturally in your response
3. **Persist** — After the conversation, store new important information
4. **Consolidate** — Periodically run `memory_dream` to prune, merge, and recompute graph scores

## Tool Quick Reference

### memory_search (primary retrieval tool)

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

**Tips:**
- Use natural language queries — vector component handles semantic matching
- Add `tags=["python", "backend"]` to disambiguate overloaded terms (e.g. "workspace" could mean Brazil, Kiro, or ATX)
- Set `global_search=True` to search across all workspaces
- Adjust `preview_chars` (default 200) if you need more context per result
- `top_k` caps at 10; use 5 for focused retrieval, 10 when exploring

### memory_store (persist new information)

Single mode:
```
memory_store(
    content="User prefers Python for backend, TypeScript for frontend",
    category="preference",
    tags=["python", "typescript", "tech-stack"],
    importance=4
)
```

**Batch mode** (faster — single embedding call for all items):
```
memory_store(items=[
    {"content": "CDK uses NpmPrettyMuch for deps", "category": "learning", "tags": ["cdk", "npm", "brazil"], "importance": 3},
    {"content": "User prefers Sonnet for complex tasks, Haiku for fast ones", "category": "preference", "tags": ["llm", "model-selection"], "importance": 4},
    {"content": "Auth module lives in src/auth/, uses Cognito", "category": "learning", "tags": ["auth", "cognito", "architecture"], "importance": 3}
])
```

**Categories:**
- `learning` — Facts, how things work, technical knowledge
- `preference` — User choices and style preferences
- `decision` — Architecture decisions with rationale (high importance)
- `pattern` — Recurring workflows, conventions
- `general` — Everything else (default)

**Importance:** 1=trivial, 2=low, 3=neutral (default), 4=important, 5=critical

### memory_relate (link memories in the graph)

```
memory_relate(from_id=10, to_id=5, relationship="RELATED_TO", confidence=0.9)
memory_relate(from_id=10, to_id=3, relationship="SUPERSEDES")  # 10 replaces 3
memory_relate(from_id=10, to_id=7, relationship="EXPLAINS")    # 10 explains 7
```

**Relationship types:**
- `RELATED_TO` — General association. Accepts `confidence` (0-1) and `provenance` (EXTRACTED|INFERRED|AMBIGUOUS)
- `SUPERSEDES` — Newer memory corrects/replaces older. Creates a correction chain.
- `EXPLAINS` — One memory explains/elaborates another

**Batch mode:**
```
memory_relate(relations=[
    {"from_id": 10, "to_id": 5, "relationship": "RELATED_TO", "confidence": 0.8},
    {"from_id": 10, "to_id": 3, "relationship": "SUPERSEDES"}
])
```

### memory_update (modify existing)

```
memory_update(memory_id=42, content="Updated preference", importance=5, tags=["new-tag"])
```

Batch mode:
```
memory_update(updates=[
    {"memory_id": 42, "importance": 5},
    {"memory_id": 43, "tags": ["deprecated"], "importance": 1}
])
```

### memory_dream (consolidation + graph recomputation)

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

### memory_get (full content by ID)

```
memory_get(memory_id=42)
```

Returns untruncated content + metadata. Use after search returns truncated previews.

### memory_topics (discover tags)

```
memory_topics(limit=20, min_count=2)
```

Lists all topics (tags) with memory counts. Use to discover available tag filters for search.

### memory_stats (health check)

```
memory_stats()
```

Returns total counts, category distribution, importance distribution, top topics, and graph metrics.

## What to Store

**Do store:**
- User preferences and choices (importance 4-5)
- Technical decisions with rationale (importance 4-5)
- Bug root causes and fixes (importance 3-4)
- Project architecture and conventions (importance 3-4)
- Package-specific gotchas and workflows (importance 3)
- Recurring patterns the user follows (importance 3)

**Don't store:**
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

## Workspace Scoping

Memories are scoped to the current workspace (project directory). Pass `global_search=True` to search across all workspaces. Cross-workspace memories still appear in graph traversals.

The workspace is auto-detected: explicit `MEMORY_WORKSPACE` env var, then the MCP client's workspace root (roots/list), then the server's cwd. The database lives at `<workspace>/.memnest/memory.lbug`.

**Verify scoping at the start of a session**: call `memory_stats()` and check the `workspace` field plus `runtime.workspace_source`. If `workspace` is `''` or doesn't match the current project (auto-detection can fail on clients that launch servers from `/` without roots support), pin it yourself:

```
memory_set_workspace(path="/absolute/path/to/current/project")
```

This re-homes the database to that project's `.memnest/` directory. Memories stored before the switch stay in the previous database file.
