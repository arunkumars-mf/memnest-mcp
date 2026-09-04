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

**You MUST read the `related` list before answering.** This is a requirement,
not an optimisation. Those are memories connected by graph edges to your top
hits (each names its anchor via `linked_to`), reported separately because
similarity ranking cannot surface them — the incident caused by a decision, the
rationale behind a convention, the version a correction replaced.

Treat `results` and `related` as one answer set. In benchmark runs on a
38-fact graph, the correct answer was frequently NOT the #1 result: it sat at
rank 2–3, or existed only in `related`. Examples that failed when `related` was
ignored, and succeeded when it was read:
- "what deprecated infrastructure does checkout transitively depend on?" — the
  deprecated component is two hops away and never ranks; it is only in `related`
- "which of these dependencies is still maintained?" — the lifecycle fact is a
  separate memory from the dependency fact, reachable only by edge
- "what breaks if this service shuts down?" — the mitigation project is linked,
  not similar

An agent that reads only `results` loses those questions outright. The ranking
answers "what matches?"; `related` answers "what else do you need to know?".
Both are required to answer well.

**Superseded results** carry `"superseded": true`. Never present those as
current — the memory that supersedes them ranks above them, or pass
`include_superseded=False` to drop them.

**If the response contains `potential_conflicts`**, two returned memories
disagree with no edge marking which is current — typically a fact learned in one
session and a conflicting one learned later. Each entry carries a `reason`:

- `near_duplicate` — they read almost identically (high similarity)
- `value_disagreement` — they are about the same subject and either one reads
  like a correction of the other, or both state a comparable quantity with
  different magnitudes (`30 days` vs `one year`). Worded differently enough that
  similarity alone would never surface them, and the stale one often ranks higher
  because the query's wording matches it better — so check this list before
  presenting the top hit as current.

Resolve it rather than picking one silently:
- If one replaces the other → re-store the current version with
  `memory_store(..., supersedes=<old_id>)`
- If both are true (different scopes, environments, time periods) →
  `memory_relate(from_id=<a>, to_id=<b>, relationship="RELATED_TO")`. **That
  dismisses the flag permanently** — it is the recorded answer to "I looked, and
  both hold", so the pair is never reported again. Do not use `SUPERSEDES` for
  this: it would demote a true fact out of results.
- If you cannot tell → ask the user; do not guess which is current

This is flagged as *potential* because the server does no LLM inference: it knows
the two are about the same subject and disagree on something, not which one is
right. Complementary facts that differ in kind rather than magnitude ("depends on
Redis for caching" / "depends on Kafka for event delivery", "port 8080 for HTTP" /
"port 9090 for metrics") are deliberately not flagged.

Two measurements of the *same* dimension under different qualifiers ("connect
timeout 500ms" / "read timeout 2000ms") will flag even though both are true —
telling a qualifier from a synonym needs semantics the server does not have. One
`memory_relate(..., relationship="RELATED_TO")` dismisses it for good, which is
why the detector errs toward flagging: a dismissal costs one call, a miss serves
a stale value as the answer.

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

Importance is yours to set — the server never raises it on its own. Re-storing a
fact you already know is a no-op (`already_exists`) and changes nothing, so
re-learning something across sessions cannot quietly promote it up the rankings.
On a merge, an omitted `importance` leaves the existing value alone and an
explicit one takes the higher of the two, so restating a fact is safe whether or
not you pass it. `updated_at`, which feeds the recency term, tracks when a
memory's *content* changed, not when it was last mentioned.

**Recording a corrected value?** Pass `supersedes=<old_id>` when you know the
new fact replaces an older one. That writes the `SUPERSEDES` edge and skips
dedup in a single call.

**If the result contains `potential_conflict_with`**, the store found a
near-identical memory whose *values* disagree (`500 milliseconds` vs
`900 milliseconds`, `Kafka` vs `Kinesis`, `prod-checkout` vs `prod-inventory`)
and deliberately kept both rather than merging one away. The status is
`stored_new`, not an error — nothing was lost. Resolve it now, while you have
the context:
- If the new fact replaces the old one → `memory_relate(from_id=<new>,
  to_id=<conflict_id>, relationship="SUPERSEDES")`
- If both are true in different scopes → say so in the content, then link them
- If you cannot tell → surface it to the user

Ignoring it is safe but leaves an unresolved pair that `potential_conflicts`
will keep reporting on every future search that touches it.

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

**Recording a correction: always use `memory_store(..., supersedes=<old_id>)`**
rather than storing then relating. A correction is textually near-identical to
what it corrects, so a plain store can be silently absorbed by semantic dedup
(measured: two consecutive retry-policy versions at 0.9284 similarity, above
the 0.92 threshold — the intermediate version vanished). Passing `supersedes`
disables dedup for that store and wires the edge in one call, so the chain
cannot be lost. Works per-item in batch mode too.

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

Creating an edge twice is a no-op (`status: "exists"`), so re-asserting a link
you are unsure about is safe.

**Wrote the wrong edge?** `memory_unrelate(from_id=..., to_id=...)` removes it.
Omit `relationship` to drop every edge between the pair, or name one to drop
just that type. The memories are untouched. This is also how you break a
circular `SUPERSEDES` chain if `memory_dream` reports one under
`contradictions`.

**Inspecting what a memory is connected to:** `memory_get(memory_id=42)` returns
an `edges` block with both directions, plus `superseded: true` and
`superseded_by` when a newer version exists. Reach for that before writing
Cypher.

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
5. Detects SCC contradictions (circular SUPERSEDES chains only — for
   unresolved semantic conflicts see `potential_conflicts` in search results)

Pairs joined by `SUPERSEDES` or `EXPLAINS` are never merged and never offered
for review — they are distinct versions by assertion, and `protected_by_edges`
in the response counts how many were left alone. Review clusters carry a
`resolution` field: a cluster may be a true duplicate to merge, competing
versions that need a `SUPERSEDES` edge, or distinct facts that merely read
alike. Decide per cluster; do not merge reflexively.

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
| `MEMORY_MERGE_TAG_OVERLAP` | `0.5` | Min tag Jaccard before two similar memories may merge. Blocks same-shape/different-subject merges |
| `MEMORY_MERGE_VALUE_GATE` | `1` | Refuse to merge near-identical memories whose values disagree. Set `0` to restore pure-similarity merging (unsafe: a corrected value can silently keep the stale one) |
| `MEMORY_CONFLICT_THRESHOLD` | `0.85` | Similarity at which two results are flagged as `near_duplicate` |
| `MEMORY_CONFLICT_VALUE_FLOOR` | `0.5` | Similarity floor for `value_disagreement` flagging (same subject, different value, however differently worded) |
| `MEMORY_SEARCH_CANDIDATES` | `100` | Rows each channel retrieves before fusion. Independent of `top_k`, so page size never decides which memories get scored |
| `MEMORY_ALLOW_DESTRUCTIVE` | `false` | Allow `memory_query` to run DELETE/DROP/TRUNCATE/REMOVE/SET/COPY. Leave off; use `memory_update`, `memory_delete`, `memory_unrelate` |
| `MEMORY_MAX_STORE_CHARS` | `20000` | Content longer than this is truncated on store |
| `MEMORY_MAX_BATCH` | `500` | Max items per batch call |
| `MEMORY_RESPONSE_FORMAT` | `toon` | Response format (`toon` or `json`) |
| `MEMORY_SEARCH_LIMIT` | `10` | Max search results |
| `MEMORY_FUSION` | `legacy` | Vector-channel scaling: `legacy` (raw cosine) or `normalized` (min-max, matching the FTS channel). `normalized` is more robust on very small memory sets |

## Backup and transfer

`memory_export()` writes every memory and edge to a JSON file (defaults to a
timestamped file next to the database). `memory_import(path=...)` restores it.

Ids are remapped rather than preserved, so an import can be merged into a
database that already has memories — edges are rewired onto the new ids, and
imported content goes through normal dedup, so re-importing the same file is a
no-op rather than a duplication. Use `dry_run=True` to see what a file would do.
`include_embeddings=True` makes the file much larger but avoids re-embedding on
restore; without it, content is re-embedded with the current model.

Worth doing before an upgrade, before `memory_set_workspace` (which leaves the
old database behind rather than moving it), and on any schedule you like.

## Troubleshooting

### A memory ranks far lower than it should

Use `memory_search(query="...", explain=True)`. Each result gains an `explain`
block with the raw channel values (`vector`, `fts`, `graph`, `recency`,
`importance_norm`), their weighted contributions, and `in_vector_window`. The
weighted values sum to the score, so the shortfall names the broken channel:

| Reading | Meaning | Fix |
|---------|---------|-----|
| `fts: 0.0` on a query whose words appear in the content | The keyword channel is not matching this memory — costs up to 0.30 of score | `memory_reindex()` (rebuilds FTS too) |
| `in_vector_window: false` | The memory is not among the k nearest for this query; it is scoring on keywords alone | `memory_reindex()`, then compare again |
| `vector` far below expectation but `in_vector_window: true` | Semantic similarity really is low — a phrasing mismatch, not a fault | Rely on `related`, or store an alias phrasing |

A single degraded channel is invisible in aggregate health fields: they report
per-index liveness, not per-memory scoring. `explain=True` is the only view that
attributes a score.

### Memory database is locked

LadybugDB allows a single read-write process per database file. Another memnest server (usually another IDE window on the same project) is holding it. Close the other session, or give this one its own database via `memory_set_workspace` / `MEMORY_DB_PATH`.

### Search returns results from other projects, or workspace is ''

Auto-detection failed (client launched the server from `/` without roots support). Check `memory_stats().runtime.workspace_source`, then pin with `memory_set_workspace(path=...)`.

### Search results include a `degraded` field

Semantic search is unavailable, so results are keyword-only and recall is
noticeably worse. Call `memory_stats()` and read `runtime.embeddings`:

| Reading | Meaning | Fix |
|---------|---------|-----|
| `missing` > 0 | Those memories were stored while the model was down and are invisible to semantic search | Re-store them |
| `stored_ok: true`, `index_returns_rows: false` | Vectors exist but the HNSW index does not return them (stale index) | `memory_reindex()` |
| `missing: 0`, `index_returns_rows: true` | Storage and index are fine; the model is failing at query time | Check server logs (ERROR level) |

A stale index is repaired automatically when the server reconnects, and
`memory_reindex()` forces it at any time. It only rebuilds the index — it
never modifies memories.

### First call is slow

`uvx` downloads the package and the ~130MB embedding model on first run. Subsequent starts are fast.

## Best Practices

- Verify workspace scoping once at session start (Step 1) before storing anything
- Prefer batch mode for multi-item stores — single embedding call
- Record corrections with `SUPERSEDES` instead of deleting old memories — the chain preserves history
- For graph traversal, aggregation, or relationship-based filtering beyond search, see the [graph-queries skill](../graph-queries/SKILL.md)
