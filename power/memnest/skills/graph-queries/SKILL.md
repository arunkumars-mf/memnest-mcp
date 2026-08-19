---
name: "graph-queries"
description: "Run Cypher graph queries against Memnest memory — traversals, correction chains, PageRank/Louvain analytics, workspace-scoped queries. Use when memory_search is not enough and you need graph traversal, aggregation, or relationship-based filtering."
license: "MIT"
metadata:
  author: "arunkse"
  version: "1.0.0"
---

# Memnest — Graph Queries

## Overview

Memnest stores memories as a graph with typed relationships. Use `memory_query` for complex traversals that go beyond what `memory_search` provides. For most use cases, `memory_search` is sufficient — reach for Cypher only when you need graph traversal, aggregation, or relationship-based filtering.

## Prerequisites Checklist

- [ ] Call `memory_schema()` first to get live table/column names
- [ ] Use `read_only=True` for exploration (rejects DELETE/DROP/TRUNCATE)
- [ ] Never SET `m.embedding` via Cypher (fails silently due to the HNSW vector index) — use `memory_update` for content changes

## Graph Data Model

```
(:Memory) — content, embedding, category, tags, importance, timestamps
(:Topic)  — auto-created from tags

(:Memory)-[:ABOUT]->(:Topic)
(:Memory)-[:RELATED_TO]->(:Memory)
(:Memory)-[:SUPERSEDES]->(:Memory)
(:Memory)-[:EXPLAINS]->(:Memory)
```

## Common Queries

### Find memories about a specific topic

```cypher
MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: 'python'})
RETURN m.id, m.content, m.importance
ORDER BY m.importance DESC;
```

### Find all topics with memory counts

```cypher
MATCH (m:Memory)-[:ABOUT]->(t:Topic)
RETURN t.name, COUNT(m) AS count
ORDER BY count DESC
LIMIT 20;
```

### Find related memories (1 hop)

```cypher
MATCH (m:Memory {id: 42})-[:RELATED_TO]-(other:Memory)
RETURN other.id, other.content, other.importance;
```

### Find memories that supersede others (correction chain)

```cypher
MATCH path = (newer:Memory)-[:SUPERSEDES*]->(older:Memory)
WHERE newer.id = 10
RETURN [n IN nodes(path) | n.content];
```

### Find the latest version of a memory (no newer supersedes it)

```cypher
MATCH (m:Memory)
WHERE NOT EXISTS { MATCH (newer:Memory)-[:SUPERSEDES]->(m) }
RETURN m.id, m.content
ORDER BY m.updated_at DESC
LIMIT 10;
```

### Find memories connected through shared topics

```cypher
MATCH (m1:Memory {id: 5})-[:ABOUT]->(t:Topic)<-[:ABOUT]-(m2:Memory)
WHERE m1 <> m2
RETURN m2.id, m2.content, t.name AS shared_topic;
```

### Find explanation chains

```cypher
MATCH (explainer:Memory)-[r:EXPLAINS]->(explained:Memory)
RETURN explainer.content, r.rationale_type, explained.content
ORDER BY explained.updated_at DESC;
```

## Graph Analytics

### Most connected memories (hub nodes)

```cypher
MATCH (m:Memory)-[r]-(other)
RETURN m.id, m.content, COUNT(r) AS connections
ORDER BY connections DESC
LIMIT 10;
```

### Memories with highest PageRank (graph centrality)

```cypher
MATCH (m:Memory)
WHERE m.pagerank IS NOT NULL
RETURN m.id, m.content, m.pagerank
ORDER BY m.pagerank DESC
LIMIT 10;
```

### Memories in the same Louvain community

```cypher
MATCH (m:Memory {id: 42})
WITH m.community_id AS cid
MATCH (other:Memory {community_id: cid})
RETURN other.id, other.content
ORDER BY other.pagerank DESC;
```

### High K-Core memories (densely connected knowledge)

```cypher
MATCH (m:Memory)
WHERE m.k_degree IS NOT NULL AND m.k_degree >= 3
RETURN m.id, m.content, m.k_degree
ORDER BY m.k_degree DESC;
```

## Workspace-Scoped Queries

### All memories in current workspace

```cypher
MATCH (m:Memory {workspace: '/path/to/project'})
RETURN m.id, m.content, m.category
ORDER BY m.updated_at DESC;
```

### Cross-workspace topic analysis

```cypher
MATCH (m:Memory)-[:ABOUT]->(t:Topic)
RETURN m.workspace, t.name, COUNT(*) AS count
ORDER BY count DESC;
```

## Write Queries

### Create a relationship

```cypher
MATCH (a:Memory {id: 5}), (b:Memory {id: 10})
CREATE (a)-[:RELATED_TO {provenance: 'EXTRACTED', confidence: 0.9}]->(b);
```

### Mark a memory as superseded

```cypher
MATCH (newer:Memory {id: 15}), (older:Memory {id: 3})
CREATE (newer)-[:SUPERSEDES]->(older);
```

**Note:** `MEMORY_ALLOW_DESTRUCTIVE=false` by default. DELETE/DROP/TRUNCATE queries are blocked unless explicitly enabled.

## Extension Calls

### Run PageRank

```cypher
CALL pagerank('Memory', 'RELATED_TO', {dampingFactor: 0.85, maxIterations: 100})
RETURN node.id, node.content, rank
ORDER BY rank DESC
LIMIT 10;
```

### Run Louvain community detection

```cypher
CALL community_detection('Memory', 'RELATED_TO')
RETURN node.id, community_id, node.content;
```

### K-Core decomposition

```cypher
CALL k_core('Memory', 'RELATED_TO')
RETURN node.id, node.content, core_number
ORDER BY core_number DESC;
```

## Troubleshooting

### Query rejected as destructive

`memory_traverse` and `read_only=True` always reject DELETE/DROP/TRUNCATE. Use `memory_query` without `read_only` for writes; destructive operations additionally require `MEMORY_ALLOW_DESTRUCTIVE=true`.

### SET on m.embedding has no effect

The HNSW vector index makes direct embedding writes fail. Use `memory_update` — it recreates the node with a fresh embedding while preserving relationships.

### pagerank/community_id/k_degree are NULL

Graph scores are computed by `memory_dream`. Run `memory_dream(force=True)` to populate them.
