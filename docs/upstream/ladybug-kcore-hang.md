# K_CORE_DECOMPOSITION hangs (unkillable) on a filtered projection

**Component:** `algo` extension, `K_CORE_DECOMPOSITION`
**Found with:** real_ladybug 0.15.3 (Python), macOS arm64
**Severity:** liveness — the call never returns and cannot be interrupted from
Python (`KeyboardInterrupt` does not fire; the process must be killed)

## Summary

`CALL K_CORE_DECOMPOSITION` on a **filtered** projected graph never returns
when the graph combines three properties:

1. a dense component (e.g. a K10 of single edges),
2. parallel edges (a second relationship row between endpoints that already
   share one), and
3. nodes **excluded** by the projection's node predicate that hold at least one
   edge among themselves.

Each ingredient alone completes in milliseconds — verified individually:
dense-only, parallels-only, exclusion-only, dense+parallels (unfiltered),
dense+filtered, sparse+exclusion. Only the three together hang.

`PAGE_RANK`, `LOUVAIN` and `STRONGLY_CONNECTED_COMPONENTS` complete on the
identical projection, so the projection itself is readable.

## Minimal reproduction

```python
import real_ladybug as lb

db = lb.Database(":memory:")
conn = lb.Connection(db)
conn.execute("INSTALL algo; LOAD EXTENSION algo;")
conn.execute("CREATE NODE TABLE Memory(id INT64 PRIMARY KEY, workspace STRING);")
conn.execute("CREATE REL TABLE RELATED_TO(FROM Memory TO Memory, provenance STRING);")

# 4 excluded nodes with one edge among themselves
for i in range(1, 5):
    conn.execute(f"CREATE (:Memory {{id: {i}, workspace: '/ws/alpha'}});")
conn.execute("MATCH (a:Memory {id:1}), (b:Memory {id:2}) "
             "CREATE (a)-[:RELATED_TO {provenance:'EXTRACTED'}]->(b);")

# 10 included nodes: complete graph K10, plus 7 parallel hub edges
for i in range(5, 15):
    conn.execute(f"CREATE (:Memory {{id: {i}, workspace: '/ws/beta'}});")
for i in range(5, 15):
    for j in range(i + 1, 15):
        conn.execute(f"MATCH (a:Memory {{id:{i}}}), (b:Memory {{id:{j}}}) "
                     f"CREATE (a)-[:RELATED_TO {{provenance:'INFERRED'}}]->(b);")
for i in range(6, 13):
    conn.execute(f"MATCH (a:Memory {{id:5}}), (b:Memory {{id:{i}}}) "
                 f"CREATE (a)-[:RELATED_TO {{provenance:'EXTRACTED'}}]->(b);")

conn.execute("CALL PROJECT_GRAPH('k', "
             "{'Memory': \"n.workspace = '/ws/beta'\"}, {'RELATED_TO': 'true'});")

# Never returns; CPU pinned; SIGINT ignored.
conn.execute("CALL K_CORE_DECOMPOSITION('k') RETURN node.id, k_degree;")
```

## Expected

10 rows (the K10 members, each with core number 9 — or degree-counting
parallels, 5 with a higher value), in milliseconds like the unfiltered case.

## Additional observation (correctness, not liveness)

When `K_CORE_DECOMPOSITION` completes on an *unfiltered* graph containing
parallel edges, it counts each parallel row toward degree — two RELATED_TO rows
between the same pair yield degree 2 for a node with one distinct neighbour.
Whether that is intended is unclear, but combined with the hang it suggests the
peeling loop's bookkeeping misbehaves when the effective degree and the row
count diverge under a filter.
