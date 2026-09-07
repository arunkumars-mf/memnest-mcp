# LOUVAIN silently ignores node predicates in filtered projections

**Component:** `algo` extension, `LOUVAIN`
**Found with:** real_ladybug 0.15.3 (Python), macOS arm64
**Severity:** correctness — the filter appears to work but does not; callers
that trust it get results for nodes they explicitly excluded, with no error or
warning

## Summary

`CALL LOUVAIN` on a projected graph created with a node predicate returns
`louvain_id`s for **every node in the table**, not just those matching the
predicate. `PAGE_RANK` and `STRONGLY_CONNECTED_COMPONENTS` honour the same
predicate on the same projection.

This is arguably more dangerous than an error: downstream code that scopes a
community computation (e.g. per-tenant) will silently mix excluded nodes into
the modularity optimisation and receive community assignments influenced by —
and emitted for — data it filtered out.

## Minimal reproduction

```python
import real_ladybug as lb

db = lb.Database(":memory:")
conn = lb.Connection(db)
conn.execute("INSTALL algo; LOAD EXTENSION algo;")
conn.execute("CREATE NODE TABLE Memory(id INT64 PRIMARY KEY, workspace STRING);")
conn.execute("CREATE REL TABLE RELATED_TO(FROM Memory TO Memory);")

for i in range(1, 5):     # 4 nodes that should be EXCLUDED
    conn.execute(f"CREATE (:Memory {{id: {i}, workspace: '/ws/alpha'}});")
for i in range(5, 15):    # 10 nodes that match the predicate
    conn.execute(f"CREATE (:Memory {{id: {i}, workspace: '/ws/beta'}});")
for a, b in [(5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]:
    conn.execute(f"MATCH (a:Memory {{id:{a}}}), (b:Memory {{id:{b}}}) "
                 f"CREATE (a)-[:RELATED_TO]->(b);")

conn.execute("CALL PROJECT_GRAPH('g', "
             "{'Memory': \"n.workspace = '/ws/beta'\"}, {'RELATED_TO': 'true'});")

rows = conn.execute("CALL LOUVAIN('g') RETURN node.id, louvain_id;")
ids = []
while rows.has_next():
    ids.append(rows.get_next()[0])
print(sorted(ids))
# Actual:   [1, 2, ..., 14]  — 14 rows, excluded nodes included
# Expected: [5, 6, ..., 14]  — 10 rows

# Control on the SAME projection: PAGE_RANK returns 10 rows.
rows = conn.execute("CALL PAGE_RANK('g') RETURN node.id, rank;")
n = 0
while rows.has_next():
    rows.get_next(); n += 1
print(n)  # 10
```

## Expected

`LOUVAIN` should either honour the node predicate like `PAGE_RANK` and
`STRONGLY_CONNECTED_COMPONENTS` do, or reject filtered projections with an
error rather than silently computing on the unfiltered node set.
