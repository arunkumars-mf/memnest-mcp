# HNSW index progressively loses reachability under insert+delete churn

**Engine:** real-ladybug 0.15.3 (storage 40), VECTOR extension
**Severity:** silent wrong results — `QUERY_VECTOR_INDEX` returns a shrinking
subset of the corpus with no error, no warning, and `k` large enough to return
everything.

## Summary

Repeated bursts of *insert N nodes, then delete those same N nodes* leave the
persistent nodes of an HNSW index progressively unreachable. The damage:

- is **monotonic** — each churn round strictly grows the unreachable set
  (observed 38 → 28 → 23 → 11 → 10 → 8 reachable over five rounds on a real
  corpus; the unreachable set at round *k* is a superset of round *k-1*),
- happens **in-process** — a close/reopen is *not* required (control A below
  degrades without ever closing), and reopen-only never degrades (control B),
  so serialization is innocent; it merely persists graph damage that delete
  maintenance already caused,
- is **deterministic** for a given graph and churn pattern — a production
  database twice collapsed to *exactly* 26-of-38 reachable after equivalent
  sessions,
- survives checkpoints, so every future session inherits it.

Deleting rows and recreating the *same* rows (id-stable delete+recreate, the
pattern used for embedding updates) does **not** reproduce this — that
distinction previously exonerated churn incorrectly. The damaging pattern is
transient population: grow the graph, then remove the growth. Deleted nodes
evidently participated in the surviving nodes' neighbour lists, and delete
maintenance leaves the survivors orphaned rather than re-linking them.

## Minimal reproduction

Single table, no properties beyond the vector, uniform random vectors,
`DELETE` (not `DETACH DELETE`), one process, no reopen:

```python
import random
import real_ladybug as lb

DIM, PERSISTENT, TRANSIENT, ROUNDS = 384, 38, 65, 5
rng = random.Random(1041)
vec = lambda: [rng.uniform(-1, 1) for _ in range(DIM)]

db = lb.Database("/tmp/lbug-churn-db")
conn = lb.Connection(db)
conn.execute("INSTALL VECTOR; LOAD EXTENSION VECTOR;")
conn.execute("CREATE NODE TABLE Item(id INT64 PRIMARY KEY, embedding FLOAT[384]);")
for i in range(PERSISTENT):
    conn.execute("CREATE (:Item {id: $id, embedding: $e});", {"id": i, "e": vec()})
conn.execute("CALL CREATE_VECTOR_INDEX('Item', 'item_vec_idx', 'embedding', "
             "metric := 'cosine');")

def reachable():
    rows = []
    r = conn.execute("MATCH (m:Item) RETURN m.id, m.embedding;")
    while r.has_next():
        rows.append(r.get_next())
    union = set()
    for _, emb in rows:  # union over every stored vector as query point
        q = conn.execute(
            "CALL QUERY_VECTOR_INDEX('Item', 'item_vec_idx', $q, $k) "
            "WITH node AS m RETURN m.id;", {"q": list(emb), "k": len(rows)})
        while q.has_next():
            union.add(q.get_next()[0])
    return len(rows), len(union)

print("baseline:", reachable())          # (38, 38)
nid = 1000
for rnd in range(1, ROUNDS + 1):
    ids = []
    for _ in range(TRANSIENT):
        conn.execute("CREATE (:Item {id: $id, embedding: $e});",
                     {"id": nid, "e": vec()})
        ids.append(nid); nid += 1
    for i in ids:
        conn.execute("MATCH (m:Item {id: $id}) DELETE m;", {"id": i})
    print(f"round {rnd}:", reachable())  # reachable shrinks from round ~3
```

Observed output (0.15.3, this seed):

```
baseline: (38, 38)
round 1: (38, 38)
round 2: (38, 38)
round 3: (38, 37)
round 4: (38, 37)
round 5: (38, 37)
```

With real text embeddings (clustered geometry) and the same protocol the
collapse is much faster and deeper: 38 → 21 → 19 → 13 → 10 across four rounds
on a fresh database seeded with 38 production memories. Aged files behave the
same.

## Onset is non-monotonic and seed-dependent — read before dismissing

Do not expect "bigger burst, more damage". On one production graph,
**30-node bursts caused a reachability shortfall on every round while
65-node bursts never did** — measured across 20 alternating rounds on the
same database. Onset also varies with RNG seed for synthetic vectors: some
seeds survive five rounds untouched.

This matters for triage: a reproduction attempt with a single burst size or
a single seed that happens to pass is **not** evidence the bug is absent —
that exact shape produced multiple independent false exonerations downstream
(same-id churn tests passing, large-burst rounds passing while small-burst
rounds failed on the same graph). Sweep burst sizes and seeds before
concluding safety.

## Controls

| Variant | Result |
|---|---|
| churn + reopen each round | degrades (monotonic) |
| churn only, never reopen (control A) | degrades — in-process |
| reopen only, no churn (control B) | never degrades |
| delete + recreate same ids (embedding-update pattern) | never degrades |
| checked: payload column / probe-reads / DETACH DELETE / clustering | none required; clustering worsens depth |

## Impact on applications

`QUERY_VECTOR_INDEX` silently serves a subset. Any application that stores
transient batches (session scratch, test fixtures, TTL'd items) into an
indexed table and deletes them will progressively lose vector search over its
*permanent* data, with no error signal at any layer. Detection requires an
application-level census (count distinct ids returnable at k ≥ corpus vs rows
with embeddings); repair requires dropping and rebuilding the index.

## Workaround used downstream

memnest-mcp runs a reachability census inside search (pool ≥ corpus), stats,
and its maintenance pass, and force-rebuilds the index on shortfall
(`CALL DROP_VECTOR_INDEX` + `CREATE_VECTOR_INDEX`). That converts silent wrong
results into a one-time repair cost per event, but the underlying delete-path
maintenance needs an engine-side fix.
