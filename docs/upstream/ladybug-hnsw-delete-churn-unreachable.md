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
| insert only, no deletes at all | not reproduced in 24 trials — see below |
| insert only, on a file with prior delete history | not reproduced (3 prior churn rounds, then +65: 103/103) |
| checked: payload column / probe-reads / DETACH DELETE / clustering | none required; clustering worsens depth |

### Insert-only: one field report, unreproduced in 24 trials

Deletion is the only operation reproduced as damaging here. Insert-only was
tried 24 times without a single degradation: 4 sequential rounds of +65 into a
38-node index in both uniform and clustered geometry (38 → 103 → 168 → 233 →
298, fully reachable throughout), 16 independent seeds of base-38-then-insert-65
with the interleaved neighbour query a dedup path performs, and an insert into
a database carrying prior delete history.

Against that, one downstream field observation stands unexplained: a 38-node
index read 38/38 reachable by an independent census, a 65-item batch insert
followed with no deletes, and the next query measured a shortfall and rebuilt.
A repeat of the identical sequence on the same database was clean. Since onset
for the delete case is already known to be non-monotonic in batch size and
seed-dependent, a stochastic insert-side trigger cannot be ruled out from
negative trials alone.

Two cautions for whoever triages this:

- Do not treat "a shortfall appeared after an insert" as evidence that
  insertion damages the graph; the reproducible trigger is deletion, and
  chasing the insert path first will burn time.
- Be careful with small-corpus baselines in general. A `k = corpus` probe on a
  small index approaches an exhaustive scan, so `reachable == embedded` there
  is weaker evidence of a sound graph than it looks. Growing a corpus can make
  an existing defect measurable, which is easily mistaken for the growth
  having caused it.

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
