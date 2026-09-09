# FTS makes some identifiers unsearchable, and the documented fix segfaults

**Engine:** real-ladybug 0.15.3, FTS extension
**Severity:** silent retrieval loss on exact-identifier queries; plus a crash in
the configuration option that would fix it.

## Part 1 — `INC-3300` is unsearchable, `BUG-7734` is fine

Two independent tokenizer behaviours combine into a total loss:

1. **`inc` and `ltd` are English stopwords**, so they are dropped at index time.
2. **Bare numeric tokens are never indexed** — `3300`, `7734`, `1111`, `2222`
   all return nothing, in every configuration tested.

A memory whose content literally contains `INC-3300` therefore has *nothing*
searchable from that identifier: the alphabetic half is a stopword and the
numeric half is not indexed. `QUERY_FTS_INDEX` scores it 0.0 for the exact
string `INC-3300`.

Measured on identical documents differing only in the identifier prefix:

| query | indexed / matches |
|---|---|
| `INC-3300`, `INC`, `3300`, `INC 3300`, `"INC-3300"` | **no** |
| `BUG-7734`, `CVE-2024`, `SEV-1`, `RFC-2119` | yes |
| `us-east-1`, `query-index`, `payments-core` | yes |
| `corp`, `bug`, `aud`, `sev`, `req`, `cve`, `rfc` | yes |
| `inc`, `ltd` | **no** |

Note `BUG-7734` matches only via `bug` — its numeric half is unindexed too, at
an identical BM25 score. Hyphenation is *not* the problem, and neither is
trailing punctuation: `AAA-1111:` with a colon indexes and matches normally.

Why this matters more than a prose-search quirk: hyphenated identifiers are
exactly what a memory system stores as retrieval anchors — incident IDs, ticket
keys, CVEs, ISO dates, semver. `INC-` is the most common incident prefix in
industry, and an agent that records `INC-3300` and later searches for it
verbatim gets no keyword contribution at all, falling back to a vector channel
that treats an opaque identifier as near-noise.

Reproduction:

```python
import real_ladybug as lb
db = lb.Database("/tmp/fts-id"); c = lb.Connection(db)
c.execute("INSTALL FTS; LOAD EXTENSION FTS;")
c.execute("CREATE NODE TABLE D(id INT64 PRIMARY KEY, content STRING);")
c.execute("CREATE (:D {id:1, content:'INC-3300 the ledger failover degraded checkout'});")
c.execute("CREATE (:D {id:2, content:'BUG-7734 tracks the invoice date off-by-one'});")
c.execute("CALL CREATE_FTS_INDEX('D','ix',['content'], stemmer := 'english');")

for q in ("INC-3300", "INC", "3300", "BUG-7734", "BUG", "7734", "ledger"):
    r = c.execute("CALL QUERY_FTS_INDEX('D','ix',$q,top:=3) "
                  "WITH node AS n, score RETURN n.id;", {"q": q})
    rows = []
    while r.has_next():
        rows.append(r.get_next()[0])
    print(f"{q:<10} -> {rows}")
```

Observed: `INC-3300 -> []`, `INC -> []`, `3300 -> []`, `BUG-7734 -> [2]`,
`BUG -> [2]`, `7734 -> []`, `ledger -> [1]`.

## Part 2 — the fix crashes

`CREATE_FTS_INDEX` accepts `stopWords := '<NodeTableName>'`, and supplying a
custom table does fix Part 1: with an empty stopword table, `INC`, `inc` and
`INC-3300` all match. (`3300` still does not — the numeric behaviour is
independent of stopwords.)

Standalone this works, including with a populated stopword table (verified at 0,
1, 5 and 10 rows). But wiring the same option into a real application — a
~150-word stopword table created and populated at schema-init time, then
`CREATE_FTS_INDEX(..., stopWords := 'FtsStopWord')`, then storing rows and
running `QUERY_FTS_INDEX` — reliably produced:

```
Segmentation fault: 11
```

The crash comes after "Schema initialized", on the first query path, and takes
the process down with no Python-level exception. Reverting to the default
stopword list removes it. We have not isolated the minimal trigger (candidate
differences from the working standalone: number of stopword rows, the stopword
table being created *after* the indexed node table, or interaction with other
indexes present on the same table — vector/HNSW plus FTS).

Net effect for applications: the identifier problem in Part 1 has a documented
configuration fix that cannot currently be used in production, so the practical
workaround is a shadow content field carrying de-hyphenated identifier variants,
with matching query rewriting — considerably more machinery than
`stopWords :=` should require.

## What we would ask for

1. Do not treat `inc` / `ltd` as stopwords by default, or document prominently
   that they are, since they collide with the most common identifier prefixes.
2. Index bare numeric tokens, or provide an option to.
3. Fix the segfault in the `stopWords :=` path, and in the meantime make it
   raise rather than crash.
