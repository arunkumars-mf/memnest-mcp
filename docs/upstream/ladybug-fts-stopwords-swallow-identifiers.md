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

### The unified rule, and what it predicts

Measured token-by-token against the raw index:

| token in content | searchable | why |
|---|---|---|
| `abcd88` | yes | mixed token, alpha run ≥ 2 |
| `bug`, `cve`, `rfc`, `sev`, `req`, `aud`, `corp` | yes | ordinary words |
| `inc`, `ltd`, `it`, `the` | **no** | stopwords |
| `3300`, `7734`, `2026`, `17`, `8080`, `1.29` | **no** | bare numerics |
| `q2`, `p99` | **no** | mixed token, alpha run of 1 |

The alpha-run threshold is exactly 2, and it is a hard boundary. Four tokens
placed in a **single document** and queried against the same index:

| token | alpha run | matches |
|---|---|---|
| `q7` | 1 | **no** |
| `ab7` | 2 | yes |
| `abc7` | 3 | yes |
| `abcd7` | 4 | yes |

**Complete rule: a token is indexed iff it contains an alphabetic run of at
least two characters that is not a stopword.**

So a hyphenated or spaced identifier is split, and **survives only through its
non-stopword alphabetic parts**. Two consequences:

1. If every alphabetic part is a stopword, the identifier is entirely
   unsearchable: `INC-3300` (`inc`) and `IT-4471` (`it`). Both are among the
   most common ticket prefixes in existence — `INC-` for incidents, `IT-` for
   service desks.
2. **Otherwise the identifier matches, but cannot be distinguished from any
   variant differing only in its numeral** — and this is worse than a miss,
   because a miss is visible. Measured: for query `eu-west-2`, both
   `eu-west-1` and `eu-west-2` documents score fts **1.0**. The discriminating
   numeral is never read; which one ranks first is decided by embedding
   similarity. The caller receives a confident, well-scored, wrong region.

Corpus-realistic instances of (2) are pervasive: `us-east-1`/`us-east-2`,
`Java 17`/`Java 21`, `PostgreSQL 15`/`14`, ports `8080`/`9090`.

The alpha-run-of-1 rule additionally erases whole categories of technical
shorthand that applications record verbatim:

- **Quarters** — `Q1`–`Q4`, i.e. the entire temporal dimension of a roadmap.
- **Percentiles** — `p50`, `p95`, `p99`, i.e. every latency SLO.
- **Version prefixes** — `v1`, `v2`.
- **Cloud instance families** — verified unindexed: `c6g`, `m5`, `t3`. The
  family designator is where the sizing lives, so instance types are invisible.

Worked example, on three documents differing **only** in the quarter
("…decommissioned in Q2/Q3/Q4 2026") queried with `decommissioned in Q4 2026`:
all three score fts **1.0**. The quarter contributes nothing, because `q2`,
`q4` and `2026` are all unindexed and the entire match comes from
`decommissioned`. Which quarter ranks first is then decided solely by embedding
similarity — in one run of this fixture the correct one won; on a real corpus
the same query ranked the **Q3** memory first while the correct Q4 answer was
present and unambiguous. That is the failure mode: not a wrong answer every
time, but a coin flip presented with a confident score, on a question the
keyword index should have settled outright.

Any application relying on FTS to separate versions, regions, ports, quarters,
percentiles or instance types is relying on the embedding channel without
knowing it.

### An asymmetry that localises the gap

The same numerals are perfectly legible elsewhere in the same database. A
value-conflict detector reading the same `content` strings distinguishes
"30 days" from "one year", "512 MB" from "2 GB", and "500 ms" from "900 ms".
So numbers are not lost in storage — they are lost in the FTS index
specifically, which is what makes this an index-configuration bug rather than
anything intrinsic to the data.

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

1. Do not treat `inc`, `ltd` or `it` as stopwords by default, or document
   prominently that they are, since they collide with the most common
   identifier prefixes (`INC-`, `IT-`).
2. Index bare numeric tokens, or provide an option to. This is the item that
   fixes the confident-wrong-answer class, not just the retrieval miss.
3. Index mixed tokens whose alphabetic run is a single character. The threshold
   is exactly 2 and can be checked against the tokenizer in one read:
   `q7` is dropped, `ab7` is kept.
4. Fix the segfault in the `stopWords :=` path, and in the meantime make it
   raise rather than crash.

Priority order from an application's perspective: (2) first, because a silent
wrong answer is worse than a visible miss; then (4), because it unblocks the
only documented workaround; then (1) and (3).
