"""Dedup must not fail open when the vector index goes stale.

Found while chasing an unrelated inconsistency: the same paraphrase pair merged
in one database (updated_existing, similarity 0.9488) but stored as new in
another, deterministically. The cause was not the threshold — the dedup probe
was getting ZERO candidates back:

    3 of 3 memories have embeddings, HNSW index returns nothing
    -> dedup sees no near-duplicates -> every store becomes stored_new

The merge branch itself is the trigger: it does DETACH DELETE + CREATE to
re-embed the surviving text, which can leave the index returning nothing. So one
merge could switch dedup off for every subsequent store, with no error, no
`degraded` flag, and no recovery until some later memory_search happened to
trip its own repair-on-use path.

memory_search has had repair-on-use since 0.9.0; the store path never did. A
healthy HNSW index always returns the k nearest neighbours, because cosine
distance is defined for every vector pair, so zero rows while embeddings exist
is an unambiguous signal that the index is broken.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/dedup-index-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

A = "Vega uses Redis for sessions."
B = "The Vega service uses Redis for session storage."
TAGS = ["vega", "cache"]

FILLER = [
    ("Helios checkout p99 latency is 850ms at peak load.", ["helios", "latency"]),
    ("Atlas stores its event log in Kinesis.", ["atlas", "eventlog"]),
    ("The ledger-db cluster is decommissioned in Q2 2026.", ["ledger", "lifecycle"]),
]


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server._index_repair_attempts = 0
    server._index_repair_last = 0.0
    yield
    server._conn = None
    server._db = None
    server._index_repair_attempts = 0
    server._index_repair_last = 0.0


def _break_the_index(conn):
    """Drop the HNSW index, leaving embeddings in place — the stale state."""
    server._safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                         expected_errors=("does not exist", "not found"))


def _probe_count(conn, text):
    emb = server._embed(text)
    try:
        res = conn.execute(
            """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
               WITH node AS m, distance RETURN m.id ORDER BY distance LIMIT 5;""",
            {"query": emb, "k": 20},
        )
        return len(server._collect_results(res))
    except Exception:
        return 0


def test_dedup_still_merges_after_the_index_goes_stale():
    a = server.memory_store.__wrapped__(content=A, tags=TAGS, importance=2)
    conn = server.get_conn()
    _break_the_index(conn)

    r = server.memory_store.__wrapped__(content=B, tags=TAGS)
    assert r["status"] == "updated_existing", (
        "a stale index must not make dedup fail open and store a duplicate as new"
    )
    assert r["id"] == a["id"]


def test_stale_index_is_actually_repaired_not_just_worked_around():
    """Probed on a store that does NOT merge, since a merge re-breaks the index."""
    server.memory_store.__wrapped__(content=A, tags=TAGS, importance=2)
    conn = server.get_conn()
    _break_the_index(conn)
    assert _probe_count(conn, B) == 0, "precondition: index returns nothing"

    unrelated = server.memory_store.__wrapped__(
        content="Billing reconciliation runs nightly against the ledger snapshot.",
        tags=["billing", "batch"])
    assert unrelated["status"] == "stored_new"
    assert _probe_count(conn, B) > 0, "the index should have been rebuilt in place"


def test_successful_repair_does_not_consume_the_budget():
    """The budget must count consecutive FAILURES, not total repairs.

    A merge's delete+recreate leaves the index unable to answer, so repairs are
    routine. Counting them meant the 3-attempt budget plus 300s cooldown was
    spent almost immediately, after which dedup silently failed open.
    """
    server.memory_store.__wrapped__(content=A, tags=TAGS, importance=2)
    conn = server.get_conn()

    _break_the_index(conn)
    server.memory_store.__wrapped__(
        content="Atlas stores its event log in Kinesis.", tags=["atlas", "eventlog"])
    assert server._index_repair_attempts == 0, "a successful repair should reset the counter"
    assert server._index_repair_allowed(), "the cooldown must not block the next repair"

    # A second independent breakage must still be repairable straight away.
    _break_the_index(conn)
    r = server.memory_store.__wrapped__(content=B, tags=TAGS)
    assert r["status"] == "updated_existing", \
        "dedup must keep working across repeated index breakages"


def test_dedup_keeps_working_across_many_consecutive_merges():
    """End-to-end: each merge re-breaks the index, so this is the real invariant."""
    base = server.memory_store.__wrapped__(
        content="Vega request timeout is 500 milliseconds.", tags=["vega", "config"],
        importance=3)
    paraphrases = [
        "The Vega request timeout is 500 milliseconds.",
        "Vega's request timeout is 500 milliseconds.",
        "The Vega service request timeout is 500 milliseconds.",
        "Vega request timeout: 500 milliseconds.",
    ]
    merged = 0
    for text in paraphrases:
        r = server.memory_store.__wrapped__(content=text, tags=["vega", "config"])
        if r["status"] in ("updated_existing", "already_exists"):
            merged += 1
    assert merged == len(paraphrases), (
        f"only {merged}/{len(paraphrases)} restatements were recognised as duplicates; "
        "dedup fails open once the index goes stale"
    )
    assert server._count_memories(server.get_conn()) == 1


def test_dedup_survives_a_merge_that_rebuilds_a_node():
    """The merge path's DETACH DELETE + CREATE is what triggered this."""
    for text, tg in FILLER:
        server.memory_store.__wrapped__(content=text, tags=tg)

    a = server.memory_store.__wrapped__(content=A, tags=TAGS, importance=2)
    # This merge re-embeds via delete + recreate.
    first = server.memory_store.__wrapped__(content=B, tags=TAGS)
    assert first["status"] == "updated_existing"

    # A further near-duplicate must still be recognised afterwards.
    again = server.memory_store.__wrapped__(
        content="The Vega service uses Redis for session storage and caching.", tags=TAGS)
    assert again["status"] == "updated_existing", (
        "dedup stopped working for every store after the first merge"
    )
    assert again["id"] == a["id"]


def test_repair_is_budgeted_and_does_not_loop():
    """An unfixable index must not trigger a rebuild on every single store."""
    server.memory_store.__wrapped__(content=A, tags=TAGS, importance=2)
    conn = server.get_conn()
    _break_the_index(conn)
    server._index_repair_attempts = server.INDEX_REPAIR_MAX_ATTEMPTS
    server._index_repair_last = 0.0

    # Budget exhausted: the store must still succeed rather than raise.
    r = server.memory_store.__wrapped__(content=B, tags=TAGS)
    assert r["status"] in ("stored_new", "updated_existing")


# ---------------------------------------------------------------------------
# Partial recall degradation: the blind spot the zero-rows check cannot see.
#
# An HNSW index can return rows for most queries yet miss specific memories
# (the classic failure mode after accumulated delete/recreate churn). Because
# rows do come back, the search/dedup self-heal never fires — observed on a
# long-lived DB as: a memory absent from results for a query it answered at
# 0.74 (surviving on FTS alone at 0.44), while still reachable via its own
# wording; and dream's edge-protection counter reading 0 because the protected
# partner was never surfaced to be protected.
#
# Dream now audits self-recall during its merge scan: querying the index with a
# memory's OWN embedding must return that memory. A miss (or a probe error —
# a missing index raises rather than returning empty) triggers one rebuild.
# ---------------------------------------------------------------------------


def _seed_corpus(n=25):
    for i in range(n):
        server.memory_store.__wrapped__(
            content=f"The service-{i} component exposes endpoint number {1000 + i} for diagnostics.",
            tags=[f"svc{i}", "diag"])


def test_dream_reports_zero_self_misses_on_a_healthy_index():
    _seed_corpus()
    out = server.memory_dream.__wrapped__(force=True)
    assert out["index_self_misses"] == 0
    assert out["index_rebuilt"] is False


def test_dream_detects_and_rebuilds_a_dead_index():
    _seed_corpus()
    conn = server.get_conn()
    server._safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                         expected_errors=("does not exist",))

    out = server.memory_dream.__wrapped__(force=True)
    assert out["index_self_misses"] >= 1, "the audit should notice the index cannot answer"
    assert out["index_rebuilt"] is True

    # And the rebuild is real: the index answers again.
    emb = server._embed("service-3 diagnostics endpoint")
    rows = server._collect_results(conn.execute(
        """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
           WITH node AS m, distance RETURN m.id LIMIT 5;""", {"query": emb, "k": 5}))
    assert len(rows) > 0


def test_dream_rebuilds_at_most_once_per_run():
    """The rebuild is a one-shot per dream: if the index is still broken after
    one rebuild, later misses must not trigger a rebuild storm."""
    _seed_corpus()
    conn = server.get_conn()
    server._safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                         expected_errors=("does not exist",))

    calls = {"n": 0}
    original = server._ensure_vector_index

    def counting(conn_, force_rebuild=False):
        calls["n"] += 1
        return original(conn_, force_rebuild=force_rebuild)

    server._ensure_vector_index = counting
    try:
        server.memory_dream.__wrapped__(force=True)
    finally:
        server._ensure_vector_index = original
    assert calls["n"] <= 1, f"dream attempted {calls['n']} rebuilds in one run"


def test_dry_run_does_not_rebuild():
    """dry_run previews; it must not mutate anything, index included."""
    _seed_corpus()
    conn = server.get_conn()
    server._safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                         expected_errors=("does not exist",))

    out = server.memory_dream.__wrapped__(force=True, dry_run=True)
    assert out.get("index_rebuilt") is False

    # Still broken afterwards — the preview changed nothing.
    assert server._probe_vector_index(conn) in (0, None)
