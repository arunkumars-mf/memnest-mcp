"""Tests for retrieval health signalling and fusion configuration.

Regression cover for the silent-degradation bug: when the embedding model is
unavailable, hybrid search quietly became keyword-only and still returned
plausible scores, so callers could not tell that retrieval was degraded.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/health-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def fresh_db():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _search(query, **kw):
    return server.memory_search.__wrapped__(query=query, **kw)


def test_healthy_search_has_no_degraded_flag():
    server.memory_store.__wrapped__(content="Redis is used for session caching")
    res = _search("what caches sessions")
    assert res["results"], "expected a hit"
    assert "degraded" not in res, f"unexpected degraded flag: {res.get('degraded')}"


def test_degraded_flag_when_embedding_unavailable(monkeypatch):
    """Simulate a failed embedding model: search must still work AND say so."""
    server.memory_store.__wrapped__(content="Redis is used for session caching")

    monkeypatch.setattr(server, "_embed", lambda text: None)
    res = _search("redis session caching")

    assert "degraded" in res, "silent degradation: no signal that vector search died"
    assert "keyword-only" in res["degraded"]
    # Must still return usable keyword results rather than failing outright
    assert res["results"], "keyword fallback should still return results"


def test_no_degraded_flag_on_empty_database(monkeypatch):
    """An empty DB is not 'degraded' — there is simply nothing to find."""
    monkeypatch.setattr(server, "_embed", lambda text: None)
    res = _search("anything at all")
    assert res["results"] == []
    assert "degraded" not in res


def test_scores_never_negative():
    """Cosine distance is 0..2, so 1-distance can go negative without a clamp.
    A dissimilar hit must never subtract from a memory's relevance."""
    server.memory_store.__wrapped__(items=[
        {"content": "Postgres connection pooling uses pgbouncer"},
        {"content": "The mobile app uses Kotlin Multiplatform"},
    ])
    # Query deliberately unrelated to everything stored
    res = _search("medieval Byzantine coinage debasement", top_k=10)
    for r in res["results"]:
        assert r["score"] >= 0.0, f"negative fused score: {r}"


def test_stats_reports_embedding_health_and_fusion_mode():
    server.memory_store.__wrapped__(content="Kinesis feeds the search indexer")
    rt = server.memory_stats.__wrapped__()["runtime"]

    assert rt["fusion_mode"] in ("legacy", "normalized")
    emb = rt["embeddings"]
    assert emb["model"], "embedding model name should be reported"
    assert emb["missing"] == 0
    assert emb["healthy"] is True


def test_stats_flags_memories_missing_embeddings(monkeypatch):
    """Memories stored while the model was down are invisible to semantic
    search; stats must surface the count so it is recoverable."""
    monkeypatch.setattr(server, "_embed", lambda text: None)
    monkeypatch.setattr(server, "_embed_batch", lambda texts: [None] * len(texts))
    res = server.memory_store.__wrapped__(content="stored while model was down")
    assert res["status"] == "stored_new_no_embedding"

    rt = server.memory_stats.__wrapped__()["runtime"]
    assert rt["embeddings"]["missing"] == 1
    assert rt["embeddings"]["healthy"] is False


def test_both_fusion_modes_produce_valid_rankings():
    """Switching fusion must not break scoring in either mode.

    Uses the 6-fact corpus from the published side-by-side comparison, where
    both modes rank correctly. (With a degenerate 3-memory corpus, BM25 IDF is
    meaningless — every document contains the query's prominent tokens — and
    'legacy' can let a keyword distractor win. That is the case
    MEMORY_FUSION=normalized is available for.)
    """
    server.memory_store.__wrapped__(items=[
        {"content": "The payments service uses DynamoDB table 'PaymentsLedger' in us-east-1 for transaction records."},
        {"content": "The payments service on-call rotation is owned by the team 'payments-core'."},
        {"content": "The payments service is written in Java 17 and deployed via Apollo."},
        {"content": "Retry policy for the payments service: exponential backoff, max 3 attempts."},
        {"content": "Updated retry policy for the payments service: max 5 attempts, jitter enabled."},
        {"content": "Incident INC-4821 was caused by a DynamoDB throttling event on PaymentsLedger."},
    ])
    original = server.FUSION_MODE
    try:
        for mode in ("legacy", "normalized"):
            server.FUSION_MODE = mode
            res = _search("what database does the payments service use", top_k=3)
            assert res["results"], f"{mode}: no results"
            scores = [r["score"] for r in res["results"]]
            assert scores == sorted(scores, reverse=True), f"{mode}: not sorted"
            assert all(0.0 <= s <= 1.0 for s in scores), f"{mode}: score out of range {scores}"
            top = res["results"][0]["content"]
            assert "DynamoDB" in top, f"{mode}: keyword distractor outranked answer ({top!r})"
    finally:
        server.FUSION_MODE = original


def test_invalid_fusion_mode_is_rejected():
    """Config typos should fail loudly at import, not silently pick a mode."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src'); import memnest_mcp.server"],
        env={**os.environ, "MEMORY_FUSION": "bogus", "MEMORY_DB_PATH": ":memory:"},
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert r.returncode != 0
    assert "MEMORY_FUSION" in (r.stdout + r.stderr)


# --- Stale vector index: detection, self-heal, and manual repair -------------
#
# Observed in the field: a database carried across versions and delete cycles
# had memory_vec_idx present and all 127 embeddings stored, yet
# QUERY_VECTOR_INDEX returned zero rows for every query. Semantic search
# silently degraded to keyword-only while reporting healthy storage.
# These tests simulate that state by removing the index from under the data.

def _drop_vector_index(conn):
    conn.execute("CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');")


def _seed(n=12):
    server.memory_store.__wrapped__(items=[
        {"content": f"The {s} service retrains its classifier on schedule {i}."}
        for i, s in enumerate(["billing", "fraud", "search", "checkout", "ledger",
                               "pricing", "identity", "cart", "shipping",
                               "inventory", "auth", "notification"][:n])
    ])


def test_probe_detects_missing_vector_index():
    _seed()
    conn = server.get_conn()
    assert (server._probe_vector_index(conn) or 0) > 0, "healthy index should return rows"
    _drop_vector_index(conn)
    # A dropped index makes the probe fail rather than return rows
    assert not (server._probe_vector_index(conn) or 0) > 0


def test_search_reports_degraded_when_index_is_stale():
    _seed()
    conn = server.get_conn()
    _drop_vector_index(conn)
    res = _search("how often is the anti-abuse classifier refreshed")
    assert "degraded" in res, "a stale index must be reported, not silently tolerated"
    assert "memory_reindex" in res["degraded"], "message should name the remedy"


def test_memory_reindex_repairs_a_stale_index():
    _seed()
    conn = server.get_conn()
    _drop_vector_index(conn)
    assert "degraded" in _search("classifier retraining schedule")

    out = server.memory_reindex.__wrapped__()
    assert out["status"] == "rebuilt", out
    assert out["embeddings_indexed"] == 12
    assert out["index_rows_after"] > 0

    res = _search("classifier retraining schedule")
    assert "degraded" not in res, "search should be healthy after reindex"
    assert res["results"]


def test_connection_self_heals_stale_index(tmp_path, monkeypatch):
    """Reopening a persistent database must restore working semantic search
    with no user action.

    Uses a file-backed DB because the data has to survive the reconnect (the
    default ':memory:' would start empty). Note that a *dropped* index is
    recreated by _init_schema, whose CREATE is only a no-op when the index
    already exists. The field failure was subtler — the index existed but
    returned nothing, so _init_schema skipped it and only the
    _ensure_vector_index probe catches it. Either way, what a caller needs is
    that search works again after a reconnect.
    """
    def close_conn():
        """Close before reopening. Nulling the globals alone leaves the old
        handle holding the WAL, which corrupts the next open on a file DB."""
        try:
            if server._conn is not None:
                server._conn.close()
            if server._db is not None:
                server._db.close()
        except Exception:
            pass
        server._conn = None
        server._db = None

    db = tmp_path / "persist.lbug"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db))
    close_conn()

    _seed()
    conn = server.get_conn()
    assert str(db) == server.DB_PATH, server.DB_PATH
    _drop_vector_index(conn)
    assert "degraded" in _search("classifier retraining schedule")

    # Force a reconnect (as a server restart would)
    close_conn()
    server.get_conn()

    assert server._vector_index_state.get("status") in ("ok", "rebuilt"), \
        server._vector_index_state
    res = _search("classifier retraining schedule")
    assert "degraded" not in res, "reconnect should leave search healthy"
    assert res["results"]

    close_conn()


def test_stats_distinguishes_stored_from_queryable():
    """The old health check said 'healthy' whenever vectors were stored. It must
    now also report whether the index actually returns them."""
    _seed()
    conn = server.get_conn()
    _drop_vector_index(conn)

    emb = server.memory_stats.__wrapped__()["runtime"]["embeddings"]
    assert emb["missing"] == 0
    assert emb["stored_ok"] is True, "vectors are stored"
    assert emb["index_returns_rows"] is False, "but the index returns nothing"
    assert emb["healthy"] is False, "so overall health must be False"


def test_reindex_on_empty_database():
    out = server.memory_reindex.__wrapped__()
    assert out["status"] == "empty"


def test_reindex_reports_when_nothing_has_embeddings(monkeypatch):
    monkeypatch.setattr(server, "_embed", lambda text: None)
    monkeypatch.setattr(server, "_embed_batch", lambda texts: [None] * len(texts))
    server.memory_store.__wrapped__(content="stored while the model was down")

    out = server.memory_reindex.__wrapped__()
    assert out["status"] == "no_embeddings"
    assert out["memories"] == 1
    assert "re-store" in out["message"]
