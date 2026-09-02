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
