"""memory_search(explain=True): per-channel score breakdown.

Added for a ranking anomaly that was undiagnosable from outside: on one
long-lived database, a memory scored exactly its vector contribution below
expectation for one query while beating a competitor on every input visible
through the MCP surface (importance, recency, access, graph, lexical overlap).
The per-channel values were the only place the difference could live, and
nothing exposed them. explain makes the fusion auditable: raw channel values,
weighted contributions, and whether the memory came back from the vector index
for this query.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/explain-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

FACTS = [
    ("Helios considered adopting a service mesh in 2024 but the proposal was rejected.",
     ["helios", "servicemesh"]),
    ("Helios checkout p99 latency is 850ms at peak load.", ["helios", "latency"]),
    ("Atlas stores its event log in Kinesis.", ["atlas", "eventlog"]),
    ("The ledger-db cluster is decommissioned in Q2 2026.", ["ledger", "lifecycle"]),
]
QUERY = "Did Helios adopt a service mesh?"


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _seed():
    for text, tg in FACTS:
        server.memory_store.__wrapped__(content=text, tags=tg)


def test_explain_is_absent_by_default():
    _seed()
    out = server.memory_search.__wrapped__(query=QUERY, top_k=3)
    assert "explain_meta" not in out
    assert all("explain" not in r for r in out["results"])


def test_explain_reconstructs_the_reported_score():
    """The weighted contributions must sum to the score — if they do not, the
    explain block is describing a different formula than the one that ran."""
    _seed()
    out = server.memory_search.__wrapped__(query=QUERY, top_k=4, explain=True)
    assert out["results"]
    for r in out["results"]:
        w = r["explain"]["weighted"]
        recon = sum(w.values())
        if "superseded_penalty" in r["explain"]:
            recon *= r["explain"]["superseded_penalty"]
        assert abs(recon - r["score"]) < 0.002, \
            f"id {r['id']}: reported {r['score']} but channels sum to {recon}"


def test_explain_meta_describes_the_vector_window():
    _seed()
    out = server.memory_search.__wrapped__(query=QUERY, top_k=3, explain=True)
    meta = out["explain_meta"]
    assert meta["candidate_pool"] >= 9
    assert meta["query_embedded"] is True
    assert meta["fusion_mode"] in ("legacy", "normalized")
    assert 0 < meta["vector_hits"] <= meta["candidates_scored"]
    assert meta["weights"]["vector"] == 0.4


def test_in_vector_window_reflects_semantic_reachability():
    """The diagnostic bit: a healthy corpus answering a matching query should
    have its top hit inside the vector window."""
    _seed()
    out = server.memory_search.__wrapped__(query=QUERY, top_k=3, explain=True)
    top = out["results"][0]
    assert top["explain"]["in_vector_window"] is True
    assert top["explain"]["vector"] > 0


def test_superseded_results_expose_the_penalty():
    a = server.memory_store.__wrapped__(
        content="The Thuban cache TTL is 60 seconds.", tags=["thuban", "cache"])
    server.memory_store.__wrapped__(
        content="The Thuban cache TTL is 300 seconds.", tags=["thuban", "cache"],
        supersedes=a["id"])

    out = server.memory_search.__wrapped__(
        query="what is the Thuban cache TTL", top_k=5,
        include_superseded=True, explain=True)
    stale = [r for r in out["results"] if r.get("superseded")]
    assert stale, "the superseded version should still be returned with the flag"
    for r in stale:
        assert r["explain"]["superseded_penalty"] == server.SUPERSEDED_PENALTY
        w = r["explain"]["weighted"]
        assert abs(sum(w.values()) * server.SUPERSEDED_PENALTY - r["score"]) < 0.002
