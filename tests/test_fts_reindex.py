"""FTS index rebuild path and health signal.

Field diagnosis (explain=True traces on the affected database): a memory
scored fts 0.0 on the full question "Did Helios adopt a service mesh?" while
the same query returned fts 1.0 on a fresh ingest of identical content — with
the vector channel bit-identical (cosine 0.7987 both sides). The missing 0.30
was exactly the FTS weight. Rare-term probes ("mesh" -> 1.0) and exact-text
probes passed on the same broken index, so the failure is query-shape
dependent and has no cheap server-side detector.

Until now FTS was the only channel with no rebuild path and no health field:
memory_reindex() rebuilt just the HNSW index, which is why a full "reindex"
produced byte-identical scores on the affected database. Now:
  - memory_reindex() rebuilds BOTH indexes and reports fts_rebuilt
  - dream rebuilds FTS unconditionally (bounded: at most daily; no detector
    can be trusted, see above)
  - memory_stats runtime carries fts_index.answering (necessary-not-sufficient)
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/fts-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

MESH = "Helios considered adopting a service mesh in 2024 but the proposal was rejected due to operational complexity."
QUERY = "Did Helios adopt a service mesh?"


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _seed():
    a = server.memory_store.__wrapped__(content=MESH, tags=["helios", "servicemesh"])
    for text, tg in (
        ("Helios checkout p99 latency is 850ms at peak load.", ["helios", "latency"]),
        ("Atlas stores its event log in Kinesis.", ["atlas", "eventlog"]),
        ("The ledger-db cluster is decommissioned in Q2 2026.", ["ledger", "lifecycle"]),
    ):
        server.memory_store.__wrapped__(content=text, tags=tg)
    return a["id"]


def _fts_of(mid):
    out = server.memory_search.__wrapped__(query=QUERY, top_k=4, explain=True)
    for r in out["results"]:
        if r["id"] == mid:
            return r["explain"]["fts"]
    return None


def _drop_fts(conn):
    server._safe_execute(conn, "CALL DROP_FTS_INDEX('Memory', 'memory_fts_idx');",
                         expected_errors=("does not exist",))


def test_reindex_rebuilds_the_fts_index_too():
    mid = _seed()
    conn = server.get_conn()
    assert _fts_of(mid) == 1.0, "precondition: healthy index gives fts 1.0"

    _drop_fts(conn)
    assert _fts_of(mid) in (None, 0.0), "precondition: dropped index loses the term"

    out = server.memory_reindex.__wrapped__()
    assert out["fts_rebuilt"] is True
    assert out["status"] == "rebuilt"
    assert _fts_of(mid) == 1.0, "reindex must restore the keyword channel"


def test_reindex_reports_fts_state_even_without_embeddings():
    conn = server.get_conn()
    # Store without embeddings by disabling the model path
    original = server._embed
    server._embed = lambda text: None
    try:
        server.memory_store.__wrapped__(content=MESH, tags=["helios"])
    finally:
        server._embed = original

    out = server.memory_reindex.__wrapped__()
    assert out["status"] == "no_embeddings"
    assert "fts_rebuilt" in out, "FTS is independent of embeddings and still rebuilt"


def test_dream_rebuilds_fts_routinely():
    _seed()
    for i in range(20):
        server.memory_store.__wrapped__(
            content=f"The svc-{i} component exposes endpoint {2000 + i}.", tags=[f"svc{i}"])
    conn = server.get_conn()
    _drop_fts(conn)

    out = server.memory_dream.__wrapped__(force=True)
    assert out["fts_rebuilt"] is True
    assert server._probe_fts_index(conn) is True


def test_dream_dry_run_does_not_touch_fts():
    _seed()
    conn = server.get_conn()
    _drop_fts(conn)
    out = server.memory_dream.__wrapped__(force=True, dry_run=True)
    assert out.get("fts_rebuilt") is False
    assert server._probe_fts_index(conn) is False, "dry_run must not rebuild"


def test_stats_exposes_fts_health():
    _seed()
    conn = server.get_conn()
    st = server.memory_stats.__wrapped__()
    assert st["runtime"]["fts_index"]["answering"] is True

    _drop_fts(conn)
    st = server.memory_stats.__wrapped__()
    assert st["runtime"]["fts_index"]["answering"] is False
