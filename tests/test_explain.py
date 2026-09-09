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
    assert meta["fusion_mode"] in ("legacy", "normalized", "rrf")
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


# --- the explain block must not advertise a penalty it did not apply ---------
#
# Cycle members are exempt from the x0.5 supersession multiplier, but the block
# still printed `superseded_penalty: 0.5` beside a score that was never halved.
# The sum-to-score invariant still held, so the formula was not misdescribed —
# but a multiplier reported next to a score it was not applied to is the same
# class of dishonest diagnostic as `status: "ok"` next to `fully_reachable:
# false`, or a degraded notice asserting repair failure after a successful
# repair. Individually cosmetic; collectively the reason a diagnostic block
# stops being trusted.

def _sum_weighted(entry):
    return sum(entry["explain"]["weighted"].values())


def test_penalised_row_reports_the_multiplier_and_the_score_reflects_it():
    old = server.memory_store.__wrapped__(
        content="The Menkar worker pool holds 8 threads.", tags=["menkar", "threads"])
    server.memory_store.__wrapped__(
        content="Correction: the Menkar worker pool now holds 32 threads.",
        tags=["menkar", "threads"], supersedes=old["id"])

    out = server.memory_search.__wrapped__(
        query="how many threads in the Menkar worker pool", top_k=5, explain=True)
    stale = next((r for r in out["results"] if r["id"] == old["id"]), None)
    if stale is None:
        pytest.skip("stale version demoted out of the window in this fusion mode")

    assert stale["explain"]["superseded_penalty"] == server.SUPERSEDED_PENALTY
    assert abs(stale["score"]
               - _sum_weighted(stale) * server.SUPERSEDED_PENALTY) < 0.0005, \
        "a reported multiplier must be the one the score actually used"


def test_exempt_row_reports_no_multiplier_and_names_the_exemption():
    ids, prev = [], None
    for v in ("The Rukbat cache size is 256 megabytes.",
              "Correction: the Rukbat cache size is 512 megabytes.",
              "Correction: the Rukbat cache size is 1 gigabyte."):
        kw = {"supersedes": prev} if prev else {}
        ids.append(server.memory_store.__wrapped__(
            content=v, tags=["rukbat", "cachesize"], **kw)["id"])
        prev = ids[-1]
    server.memory_relate.__wrapped__(from_id=ids[0], to_id=ids[2],
                                     relationship="SUPERSEDES")

    out = server.memory_search.__wrapped__(query="what is the Rukbat cache size",
                                           top_k=3, explain=True)
    members = [r for r in out["results"] if r["id"] in ids]
    assert members, "cycle members must be retrievable"
    for r in members:
        ex = r["explain"]
        assert r.get("superseded") is True, \
            "each member genuinely IS superseded — that stays reported"
        assert ex["superseded_penalty"] is None, \
            "no multiplier may be advertised when none was charged"
        assert ex["superseded_penalty_exempt"] == "supersession_cycle"
        assert abs(r["score"] - _sum_weighted(r)) < 0.0005, \
            "an exempt row's score must equal its unpenalised weighted sum"
