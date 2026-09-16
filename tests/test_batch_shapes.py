"""Batch shapes across the write and read tools.

Five tools already accepted batches (store/items, update/updates,
relate/relations, unrelate/relations, delete/int-or-list). Two did not, and the
gaps were not symmetrical:

- `memory_keep_separate` LOOKED batched -- it takes a list -- but the list is a
  CLIQUE: every pair among the ids is recorded. An agent resolving three dream
  clusters in one call by passing all their ids would record verdicts on
  cross-cluster pairs nobody examined, permanently suppressing real conflicts
  between them. That is a silent correctness loss dressed as an optimisation,
  which is worse than having no batch path at all.
- `memory_get` took only an int, though `memory_delete` accepts int-or-list for
  the same parameter. Every id-reporting surface (conflict flags, review
  clusters, supersession cycles) reports several ids at once, so reading them
  back was N round trips.
"""
import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/batch-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server._index_repair_attempts = 0
    yield


def _seed(n=4):
    out = server.memory_store.__wrapped__(items=[
        {"content": f"Rigel stage {i} uses batch window {i * 5} minutes.",
         "tags": [f"rigel{i}"]}
        for i in range(n)])
    return [r["id"] for r in out["results"]]


# --- memory_keep_separate ----------------------------------------------------

def test_clique_records_every_pair():
    a, b, c, _ = _seed()
    out = server.memory_keep_separate.__wrapped__(memory_ids=[a, b, c])
    pairs = {tuple(p) for p in out["pairs_recorded"]}
    assert pairs == {(a, b), (a, c), (b, c)}, (
        "memory_ids is documented as a clique; if this changes, the docstring "
        "and the pairs= alternative both need revisiting")


def test_pairs_records_only_the_pairs_given():
    a, b, c, d = _seed()
    out = server.memory_keep_separate.__wrapped__(pairs=[[a, b], [c, d]])
    pairs = {tuple(p) for p in out["pairs_recorded"]}
    assert pairs == {(a, b), (c, d)}
    for cross in ((a, c), (a, d), (b, c), (b, d)):
        assert cross not in pairs, (
            f"{cross} was never examined by the agent; recording it would "
            f"suppress a conflict nobody looked at")


def test_pairs_is_the_only_safe_way_to_resolve_two_clusters_at_once():
    """The point of the parameter, stated as a comparison."""
    a, b, c, d = _seed()
    conn = server.get_conn()

    server.memory_keep_separate.__wrapped__(pairs=[[a, b], [c, d]])
    assert server._pair_kept_separate(conn, a, b)
    assert server._pair_kept_separate(conn, c, d)
    assert not server._pair_kept_separate(conn, a, c), (
        "cross-cluster pair must remain eligible for conflict detection")

    # What the clique shape would have done with the same intent.
    server._conn = None
    server._db = None
    a, b, c, d = _seed()
    conn = server.get_conn()
    server.memory_keep_separate.__wrapped__(memory_ids=[a, b, c, d])
    assert server._pair_kept_separate(conn, a, c), (
        "demonstrates the hazard the pairs= parameter exists to avoid")


def test_pairs_normalises_order_and_deduplicates_against_the_clique():
    a, b, c, _ = _seed()
    out = server.memory_keep_separate.__wrapped__(
        memory_ids=[a, b], pairs=[[b, a], [a, c]])
    pairs = [tuple(p) for p in out["pairs_recorded"]]
    assert len(pairs) == len(set(pairs)), f"duplicate writes: {pairs}"
    assert set(pairs) == {(a, b), (a, c)}, (
        "[b, a] is the same verdict as [a, b] and must not be written twice")


def test_pairs_rejects_malformed_entries():
    a, b, _, _ = _seed()
    for bad in ([[a]], [[a, b, a]], [[a, a]], [["x", b]]):
        out = server.memory_keep_separate.__wrapped__(pairs=bad)
        assert out["status"] == "error", f"{bad} should be rejected"
        assert "bad_pair" in out


def test_pairs_reports_missing_memories_without_failing_the_rest():
    a, b, c, _ = _seed()
    ghost = 999_999
    out = server.memory_keep_separate.__wrapped__(pairs=[[a, b], [c, ghost]])
    assert {tuple(p) for p in out["pairs_recorded"]} == {(a, b)}
    assert out["skipped_pairs"] == [[c, ghost]]
    assert ghost in out["not_found"]


def test_neither_shape_given_is_an_error_naming_both():
    out = server.memory_keep_separate.__wrapped__()
    assert out["status"] == "error"
    assert "memory_ids" in out["message"] and "pairs" in out["message"]


# --- memory_get --------------------------------------------------------------

def test_get_single_id_shape_is_unchanged():
    a, _, _, _ = _seed()
    out = server.memory_get.__wrapped__(memory_id=a)
    assert out["status"] == "found"
    assert out["id"] == a
    assert "results" not in out, (
        "wrapping a single-id response would break every existing caller")


def test_get_accepts_a_list():
    a, b, c, _ = _seed()
    out = server.memory_get.__wrapped__(memory_id=[a, b, c])
    assert out["count"] == 3
    assert [r["id"] for r in out["results"]] == [a, b, c]
    assert all(r["content"] for r in out["results"])


def test_get_list_reports_missing_ids_separately():
    a, _, _, _ = _seed()
    out = server.memory_get.__wrapped__(memory_id=[a, 999_999])
    assert out["count"] == 1
    assert out["not_found"] == [999_999]


def test_get_list_includes_edges_like_the_single_path():
    a, b, _, _ = _seed()
    server.memory_relate.__wrapped__(from_id=a, to_id=b,
                                     relationship="SUPERSEDES")
    single = server.memory_get.__wrapped__(memory_id=b)
    batch = server.memory_get.__wrapped__(memory_id=[b])["results"][0]
    assert single.get("superseded") is True
    assert batch.get("superseded") is True, (
        "the batch path must not lose the field that says a memory is stale — "
        "sharing _get_one is what guarantees it")


def test_get_list_bumps_access_count_once_per_id():
    a, _, _, _ = _seed()
    before = server.memory_get.__wrapped__(memory_id=a)["access_count"]
    server.memory_get.__wrapped__(memory_id=[a, a])  # duplicate on purpose
    after = server.memory_get.__wrapped__(memory_id=a)["access_count"]
    assert after == before + 2, (
        f"expected one bump for the deduplicated list read plus one for this "
        f"read, got {before} -> {after}")
