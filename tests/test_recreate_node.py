"""Contract of _recreate_memory_node — the ONE delete + recreate implementation.

LadybugDB cannot update an indexed embedding in place, so every content change
is a DETACH DELETE + CREATE. Store dedup-merge, update, and dream merge used
to carry three private copies of that dance, and they drifted: dream zeroed
access_count where the other two preserved it. These tests pin the resolved
semantics per call site so the next drift fails loudly instead of silently.

access_count resolution: store/update carry the existing count through
(same node identity, same usage history); dream merge SUMS the members'
counts (the survivor represents all of their usage).
"""

import os
import sys
import time

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/recreate-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server.WORKSPACE = "/recreate-test"
    yield
    server._conn = None
    server._db = None


def _set_access_count(mid, n):
    server.get_conn().execute(
        "MATCH (m:Memory {id: $id}) SET m.access_count = $n;", {"id": mid, "n": n})


def _get_row(mid):
    r = server.get_conn().execute(
        """MATCH (m:Memory {id: $id})
           RETURN m.access_count, m.created_at, m.content;""", {"id": mid})
    return r.get_next() if r.has_next() else None


def test_update_content_preserves_access_count_and_created_at():
    r = server.memory_store.__wrapped__(content="The batch window opens at 02:00 UTC.")
    _set_access_count(r["id"], 7)
    created_before = _get_row(r["id"])[1]

    server.memory_update.__wrapped__(memory_id=r["id"],
                                     content="The batch window opens at 03:00 UTC.")
    row = _get_row(r["id"])
    assert row[0] == 7, "content update must not reset usage history"
    assert row[1] == created_before, "created_at is provenance; recreate must carry it"


def test_dedup_merge_preserves_access_count():
    r = server.memory_store.__wrapped__(
        content="Deploy pipeline gate: canary must run 30 minutes before promote.",
        tags=["deploy", "canary"])
    _set_access_count(r["id"], 5)

    # Similar enough to dedup-merge, different enough to take the re-embed
    # (recreate) branch rather than the in-place tag absorb.
    merged = server.memory_store.__wrapped__(
        content="Deploy pipeline gate: the canary must run for 30 minutes before "
                "promote, and the gate blocks manual overrides.",
        tags=["deploy", "canary"])
    assert merged["status"] == "updated_existing", \
        f"fixture drifted: expected a dedup merge, got {merged['status']}"
    assert _get_row(merged["id"])[0] == 5, \
        "dedup-merge recreate must not reset usage history"


def test_dream_merge_sums_member_access_counts():
    """The path that used to zero the count. The survivor represents every
    member's usage, so it carries their sum."""
    triplet = [
        "Runbook R-2200: quiesce writers, snapshot, rotate credentials, resume.",
        "Runbook R-2200: quiesce writers, snapshot, rotate credentials, resume now.",
        "Runbook R-2200: quiesce writers, snapshot, then rotate credentials, resume.",
    ]
    ids, prev = [], None
    for c in triplet:  # around store-dedup: supersedes chain, then unlink
        kw = {"supersedes": prev} if prev is not None else {}
        r = server.memory_store.__wrapped__(content=c, tags=["runbook", "r2200"], **kw)
        ids.append(r["id"])
        prev = r["id"]
        time.sleep(0.01)
    for i in range(1, len(ids)):
        server.memory_unrelate.__wrapped__(from_id=ids[i], to_id=ids[i - 1],
                                           relationship="SUPERSEDES")
    for mid, n in zip(ids, (2, 3, 4)):
        _set_access_count(mid, n)

    out = server.memory_dream.__wrapped__(force=True)
    assert out["auto_merged"] == 2, f"fixture drifted: {out}"
    rows = server._collect_results(server.get_conn().execute(
        "MATCH (m:Memory) RETURN m.id, m.access_count;"))
    assert len(rows) == 1
    assert rows[0][1] == 9, \
        f"survivor should carry the members' summed usage (2+3+4), got {rows[0][1]}"


def test_edges_survive_all_three_recreate_paths():
    anchor = server.memory_store.__wrapped__(
        content="Decision record: the ledger runs on Postgres 16.")["id"]

    # Path 1: update
    a = server.memory_store.__wrapped__(
        content="The reconciliation job depends on the ledger decision.")["id"]
    server.memory_relate.__wrapped__(from_id=a, to_id=anchor, relationship="EXPLAINS")
    server.memory_update.__wrapped__(
        memory_id=a, content="The nightly reconciliation job depends on the ledger decision.")
    edges = server._collect_results(server.get_conn().execute(
        "MATCH (x:Memory {id: $a})-[:EXPLAINS]->(y:Memory {id: $b}) RETURN x.id;",
        {"a": a, "b": anchor}))
    assert edges, "EXPLAINS edge lost across the update recreate"

    # Path 2: dedup-merge (edge saved from the surviving match node)
    b = server.memory_store.__wrapped__(
        content="Failover drill cadence: the ledger team runs one drill per quarter.",
        tags=["ledger", "drill"])["id"]
    server.memory_relate.__wrapped__(from_id=b, to_id=anchor, relationship="RELATED_TO")
    merged = server.memory_store.__wrapped__(
        content="Failover drill cadence: the ledger team runs one full drill per "
                "quarter, alternating regions.",
        tags=["ledger", "drill"])
    assert merged["status"] == "updated_existing", f"fixture drifted: {merged}"
    edges = server._collect_results(server.get_conn().execute(
        "MATCH (x:Memory {id: $a})-[:RELATED_TO]->(y:Memory {id: $b}) RETURN x.id;",
        {"a": merged["id"], "b": anchor}))
    assert edges, "RELATED_TO edge lost across the dedup-merge recreate"
