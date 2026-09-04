"""Edge lifecycle: memory_unrelate, and edges exposed on memory_get.

Found in a surface review. memory_relate only ever CREATEd, so there was no way
to remove an edge — and memory_query's DELETE is blocked by default, meaning a
mistaken SUPERSEDES was permanent unless the operator loosened server config.
That undercut the design: the skill tells agents edges are the highest-value
action, memory_store(supersedes=) makes them cheap to create, and dream reports
circular SUPERSEDES chains as `contradictions` with no supported remedy.

Separately, memory_get returned node properties only, so "what does this
replace?" required hand-written Cypher even though the graph is the point.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/edges-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _chain():
    """old <- new (SUPERSEDES), plus an incident that EXPLAINS the old one."""
    old = server.memory_store.__wrapped__(
        content="The Vega cache TTL is 60 seconds.", tags=["vega", "cache"], importance=2)
    new = server.memory_store.__wrapped__(
        content="Correction: the Vega cache TTL is 300 seconds.",
        tags=["vega", "cache"], importance=4, supersedes=old["id"])
    inc = server.memory_store.__wrapped__(
        content="INC-7700: Vega cache stampede during a deploy.", tags=["vega", "incident"])
    server.memory_relate.__wrapped__(from_id=inc["id"], to_id=old["id"],
                                     relationship="EXPLAINS")
    return old["id"], new["id"], inc["id"]


def _edge_count(rel):
    rows = server._collect_results(server.get_conn().execute(
        f"MATCH ()-[r:{rel}]->() RETURN COUNT(r);"))
    return rows[0][0] if rows else 0


# --- memory_get edges ------------------------------------------------------

def test_get_reports_supersession_without_cypher():
    old, new, inc = _chain()
    g = server.memory_get.__wrapped__(memory_id=old)
    assert g["superseded"] is True
    assert g["superseded_by"] == [new]
    assert g["edges"]["superseded_by"] == [new]
    assert g["edges"]["explained_by"][0]["id"] == inc


def test_get_reports_outgoing_edges():
    old, new, inc = _chain()
    g = server.memory_get.__wrapped__(memory_id=inc)
    assert g["edges"]["explains"][0]["id"] == old
    assert not g.get("superseded")


def test_get_can_skip_edges():
    old, _, _ = _chain()
    g = server.memory_get.__wrapped__(memory_id=old, include_edges=False)
    assert "edges" not in g
    assert g["content"]


def test_get_validates_its_id():
    assert server.memory_get.__wrapped__(memory_id="abc")["status"] == "error"
    assert server.memory_get.__wrapped__(memory_id=999999)["status"] == "not_found"


# --- memory_unrelate -------------------------------------------------------

def test_unrelate_removes_one_edge_type():
    old, new, inc = _chain()
    before = _edge_count("EXPLAINS")
    out = server.memory_unrelate.__wrapped__(from_id=inc, to_id=old, relationship="EXPLAINS")
    assert out["status"] == "deleted"
    assert _edge_count("EXPLAINS") == before - 1
    # Both memories survive; only the edge went.
    assert server.memory_get.__wrapped__(memory_id=old)["status"] == "found"
    assert server.memory_get.__wrapped__(memory_id=inc)["status"] == "found"


def test_unrelate_without_a_type_removes_every_edge_between_the_pair():
    old, new, inc = _chain()
    server.memory_relate.__wrapped__(from_id=inc, to_id=old, relationship="RELATED_TO")
    out = server.memory_unrelate.__wrapped__(from_id=inc, to_id=old)
    assert out["status"] == "deleted"
    assert len(out["removed"]) == 2
    assert not server.memory_get.__wrapped__(memory_id=inc).get("edges")


def test_unrelate_reports_not_found_rather_than_a_false_success():
    old, new, inc = _chain()
    out = server.memory_unrelate.__wrapped__(from_id=inc, to_id=new, relationship="RELATED_TO")
    assert out["status"] == "not_found"


def test_unrelate_breaks_a_circular_supersedes_chain():
    """dream reports these as `contradictions`; this is the supported remedy."""
    old, new, inc = _chain()
    server.memory_relate.__wrapped__(from_id=old, to_id=new, relationship="SUPERSEDES")
    assert _edge_count("SUPERSEDES") == 2  # cycle

    out = server.memory_unrelate.__wrapped__(from_id=old, to_id=new, relationship="SUPERSEDES")
    assert out["status"] == "deleted"
    assert _edge_count("SUPERSEDES") == 1  # the correct direction remains
    assert server.memory_get.__wrapped__(memory_id=old)["superseded_by"] == [new]


def test_unrelate_rejects_an_arbitrary_label():
    """The label is interpolated into Cypher, so the allowlist is load-bearing."""
    old, new, _ = _chain()
    out = server.memory_unrelate.__wrapped__(
        from_id=old, to_id=new, relationship="SUPERSEDES]->(x) DETACH DELETE x //")
    assert out["status"] == "error"
    assert server._count_memories(server.get_conn()) == 3


def test_unrelate_both_directions():
    old, new, inc = _chain()
    server.memory_relate.__wrapped__(from_id=old, to_id=inc, relationship="RELATED_TO")
    server.memory_relate.__wrapped__(from_id=inc, to_id=old, relationship="RELATED_TO")
    out = server.memory_unrelate.__wrapped__(
        from_id=old, to_id=inc, relationship="RELATED_TO", both_directions=True)
    assert len(out["removed"]) == 2


def test_unrelate_batch():
    old, new, inc = _chain()
    out = server.memory_unrelate.__wrapped__(relations=[
        {"from_id": new, "to_id": old, "relationship": "SUPERSEDES"},
        {"from_id": inc, "to_id": old, "relationship": "EXPLAINS"},
    ])
    assert out["count"] == 2
    assert all(r["status"] == "deleted" for r in out["results"])
    assert _edge_count("SUPERSEDES") == 0 and _edge_count("EXPLAINS") == 0


def test_relate_is_idempotent():
    """Re-asserting an edge must not create a parallel duplicate: dream has to
    defend against those, and re-importing an export used to double every edge."""
    old, new, inc = _chain()
    before = _edge_count("EXPLAINS")
    again = server.memory_relate.__wrapped__(from_id=inc, to_id=old, relationship="EXPLAINS")
    assert again["status"] == "exists"
    assert _edge_count("EXPLAINS") == before
