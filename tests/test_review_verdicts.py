"""memory_keep_separate: making an agent's "these are distinct" verdict durable.

Dream offers `leave_separate` as a resolution and nothing recorded it, so the
judgement evaporated: the same cluster was re-offered on the next run and the
same pair was re-flagged on the next search. 0.25.0 reduced that noise by
INFERRING permanent verdicts from the merge gates; this records the verdict an
agent actually made, which is the case the gates cannot infer — two facts that
read alike, share a subject, and genuinely both hold.

Mechanism borrowed from the Procedural Graph paper's refiner, which keeps
rejected edits on file so the same edit is not proposed twice. Only the
mechanism transfers: it needs neither an LLM nor labelled data, unlike that
paper's held-out validation gate.

Stored in a side table rather than as a new edge type on purpose. A new edge
would have to be threaded through the EDGE_TYPES allowlist, the save/restore
pair that carries edges across delete+recreate, the graph projections,
memory_get and memory_unrelate — and an unthreaded copy is exactly how the
access_count drift happened.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/verdict-test")
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


PAIR = ("The Nunki service request timeout is 500ms.",
        "The Nunki service request timeout is 900ms.")
QUERY = "Nunki service request timeout"


def _unlinked_pair():
    """A same-subject look-alike pair with no edge between them."""
    a = server.memory_store.__wrapped__(content=PAIR[0],
                                        tags=["nunki", "timeout"])["id"]
    b = server.memory_store.__wrapped__(content=PAIR[1], tags=["nunki", "timeout"],
                                        supersedes=a)["id"]
    server.memory_unrelate.__wrapped__(from_id=b, to_id=a,
                                       relationship="SUPERSEDES")
    server.memory_store.__wrapped__(items=[
        {"content": f"Filler {i} on unrelated capacity planning {i}.",
         "tags": [f"vf{i}"]} for i in range(24)])
    return a, b


def _conflict_flagged():
    out = server.memory_search.__wrapped__(query=QUERY, top_k=5)
    return bool(out.get("potential_conflicts"))


def _in_review(a, b):
    res = server.memory_dream.__wrapped__(force=True, dry_run=True)
    for c in res.get("clusters_for_review") or []:
        ids = {c["anchor"]["id"]} | {m["id"] for m in c["similar"]}
        if {a, b} <= ids:
            return True
    return False


def test_unreviewed_pair_is_surfaced_on_both_surfaces():
    """The premise. Without this the suppression tests would be vacuous."""
    a, b = _unlinked_pair()
    assert _conflict_flagged(), "an unreviewed look-alike pair should be flagged"
    assert _in_review(a, b), "and should be offered for review"


def test_verdict_silences_the_conflict_flag_and_the_review_cluster():
    a, b = _unlinked_pair()
    res = server.memory_keep_separate.__wrapped__(memory_ids=[a, b])
    assert res["status"] == "recorded"
    assert res["pairs_recorded"] == [[min(a, b), max(a, b)]]

    assert not _conflict_flagged(), "a ruled-on pair must not be re-flagged"
    assert not _in_review(a, b), "and must not be re-offered for review"


def test_verdict_creates_no_edge():
    """The distinction from memory_relate: 'separate and I checked', not
    'connected'. An edge would feed traversal and centrality."""
    a, b = _unlinked_pair()
    server.memory_keep_separate.__wrapped__(memory_ids=[a, b])

    edges = server._collect_results(server.get_conn().execute(
        "MATCH (x:Memory)-[r]->(y:Memory) WHERE x.id IN [$a,$b] AND y.id IN [$a,$b] "
        "RETURN label(r);", {"a": a, "b": b}))
    assert not edges, f"no edge may be created between the pair, got {edges}"


def test_recording_twice_is_idempotent():
    a, b = _unlinked_pair()
    server.memory_keep_separate.__wrapped__(memory_ids=[a, b])
    again = server.memory_keep_separate.__wrapped__(memory_ids=[a, b])
    assert again["status"] == "unchanged"
    assert again["pairs_already_recorded"] == [[min(a, b), max(a, b)]]


def test_more_than_two_ids_records_every_pair():
    a, b = _unlinked_pair()
    c = server.memory_store.__wrapped__(
        content="The Nunki service connect timeout is 300ms.",
        tags=["nunki", "timeout"])["id"]
    res = server.memory_keep_separate.__wrapped__(memory_ids=[a, b, c])
    assert len(res["pairs_recorded"]) == 3, "all three pairs among three ids"


def test_verdicts_are_reaped_when_a_member_is_deleted():
    """A side table nobody collects grows without bound — the Topic lesson."""
    a, b = _unlinked_pair()
    server.memory_keep_separate.__wrapped__(memory_ids=[a, b])
    conn = server.get_conn()
    before = server._collect_results(conn.execute(
        f"MATCH (v:{server.REVIEW_VERDICT_TABLE}) RETURN COUNT(v);"))[0][0]
    assert before == 1

    server.memory_delete.__wrapped__(memory_id=b)
    after = server._collect_results(conn.execute(
        f"MATCH (v:{server.REVIEW_VERDICT_TABLE}) RETURN COUNT(v);"))[0][0]
    assert after == 0, "a verdict whose memory is gone must be collected"


def test_bad_input_is_rejected_clearly():
    a, _b = _unlinked_pair()
    assert server.memory_keep_separate.__wrapped__(
        memory_ids=[a])["status"] == "error", "one id is not a pair"
    assert server.memory_keep_separate.__wrapped__(
        memory_ids=[])["status"] == "error"
    res = server.memory_keep_separate.__wrapped__(memory_ids=[a, 999999])
    assert res["status"] == "error"
    assert res["not_found"] == [999999]
