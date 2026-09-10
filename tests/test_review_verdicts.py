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


# --- the remedy must be named where the problem is reported -------------------
#
# 0.30.0 shipped the tool and left every hint pointing elsewhere. The read-time
# hint was the bad case: it named memory_relate(RELATED_TO) as the dismissal,
# which does dismiss the flag but MEANS "these are connected" and feeds
# traversal and centrality — so an agent following the advice on a both-hold
# pair creates graph structure it never intended. That is the 0.19.0
# closed-loop bug (a hint recommending an ineffective action) one step further:
# a hint recommending an action with unintended side effects.
#
# A resolution an agent cannot call is not a resolution.

VALUE_PAIR = ("Izar retains audit logs for 30 days.",
              "Retention on Izar was extended to a full year.")


def test_write_time_hint_names_the_tool():
    server.memory_store.__wrapped__(content=VALUE_PAIR[0], tags=["izar", "audit"])
    second = server.memory_store.__wrapped__(content=VALUE_PAIR[1],
                                             tags=["izar", "audit"])
    hint = second.get("hint", "")
    assert hint, "fixture drifted: the write-time conflict hint did not fire"
    assert "memory_keep_separate" in hint


def test_read_time_hint_names_the_tool_and_deprecates_the_edge_route():
    server.memory_store.__wrapped__(content=VALUE_PAIR[0], tags=["izar", "audit"])
    server.memory_store.__wrapped__(content=VALUE_PAIR[1], tags=["izar", "audit"])
    server.memory_store.__wrapped__(items=[
        {"content": f"Filler {i} on capacity planning.", "tags": [f"hf{i}"]}
        for i in range(6)])

    out = server.memory_search.__wrapped__(query="Izar audit log retention", top_k=5)
    conflicts = out.get("potential_conflicts") or []
    assert conflicts, "fixture drifted: no conflict flagged"
    hint = conflicts[0]["hint"]

    assert "memory_keep_separate" in hint, \
        "the both-hold branch must name the tool that records the verdict"
    assert "genuinely connected" in hint, \
        "RELATED_TO must survive only as a caveat, not as the recommended route"


def test_dream_resolution_names_callables_not_labels():
    a, b = _unlinked_pair()
    res = server.memory_dream.__wrapped__(force=True, dry_run=True)
    clusters = res.get("clusters_for_review") or []
    assert clusters, "fixture drifted: no review cluster offered"
    resolution = clusters[0]["resolution"]

    assert "memory_keep_separate" in resolution, \
        "leave_separate was a label with no callable behind it"
    assert "memory_delete" in resolution and "SUPERSEDES" in resolution, \
        "every branch of the resolution should name what to call"


# --- enumerate the emission sites, do not count them from a description -------
#
# 0.30.1 fixed four surfaces and missed a fifth. The write-time hint has TWO
# variants — near_duplicate (>= DEDUP_THRESHOLD) and value_disagreement (below
# it) — and the release description had already collapsed them into one
# "write-time conflict hint", so the count came out at four. The read-time pair
# was enumerated separately and both were fixed; the write-time pair was
# enumerated as one and only one was fixed.
#
# This is not rule 4 and not a vacuous test: three tests were added, each
# asserting its fixture fires first, and all three passed. The gap was an
# UNENUMERATED SURFACE — there was never a test to write. The tell would have
# been counting emission sites in the code rather than counting them from the
# description, which is where the merge happened.
#
# So this test drives every branch and asserts on the set, which fails on
# addition of a variant rather than requiring someone to notice it.

def _write_time_store(content_a, content_b, tags):
    server._conn = None
    server._db = None
    server.get_conn()
    server.memory_store.__wrapped__(content=content_a, tags=tags)
    return server.memory_store.__wrapped__(content=content_b, tags=tags)


PAIR_A = "The Nunki service request timeout is 500ms."
PAIR_B = "The Nunki service request timeout is 900ms."


def test_every_write_time_conflict_variant_names_the_tool(monkeypatch):
    """Both branches of the write-time hint, driven DETERMINISTICALLY.

    The branch is chosen by `conflict_similarity >= DEDUP_THRESHOLD`, and a
    text fixture lands wherever the embedding puts it — the first version of
    this test scored 0.9226 against a 0.92 threshold on one machine and below
    it on another, so it silently exercised one branch twice and passed with
    the other branch broken. Moving the threshold forces each branch, and the
    assertion on `conflict_similarity` proves which one ran.
    """
    results = {}

    monkeypatch.setattr(server, "DEDUP_THRESHOLD", 0.50)
    res = _write_time_store(PAIR_A, PAIR_B, ["nunki", "timeout"])
    assert res.get("conflict_similarity", 0) >= 0.50, "near_duplicate branch not taken"
    results["near_duplicate"] = res.get("hint", "")

    monkeypatch.setattr(server, "DEDUP_THRESHOLD", 0.999)
    res = _write_time_store(PAIR_A, PAIR_B, ["nunki", "timeout"])
    assert res.get("conflict_similarity", 1) < 0.999, \
        "value_disagreement branch not taken"
    results["value_disagreement"] = res.get("hint", "")

    for label, hint in results.items():
        assert hint, f"fixture drifted: the write-time {label} hint did not fire"
        assert "memory_keep_separate" in hint, \
            f"the write-time {label} hint does not name the tool: {hint!r}"


def test_no_conflict_hint_recommends_an_edge_as_the_dismissal():
    """The property that generalises across every surface: an agent told to
    resolve a both-hold pair must never be pointed at an edge-creating call as
    THE remedy. RELATED_TO may appear only conditionally, with its cost named."""
    hints = [
        _write_time_store(PAIR_A, PAIR_B, ["nunki", "timeout"]).get("hint", ""),
        _write_time_store("Izar retains audit logs for 30 days.",
                          "Retention on Izar was extended to a full year.",
                          ["izar", "audit"]).get("hint", ""),
    ]
    for hint in hints:
        if "RELATED_TO" in hint:
            assert "genuinely connected" in hint, \
                f"RELATED_TO offered without its condition: {hint!r}"
