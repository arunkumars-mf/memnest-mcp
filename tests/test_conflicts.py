"""Unresolved-conflict awareness and consolidation safety.

Both behaviours come from an AI-memory stress test:

1. Two contradictory facts stored in different sessions with no SUPERSEDES
   edge ("Zephyr uses PostgreSQL" / "Zephyr uses DynamoDB") were returned at
   near-equal scores with nothing indicating they conflict. dream's
   `contradictions` field only detects circular SUPERSEDES chains (via SCC),
   which is a much narrower thing.

2. Worse, dream's merge path keyed purely on textual similarity, so a
   supersedes=-protected correction chain appeared as a merge candidate — the
   consolidation step could cannibalise the history the edge exists to record.

Conflicts are reported as "potential": telling a genuine contradiction from two
complementary facts needs entailment, which would mean an LLM call in the
server. Flagging the ambiguity is free; resolving it is the agent's job.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/conflict-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

PG = "The Zephyr service uses PostgreSQL 15 as its primary datastore."
DDB = "The Zephyr service uses DynamoDB as its primary datastore."
TAGS = ["zephyr", "datastore"]
QUERY = "what datastore does the Zephyr service use"


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def test_contradiction_without_an_edge_is_flagged():
    a = server.memory_store.__wrapped__(content=PG, tags=TAGS)
    b = server.memory_store.__wrapped__(content=DDB, tags=TAGS)

    out = server.memory_search.__wrapped__(query=QUERY, top_k=4)
    pairs = [set(c["ids"]) for c in out.get("potential_conflicts", [])]
    assert {a["id"], b["id"]} in pairs, \
        "two near-identical facts with no SUPERSEDES edge should be flagged"


def test_conflict_entry_explains_the_remedy():
    server.memory_store.__wrapped__(content=PG, tags=TAGS)
    server.memory_store.__wrapped__(content=DDB, tags=TAGS)

    conflict = server.memory_search.__wrapped__(
        query=QUERY, top_k=4)["potential_conflicts"][0]
    assert "similarity" in conflict
    assert "supersedes" in conflict["hint"], "hint should name the fix"


def test_resolved_pair_is_not_flagged():
    """Once an edge records which version is current, it is no longer a conflict."""
    a = server.memory_store.__wrapped__(content=PG, tags=TAGS)
    b = server.memory_store.__wrapped__(content=DDB, tags=TAGS, supersedes=a["id"])

    out = server.memory_search.__wrapped__(query=QUERY, top_k=4)
    pairs = [set(c["ids"]) for c in out.get("potential_conflicts", [])]
    assert {a["id"], b["id"]} not in pairs, "an edge-resolved pair must not be flagged"


def test_parallel_facts_about_different_subjects_are_not_flagged():
    """Templated facts read alike without conflicting. Tag overlap is the
    subject proxy that prevents flagging them."""
    server.memory_store.__wrapped__(items=[
        {"content": "The Checkout service is written in Java 17 and deployed via Apollo.",
         "tags": ["checkout", "deploy"]},
        {"content": "The Billing service is written in Java 17 and deployed via Apollo.",
         "tags": ["billing", "deploy"]},
        {"content": "The Search service is written in Java 17 and deployed via Apollo.",
         "tags": ["search", "deploy"]},
    ])
    out = server.memory_search.__wrapped__(
        query="which services are written in Java 17", top_k=5)
    assert not out.get("potential_conflicts"), \
        f"parallel facts flagged as conflicts: {out.get('potential_conflicts')}"


def test_detection_can_be_disabled(monkeypatch):
    server.memory_store.__wrapped__(content=PG, tags=TAGS)
    server.memory_store.__wrapped__(content=DDB, tags=TAGS)
    monkeypatch.setattr(server, "CONFLICT_DETECTION", False)
    assert "potential_conflicts" not in server.memory_search.__wrapped__(
        query=QUERY, top_k=4)


# --- Consolidation safety ----------------------------------------------------

V2 = "Correction: the Aurora ingestion retry policy changed to 5 attempts with exponential backoff."
V3 = ("Correction: the Aurora ingestion retry policy is finalized at 5 attempts, exponential "
      "backoff with full jitter and a 10s cap, adopted after incident INC-5501.")


def test_semantically_linked_detects_supersedes():
    a = server.memory_store.__wrapped__(content=V2)
    b = server.memory_store.__wrapped__(content=V3, supersedes=a["id"])
    conn = server.get_conn()
    assert server._semantically_linked(conn, b["id"], a["id"]), "edge not detected"
    assert server._semantically_linked(conn, a["id"], b["id"]), "should be direction-agnostic"


def test_unlinked_pair_is_not_reported_as_linked():
    a = server.memory_store.__wrapped__(content=PG, tags=TAGS)
    b = server.memory_store.__wrapped__(content="Unrelated note about queue depth.")
    assert not server._semantically_linked(server.get_conn(), a["id"], b["id"])


def test_dream_never_merges_a_protected_chain():
    """The regression that matters: consolidation must not undo supersedes=."""
    a = server.memory_store.__wrapped__(content=V2)
    b = server.memory_store.__wrapped__(content=V3, supersedes=a["id"])
    # Enough memories for dream to act on
    server.memory_store.__wrapped__(items=[
        {"content": f"Operational note {i}: queue depth alarm tuning detail {i}."}
        for i in range(25)])

    conn = server.get_conn()
    before = conn.execute("MATCH (m:Memory) RETURN COUNT(m);").get_next()[0]
    res = server.memory_dream.__wrapped__(force=True)
    after = conn.execute("MATCH (m:Memory) RETURN COUNT(m);").get_next()[0]

    assert res.get("protected_by_edges", 0) >= 1, \
        "dream should report skipping the linked pair"
    assert after >= before, "no memory should have been merged away"
    edge = server._collect_results(conn.execute(
        "MATCH (x:Memory)-[:SUPERSEDES]->(y:Memory) WHERE x.id=$a AND y.id=$b RETURN x.id;",
        {"a": b["id"], "b": a["id"]}))
    assert edge, "the SUPERSEDES edge must survive consolidation"


def test_dream_does_not_offer_linked_pairs_for_review():
    """A linked pair is resolved history, not a duplicate to review — surfacing
    it invites an agent to merge a correction chain away."""
    a = server.memory_store.__wrapped__(content=V2)
    b = server.memory_store.__wrapped__(content=V3, supersedes=a["id"])
    server.memory_store.__wrapped__(items=[
        {"content": f"Operational note {i}: unrelated tuning detail {i}."}
        for i in range(25)])

    res = server.memory_dream.__wrapped__(force=True, dry_run=True)
    for cluster in res.get("clusters_for_review") or []:
        members = {m["id"] for m in cluster.get("similar", [])}
        pair = {cluster["anchor"]["id"]} | members
        assert not {a["id"], b["id"]} <= pair, \
            f"the protected chain was offered for review: {cluster}"


def test_review_clusters_state_the_available_resolutions():
    """Clusters are not always duplicates — they may be competing versions."""
    server.memory_store.__wrapped__(items=[
        {"content": "Cache TTL for the Titan service is set to 60 seconds."},
        {"content": "Cache TTL for the Titan service is configured at 65 seconds."},
    ] + [{"content": f"Filler note {i} about unrelated infrastructure {i}."}
         for i in range(25)])

    res = server.memory_dream.__wrapped__(force=True, dry_run=True)
    clusters = res.get("clusters_for_review") or []
    if clusters:
        assert "resolution" in clusters[0]
        assert "supersedes" in clusters[0]["resolution"]


# --- Subject gate on destructive merges --------------------------------------
#
# Store-time dedup (>=0.92) and dream auto-merge (>=0.95) are the two paths that
# destroy a memory without review. Neither checked whether the pair was even
# about the same thing. Measured on a 127-fact corpus of per-service templated
# facts: 30 memories were silently absorbed on ingest — runbooks and logging
# decisions for DIFFERENT services merged at 0.94-0.95, losing which service
# each described. With the gate, all 127 survive.
#
# Asymmetry is deliberate: a missed merge leaves a recoverable duplicate that
# the review band surfaces again; a wrong merge destroys a fact irreversibly.

RUNBOOK_A = "Runbook: to roll back the Zephyr service, redeploy the previous Apollo version."
RUNBOOK_B = "Runbook: to roll back the Titan service, redeploy the previous Apollo version."
DUP_A = "The Vega service caches sessions in Redis with a 30 minute TTL."
DUP_B = "The Vega service caches user sessions in Redis using a 30-minute TTL."


def test_same_subject_helper():
    assert server._same_subject(["zephyr", "cache"], ["zephyr", "cache"])
    assert server._same_subject(["zephyr", "cache"], ["zephyr"])          # 0.5
    assert not server._same_subject(["zephyr", "runbook"], ["titan", "runbook"])  # 0.33
    assert not server._same_subject(["a"], ["b"])
    # No tags on either side means no signal, so prior behaviour is preserved
    assert server._same_subject([], ["zephyr"])
    assert server._same_subject(["zephyr"], [])


def test_distinct_subjects_are_not_merged_on_store():
    a = server.memory_store.__wrapped__(content=RUNBOOK_A, tags=["zephyr", "runbook"])
    b = server.memory_store.__wrapped__(content=RUNBOOK_B, tags=["titan", "runbook"])
    assert b["status"] == "stored_new", \
        "near-identical runbooks for different services must both survive"
    assert a["id"] != b["id"]


def test_genuine_duplicates_still_merge():
    """The gate must not disable dedup for real duplicates."""
    server.memory_store.__wrapped__(content=DUP_A, tags=["vega", "cache"])
    b = server.memory_store.__wrapped__(content=DUP_B, tags=["vega", "cache"])
    assert b["status"] == "updated_existing", "same-subject duplicates should merge"


def test_untagged_memories_keep_prior_dedup_behaviour():
    """With no tags there is no subject signal, so similarity alone decides."""
    server.memory_store.__wrapped__(content=DUP_A)
    b = server.memory_store.__wrapped__(content=DUP_B)
    assert b["status"] == "updated_existing"


def test_dream_reports_subject_protected_pairs():
    server.memory_store.__wrapped__(items=[
        {"content": f"Runbook: to roll back the svc{i} service, redeploy the "
                    f"previous Apollo version.", "tags": [f"svc{i}", "runbook"]}
        for i in range(6)
    ] + [{"content": f"Filler note {i} about unrelated capacity planning {i}.",
          "tags": [f"filler{i}"]} for i in range(20)])

    res = server.memory_dream.__wrapped__(force=True, dry_run=True)
    assert res.get("protected_by_subject", 0) >= 1, \
        "dream should report pairs it declined to merge on subject grounds"


def test_merge_gate_can_be_disabled(monkeypatch):
    """Setting the overlap to 0 restores pure-similarity merging."""
    monkeypatch.setattr(server, "MERGE_TAG_OVERLAP", 0.0)
    server.memory_store.__wrapped__(content=RUNBOOK_A, tags=["zephyr", "runbook"])
    b = server.memory_store.__wrapped__(content=RUNBOOK_B, tags=["titan", "runbook"])
    assert b["status"] == "updated_existing", "gate should be bypassable"
