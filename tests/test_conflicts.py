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
    """Disabling both merge gates restores pure-similarity merging.

    RUNBOOK_A/B are blocked twice over: different subjects (tags) and different
    named values (Zephyr vs Titan), so bypassing the pair requires switching off
    both gates.
    """
    monkeypatch.setattr(server, "MERGE_TAG_OVERLAP", 0.0)
    monkeypatch.setattr(server, "MERGE_VALUE_GATE", False)
    server.memory_store.__wrapped__(content=RUNBOOK_A, tags=["zephyr", "runbook"])
    b = server.memory_store.__wrapped__(content=RUNBOOK_B, tags=["titan", "runbook"])
    assert b["status"] == "updated_existing", "gates should be bypassable"


def test_value_gate_blocks_independently_of_subject_gate(monkeypatch):
    """The value gate alone is enough to stop a destructive merge.

    With the subject gate wide open, differing named values must still keep
    both memories — the two gates are independent protections.
    """
    monkeypatch.setattr(server, "MERGE_TAG_OVERLAP", 0.0)
    server.memory_store.__wrapped__(content=RUNBOOK_A, tags=["zephyr", "runbook"])
    b = server.memory_store.__wrapped__(content=RUNBOOK_B, tags=["titan", "runbook"])
    assert b["status"] == "stored_new", "value gate must block on its own"


# ---------------------------------------------------------------------------
# Same-subject contradictions must not be merged.
#
# The subject gate above stops two DIFFERENT subjects merging. It cannot help
# here, because a contradiction is same-subject by construction: identical tags,
# near-identical wording, one differing value. Observed on 0.13.1:
#
#   store "Vega request timeout is 500 milliseconds."  -> stored_new id=372
#   store "Vega request timeout is 900 milliseconds."  -> updated_existing 0.929
#   memory_get(372) -> "...500 milliseconds."   <- STALE value kept
#
# Three compounding faults: the older value wins (the two strings are the same
# length, so the `keep = longer` tie-break falls through to the existing text),
# the call reports success, and potential_conflicts can never fire afterwards
# because one of the two facts no longer exists.
# ---------------------------------------------------------------------------

TIMEOUT_500 = "Vega request timeout is 500 milliseconds."
TIMEOUT_900 = "Vega request timeout is 900 milliseconds."
VEGA_TAGS = ["vega", "config"]

# Genuine restatement of one fact: differing wording, identical values.
TTL_A = "The Vega service caches sessions in Redis with a 30 minute TTL."
TTL_B = "The Vega service caches user sessions in Redis using a 30-minute TTL."
TTL_TAGS = ["vega", "redis"]


@pytest.mark.parametrize(
    "a,b,conflict",
    [
        # Numeric drift: the highest-risk shape, since only a digit changes and
        # similarity is therefore maximal.
        (TIMEOUT_500, TIMEOUT_900, True),
        ("Ledger-db is decommissioned in Q2 2026.",
         "Ledger-db is decommissioned in Q3 2026.", True),
        ("Checkout runs Java 17.", "Checkout runs Java 21.", True),
        # Enum-ish and named values, no numbers involved.
        ("Atlas stores its event log in Kafka.",
         "Atlas stores its event log in Kinesis.", True),
        ("Zephyr rounds currency with HALF_UP.",
         "Zephyr rounds currency with HALF_EVEN.", True),
        # Lowercase service identifiers have neither a capital nor a digit, so
        # they need their own token class.
        ("Deployed via Apollo to prod-checkout.",
         "Deployed via Apollo to prod-inventory.", True),
        ("Checkout depends on payments-core.",
         "Checkout depends on inventory-cache.", True),
        # Formatting-only differences must NOT read as conflicts.
        ("Vega timeout is 500ms.", "Vega timeout is 500 ms.", False),
        ("Throughput is 12,000 msg/sec.", "Throughput is 12000 msg/sec.", False),
        ("Uses Redis for cache.", "uses redis for cache.", False),
        (TTL_A, TTL_B, False),
        # A superset elaborates rather than contradicts; the merge keeps the
        # longer text, so no value is lost.
        ("Retry policy is 5 attempts.",
         "Retry policy is 5 attempts with exponential backoff.", False),
        ("Retry policy is 5 attempts.",
         "Retry policy is 5 attempts, backoff with jitter and a 10s cap.", False),
    ],
)
def test_values_conflict_boundaries(a, b, conflict):
    assert server._values_conflict(a, b) is conflict


def test_values_conflict_is_symmetric():
    assert server._values_conflict(TIMEOUT_500, TIMEOUT_900) is True
    assert server._values_conflict(TIMEOUT_900, TIMEOUT_500) is True


def test_contradicting_value_is_not_merged_at_store_time():
    """The reported bug: both facts must survive the second store."""
    a = server.memory_store.__wrapped__(content=TIMEOUT_500, tags=VEGA_TAGS)
    b = server.memory_store.__wrapped__(content=TIMEOUT_900, tags=VEGA_TAGS)

    assert a["status"] == "stored_new"
    assert b["status"] == "stored_new", \
        "a differing value must never be absorbed into an existing memory"
    assert a["id"] != b["id"]

    kept = server.memory_get.__wrapped__(memory_id=a["id"])
    assert "500" in kept["content"], "original fact must be untouched"
    fresh = server.memory_get.__wrapped__(memory_id=b["id"])
    assert "900" in fresh["content"], "new fact must be stored verbatim"


def test_store_reports_the_competing_memory():
    """The caller learns about the conflict at write time, not only at search."""
    a = server.memory_store.__wrapped__(content=TIMEOUT_500, tags=VEGA_TAGS)
    b = server.memory_store.__wrapped__(content=TIMEOUT_900, tags=VEGA_TAGS)

    assert b["potential_conflict_with"] == a["id"]
    assert b["conflict_similarity"] >= server.DEDUP_THRESHOLD
    assert "supersedes" in b["hint"].lower(), \
        "the hint should name the remedy, not just the problem"


def test_both_facts_survive_so_search_can_flag_them():
    """Detection and destruction were in direct conflict; destruction ran first.

    With the merge refused, the pair stays intact and potential_conflicts —
    which needs both sides to exist — can finally fire on it.
    """
    a = server.memory_store.__wrapped__(content=TIMEOUT_500, tags=VEGA_TAGS)
    b = server.memory_store.__wrapped__(content=TIMEOUT_900, tags=VEGA_TAGS)

    out = server.memory_search.__wrapped__(
        query="what is the Vega request timeout", top_k=4
    )
    ids = {r["id"] for r in out["results"]}
    assert {a["id"], b["id"]} <= ids, "both values should be retrievable"

    pairs = [set(c["ids"]) for c in out.get("potential_conflicts", [])]
    assert {a["id"], b["id"]} in pairs


def test_paraphrase_of_one_fact_still_merges():
    """Regression guard: the gate must not break ordinary dedup."""
    server.memory_store.__wrapped__(content=TTL_A, tags=TTL_TAGS)
    b = server.memory_store.__wrapped__(content=TTL_B, tags=TTL_TAGS)
    assert b["status"] == "updated_existing", \
        "same values, different wording is a duplicate and should merge"


def test_elaboration_merges_and_keeps_the_richer_text():
    short = "Retry policy is 5 attempts."
    long = "Retry policy is 5 attempts with exponential backoff."
    server.memory_store.__wrapped__(content=short, tags=["retry", "policy"])
    b = server.memory_store.__wrapped__(content=long, tags=["retry", "policy"])

    assert b["status"] == "updated_existing"
    assert "backoff" in server.memory_get.__wrapped__(memory_id=b["id"])["content"]


def test_supersedes_still_bypasses_dedup_for_a_known_correction():
    a = server.memory_store.__wrapped__(content=TIMEOUT_500, tags=VEGA_TAGS)
    b = server.memory_store.__wrapped__(
        content=TIMEOUT_900, tags=VEGA_TAGS, supersedes=a["id"]
    )
    assert b["status"] == "stored_new"
    assert b["supersedes"] == a["id"]
    # An explicit correction is already resolved, so there is nothing to flag.
    assert "potential_conflict_with" not in b


def test_dream_does_not_auto_merge_contradicting_values():
    """Auto-merge at >=0.95 is unreviewed, so it needs the same gate."""
    a = server.memory_store.__wrapped__(content=TIMEOUT_500, tags=VEGA_TAGS)
    b = server.memory_store.__wrapped__(content=TIMEOUT_900, tags=VEGA_TAGS)

    out = server.memory_dream.__wrapped__(force=True)
    assert out.get("protected_by_value_conflict", 0) >= 1, \
        "dream should report pairs it declined to merge on value grounds"

    for mid, expect in ((a["id"], "500"), (b["id"], "900")):
        got = server.memory_get.__wrapped__(memory_id=mid)
        assert "error" not in got, f"memory {mid} was consumed by auto-merge"
        assert expect in got["content"]


def test_elaboration_carve_out_requires_the_richer_text_to_survive():
    """A shorter-but-value-richer text must not be absorbed into a longer one.

    Both merge paths keep the LONGER string. So "extra values are harmless" only
    holds while the value-richer side is also the longer side; otherwise those
    extra values would be silently dropped.
    """
    terse_rich = "Retry: 5 attempts, 10s cap, 3 hosts."
    verbose_poor = (
        "The retry policy for the ingestion pipeline is configured with a "
        "total of 5 attempts before the request is finally abandoned."
    )
    assert len(verbose_poor) > len(terse_rich)

    # The terse text carries values (10, 3) the verbose one lacks, and it is the
    # shorter of the two, so merging would discard them.
    assert server._values_conflict(terse_rich, verbose_poor) is True
    assert server._values_conflict(verbose_poor, terse_rich) is True


def test_elaboration_merges_when_the_longer_text_is_the_richer_one():
    short_poor = "Retry policy is 5 attempts."
    long_rich = "Retry policy is 5 attempts, backoff with jitter and a 10s cap."
    assert server._values_conflict(long_rich, short_poor) is False
    assert server._values_conflict(short_poor, long_rich) is False


# ---------------------------------------------------------------------------
# The middle gap: contradictions too differently worded to look like duplicates.
#
# Cosine-only detection has a floor problem. A correction REWRITTEN rather than
# edited scores low, so nothing flagged it — and the stale version outranked the
# current one because the query's wording matched it better. Measured:
#
#   "Izar retains audit logs for 30 days."                     -> 0.7748  #1
#   "Correction: retention on Izar was extended to a full year." -> 0.4533  #2
#   potential_conflicts: none
#
# Perversely, the more thoroughly an agent rewords a correction, the less likely
# cosine was to notice it. The fix uses signals already computed: same subject
# (tag Jaccard) plus a disagreeing value token, at any cosine above a low floor.
#
# Separately, write-time warning used to be a side effect of the dedup branch,
# so it only fired at >=0.92 while read-time detection reached 0.85. A pair at
# 0.8925 returned a clean stored_new and the agent found out only if it later
# searched that topic — after the cheap moment to fix it had passed.
# ---------------------------------------------------------------------------

IZAR_OLD = "Izar retains audit logs for 30 days."
IZAR_NEW = "Correction: retention on Izar was extended to a full year."
IZAR_TAGS = ["izar", "retention"]
IZAR_QUERY = "how long does Izar retain audit logs"


def test_reworded_contradiction_is_flagged_despite_low_similarity():
    a = server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)

    out = server.memory_search.__wrapped__(query=IZAR_QUERY, top_k=3)
    pairs = {frozenset(c["ids"]): c for c in out.get("potential_conflicts", [])}
    key = frozenset({a["id"], b["id"]})
    assert key in pairs, "a same-subject value contradiction must be flagged at any similarity"
    entry = pairs[key]
    assert entry["reason"] == "value_disagreement"
    assert entry["similarity"] < server.CONFLICT_THRESHOLD, \
        "this pair is below the near-duplicate threshold — that is the point"


def test_value_disagreement_hint_names_the_differing_tokens():
    server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    out = server.memory_search.__wrapped__(query=IZAR_QUERY, top_k=3)
    hint = out["potential_conflicts"][0]["hint"]
    assert "30" in hint, "the hint should show what actually differs"
    assert "supersedes" in hint.lower()


def test_parallel_facts_do_not_trigger_value_disagreement():
    """False-positive guard: same shape, DIFFERENT subjects, no conflict."""
    for text, tg in (
        ("The Checkout service is written in Java 17.", ["checkout", "lang"]),
        ("The Billing service is written in Java 17.", ["billing", "lang"]),
        ("The Inventory service is written in Java 17.", ["inventory", "lang"]),
        ("The Search service is written in Java 17.", ["search", "lang"]),
    ):
        server.memory_store.__wrapped__(content=text, tags=tg)

    out = server.memory_search.__wrapped__(
        query="what language are the services written in", top_k=4)
    assert not out.get("potential_conflicts"), \
        "distinct subjects must not be flagged, however alike they read"


def test_resolving_a_reworded_contradiction_silences_it():
    a = server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    server.memory_relate.__wrapped__(from_id=b["id"], to_id=a["id"],
                                     relationship="SUPERSEDES")

    out = server.memory_search.__wrapped__(query=IZAR_QUERY, top_k=3)
    assert not out.get("potential_conflicts"), "the edge resolves the conflict"

    by_id = {r["id"]: r for r in out["results"]}
    assert by_id[a["id"]].get("superseded") is True, \
        "the replaced version must be marked so it is not presented as current"

    # Whether the demotion actually FLIPS the order depends on the score margin
    # and the fusion mode (normalized min-max widens gaps, so a 0.5 penalty
    # cannot always overtake a large lead). Dropping superseded results is the
    # mode-independent guarantee.
    strict = server.memory_search.__wrapped__(
        query=IZAR_QUERY, top_k=3, include_superseded=False)
    assert strict["results"][0]["id"] == b["id"], \
        "with superseded results excluded, the current version must lead"


def test_write_time_warning_covers_the_sub_dedup_band():
    """A contradiction between the value floor and the dedup threshold must warn
    at write time, when one memory_relate call still fixes it cheaply."""
    a = server.memory_store.__wrapped__(
        content="The Alkaid queue depth limit is 500 messages.", tags=["alkaidq", "config"])
    b = server.memory_store.__wrapped__(
        content="The Alkaid queue depth limit is 800 messages.", tags=["alkaidq", "config"])

    assert b["status"] == "stored_new"
    assert b["potential_conflict_with"] == a["id"]
    assert server.CONFLICT_VALUE_FLOOR <= b["conflict_similarity"] < server.DEDUP_THRESHOLD
    assert "supersedes" in b["hint"].lower()


def test_write_time_warning_wording_matches_the_band():
    """Below the dedup threshold nothing was at risk of being merged, so the
    hint must not claim the memories were near-identical."""
    server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    assert b.get("potential_conflict_with") is not None
    assert "near-identical" not in b["hint"]


def test_unrelated_memories_sharing_tags_are_not_flagged():
    """The cosine floor still excludes pairs that share tags by accident."""
    server.memory_store.__wrapped__(
        content="The Izar service was commissioned in 2019 by the platform team.",
        tags=IZAR_TAGS)
    server.memory_store.__wrapped__(
        content="Quarterly capacity review meetings happen on the first Tuesday.",
        tags=IZAR_TAGS)
    out = server.memory_search.__wrapped__(query="Izar service history", top_k=3)
    for c in out.get("potential_conflicts", []):
        assert c["similarity"] >= server.CONFLICT_VALUE_FLOOR


# ---------------------------------------------------------------------------
# value_disagreement false-positived on complementary facts, with no way out.
#
# Reported against 0.19.0, two cases organic rather than constructed:
#
#   "Rigel depends on Redis for caching" / "... Kafka for event delivery"   0.7966
#   "Vela exposes 8080 for HTTP" / "... 9090 for metrics"                   0.7804
#   "Helios checkout depends on payments-core" / "... inventory-cache"      0.8235
#   "Helios SEARCH depends on query-index" / "Helios CHECKOUT depends on
#    payments-core"                                                        0.7297
#
# All four are true simultaneously. The last is worse than a false positive on
# the value: the two memories are about different subjects and coincide only on
# tags, which is tag-Jaccard being too coarse a subject proxy for one-to-many
# relations. "X depends on A" / "X depends on B" is complementary by nature, and
# that shape is the norm in a dependency or architecture graph — the primary
# AI-memory shape — so the flag was firing on most same-entity pairs.
#
# And there was no exit. The hint said "if both hold, link them with
# memory_relate"; doing exactly that left the flag firing on every subsequent
# search, because only SUPERSEDES cleared it — which would have been factually
# false and would have demoted a true fact out of results.
#
# Fix: below the near-duplicate threshold, require a CORRECTION MARKER. A
# correction announces itself ("Correction:", "now uses", "extended to", "no
# longer"); complementary facts never do. And any agent-asserted edge dismisses
# the flag, RELATED_TO included.
# ---------------------------------------------------------------------------

COMPLEMENTARY = [
    ("Rigel Redis/Kafka",
     "The Rigel service depends on Redis for caching.",
     "The Rigel service depends on Kafka for event delivery.",
     ["rigel", "dependency"], "what does Rigel depend on"),
    ("Vela ports",
     "The Vela service exposes port 8080 for HTTP traffic.",
     "The Vela service exposes port 9090 for metrics.",
     ["vela", "ports"], "what ports does Vela expose"),
    ("Helios two deps",
     "Helios checkout depends on payments-core for authorisation.",
     "Helios checkout depends on inventory-cache for stock lookups.",
     ["helios", "dependency"], "what does Helios checkout depend on"),
    ("different subjects, shared tags",
     "The Helios search feature depends on the query-index service.",
     "Helios checkout depends on the payments-core service.",
     ["helios", "dependency"], "Helios dependencies"),
]


def _flag_for(a_id, b_id, query):
    out = server.memory_search.__wrapped__(query=query, top_k=4)
    for c in out.get("potential_conflicts", []):
        if set(c["ids"]) == {a_id, b_id}:
            return c
    return None


@pytest.mark.parametrize("label,text_a,text_b,tags,query", COMPLEMENTARY,
                         ids=[c[0] for c in COMPLEMENTARY])
def test_complementary_facts_are_not_flagged(label, text_a, text_b, tags, query):
    a = server.memory_store.__wrapped__(content=text_a, tags=tags)
    b = server.memory_store.__wrapped__(content=text_b, tags=tags)
    assert _flag_for(a["id"], b["id"], query) is None, \
        f"{label}: both facts are true; flagging them makes a compliant agent hedge"


def test_complementary_facts_do_not_warn_at_write_time_either():
    server.memory_store.__wrapped__(content="The Rigel service depends on Redis for caching.",
                                    tags=["rigel", "dependency"])
    r = server.memory_store.__wrapped__(
        content="The Rigel service depends on Kafka for event delivery.",
        tags=["rigel", "dependency"])
    assert "potential_conflict_with" not in r


CORRECTIONS = [
    ("explicit Correction: prefix", IZAR_OLD, IZAR_NEW, IZAR_TAGS, IZAR_QUERY),
    ("'now uses' phrasing",
     "Zephyr rounds currency with HALF_UP.",
     "Zephyr now uses HALF_EVEN for currency rounding.",
     ["zephyr", "rounding"], "how does Zephyr round currency"),
    ("'no longer' phrasing",
     "The Mizar service writes audit events to S3.",
     "Mizar no longer writes audit events to S3; they go to CloudWatch.",
     ["mizar", "audit"], "where does Mizar write audit events"),
]


@pytest.mark.parametrize("label,text_a,text_b,tags,query", CORRECTIONS,
                         ids=[c[0] for c in CORRECTIONS])
def test_corrections_are_still_flagged(label, text_a, text_b, tags, query):
    """The middle gap must stay closed: these are the cases nothing else catches."""
    a = server.memory_store.__wrapped__(content=text_a, tags=tags)
    b = server.memory_store.__wrapped__(content=text_b, tags=tags)
    assert _flag_for(a["id"], b["id"], query) is not None, f"{label} was missed"


def test_related_to_dismisses_the_flag_permanently():
    """The documented remedy has to actually work, or the hint is a dead end."""
    a = server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    assert _flag_for(a["id"], b["id"], IZAR_QUERY) is not None

    rel = server.memory_relate.__wrapped__(from_id=b["id"], to_id=a["id"],
                                           relationship="RELATED_TO")
    assert rel["status"] == "created"

    assert _flag_for(a["id"], b["id"], IZAR_QUERY) is None
    # And it stays dismissed on later searches.
    assert _flag_for(a["id"], b["id"], IZAR_QUERY) is None
    # Both facts still rank — RELATED_TO must not demote anything.
    ids = {r["id"] for r in
           server.memory_search.__wrapped__(query=IZAR_QUERY, top_k=4)["results"]}
    assert {a["id"], b["id"]} <= ids


def test_inferred_related_to_does_not_dismiss():
    """_compute_graph_scores auto-creates RELATED_TO from shared topics, so an
    INFERRED edge asserts nothing and must not silence a real conflict."""
    a = server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    server.memory_relate.__wrapped__(from_id=b["id"], to_id=a["id"],
                                     relationship="RELATED_TO", provenance="INFERRED")
    assert _flag_for(a["id"], b["id"], IZAR_QUERY) is not None


def test_hint_names_related_to_as_the_dismissal():
    a = server.memory_store.__wrapped__(content=IZAR_OLD, tags=IZAR_TAGS)
    b = server.memory_store.__wrapped__(content=IZAR_NEW, tags=IZAR_TAGS)
    hint = _flag_for(a["id"], b["id"], IZAR_QUERY)["hint"]
    assert "RELATED_TO" in hint
    assert "supersedes" in hint.lower()


def test_correction_marker_helper():
    assert server._has_correction_marker("Correction: the TTL is now 300s") is True
    assert server._has_correction_marker("Zephyr now uses HALF_EVEN") is True
    assert server._has_correction_marker("retention was extended to a year") is True
    assert server._has_correction_marker("Mizar no longer writes to S3") is True
    assert server._has_correction_marker("This supersedes the earlier value") is True
    # Complementary phrasings carry no marker.
    assert server._has_correction_marker("Rigel depends on Redis for caching") is False
    assert server._has_correction_marker("Vela exposes port 8080 for HTTP") is False
    assert server._has_correction_marker("Helios checkout depends on payments-core") is False
