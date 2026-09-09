"""Supersession-aware ranking.

The failure this fixes (reproduced from a community scale test): a query's
wording matches a stale fact better than its own correction, so plain
similarity ranks the outdated answer first.

    query      "how does the pricing service round monetary amounts"
    rank 1     "The Pricing service rounds ... using HALF_UP"      (superseded)
    rank 2     "Correction: ... now rounds using HALF_EVEN"        (current)

Verified before the fix: legacy scored 0.7598 vs 0.5753, and
MEMORY_FUSION=normalized made it worse (0.8250 vs 0.5442). Only the
SUPERSEDES edge carries the information, so search must use it.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/supersession-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

STALE = "The Pricing service rounds monetary amounts to 2 decimal places using HALF_UP."
CURRENT = "Correction: the Pricing service now rounds using HALF_EVEN for audit compliance."
QUERY = "how does the pricing service round monetary amounts"

DISTRACTORS = [
    "The Pricing service uses SQS as its primary datastore.",
    "The Pricing service is written in Java 17 and deployed via Apollo.",
    "The Pricing service on-call rotation is owned by the team pricing-core.",
    "The Billing service emits metrics to the billing-metrics namespace.",
]


@pytest.fixture
def store():
    server._conn = None
    server._db = None
    res = server.memory_store.__wrapped__(items=[
        {"content": c} for c in [STALE, CURRENT] + DISTRACTORS
    ])
    ids = [r["id"] for r in res["results"]]
    yield {"stale": ids[0], "current": ids[1]}
    server._conn = None
    server._db = None


def _search(top_k=10, **kw):
    """top_k=10 by default so demoted memories stay observable — with a 0.5
    penalty a superseded memory can fall below unrelated distractors, which is
    correct behaviour but hides it from a narrow window."""
    res = server.memory_search.__wrapped__(query=QUERY, top_k=top_k, **kw)
    return res["results"]


def test_stale_answer_wins_without_an_edge(store):
    """Baseline: with no SUPERSEDES edge, similarity favours the stale fact.
    This documents WHY the edge is needed — it is not a bug in the fusion."""
    rows = _search()
    assert rows[0]["id"] == store["stale"], (
        "expected the stale fact to win on raw similarity; if this changes, "
        "the supersession test below is no longer meaningful"
    )
    assert not any(r.get("superseded") for r in rows)


def test_supersedes_edge_promotes_the_current_answer(store):
    server.memory_relate.__wrapped__(
        from_id=store["current"], to_id=store["stale"],
        relationship="SUPERSEDES")

    rows = _search()
    assert rows[0]["id"] == store["current"], \
        f"current answer should rank first, got id={rows[0]['id']}"


def test_superseded_results_are_flagged(store):
    server.memory_relate.__wrapped__(
        from_id=store["current"], to_id=store["stale"],
        relationship="SUPERSEDES")

    rows = _search()
    stale = next(r for r in rows if r["id"] == store["stale"])
    current = next(r for r in rows if r["id"] == store["current"])
    assert stale.get("superseded") is True, "stale memory must be marked"
    assert "superseded" not in current, "current memory must not be marked"


def test_superseded_memories_remain_retrievable(store):
    """Demotion, not deletion: history stays available for auditing.

    Note it may rank below unrelated memories, so a narrow top_k can exclude
    it — that is intended. What matters is that it is still reachable.
    """
    server.memory_relate.__wrapped__(
        from_id=store["current"], to_id=store["stale"],
        relationship="SUPERSEDES")
    assert any(r["id"] == store["stale"] for r in _search(top_k=10))


def test_include_superseded_false_excludes_them(store):
    server.memory_relate.__wrapped__(
        from_id=store["current"], to_id=store["stale"],
        relationship="SUPERSEDES")

    rows = _search(include_superseded=False)
    assert all(r["id"] != store["stale"] for r in rows), "stale must be dropped"
    assert rows[0]["id"] == store["current"]


def test_penalty_is_configurable(store, monkeypatch):
    server.memory_relate.__wrapped__(
        from_id=store["current"], to_id=store["stale"],
        relationship="SUPERSEDES")

    # A penalty of 1.0 disables demotion, restoring similarity-only ordering
    monkeypatch.setattr(server, "SUPERSEDED_PENALTY", 1.0)
    rows = _search()
    assert rows[0]["id"] == store["stale"], "penalty 1.0 should disable demotion"
    # ...but the flag must still be present so callers can react
    assert rows[0].get("superseded") is True


def test_chained_supersession_surfaces_the_newest(store):
    """A -> B -> C: only C is current, both A and B are superseded."""
    third = server.memory_store.__wrapped__(
        content="Final: the Pricing service rounds using HALF_EVEN with 4 decimal "
                "places for FX conversions.")["id"]
    server.memory_relate.__wrapped__(relations=[
        {"from_id": store["current"], "to_id": store["stale"],
         "relationship": "SUPERSEDES"},
        {"from_id": third, "to_id": store["current"],
         "relationship": "SUPERSEDES"},
    ])

    rows = _search()
    flagged = {r["id"] for r in rows if r.get("superseded")}
    assert store["stale"] in flagged
    assert store["current"] in flagged, "middle of the chain is also superseded"
    assert third not in flagged, "newest must not be flagged"


# --- Correction chains must survive dedup ------------------------------------
#
# Field report: storing three consecutive retry-policy versions in one batch
# merged v2 into v3 at similarity 0.9284, above the 0.92 dedup threshold, and
# silently collapsed a 3-version chain into 2. A correction is textually
# near-identical to what it corrects by construction, so the mechanism that
# prevents duplicate accumulation was destroying the history the graph exists
# to record. Passing supersedes= disables dedup for that store and wires the
# edge in the same call.

V1 = "The Aurora ingestion service retry policy is 3 attempts with a fixed 500ms delay."
V2 = "Correction: the Aurora ingestion retry policy changed to 5 attempts with exponential backoff."
V3 = ("Correction: the Aurora ingestion retry policy is finalized at 5 attempts, exponential "
      "backoff with full jitter and a 10s cap, adopted after incident INC-5501.")


@pytest.fixture
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _count(conn):
    return conn.execute("MATCH (m:Memory) RETURN COUNT(m);").get_next()[0]


def test_dedup_collapses_a_chain_without_supersedes(clean):
    """Documents the hazard: near-identical corrections merge by default."""
    server.memory_store.__wrapped__(items=[{"content": c} for c in (V1, V2, V3)])
    assert _count(server.get_conn()) < 3, (
        "expected dedup to merge the near-identical corrections; if this "
        "changes, the supersedes= guard may no longer be necessary"
    )


def test_supersedes_preserves_every_version(clean):
    r1 = server.memory_store.__wrapped__(content=V1)
    r2 = server.memory_store.__wrapped__(content=V2, supersedes=r1["id"])
    r3 = server.memory_store.__wrapped__(content=V3, supersedes=r2["id"])

    assert {r1["status"], r2["status"], r3["status"]} == {"stored_new"}
    assert _count(server.get_conn()) == 3, "all three versions must survive"
    assert r2["supersedes"] == r1["id"]
    assert r3["supersedes"] == r2["id"]


def test_supersedes_creates_the_edge(clean):
    r1 = server.memory_store.__wrapped__(content=V1)
    r2 = server.memory_store.__wrapped__(content=V2, supersedes=r1["id"])

    rows = server._collect_results(server.get_conn().execute(
        "MATCH (a:Memory)-[:SUPERSEDES]->(b:Memory) RETURN a.id, b.id;"))
    assert [r2["id"], r1["id"]] in [list(r) for r in rows], \
        "storing with supersedes= must create the edge in the same call"


def test_full_lineage_is_walkable(clean):
    r1 = server.memory_store.__wrapped__(content=V1)
    r2 = server.memory_store.__wrapped__(content=V2, supersedes=r1["id"])
    r3 = server.memory_store.__wrapped__(content=V3, supersedes=r2["id"])

    rows = server._collect_results(server.get_conn().execute(
        "MATCH (c:Memory)-[:SUPERSEDES*]->(o:Memory) WHERE c.id = $i RETURN o.id;",
        {"i": r3["id"]}))
    assert {r[0] for r in rows} == {r1["id"], r2["id"]}, \
        "the newest version should reach every prior one"


def test_current_version_ranks_first_over_a_chain(clean):
    r1 = server.memory_store.__wrapped__(content=V1)
    r2 = server.memory_store.__wrapped__(content=V2, supersedes=r1["id"])
    r3 = server.memory_store.__wrapped__(content=V3, supersedes=r2["id"])

    rows = server.memory_search.__wrapped__(
        query="what is the current retry policy for Aurora ingestion",
        top_k=5)["results"]
    assert rows[0]["id"] == r3["id"], f"current version should rank first, got {rows[0]['id']}"
    older = {r["id"] for r in rows if r.get("superseded")}
    assert {r1["id"], r2["id"]} <= older, "both prior versions must be flagged superseded"


def test_supersedes_works_in_batch_mode(clean):
    first = server.memory_store.__wrapped__(content=V1)
    res = server.memory_store.__wrapped__(items=[
        {"content": V2, "supersedes": first["id"]},
        {"content": "Aurora ingestion adopted a dead-letter queue for exhausted retries."},
    ])
    statuses = [r["status"] for r in res["results"]]
    assert statuses == ["stored_new", "stored_new"]
    assert res["results"][0]["supersedes"] == first["id"]
    assert _count(server.get_conn()) == 3


def test_merge_response_reports_similarity(clean):
    """When dedup does merge, the caller should be able to see why."""
    server.memory_store.__wrapped__(content=V2)
    res = server.memory_store.__wrapped__(content=V3)
    if res["status"] == "updated_existing":
        assert "similarity" in res, "a merge should report the similarity that caused it"
        assert res["similarity"] >= server.DEDUP_THRESHOLD


# --- circular SUPERSEDES ------------------------------------------------------
#
# A cycle means no memory is the current version. Each individual behaviour is
# defensible and the combination was a hole: every member is superseded so the
# penalty cannot discriminate and the OLDEST value can rank first; the
# documented current-answer query returns zero rows, which reads as "no
# information" rather than "contradictory information"; and the one component
# that knows — dream's SCC pass — was gated behind `not dry_run`, so an
# operator inspecting safely was told there were no contradictions.

_CYCLE = [
    "The Sadr ingest pipeline is throttled to 100 rps.",
    "Correction: the Sadr ingest pipeline is throttled to 250 rps.",
    "Correction: the Sadr ingest pipeline is throttled to 400 rps.",
]


def _make_cycle():
    ids, prev = [], None
    for v in _CYCLE:
        kw = {"supersedes": prev} if prev else {}
        ids.append(server.memory_store.__wrapped__(
            content=v, tags=["sadr", "throttle"], **kw)["id"])
        prev = ids[-1]
    # Close the loop: the oldest supersedes the newest.
    server.memory_relate.__wrapped__(from_id=ids[0], to_id=ids[2],
                                     relationship="SUPERSEDES")
    server.memory_store.__wrapped__(items=[
        {"content": f"Filler {i} on unrelated capacity planning {i}.",
         "tags": [f"cyc-f{i}"]} for i in range(24)])
    return ids


def test_dream_reports_a_cycle_identically_in_dry_run_and_for_real():
    """A diagnostic must not read clean on a state that is not."""
    _make_cycle()
    dry = server.memory_dream.__wrapped__(force=True, dry_run=True)
    wet = server.memory_dream.__wrapped__(force=True)

    assert dry.get("contradictions"), \
        "dry_run reported no contradictions on a graph that has one"
    assert [c["memory_ids"] for c in dry["contradictions"]] == \
           [c["memory_ids"] for c in wet["contradictions"]], \
        "preview and real runs must agree about contradictions"
    assert "unrelate" in dry["contradictions"][0]["resolution"].lower()


def test_search_tells_the_agent_the_supersession_data_is_circular():
    ids = _make_cycle()
    # Passes in every fusion mode since cycle members are exempt from the
    # supersession penalty: the multiplier used to sink them below unrelated
    # fillers (guaranteed under rrf's compressed spread, and reachable under
    # legacy whenever the margin over the noise floor was under 2x).
    out = server.memory_search.__wrapped__(query="what is the Sadr ingest throttle",
                                           top_k=30)
    cyc = out.get("supersession_cycle")
    assert cyc, "a cycle must be surfaced where the caller is already looking"
    assert set(cyc["memory_ids"]) == set(ids)
    assert "arbitrary" in cyc["issue"], \
        "the notice must say the ranking cannot be trusted, not merely that it is stale"


def test_breaking_the_cycle_clears_the_flag_and_restores_a_head():
    ids = _make_cycle()
    server.memory_unrelate.__wrapped__(from_id=ids[0], to_id=ids[2],
                                       relationship="SUPERSEDES")
    out = server.memory_search.__wrapped__(query="what is the Sadr ingest throttle",
                                           top_k=30)
    assert "supersession_cycle" not in out, "the flag must clear once repaired"
    ranked = [r["id"] for r in out["results"]]
    members = [m for m in ranked if m in ids]
    assert members and members[0] == ids[2], \
        "with the loop broken the newest version must outrank the older ones"


def test_a_healthy_chain_is_never_flagged_as_circular():
    """No false positives on the ordinary correction chain this feature protects."""
    ids, prev = [], None
    for v in _CYCLE:
        kw = {"supersedes": prev} if prev else {}
        ids.append(server.memory_store.__wrapped__(
            content=v, tags=["sadr", "throttle"], **kw)["id"])
        prev = ids[-1]

    out = server.memory_search.__wrapped__(query="what is the Sadr ingest throttle",
                                           top_k=30)
    assert "supersession_cycle" not in out
    ranked = [r["id"] for r in out["results"]]
    members = [m for m in ranked if m in ids]
    assert members and members[0] == ids[2], \
        "the newest version must outrank the older ones in a healthy chain"


def test_cycle_warning_does_not_attach_to_unrelated_queries():
    """The warning must be scoped to the RETURNED rows, not to every scored
    candidate. It was originally keyed off the superseded set derived from
    final_scores — which holds the whole candidate pool (100), i.e. the entire
    corpus on any smaller workspace — so one unresolved cycle anywhere
    attached the warning to every search. Self-defeating for a warning whose
    value is its rarity, and the same permanent-verdict noise the review
    clusters were fixed for."""
    _make_cycle()
    for text, tg in (
        ("The Selene platform is owned by the infrastructure guild.",
         ["selene", "ownership"]),
        ("Selene architecture uses an event-sourced ledger.",
         ["selene", "architecture"]),
    ):
        server.memory_store.__wrapped__(content=text, tags=tg)

    on_topic = server.memory_search.__wrapped__(
        query="what is the Sadr ingest throttle", top_k=30)
    off_topic = server.memory_search.__wrapped__(
        query="Selene platform architecture and ownership", top_k=2)

    assert on_topic.get("supersession_cycle"), \
        "a query that returns cycle members must still warn"
    assert "supersession_cycle" not in off_topic, \
        "a query returning no cycle member must not carry the warning"


def test_cycle_members_rank_on_relevance_not_penalised_into_invisibility():
    """The composite failure this exemption fixes.

    Measured in the field: for "what is the Rukbat cache size" the answering
    memory scored vector 0.8569 / fts 1.0 — an unpenalised 0.769 — and the
    x0.5 multiplier dropped it to 0.3845, below three unrelated memories at
    ~0.41. At top_k=2 it was not returned at all, and because the cycle
    warning is (correctly) scoped to returned rows, the warning disappeared
    with it. The caller asking exactly the affected question received
    unrelated memories and no indication anything was wrong.

    In a cycle the superseded flag says nothing about which member is stale,
    so the penalty is pure damage. Rank on relevance and flag the cycle.
    """
    V = ["The Rukbat cache size is 40 GB.",
         "Correction: the Rukbat cache size is 80 GB.",
         "Correction: the Rukbat cache size is 120 GB."]
    ids, prev = [], None
    for v in V:
        kw = {"supersedes": prev} if prev else {}
        ids.append(server.memory_store.__wrapped__(
            content=v, tags=["rukbat", "cache"], **kw)["id"])
        prev = ids[-1]
    server.memory_relate.__wrapped__(from_id=ids[0], to_id=ids[2],
                                     relationship="SUPERSEDES")
    for i, txt in enumerate([
        "INC-9100: a Selene cache stampede caused elevated latency for 40 minutes.",
        "Manifest checksums are verified on every cache warm start.",
        "The Vega cache is partitioned by tenant id.",
        "Cache eviction uses LRU across all services.",
    ]):
        server.memory_store.__wrapped__(content=txt, tags=[f"cyc-noise{i}"])

    # The narrow window is the point: this is where the answer used to vanish.
    out = server.memory_search.__wrapped__(query="what is the Rukbat cache size",
                                           top_k=2)
    returned = [r["id"] for r in out["results"]]
    assert any(m in ids for m in returned), \
        f"the answering cycle member must be reachable at top_k=2, got {returned}"
    assert out.get("supersession_cycle"), \
        "and the caller must be told the supersession data is unreliable"


def test_ordinary_superseded_memories_are_still_penalised():
    """The exemption is for cycles only — a normal correction chain must still
    demote its stale versions, which is the whole point of the feature."""
    old = server.memory_store.__wrapped__(
        content="The Izar retry budget is 3 attempts.", tags=["izar", "retry"])
    new = server.memory_store.__wrapped__(
        content="Correction: the Izar retry budget is 7 attempts.",
        tags=["izar", "retry"], supersedes=old["id"])

    out = server.memory_search.__wrapped__(query="what is the Izar retry budget",
                                           top_k=5)
    ranked = [r["id"] for r in out["results"]]
    assert new["id"] in ranked, "the current version must be retrievable"
    if old["id"] in ranked:
        assert ranked.index(new["id"]) < ranked.index(old["id"]), \
            "the current version must outrank the superseded one"
    # Under rrf the multiplier can demote the stale version clear out of the
    # window — a documented rrf limitation (see README), and still a correct
    # demotion. What must NOT happen is the stale version outranking.
    assert "supersession_cycle" not in out
