"""Graph neighbours reported alongside search results.

Why they are separate from `results` rather than fused into the ranking:
raw cosine similarity has a high floor (~0.5 even for unrelated text), so
every similarity candidate carries a large constant contribution while a
graph-only hit caps at the 0.15 graph weight. Measured: a linked but
semantically distant memory scored ~0.2 against a rank-5 cutoff of ~0.52, so
it could never place. Raising the graph weight enough to compete would let
loosely-linked memories displace direct answers — so ranking stays
similarity-driven and the graph is surfaced explicitly instead.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/graph-related-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

DECISION = "Decision: the Checkout service writes orders to DynamoDB with on-demand capacity."
# Deliberately shares almost no vocabulary with a query about the decision
CONSEQUENCE = "Postmortem INC-9931: the flash sale melted a downstream shard replica overnight."
NOISE = [
    "The Search service indexes documents into OpenSearch nightly.",
    "The mobile app uses Kotlin Multiplatform for shared business logic.",
    "Feature flags refresh from a cached ruleset every 60 seconds.",
    "The auth module validates tokens against Cognito user pools.",
]
QUERY = "why does checkout use on-demand capacity for orders"


@pytest.fixture
def store():
    server._conn = None
    server._db = None
    res = server.memory_store.__wrapped__(items=[
        {"content": c} for c in [DECISION, CONSEQUENCE] + NOISE])
    ids = [r["id"] for r in res["results"]]
    yield {"decision": ids[0], "consequence": ids[1]}
    server._conn = None
    server._db = None


def _search(top_k=3, **kw):
    return server.memory_search.__wrapped__(query=QUERY, top_k=top_k, **kw)


def test_no_related_field_without_edges(store):
    out = _search()
    assert "related" not in out, "an edge-free graph should report no neighbours"


def test_related_surfaces_a_linked_memory(store):
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO", confidence=0.9)

    out = _search()
    related_ids = {r["id"] for r in out.get("related", [])}
    assert store["consequence"] in related_ids, \
        "a linked memory should surface even when similarity ranks it low"


def test_related_names_its_anchor(store):
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")

    entry = next(r for r in _search()["related"] if r["id"] == store["consequence"])
    assert entry["linked_to"] == store["decision"], \
        "linked_to should identify which result the neighbour hangs off"


def test_related_never_duplicates_results(store):
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")

    out = _search(top_k=10)  # wide enough that both are ranked
    result_ids = {r["id"] for r in out["results"]}
    related_ids = {r["id"] for r in out.get("related", [])}
    assert not (result_ids & related_ids), \
        "a memory ranked on its own merits must not also appear as related"


def test_ranking_is_unaffected_by_edges(store):
    """Adding an edge must not reorder or rescore the ranked results."""
    before = _search()["results"]
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")
    after = _search()["results"]

    assert [r["id"] for r in before] == [r["id"] for r in after], "order changed"
    assert [r["score"] for r in before] == [r["score"] for r in after], "scores changed"


def test_expansion_can_be_disabled(store, monkeypatch):
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")
    monkeypatch.setattr(server, "GRAPH_EXPAND_SEEDS", 0)
    assert "related" not in _search()


def test_related_respects_its_limit(store, monkeypatch):
    """Many neighbours must not flood the response."""
    extra = server.memory_store.__wrapped__(items=[
        {"content": f"Follow-up note {i}: an unrelated operational detail about "
                    f"queue depth {i}."} for i in range(6)])
    relations = [{"from_id": r["id"], "to_id": store["decision"],
                  "relationship": "RELATED_TO"} for r in extra["results"]]
    server.memory_relate.__wrapped__(relations=relations)

    monkeypatch.setattr(server, "GRAPH_EXPAND_LIMIT", 3)
    out = _search()
    assert len(out.get("related", [])) <= 3


def test_superseded_version_available_as_context(store):
    """After a correction outranks a stale fact, the stale one is still
    reachable as a neighbour — history stays visible without polluting the
    answer."""
    correction = server.memory_store.__wrapped__(
        content="Correction: Checkout now uses provisioned capacity with "
                "autoscaling, not on-demand.")["id"]
    server.memory_relate.__wrapped__(
        from_id=correction, to_id=store["decision"], relationship="SUPERSEDES")

    out = _search()
    top_ids = [r["id"] for r in out["results"]]
    assert correction in top_ids, "the correction should rank"
    # the superseded original is either demoted in results or offered as related
    stale_visible = (store["decision"] in top_ids
                     or store["decision"] in {r["id"] for r in out.get("related", [])})
    assert stale_visible, "superseded memory should remain discoverable"


def test_weak_matches_do_not_seed_expansion(store):
    """A weak hit is a coincidence; its neighbours are noise.

    Observed in a 127-memory corpus: a pricing correction placed rank 3 at
    0.4185 on a billing-bug query (53% of the 0.7879 top score), which seeded
    expansion and pulled its superseded pair into `related` where it was
    irrelevant. Seeds must clear GRAPH_EXPAND_MIN_RATIO of the top score.
    """
    # Link something to a memory that will only ever be a weak match here
    weak_neighbour = server.memory_store.__wrapped__(
        content="Quarterly finance review scheduled for the fifteenth.")["id"]
    weak_target = next(r["id"] for r in server.memory_search.__wrapped__(
        query="cognito token validation", top_k=1)["results"])
    server.memory_relate.__wrapped__(from_id=weak_neighbour, to_id=weak_target,
                                     relationship="RELATED_TO")

    out = _search()  # a checkout query; the auth memory is at best a weak hit
    top = out["results"][0]["score"]
    floor = top * server.GRAPH_EXPAND_MIN_RATIO
    for r in out["results"]:
        if r["score"] < floor:
            assert r["id"] != weak_target or weak_neighbour not in {
                x["id"] for x in out.get("related", [])
            }, "expanded from a hit below the relevance floor"


def test_strong_match_still_seeds_expansion(store):
    """The floor must not block expansion from a confident top hit."""
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")

    out = _search()
    assert out["results"][0]["id"] == store["decision"]
    assert out["results"][0]["score"] >= (
        out["results"][0]["score"] * server.GRAPH_EXPAND_MIN_RATIO)
    assert store["consequence"] in {r["id"] for r in out.get("related", [])}, \
        "a strong top hit must still expand"


def test_min_ratio_is_configurable(store, monkeypatch):
    server.memory_relate.__wrapped__(
        from_id=store["consequence"], to_id=store["decision"],
        relationship="RELATED_TO")

    # A ratio above 1.0 makes even the top result fail the floor
    monkeypatch.setattr(server, "GRAPH_EXPAND_MIN_RATIO", 1.5)
    assert "related" not in _search(), "no seed should clear an impossible floor"


def test_rrf_seed_floor_reads_raw_cosine(monkeypatch):
    """Rank fusion compresses fused scores: rank 2 lands at ~0.96x the top
    regardless of how weak the underlying match is, which would let junk
    seeds sail over GRAPH_EXPAND_MIN_RATIO and neuter the guard the previous
    test exists for. In rrf mode the floor must read the raw pre-rank
    cosines snapshotted before the transform.
    """
    server._conn = None
    server._db = None
    monkeypatch.setattr(server, "FUSION_MODE", "rrf")
    monkeypatch.setattr(server, "GRAPH_EXPAND_MIN_RATIO", 0.9)

    res = server.memory_store.__wrapped__(items=[
        {"content": DECISION},
        # Shares a token ("orders") so it places in BOTH channels — the
        # shape where rank compression makes fused scores incomparable.
        {"content": "Order fulfillment reports are archived quarterly."},
        {"content": "Parking garage badge readers were replaced in March."},
    ])
    a, b, c = [r["id"] for r in res["results"]]
    server.memory_relate.__wrapped__(from_id=c, to_id=b,
                                     relationship="RELATED_TO")

    out = _search(top_k=2)
    assert out["results"][0]["id"] == a

    # The premise: rank compression lifts the junk rank-2 hit over a 0.9
    # FUSED-score floor, so a fused test cannot see its weakness.
    assert out["results"][1]["score"] >= out["results"][0]["score"] * 0.9, \
        "fixture drifted: rank 2 no longer demonstrates rank compression"

    # The fix: its raw cosine does NOT clear 0.9x the top's cosine, so it
    # must not seed expansion — its neighbour stays out of `related`.
    ranked = {r["id"] for r in out["results"]}
    related = {r["id"] for r in out.get("related", [])}
    assert not (({b, c} - ranked) & related), \
        "expanded from a junk seed that only rank compression made look strong"


# --- neighbour selection must not depend on insertion order ------------------
#
# Third site of one defect class. `id` was the sole tiebreak in neighbour
# selection, and ids encode insertion order, so which neighbour won the last
# capped slot depended on the order memories were stored. The same mistake
# was already found and fixed twice: in the dream survivor choice, and in the
# rrf rank transform. This site has the most reach, because `related` decides
# whether the answer to a transitive question appears at all.
#
# The rule that prevents a fourth instance: id may appear only as the terminal
# element of an explicitly documented ordering key, never as an incidental
# one. Here the full key is: hop-decayed relevance, fewer hops, importance,
# recency, id.

_ANCHOR = "Decision: the Perseus gateway routes checkout traffic through the edge tier."
_RIVALS = [
    # Two neighbours that are equally (ir)relevant to the anchor query and sit
    # at the same hop, differing only in importance.
    ("Ownership note: the Perseus gateway is owned by the platform group.", 2),
    ("Ownership note: the Perseus gateway is owned by the traffic group.", 5),
]
_FILLER = [
    "The Perseus gateway emits request metrics to the telemetry bus.",
    "The Perseus gateway holds a 30 second idle timeout.",
    "The Perseus gateway terminates TLS at the edge tier.",
    "The Perseus gateway rate limits by API key.",
]


def _related_run(rival_order, monkeypatch):
    server._conn = None
    server._db = None
    server.get_conn()
    anchor = server.memory_store.__wrapped__(content=_ANCHOR, tags=["perseus"])["id"]
    # Fill every slot but one so the two rivals compete for the last.
    monkeypatch.setattr(server, "GRAPH_EXPAND_LIMIT", len(_FILLER) + 1)
    for text in _FILLER:
        fid = server.memory_store.__wrapped__(content=text, tags=["perseus"])["id"]
        server.memory_relate.__wrapped__(from_id=fid, to_id=anchor,
                                         relationship="RELATED_TO")
    ids = {}
    for i in rival_order:
        text, imp = _RIVALS[i]
        rid = server.memory_store.__wrapped__(content=text, tags=["perseus"],
                                             importance=imp)["id"]
        server.memory_relate.__wrapped__(from_id=rid, to_id=anchor,
                                        relationship="RELATED_TO")
        ids[i] = rid
    out = server.memory_search.__wrapped__(
        query="how does the Perseus gateway route checkout traffic", top_k=1)
    related = {r["id"] for r in out.get("related", [])}
    return {i: (ids[i] in related) for i in ids}


def test_neighbour_selection_is_independent_of_insertion_order(monkeypatch):
    """NOTE on the strength of this test: it passes with the old id-only
    tiebreak too, because forcing an exact relevance tie needs two candidates
    with identical embeddings — i.e. near-identical text, which dedup merges
    before it can reach this code. So the defect here is real by inspection
    (id was the terminal key with nothing between it and relevance) but its
    exposure is narrow, unlike the rrf transform where BM25 ties across a
    template made it routine. This stands as a guard against a future change
    that widens the exposure, not as a reproduction of the original."""
    forward = _related_run([0, 1], monkeypatch)
    reverse = _related_run([1, 0], monkeypatch)
    server._conn = None
    server._db = None

    assert forward == reverse, (
        f"which neighbour won the capped slot changed with insertion order: "
        f"forward={forward} reverse={reverse}"
    )
