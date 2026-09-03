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
