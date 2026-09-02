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
