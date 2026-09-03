"""Restating a memory must not promote it.

From an AI-memory regression run: re-encountering a fact across sessions is the
NORMAL case, but every restatement raised importance by 1 and reset updated_at.
Both feed the ranking (importance 0.05, recency 0.1), the bump was monotonic
with no decay, and the effect was measured end-to-end:

    fact "Helios checkout p99 latency is 850ms at peak load."  importance 2
      paraphrase re-store -> updated_existing -> importance 4
      paraphrase re-store -> updated_existing -> importance 5  (ceiling)

That mundane latency note then took a top slot on "What is the current Helios
architecture?" and displaced every architecture fact from the results.

The principle now: importance is caller-owned metadata the server does not
editorialise, and updated_at means "when the content last changed", not "when
somebody last mentioned it". Note dream's merge has always taken a plain max
without incrementing, so the store path was also internally inconsistent.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/importance-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

LATENCY = "Helios checkout p99 latency is 850ms at peak load."
LATENCY_PARA = "The Helios checkout service has a p99 latency of 850ms at peak load."
TAGS = ["helios", "latency"]


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _state(mid):
    conn = server.get_conn()
    row = server._collect_results(conn.execute(
        "MATCH (m:Memory {id: $id}) RETURN m.importance, m.updated_at;", {"id": mid}))[0]
    return {"importance": row[0], "updated_at": row[1]}


def test_identical_restatement_changes_nothing():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    before = _state(a["id"])

    for _ in range(3):
        r = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
        assert r["status"] == "already_exists"
        assert r["id"] == a["id"]

    after = _state(a["id"])
    assert after["importance"] == before["importance"] == 2, \
        "restating a fact must not raise its importance"
    assert after["updated_at"] == before["updated_at"], \
        "restating a fact must not reset its recency"


def test_paraphrase_merge_does_not_increment_importance():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    b = server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS, importance=2)
    assert b["status"] == "updated_existing"
    assert _state(a["id"])["importance"] == 2, \
        "a merge must not treat restatement as evidence of importance"


def test_repeated_merges_do_not_ratchet_to_the_ceiling():
    """The reported escalation: 2 -> 4 -> 5 across two restatements."""
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    for text in (
        LATENCY_PARA,
        "Helios checkout p99 latency is 850ms at peak load in production.",
        "Helios checkout has a p99 latency of 850ms under peak load.",
    ):
        server.memory_store.__wrapped__(content=text, tags=TAGS, importance=2)
    assert _state(a["id"])["importance"] <= 2, \
        "importance must not climb with the number of restatements"


def test_merge_keeps_the_higher_of_the_two_importances():
    """Not incrementing must not mean discarding a caller's higher value."""
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS, importance=5)
    assert _state(a["id"])["importance"] == 5, \
        "an explicit higher importance from the caller should win"


def test_merge_does_not_lower_importance():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=5)
    server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS, importance=1)
    assert _state(a["id"])["importance"] == 5, \
        "a low-importance restatement must not demote an important memory"


def test_store_path_matches_dream_merge_semantics():
    """Both destructive merge paths should agree on importance handling."""
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=3)
    server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS, importance=4)
    store_result = _state(a["id"])["importance"]
    # dream computes: min(5, max(importance or 3, other_imp or 3))
    assert store_result == min(5, max(3, 4)), \
        "store dedup should take a plain max, exactly like dream's merge"


def test_restatement_does_not_displace_an_unrelated_answer():
    """End-to-end version of the reported symptom."""
    arch = server.memory_store.__wrapped__(
        content="Helios consolidated back to a modular monolith in 2025 after the "
                "microservices split proved too costly.",
        tags=["helios", "architecture"], importance=3)
    lat = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    for text, tg in (
        ("Helios services are written in Java 17.", ["helios", "java"]),
        ("Helios deploys via Apollo to prod-checkout.", ["helios", "deploy"]),
        ("Selene remained on microservices after evaluating a monolith.",
         ["selene", "architecture"]),
    ):
        server.memory_store.__wrapped__(content=text, tags=tg)

    query = "What is the current Helios architecture?"

    def scores():
        out = server.memory_search.__wrapped__(query=query, top_k=5)
        return {r["id"]: r["score"] for r in out["results"]}

    before = scores()
    order_before = [r for r, _ in sorted(before.items(), key=lambda x: -x[1])]
    for _ in range(3):
        server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    after = scores()
    order_after = [r for r, _ in sorted(after.items(), key=lambda x: -x[1])]

    # The invariant under test is that restatement is inert. Whether the
    # latency note outranks the architecture fact to begin with is a separate
    # ranking-quality question and is deliberately not asserted here.
    assert before == after, \
        "restating an unrelated fact must not perturb this query's scores at all"
    assert order_before == order_after, \
        "restating an unrelated fact must not reorder results"
    assert _state(lat["id"])["importance"] == 2
    assert arch["id"] in after or arch["id"] not in before, \
        "the fix must not push the architecture fact out of the results"


def test_access_count_is_not_part_of_the_ranking_formula():
    """Guard against a retrieval feedback loop being introduced later.

    Being retrieved must not make a memory more retrievable, or ranking would
    drift toward whatever the agent happened to look at most.
    """
    a = server.memory_store.__wrapped__(
        content="Helios consolidated back to a modular monolith in 2025.",
        tags=["helios", "architecture"])
    for text, tg in (
        ("Helios services are written in Java 17.", ["helios", "java"]),
        ("Atlas stores its event log in Kinesis.", ["atlas", "eventlog"]),
        ("The Helios platform team is led by Dana Cruz.", ["helios", "ownership"]),
    ):
        server.memory_store.__wrapped__(content=text, tags=tg)

    query = "What is the current Helios architecture?"

    def scores():
        out = server.memory_search.__wrapped__(query=query, top_k=5)
        return {r["id"]: r["score"] for r in out["results"]}

    before = scores()
    conn = server.get_conn()
    conn.execute("MATCH (m:Memory {id: $i}) SET m.access_count = 9999;", {"i": a["id"]})
    assert scores() == before, "access_count must not influence the score"


# ---------------------------------------------------------------------------
# An omitted importance must not be read as the caller asserting 3.
#
# Follow-up to the fix above. Taking max(existing, incoming) is right, but the
# tool substituted its default of 3 before reaching the merge, so a memory
# deliberately stored at importance 2 was silently raised to 3 by any
# restatement that simply left the argument out. Bounded at the default rather
# than ratcheting to the ceiling, so not the original bug — but "importance is
# caller-owned" only holds if an unstated value stays unstated.
# ---------------------------------------------------------------------------


def test_restatement_omitting_importance_leaves_it_untouched():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    assert _state(a["id"])["importance"] == 2

    server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS)
    assert _state(a["id"])["importance"] == 2, \
        "omitting importance must not apply the default to an existing memory"

    server.memory_store.__wrapped__(content=LATENCY, tags=TAGS)
    assert _state(a["id"])["importance"] == 2


def test_new_memory_without_importance_still_defaults_to_three():
    a = server.memory_store.__wrapped__(content="Atlas stores its event log in Kinesis.",
                                        tags=["atlas", "eventlog"])
    assert _state(a["id"])["importance"] == server.DEFAULT_IMPORTANCE == 3


def test_explicit_importance_on_a_restatement_still_applies():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    server.memory_store.__wrapped__(content=LATENCY_PARA, tags=TAGS, importance=5)
    assert _state(a["id"])["importance"] == 5, \
        "a caller that states a value must still be able to raise importance"


def test_batch_item_without_importance_does_not_promote():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    out = server.memory_store.__wrapped__(items=[
        {"content": LATENCY_PARA, "tags": TAGS},
        {"content": "Rigel caches lookups in Memcached.", "tags": ["rigel", "cache"]},
    ])
    assert _state(a["id"])["importance"] == 2
    new_id = out["results"][1]["id"]
    assert _state(new_id)["importance"] == 3, "a new batch item still gets the default"


@pytest.mark.parametrize("given,expected", [(99, 5), (6, 5), (5, 5), (1, 1), (0, 1), (-4, 1)])
def test_importance_is_clamped_to_the_documented_range(given, expected):
    """The ranking term is (importance - 1) / 4, so an unclamped 9 would carry
    double the weight the 1-5 scale allows."""
    a = server.memory_store.__wrapped__(
        content=f"Sirius exposes health endpoint number {given}.",
        tags=["sirius", "health"], importance=given)
    assert _state(a["id"])["importance"] == expected


# ---------------------------------------------------------------------------
# The exact-duplicate short-circuit dropped explicit caller intent.
#
# The hash match returned before importance or tags were considered, so a caller
# could raise importance by restating a PARAPHRASE but not the identical text —
# inconsistent with the merge path, and precisely the case an agent hits when it
# re-encounters a fact verbatim and wants to promote it. The rule is now uniform
# across every store path: an omitted value changes nothing, an explicit value
# is honoured, and neither can demote.
# ---------------------------------------------------------------------------


def test_exact_restatement_honours_an_explicit_importance():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    r = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=4)
    assert r["status"] == "already_exists"
    assert _state(a["id"])["importance"] == 4, \
        "an explicit importance must apply on the exact-duplicate path too"
    assert r.get("importance") == 4, "the change should be reported, not silent"


def test_exact_restatement_without_importance_is_still_a_total_no_op():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    before = _state(a["id"])
    r = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS)
    assert r["status"] == "already_exists"
    assert "importance" not in r and "tags_added" not in r
    assert _state(a["id"]) == before, "nothing may change when nothing was stated"


def test_exact_restatement_cannot_demote():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=5)
    server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=1)
    assert _state(a["id"])["importance"] == 5


def test_exact_restatement_never_touches_recency():
    """Content is identical, so updated_at must not move even when metadata does."""
    a = server.memory_store.__wrapped__(content=LATENCY, tags=TAGS, importance=2)
    before = _state(a["id"])["updated_at"]
    server.memory_store.__wrapped__(content=LATENCY, tags=TAGS + ["perf"], importance=4)
    assert _state(a["id"])["updated_at"] == before


def test_exact_restatement_adds_new_tags():
    a = server.memory_store.__wrapped__(content=LATENCY, tags=["helios"], importance=2)
    r = server.memory_store.__wrapped__(content=LATENCY, tags=["helios", "perf", "sla"])
    assert sorted(r.get("tags_added", [])) == ["perf", "sla"]

    conn = server.get_conn()
    tags = server._parse_tags(server._collect_results(conn.execute(
        "MATCH (m:Memory {id: $i}) RETURN m.tags;", {"i": a["id"]}))[0][0])
    assert {"helios", "perf", "sla"} <= set(tags)

    # Topic nodes must be wired too, or the tag is invisible to tag-filtered search.
    topics = {t[0] for t in server._collect_results(conn.execute(
        "MATCH (m:Memory {id: $i})-[:ABOUT]->(t:Topic) RETURN t.name;", {"i": a["id"]}))}
    assert {"perf", "sla"} <= topics
