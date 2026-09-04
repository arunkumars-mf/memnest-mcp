"""Input validation and search pagination.

Each case below is a verified misbehaviour, not a hypothetical:

  preview_chars=-5     -> _truncate did text[:-5], silently corrupting output
  top_k=0 / -3         -> 0 results PLUS a `degraded` message blaming the
                          embedding model, so a caller-side bug was reported as
                          a server-side failure
  120,000-char content -> accepted, embedded and stored
  memory_delete        -> status "deleted" even when every id was not_found
  min_importance='high'-> raw ValueError out of the tool
  offset               -> did not exist; rank 11 was unreachable for any query

The candidate pool also mattered: it was top_k * 3, so the size of the page
determined which memories were even scored. On a 25-memory corpus, top_k=10
surfaced two memories that outranked every result top_k=5 returned.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/validation-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

QUERY = "alpha service timeout production"


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


def _seed(n=25):
    for i in range(n):
        server.memory_store.__wrapped__(
            content=f"The alpha-{i} service handles request class {i} with a "
                    f"timeout of {100 + i * 13} milliseconds in production.",
            tags=[f"alpha{i}", "svc"])


# --- clamping --------------------------------------------------------------

@pytest.mark.parametrize("preview", [-5, 0, -1])
def test_negative_preview_chars_does_not_corrupt_content(preview):
    _seed(5)
    out = server.memory_search.__wrapped__(query=QUERY, top_k=2, preview_chars=preview)
    for r in out["results"]:
        assert r["content"], "preview must not be empty"
        assert not r["content"].startswith("..."), "content was sliced from the wrong end"


@pytest.mark.parametrize("k", [0, -3])
def test_non_positive_top_k_is_clamped_and_not_reported_as_degraded(k):
    _seed(5)
    out = server.memory_search.__wrapped__(query=QUERY, top_k=k)
    assert len(out["results"]) >= 1
    assert "degraded" not in out, "a caller-side top_k must not look like a dead channel"


def test_clamp_int_helper():
    assert server._clamp_int(5, 1, 10, 3) == 5
    assert server._clamp_int(0, 1, 10, 3) == 1
    assert server._clamp_int(99, 1, 10, 3) == 10
    assert server._clamp_int("nope", 1, 10, 3) == 3
    assert server._clamp_int(None, 1, 10, 3) == 3


# --- size caps -------------------------------------------------------------

def test_oversized_content_is_capped_and_reported():
    big = "y " * 40000
    r = server.memory_store.__wrapped__(content=big, tags=["big"])
    assert r["content_truncated_to"] == server.MAX_STORE_CHARS
    stored = server.memory_get.__wrapped__(memory_id=r["id"])["content"]
    assert len(stored) == server.MAX_STORE_CHARS


def test_oversized_batch_is_refused():
    r = server.memory_store.__wrapped__(
        items=[{"content": f"batch fact {i}"} for i in range(server.MAX_BATCH_ITEMS + 1)])
    assert r["status"] == "error"
    assert "Too many" in r["message"]


def test_tag_count_and_length_are_capped():
    r = server.memory_store.__wrapped__(
        content="Tag cap probe fact.",
        tags=[f"t{i}" for i in range(60)] + ["z" * 500])
    tags = server.memory_get.__wrapped__(memory_id=r["id"])["tags"]
    assert len(tags) <= server.MAX_TAGS_PER_MEMORY
    assert max(len(t) for t in tags) <= server.MAX_TAG_CHARS


# --- honest statuses -------------------------------------------------------

def test_delete_status_reflects_what_happened():
    _seed(2)
    ids = [r[0] for r in server._collect_results(
        server.get_conn().execute("MATCH (m:Memory) RETURN m.id ORDER BY m.id;"))]

    assert server.memory_delete.__wrapped__(
        memory_id=[999998, 999999])["status"] == "not_found"
    assert server.memory_delete.__wrapped__(
        memory_id=[ids[0], 999999])["status"] == "partial"
    assert server.memory_delete.__wrapped__(memory_id=ids[1])["status"] == "deleted"


def test_batch_store_reports_failures_in_the_envelope():
    out = server.memory_store.__wrapped__(items=[
        {"content": "A genuine fact about the alpha service."},
        {"content": "   "},
    ])
    assert out["status"] == "partial"
    assert out["errors"] == 1


def test_list_rejects_a_non_numeric_min_importance():
    _seed(3)
    out = server.memory_list.__wrapped__(min_importance="high")
    assert out["status"] == "error"
    assert "min_importance" in out["message"]


@pytest.mark.parametrize("call,kwargs", [
    ("memory_store", {"content": "   "}),
    ("memory_search", {"query": ""}),
    ("memory_query", {"cypher_query": "  "}),
])
def test_empty_required_inputs_return_the_error_shape(call, kwargs):
    out = getattr(server, call).__wrapped__(**kwargs)
    assert out["status"] == "error"
    assert "message" in out


def test_update_with_nothing_to_change_is_an_error():
    _seed(1)
    out = server.memory_update.__wrapped__(memory_id=1)
    assert out["status"] == "error"


# --- pagination ------------------------------------------------------------

def test_pages_are_disjoint_and_ordered():
    _seed()
    p1 = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=0)
    p2 = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=5)
    p3 = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=10)

    ids1 = [r["id"] for r in p1["results"]]
    ids2 = [r["id"] for r in p2["results"]]
    ids3 = [r["id"] for r in p3["results"]]

    assert ids1 and ids2 and ids3
    assert not set(ids1) & set(ids2), "page 1 and 2 overlap"
    assert not set(ids2) & set(ids3), "page 2 and 3 overlap"
    # Descending score across page boundaries.
    assert p1["results"][-1]["score"] >= p2["results"][0]["score"]


def test_paging_matches_one_large_page():
    """offset must slice the same ranking, not re-rank."""
    _seed()
    big = server.memory_search.__wrapped__(query=QUERY, top_k=10, offset=0)
    p1 = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=0)
    p2 = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=5)

    big_ids = [r["id"] for r in big["results"]]
    assert [r["id"] for r in p1["results"]] == big_ids[:5]
    assert [r["id"] for r in p2["results"]] == big_ids[5:10]


def test_offset_and_has_more_are_reported():
    _seed()
    out = server.memory_search.__wrapped__(query=QUERY, top_k=5, offset=5)
    assert out["offset"] == 5
    assert out["has_more"] is True


def test_negative_offset_is_clamped():
    _seed(5)
    a = server.memory_search.__wrapped__(query=QUERY, top_k=3, offset=-10)
    b = server.memory_search.__wrapped__(query=QUERY, top_k=3, offset=0)
    assert [r["id"] for r in a["results"]] == [r["id"] for r in b["results"]]


def test_candidate_pool_is_independent_of_page_size():
    """The pool used to be top_k * 3, so the page size decided which memories
    were scored at all — the top result could be missing from a small page."""
    _seed()
    small = server.memory_search.__wrapped__(query=QUERY, top_k=3, explain=True)
    large = server.memory_search.__wrapped__(query=QUERY, top_k=10, explain=True)
    assert small["explain_meta"]["candidate_pool"] == large["explain_meta"]["candidate_pool"]
    # And the small page is a true prefix of the large one.
    assert [r["id"] for r in small["results"]] == [r["id"] for r in large["results"]][:3]


def test_scores_are_unchanged_by_the_pool_size(monkeypatch):
    """A wider pool scores MORE candidates but must not change any score:
    vector similarity is per-pair, FTS rows arrive score-ordered so the
    normalising max is the first row regardless of limit, and
    recency/importance/graph are per-memory."""
    _seed()
    wide = {r["id"]: r["score"]
            for r in server.memory_search.__wrapped__(query=QUERY, top_k=5)["results"]}

    monkeypatch.setattr(server, "SEARCH_CANDIDATE_POOL", 500)
    wider = {r["id"]: r["score"]
             for r in server.memory_search.__wrapped__(query=QUERY, top_k=5)["results"]}

    shared = set(wide) & set(wider)
    assert shared, "expected overlapping results"
    for mid in shared:
        assert abs(wide[mid] - wider[mid]) < 1e-9, f"score for {mid} moved with the pool size"
