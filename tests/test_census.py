"""The vector census: complete reachability checking on every search.

Field event this exists for: 12 of 38 memories became unreachable from
fresh-text query points on a long-lived database, while every existing health
signal read green — k=1 probe (the index answers), stored-embedding self-recall
(the stored vector IS the query point, so HNSW returns it trivially even on a
broken graph, index_self_misses: 0), FTS term probe. vector_hits sat at 26 on
every query and nothing compared it to anything.

The census invariant: when the candidate pool covers the corpus, a healthy
HNSW index must return EVERY embedded memory for ANY query, because cosine
distance is defined for all vector pairs — nothing can be "too far" to appear
in a window bigger than the corpus. vector_hits < the workspace-visible
embedded count is therefore proof of unreachable nodes, free, on every search.

Also here: the engine-version stamp. The dependency is an open range
(real-ladybug>=0.15.0) and clients run @latest, so upgrades silently swap the
engine under a persistent DB whose indexes were built by an older one — the
prime trigger for both field degradations (churn alone does not reproduce
either). Indexes are derived state; on an engine change they are rebuilt.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/census-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server._index_repair_attempts = 0
    server._index_repair_last = 0.0
    yield
    server._conn = None
    server._db = None
    server._index_repair_attempts = 0
    server._index_repair_last = 0.0


def _seed(n=12):
    for i in range(n):
        server.memory_store.__wrapped__(
            content=f"The svc-{i} component exposes endpoint {3000 + i} for diagnostics.",
            tags=[f"svc{i}", "diag"])


class _PartialIndex:
    """Wrap conn.execute so QUERY_VECTOR_INDEX drops rows until a rebuild.

    Simulates the field failure shape (index answers, but from a subset)
    deterministically. The real thing IS constructible through the engine —
    transient insert+delete churn progressively orphans surviving nodes
    (docs/upstream/ladybug-hnsw-delete-churn-unreachable.md) — but onset
    varies with vector geometry and RNG seed, so tests use this wrapper.
    (Same-id delete/recreate, the embedding-update pattern, never damages
    the graph; only transient population does.)
    """

    def __init__(self, conn, drop: set):
        self._conn = conn
        self._drop = set(drop)
        self.broken = True
        self._orig = conn.execute

    def install(self):
        def wrapper(query, *a, **k):
            q = str(query)
            if "CREATE_VECTOR_INDEX" in q.upper():
                self.broken = False  # a rebuild heals the graph
            result = self._orig(query, *a, **k)
            if self.broken and "QUERY_VECTOR_INDEX" in q.upper():
                rows = server._collect_results(result)
                kept = [r for r in rows if r[0] not in self._drop]

                class _Fake:
                    def __init__(self, rows_):
                        self._rows = list(rows_)
                        self._i = 0

                    def has_next(self):
                        return self._i < len(self._rows)

                    def get_next(self):
                        row = self._rows[self._i]
                        self._i += 1
                        return row

                return _Fake(kept)
            return result

        self._conn.execute = wrapper


def test_census_expected_matches_corpus_on_healthy_index():
    _seed()
    out = server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=3,
                                           explain=True)
    meta = out["explain_meta"]
    assert meta["vector_census_expected"] == 12
    assert meta["vector_hits"] == 12
    assert "degraded" not in out


def test_census_shortfall_triggers_repair_and_recovers():
    """The 26-of-38 shape: index answers from a subset. One search must detect
    the shortfall, rebuild, and return the full corpus."""
    _seed()
    conn = server.get_conn()
    fake = _PartialIndex(conn, drop={3, 4, 5, 6})
    fake.install()

    out = server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=3,
                                           explain=True)
    meta = out["explain_meta"]
    assert fake.broken is False, "the census shortfall should have forced a rebuild"
    assert meta["vector_hits"] == 12, "post-repair rerun should reach everything"
    assert "degraded" not in out


def test_census_shortfall_reports_degraded_when_budget_spent():
    """If repair cannot run, the caller must be TOLD results are incomplete —
    the field event served silently truncated results for days."""
    _seed()
    conn = server.get_conn()
    server._index_repair_attempts = server.INDEX_REPAIR_MAX_ATTEMPTS  # budget gone
    fake = _PartialIndex(conn, drop={3, 4, 5, 6})
    fake.install()

    out = server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=3,
                                           explain=True)
    assert out["explain_meta"]["vector_hits"] == 8
    assert "degraded" in out
    assert "8 of 12" in out["degraded"]


def test_stats_census_is_a_full_count_not_a_liveness_poke():
    """runtime.vector_index.reachable must census the corpus; the old k=1 probe
    read healthy while a third of the corpus was unreachable."""
    _seed()
    st = server.memory_stats.__wrapped__()
    vi = st["runtime"]["vector_index"]
    assert vi["embedded"] == 12
    assert vi["reachable"] == 12
    assert vi["fully_reachable"] is True
    assert st["runtime"]["engine"] == server._engine_signature()


def test_stats_status_cannot_say_ok_while_census_says_degraded():
    """Observed in the field: `status: "ok"` (a stale probe verdict) sitting
    next to `fully_reachable: false` from the fresh census in the same block.
    A field reading "ok" beside fields reading "not ok" invites exactly the
    misreading the null-census sentinel was added to prevent — the census
    verdict must win."""
    _seed()
    conn = server.get_conn()
    fake = _PartialIndex(conn, drop={3, 4, 5, 6})
    fake.install()

    vi = server.memory_stats.__wrapped__()["runtime"]["vector_index"]
    if vi["fully_reachable"] is False:
        assert vi["status"] == "degraded", \
            f"status {vi['status']!r} contradicts fully_reachable=False"
    else:
        # Stats repaired on the way (acceptable) — then everything must agree.
        assert vi["reachable"] >= vi["embedded"]
        assert vi["status"] != "degraded"


# --- engine-version stamp -----------------------------------------------------

def test_new_database_records_the_engine_signature():
    _seed(2)
    conn = server.get_conn()
    rows = server._collect_results(conn.execute(
        "MATCH (s:SchemaMeta {key: 'engine_sig'}) RETURN s.value;"))
    assert rows and rows[0][0] == server._engine_signature()


def test_engine_change_rebuilds_both_indexes():
    _seed(6)
    conn = server.get_conn()
    # Pretend the DB was last opened by an older engine, with an index state
    # that engine left broken (drop the vector index).
    conn.execute("MATCH (s:SchemaMeta {key: 'engine_sig'}) DETACH DELETE s;")
    conn.execute("CREATE (:SchemaMeta {key: 'engine_sig', value: 'old/39'});")
    server._safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                         expected_errors=("does not exist",))

    prev = server._rebuild_indexes_on_engine_change(conn)
    assert prev == "old/39"
    assert (server._probe_vector_index(conn, k=6) or 0) == 6, \
        "the upgrade rebuild must restore full reachability"
    assert server._probe_fts_index(conn) is True

    # Stamp updated; second call is a no-op.
    assert server._rebuild_indexes_on_engine_change(conn) is None


def test_same_engine_does_not_rebuild():
    _seed(2)
    conn = server.get_conn()
    calls = {"n": 0}
    orig = server._ensure_vector_index

    def counting(c, force_rebuild=False):
        calls["n"] += 1
        return orig(c, force_rebuild=force_rebuild)

    server._ensure_vector_index = counting
    try:
        assert server._rebuild_indexes_on_engine_change(conn) is None
        assert calls["n"] == 0, "matching signature must not touch the indexes"
    finally:
        server._ensure_vector_index = orig


# --- census completeness boundary ---------------------------------------------
#
# Census coverage must hold at EVERY corpus size. The cheap path compares the
# ranked vector hits against the corpus, which only works while the ranking
# pool covers it — the regime a real workspace outgrows in weeks. That used to
# make the census "inapplicable" above the pool, so the detector for the worst
# bug class protected a shrinking slice (2% at 5,000 memories) exactly as the
# graph got big enough for partial unreachability to matter. Above the pool a
# dedicated id-only probe at k=corpus runs instead: ~10% of a search's cost,
# 100% coverage.


def test_search_census_covers_corpus_even_when_pool_is_smaller(monkeypatch):
    _seed(12)
    monkeypatch.setattr(server, "SEARCH_CANDIDATE_POOL", 6)
    out = server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=2,
                                           explain=True)
    meta = out["explain_meta"]
    assert meta["census_complete"] is True, \
        "coverage must not lapse just because the corpus outgrew the pool"
    assert meta["census_mode"] == "probe", \
        "above the pool the index must be probed directly"
    # And crucially: the pool-capped ranked list must not read as a shortfall.
    assert "degraded" not in out


def test_search_census_uses_the_free_path_when_pool_covers_corpus():
    _seed(12)
    meta = server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=2,
                                            explain=True)["explain_meta"]
    assert meta["census_complete"] is True
    assert meta["census_mode"] == "pool", \
        "no extra probe should be paid for when the pool already proves coverage"
    assert meta["vector_census_expected"] == 12


def test_small_pool_still_detects_a_real_shortfall(monkeypatch):
    """The point of keeping coverage above the pool: damage must still be
    caught there. Index answers from a subset, pool smaller than corpus."""
    _seed(12)
    monkeypatch.setattr(server, "SEARCH_CANDIDATE_POOL", 6)
    conn = server.get_conn()
    fake = _PartialIndex(conn, drop={3, 4, 5, 6})
    fake.install()

    server.memory_search.__wrapped__(query="diagnostics endpoint", top_k=2,
                                     explain=True)
    assert fake.broken is False, \
        "a shortfall must trigger repair even when the pool cannot see it"


def test_dream_full_census_reports_healthy_counts():
    _seed(12)
    out = server.memory_dream.__wrapped__(force=True)
    assert out["vector_census"] == {"reachable": 12, "embedded": 12, "rebuilt": False}


def test_dream_full_census_detects_and_repairs_partial_unreachability():
    """The backstop must act at ANY corpus size, including past the pool."""
    _seed(12)
    conn = server.get_conn()
    fake = _PartialIndex(conn, drop={3, 4, 5, 6})
    fake.install()

    out = server.memory_dream.__wrapped__(force=True)
    census = out["vector_census"]
    # Two layers can catch this: the self-recall audit (fires if the damage is
    # visible to stored-embedding probes, as this simulation is) or the full
    # census (fires regardless). Either way the run must END fully reachable.
    assert census["rebuilt"] or out["index_rebuilt"], \
        "no layer repaired the unreachable nodes"
    assert census["reachable"] == census["embedded"] == 12, \
        "the dream must not finish with unreachable memories"


def test_dream_dry_run_census_reports_without_repairing():
    _seed(12)
    conn = server.get_conn()
    fake = _PartialIndex(conn, drop={3, 4})
    fake.install()

    out = server.memory_dream.__wrapped__(force=True, dry_run=True)
    census = out["vector_census"]
    assert census["reachable"] == 10
    assert census["rebuilt"] is False, "dry_run must not mutate the index"
    assert fake.broken is True


def test_dream_full_census_catches_audit_invisible_damage():
    """The field shape: stored-embedding self-recall PASSES (the audit reads
    index_self_misses: 0) while fresh query points miss nodes. Simulated by
    dropping rows only from large-k probes (the census asks k=embedded; the
    audit's merge probes ask k=4), so the audit sees a healthy index and the
    census is the only layer that can fire."""
    _seed(12)
    conn = server.get_conn()

    state = {"broken": True}
    orig = conn.execute

    def wrapper(query, *a, **k):
        q = str(query)
        if "CREATE_VECTOR_INDEX" in q.upper():
            state["broken"] = False
        result = orig(query, *a, **k)
        params = a[0] if a else (k.get("parameters") or {})
        big_k = isinstance(params, dict) and (params.get("k") or 0) >= 12
        if state["broken"] and "QUERY_VECTOR_INDEX" in q.upper() and big_k:
            rows = [r for r in server._collect_results(result) if r[0] not in {3, 4, 5}]

            class _Fake:
                def __init__(self, rows_):
                    self._rows = list(rows_)
                    self._i = 0

                def has_next(self):
                    return self._i < len(self._rows)

                def get_next(self):
                    row = self._rows[self._i]
                    self._i += 1
                    return row

            return _Fake(rows)
        return result

    conn.execute = wrapper

    out = server.memory_dream.__wrapped__(force=True)
    assert out["index_self_misses"] == 0, \
        "precondition: the audit must be blind to this damage, as in the field"
    census = out["vector_census"]
    assert census["rebuilt"] is True, "the census is the only layer that can fire here"
    assert census["reachable"] == census["embedded"] == 12


# --- post-delete census -------------------------------------------------------

def test_delete_runs_census_and_repairs_shortfall():
    """Deletes are the damage source (engine delete maintenance orphans
    SURVIVING nodes — reproduced standalone, monotonic, in-process). A session
    that deletes and exits must not hand the next session a broken graph, so
    memory_delete itself runs the census and rebuilds on shortfall."""
    _seed()
    conn = server.get_conn()
    victim = server._collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.id LIMIT 1;"))[0][0]
    fake = _PartialIndex(conn, drop={3, 4, 5, 6} - {victim})
    fake.install()

    out = server.memory_delete.__wrapped__(memory_id=victim)
    assert out["status"] == "deleted"
    assert fake.broken is False, \
        "the post-delete census should have detected the shortfall and rebuilt"

    st = server.memory_stats.__wrapped__()["runtime"]["vector_index"]
    assert st["fully_reachable"] is True, f"not repaired: {st}"


def test_delete_census_is_quiet_on_a_healthy_index():
    """No shortfall, no rebuild — the census must not churn the index."""
    _seed()
    conn = server.get_conn()
    victim = server._collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.id LIMIT 1;"))[0][0]

    rebuilds = []
    orig = conn.execute

    def spy(query, *a, **k):
        if "CREATE_VECTOR_INDEX" in str(query).upper():
            rebuilds.append(str(query))
        return orig(query, *a, **k)

    conn.execute = spy
    out = server.memory_delete.__wrapped__(memory_id=victim)
    conn.execute = orig
    assert out["status"] == "deleted"
    assert not rebuilds, "healthy index must not be rebuilt on delete"


# --- dream scan coverage ------------------------------------------------------
#
# MAX_CONSOLIDATE_SCAN bounds what one dream examines. It used to be a
# permanent HORIZON: always the newest N by updated_at, so once a workspace
# passed the cap everything older was never considered for merge or prune
# again — silent coverage loss at ordinary sizes, not extreme ones. It is now
# a rotating WINDOW: same per-run cost, full coverage over successive runs.

def test_dream_scan_window_rotates_and_wraps(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MAX_CONSOLIDATE_SCAN", 20)
    monkeypatch.setattr(server, "_dream_scan_cursor", 0)
    _seed(100)

    offsets = []
    for _ in range(6):
        out = server.memory_dream.__wrapped__(force=True)
        offsets.append(out["scan_coverage"]["offset"])

    assert offsets == [0, 20, 40, 60, 80, 0], \
        f"window must advance then wrap, got {offsets}"


def test_dream_reports_runs_needed_for_full_coverage(monkeypatch):
    monkeypatch.setattr(server, "MAX_CONSOLIDATE_SCAN", 20)
    monkeypatch.setattr(server, "_dream_scan_cursor", 0)
    _seed(100)

    cov = server.memory_dream.__wrapped__(force=True)["scan_coverage"]
    assert cov["corpus"] == 100
    assert cov["window"] == 20
    assert cov["runs_for_full_coverage"] == 5, \
        "a caller must be able to see that one run is not the whole corpus"


def test_dream_scan_does_not_rotate_while_corpus_fits(monkeypatch):
    """No cursor churn when the window already covers everything."""
    monkeypatch.setattr(server, "MAX_CONSOLIDATE_SCAN", 1000)
    monkeypatch.setattr(server, "_dream_scan_cursor", 0)
    _seed(12)

    for _ in range(3):
        cov = server.memory_dream.__wrapped__(force=True)["scan_coverage"]
        assert cov["offset"] == 0
        assert cov["runs_for_full_coverage"] == 1
