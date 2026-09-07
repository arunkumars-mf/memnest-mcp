"""Algorithmic replacements for hand-rolled graph behaviours.

Three changes, each replacing ad-hoc iteration with the standard method, plus
workarounds for two LadybugDB engine defects found while wiring them in:

  CALL K_CORE_DECOMPOSITION hangs (unkillable, C-level) on a filtered
  projection whose graph combines density, parallel edges and excluded nodes.
  Minimal repro: K10 of inferred edges + 7 parallel hub edges in one workspace,
  4 filtered-out nodes holding one edge in another. Each ingredient alone
  completes. -> replaced with in-process peeling (_kcore_peel).

  CALL LOUVAIN silently ignores node predicates — a projection filtered to 10
  nodes returned louvain_ids for all 14. -> run global, scope at write-back.

PAGE_RANK and SCC honour predicates (verified) and use filtered projections.
"""

import os
import sys
import time

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/algo-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server.WORKSPACE = "/algo-test"
    yield
    server._conn = None
    server._db = None
    server.WORKSPACE = "/algo-test"


# --- _kcore_peel unit behaviour ---------------------------------------------

def test_kcore_peel_triangle_plus_tail():
    # Triangle (core 2) with a pendant (core 1).
    core = server._kcore_peel([(1, 2), (2, 3), (3, 1), (3, 4)])
    assert core == {1: 2, 2: 2, 3: 2, 4: 1}


def test_kcore_peel_collapses_parallel_edges():
    """Two rows between one pair (inferred + asserted) are one adjacency.
    The engine call counted them separately, inflating structural importance."""
    single = server._kcore_peel([(1, 2)])
    doubled = server._kcore_peel([(1, 2), (1, 2), (2, 1)])
    assert single == doubled == {1: 1, 2: 1}


def test_kcore_peel_ignores_self_loops_and_handles_empty():
    assert server._kcore_peel([(1, 1)]) == {}
    assert server._kcore_peel([]) == {}


def test_kcore_peel_dense_with_exclusion_shape():
    """The exact shape that hangs the engine call, as a regression canary:
    K10 + parallel hub edges must complete instantly in-process."""
    edges = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    edges += [(0, i) for i in range(1, 8)]  # parallels
    t0 = time.perf_counter()
    core = server._kcore_peel(edges)
    assert (time.perf_counter() - t0) < 1.0
    assert all(v == 9 for v in core.values())  # K10: everyone has 9 neighbours


# --- workspace scoping of graph scores ---------------------------------------

def _seed_two_workspaces():
    store = server.memory_store.__wrapped__
    relate = server.memory_relate.__wrapped__
    server.WORKSPACE = "/ws/alpha"
    alpha = [store(content=f"Alpha fact {i} about the checkout pipeline step {i}.",
                   tags=[f"a{i}", "alpha"])["id"] for i in range(4)]
    relate(from_id=alpha[0], to_id=alpha[1], relationship="RELATED_TO")
    server.WORKSPACE = "/ws/beta"
    beta = [store(content=f"Beta billing rule {i}: reconcile ledger partition {i} nightly.",
                  tags=["beta", "billing", "ledger", f"rule{i}"])["id"] for i in range(10)]
    for i in range(1, 8):
        relate(from_id=beta[0], to_id=beta[i], relationship="RELATED_TO")
    return alpha, beta


def test_graph_scores_do_not_leak_across_workspaces():
    alpha, beta = _seed_two_workspaces()
    conn = server.get_conn()

    server.WORKSPACE = "/ws/alpha"
    server._compute_graph_scores(conn)

    rows = server._collect_results(conn.execute(
        "MATCH (m:Memory) WHERE m.workspace = '/ws/beta' "
        "RETURN m.pagerank, m.k_degree, m.community_id;"))
    for pr, k, cid in rows:
        assert pr == 0.0 and k == 0 and cid == -1, \
            "alpha's run scored beta's memories"


def test_each_workspace_scores_its_own_and_leaves_others_alone():
    alpha, beta = _seed_two_workspaces()
    conn = server.get_conn()

    server.WORKSPACE = "/ws/alpha"
    server._compute_graph_scores(conn)
    before = {r[0]: (r[1], r[2]) for r in server._collect_results(conn.execute(
        "MATCH (m:Memory) WHERE m.workspace = '/ws/alpha' RETURN m.id, m.pagerank, m.k_degree;"))}

    server.WORKSPACE = "/ws/beta"
    server._compute_graph_scores(conn)  # this exact call used to hang the engine

    after = {r[0]: (r[1], r[2]) for r in server._collect_results(conn.execute(
        "MATCH (m:Memory) WHERE m.workspace = '/ws/alpha' RETURN m.id, m.pagerank, m.k_degree;"))}
    assert before == after, "beta's run rewrote alpha's scores"

    beta_scored = server._collect_results(conn.execute(
        "MATCH (m:Memory) WHERE m.workspace = '/ws/beta' AND m.pagerank > 0 RETURN COUNT(m);"))[0][0]
    assert beta_scored == 10


def test_inferred_edges_never_cross_workspaces():
    """Shared topic NAMES are common across projects; an inferred edge between
    them would wire two projects' graphs together."""
    store = server.memory_store.__wrapped__
    server.WORKSPACE = "/ws/alpha"
    for i in range(2):
        store(content=f"Alpha auth config note {i} for the login service.",
              tags=["auth", "config", "login"])
    server.WORKSPACE = "/ws/beta"
    for i in range(2):
        store(content=f"Beta auth config note {i} for the payments login.",
              tags=["auth", "config", "login"])
    conn = server.get_conn()
    server._compute_graph_scores(conn)

    cross = server._collect_results(conn.execute(
        """MATCH (a:Memory)-[r:RELATED_TO {provenance: 'INFERRED'}]->(b:Memory)
           WHERE a.workspace <> b.workspace RETURN COUNT(r);"""))[0][0]
    assert cross == 0


# --- multi-hop related --------------------------------------------------------

def _seed_chain():
    store = server.memory_store.__wrapped__
    relate = server.memory_relate.__wrapped__
    checkout = store(content="Helios checkout depends on payments-core for authorisation.",
                     tags=["helios", "deps"])["id"]
    pcore = store(content="The payments-core service depends on the ledger-db cluster.",
                  tags=["payments", "deps"])["id"]
    ledger = store(content="The ledger-db cluster is deprecated, decommissioned Q2 2026.",
                   tags=["ledger", "lifecycle"])["id"]
    relate(from_id=checkout, to_id=pcore, relationship="RELATED_TO")
    relate(from_id=pcore, to_id=ledger, relationship="RELATED_TO")
    return checkout, pcore, ledger


def test_related_reaches_two_hops_with_hop_annotation():
    """The measured gap: the transitive deprecation was invisible at 1 hop."""
    checkout, pcore, ledger = _seed_chain()
    out = server.memory_search.__wrapped__(
        query="What deprecated infrastructure does the Helios checkout flow depend on?",
        top_k=1)
    related = {r["id"]: r for r in out.get("related", [])}
    assert pcore in related and "hops" not in related[pcore], \
        "direct neighbour should carry no hop annotation"
    assert ledger in related and related[ledger]["hops"] == 2, \
        "the two-hop deprecation must surface, annotated"


def test_related_selection_is_hop_decayed_relevance():
    """Distance-only ordering starved deeper hops of the capped slots and broke
    equal-distance ties by id. Reported shape: four 1-hop neighbours filled the
    list, and of two 2-hop candidates the slot went to an ownership fact rather
    than the deprecation that ANSWERED the query. Selection is now
    cosine(query, candidate) * DECAY^(hops-1), so the answer must win the
    contested slot even when the noise has the smaller id."""
    store = server.memory_store.__wrapped__
    relate = server.memory_relate.__wrapped__
    checkout = store(content="Helios checkout depends on payments-core for authorisation.",
                     tags=["helios", "deps"])["id"]
    one_hop = []
    for text, tg in [
        ("INC-4400: a ledger-db failover caused twelve minutes of checkout errors.",
         ["helios", "incident"]),
        ("The payments-core service depends on the ledger-db cluster.",
         ["payments", "deps"]),
        ("Helios checkout also depends on inventory-cache for stock lookups.",
         ["helios", "deps"]),
        ("The fintech-infra team operates the checkout escalation rota.",
         ["fintech", "ownership"]),
    ]:
        mid = store(content=text, tags=tg)["id"]
        one_hop.append(mid)
        relate(from_id=checkout, to_id=mid, relationship="RELATED_TO")
    pcore = one_hop[1]
    # Noise stored FIRST (smaller id): the old (hops, id) ordering picked it.
    noise = store(content="The payments-core service is owned by Marcus Webb's team.",
                  tags=["payments", "ownership"])["id"]
    answer = store(content="The ledger-db cluster is deprecated and will be decommissioned in Q2 2026.",
                   tags=["ledger", "lifecycle"])["id"]
    relate(from_id=pcore, to_id=noise, relationship="RELATED_TO")
    relate(from_id=pcore, to_id=answer, relationship="RELATED_TO")

    out = server.memory_search.__wrapped__(
        query="What deprecated infrastructure does the Helios checkout flow "
              "transitively depend on?",
        top_k=1)
    ids = [r["id"] for r in out.get("related", [])]
    assert answer in ids, "the 2-hop answer was starved out of the capped list"
    assert noise not in ids, "the irrelevant equal-distance candidate took the slot"


def test_related_falls_back_to_hop_order_without_a_query_embedding(monkeypatch):
    """Degraded mode has no relevance signal; distance ordering must still work."""
    _seed_chain()
    monkeypatch.setattr(server, "_embed", lambda t: None)
    out = server.memory_search.__wrapped__(
        query="ledger-db deprecated checkout", top_k=1)
    related = out.get("related", [])
    assert related, "expected FTS-ranked results to still expand"
    hops = [r.get("hops", 1) for r in related]
    assert hops == sorted(hops), "fallback should preserve distance ordering"


def test_related_walk_respects_workspace():
    checkout, pcore, ledger = _seed_chain()
    server.WORKSPACE = "/elsewhere"
    foreign = server.memory_store.__wrapped__(
        content="Foreign note about ledger-db maintenance windows.", tags=["ledger"])["id"]
    server.WORKSPACE = "/algo-test"
    server.memory_relate.__wrapped__(from_id=ledger, to_id=foreign, relationship="RELATED_TO")

    out = server.memory_search.__wrapped__(
        query="What deprecated infrastructure does the Helios checkout flow depend on?",
        top_k=1)
    ids = {r["id"] for r in out.get("related", [])}
    assert foreign not in ids, "the walk crossed into another workspace"


def test_one_hop_config_restores_old_behaviour(monkeypatch):
    checkout, pcore, ledger = _seed_chain()
    monkeypatch.setattr(server, "GRAPH_EXPAND_HOPS", 1)
    out = server.memory_search.__wrapped__(
        query="What deprecated infrastructure does the Helios checkout flow depend on?",
        top_k=1)
    ids = {r["id"] for r in out.get("related", [])}
    assert pcore in ids and ledger not in ids


# --- union-find dream merge ---------------------------------------------------

TRIPLET = [
    "Runbook R-1041: drain, patch, verify, restore traffic to the cluster.",
    "Runbook R-1041: drain, patch, verify, restore traffic to the cluster now.",
    "Runbook R-1041: drain, patch, verify then restore traffic to the cluster.",
]


def _triplet(order):
    """Insert three near-duplicates around dedup (supersedes chain), unlink."""
    store = server.memory_store.__wrapped__
    ids, prev = [], None
    for k in order:
        kw = {"supersedes": prev} if prev is not None else {}
        r = store(content=TRIPLET[k], tags=["runbook", "r1041"], **kw)
        ids.append(r["id"])
        prev = r["id"]
        time.sleep(0.01)
    for i in range(1, len(ids)):
        server.memory_unrelate.__wrapped__(from_id=ids[i], to_id=ids[i - 1],
                                           relationship="SUPERSEDES")
    assert server._count_memories(server.get_conn()) >= 3
    return ids


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_dream_merge_survivor_is_insertion_order_independent(order):
    """Greedy merged in scan order, so the survivor depended on updated_at.
    The survivor must be a pure function of the content SET."""
    _triplet(order)
    out = server.memory_dream.__wrapped__(force=True)
    assert out["auto_merged"] == 2
    survivors = [r[0] for r in server._collect_results(server.get_conn().execute(
        "MATCH (m:Memory) RETURN m.content;"))]
    assert len(survivors) == 1
    # Longest content wins; contents 1 and 2 tie on length, so the tiebreak is
    # the content itself — lexicographically larger of the two.
    expected = max(TRIPLET, key=lambda c: (len(c), c))
    assert survivors[0] == expected


def test_dream_refuses_component_with_internally_protected_pair():
    """Transitivity can join A and C through B when the DIRECT A-C pair is
    protected. Greedy never faced this; union-find must check every internal
    pair and refuse the whole component."""
    ids = _triplet([0, 1, 2])
    server.memory_relate.__wrapped__(from_id=ids[2], to_id=ids[0],
                                     relationship="SUPERSEDES")
    out = server.memory_dream.__wrapped__(force=True)
    assert out["auto_merged"] == 0
    assert server._count_memories(server.get_conn()) == 3


def test_dream_merge_migrates_edges_from_all_dropped_members():
    ids = _triplet([0, 1, 2])
    anchor = server.memory_store.__wrapped__(
        content="INC-8800: the R-1041 runbook was exercised during the outage.",
        tags=["incident"])["id"]
    server.memory_relate.__wrapped__(from_id=anchor, to_id=ids[0],
                                     relationship="EXPLAINS")
    out = server.memory_dream.__wrapped__(force=True)
    assert out["auto_merged"] == 2

    rows = server._collect_results(server.get_conn().execute(
        "MATCH (a:Memory)-[:EXPLAINS]->(b:Memory) RETURN a.id, b.id;"))
    assert len(rows) == 1 and rows[0][0] == anchor, \
        "the EXPLAINS edge must survive onto the merge survivor"
