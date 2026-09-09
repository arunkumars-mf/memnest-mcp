"""Scale + longevity soak for memnest-mcp.

Every serious bug in this server's history came from ACCUMULATED STATE —
churn, upgrades, a long-lived file — not from logic visible in a fresh
database. This harness compresses that history: it seeds a corpus far above
anything the test suite touches (the suite tops out around 100 memories),
then runs cycles of store / update / search / delete / dream against a real
on-disk file, reopening the connection between cycles to simulate process
restarts, and asserts the retrieval contract after every cycle.

Invariants checked each cycle:
  1. Census: every workspace-visible memory with an embedding is reachable
     through the vector index (stats reads reachable == embedded).
  2. Bookkeeping: the node count in the DB equals the tracked expectation
     given every reported merge, prune, and delete. A silent loss or a
     silent duplicate both fail this.
  3. FTS: the index answers.
  4. Retrieval: a planted sentinel fact is findable by its exact wording
     at every corpus size (top-3, not top-1: the corpus is adversarially
     repetitive by design).
  5. IDs stay unique.

Usage:
  ./venv/bin/python benchmark/soak.py --seed-size 1500 --cycles 10
  ./venv/bin/python benchmark/soak.py --seed-size 300 --cycles 4   # quick

The DB persists in benchmark/results/soak/ so successive runs extend the
same file's history — closer to the n=1 long-lived DB where both index
events actually happened. Pass --fresh to start over.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOAK_DIR = os.path.join(HERE, "results", "soak")

parser = argparse.ArgumentParser()
parser.add_argument("--seed-size", type=int, default=1500)
parser.add_argument("--cycles", type=int, default=10)
parser.add_argument("--fresh", action="store_true", help="delete the soak DB first")
parser.add_argument("--rng-seed", type=int, default=1041)
parser.add_argument("--sessions", type=int, default=0,
                    help="Run N short SESSIONS (separate processes) instead of "
                         "in-process cycles. This is the realistic stress "
                         "pattern for one-connection-per-workspace: the worst "
                         "bug of the series was one session's deletes handing "
                         "the next a broken index.")
parser.add_argument("--session-child", action="store_true",
                    help=argparse.SUPPRESS)
args = parser.parse_args()

if args.fresh and os.path.isdir(SOAK_DIR):
    shutil.rmtree(SOAK_DIR)
os.makedirs(SOAK_DIR, exist_ok=True)

os.environ["MEMORY_DB_PATH"] = os.path.join(SOAK_DIR, "memory.lbug")
os.environ["MEMORY_WORKSPACE"] = "/soak"
os.environ["MEMORY_RESPONSE_FORMAT"] = "json"
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from memnest_mcp import server as S  # noqa: E402

rng = random.Random(args.rng_seed)

SUBJECTS = ["checkout", "ledger", "auth", "search", "billing", "fraud",
            "pricing", "catalog", "shipping", "notifications", "payments",
            "inventory", "reporting", "identity", "sessions", "gateway"]
FACTS = [
    "The {s} service p99 latency budget is {n} ms.",
    "{S} deploys ride the Tuesday release train, cut at {n}:00 UTC.",
    "Incident INC-{n}: the {s} service exhausted its connection pool.",
    "The {s} database failover drill is scheduled every {n} weeks.",
    "{S} retention policy: raw events are kept for {n} days.",
    "The {s} cache is sized at {n} GB with LRU eviction.",
    "Decision: {s} moves to the shared event bus in Q{q}.",
    "The {s} team owns the {s2} integration and its runbook.",
    "Alert threshold for {s} error rate is {n} basis points.",
    "The {s} batch job runs at 0{q}:30 and writes to the {s2} store.",
]
SENTINEL = ("The soak sentinel fact: the archival vault rotates its signing "
            "keys every 17 days under runbook R-9017.")


def _fact(i):
    s = rng.choice(SUBJECTS)
    s2 = rng.choice([x for x in SUBJECTS if x != s])
    t = rng.choice(FACTS)
    return (t.format(s=s, S=s.capitalize(), s2=s2, n=rng.randint(2, 900),
                     q=rng.randint(1, 4)),
            [s, "soak"], rng.randint(1, 5))


class Tracker:
    """Expected live-node count, adjusted by every reported outcome."""
    def __init__(self):
        self.expected = 0
        self.ids = set()

    def stored(self, res):
        for r in (res["results"] if "results" in res else [res]):
            if r.get("status") in ("stored_new", "stored_new_no_embedding"):
                self.expected += 1
                assert r["id"] not in self.ids, f"duplicate id issued: {r['id']}"
                self.ids.add(r["id"])
            # updated_existing / merged: no count change

    def dreamed(self, res):
        self.expected -= res.get("auto_merged", 0)
        self.expected -= res.get("pruned", 0)

    def deleted(self, n):
        self.expected -= n


def _count():
    r = S.get_conn().execute("MATCH (m:Memory) RETURN COUNT(m);")
    return r.get_next()[0]


def _reopen():
    """Simulate a process restart between cycles."""
    if S._conn is not None:
        try:
            S._conn.close()
        except Exception:
            pass
    S._conn = None
    S._db = None
    S.get_conn()


def check_invariants(track, cycle, phase):
    stats = S.memory_stats.__wrapped__()
    rt = stats["runtime"]
    vi = rt.get("vector_index") or {}
    assert vi.get("fully_reachable") is True, \
        f"[c{cycle} {phase}] census: {vi}"
    fts = rt.get("fts_index") or {}
    assert fts.get("answering") is not False, \
        f"[c{cycle} {phase}] FTS not answering: {fts}"
    actual = _count()
    assert actual == track.expected, \
        f"[c{cycle} {phase}] count drift: db={actual} expected={track.expected}"
    hit = S.memory_search.__wrapped__(
        query="archival vault signing key rotation runbook", top_k=3)
    contents = [r["content"] for r in hit["results"]]
    assert any("R-9017" in c for c in contents), \
        f"[c{cycle} {phase}] sentinel unfindable at corpus {actual}"
    return actual


def session_child():
    """One short session: open, small burst of work with churn, close.

    Deliberately a separate PROCESS per session so the index is serialized and
    deserialized between them — the boundary where inherited index damage
    became visible in the field. Asserts the contract on entry (what the
    previous session left behind) and on exit.
    """
    before = check_invariants(_adopt(), 0, "session-entry")
    track = _adopt()

    items = []
    for i in range(15):
        c, tags, imp = _fact(i)
        items.append({"content": c, "tags": tags, "importance": imp})
    track.stored(S.memory_store.__wrapped__(items=items))

    # transient burst: the pattern that orphans surviving HNSW nodes
    burst = [{"content": f"session transient {i}: {_fact(i)[0]}", "tags": ["tmp"]}
             for i in range(12)]
    bres = S.memory_store.__wrapped__(items=burst)
    track.stored(bres)
    bids = [r["id"] for r in bres["results"]
            if r.get("status", "").startswith("stored_new")]
    if bids:
        S.memory_delete.__wrapped__(memory_id=bids)
        track.deleted(len(bids))

    for q in ("release train", "retention policy", "cache eviction"):
        assert S.memory_search.__wrapped__(query=q, top_k=5)["results"]

    after = check_invariants(track, 0, "session-exit")
    print(f"  session ok: {before} -> {after} live", flush=True)
    try:
        S._conn.close()
    except Exception:
        pass


def _adopt():
    """Tracker seeded from what is actually in the DB (new process)."""
    t = Tracker()
    t.expected = _count()
    rows = S._collect_results(S.get_conn().execute("MATCH (m:Memory) RETURN m.id;"))
    t.ids = {r[0] for r in rows}
    return t


def run_sessions():
    """Parent: spawn N short session processes in sequence."""
    print(f"=== {args.sessions} short sessions (separate processes) ===", flush=True)
    t0 = time.time()
    for s in range(1, args.sessions + 1):
        p = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--session-child",
             "--rng-seed", str(args.rng_seed + s)],
            capture_output=True, text=True, timeout=900)
        line = [l for l in p.stdout.splitlines() if "session ok" in l]
        print(f"session {s}/{args.sessions}: {line[-1].strip() if line else '(no output)'} "
              f"exit={p.returncode}  t={time.time() - t0:.0f}s", flush=True)
        if p.returncode != 0:
            print("=== SESSION FAILED ===")
            print(p.stdout[-1500:])
            print(p.stderr[-2500:])
            sys.exit(1)
    print(f"\nSESSIONS PASSED: {args.sessions} sessions, corpus {_count()}, "
          f"{time.time() - t0:.0f}s")


def main():
    t0 = time.time()
    track = Tracker()
    existing = _count()
    if existing:
        # Resuming a previous soak file: adopt its state.
        track.expected = existing
        rows = S._collect_results(S.get_conn().execute("MATCH (m:Memory) RETURN m.id;"))
        track.ids = {r[0] for r in rows}
        print(f"resuming soak DB with {existing} memories")

    # --- Seed ---
    sentinel_present = any(
        True for _ in S._collect_results(S.get_conn().execute(
            "MATCH (m:Memory) WHERE m.content CONTAINS 'R-9017' RETURN m.id;")))
    if not sentinel_present:
        track.stored(S.memory_store.__wrapped__(content=SENTINEL,
                                                tags=["soak", "sentinel"],
                                                importance=5))
    to_seed = max(0, args.seed_size - track.expected)
    print(f"seeding {to_seed} memories...")
    BATCH = 400
    while to_seed > 0:
        n = min(BATCH, to_seed)
        items = []
        for i in range(n):
            c, tags, imp = _fact(i)
            items.append({"content": c, "tags": tags, "importance": imp})
        track.stored(S.memory_store.__wrapped__(items=items))
        to_seed -= n
        print(f"  ...{track.expected} live "
              f"({time.time() - t0:.0f}s)")
    check_invariants(track, 0, "post-seed")
    print(f"seeded. corpus={track.expected}  t={time.time() - t0:.0f}s")

    # --- Cycles ---
    for cycle in range(1, args.cycles + 1):
        _reopen()

        # stores (some will dedup-merge, tracker handles both outcomes)
        items = []
        for i in range(30):
            c, tags, imp = _fact(i)
            items.append({"content": c, "tags": tags, "importance": imp})
        track.stored(S.memory_store.__wrapped__(items=items))

        # content updates on random survivors (exercises recreate path)
        live = [r[0] for r in S._collect_results(S.get_conn().execute(
            "MATCH (m:Memory) WHERE m.workspace = '/soak' "
            "AND NOT m.content CONTAINS 'R-9017' RETURN m.id LIMIT 500;"))]
        for mid in rng.sample(live, min(10, len(live))):
            S.memory_update.__wrapped__(
                memory_id=mid,
                content=_fact(0)[0] + f" (revised c{cycle})")

        # deletes
        victims = rng.sample(live, min(5, len(live)))
        res = S.memory_delete.__wrapped__(memory_id=victims)
        track.deleted(len(res.get("deleted", victims)))
        track.ids -= set(victims)

        # transient burst: store a batch, then delete exactly that batch.
        # This is the pattern that progressively orphans SURVIVING nodes in
        # the HNSW graph (engine 0.15.3; see docs/upstream/
        # ladybug-hnsw-delete-churn-unreachable.md) — the census invariant
        # below is what catches it if the post-delete repair ever regresses.
        burst = []
        for i in range(20):
            c, tags, imp = _fact(i)
            burst.append({"content": f"transient c{cycle}: {c}", "tags": tags})
        bres = S.memory_store.__wrapped__(items=burst)
        burst_ids = [r["id"] for r in bres["results"]
                     if r.get("status", "").startswith("stored_new")]
        track.stored(bres)
        if burst_ids:
            bdel = S.memory_delete.__wrapped__(memory_id=burst_ids)
            track.deleted(len(bdel.get("deleted", burst_ids)))
            track.ids -= set(burst_ids)

        # searches interleaved (census fast-path exercised at corpus > pool)
        for q in ("incident connection pool", "release train cut time",
                  "retention policy raw events", "cache eviction sizing"):
            out = S.memory_search.__wrapped__(query=q, top_k=5)
            assert out["results"], f"[c{cycle}] no results for {q!r}"

        # dream every 3rd cycle: merge + prune + census, unattended cadence
        if cycle % 3 == 0:
            dres = S.memory_dream.__wrapped__(force=True)
            track.dreamed(dres)
            vc = dres.get("vector_census") or {}
            print(f"  c{cycle} dream: merged={dres.get('auto_merged', 0)} "
                  f"pruned={dres.get('pruned', 0)} census={vc}")

        actual = check_invariants(track, cycle, "end")
        print(f"cycle {cycle}/{args.cycles} ok  corpus={actual}  "
              f"t={time.time() - t0:.0f}s")

    print(f"\nSOAK PASSED: {args.cycles} cycles, corpus {_count()}, "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    if args.session_child:
        session_child()
    elif args.sessions:
        run_sessions()
    else:
        main()
