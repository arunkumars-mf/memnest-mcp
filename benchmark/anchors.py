"""Canonical retrieval anchors, with the graph state they are valid for.

These numbers have been used throughout development as a "did ranking move?"
regression check. A field observation showed the check was under-specified:
graph centrality (PageRank + k-core) is computed by `memory_dream`, never at
store time, so a corpus where dream has run scores every CONNECTED memory
higher than the same corpus where it has not. Asserting one value therefore
reads as a regression on any real workspace, where dream has run at least once.

So both states are pinned, and each run states which one it checked:

  COLD  no dream has run; graph channel contributes 0 to every memory
  WARM  dream has run at least once; centrality is live

WARM is stable across repeated dreams (verified), so it is a legitimate anchor
rather than a moving target. On this fixture the warm graph term happens to be
UNIFORM across the top results (every one is equally connected), so it lifts
scores without reordering them — a corpus with a genuine hub would also see
order change, which is the case worth watching.

Usage:
  ./venv/bin/python benchmark/anchors.py            # check both states
  ./venv/bin/python benchmark/anchors.py --update   # reprint for pasting
"""

import glob
import json
import os
import sys

HELIOS = "/Users/arunkse/test-tmp1/helios_files/*.md"

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/anchor")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memnest_mcp import server as S  # noqa: E402

Q7 = "Did Helios adopt a service mesh?"
Q6 = "Which Helios checkout dependency is actively maintained and not deprecated?"

# fusion mode -> state -> query -> [(id, score), ...]
CANONICAL = {
    "legacy": {
        "cold": {
            "Q7": [(26, 0.7445), (21, 0.5384), (6, 0.5153)],
            "Q6": [(29, 0.6638), (6, 0.6167), (7, 0.6048)],
        },
        # Q6 is the informative one: id 29 (the CORRECT answer) is unconnected,
        # so it gets no centrality lift while its two competitors each gain
        # 0.0192 — narrowing the correct answer's lead from 0.0471 to 0.0443.
        # Harmless at this margin, and the exact shape that becomes a wrong
        # answer when the gap is thinner and the competitor is a hub.
        "warm": {
            "Q7": [(26, 0.7474), (21, 0.5413), (6, 0.5182)],
            "Q6": [(29, 0.6638), (6, 0.6195), (7, 0.6077)],
        },
    },
}


def _ingest():
    files = sorted(glob.glob(HELIOS))
    if not files:
        print(f"SKIP: no fixture at {HELIOS}")
        sys.exit(0)
    for f in files:
        S.memory_store.__wrapped__(content=open(f).read().strip(),
                                   tags=["helios-sbs"])


def _measure():
    out = {}
    for name, q in (("Q7", Q7), ("Q6", Q6)):
        res = S.memory_search.__wrapped__(query=q, top_k=3, explain=True)
        out[name] = [(r["id"], r["score"]) for r in res["results"]]
        out[name + "_graph"] = [round(r["explain"]["graph"], 4)
                                for r in res["results"]]
    return out


def main():
    update = "--update" in sys.argv
    mode = S.FUSION_MODE
    _ingest()

    results = {"cold": _measure()}
    S.memory_dream.__wrapped__(force=True)          # computes centrality
    results["warm"] = _measure()

    if update or mode not in CANONICAL:
        print(json.dumps({k: {kk: vv for kk, vv in v.items()}
                          for k, v in results.items()}, indent=2))
        if mode not in CANONICAL:
            print(f"\n(no canonical values recorded for fusion mode {mode!r})")
        return

    failures = []
    for state in ("cold", "warm"):
        for name in ("Q7", "Q6"):
            got = results[state][name]
            want = CANONICAL[mode][state][name]
            n = min(len(want), len(got))
            if [tuple(x) for x in got[:n]] != [tuple(x) for x in want[:n]]:
                failures.append((state, name, want[:n], got[:n]))
            print(f"  {'FAIL' if failures and failures[-1][:2] == (state, name) else 'PASS'}"
                  f"  {mode}/{state}/{name}: {got[:n]}"
                  f"  graph={results[state][name + '_graph'][:n]}")

    if failures:
        print("\nANCHORS MOVED:")
        for state, name, want, got in failures:
            print(f"  {state}/{name}\n    want {want}\n    got  {got}")
        sys.exit(1)
    print("\nANCHORS OK (both graph-cold and graph-warm)")


if __name__ == "__main__":
    main()
