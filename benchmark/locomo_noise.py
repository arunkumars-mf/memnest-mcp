"""Noise floor: legacy vs legacy replicate, compared to legacy vs rrf."""
import json
from collections import Counter

R = "/Users/arunkse/test/ladybug-memory-mcp/benchmark/results"
CAT = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain", 5: "adversarial"}


def load(p):
    return {(j["conv_id"], j["q_idx"]): j for j in json.load(open(p))}


leg1 = load(f"{R}/archive/v0241-legacy/judgments.json")
rrf = load(f"{R}/archive/v0241-rrf/judgments.json")
leg2 = load(f"{R}/judgments.json")


def compare(a, b, label):
    lost, won = [], []
    for k, jb in b.items():
        if k not in a:
            continue
        sa, sb = a[k].get("score", 0), jb.get("score", 0)
        if sa and not sb:
            lost.append(jb)
        elif sb and not sa:
            won.append(jb)
    sa_tot = sum(1 for j in a.values() if j.get("score"))
    sb_tot = sum(1 for j in b.values() if j.get("score"))
    print(f"{label}: {sa_tot}/199 -> {sb_tot}/199  "
          f"({(sb_tot - sa_tot) / 199 * 100:+.2f} pts)  "
          f"flips -{len(lost)} +{len(won)} = {len(lost) + len(won)} total churn")
    print(f"   lost by category: {dict(Counter(CAT.get(j.get('category')) for j in lost))}")
    print(f"   won  by category: {dict(Counter(CAT.get(j.get('category')) for j in won))}")


compare(leg1, leg2, "legacy -> legacy (SAME CONFIG, noise floor)")
compare(leg1, rrf, "legacy -> rrf   (the actual comparison)")
