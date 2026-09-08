"""Does rrf inflate the top score on UNANSWERABLE questions?

Refusal depends on an absolute signal: "nothing here matches well". Legacy
cosine has a real floor/ceiling spread, so a bad match scores low. Under rrf
the rank-1 item always scores (K+1)/(K+1)=1.0 per channel regardless of
absolute quality — the signal is gone by construction. Measure it.
"""
import os, sys, json, statistics

os.environ["MEMORY_DB_PATH"] = \
    "/Users/arunkse/test/ladybug-memory-mcp/benchmark/results/dbs/conv-26/memory.lbug"
os.environ["MEMORY_WORKSPACE"] = "conv-26"
os.environ["MEMORY_RESPONSE_FORMAT"] = "json"
sys.path.insert(0, "/Users/arunkse/test/ladybug-memory-mcp/src")
from memnest_mcp import server as S

qa = json.load(open(
    "/Users/arunkse/test/ladybug-memory-mcp/benchmark/data/locomo10.json"))[0]["qa"]

# category 5 = adversarial (unanswerable); category 1 = single_hop (answerable)
unanswerable = [q["question"] for q in qa if q.get("category") == 5][:40]
answerable = [q["question"] for q in qa if q.get("category") == 1][:40]


def tops(questions):
    out = []
    for q in questions:
        r = S.memory_search.__wrapped__(query=q, top_k=1)
        if r.get("results"):
            out.append(r["results"][0]["score"])
    return out


un = tops(unanswerable)
an = tops(answerable)
sep = statistics.mean(an) - statistics.mean(un)
print(json.dumps({
    "mode": S.FUSION_MODE,
    "unanswerable_top1_mean": round(statistics.mean(un), 4),
    "unanswerable_top1_max": round(max(un), 4),
    "answerable_top1_mean": round(statistics.mean(an), 4),
    "separation": round(sep, 4),
    "unanswerable_spread": round(max(un) - min(un), 4),
}))
