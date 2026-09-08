"""Retrieval-only LOCOMO A/B: legacy vs rrf on the same ingested conv-26 DB.

No LLM. For each question with evidence turn IDs, search top-20 and measure
whether the evidence-bearing memories surface, and at what rank.
MEMORY_FUSION comes from the environment (run once per mode).
"""
import os, sys, json

os.environ["MEMORY_DB_PATH"] = \
    "/Users/arunkse/test/ladybug-memory-mcp/benchmark/results/dbs/conv-26/memory.lbug"
os.environ["MEMORY_WORKSPACE"] = "conv-26"
os.environ["MEMORY_RESPONSE_FORMAT"] = "json"
sys.path.insert(0, "/Users/arunkse/test/ladybug-memory-mcp/src")
from memnest_mcp import server as S

data = json.load(open(
    "/Users/arunkse/test/ladybug-memory-mcp/benchmark/data/locomo10.json"))
conv = data[0]
qa = conv["qa"]

conn = S.get_conn()
# Map dia_id tag -> memory ids (turn memories carry the dia_id tag)
rows = S._collect_results(conn.execute("MATCH (m:Memory) RETURN m.id, m.tags;"))
by_dia = {}
for mid, tags in rows:
    for t in S._parse_tags(tags):
        if ":" in t and t[0] == "d":
            by_dia.setdefault(t, set()).add(mid)

K = 20
hits5 = hits20 = mrr_sum = n = 0
misses = []
for q in qa:
    ev = [e.lower() for e in q.get("evidence", []) if isinstance(e, str)]
    ev_ids = set()
    for e in ev:
        ev_ids |= by_dia.get(e, set())
    if not ev_ids:
        continue  # adversarial / no mappable evidence
    n += 1
    out = S.memory_search.__wrapped__(query=q["question"], top_k=K)
    ranked = [r["id"] for r in out["results"]]
    # related surfaces evidence too — count it as a hit at rank of its anchor? No:
    # strict ranked-list metric; related counted separately.
    rel = {r["id"] for r in out.get("related", [])}
    first = next((i + 1 for i, mid in enumerate(ranked) if mid in ev_ids), None)
    if first:
        mrr_sum += 1.0 / first
        hits20 += 1
        if first <= 5:
            hits5 += 1
    else:
        misses.append({"q": q["question"][:60], "cat": q.get("category"),
                       "via_related": bool(ev_ids & rel)})

res = {
    "mode": S.FUSION_MODE, "questions": n,
    "hit@5": round(hits5 / n, 4), "hit@20": round(hits20 / n, 4),
    "mrr@20": round(mrr_sum / n, 4),
    "misses": len(misses),
    "missed_but_in_related": sum(1 for m in misses if m["via_related"]),
}
print(json.dumps(res))
json.dump(misses, open(f"/tmp/locomo_misses_{S.FUSION_MODE}.json", "w"), indent=1)
