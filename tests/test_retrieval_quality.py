"""Retrieval quality regression tests.

Guards the hybrid fusion ranking against regressions. Each query has exactly
one correct answer, and the fixture deliberately includes keyword-overlap
distractors: memories that share prominent tokens with the query but are not
the answer. That is the failure mode where a keyword channel can outrank the
semantic channel (see FUSION_MODE in server.py).

Metrics reported: precision@1, MRR, recall@3.
"""

import os
import sys
import time

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/retrieval-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server

# (content, tags) — kept semantically distinct so the 0.92 dedup never merges them
FACTS = [
    # Cluster sharing the phrase "payments service" (keyword-overlap distractors)
    "The payments service uses DynamoDB table 'PaymentsLedger' in us-east-1 for transaction records.",
    "The payments service is written in Java 17 and deployed via Apollo to the prod-payments stage.",
    "The payments service on-call rotation is owned by the team 'payments-core'.",
    "Retry policy for the payments service: exponential backoff, max 5 attempts, jitter enabled.",
    # Cluster sharing "auth"
    "The auth module lives in src/auth/ and validates tokens against Cognito user pools.",
    "Auth integration tests require a seeded Cognito local emulator on port 9229.",
    "The auth service rate limits token refresh to 10 requests per minute per client.",
    # Cluster sharing "build"
    "Build failures with CannotFindBuildDirectoryException mean you are running brazil-build outside a package.",
    "The build pipeline publishes artifacts to the internal maven mirror after unit tests pass.",
    "Nightly builds run integration tests against the gamma stage, not prod.",
    # Distinct singletons
    "Incident INC-4821 was caused by a DynamoDB throttling event during a traffic spike.",
    "The frontend bundles with esbuild and targets evergreen browsers only.",
    "Postgres connection pooling is handled by pgbouncer in transaction mode.",
    "Terraform state for the networking stack is stored in an S3 backend with DynamoDB locking.",
    "The search indexer consumes a Kinesis stream and writes to OpenSearch in batches of 500.",
    "Feature flags are evaluated client-side using a cached ruleset refreshed every 60 seconds.",
    "Log retention in CloudWatch is 30 days for non-prod and 400 days for prod accounts.",
    "The mobile app uses Kotlin Multiplatform for shared business logic across iOS and Android.",
    "Canary deployments shift 10 percent of traffic for 15 minutes before full rollout.",
    "Secrets are injected at runtime from Secrets Manager, never baked into container images.",
]

# query -> substring identifying the single correct answer
QUERIES = {
    "what database does the payments service use": "DynamoDB table 'PaymentsLedger'",
    "which team is on call for payments": "payments-core",
    "what language is the payments service written in": "Java 17",
    "how many times do we retry payments requests": "max 5 attempts",
    "where does token validation happen": "Cognito user pools",
    "how do we throttle refreshing tokens": "rate limits token refresh",
    "why does my brazil build fail with a directory error": "CannotFindBuildDirectoryException",
    "what caused the throttling incident": "INC-4821",
    "how is terraform state locked": "S3 backend with DynamoDB locking",
    "what does the indexer read from": "Kinesis stream",
    "how long are production logs kept": "400 days for prod",
    "how do containers get their credentials": "Secrets Manager",
}


@pytest.fixture(scope="module")
def populated():
    """One fresh DB for the whole module, loaded with the fixture."""
    server._conn = None
    server._db = None
    res = server.memory_store.__wrapped__(items=[{"content": f} for f in FACTS])
    stored = res.get("results", [])
    assert len(stored) == len(FACTS)
    # Every fact must be genuinely new — if dedup merged any, the fixture is broken
    merged = [r for r in stored if r.get("status") != "stored_new"]
    assert not merged, f"fixture facts collided under dedup: {merged}"
    yield
    server._conn = None
    server._db = None


def _rank_of(query: str, needle: str) -> int:
    """1-based rank of the correct answer, or 0 if absent from top-10."""
    res = server.memory_search.__wrapped__(query=query, top_k=10)
    results = res.get("results", res) if isinstance(res, dict) else res
    for i, r in enumerate(results, start=1):
        if needle in r.get("content", ""):
            return i
    return 0


def test_retrieval_quality_metrics(populated):
    """Report and enforce ranking quality across the fixture."""
    ranks = {q: _rank_of(q, needle) for q, needle in QUERIES.items()}

    found = [r for r in ranks.values() if r > 0]
    p_at_1 = sum(1 for r in ranks.values() if r == 1) / len(ranks)
    recall_3 = sum(1 for r in ranks.values() if 1 <= r <= 3) / len(ranks)
    mrr = sum(1.0 / r for r in found) / len(ranks)

    print("\n--- retrieval quality ---")
    for q, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        print(f"  rank {r if r else '>10'}  {q}")
    print(f"  precision@1: {p_at_1:.2%}")
    print(f"  recall@3:    {recall_3:.2%}")
    print(f"  MRR:         {mrr:.4f}")

    # Thresholds guard against regression. Raise them if quality improves.
    assert recall_3 >= 0.75, f"recall@3 regressed to {recall_3:.2%}"
    assert p_at_1 >= 0.58, f"precision@1 regressed to {p_at_1:.2%}"
    assert mrr >= 0.70, f"MRR regressed to {mrr:.4f}"


def test_keyword_distractor_does_not_outrank_semantic_answer(populated):
    """The exact failure from the side-by-side test: a memory sharing the
    query's prominent tokens ('payments service') must not outrank the memory
    that actually answers it."""
    rank = _rank_of("what database does the payments service use",
                    "DynamoDB table 'PaymentsLedger'")
    assert rank == 1, (
        f"keyword distractor outranked the semantic answer (correct answer at "
        f"rank {rank}); the fusion channels are likely on incomparable scales"
    )


# --- tie ordering must not depend on insertion order -------------------------
#
# Sorting by score alone left equal scores to dict order — the order memories
# happened to enter the channels — so ranking looked stable while depending on
# corpus composition. Exact ties are rare under 'legacy' float scores but
# common under rank fusion, where two memories holding the same ranks across
# channels score identically (observed in the field at 0.715/0.715 on adjacent
# results, with the top two exactly equal). Ties now break on memory
# properties: importance, then recency, then id.

_TIE_FACTS = [
    ("The alpha collector batches settlement rows every cycle.", 1, "alpha"),
    ("The beta collector batches settlement rows every cycle.", 5, "beta"),
    ("The gamma collector batches settlement rows every cycle.", 2, "gamma"),
    ("The delta collector batches settlement rows every cycle.", 4, "delta"),
]
_TIE_QUERY = "collector batches settlement rows every cycle"


def _tie_run(order):
    server._conn = None
    server._db = None
    server.get_conn()
    for i in order:
        content, imp, tag = _TIE_FACTS[i]
        server.memory_store.__wrapped__(content=content, tags=[tag], importance=imp)
    # Freeze recency so it cannot silently do the tiebreaking for us.
    server.get_conn().execute("MATCH (m:Memory) SET m.updated_at = $t;",
                              {"t": time.time() - 30 * 86400})
    out = server.memory_search.__wrapped__(query=_TIE_QUERY, top_k=4)
    return [(round(r["score"], 6), r["content"].split()[1]) for r in out["results"]]


def test_ranking_is_independent_of_insertion_order():
    a = _tie_run([0, 1, 2, 3])
    b = _tie_run([3, 2, 1, 0])
    c = _tie_run([2, 0, 3, 1])
    server._conn = None
    server._db = None

    names = [[w for _, w in r] for r in (a, b, c)]
    assert names[0] == names[1] == names[2], \
        f"result order changed with insertion order: {names}"


def test_tied_scores_break_toward_higher_importance():
    rows = _tie_run([0, 1, 2, 3])
    server._conn = None
    server._db = None
    imp_of = {tag: imp for _, imp, tag in _TIE_FACTS}

    groups: dict = {}
    for score, name in rows:
        groups.setdefault(score, []).append(name)

    for score, names in groups.items():
        if len(names) < 2:
            continue  # not a tie
        imps = [imp_of[n] for n in names]
        assert imps == sorted(imps, reverse=True), \
            f"tie at {score} ordered {imps}, expected descending importance"
