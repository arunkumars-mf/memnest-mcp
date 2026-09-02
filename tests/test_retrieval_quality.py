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
