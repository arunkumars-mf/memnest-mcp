"""Smoke tests for memnest-mcp server.

Tests use an in-memory database, exercise the core tool surface, and verify
that key invariants are preserved (especially relationship preservation across
the delete + recreate workaround for vector-indexed embeddings).
"""

import json
import os
import sys

# Force in-memory DB before importing the server module
os.environ["MEMORY_DB_PATH"] = ":memory:"
os.environ["MEMORY_WORKSPACE"] = "/test-workspace"
os.environ["MEMORY_RESPONSE_FORMAT"] = "json"  # Tests parse JSON; production can use TOON

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def reset_db(monkeypatch):
    """Reset the connection between tests so each starts fresh."""
    # Force a brand-new in-memory DB by closing any existing handle
    server._conn = None
    server._db = None
    server._dream_ops_since_last = 0
    server._dream_last_time = 0.0
    yield
    # Cleanup: clear globals so next test gets a fresh DB
    server._conn = None
    server._db = None


def _unwrap(res):
    """Helpers below call `__wrapped__` to bypass the @_timed serializer.
    Inner tool functions return dicts/lists. Decoded JSON strings are also
    accepted for forward-compat with any tool still emitting them.
    """
    if isinstance(res, (dict, list)):
        return res
    if isinstance(res, str):
        try:
            return json.loads(res)
        except (ValueError, TypeError):
            return res
    return res


def _store(content, **kwargs):
    return _unwrap(server.memory_store.__wrapped__(content=content, **kwargs))


def _search(query, **kwargs):
    res = _unwrap(server.memory_search.__wrapped__(query=query, **kwargs))
    # memory_search wraps results in {"results": [...]} for TOON-friendliness
    return res.get("results", res) if isinstance(res, dict) else res


def _query(cypher):
    return _unwrap(server.memory_query.__wrapped__(cypher_query=cypher))


def test_store_and_search_roundtrip():
    res = _store("Python is a programming language", category="learning", tags=["python"])
    assert res["status"] == "stored_new"
    mid = res["id"]
    assert mid > 0

    results = _search("python language")
    assert len(results) > 0
    assert any(r["id"] == mid for r in results)


def test_exact_dedup_returns_already_exists():
    r1 = _store("Identical content", category="general")
    r2 = _store("Identical content", category="general")
    assert r1["status"] == "stored_new"
    assert r2["status"] == "already_exists"
    assert r1["id"] == r2["id"]


def test_batch_store():
    res = _unwrap(server.memory_store.__wrapped__(items=[
        {"content": "First batch memory", "tags": ["batch"]},
        {"content": "Second batch memory", "tags": ["batch"]},
        {"content": "Third batch memory", "tags": ["batch"]},
    ]))
    assert res["count"] == 3
    assert all(r["status"] == "stored_new" for r in res["results"])


def test_relationships_survive_content_update():
    """Critical: updating content must NOT lose relationships."""
    a = _store("Apples are red fruit grown on trees")["id"]
    b = _store("Submarines navigate underwater oceans")["id"]
    c = _store("Mountains are formed by tectonic plates")["id"]

    server.memory_relate.__wrapped__(from_id=a, to_id=b, relationship="RELATED_TO",
                                      confidence=0.9, provenance="EXTRACTED")
    server.memory_relate.__wrapped__(from_id=b, to_id=c, relationship="EXPLAINS")

    # Verify edges exist
    rels_before = _query(
        "MATCH (a:Memory)-[r]->(b:Memory) RETURN COUNT(r);"
    )
    assert rels_before["rows"][0][0] == 2

    # Update memory B's content
    upd = _unwrap(server.memory_update.__wrapped__(
        memory_id=b, content="Submarines navigate underwater oceans with sonar"
    ))
    assert upd["status"] == "updated"

    # Edges should still exist
    rels_after = _query(
        "MATCH (a:Memory)-[r]->(b:Memory) RETURN COUNT(r);"
    )
    assert rels_after["rows"][0][0] == 2, \
        f"Lost relationships across update: was 2, now {rels_after['rows'][0][0]}"


def test_explains_rationale_type_preserved_on_update():
    a = _store("Volcanic eruptions release magma")["id"]
    b = _store("Birds migrate seasonally for food")["id"]
    server._conn.execute(
        """MATCH (a:Memory {id: $a}), (b:Memory {id: $b})
           CREATE (a)-[:EXPLAINS {rationale_type: 'counterexample'}]->(b);""",
        {"a": a, "b": b},
    )

    _unwrap(server.memory_update.__wrapped__(
        memory_id=b, content="Birds migrate seasonally for warmer climates"
    ))

    res = _query(
        "MATCH (a:Memory)-[r:EXPLAINS]->(b:Memory) RETURN r.rationale_type;"
    )
    assert res["rows"], "EXPLAINS edge lost"
    assert res["rows"][0][0] == "counterexample", \
        f"rationale_type stripped: got {res['rows'][0][0]}"


def test_workspace_isolation_in_dedup():
    """Dedup must not merge memories across workspaces."""
    server.WORKSPACE = "/workspace-A"
    a_res = _store("Workspace A unique memory content here for similarity testing")
    a = a_res["id"]

    server.WORKSPACE = "/workspace-B"
    # Slightly different content so exact-hash dedup doesn't fire — we want
    # the SEMANTIC dedup to be tested, which is the path that should respect workspace.
    b_res = _store("Workspace A unique memory content here for similarity test")
    b = b_res["id"]

    # Should be stored as new in workspace B, not merged into A
    assert b_res["status"] == "stored_new", \
        f"Expected stored_new (workspace isolation), got {b_res['status']}"
    assert a != b


def test_memory_query_destructive_block():
    """Server-level kill switch should reject DELETE when env disables it."""
    server.ALLOW_DESTRUCTIVE_QUERIES = False
    try:
        res = _query("MATCH (m:Memory) DETACH DELETE m;")
        assert res.get("status") == "error"
        assert "blocked" in res["message"].lower()
    finally:
        server.ALLOW_DESTRUCTIVE_QUERIES = True


def test_memory_query_read_only_blocks_writes():
    res = _unwrap(server.memory_query.__wrapped__(
        cypher_query="MATCH (m:Memory) DETACH DELETE m;",
        read_only=True,
    ))
    assert res.get("status") == "error"
    assert "read_only" in res["message"].lower()


def test_memory_schema_returns_tables():
    res = _unwrap(server.memory_schema.__wrapped__())
    assert "Memory" in res["nodes"]
    assert "Topic" in res["nodes"]
    assert "RELATED_TO" in res["rels"]
    assert "EXPLAINS" in res["rels"]


def test_memory_delete_batch():
    a = _store("Pineapples taste sweet and tangy")["id"]
    b = _store("Helicopters fly using rotor blades")["id"]
    res = _unwrap(server.memory_delete.__wrapped__(memory_id=[a, b, 99999]))
    assert set(res["deleted"]) == {a, b}
    assert res["not_found"] == [99999]


def test_dream_skips_when_too_few_memories():
    _store("Just one memory")
    res = _unwrap(server.memory_dream.__wrapped__())
    assert res["status"] == "not_needed"
    assert res["reason"] == "memory_count_too_low"


def test_dream_force_runs_with_few_memories():
    _store("Memory 1")
    _store("Memory 2")
    res = _unwrap(server.memory_dream.__wrapped__(force=True))
    assert res["status"] == "completed"
    assert "pruned" in res
    assert "auto_merged" in res


def test_toon_format_when_enabled():
    """Verify TOON serialization works when configured."""
    if not server._TOON_AVAILABLE:
        pytest.skip("toons package not installed")

    # Switch to TOON mode for this test. Call the *decorated* function so we
    # exercise the @_timed serialization path that produces a TOON string.
    original = server.RESPONSE_FORMAT
    server.RESPONSE_FORMAT = "toon"
    try:
        result = server.memory_stats()
        assert isinstance(result, str), f"Expected serialized string, got {type(result)}"
        # TOON output uses key:value or array notation, not JSON's '{'
        assert not result.lstrip().startswith("{"), \
            f"Expected TOON format, got JSON: {result[:100]}"
        # And elapsed_ms must be present (regression: previously dropped under TOON)
        assert "elapsed_ms" in result, \
            f"elapsed_ms missing from TOON response: {result[:200]}"
    finally:
        server.RESPONSE_FORMAT = original


def test_elapsed_ms_present_in_json_responses():
    """elapsed_ms must be injected by @_timed regardless of format."""
    server.RESPONSE_FORMAT = "json"
    raw = server.memory_stats()
    parsed = json.loads(raw)
    assert "elapsed_ms" in parsed, "elapsed_ms missing under JSON format"
    assert isinstance(parsed["elapsed_ms"], (int, float))


def test_serialize_helper_is_compact():
    """JSON output should be compact (no indent, no extra whitespace)."""
    server.RESPONSE_FORMAT = "json"
    result = server._serialize({"a": 1, "b": [1, 2, 3]})
    # Compact JSON has no spaces after colons/commas in stdlib by default
    assert "  " not in result, f"Expected compact JSON, got: {result}"


def test_memory_query_blocks_set_on_embedding():
    """SET on m.embedding should be rejected (HNSW index limitation)."""
    _store("test memory")
    res = _unwrap(server.memory_query.__wrapped__(
        cypher_query="MATCH (m:Memory) SET m.embedding = [0.1, 0.2];"
    ))
    assert res.get("status") == "error"
    assert "embedding" in res["message"].lower()


def test_memory_topics_returns_topics_with_counts():
    # Distinct content so semantic dedup doesn't merge
    _store("Apples are red fruit", tags=["fruit", "food"])
    _store("Submarines navigate underwater", tags=["fruit", "vehicle"])  # share 'fruit' tag intentionally
    _store("Mountains form via tectonics", tags=["standalone"])

    res = _unwrap(server.memory_topics.__wrapped__())
    topic_names = [t["topic"] for t in res["topics"]]
    assert "fruit" in topic_names

    # 'fruit' is on 2 memories
    fruit = next(t for t in res["topics"] if t["topic"] == "fruit")
    assert fruit["count"] == 2


def test_memory_topics_min_count_filter():
    _store("Apples are red fruit", tags=["unique-tag"])
    _store("Submarines navigate underwater", tags=["shared"])
    _store("Mountains form via tectonics", tags=["shared"])

    res = _unwrap(server.memory_topics.__wrapped__(min_count=2))
    topic_names = [t["topic"] for t in res["topics"]]
    assert "shared" in topic_names
    assert "unique-tag" not in topic_names


def test_tag_canonicalization_prevents_fragmentation():
    """Tags like 'Kuzu', 'kuzu', 'KUZU' should all merge into one Topic node."""
    _store("Apples are red fruit", tags=["Kuzu", "Database"])
    _store("Submarines navigate underwater", tags=["kuzu", "database"])
    _store("Mountains form via tectonics", tags=[" KUZU ", "DataBase"])

    # All three should map to the SAME canonical 'kuzu' and 'database' topics
    res = _unwrap(server.memory_topics.__wrapped__())
    topic_names = [t["topic"] for t in res["topics"]]
    assert "kuzu" in topic_names
    assert "database" in topic_names
    # No fragmentation: no Kuzu/KUZU variants
    assert "Kuzu" not in topic_names
    assert "KUZU" not in topic_names

    # 'kuzu' should connect to all 3 memories (one Topic, three edges)
    kuzu = next(t for t in res["topics"] if t["topic"] == "kuzu")
    assert kuzu["count"] == 3


def test_canonicalization_handles_whitespace_and_punctuation():
    """Internal whitespace becomes hyphens, trailing punctuation stripped."""
    _store("Test memory one", tags=["aws transform", "MCP."])
    _store("Test memory two", tags=["aws-transform", "mcp"])

    res = _unwrap(server.memory_topics.__wrapped__())
    topic_names = [t["topic"] for t in res["topics"]]
    assert "aws-transform" in topic_names
    assert "mcp" in topic_names
    # Both spellings collapsed
    assert "aws transform" not in topic_names
    assert "MCP." not in topic_names


def test_search_with_uppercase_tag_finds_canonical():
    """Searching for tags=['Kuzu'] should still match memories tagged with canonical 'kuzu'."""
    _store("Apples are red fruit", tags=["kuzu"])
    _store("Mountains form via tectonics", tags=["unrelated"])

    # Even though we stored 'kuzu', the agent might pass 'Kuzu' or 'KUZU' on search
    results = _search("fruit", tags=["KUZU"])
    assert any(r["id"] == 1 for r in results), \
        "Tag search should canonicalize input to match stored form"


# -----------------------------------------------------------------------------
# Regression tests for fixes in 0.2.0
# -----------------------------------------------------------------------------


def test_tags_with_comma_round_trip():
    """Tags containing a comma must survive store -> read -> Topic linking
    without being split. Fixes the JSON-vs-comma storage inconsistency."""
    res = _store("Memory with comma-bearing tag",
                 tags=["hello,world", "three"])
    mid = res["id"]

    # Search returns tags as a list, with the comma intact in the first tag.
    results = _search("comma-bearing")
    assert results, "expected at least one search hit"
    hit = next(r for r in results if r["id"] == mid)
    assert "hello,world" in hit["tags"], \
        f"expected 'hello,world' as a single tag, got {hit['tags']!r}"
    assert "three" in hit["tags"]
    # Length 2: should NOT have been split into hello/world/three.
    assert len(hit["tags"]) == 2, \
        f"tags should be 2 entries (comma preserved), got {hit['tags']!r}"

    # Topic graph should also have exactly two topics for this memory.
    res2 = _query(
        "MATCH (m:Memory {id: " + str(mid) + "})-[:ABOUT]->(t:Topic) RETURN t.name;"
    )
    topic_names = sorted(r[0] for r in res2["rows"])
    assert topic_names == ["hello,world", "three"], \
        f"Topic nodes diverged from stored tags: {topic_names!r}"


def test_relate_rejects_missing_endpoints():
    """memory_relate must NOT silently succeed when the source/target IDs
    don't exist. Fixes the no-op silently-reporting-success bug."""
    a = _store("Anchor memory")["id"]
    res = _unwrap(server.memory_relate.__wrapped__(
        from_id=a, to_id=99999, relationship="RELATED_TO"
    ))
    assert res["status"] == "not_found", \
        f"expected not_found for missing target, got {res!r}"

    # Also confirm no edge was created.
    edges = _query("MATCH ()-[r]->() RETURN COUNT(r);")
    assert edges["rows"][0][0] == 0, \
        f"no edge should exist after a not_found relate, got {edges['rows'][0][0]}"


def test_relate_succeeds_with_valid_endpoints():
    a = _store("First")["id"]
    b = _store("Second")["id"]
    res = _unwrap(server.memory_relate.__wrapped__(
        from_id=a, to_id=b, relationship="RELATED_TO"
    ))
    assert res["status"] == "created"
    assert res["from"] == a
    assert res["to"] == b
    assert res["type"] == "RELATED_TO"


def test_memory_query_default_blocks_destructive():
    """With MEMORY_ALLOW_DESTRUCTIVE flipped to default-false in 0.2.0,
    bare memory_query DELETEs are blocked unless explicitly enabled."""
    server.ALLOW_DESTRUCTIVE_QUERIES = False
    res = _query("MATCH (m:Memory) DETACH DELETE m;")
    assert res.get("status") == "error"
    assert "blocked" in res["message"].lower()


def test_memory_traverse_alias_blocks_destructive_unconditionally():
    """memory_traverse is the safe alias and must reject writes regardless of
    the global ALLOW_DESTRUCTIVE_QUERIES setting."""
    server.ALLOW_DESTRUCTIVE_QUERIES = True  # try to lift the global guard
    try:
        res = _unwrap(server.memory_traverse.__wrapped__(
            cypher_query="MATCH (m:Memory) DETACH DELETE m;"
        ))
        # The wrapped memory_query enforces read_only=True, returning the
        # read-only block message.
        assert res.get("status") == "error"
        assert "read_only" in res["message"].lower()
    finally:
        server.ALLOW_DESTRUCTIVE_QUERIES = False


def test_memory_get_alias_returns_full_content():
    long_content = "x" * 1500  # exceeds MAX_CONTENT_LENGTH
    mid = _store(long_content)["id"]
    res = _unwrap(server.memory_get.__wrapped__(memory_id=mid))
    assert res["status"] == "found"
    assert res["content"] == long_content, "memory_get must return full content untruncated"


def test_memory_list_alias_filters_by_workspace():
    """memory_list should respect MEMORY_WORKSPACE by default."""
    server.WORKSPACE = "/ws-list-A"
    a = _store("Workspace A list memory")["id"]
    server.WORKSPACE = "/ws-list-B"
    b = _store("Workspace B list memory")["id"]

    # Default search is scoped to current workspace (B).
    res = _unwrap(server.memory_list.__wrapped__())
    ids = [item["id"] for item in res["items"]]
    assert b in ids
    assert a not in ids, "expected workspace isolation in memory_list"

    # global_search exposes both.
    res2 = _unwrap(server.memory_list.__wrapped__(global_search=True))
    ids2 = [item["id"] for item in res2["items"]]
    assert a in ids2 and b in ids2


def test_graph_html_refuses_oversized_graph(monkeypatch):
    """memory_graph_html must refuse to render when node count > limit."""
    monkeypatch.setattr(server, "GRAPH_HTML_MAX_NODES", 2)
    _store("memory one")
    _store("memory two")
    _store("memory three")
    res = _unwrap(server.memory_graph_html.__wrapped__(open_browser=False))
    assert res["status"] == "too_large", \
        f"expected too_large, got {res!r}"
    assert res["limit"] == 2
    assert res["node_count"] >= 3


def test_graph_html_writes_stable_latest_path():
    """Successful renders write a fixed `memory_graph.html` plus a snapshot."""
    _store("simple memory for graph render")
    res = _unwrap(server.memory_graph_html.__wrapped__(open_browser=False))
    assert res["status"] == "generated"
    assert res["path"].endswith("memory_graph.html"), \
        "expected the stable filename, got " + res["path"]
    assert res["snapshot"] != res["path"], "snapshot should be a separate file"
    assert "browser_opened" in res
    assert os.path.exists(res["path"]), "stable file must exist on disk"


def test_graph_html_escapes_untrusted_content():
    """Memory content containing HTML must NOT appear in tooltip HTML
    (rendered as text, not parsed) and must be fed via DOM textContent in the
    detail panel. We accept the payload appearing inside the JSON literal
    embedded in the `<script>` block (it's a string property and the JS reads
    it via `textContent`/escape, not `innerHTML`)."""
    payload = "<img src=x onerror=alert(1)>"
    _store(payload, tags=["xss-test"])
    res = _unwrap(server.memory_graph_html.__wrapped__(open_browser=False))
    assert res["status"] == "generated"
    rendered = open(res["path"], "r", encoding="utf-8").read()

    # 1. The detail panel must NOT use innerHTML on memory content.
    assert ".innerHTML = `" not in rendered, \
        "innerHTML usage on untrusted content reintroduces XSS"
    # 2. Tooltip lines (the `title` field assembled by Python) must contain the
    #    escaped form, not the raw form.
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered, \
        "tooltip text was not HTML-escaped"
    # 3. The new code uses .textContent and replaceChildren — sanity-check.
    assert "textContent" in rendered
    assert "buildField" in rendered
