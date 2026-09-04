"""memory_query's safety gate.

The previous guard matched raw substrings against query.upper():
("DETACH DELETE", "DELETE ", "DROP ", "TRUNCATE"). Two verified holes:

  1. "DELETE " requires a literal trailing SPACE. With
     MEMORY_ALLOW_DESTRUCTIVE=false, "MATCH (m:Memory)\\nDETACH\\nDELETE\\nm;"
     reported ordinary success and deleted every memory in the database.
  2. Only removal was considered, so read_only=True permitted overwrites:
     "MATCH (m:Memory {id: 1}) SET m.content = 'OVERWRITTEN';" succeeded and
     replaced the content.

A safety control that reports enforcement it does not have is worse than no
control, so these are regression tests for the parser, not the wording.
"""

import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/guard-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    server._conn = None
    server._db = None
    monkeypatch.setattr(server, "ALLOW_DESTRUCTIVE_QUERIES", False)
    yield
    server._conn = None
    server._db = None


def _seed(n=4):
    for i in range(n):
        server.memory_store.__wrapped__(
            content=f"Fact number {i} about service alpha-{i}.", tags=[f"svc{i}"])
    return server._count_memories(server.get_conn())


BYPASSES = [
    ("newline DETACH DELETE", "MATCH (m:Memory)\nDETACH\nDELETE\nm;"),
    ("tab DELETE", "MATCH (m:Memory)\tDELETE\tm;"),
    ("block-comment split", "MATCH (m:Memory) /*x*/DELETE/*y*/ m;"),
    ("line-comment prefix", "// harmless\nMATCH (m:Memory) DETACH DELETE m;"),
    ("multiple spaces", "MATCH (m:Memory) DETACH     DELETE     m;"),
    ("DROP procedure call", "CALL DROP_VECTOR_INDEX('Memory','memory_vec_idx');"),
    ("DROP table", "DROP TABLE Memory;"),
    ("TRUNCATE", "TRUNCATE TABLE Memory;"),
    ("REMOVE property", "MATCH (m:Memory) REMOVE m.category;"),
    ("SET overwrite", "MATCH (m:Memory) SET m.content = 'OVERWRITTEN';"),
    ("newline SET on embedding", "MATCH (m:Memory {id:1})\nSET\nm.embedding = [0.1];"),
]


@pytest.mark.parametrize("label,query", BYPASSES, ids=[b[0] for b in BYPASSES])
def test_destructive_queries_are_blocked_and_change_nothing(label, query):
    before = _seed()
    out = server.memory_query.__wrapped__(cypher_query=query)
    assert out.get("status") == "error", f"{label} was not blocked"
    assert server._count_memories(server.get_conn()) == before, \
        f"{label} mutated the database despite being 'blocked'"


@pytest.mark.parametrize("query", [
    "MATCH (m:Memory {id:1}) SET m.content = 'X';",
    "CREATE (m:Memory {id: 9001});",
    "MERGE (t:Topic {name: 'sneaky'});",
    "MATCH (m:Memory) DETACH DELETE m;",
])
def test_read_only_rejects_every_mutation(query):
    """read_only=True used to permit CREATE/MERGE/SET, making the name untrue."""
    before = _seed()
    out = server.memory_query.__wrapped__(cypher_query=query, read_only=True)
    assert out.get("status") == "error"
    assert "read_only" in out["message"]
    assert server._count_memories(server.get_conn()) == before


@pytest.mark.parametrize("query", [
    "MATCH (m:Memory) RETURN m.id LIMIT 2;",
    "MATCH (m:Memory) RETURN COUNT(m);",
    # A keyword inside a string literal is data, not code.
    "MATCH (m:Memory) WHERE m.content = 'DELETE me' RETURN m.id;",
    # A keyword only inside a comment must not trip the guard either.
    "// DELETE nothing\nMATCH (m:Memory) RETURN COUNT(m);",
])
def test_reads_are_not_blocked(query):
    _seed()
    out = server.memory_query.__wrapped__(cypher_query=query, read_only=True)
    assert out.get("status") != "error", out


def test_additive_writes_are_allowed_by_default():
    """CREATE adds; it does not destroy. Only read_only should stop it."""
    _seed()
    out = server.memory_query.__wrapped__(cypher_query="CREATE (t:Topic {name: 'legit'});")
    assert out.get("status") != "error", out


def test_destructive_still_works_when_explicitly_enabled(monkeypatch):
    before = _seed()
    monkeypatch.setattr(server, "ALLOW_DESTRUCTIVE_QUERIES", True)
    out = server.memory_query.__wrapped__(
        cypher_query="MATCH (m:Memory {id: 1}) DETACH DELETE m;")
    assert out.get("status") != "error", out
    assert server._count_memories(server.get_conn()) == before - 1


def test_oversized_and_empty_queries_are_rejected():
    _seed()
    assert server.memory_query.__wrapped__(
        cypher_query="MATCH (m) RETURN m; " + "x" * (server.MAX_QUERY_CHARS + 1)
    )["status"] == "error"
    assert server.memory_query.__wrapped__(cypher_query="   ")["status"] == "error"


def test_classifier_units():
    """The keyword matcher itself, independent of the tool."""
    assert server._is_destructive("MATCH (m)\nDELETE\nm") is True
    assert server._is_destructive("MATCH (m) SET m.x = 1") is True
    assert server._is_destructive("CALL DROP_FTS_INDEX('Memory','i')") is True
    assert server._is_destructive("MATCH (m) RETURN m") is False
    # 'deleted' must not read as DELETE — word boundaries, not substrings.
    assert server._is_destructive("MATCH (m) WHERE m.deleted RETURN m") is False
    assert server._is_write("CREATE (n)") is True
    assert server._is_write("MERGE (n)") is True
    assert server._is_write("MATCH (m) RETURN m") is False
    assert server._is_unsafe_embedding_set("MATCH (m)\tSET\tm.embedding = [1]") is True
