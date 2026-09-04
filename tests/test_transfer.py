"""memory_export / memory_import round-trip.

There was no backup path at all, which is uncomfortable for the single copy of
an agent's long-term memory: the database allows one writer, index state has
been observed to degrade across library upgrades, and memory_set_workspace
strands the old file rather than moving it.

Ids are remapped rather than preserved so a file can be merged into a database
that already has memories; edges are rewired onto the new ids.
"""

import os
import sys
import json
import tempfile

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/transfer-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    yield
    server._conn = None
    server._db = None


@pytest.fixture
def tmp_json(tmp_path):
    return str(tmp_path / "export.json")


def _seed():
    old = server.memory_store.__wrapped__(
        content="The Vega cache TTL is 60 seconds.", tags=["vega", "cache"], importance=2)
    new = server.memory_store.__wrapped__(
        content="Correction: the Vega cache TTL is 300 seconds.",
        tags=["vega", "cache"], importance=4, supersedes=old["id"])
    inc = server.memory_store.__wrapped__(
        content="INC-7700: Vega cache stampede during a deploy.", tags=["vega", "incident"])
    server.memory_relate.__wrapped__(from_id=inc["id"], to_id=new["id"],
                                     relationship="EXPLAINS")
    server.memory_relate.__wrapped__(from_id=old["id"], to_id=inc["id"],
                                     relationship="RELATED_TO", confidence=0.7)
    return old["id"], new["id"], inc["id"]


def _reset_db():
    """Drop to a fresh in-memory database, simulating a restore target."""
    server._conn = None
    server._db = None
    server.get_conn()


def test_export_writes_memories_and_edges(tmp_json):
    _seed()
    out = server.memory_export.__wrapped__(path=tmp_json)
    assert out["status"] == "exported"
    assert out["memories"] == 3
    assert out["edges"] == {"related_to": 1, "supersedes": 1, "explains": 1}

    payload = json.load(open(tmp_json))
    assert payload["format"] == server.EXPORT_FORMAT
    assert payload["embedding_dim"] == server.EMBEDDING_DIM
    assert payload["includes_embeddings"] is False
    assert len(payload["memories"]) == 3


def test_round_trip_restores_content_metadata_and_edges(tmp_json):
    _seed()
    server.memory_export.__wrapped__(path=tmp_json)
    _reset_db()
    assert server._count_memories(server.get_conn()) == 0

    out = server.memory_import.__wrapped__(path=tmp_json)
    assert out["status"] == "imported"
    assert out["stored_new"] == 3
    assert out["edges_created"] == {"related_to": 1, "supersedes": 1, "explains": 1}

    conn = server.get_conn()
    assert server._count_memories(conn) == 3
    # Importance survived, so ranking behaves the same after a restore.
    imps = sorted(r[0] for r in server._collect_results(
        conn.execute("MATCH (m:Memory) RETURN m.importance;")))
    assert imps == [2, 3, 4]
    # The correction still supersedes the original, by its NEW id.
    sup = server._collect_results(conn.execute(
        "MATCH (a:Memory)-[:SUPERSEDES]->(b:Memory) RETURN a.content, b.content;"))
    assert len(sup) == 1
    assert "300 seconds" in sup[0][0] and "60 seconds" in sup[0][1]


def test_restored_database_is_searchable(tmp_json):
    _seed()
    server.memory_export.__wrapped__(path=tmp_json)
    _reset_db()
    server.memory_import.__wrapped__(path=tmp_json)

    out = server.memory_search.__wrapped__(query="what is the Vega cache TTL", top_k=3)
    assert out["results"], "a restored database must be searchable"
    # Supersession demotion still applies after the restore.
    strict = server.memory_search.__wrapped__(
        query="what is the Vega cache TTL", top_k=3, include_superseded=False)
    assert "300 seconds" in strict["results"][0]["content"]


def test_dry_run_changes_nothing(tmp_json):
    _seed()
    server.memory_export.__wrapped__(path=tmp_json)
    _reset_db()

    out = server.memory_import.__wrapped__(path=tmp_json, dry_run=True)
    assert out["status"] == "preview"
    assert out["memories"] == 3
    assert server._count_memories(server.get_conn()) == 0


def test_reimport_merges_instead_of_duplicating(tmp_json):
    _seed()
    server.memory_export.__wrapped__(path=tmp_json)

    out = server.memory_import.__wrapped__(path=tmp_json)
    assert out["stored_new"] == 0
    assert out["merged_into_existing"] == 3
    assert server._count_memories(server.get_conn()) == 3
    # And edges are not doubled.
    for rel in ("SUPERSEDES", "EXPLAINS", "RELATED_TO"):
        rows = server._collect_results(server.get_conn().execute(
            f"MATCH ()-[r:{rel}]->() RETURN COUNT(r);"))
        assert rows[0][0] == 1, f"{rel} was duplicated by re-import"


def test_embeddings_can_be_carried_and_reused(tmp_json):
    _seed()
    out = server.memory_export.__wrapped__(path=tmp_json, include_embeddings=True)
    assert out["includes_embeddings"] is True
    payload = json.load(open(tmp_json))
    assert len(payload["memories"][0]["embedding"]) == server.EMBEDDING_DIM

    _reset_db()
    res = server.memory_import.__wrapped__(path=tmp_json)
    assert res["reused_embeddings"] is True
    assert res["stored_new"] == 3


def test_malformed_and_missing_files_are_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "something-else"}')
    assert server.memory_import.__wrapped__(path=str(bad))["status"] == "error"

    assert server.memory_import.__wrapped__(
        path=str(tmp_path / "nope.json"))["status"] == "error"

    notjson = tmp_path / "notjson.json"
    notjson.write_text("this is not json")
    assert server.memory_import.__wrapped__(path=str(notjson))["status"] == "error"


def test_future_format_version_is_refused(tmp_path):
    f = tmp_path / "future.json"
    f.write_text(json.dumps({
        "format": server.EXPORT_FORMAT,
        "format_version": server.EXPORT_FORMAT_VERSION + 1,
        "memories": [],
    }))
    out = server.memory_import.__wrapped__(path=str(f))
    assert out["status"] == "error"
    assert "newer" in out["message"]


def test_export_default_path_is_written(tmp_path, monkeypatch):
    """With no path, the export lands next to the database rather than nowhere.

    DB_PATH must be patched AFTER seeding: get_conn() re-resolves it on connect
    (the workspace root is adopted on the first tool call), so patching earlier
    is silently reverted.
    """
    _seed()
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "memory.lbug"))
    out = server.memory_export.__wrapped__()
    assert out["status"] == "exported"
    assert os.path.isfile(out["path"])
    assert out["path"].startswith(str(tmp_path))
