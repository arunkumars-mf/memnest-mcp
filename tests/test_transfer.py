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


# --- real disaster recovery ---------------------------------------------------
#
# The round-trip tests above restore into a fresh IN-MEMORY database, which
# proves the import logic. It does not prove recovery: a backup you have never
# restored to a real database file is not a backup. This exercises the actual
# DR shape — a separate .lbug file, opened cold, with embeddings recomputed
# because the default export deliberately omits them — and checks the property
# that matters, which is that RETRIEVAL works afterwards rather than that the
# counts line up.

def test_restore_into_a_separate_database_file_recovers_retrieval(tmp_path,
                                                                  monkeypatch):
    _seed()
    server.memory_store.__wrapped__(
        content="The archival vault rotates signing keys every 17 days "
                "under runbook R-9017.", tags=["vault"])
    export = str(tmp_path / "backup.json")
    out = server.memory_export.__wrapped__(path=export)
    assert out["status"] == "exported"
    assert json.load(open(export))["includes_embeddings"] is False, \
        "the default backup omits embeddings, so restore must re-embed"

    # Cold, separate, on-disk restore target.
    server._conn = None
    server._db = None
    target = str(tmp_path / "restored" / "memory.lbug")
    monkeypatch.setenv("MEMORY_DB_PATH", target)
    monkeypatch.setattr(server, "DB_PATH", target)
    conn = server.get_conn()
    assert server._count_memories(conn) == 0, "restore target must start empty"

    res = server.memory_import.__wrapped__(path=export)
    assert res["status"] == "imported"
    assert res["reused_embeddings"] is False

    conn = server.get_conn()
    total = server._count_memories(conn)
    embedded = server._collect_results(conn.execute(
        "MATCH (m:Memory) WHERE m.embedding IS NOT NULL RETURN COUNT(m);"))[0][0]
    assert embedded == total, "every restored memory must be re-embedded"
    assert server._probe_vector_index(conn, k=embedded) == embedded, \
        "the restored index must be fully reachable"

    # Retrieval, not just row counts.
    hit = server.memory_search.__wrapped__(
        query="archival vault signing key rotation runbook", top_k=3)
    assert any("R-9017" in r["content"] for r in hit["results"]), \
        "a restored memory must be findable by content"

    # And the graph still does the thing the graph is for.
    cur = server.memory_search.__wrapped__(query="what is the Vega cache TTL",
                                           top_k=3)
    assert "300 seconds" in cur["results"][0]["content"], \
        "supersession must still resolve to the current value after a restore"

    server._conn = None
    server._db = None


# --- exports must not disclose filesystem paths ------------------------------
#
# Workspace values are absolute paths, and they appeared once per memory plus
# once in the header — a 38-memory export disclosed the user's directory layout
# in 39 places. Exports are the artefact people attach to bug reports, support
# threads and shared fixtures, so sharing is the primary use, not an edge case.
# Import never reads the field, so scrubbing is free.

def test_export_does_not_leak_the_workspace_path(tmp_json, monkeypatch):
    secret = "/Users/someone/private-project-name"
    monkeypatch.setattr(server, "WORKSPACE", secret)
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.",
                                    tags=["ledger"])

    server.memory_export.__wrapped__(path=tmp_json)
    raw = open(tmp_json).read()
    assert secret not in raw, "the export discloses the workspace path"

    payload = json.loads(raw)
    assert payload["workspace"] == "workspace-1"
    assert payload["memories"][0]["workspace"] == "workspace-1"
    assert payload["workspace_paths_included"] is False


def test_paths_can_be_kept_deliberately_for_a_local_backup(tmp_json, monkeypatch):
    secret = "/Users/someone/private-project-name"
    monkeypatch.setattr(server, "WORKSPACE", secret)
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.",
                                    tags=["ledger"])

    server.memory_export.__wrapped__(path=tmp_json, include_workspace_paths=True)
    payload = json.loads(open(tmp_json).read())
    assert payload["workspace"] == secret
    assert payload["workspace_paths_included"] is True


def test_scrubbing_preserves_distinctions_between_workspaces(tmp_json, monkeypatch):
    """A global export must still distinguish workspaces — labels, not paths."""
    monkeypatch.setattr(server, "WORKSPACE", "/Users/someone/project-a")
    server.memory_store.__wrapped__(content="Project A owns the ledger runbook.",
                                    tags=["a"])
    monkeypatch.setattr(server, "WORKSPACE", "/Users/someone/project-b")
    server.memory_store.__wrapped__(content="Project B owns the gateway runbook.",
                                    tags=["b"])

    server.memory_export.__wrapped__(path=tmp_json, global_export=True)
    raw = open(tmp_json).read()
    assert "project-a" not in raw and "project-b" not in raw
    labels = {m["workspace"] for m in json.loads(raw)["memories"]}
    assert labels == {"workspace-1", "workspace-2"}, \
        f"distinct workspaces must stay distinct after scrubbing, got {labels}"


def test_scrubbed_export_still_round_trips(tmp_json, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", "/Users/someone/private-project-name")
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.",
                                    tags=["ledger"])
    server.memory_export.__wrapped__(path=tmp_json)

    _reset_db()
    res = server.memory_import.__wrapped__(path=tmp_json)
    assert res["status"] == "imported"
    assert server.memory_search.__wrapped__(
        query="ledger pipeline", top_k=3)["results"], \
        "a scrubbed export must remain importable and searchable"
