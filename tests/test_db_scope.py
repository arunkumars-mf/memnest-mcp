"""The 1:1 workspace -> database invariant, reported from evidence.

Why it matters: this server does no locking on the connection, the embedding
model or the repair counters. That is safe because the deployment model is one
MCP connection per workspace and one database per workspace — never two
callers. The invariant therefore deserves a check rather than an assumption.

Two ways it can break:
  concurrent  two live clients on one file. Already loud: LadybugDB refuses
              the second writer and get_conn explains it.
  sequential  a globally pinned MEMORY_DB_PATH; project A and project B open
              the same file in turn and both graphs land in one database.
              Silent — retrieval stays workspace-scoped, so nothing surfaces.

The sequential case is what these tests cover.
"""

import json
import os
import sys

os.environ.setdefault("MEMORY_DB_PATH", ":memory:")
os.environ.setdefault("MEMORY_WORKSPACE", "/db-scope-test")
os.environ.setdefault("MEMORY_RESPONSE_FORMAT", "json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import server


@pytest.fixture(autouse=True)
def clean():
    server._conn = None
    server._db = None
    server.WORKSPACE = "/db-scope-test"
    yield
    server._conn = None
    server._db = None
    server.WORKSPACE = "/db-scope-test"


def _scope():
    return server.memory_stats.__wrapped__()["runtime"]["db_scope"]


def test_single_workspace_database_reports_private():
    server.memory_store.__wrapped__(content="Only this project's fact about shipping.")
    s = _scope()
    assert s["private_to_workspace"] is True
    # Identity, not path — see the redaction tests below.
    assert s["workspaces_in_db"] == [server._redact_path("/db-scope-test")]
    assert "warning" not in s


def test_shared_database_is_reported_with_a_warning():
    """Two projects in one file: the silent misconfiguration."""
    server.memory_store.__wrapped__(content="Project alpha owns the ledger runbook.")
    server.WORKSPACE = "/other-project"
    server.memory_store.__wrapped__(content="Project beta owns the gateway runbook.")

    s = _scope()
    assert s["private_to_workspace"] is False
    assert set(s["workspaces_in_db"]) == {
        server._redact_path("/db-scope-test"), server._redact_path("/other-project")}
    assert "warning" in s
    assert "one database per workspace" in s["warning"]


def test_scope_report_survives_an_empty_database():
    s = _scope()
    assert s["workspaces_in_db"] == []
    assert s["private_to_workspace"] is True


def test_scope_report_names_the_resolution_inputs():
    """A user debugging this needs to see WHY the paths resolved as they did."""
    server.memory_store.__wrapped__(content="A fact to make the database non-empty.")
    s = _scope()
    assert s["db_path"] == server._redact_path(server.DB_PATH)
    assert s["workspace"] == server._redact_path(server.WORKSPACE)
    assert s["workspace_source"] == server._workspace_source


# --- stats must not disclose the filesystem tree ------------------------------
#
# Same class as the export leak, with a LARGER surface: memory_stats carried the
# absolute path in five places, and it is pasted reflexively — it is the first
# thing anyone shares when asking "is my index healthy". The export at least had
# to be deliberately attached.
#
# It differs in that the path is diagnostically load-bearing here, so scrubbing
# must not cost diagnostic value. The resolution: the derived answers
# (db_inside_workspace, private_to_workspace, the workspace count) are computed
# server-side and stay valid regardless, so only IDENTITY needs to survive.
# basename#hash gives identity without naming the tree.

def test_stats_does_not_disclose_the_workspace_path(monkeypatch, tmp_path):
    secret = str(tmp_path / "private-project-name")
    monkeypatch.setattr(server, "WORKSPACE", secret)
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.")

    st = server.memory_stats.__wrapped__()
    assert secret not in json.dumps(st), "stats discloses the workspace path"
    assert st["workspace"].startswith("private-project-name#")


def test_stats_keeps_every_derived_diagnostic_while_redacted(monkeypatch, tmp_path):
    """Scrubbing must not cost diagnostic value — that is the whole design."""
    monkeypatch.setattr(server, "WORKSPACE", str(tmp_path / "proj"))
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.")

    scope = server.memory_stats.__wrapped__()["runtime"]["db_scope"]
    assert scope["private_to_workspace"] is True
    assert scope["db_inside_workspace"] in (True, False, None)
    assert len(scope["workspaces_in_db"]) == 1
    assert scope["workspace_source"] == server._workspace_source


def test_paths_available_on_request_for_local_debugging(monkeypatch, tmp_path):
    secret = str(tmp_path / "private-project-name")
    monkeypatch.setattr(server, "WORKSPACE", secret)
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.")

    st = server.memory_stats.__wrapped__(include_paths=True)
    assert st["workspace"] == secret
    assert st["runtime"]["db_scope"]["workspace"] == secret


def test_redacted_identity_is_stable_and_distinguishing():
    assert server._redact_path("/a/proj") == server._redact_path("/a/proj"), \
        "two calls on one database must produce the same identity"
    assert server._redact_path("/a/proj") != server._redact_path("/a/other"), \
        "different databases must be distinguishable"
    assert server._redact_path("/one/app") != server._redact_path("/two/app"), \
        "same basename in a different tree must still be distinguishable"
    assert server._redact_path(":memory:") == ":memory:"


# --- the scrub happens at the serialization boundary, not per call site -------
#
# Rule 3 applied to the whole class rather than to the two sites that had
# already bitten. ~17 places return engine error text verbatim via str(e), and
# engine errors embed file paths on IO and lock failures — so patching the
# places that build responses would leave the next str(e) free to reopen the
# hole. Scrubbing on the way out covers every tool, present and future.

def test_engine_error_text_is_scrubbed():
    faked = f"IO error: could not read {server.DB_PATH} (errno 5)"
    scrubbed = server._scrub_paths(faked)
    if server.DB_PATH != ":memory:":
        assert server.DB_PATH not in scrubbed
    assert "IO error" in scrubbed, "scrubbing must not destroy the message"


def test_scrub_leaves_unrelated_content_alone():
    """Deliberately narrow: only THIS server's paths are substituted."""
    text = "The user mentioned /etc/hosts and C:\\Windows in a memory."
    assert server._scrub_paths(text) == text


def test_export_returns_a_filename_the_caller_can_use():
    """The scrub reaches the export's own returned path, so the filename is
    carried separately — it has no directory component to disclose, and it is
    what the caller needs to find the file inside a workspace it knows."""
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "backup.json")
    server.memory_store.__wrapped__(content="A fact about the ledger pipeline.")
    out = server.memory_export.__wrapped__(path=p)
    assert out["filename"] == "backup.json"
    assert os.sep not in out["filename"]
