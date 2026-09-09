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
    assert s["workspaces_in_db"] == ["/db-scope-test"]
    assert "warning" not in s


def test_shared_database_is_reported_with_a_warning():
    """Two projects in one file: the silent misconfiguration."""
    server.memory_store.__wrapped__(content="Project alpha owns the ledger runbook.")
    server.WORKSPACE = "/other-project"
    server.memory_store.__wrapped__(content="Project beta owns the gateway runbook.")

    s = _scope()
    assert s["private_to_workspace"] is False
    assert set(s["workspaces_in_db"]) == {"/db-scope-test", "/other-project"}
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
    assert s["db_path"] == server.DB_PATH
    assert s["workspace"] == server.WORKSPACE
    assert s["workspace_source"] == server._workspace_source
