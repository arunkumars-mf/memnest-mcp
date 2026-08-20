"""Tests for the memnest-mcp CLI (config kiro).

The CLI module must import without pulling in the server's heavy
dependencies; serving is dispatched lazily.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from memnest_mcp import cli


def run_config(root, *extra):
    argv_backup = sys.argv
    sys.argv = ["memnest-mcp", "config", "kiro", "--root", str(root), *extra]
    try:
        return cli.main()
    finally:
        sys.argv = argv_backup


def read_cfg(root):
    with open(os.path.join(root, ".kiro", "settings", "mcp.json")) as f:
        return json.load(f)


def test_fresh_project_configures_server_and_hooks(tmp_path, capsys):
    assert run_config(tmp_path) == 0
    cfg = read_cfg(tmp_path)
    entry = cfg["mcpServers"]["memnest"]
    assert entry["command"] == "uvx"
    assert entry["args"] == ["memnest-mcp@latest"]
    assert entry["env"]["MEMORY_WORKSPACE"] == str(tmp_path)
    assert "memory_set_workspace" in entry["autoApprove"]

    hooks_dir = tmp_path / ".kiro" / "hooks"
    assert sorted(os.listdir(hooks_dir)) == sorted(cli.KIRO_HOOKS)
    for name in cli.KIRO_HOOKS:
        hook = json.loads((hooks_dir / name).read_text())
        assert hook["then"]["type"] == "askAgent"
        assert "kiroPowers" not in hook["then"]["prompt"], \
            "CLI hooks must use direct MCP tools, not the power indirection"


def test_check_mode_fails_then_passes(tmp_path):
    assert run_config(tmp_path, "--check") == 1
    assert run_config(tmp_path) == 0
    assert run_config(tmp_path, "--check") == 0


def test_merge_preserves_other_servers_and_backs_up(tmp_path):
    settings = tmp_path / ".kiro" / "settings"
    settings.mkdir(parents=True)
    (settings / "mcp.json").write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "python", "args": ["x.py"],
                             "env": {"KEY": "value"}}
        }
    }))
    assert run_config(tmp_path) == 0
    cfg = read_cfg(tmp_path)
    assert cfg["mcpServers"]["other-server"]["env"]["KEY"] == "value"
    assert cfg["mcpServers"]["memnest"]["env"]["MEMORY_WORKSPACE"] == str(tmp_path)
    assert (settings / "mcp.json.bak").exists()


def test_idempotent_second_run(tmp_path, capsys):
    assert run_config(tmp_path) == 0
    capsys.readouterr()
    assert run_config(tmp_path) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_repairs_wrong_workspace_keeps_custom_env(tmp_path):
    assert run_config(tmp_path) == 0
    cfg_path = tmp_path / ".kiro" / "settings" / "mcp.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mcpServers"]["memnest"]["env"]["MEMORY_WORKSPACE"] = "/somewhere/else"
    cfg["mcpServers"]["memnest"]["env"]["CUSTOM"] = "keep-me"
    cfg_path.write_text(json.dumps(cfg))

    assert run_config(tmp_path) == 0
    entry = read_cfg(tmp_path)["mcpServers"]["memnest"]
    assert entry["env"]["MEMORY_WORKSPACE"] == str(tmp_path)
    assert entry["env"]["CUSTOM"] == "keep-me"


def test_existing_hooks_skipped_without_force(tmp_path):
    assert run_config(tmp_path) == 0
    hook = tmp_path / ".kiro" / "hooks" / "memnest-recall.kiro.hook"
    hook.write_text('{"custom": true}')

    # Force a config change so fix runs again
    cfg_path = tmp_path / ".kiro" / "settings" / "mcp.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mcpServers"]["memnest"]["disabled"] = True
    cfg_path.write_text(json.dumps(cfg))

    assert run_config(tmp_path) == 0
    assert json.loads(hook.read_text()) == {"custom": True}, "hook overwritten without --force"

    cfg = json.loads(cfg_path.read_text())
    cfg["mcpServers"]["memnest"]["disabled"] = True
    cfg_path.write_text(json.dumps(cfg))
    assert run_config(tmp_path, "--force") == 0
    assert json.loads(hook.read_text())["name"].startswith("Memnest"), "not overwritten with --force"


def test_no_hooks_flag(tmp_path, capsys):
    assert run_config(tmp_path, "--no-hooks") == 0
    assert "memnest" in read_cfg(tmp_path)["mcpServers"]
    assert not (tmp_path / ".kiro" / "hooks").exists()
    # idempotent under the same flag; hooks absence is not a defect here
    capsys.readouterr()
    assert run_config(tmp_path, "--no-hooks") == 0
    assert "nothing to do" in capsys.readouterr().out


def test_local_command_mode(tmp_path):
    run_config(tmp_path, "--command", "local")
    entry = read_cfg(tmp_path)["mcpServers"]["memnest"]
    assert entry["command"] == "memnest-mcp"
    assert entry["args"] == []


def test_bogus_roots_rejected():
    home = os.path.expanduser("~")
    for root in ("/", home, os.path.join(home, ".kiro", "powers", "installed", "x")):
        assert run_config(root) == 1


def test_cli_has_no_module_level_server_import():
    """The config path must not load embedding/DB libraries."""
    src = open(cli.__file__).read()
    module_level = [
        line for line in src.splitlines()
        if line.startswith(("import ", "from "))  # top-level only, no indent
    ]
    for heavy in ("server", "fastembed", "real_ladybug"):
        assert not any(heavy in line for line in module_level), \
            f"{heavy} imported at module level: {module_level}"
