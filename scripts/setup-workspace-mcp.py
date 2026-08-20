#!/usr/bin/env python3
"""Verify and configure workspace-level memnest MCP config for Kiro.

Kiro launches power-installed MCP servers from the power's install
directory (Agent Plugins spec) or '/' (legacy powers), so those servers
cannot discover your project from cwd. A workspace-level entry in
<project>/.kiro/settings/mcp.json pins the scope explicitly via
MEMORY_WORKSPACE, which always wins over auto-detection.

This script verifies the current state and merges a memnest server entry
into the workspace config without touching other servers.

Usage:
  python3 setup-workspace-mcp.py            # verify + fix current directory
  python3 setup-workspace-mcp.py --root P   # target project P instead of cwd
  python3 setup-workspace-mcp.py --check    # verify only, exit 1 if not configured

After running, reconnect MCP servers in Kiro (or restart the window) so
the new config takes effect. Keep the memnest power installed: it still
provides the skills and IDE hooks; this workspace server provides
correctly-scoped tools.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

SERVER_NAME = "memnest"
AUTO_APPROVE = [
    "memory_store", "memory_search", "memory_update", "memory_delete",
    "memory_relate", "memory_dream", "memory_query", "memory_schema",
    "memory_topics", "memory_stats", "memory_graph_html", "memory_get",
    "memory_list", "memory_traverse", "memory_set_workspace",
]


def is_bogus_workspace(path: str) -> bool:
    """Mirror the server's rejection rules: never a real project."""
    home = os.path.expanduser("~")
    kiro = os.path.join(home, ".kiro")
    return (
        not path
        or path == "/"
        or "${" in path
        or path == home
        or path == kiro
        or path.startswith(kiro + os.sep)
    )


def desired_entry(root: str) -> dict:
    return {
        "command": "uvx",
        "args": ["memnest-mcp@latest"],
        "env": {
            "MEMORY_WORKSPACE": root,
            "FASTMCP_LOG_LEVEL": "ERROR",
        },
        "disabled": False,
        "autoApprove": AUTO_APPROVE,
    }


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def verify(root: str, config_path: str) -> tuple[bool, list[str]]:
    """Return (configured_correctly, report_lines)."""
    lines = []
    ok = True

    if is_bogus_workspace(root):
        lines.append(f"ERROR: {root!r} is not a valid project workspace "
                     f"('/', home, and paths under ~/.kiro are rejected)")
        return False, lines
    lines.append(f"workspace:    {root}")
    lines.append(f"config file:  {config_path} "
                 f"({'exists' if os.path.exists(config_path) else 'missing'})")

    cfg = {}
    try:
        cfg = load_config(config_path)
    except (json.JSONDecodeError, OSError) as e:
        lines.append(f"ERROR: cannot parse config: {e}")
        return False, lines

    servers = cfg.get("mcpServers", {})
    entry = servers.get(SERVER_NAME)
    if entry is None:
        lines.append(f"memnest:      NOT configured at workspace level")
        ok = False
    else:
        ws = entry.get("env", {}).get("MEMORY_WORKSPACE", "")
        disabled = entry.get("disabled", False)
        lines.append(f"memnest:      configured, MEMORY_WORKSPACE={ws!r}"
                     f"{' (DISABLED)' if disabled else ''}")
        if ws != root:
            lines.append(f"              MISMATCH: expected {root!r}")
            ok = False
        if disabled:
            ok = False

    others = [k for k in servers if k != SERVER_NAME]
    if others:
        lines.append(f"other servers: {', '.join(others)} (untouched)")

    # Informational: any memnest server processes currently running
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "memnest"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        n = len([l for l in out.splitlines() if "memnest" in l])
        if n:
            lines.append(f"running memnest processes: {n} "
                         f"(reconnect MCP servers in Kiro after changes)")
    except Exception:
        pass

    db = os.path.join(root, ".memnest", "memory.lbug")
    lines.append(f"project DB:   {db} "
                 f"({'exists' if os.path.exists(db) else 'will be created on first use'})")
    return ok, lines


def fix(root: str, config_path: str) -> list[str]:
    """Merge the memnest entry into the workspace config. Returns report."""
    lines = []
    cfg = load_config(config_path)

    if os.path.exists(config_path):
        backup = config_path + ".bak"
        shutil.copy2(config_path, backup)
        lines.append(f"backup:       {backup}")

    servers = cfg.setdefault("mcpServers", {})
    existing = servers.get(SERVER_NAME)
    if existing is None:
        servers[SERVER_NAME] = desired_entry(root)
        lines.append(f"memnest:      added with MEMORY_WORKSPACE={root!r}")
    else:
        # Non-destructive: only pin the workspace and re-enable; keep any
        # customizations (different command, extra env, trimmed autoApprove).
        existing.setdefault("env", {})["MEMORY_WORKSPACE"] = root
        existing["disabled"] = False
        existing.setdefault("command", "uvx")
        existing.setdefault("args", ["memnest-mcp@latest"])
        lines.append(f"memnest:      updated, MEMORY_WORKSPACE={root!r}")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    lines.append(f"wrote:        {config_path}")
    lines.append("next:         reconnect MCP servers in Kiro (command palette: "
                 "'MCP' -> reconnect), then check memory_stats -> runtime")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.getcwd(),
                    help="project root (default: current directory)")
    ap.add_argument("--check", action="store_true",
                    help="verify only; exit 1 if not configured correctly")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    config_path = os.path.join(root, ".kiro", "settings", "mcp.json")

    ok, report = verify(root, config_path)
    print("\n".join(report))

    if args.check:
        print(f"\nstatus: {'OK' if ok else 'NOT CONFIGURED'}")
        return 0 if ok else 1

    if ok:
        print("\nstatus: already configured, nothing to do")
        return 0

    if is_bogus_workspace(root):
        return 1

    print()
    print("\n".join(fix(root, config_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
