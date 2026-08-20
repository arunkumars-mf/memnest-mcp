"""Command-line interface for memnest-mcp.

memnest-mcp                  Run the MCP server over stdio (default, used
                             by MCP host configs like `uvx memnest-mcp@latest`)
memnest-mcp serve            Same as above, explicit
memnest-mcp config kiro      Configure the current project for Kiro:
                             workspace-level MCP config + agent hooks

`config` is always workspace-level: it writes into <project>/.kiro/.
Client-specific subcommands keep room for other assistants with different
config layouts (e.g. a future `config claude-code`).
"""

import argparse
import json
import os
import shutil
import sys

SERVER_NAME = "memnest"
AUTO_APPROVE = [
    "memory_store", "memory_search", "memory_update", "memory_delete",
    "memory_relate", "memory_dream", "memory_query", "memory_schema",
    "memory_topics", "memory_stats", "memory_graph_html", "memory_get",
    "memory_list", "memory_traverse", "memory_set_workspace",
]

# --- Kiro agent hooks (direct MCP tool usage — no power required) ---

_RECALL_PROMPT = (
    "Before responding to the user's request, recall relevant context from "
    "memory using the memnest MCP server tools.\n\n"
    "Retrieval strategy:\n"
    "- If the user's message has a clear topic/question: call memory_search "
    "with {\"query\": \"<derived query>\", \"top_k\": 5, \"preview_chars\": 300}.\n"
    "- If specific domains are involved, add \"tags\" to narrow results "
    "(e.g. tags=[\"python\",\"backend\"] for language questions, "
    "tags=[\"architecture\"] for design questions). Tags disambiguate "
    "overloaded terms.\n"
    "- If the message is vague, a greeting, or lacks a searchable topic: call "
    "memory_list with {\"limit\": 5, \"sort\": \"importance\"} instead.\n"
    "- Only call one retrieval tool — not both.\n\n"
    "The search uses hybrid scoring (vector similarity 40% + BM25 full-text "
    "30% + graph PageRank/community 15% + recency 10% + importance 5%), so "
    "natural language queries work well.\n\n"
    "Pay special attention to recalling:\n"
    "- Prior architecture decisions and design rationale relevant to the current task\n"
    "- Known bug root causes and fixes for the packages being discussed\n"
    "- Package-specific conventions, gotchas, or workflows\n"
    "- User preferences and working patterns from past sessions\n"
    "- Related memories that form a knowledge cluster (the graph expansion "
    "handles this automatically)\n\n"
    "If relevant memories are found, incorporate them naturally into your "
    "approach. Do not mention the memory system to the user unless they ask "
    "about it."
)

_PERSIST_PROMPT = (
    "Review this conversation for any new learnings, user preferences, "
    "decisions, or patterns worth remembering, and store them using the "
    "memnest MCP server tools.\n\n"
    "What to store — pay special attention to:\n"
    "- Project architecture decisions and design rationale\n"
    "- Bug root causes and their fixes\n"
    "- Package-specific gotchas or conventions (build, deploy, review workflows)\n"
    "- Important technical context about the project or platform\n"
    "- User preferences, working style, and tool choices\n"
    "- Recurring patterns or workflows the user follows\n\n"
    "How to store — use these features for maximum retrieval quality:\n"
    "1. Use BATCH mode when storing multiple memories: call memory_store with "
    "{\"items\": [{\"content\": \"...\", \"category\": \"...\", \"tags\": [...], "
    "\"importance\": N}, ...]}. This is faster than multiple single calls.\n"
    "2. Choose the right CATEGORY: \"learning\" (facts, how things work), "
    "\"preference\" (user choices), \"decision\" (architecture/tool picks with "
    "rationale), \"pattern\" (recurring workflows), \"general\" (other).\n"
    "3. Add specific TAGS for each memory — these become graph Topic nodes "
    "enabling traversal. Use lowercase, specific terms (e.g. [\"python\", "
    "\"fastapi\", \"error-handling\"] not [\"code\"]).\n"
    "4. Set IMPORTANCE appropriately: 1=trivial, 2=low, 3=neutral, "
    "4=important, 5=critical. Decisions and preferences should be 4-5; "
    "routine facts 2-3.\n"
    "5. Link related memories with SUPERSEDES relationships: if new "
    "information corrects or updates something previously stored, call "
    "memory_relate with {\"from_id\": <new_id>, \"to_id\": <old_id>, "
    "\"relationship\": \"SUPERSEDES\"} so the graph reflects the correction "
    "chain.\n\n"
    "Skip anything trivial, ephemeral, or already stored (the system "
    "auto-deduplicates via hash + semantic similarity, but don't rely on it "
    "for obviously redundant stores). If nothing worth storing, do nothing — "
    "no explanation needed."
)

_DREAM_PROMPT = (
    "Run memory consolidation using the memnest MCP server tools.\n\n"
    "Step 1 — Preview: Call memory_dream with {\"dry_run\": true} to see what "
    "would happen without making changes. Review the output.\n\n"
    "Step 2 — Execute: If the preview looks reasonable, call memory_dream "
    "with {\"force\": true} to run consolidation. This will:\n"
    "- Recompute graph algorithms (PageRank, Louvain communities, K-Core) for "
    "better search ranking\n"
    "- Auto-prune memories older than 30 days with importance <= 2\n"
    "- Auto-merge near-duplicates (similarity >= 0.95)\n"
    "- Surface clusters with similarity 0.88-0.95 for your review\n"
    "- Detect SCC contradictions (circular SUPERSEDES chains)\n\n"
    "Step 3 — Handle clusters: For each surfaced cluster:\n"
    "- Use your judgment — some related memories are intentionally distinct "
    "(different concerns, different contexts)\n"
    "- For clusters that SHOULD merge: write a comprehensive merged memory "
    "with memory_store, then delete the old ones with memory_delete\n"
    "- For clusters that should stay separate: optionally link them with "
    "memory_relate using {\"relationship\": \"RELATED_TO\", \"confidence\": 0.9} "
    "so the graph captures their relationship without merging\n\n"
    "Step 4 — Report: Summarize what was pruned, merged, kept separate, and "
    "any contradictions detected."
)

# Hook files use the v1 schema introduced in Kiro IDE 1.0 / CLI 3.0:
# standalone .json files in .kiro/hooks/ with PascalCase trigger names.
# See https://kiro.dev/docs/hooks
KIRO_HOOKS = {
    "memnest-recall.json": {
        "version": "v1",
        "hooks": [
            {
                "name": "Memnest: Recall Relevant Memories",
                "description": "Before responding to each user prompt, recall "
                               "relevant context from Memnest memory using hybrid "
                               "search (vector + FTS + graph) to provide continuity "
                               "across sessions.",
                "trigger": "UserPromptSubmit",
                "action": {"type": "agent", "prompt": _RECALL_PROMPT},
                "enabled": True,
            }
        ],
    },
    "memnest-persist.json": {
        "version": "v1",
        "hooks": [
            {
                "name": "Memnest: Persist Learnings to Memory",
                "description": "After the agent finishes a task, review the "
                               "conversation for new learnings, preferences, "
                               "decisions, or patterns and batch-store them in "
                               "Memnest memory.",
                "trigger": "Stop",
                "action": {"type": "agent", "prompt": _PERSIST_PROMPT},
                "enabled": True,
            }
        ],
    },
}

# The old 'userTriggered' hook trigger was removed in Kiro IDE 1.0 and replaced
# by manual steering files, invoked as /memnest-dream (or #memnest-dream) in chat.
KIRO_STEERING = {
    "memnest-dream.md": (
        "---\n"
        "inclusion: manual\n"
        'description: "Consolidate Memnest memory — run graph algorithms, prune '
        'stale memories, merge near-duplicates, and review surfaced clusters."\n'
        "---\n"
        "\n"
        "# Memnest: Consolidate Memory\n"
        "\n" + _DREAM_PROMPT + "\n"
    ),
}

# Files written by earlier versions of this command (pre-1.0 hook format).
# Removed on reconfigure so hooks don't double-fire and Kiro stops offering
# to migrate them.
LEGACY_KIRO_HOOKS = (
    "memnest-recall.kiro.hook",
    "memnest-persist.kiro.hook",
    "memnest-dream.kiro.hook",
)


def _is_bogus_workspace(path: str) -> bool:
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


def _server_entry(root: str, command_mode: str) -> dict:
    if command_mode == "local":
        command, args = "memnest-mcp", []
    else:
        command, args = "uvx", ["memnest-mcp@latest"]
    return {
        "command": command,
        "args": args,
        "env": {
            "MEMORY_WORKSPACE": root,
            "FASTMCP_LOG_LEVEL": "ERROR",
        },
        "disabled": False,
        "autoApprove": AUTO_APPROVE,
    }


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _verify_kiro(root: str, require_hooks: bool = True) -> tuple[bool, list[str]]:
    """Return (configured_correctly, report_lines)."""
    lines = []
    ok = True
    config_path = os.path.join(root, ".kiro", "settings", "mcp.json")

    if _is_bogus_workspace(root):
        lines.append(f"ERROR: {root!r} is not a valid project workspace "
                     f"('/', home, and paths under ~/.kiro are rejected)")
        return False, lines

    lines.append(f"workspace:    {root}")
    lines.append(f"config file:  {config_path} "
                 f"({'exists' if os.path.exists(config_path) else 'missing'})")

    try:
        cfg = _load_json(config_path)
    except (json.JSONDecodeError, OSError) as e:
        lines.append(f"ERROR: cannot parse config: {e}")
        return False, lines

    servers = cfg.get("mcpServers", {})
    entry = servers.get(SERVER_NAME)
    if entry is None:
        lines.append("memnest:      NOT configured at workspace level")
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

    hooks_dir = os.path.join(root, ".kiro", "hooks")
    present = [h for h in KIRO_HOOKS if os.path.exists(os.path.join(hooks_dir, h))]
    lines.append(f"hooks:        {len(present)}/{len(KIRO_HOOKS)} installed "
                 f"in {hooks_dir}")
    if require_hooks and len(present) < len(KIRO_HOOKS):
        ok = False

    steering_dir = os.path.join(root, ".kiro", "steering")
    s_present = [s for s in KIRO_STEERING
                 if os.path.exists(os.path.join(steering_dir, s))]
    lines.append(f"steering:     {len(s_present)}/{len(KIRO_STEERING)} installed "
                 f"(/memnest-dream)")
    if require_hooks and len(s_present) < len(KIRO_STEERING):
        ok = False

    stale = [h for h in LEGACY_KIRO_HOOKS
             if os.path.exists(os.path.join(hooks_dir, h))]
    if stale:
        lines.append(f"legacy hooks: {', '.join(stale)} (pre-1.0 format, "
                     f"will be removed)")
        ok = False

    db = os.path.join(root, ".memnest", "memory.lbug")
    lines.append(f"project DB:   {db} "
                 f"({'exists' if os.path.exists(db) else 'created on first use'})")
    return ok, lines


def _fix_kiro(root: str, command_mode: str, with_hooks: bool, force: bool) -> list[str]:
    """Merge the memnest entry + hooks into <root>/.kiro/. Returns report."""
    lines = []
    config_path = os.path.join(root, ".kiro", "settings", "mcp.json")
    cfg = _load_json(config_path)

    if os.path.exists(config_path):
        backup = config_path + ".bak"
        shutil.copy2(config_path, backup)
        lines.append(f"backup:       {backup}")

    servers = cfg.setdefault("mcpServers", {})
    existing = servers.get(SERVER_NAME)
    if existing is None:
        servers[SERVER_NAME] = _server_entry(root, command_mode)
        lines.append(f"memnest:      added with MEMORY_WORKSPACE={root!r}")
    else:
        # Non-destructive: pin the workspace and re-enable; keep any
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

    if with_hooks:
        hooks_dir = os.path.join(root, ".kiro", "hooks")
        os.makedirs(hooks_dir, exist_ok=True)
        for filename, hook in KIRO_HOOKS.items():
            path = os.path.join(hooks_dir, filename)
            if os.path.exists(path) and not force:
                lines.append(f"hook:         {filename} exists, skipped (--force to overwrite)")
                continue
            with open(path, "w") as f:
                json.dump(hook, f, indent=2)
                f.write("\n")
            lines.append(f"hook:         {filename} written")

        # Drop hooks written in the pre-1.0 format by earlier versions of this
        # command. Leaving them causes duplicate firing and migration prompts.
        for filename in LEGACY_KIRO_HOOKS:
            path = os.path.join(hooks_dir, filename)
            if os.path.exists(path):
                os.remove(path)
                lines.append(f"removed:      {filename} (pre-1.0 hook format)")

        steering_dir = os.path.join(root, ".kiro", "steering")
        os.makedirs(steering_dir, exist_ok=True)
        for filename, body in KIRO_STEERING.items():
            path = os.path.join(steering_dir, filename)
            if os.path.exists(path) and not force:
                lines.append(f"steering:     {filename} exists, skipped (--force to overwrite)")
                continue
            with open(path, "w") as f:
                f.write(body)
            lines.append(f"steering:     {filename} written (run with /memnest-dream)")

    lines.append("next:         reconnect MCP servers in Kiro (command palette: "
                 "'MCP') or restart the window, then check memory_stats -> runtime")
    return lines


def _cmd_config_kiro(args: argparse.Namespace) -> int:
    root = os.path.abspath(os.path.expanduser(args.root))

    ok, report = _verify_kiro(root, require_hooks=not args.no_hooks)
    print("\n".join(report))

    if args.check:
        print(f"\nstatus: {'OK' if ok else 'NOT CONFIGURED'}")
        return 0 if ok else 1

    if _is_bogus_workspace(root):
        return 1

    if ok:
        print("\nstatus: already configured, nothing to do")
        return 0

    print()
    print("\n".join(_fix_kiro(root, args.command, not args.no_hooks, args.force)))
    return 0


def main() -> int:
    argv = sys.argv[1:]

    # Default action (no arguments): run the MCP server. This keeps every
    # existing host config (`uvx memnest-mcp@latest`, `python -m memnest_mcp`)
    # working unchanged. Heavy imports happen only on this path.
    if not argv or argv[0] == "serve":
        from .server import main as serve
        serve()
        return 0

    parser = argparse.ArgumentParser(
        prog="memnest-mcp",
        description="Memnest memory MCP server and project configurator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the MCP server over stdio (default)")

    p_config = sub.add_parser("config", help="configure a project for a client")
    config_sub = p_config.add_subparsers(dest="client", required=True)

    p_kiro = config_sub.add_parser(
        "kiro",
        help="write workspace-level Kiro config: .kiro/settings/mcp.json + .kiro/hooks/",
    )
    p_kiro.add_argument("--root", default=os.getcwd(),
                        help="project root (default: current directory)")
    p_kiro.add_argument("--check", action="store_true",
                        help="verify only; exit 1 if not configured")
    p_kiro.add_argument("--no-hooks", action="store_true",
                        help="configure the MCP server only, skip agent hooks")
    p_kiro.add_argument("--force", action="store_true",
                        help="overwrite existing hook files")
    p_kiro.add_argument("--command", choices=["uvx", "local"], default="uvx",
                        help="how the config launches the server: uvx (default, "
                             "always latest) or local (memnest-mcp on PATH)")
    p_kiro.set_defaults(func=_cmd_config_kiro)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        from .server import main as serve
        serve()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
