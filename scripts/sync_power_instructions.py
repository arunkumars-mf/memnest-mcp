#!/usr/bin/env python3
"""Sync the power's hook prompts and autoApprove list from src/memnest_mcp/cli.py.

The same instruction text exists twice: once in `cli.py` (written into a user's
.kiro/ by `memnest-mcp config kiro`) and once in `power/memnest/`, which ships
standalone JSON and cannot import Python. That duplication silently drifted --
0.30.0 added `memory_keep_separate`, five hint surfaces in the server were
updated to name it, and neither copy of the hook prompts nor the autoApprove
lists were touched, so agents driven by the hooks were still taught the
pre-0.30.0 option set months later.

`cli.py` is the single source. Run this after editing a prompt there;
tests/test_instruction_surfaces.py fails if the two copies disagree.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memnest_mcp import cli  # noqa: E402

POWER = ROOT / "power" / "memnest"
TARGETS = {
    POWER / "dev.kiro" / "hooks" / "memnest-recall.json": cli._RECALL_PROMPT,
    POWER / "dev.kiro" / "hooks" / "memnest-persist.json": cli._PERSIST_PROMPT,
}


def main() -> int:
    changed = []

    for path, prompt in TARGETS.items():
        doc = json.loads(path.read_text())
        hook = doc["hooks"][0]
        if hook["action"].get("prompt") != prompt:
            hook["action"]["prompt"] = prompt
            path.write_text(json.dumps(doc, indent=2) + "\n")
            changed.append(path.name)

    plugin_path = POWER / "plugin.json"
    plugin = json.loads(plugin_path.read_text())
    approve = plugin["extensions"]["dev.kiro"]["autoApprove"]
    if approve != list(cli.AUTO_APPROVE):
        plugin["extensions"]["dev.kiro"]["autoApprove"] = list(cli.AUTO_APPROVE)
        plugin_path.write_text(json.dumps(plugin, indent=2) + "\n")
        changed.append("plugin.json")

    print("synced: " + (", ".join(changed) if changed else "nothing (already in sync)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
