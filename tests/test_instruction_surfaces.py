"""The instruction surfaces are as much a contract as the tool signatures.

0.30.0 added `memory_keep_separate`; five hint surfaces inside the server were
updated to name it and *seven* other places that instruct an agent were not: the
three prompts in cli.py, the two power hook JSONs, the power's autoApprove list,
and the getting-started skill. The consequence was concrete rather than cosmetic
-- an agent following the shipped hooks was taught an option set from which the
new tool was absent, and the dream steering actively recommended the option
0.30.1 had demoted. Enumerating emission sites inside server.py and stopping
there is what made a fix look complete for two releases.

These tests treat "every tool is either auto-approved or deliberately excluded"
and "the duplicated prompt text agrees" as assertions, so the next tool added
cannot be silently missing from the places that tell an agent it exists.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memnest_mcp import cli  # noqa: E402
from memnest_mcp import server  # noqa: E402

POWER = ROOT / "power" / "memnest"
HOOKS = POWER / "dev.kiro" / "hooks"


def _live_tool_names() -> set:
    """Tool names from the server itself, not from a list someone maintained."""
    names = set()
    for attr in dir(server):
        obj = getattr(server, attr)
        if attr.startswith("memory_") and hasattr(obj, "__wrapped__"):
            names.add(attr)
    assert len(names) >= 15, f"tool discovery looks broken: {sorted(names)}"
    return names


def test_every_tool_is_either_auto_approved_or_deliberately_excluded():
    live = _live_tool_names()
    approved = set(cli.AUTO_APPROVE)
    excluded = set(cli.NOT_AUTO_APPROVED)

    unaccounted = live - approved - excluded
    assert not unaccounted, (
        f"tools nobody decided about: {sorted(unaccounted)}. Add each to "
        f"AUTO_APPROVE or to NOT_AUTO_APPROVED so the choice is recorded. A tool "
        f"that is neither prompts the user on every call, which in practice "
        f"means an agent stops using it.")

    assert not (approved & excluded), "a tool cannot be both approved and excluded"
    phantom = (approved | excluded) - live
    assert not phantom, f"listed but not a real tool: {sorted(phantom)}"


def test_keep_separate_is_auto_approved():
    """It records a verdict, creates no edge, and is idempotent — the lowest-risk
    write in the set, and useless behind a prompt: an agent that must interrupt
    to dismiss a flag leaves the flag firing forever.
    """
    assert "memory_keep_separate" in cli.AUTO_APPROVE


def test_power_hook_prompts_match_cli():
    """The power ships JSON and cannot import cli.py, so the text is duplicated.
    Run scripts/sync_power_instructions.py when this fails.
    """
    for name, expected in (("memnest-recall.json", cli._RECALL_PROMPT),
                           ("memnest-persist.json", cli._PERSIST_PROMPT)):
        doc = json.loads((HOOKS / name).read_text())
        actual = doc["hooks"][0]["action"]["prompt"]
        assert actual == expected, (
            f"{name} has drifted from cli.py. Run "
            f"scripts/sync_power_instructions.py")


def test_power_auto_approve_matches_cli():
    plugin = json.loads((POWER / "plugin.json").read_text())
    assert plugin["extensions"]["dev.kiro"]["autoApprove"] == list(cli.AUTO_APPROVE)


@pytest.mark.parametrize("prompt_name", ["_PERSIST_PROMPT", "_DREAM_PROMPT"])
def test_prompts_that_resolve_conflicts_name_keep_separate(prompt_name):
    """Both prompts tell an agent how to resolve a same-subject conflict, so both
    need the "both hold and are distinct" option. Asserting per-prompt rather
    than "somewhere in cli.py" is the point: the gap was one surface having it
    and another not.
    """
    prompt = getattr(cli, prompt_name)
    assert "memory_keep_separate" in prompt, (
        f"{prompt_name} enumerates conflict resolutions without the one that "
        f"records 'both hold'")


def test_dream_prompt_does_not_recommend_related_to_for_distinct_clusters():
    """RELATED_TO clears the search-time flag but does NOT stop dream re-offering
    the cluster (measured), so recommending it for clusters judged distinct means
    answering the same question on every run.
    """
    prompt = cli._DREAM_PROMPT
    idx = prompt.find("stay separate")
    assert idx != -1, "the dream prompt no longer covers the stay-separate case"
    window = prompt[idx:idx + 400]
    assert "memory_keep_separate" in window
    assert "optionally link them with memory_relate" not in prompt


def test_recall_prompt_handles_untrusted_content_and_cycles():
    """Screening stored content is worth nothing if the consumer of a recalled
    memory is never told to treat a flagged one as data.
    """
    prompt = cli._RECALL_PROMPT
    assert "untrusted_content" in prompt
    assert "supersession_cycle" in prompt


def test_recall_prompt_does_not_hardcode_channel_weights():
    """The weights are wrong under MEMORY_FUSION=rrf, where only ranks enter the
    score, and wrong whenever MEMORY_GRAPH_WEIGHT is changed — which the README
    recommends considering. A derived description that can drift is worse than
    none.
    """
    prompt = cli._RECALL_PROMPT
    for stale in ("40%", "30%", "15%", "10%", "5%"):
        assert stale not in prompt, f"hardcoded channel weight {stale} in prompt"
