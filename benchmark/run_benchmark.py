"""LOCOMO benchmark runner for the Memnest memory MCP server.

Phases:
  1. ingest — load each conversation into a per-conversation memory DB + dream
  2. answer — Strands agent uses memory tools to answer each question
  3. judge  — Strands agent scores predictions vs ground truth
  4. score  — compute final scorecard by category

The answer agent has full access to all memory tools (search, get, query, topics)
and decides what to retrieve on its own — the agentic memory pattern.

Usage:
  python benchmark/run_benchmark.py --phase all
  python benchmark/run_benchmark.py --phase all --max-convs 1 --max-questions 20
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "locomo10.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PRED_PATH = RESULTS_DIR / "predictions.json"
JUDGE_PATH = RESULTS_DIR / "judgments.json"
SCORES_PATH = RESULTS_DIR / "scores.json"

# Make adapter importable
sys.path.insert(0, str(ROOT))


CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


def load_data(max_convs: int = None) -> list:
    data = json.loads(DATA_PATH.read_text())
    if max_convs:
        data = data[:max_convs]
    return data


def phase_ingest(max_convs: int = None, force: bool = False):
    """Phase 1: ingest conversations into memory, then run dream consolidation."""
    from adapter.memnest import ingest_conversation, reset_db_dir

    data = load_data(max_convs)
    print(f"=== INGEST: {len(data)} conversations ===", flush=True)

    if force:
        reset_db_dir(RESULTS_DIR)
        print("Reset existing DBs.", flush=True)

    summary = []
    for i, conv in enumerate(data):
        conv_id = conv.get("sample_id", f"conv-{i}")
        t0 = time.time()
        result = ingest_conversation(conv_id, conv, RESULTS_DIR)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        summary.append(result)
        dream_info = ""
        if result.get("dream_merged") or result.get("dream_pruned"):
            dream_info = f" (dream: merged={result.get('dream_merged',0)}, pruned={result.get('dream_pruned',0)})"
        facts_info = ""
        if result.get("facts_extracted"):
            facts_info = f" (facts: {result.get('facts_extracted',0)})"
        print(f"  [{i + 1}/{len(data)}] {conv_id}: {result['turns_stored']} turns "
              f"in {elapsed:.1f}s{dream_info}{facts_info}", flush=True)

    (RESULTS_DIR / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nIngest done. Summary: {RESULTS_DIR}/ingest_summary.json", flush=True)


def phase_answer(max_convs: int = None, max_questions: int = None):
    """Phase 2: Retrieve context + generate answer for each question."""
    from adapter.memnest import setup_server_for_conversation
    from adapter.strands_backend import agent_answer, reset_answer_agent

    data = load_data(max_convs)
    print(f"=== ANSWER: {len(data)} conversations, max_questions={max_questions} ===",
          flush=True)

    # Flatten questions across convs
    tasks = []
    for conv in data:
        conv_id = conv.get("sample_id", f"conv-{len(tasks)}")
        for q_idx, qa in enumerate(conv["qa"]):
            if max_questions and len(tasks) >= max_questions:
                break
            tasks.append({
                "conv_id": conv_id,
                "q_idx": q_idx,
                "question": qa["question"],
                "ground_truth": qa.get("answer", ""),
                "category": qa.get("category"),
            })
        if max_questions and len(tasks) >= max_questions:
            break

    print(f"Total questions: {len(tasks)}", flush=True)

    # Resume support: load existing predictions
    existing = {}
    if PRED_PATH.exists():
        for p in json.loads(PRED_PATH.read_text()):
            existing[(p["conv_id"], p["q_idx"])] = p
        print(f"Resuming with {len(existing)} existing predictions.", flush=True)

    results = []
    completed = 0
    t0 = time.time()
    current_conv_id = None

    for task in tasks:
        key = (task["conv_id"], task["q_idx"])
        if key in existing:
            results.append(existing[key])
            completed += 1
            continue

        # Reset agent when switching conversations (container pinning pattern)
        if task["conv_id"] != current_conv_id:
            reset_answer_agent()
            current_conv_id = task["conv_id"]

        # Point the server module at this conversation's DB
        try:
            setup_server_for_conversation(task["conv_id"], RESULTS_DIR)
        except Exception as e:
            results.append({**task, "predicted": "", "error": f"setup: {e}"})
            completed += 1
            continue

        # Let the Strands agent use tools to find and answer
        try:
            predicted = agent_answer(task["question"])
        except Exception as e:
            results.append({**task, "predicted": "", "error": f"agent: {e}"})
            completed += 1
            continue

        results.append({**task, "predicted": predicted})
        completed += 1

        if completed % 10 == 0 or completed == len(tasks):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(tasks)}] elapsed {elapsed:.0f}s  "
                  f"rate {rate:.1f}/s  ETA {eta:.0f}s", flush=True)

        # Periodic checkpoint
        if completed % 25 == 0:
            PRED_PATH.write_text(json.dumps(results, indent=2))

    PRED_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nAnswer phase done. {len(results)} predictions → {PRED_PATH}", flush=True)


def phase_judge():
    """Phase 3: judge predictions vs ground truth."""
    from adapter.strands_backend import judge_answer

    if not PRED_PATH.exists():
        print(f"ERROR: No predictions at {PRED_PATH}. Run answer phase first.")
        return

    predictions = json.loads(PRED_PATH.read_text())
    print(f"=== JUDGE: {len(predictions)} predictions ===", flush=True)

    # Resume
    existing = {}
    if JUDGE_PATH.exists():
        for j in json.loads(JUDGE_PATH.read_text()):
            existing[(j["conv_id"], j["q_idx"])] = j
        print(f"Resuming with {len(existing)} existing judgments.", flush=True)

    results = []
    completed = 0
    t0 = time.time()

    for pred in predictions:
        key = (pred["conv_id"], pred["q_idx"])
        if key in existing:
            results.append(existing[key])
            completed += 1
            continue

        if pred.get("error") or not pred.get("predicted"):
            results.append({**pred, "score": 0, "reasoning": "no prediction"})
            completed += 1
            continue

        try:
            result = judge_answer(
                pred["question"],
                pred["predicted"],
                pred["ground_truth"],
            )
            results.append({**pred, "score": result["score"], "reasoning": result["reasoning"]})
        except Exception as e:
            results.append({**pred, "score": 0, "reasoning": f"judge error: {e}"})

        completed += 1
        if completed % 10 == 0 or completed == len(predictions):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(predictions) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(predictions)}] elapsed {elapsed:.0f}s  "
                  f"rate {rate:.1f}/s  ETA {eta:.0f}s", flush=True)

        if completed % 25 == 0:
            JUDGE_PATH.write_text(json.dumps(results, indent=2))

    JUDGE_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nJudge phase done. → {JUDGE_PATH}", flush=True)


def phase_score():
    """Phase 4: compute final scores by category."""
    if not JUDGE_PATH.exists():
        print(f"ERROR: No judgments at {JUDGE_PATH}.")
        return

    judgments = json.loads(JUDGE_PATH.read_text())

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for j in judgments:
        cat = j.get("category", "unknown")
        cat_name = CATEGORY_NAMES.get(cat, str(cat))
        by_cat[cat_name]["total"] += 1
        by_cat[cat_name]["correct"] += int(j.get("score", 0))

    overall_correct = sum(c["correct"] for c in by_cat.values())
    overall_total = sum(c["total"] for c in by_cat.values())

    scores = {
        "by_category": {},
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "score_pct": round(100 * overall_correct / overall_total, 2) if overall_total else 0,
        },
    }
    for cat, vals in by_cat.items():
        scores["by_category"][cat] = {
            "correct": vals["correct"],
            "total": vals["total"],
            "score_pct": round(100 * vals["correct"] / vals["total"], 2)
            if vals["total"] else 0,
        }

    SCORES_PATH.write_text(json.dumps(scores, indent=2))

    print("\n========== LOCOMO SCORECARD ==========")
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'Score':>10}")
    print("-" * 48)
    for cat in ["single_hop", "multi_hop", "open_domain", "temporal", "adversarial"]:
        if cat in scores["by_category"]:
            v = scores["by_category"][cat]
            print(f"{cat:<20} {v['correct']:>8} {v['total']:>8} {v['score_pct']:>9.2f}%")
    print("-" * 48)
    o = scores["overall"]
    print(f"{'OVERALL':<20} {o['correct']:>8} {o['total']:>8} {o['score_pct']:>9.2f}%")
    print(f"\nFull scores → {SCORES_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="LOCOMO benchmark for Memnest memory MCP server (Strands Agents)")
    parser.add_argument("--phase", choices=["ingest", "answer", "judge", "score", "all"],
                        default="all")
    parser.add_argument("--max-convs", type=int, default=None,
                        help="Limit number of conversations (for smoke test)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (for smoke test)")
    parser.add_argument("--force-reingest", action="store_true",
                        help="Wipe existing DBs before ingest")
    args = parser.parse_args()

    if args.phase in ("ingest", "all"):
        phase_ingest(max_convs=args.max_convs, force=args.force_reingest)
    if args.phase in ("answer", "all"):
        phase_answer(max_convs=args.max_convs, max_questions=args.max_questions)
    if args.phase in ("judge", "all"):
        phase_judge()
    if args.phase in ("score", "all"):
        phase_score()


if __name__ == "__main__":
    main()
