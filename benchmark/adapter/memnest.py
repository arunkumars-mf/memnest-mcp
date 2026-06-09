"""Memnest memory adapter for LOCOMO benchmark.

Calls the server.py tool functions directly (no MCP transport overhead).
Each conversation gets its own isolated DB file under benchmark/results/dbs/.
"""

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

# Make src importable
_SRC_PATH = str(Path(__file__).parent.parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Module-level lock: the server module uses global state (_conn, _db) so
# concurrent access from ThreadPoolExecutor threads must be serialized.
_server_lock = threading.Lock()


def _set_db_for_conversation(conv_id: str, results_dir: Path):
    """Point the server at a fresh per-conversation DB before importing."""
    db_dir = results_dir / "dbs" / conv_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "memory.lbug"
    os.environ["MEMORY_DB_PATH"] = str(db_path)
    os.environ["MEMORY_WORKSPACE"] = conv_id
    os.environ["MEMORY_RESPONSE_FORMAT"] = "json"
    # Disable dream auto-trigger during ingest
    os.environ["MEMORY_DREAM_MIN_OPS"] = "999999"
    return db_path


def _reset_server_module():
    """Force re-import of the server module so env changes take effect.

    IMPORTANT: must close the existing DB connection first, otherwise the WAL
    file is left in a dirty state and the next open gets a corruption error.
    """
    # Close existing connection/database if loaded
    try:
        from memnest_mcp import server as _srv
        if _srv._conn is not None:
            try:
                _srv._conn.close()
            except Exception:
                pass
            _srv._conn = None
        if _srv._db is not None:
            try:
                _srv._db.close()
            except Exception:
                pass
            _srv._db = None
    except (ImportError, AttributeError):
        pass

    mods_to_drop = [m for m in list(sys.modules) if m.startswith("memnest_mcp")]
    for m in mods_to_drop:
        del sys.modules[m]


def reset_db_dir(results_dir: Path):
    """Delete all per-conversation DBs from a previous run."""
    db_root = results_dir / "dbs"
    if db_root.exists():
        shutil.rmtree(db_root)


def ingest_conversation(conv_id: str, conv_data: dict, results_dir: Path):
    """Ingest one LOCOMO conversation into a fresh memory DB.

    Strategy: dual-granularity storage.
      1. Individual turns — for precise single-hop retrieval.
      2. Sliding-window chunks (5 turns, stride 2) — for multi-hop and temporal
         context. Each chunk carries the session date as an anchor.

    This roughly doubles the memory count but dramatically improves recall for
    questions that need to connect information across adjacent turns.
    """
    with _server_lock:
        _set_db_for_conversation(conv_id, results_dir)
        _reset_server_module()
        from memnest_mcp import server  # type: ignore

        convo = conv_data["conversation"]
        items = []

        # Collect all turns per session first (for windowing)
        sessions: list[tuple[str, str, list[dict]]] = []  # (session_num, date_str, turns)
        for key, val in convo.items():
            if not key.startswith("session_"):
                continue
            if key.endswith("_date_time"):
                continue
            if not isinstance(val, list):
                continue
            session_num = key.split("_")[1]
            date_key = f"session_{session_num}_date_time"
            date_str = convo.get(date_key, "")
            sessions.append((session_num, date_str, val))

        for session_num, date_str, turns in sessions:
            # --- Layer 1: Individual turns ---
            turn_texts = []
            for turn in turns:
                speaker = turn.get("speaker", "unknown")
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                if not text:
                    turn_texts.append("")
                    continue
                content = (
                    f"[{date_str}] [{speaker}] {text}"
                    if date_str
                    else f"[{speaker}] {text}"
                )
                turn_texts.append(content)
                items.append({
                    "content": content,
                    "category": "general",
                    "tags": [conv_id, f"session-{session_num}", speaker.lower(), dia_id.lower()],
                })

            # --- Layer 2: Sliding-window chunks (5 turns, stride 2) ---
            window_size = 5
            stride = 2
            non_empty = [(i, t) for i, t in enumerate(turn_texts) if t]
            for start_idx in range(0, len(non_empty), stride):
                window = non_empty[start_idx:start_idx + window_size]
                if len(window) < 2:
                    continue  # skip single-turn windows (already stored individually)
                chunk_content = "\n".join(t for _, t in window)
                # Collect unique speakers in this window
                speakers = list({turns[i].get("speaker", "").lower()
                                 for i, _ in window if i < len(turns)})
                items.append({
                    "content": chunk_content,
                    "category": "general",
                    "tags": [conv_id, f"session-{session_num}", "chunk"] + speakers,
                    "importance": 2,  # slightly lower than individual turns
                })

        # Batch insert in chunks of 50 to avoid huge embedding calls
        chunk_size = 50
        stored = 0
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            res = server.memory_store.__wrapped__(items=chunk)
            if isinstance(res, str):
                res = json.loads(res)
            stored += res.get("count", len(chunk))

        # Run dream (consolidation) to merge near-duplicates from the
        # sliding-window overlap and prune noise.
        dream_res = server.memory_dream.__wrapped__(force=True)
        if isinstance(dream_res, str):
            dream_res = json.loads(dream_res)
        pruned = dream_res.get("pruned", 0)
        merged = dream_res.get("auto_merged", 0)

    return {
        "conv_id": conv_id,
        "turns_stored": stored,
        "total_items": len(items),
        "dream_pruned": pruned,
        "dream_merged": merged,
    }


def search_memories(conv_id: str, question: str, results_dir: Path,
                    top_k: int = 20, preview_chars: int = 1000) -> list[dict]:
    """Retrieve top_k memories for a question. Returns list of {id, content, score}."""
    with _server_lock:
        _set_db_for_conversation(conv_id, results_dir)
        _reset_server_module()
        from memnest_mcp import server  # type: ignore

        res = server.memory_search.__wrapped__(
            query=question,
            top_k=top_k,
            preview_chars=preview_chars,
        )
        if isinstance(res, str):
            res = json.loads(res)
        # New shape: {"results": [...]}
        items = res.get("results", res) if isinstance(res, dict) else res
        return items if isinstance(items, list) else []


# Track which conversation the server is currently pointed at so we can skip
# the expensive reset+reinit when consecutive questions target the same DB.
_current_conv_id: str = ""
_current_results_dir: str = ""


def setup_server_for_conversation(conv_id: str, results_dir: Path):
    """Point the server module at a conversation's DB for tool-calling agents.

    Caches the current conversation — if the same conv_id and results_dir are
    requested again, skips the expensive module reset and DB re-initialization.

    Thread-safe: acquires _server_lock. The caller should NOT hold the lock.
    """
    global _current_conv_id, _current_results_dir
    rd_str = str(results_dir)

    with _server_lock:
        if _current_conv_id == conv_id and _current_results_dir == rd_str:
            # Already pointing at the right DB — nothing to do.
            return

        _set_db_for_conversation(conv_id, results_dir)
        _reset_server_module()
        # Force the module to initialize the connection
        from memnest_mcp import server  # type: ignore  # noqa: F401
        server.get_conn()
        _current_conv_id = conv_id
        _current_results_dir = rd_str
