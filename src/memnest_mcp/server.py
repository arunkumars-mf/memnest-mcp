"""
Memnest Memory MCP Server
===========================
AI agent memory powered by LadybugDB — graph + vector search in one embedded database.

Graph data model:
  - Memory nodes (content, embeddings, metadata, workspace)
  - Topic nodes (auto-linked from tags)
  - Relationships: ABOUT, RELATED_TO, SUPERSEDES, EXPLAINS

Workspace namespacing:
  - MEMORY_WORKSPACE env var scopes memories to a project (always wins)
  - Otherwise the MCP client's first workspace root is adopted on the
    first tool call (roots/list), so per-project scoping works even from
    user-level MCP configs launched with cwd '/'
  - Falls back to cwd, then to global scope ('') when cwd is '/'
  - Pass global_search=True to bypass

Response format:
  - MEMORY_RESPONSE_FORMAT=toon (default if installed) for compact LLM-friendly output
  - MEMORY_RESPONSE_FORMAT=json for backward-compatible JSON
  - TOON typically reduces tokens by 30-60% vs JSON

Tools (see individual docstrings for params):
  Write      memory_store, memory_update, memory_delete
  Read       memory_search, memory_get, memory_list, memory_topics
  Graph      memory_relate, memory_unrelate, memory_query, memory_schema
  Maintain   memory_dream, memory_reindex, memory_stats, memory_set_workspace
  Transfer   memory_export, memory_import
  Visualise  memory_graph_html
  Deprecated memory_traverse (use memory_query with read_only=True)
"""

import hashlib
import html
import inspect
import json
import logging
import os
import re
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Literal, Optional

import anyio
import real_ladybug as lb
from fastembed import TextEmbedding
from mcp import types as mcp_types
from mcp.server.fastmcp import FastMCP

try:
    from toon_format import encode as _toon_encode  # type: ignore
    _TOON_AVAILABLE = True
except Exception:  # noqa: BLE001 — beta dep can fail in many ways at import
    _TOON_AVAILABLE = False
    _toon_encode = None  # type: ignore

# --- Logging ---
# Library policy: attach a NullHandler so we don't override host logging config.
# main() opts in to basicConfig when run as the entry point.
logger = logging.getLogger("memnest")
logger.addHandler(logging.NullHandler())

# --- Configuration ---
def _looks_unsubstituted(value: str) -> bool:
    """Detect config template placeholders that the MCP host did not expand.

    Users coming from VS Code write things like "${workspaceFolder}" in env
    values; Kiro (and other hosts) pass the literal string through. Treating
    it as a real path would scope memories to a directory named '${...}'.
    """
    return "${" in value


def _is_bogus_workspace(path: str) -> bool:
    """Paths that can never be a real project workspace.

    - '' or '/': MCP hosts often launch servers from '/'
    - unexpanded ${...} config placeholders
    - the home directory itself: hosts also launch from $HOME; a project
      that IS the home directory is pathological
    - anything under ~/.kiro: Kiro launches Agent Plugins-format power
      servers with cwd set to the power's INSTALL directory
      (~/.kiro/powers/installed/<name>). Accepting it would make every
      window share one database inside the install tree.
    """
    if not path or path == "/":
        return True
    if _looks_unsubstituted(path):
        return True
    home = os.path.expanduser("~")
    if path == home:
        return True
    kiro_dir = os.path.join(home, ".kiro")
    if path == kiro_dir or path.startswith(kiro_dir + os.sep):
        return True
    return False


def _resolve_db_path() -> str:
    """Determine the database path with sensible fallbacks.

    Priority:
    1. MEMORY_DB_PATH env var (explicit override)
    2. WORKSPACE/.memnest/memory.lbug — the resolved workspace scope
       (env var, adopted client root, or cwd), if it is a writable directory
    3. ~/.memnest/memory.lbug (global fallback)

    Called lazily at first connection (see get_conn), AFTER the client's
    workspace root may have been adopted. This gives each project its own
    database file, which matters beyond tidiness: LadybugDB allows only ONE
    read-write process per database file, so multiple IDE windows sharing a
    single global file would fight over the lock and all but one would fail.
    Per-project files keep windows on different projects out of each other's
    way.
    """
    explicit = os.environ.get("MEMORY_DB_PATH")
    if explicit and not _looks_unsubstituted(explicit):
        return explicit
    if explicit:
        logger.warning(f"Ignoring MEMORY_DB_PATH with unexpanded placeholder: {explicit}")

    if WORKSPACE and not _is_bogus_workspace(WORKSPACE) \
            and os.path.isdir(WORKSPACE) and os.access(WORKSPACE, os.W_OK):
        return os.path.join(WORKSPACE, ".memnest", "memory.lbug")

    # Global fallback
    return os.path.expanduser("~/.memnest/memory.lbug")


def _resolve_workspace() -> str:
    """Determine the workspace scope at import time, with sensible fallbacks.

    Priority:
    1. MEMORY_WORKSPACE env var (explicit override), unless it's '/'
    2. cwd, unless it's '/' (MCP hosts often launch servers from '/')
    3. '' (global scope — memories are visible to every workspace)

    This is only the static baseline. On the first tool call the server also
    asks the MCP client for its workspace roots (roots/list) and adopts the
    first one — see _adopt_client_workspace(). That makes workspace scoping
    work even when the server is registered in a user-level config and
    launched with cwd '/'. The env var always wins over roots.

    A workspace of '/' is never legitimate: it would silently merge all
    projects into one shared scope while looking like a real path.
    """
    global _workspace_source
    workspace = os.environ.get("MEMORY_WORKSPACE", "")
    if workspace and not _is_bogus_workspace(workspace):
        _workspace_source = "env"
        return workspace

    cwd = os.getcwd()
    if not _is_bogus_workspace(cwd):
        _workspace_source = "cwd"
        return cwd

    _workspace_source = "global"
    return ""


try:
    from importlib.metadata import version as _pkg_version
    SERVER_VERSION = _pkg_version("memnest-mcp")
except Exception:  # pragma: no cover - dev checkouts without install
    SERVER_VERSION = "unknown"

# How the current WORKSPACE value was determined:
# env | cwd | global | roots | manual (see memory_set_workspace)
_workspace_source: str = "unset"

# WORKSPACE must be resolved before DB_PATH: the DB path derives from it.
# Both are re-resolved at first connection (get_conn) once the client's
# workspace root may have been adopted via roots/list.
WORKSPACE = _resolve_workspace()
DB_PATH = _resolve_db_path()
EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("MEMORY_EMBEDDING_DIM", "384"))
DEDUP_THRESHOLD = float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92"))
LATENCY_WARN_MS = int(os.environ.get("MEMORY_LATENCY_WARN_MS", "200"))

# Refuse to merge two near-identical memories whose value-bearing tokens
# disagree (500ms vs 900ms, Kafka vs Kinesis, prod-checkout vs prod-inventory).
# Set MEMORY_MERGE_VALUE_GATE=0 to restore pure-similarity merging. Unsafe:
# without it, storing a corrected value can silently keep the stale one.
MERGE_VALUE_GATE = os.environ.get("MEMORY_MERGE_VALUE_GATE", "1").lower() not in (
    "0", "false", "no", "off",
)

# Vector-channel scaling in search fusion: 'legacy' (raw cosine) or
# 'normalized' (min-max, matching the FTS channel). Default 'legacy' because
# the published benchmark score was measured with it.
# Score multiplier for memories that a newer memory SUPERSEDES. 1.0 disables
# the demotion; 0.0 sinks them to the bottom of the ranking.
SUPERSEDED_PENALTY = float(os.environ.get("MEMORY_SUPERSEDED_PENALTY", "0.5"))

# Graph neighbours reported alongside search results: how many top hits to
# expand from, and how many connected memories to return. Set
# MEMORY_GRAPH_EXPAND_SEEDS=0 to disable.
GRAPH_EXPAND_SEEDS = int(os.environ.get("MEMORY_GRAPH_EXPAND_SEEDS", "3"))
GRAPH_EXPAND_LIMIT = int(os.environ.get("MEMORY_GRAPH_EXPAND_LIMIT", "5"))
# A seed must score at least this fraction of the top result to expand from.
# Weak hits are coincidences and their neighbours are noise.
GRAPH_EXPAND_MIN_RATIO = float(os.environ.get("MEMORY_GRAPH_EXPAND_MIN_RATIO", "0.6"))

# Flag returned memories that are near-duplicates of each other with no
# SUPERSEDES/EXPLAINS edge — likely competing versions nobody resolved.
CONFLICT_DETECTION = os.environ.get("MEMORY_CONFLICT_DETECTION", "1") not in ("0", "false", "no")
CONFLICT_THRESHOLD = float(os.environ.get("MEMORY_CONFLICT_THRESHOLD", "0.85"))
# Minimum Jaccard tag overlap for a similar pair to count as being about the
# same subject. Guards against flagging parallel facts about different services.
CONFLICT_TAG_OVERLAP = float(os.environ.get("MEMORY_CONFLICT_TAG_OVERLAP", "0.5"))
# Cosine floor for the value-disagreement path, which flags same-subject pairs
# whose VALUES differ regardless of how differently they are worded. Much lower
# than CONFLICT_THRESHOLD on purpose: a correction rewritten from scratch scores
# too low to look like a duplicate, which is exactly why cosine-only detection
# missed it. The floor only excludes pairs that merely share tags by accident.
CONFLICT_VALUE_FLOOR = float(os.environ.get("MEMORY_CONFLICT_VALUE_FLOOR", "0.5"))

# Minimum Jaccard tag overlap before two similar memories may be MERGED
# (store-time dedup and dream auto-merge). Both paths are destructive, so they
# get the strictest guard. Set to 0 to restore pure-similarity merging.
MERGE_TAG_OVERLAP = float(os.environ.get("MEMORY_MERGE_TAG_OVERLAP", "0.5"))

FUSION_MODE = os.environ.get("MEMORY_FUSION", "legacy").strip().lower()
if FUSION_MODE not in ("legacy", "normalized"):
    raise RuntimeError(
        f"Invalid MEMORY_FUSION={FUSION_MODE!r}: expected 'legacy' or 'normalized'."
    )

# Response format: 'json' (default) or 'toon' (compact for LLM context)
RESPONSE_FORMAT = os.environ.get("MEMORY_RESPONSE_FORMAT", "toon" if _TOON_AVAILABLE else "json").lower()

MAX_CONTENT_LENGTH = int(os.environ.get("MEMORY_MAX_CONTENT", "500"))

# Input caps. MAX_CONTENT_LENGTH above is an OUTPUT preview width, not an input
# limit — a 120,000-character memory was accepted, embedded and stored, which
# wastes the embedding budget (the model truncates to its context anyway) and
# bloats every preview path.
MAX_STORE_CHARS = int(os.environ.get("MEMORY_MAX_STORE_CHARS", "20000"))
MAX_TAG_CHARS = int(os.environ.get("MEMORY_MAX_TAG_CHARS", "80"))
MAX_TAGS_PER_MEMORY = int(os.environ.get("MEMORY_MAX_TAGS", "32"))
MAX_BATCH_ITEMS = int(os.environ.get("MEMORY_MAX_BATCH", "500"))
MAX_QUERY_CHARS = int(os.environ.get("MEMORY_MAX_QUERY_CHARS", "20000"))
MAX_PREVIEW_CHARS = int(os.environ.get("MEMORY_MAX_PREVIEW_CHARS", "4000"))
MAX_SEARCH_RESULTS = int(os.environ.get("MEMORY_SEARCH_LIMIT", "10"))
# Rows each search channel retrieves before fusion. Independent of top_k so the
# ranking does not depend on the page size — see the comment at the pool
# computation in memory_search for why this cannot change scores.
SEARCH_CANDIDATE_POOL = int(os.environ.get("MEMORY_SEARCH_CANDIDATES", "100"))
MAX_LIST_RESULTS = int(os.environ.get("MEMORY_LIST_LIMIT", "20"))
MAX_CONSOLIDATE_CLUSTERS = int(os.environ.get("MEMORY_CONSOLIDATE_CLUSTERS", "10"))
MAX_CONSOLIDATE_SCAN = int(os.environ.get("MEMORY_CONSOLIDATE_SCAN", "1000"))

# Dream (consolidation) settings
DREAM_MIN_OPERATIONS = int(os.environ.get("MEMORY_DREAM_MIN_OPS", "10"))
DREAM_MIN_INTERVAL_HOURS = float(os.environ.get("MEMORY_DREAM_MIN_HOURS", "24"))
DREAM_AUTO_PRUNE_DAYS = int(os.environ.get("MEMORY_DREAM_PRUNE_DAYS", "30"))
DREAM_AUTO_PRUNE_MAX_IMPORTANCE = int(os.environ.get("MEMORY_DREAM_PRUNE_MAX_IMP", "2"))
DREAM_TRIVIAL_MERGE_THRESHOLD = float(os.environ.get("MEMORY_DREAM_TRIVIAL_THRESHOLD", "0.95"))
# Lower bound of the cluster-review window. Anything between this and the trivial
# threshold is surfaced for agent review rather than auto-merged.
DREAM_CLUSTER_LOW_THRESHOLD = float(os.environ.get("MEMORY_DREAM_CLUSTER_LOW", "0.88"))
DREAM_MIN_MEMORIES = int(os.environ.get("MEMORY_DREAM_MIN_MEMORIES", "20"))

# Embedding model timeout (seconds). Currently used for warm-up only; per-call
# timeouts require a worker pool which is out of scope here.
EMBED_TIMEOUT_S = float(os.environ.get("MEMORY_EMBED_TIMEOUT_S", "30"))

# memory_graph_html safety caps — render budget guard
GRAPH_HTML_MAX_NODES = int(os.environ.get("MEMORY_GRAPH_MAX_NODES", "2000"))

# Validation: cluster window must be a non-empty band below the trivial threshold.
if DREAM_CLUSTER_LOW_THRESHOLD >= DREAM_TRIVIAL_MERGE_THRESHOLD:
    raise RuntimeError(
        f"Invalid configuration: MEMORY_DREAM_CLUSTER_LOW ({DREAM_CLUSTER_LOW_THRESHOLD}) "
        f"must be less than MEMORY_DREAM_TRIVIAL_THRESHOLD ({DREAM_TRIVIAL_MERGE_THRESHOLD})."
    )

# --- Globals ---
mcp = FastMCP("memnest")
_embed_model: Optional[TextEmbedding] = None
_conn: Optional[lb.Connection] = None
_db: Optional[lb.Database] = None

# Dream state tracking (in-memory, persisted to SchemaMeta)
_dream_ops_lock = threading.Lock()
_dream_ops_since_last: int = 0
_dream_last_time: float = 0.0

# Workspace adoption from MCP client roots.
# _roots_done goes True only when an attempt COMPLETES (adopted, client
# lacks the capability, or the attempt cap is hit) — never on cancellation,
# so a user-cancelled first tool call can't permanently disable adoption.
_roots_done: bool = False
_roots_attempts: int = 0
_ROOTS_MAX_ATTEMPTS = 3
_roots_lock = anyio.Lock()

# Diagnostics captured during adoption, surfaced via memory_stats.runtime so
# a single stats call answers "which client is this, does it support roots,
# and why is the workspace what it is?"
_client_info: Optional[dict] = None
_client_supports_roots: Optional[bool] = None

# Result of the vector-index health check performed at connection time.
_vector_index_state: dict = {}

# Repair-on-use budget. Rebuilding costs time proportional to memory count, so
# an index that cannot be fixed must not trigger a rebuild on every query.
_index_repair_attempts: int = 0
_index_repair_last: float = 0.0
INDEX_REPAIR_MAX_ATTEMPTS = int(os.environ.get("MEMORY_INDEX_REPAIR_MAX", "3"))
INDEX_REPAIR_COOLDOWN_S = float(os.environ.get("MEMORY_INDEX_REPAIR_COOLDOWN_S", "300"))


def _index_repair_allowed() -> bool:
    """Whether search may attempt an automatic index rebuild right now."""
    if INDEX_REPAIR_MAX_ATTEMPTS <= 0:
        return False
    if _index_repair_attempts >= INDEX_REPAIR_MAX_ATTEMPTS:
        return False
    return (time.time() - _index_repair_last) >= INDEX_REPAIR_COOLDOWN_S


def _note_index_repair_attempt() -> None:
    global _index_repair_attempts, _index_repair_last
    _index_repair_attempts += 1
    _index_repair_last = time.time()


def _note_index_repair_success() -> None:
    """A repair that worked must not consume the budget for later repairs.

    The budget exists to stop an UNFIXABLE index from triggering a rebuild on
    every call, so it should count CONSECUTIVE FAILURES. Counting successful
    repairs too was actively harmful once dedup started using this path: a
    merge's DETACH DELETE + CREATE leaves the index unable to answer, so
    repairs are routine rather than pathological. With a 3-attempt budget and a
    300s cooldown, the budget was exhausted almost immediately and dedup then
    silently failed open — only the first near-duplicate after a merge was
    recognised, and every one for the next five minutes was stored as new.
    """
    global _index_repair_attempts, _index_repair_last
    _index_repair_attempts = 0
    _index_repair_last = 0.0


def _file_uri_to_path(uri) -> Optional[str]:
    """Convert a file:// URI to a local filesystem path, or None if not file://."""
    from urllib.parse import unquote, urlparse
    s = str(uri)
    if not s.startswith("file://"):
        return None
    path = unquote(urlparse(s).path)
    # Windows URIs look like file:///C:/dir — strip the leading slash
    if re.match(r"^/[A-Za-z]:[/\\]", path):
        path = path[1:]
    return path or None


async def _adopt_client_workspace() -> None:
    """Adopt the MCP client's first workspace root as the memory scope.

    Runs on tool calls until an attempt completes (roots/list is only
    available after the session is initialized, so this can't happen at
    startup). Concurrent first calls (e.g. a recall hook racing the user's
    call) serialize on a lock, so no call proceeds with an un-adopted
    workspace while another is still asking the client.

    Adoption finishes permanently when:
    - MEMORY_WORKSPACE is explicitly set (env var always wins), or
    - the DB connection is already open (adopting after that would tag
      memories with a workspace whose DB file was never opened), or
    - the client doesn't advertise the roots capability, or
    - the client answered (with or without a usable file:// root), or
    - _ROOTS_MAX_ATTEMPTS transient failures accumulated.

    Cancellation (user aborts the tool call mid-request) propagates WITHOUT
    marking adoption done, so the next call retries. Transient errors
    (timeouts, transport hiccups) retry up to the attempt cap. When no root
    is adopted, the static _resolve_workspace() baseline stays in effect.
    """
    global _roots_done, _roots_attempts, WORKSPACE, _workspace_source
    global _client_info, _client_supports_roots
    if _roots_done:
        return

    async with _roots_lock:
        if _roots_done:
            return

        env_ws = os.environ.get("MEMORY_WORKSPACE", "")
        if env_ws and not _is_bogus_workspace(env_ws):
            _roots_done = True
            return

        if _conn is not None:
            _roots_done = True
            return

        _roots_attempts += 1
        try:
            ctx = mcp.get_context()
            session = ctx.session

            # Record who we're talking to — this shows up in memory_stats
            # so workspace problems can be diagnosed with one call.
            params = getattr(session, "client_params", None)
            ci = getattr(params, "clientInfo", None)
            if ci is not None:
                _client_info = {"name": ci.name, "version": ci.version}

            _client_supports_roots = session.check_client_capability(
                mcp_types.ClientCapabilities(roots=mcp_types.RootsCapability())
            )
            if not _client_supports_roots:
                logger.info(
                    "MCP client does not advertise roots support; workspace "
                    "stays static (use memory_set_workspace to pin one)"
                )
                _roots_done = True
                return
            with anyio.fail_after(5):
                result = await session.list_roots()
            for root in result.roots:
                path = _file_uri_to_path(root.uri)
                if path and not _is_bogus_workspace(path):
                    WORKSPACE = path
                    _workspace_source = "roots"
                    logger.info(f"Workspace adopted from MCP client root: {path}")
                    break
            else:
                logger.debug("MCP client returned no usable file:// roots")
            _roots_done = True  # the client answered definitively
        except Exception as e:
            # Transient failure — retry on a later call, up to the cap.
            # (Cancellation is a BaseException: it propagates and is NOT
            # counted as done, by design.)
            if _roots_attempts >= _ROOTS_MAX_ATTEMPTS:
                _roots_done = True
            logger.debug(
                f"Workspace adoption attempt {_roots_attempts} failed: {e}"
            )


def _dream_state_path() -> Optional[str]:
    """Return the path to the dream-state sidecar file, or None for in-memory DBs."""
    if DB_PATH == ":memory:":
        return None
    return os.path.join(os.path.dirname(DB_PATH) or ".", ".dream-state.json")


def _bump_dream_ops():
    """Atomically increment the dream operations counter.
    Persists to disk every PERSIST_EVERY_N bumps to avoid N+1 writes on batch ops.
    """
    global _dream_ops_since_last
    with _dream_ops_lock:
        _dream_ops_since_last += 1
        # Throttle persistence: only every 10 bumps, not every store
        if _dream_ops_since_last % 10 == 0:
            try:
                _persist_dream_state()
            except Exception:
                pass


def _persist_dream_state():
    """Save dream counters to a sidecar JSON file with atomic replace.

    Sidecar (rather than SchemaMeta) because LadybugDB doesn't expose an atomic
    upsert for property changes, and the prior delete+create dance lost state
    on crash. POSIX `os.replace` is atomic so a crash either leaves the old
    file or the new one — never an empty partial.
    """
    sidecar = _dream_state_path()
    if sidecar is None:
        return
    payload = {
        "dream_ops": _dream_ops_since_last,
        "dream_last_time": _dream_last_time,
        "version": 1,
    }
    tmp = sidecar + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, sidecar)
    except Exception as e:
        logger.debug(f"Could not persist dream state: {e}")
        # Clean up tmp on failure so we don't leak files.
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _load_dream_state():
    """Restore dream counters from the sidecar (or, for backward compat, from
    SchemaMeta on databases that pre-date the sidecar).
    """
    global _dream_ops_since_last, _dream_last_time

    sidecar = _dream_state_path()
    if sidecar and os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                payload = json.load(f)
            _dream_ops_since_last = int(payload.get("dream_ops", 0))
            _dream_last_time = float(payload.get("dream_last_time", 0.0))
            return
        except Exception as e:
            logger.debug(f"Could not load dream state from sidecar: {e}")

    # Backward-compat: read legacy SchemaMeta rows if present, then migrate.
    if _conn is None:
        return
    migrated = False
    try:
        result = _conn.execute(
            "MATCH (s:SchemaMeta) WHERE s.key IN ['dream_ops', 'dream_last_time'] "
            "RETURN s.key, s.value;"
        )
        for row in _collect_results(result):
            if row[0] == "dream_ops":
                _dream_ops_since_last = int(row[1])
                migrated = True
            elif row[0] == "dream_last_time":
                _dream_last_time = float(row[1])
                migrated = True
    except Exception as e:
        logger.debug(f"Could not load dream state: {e}")

    # Persist into the sidecar so next startup is atomic, then clean up SchemaMeta.
    if migrated:
        try:
            _persist_dream_state()
            _conn.execute(
                "MATCH (s:SchemaMeta) "
                "WHERE s.key IN ['dream_ops', 'dream_last_time'] DETACH DELETE s;"
            )
        except Exception as e:
            logger.debug(f"Could not migrate legacy dream state: {e}")


def _serialize(obj) -> str:
    """Serialize a Python object to the configured response format (JSON or TOON).

    TOON is significantly more token-efficient (often 30-60% reduction vs JSON).
    Falls back to compact JSON if TOON is unavailable or encoding fails.
    """
    if RESPONSE_FORMAT == "toon" and _TOON_AVAILABLE:
        try:
            return _toon_encode(obj)
        except Exception as e:
            logger.debug(f"TOON encode failed, falling back to JSON: {e}")
    # Compact JSON (no indent), default=str for things like floats
    return json.dumps(obj, default=str)


def _timed(operation: str):
    """Decorator: wraps a tool, normalizes its return into a dict (or {results: list}),
    injects elapsed_ms, logs slow operations, then serializes once at the end.

    Tool functions SHOULD return a Python dict or list. Returning a string is supported
    for legacy reasons (the string is passed through unchanged with no timing info).

    The wrapper is async so it can talk to the MCP client (roots/list for
    workspace adoption) before dispatching to the sync tool body. FastMCP
    reads the schema from __signature__, so tool parameters are unaffected.
    Tests bypass this wrapper via __wrapped__.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            await _adopt_client_workspace()
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

            # Tools that already serialized themselves (rare, legacy) bypass timing
            # injection — return as-is so we don't double-encode.
            if isinstance(result, str):
                if elapsed_ms > LATENCY_WARN_MS:
                    logger.warning(f"SLOW {operation}: {elapsed_ms}ms")
                else:
                    logger.debug(f"{operation}: {elapsed_ms}ms")
                return result

            if isinstance(result, list):
                result = {"results": result}
            elif not isinstance(result, dict):
                result = {"value": result}

            result["elapsed_ms"] = elapsed_ms

            if elapsed_ms > LATENCY_WARN_MS:
                logger.warning(f"SLOW {operation}: {elapsed_ms}ms")
            else:
                logger.debug(f"{operation}: {elapsed_ms}ms")

            return _serialize(result)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = inspect.signature(func)
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


def get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(EMBEDDING_MODEL)
    return _embed_model


def get_conn() -> lb.Connection:
    global _conn, _db, DB_PATH
    if _conn is not None:
        return _conn

    # Re-resolve now: the client's workspace root is adopted on the first tool
    # call (after import), and the DB path derives from the workspace. With an
    # explicit MEMORY_DB_PATH this is a no-op.
    DB_PATH = _resolve_db_path()

    from pathlib import Path
    if DB_PATH == ":memory:":
        _db = lb.Database(":memory:")
    else:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        try:
            _db = lb.Database(DB_PATH)
        except Exception as e:
            if "lock" in str(e).lower():
                raise RuntimeError(
                    f"Memory database is locked: {DB_PATH}. LadybugDB allows a "
                    f"single read-write process per database file, and another "
                    f"memnest server (likely another IDE window) is using this "
                    f"one. Close the other session, or give this one its own "
                    f"database via MEMORY_DB_PATH / MEMORY_WORKSPACE."
                ) from e
            raise

    _conn = lb.Connection(_db)

    # Load extensions
    _conn.execute("INSTALL vector; LOAD EXTENSION vector;")
    _conn.execute("INSTALL fts; LOAD EXTENSION fts;")
    _conn.execute("INSTALL algo; LOAD EXTENSION algo;")
    _conn.execute("INSTALL json; LOAD EXTENSION json;")
    _conn.execute("INSTALL httpfs; LOAD EXTENSION httpfs;")

    # Log loaded extensions
    try:
        result = _conn.execute("CALL SHOW_LOADED_EXTENSIONS() RETURN *;")
        loaded = []
        while result.has_next():
            loaded.append(result.get_next()[0])
        logger.info(f"Loaded extensions: {', '.join(loaded)}")
    except Exception:
        pass

    _init_schema(_conn)
    # Repair a stale HNSW index before serving any query, otherwise semantic
    # search silently degrades to keyword-only for the whole session.
    global _vector_index_state
    _vector_index_state = _ensure_vector_index(_conn)
    _load_dream_state()
    return _conn


# Schema version — bump when adding migrations to _apply_migrations().
# v1: workspace column on Memory.
# v2: provenance + confidence on RELATED_TO.
# v3: retag workspace '/' as '' (global). '/' was a bug: MCP hosts launch
#     servers with cwd '/', which the old code recorded as a real workspace.
SCHEMA_VERSION = 3


def _init_schema(conn: lb.Connection):
    """Create graph schema if not exists, then apply versioned migrations."""
    # Memory node — the core entity
    _safe_execute(conn, f"""
        CREATE NODE TABLE Memory(
            id INT64 PRIMARY KEY,
            content STRING,
            content_hash STRING,
            category STRING DEFAULT 'general',
            tags STRING DEFAULT '',
            workspace STRING DEFAULT '',
            importance INT64 DEFAULT 3,
            access_count INT64 DEFAULT 0,
            created_at DOUBLE DEFAULT 0.0,
            updated_at DOUBLE DEFAULT 0.0,
            embedding FLOAT[{EMBEDDING_DIM}]
        );
    """, expected_errors=("already exists",))

    # Topic node — auto-created from tags
    _safe_execute(conn, """
        CREATE NODE TABLE Topic(
            name STRING PRIMARY KEY
        );
    """, expected_errors=("already exists",))

    # SchemaMeta — tracks applied migration version
    _safe_execute(conn, """
        CREATE NODE TABLE SchemaMeta(
            key STRING PRIMARY KEY,
            value STRING
        );
    """, expected_errors=("already exists",))

    # Relationships
    _safe_execute(conn, "CREATE REL TABLE ABOUT(FROM Memory TO Topic);",
                  expected_errors=("already exists",))
    _safe_execute(conn,
        "CREATE REL TABLE RELATED_TO(FROM Memory TO Memory, "
        "provenance STRING DEFAULT 'EXTRACTED', confidence DOUBLE DEFAULT 1.0);",
        expected_errors=("already exists",))
    _safe_execute(conn, "CREATE REL TABLE SUPERSEDES(FROM Memory TO Memory);",
                  expected_errors=("already exists",))
    _safe_execute(conn,
        "CREATE REL TABLE EXPLAINS(FROM Memory TO Memory, "
        "rationale_type STRING DEFAULT 'why');",
        expected_errors=("already exists",))

    # Vector index
    _safe_execute(conn, """
        CALL CREATE_VECTOR_INDEX(
            'Memory', 'memory_vec_idx', 'embedding',
            metric := 'cosine'
        );
    """, expected_errors=("already exists",))

    # Full-text search index on Memory.content (BM25 with English stemmer)
    _safe_execute(conn, """
        CALL CREATE_FTS_INDEX('Memory', 'memory_fts_idx', ['content'], stemmer := 'english');
    """, expected_errors=("already exists",))

    # Verify embedding dimension matches existing schema
    _verify_embedding_dim(conn)

    # Apply versioned migrations only if behind
    _apply_migrations(conn)

    logger.info("Schema initialized")


_NUMERIC_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAPPED_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
# Lowercase identifiers: service/host/env names such as prod-checkout,
# payments-core, ledger-db, api.internal, feature/flag. Requires an internal
# separator so prose words don't qualify, and must not start with a digit so
# quantities like "30-minute" stay with the numeric class instead.
_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-_./:][a-z0-9]+)+\b")


_WORD_RE = re.compile(r"[a-z0-9_]+")


def _discriminating_tokens(text: str, name_vocab: Optional[set] = None) -> set:
    """Extract the tokens that carry a fact's VALUE rather than its phrasing.

    Three classes distinguish two statements about the same subject:
      numbers        500ms vs 900ms, Q2 2026 vs Q3 2026, Java 17 vs Java 21
      names          Kafka vs Kinesis, HALF_UP vs HALF_EVEN, PostgreSQL vs DynamoDB
      identifiers    prod-checkout vs prod-inventory, ledger-db vs inventory-cache

    Ordinary wording differences ("caches sessions" vs "caches user sessions")
    leave all three identical. Thousands separators are stripped so
    "12,000"/"12000" match, and everything is lowercased.

    `name_vocab` supplies the set of words to treat as names. It matters because
    capitalisation is evidence, not identity: deriving names from this text alone
    makes the class asymmetric, so "Uses Redis" would yield {uses, redis} while
    "uses redis" yields {} and a pure case difference would read as two missing
    values. Callers comparing two texts pass the union of both vocabularies.
    """
    norm = text.replace(",", "")
    nums = {n.rstrip(".") for n in _NUMERIC_RE.findall(norm)}
    idents = set(_IDENT_RE.findall(text.lower()))
    if name_vocab is None:
        name_vocab = {w.lower() for w in _CAPPED_RE.findall(text)}
    names = {w for w in _WORD_RE.findall(text.lower()) if w in name_vocab}
    return nums | names | idents


def _name_vocab(*texts: str) -> set:
    """Words capitalised in ANY of the given texts, lowercased."""
    vocab = set()
    for t in texts:
        vocab |= {w.lower() for w in _CAPPED_RE.findall(t)}
    return vocab


def _values_conflict(a: str, b: str) -> bool:
    """True when two texts disagree on a value-bearing token.

    Used to refuse a DESTRUCTIVE merge. Two memories about the same subject
    whose numbers or named values differ are not duplicates — they are
    competing facts, and merging them silently discards one. Observed: storing
    "Vega request timeout is 900 milliseconds" over "...is 500 milliseconds"
    merged at 0.929 and kept the STALE 500ms value while reporting success.

    A superset is not a conflict: "5 attempts" vs "5 attempts with jitter and a
    10s cap" adds tokens rather than contradicting them, so only genuine
    disagreement on shared token *kinds* blocks the merge.
    """
    if not MERGE_VALUE_GATE:
        return False
    vocab = _name_vocab(a, b)
    ta = _discriminating_tokens(a, vocab)
    tb = _discriminating_tokens(b, vocab)
    if ta == tb:
        return False
    # One side strictly richer = elaboration rather than disagreement, but that
    # is only SAFE if the richer text is the one the merge keeps. Both merge
    # paths keep the longer string, so a shorter-but-value-richer text would
    # have its extra values thrown away — treat that as a conflict instead.
    if ta < tb:
        return not len(b) > len(a)
    if tb < ta:
        return not len(a) > len(b)
    return True


def _same_subject(tags_a, tags_b) -> bool:
    """Cheap subject check gating DESTRUCTIVE merges.

    Templated facts about different subjects read almost identically — runbooks
    and logging decisions for different services measured 0.94-0.95 similarity,
    above both merge thresholds, and merging them silently discards which
    service each described. Measured on a 127-fact corpus: 30 memories were
    absorbed on ingest before this gate existed.

    Tag overlap is the subject proxy. When either side is untagged there is no
    signal, so prior behaviour is preserved (similarity alone decides) — which
    is why the guidance tells agents to tag.

    Asymmetry is deliberate: a missed merge leaves a recoverable duplicate that
    the review band will surface again, while a wrong merge destroys a fact
    irreversibly. Prefer keeping both.
    """
    sa, sb = set(tags_a or []), set(tags_b or [])
    if not sa or not sb:
        return True
    union = sa | sb
    if not union:
        return True
    return (len(sa & sb) / len(union)) >= MERGE_TAG_OVERLAP


def _conflict_dismissed(conn: lb.Connection, a: int, b: int) -> Optional[str]:
    """Whether an agent has already resolved the relationship between a and b.

    Wider than _semantically_linked on purpose. That function governs MERGE
    protection, where an auto-created RELATED_TO must not count — it is inferred
    from shared topics and asserts nothing.

    Conflict flagging is a different question. Its hint tells the agent that if
    both facts hold, it should "link them with memory_relate" — so an
    agent-asserted RELATED_TO has to silence the flag, or the advice is a dead
    end: the agent does exactly what it was told and the same warning returns on
    every subsequent search. Reported by a user who wired the edge and watched
    the flag persist, with SUPERSEDES the only thing that cleared it — which
    would have been factually false and would have demoted a true fact.

    provenance distinguishes the two: INFERRED comes from _compute_graph_scores,
    while EXTRACTED and AMBIGUOUS are written by a caller who looked at the pair.
    """
    linked = _semantically_linked(conn, a, b)
    if linked:
        return linked
    try:
        r = conn.execute(
            """MATCH (x:Memory)-[e:RELATED_TO]-(y:Memory)
               WHERE x.id = $a AND y.id = $b AND e.provenance <> 'INFERRED'
               RETURN COUNT(e);""",
            {"a": a, "b": b},
        )
        if r.has_next() and (r.get_next()[0] or 0) > 0:
            return "RELATED_TO"
    except Exception as e:
        logger.debug(f"Conflict-dismissal check failed for {a}/{b}: {e}")
    return None


# Phrasings that mark a statement as revising something already known. Corrections
# are the case value_disagreement exists to catch, and they announce themselves:
# an agent recording one naturally writes "Correction:", "now", "no longer",
# "changed to", "extended to". Complementary facts about the same subject
# ("depends on Redis for caching" / "depends on Kafka for event delivery") do not.
_CORRECTION_MARKER_RE = re.compile(
    r"\b(correct(?:ion|ed)?|update[ds]?|revis(?:ed|ion)|amend(?:ed|ment)?|"
    r"supersed(?:es|ed)|replac(?:es|ed|ing)|instead\s+of|rather\s+than|"
    r"no\s+longer|previously|formerly|used\s+to|was\s+changed|"
    r"chang(?:ed|es)\s+(?:to|from)|mov(?:ed|es)\s+(?:to|from)|"
    r"(?:extended|reduced|raised|lowered|increased|decreased|bumped)\s+(?:to|from)|"
    r"actually|in\s+fact|as\s+of\s+\d|now\s+(?:uses?|is|are|set|configured|runs?))\b",
    re.I,
)


def _has_correction_marker(*texts: str) -> bool:
    return any(_CORRECTION_MARKER_RE.search(t or "") for t in texts)


# --- Comparable quantities -------------------------------------------------
#
# The correction marker above is anti-correlated with need, which is the whole
# problem with relying on it alone: a marker means the agent KNOWS it is
# correcting something, and an agent that knows would pass supersedes= and never
# need the detector. Detection earns its keep in the opposite case — a fact
# learned in a fresh session and written down plainly, with no marker, because
# from the agent's point of view nothing is being corrected. Measured:
#
#   "The Mira service retains audit logs for 30 days."      #1  0.7695  STALE
#   "Audit logs on the Mira service are kept for one year."  #2  0.7024  CURRENT
#   potential_conflicts: none
#
# So a second, marker-free trigger: the two texts state QUANTITIES that are
# directly comparable — the same dimension, or literally the same unit noun —
# and the magnitudes differ.
#
# Word order does the work of excluding identifiers, and it does it for a real
# reason rather than a lucky one: English writes measurements as
# "<number> <unit>" ("30 days", "500 messages") and identifiers as
# "<label> <number>" ("port 8080", "version 3", "Java 17"). Requiring a unit
# AFTER the number therefore skips port numbers and version numbers without
# needing a list of them — which is what keeps "exposes port 8080 for HTTP" and
# "exposes port 9090 for metrics" out of the detector.
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "half": 0.5,
}

# Units that can be compared ACROSS wording, normalised to a base unit. Only
# unambiguous spellings: bare "m" and "s" are skipped because they collide with
# metres/seconds and with each other in practice.
_DIMENSIONAL_UNITS = {
    "duration": {
        "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001, "msec": 0.001,
        "sec": 1, "secs": 1, "second": 1, "seconds": 1,
        "min": 60, "mins": 60, "minute": 60, "minutes": 60,
        "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
        "day": 86400, "days": 86400,
        "week": 604800, "weeks": 604800,
        "month": 2592000, "months": 2592000,
        "quarter": 7776000, "quarters": 7776000,
        "year": 31536000, "years": 31536000,
    },
    "size": {
        "byte": 1, "bytes": 1,
        "kb": 1e3, "kib": 1024, "kilobyte": 1e3, "kilobytes": 1e3,
        "mb": 1e6, "mib": 1024 ** 2, "megabyte": 1e6, "megabytes": 1e6,
        "gb": 1e9, "gib": 1024 ** 3, "gigabyte": 1e9, "gigabytes": 1e9,
        "tb": 1e12, "tib": 1024 ** 4, "terabyte": 1e12, "terabytes": 1e12,
    },
}
_UNIT_TO_FAMILY = {
    unit: (family, factor)
    for family, units in _DIMENSIONAL_UNITS.items()
    for unit, factor in units.items()
}

# Words that can never BE the unit, only sit between a number and its unit
# ("8080 for HTTP", "a full year"). Without this filter "8080 for" parsed as
# 8080 of unit "for", which made two port numbers look like comparable
# magnitudes — the Vela false positive.
_UNIT_STOPWORDS = frozenset({
    "for", "of", "in", "on", "at", "to", "and", "or", "the", "a", "an", "with",
    "per", "from", "by", "as", "is", "are", "was", "were", "be", "been", "that",
    "this", "than", "then", "over", "under", "up", "down", "out", "into", "via",
    "full", "total", "about", "around", "approximately", "roughly", "only",
    "just", "more", "less", "least", "most", "max", "maximum", "min", "minimum",
    "peak", "average", "avg", "each", "every", "all", "any", "some", "no",
})
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")


def _quantity_family(unit: str) -> str:
    """Family key for a unit word. Unknown units become their own family keyed
    on the singular form, so "500 messages" and "800 messages" compare while
    "500 messages" and "800 requests" do not."""
    u = unit.lower()
    if u in _UNIT_TO_FAMILY:
        return _UNIT_TO_FAMILY[u][0]
    return "unit:" + (u[:-1] if len(u) > 3 and u.endswith("s") else u)


def _extract_quantities(text: str) -> dict:
    """Map family -> set of normalised magnitudes found in the text.

    Walks tokens rather than pattern-matching a fixed gap, because the unit can
    sit one or two words after the number ("a full year", "800 queued messages")
    and the words in between must not be mistaken for it. Within a 3-token
    lookahead a KNOWN dimensional unit wins over a nearer unknown word, so
    "a full year" reads as a duration rather than a quantity of "full".
    """
    raw = text or ""
    spans = list(_TOKEN_RE.finditer(raw))
    tokens = [m.group(0) for m in spans]
    out: dict = {}
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _WORD_NUMBERS:
            value = float(_WORD_NUMBERS[low])
        elif tok[0].isdigit():
            # A digit run glued to a letter prefix is an identifier, not a
            # measurement: p99, v2, s3, h2. Without this check "p99 latency"
            # yielded 99 of unit "latency", so p99 and p50 latency figures —
            # complementary by definition — looked like disagreeing magnitudes.
            start = spans[i].start()
            if start > 0 and raw[start - 1].isalpha():
                continue
            try:
                value = float(tok.replace(",", ""))
            except ValueError:
                continue
        else:
            continue

        window = [t.lower() for t in tokens[i + 1:i + 4] if not t[0].isdigit()]
        # A KNOWN dimensional unit counts from up to three words away, so
        # "a full year" and "30 to 60 seconds" still read as durations.
        unit = next((w for w in window if w in _UNIT_TO_FAMILY), None)
        if unit is None:
            # An UNKNOWN unit must be the very next word. Scanning further let
            # "port 8080 for HTTP" attach 8080 to "http", which then collided
            # with another port sharing the same trailing word — the number is
            # labelled by what precedes it, so nothing after the preposition is
            # its unit.
            unit = window[0] if window and window[0] not in _UNIT_STOPWORDS else None
        if unit is None:
            continue

        family = _quantity_family(unit)
        factor = _UNIT_TO_FAMILY.get(unit, (None, 1.0))[1]
        out.setdefault(family, set()).add(round(value * factor, 6))
    return out


def _quantities_disagree(a: str, b: str) -> bool:
    """True when both texts state a comparable quantity and the values differ.

    Requires the family to appear in BOTH texts: a fact that adds a quantity the
    other never mentions is elaboration, not disagreement.
    """
    qa, qb = _extract_quantities(a), _extract_quantities(b)
    for family in set(qa) & set(qb):
        if qa[family] != qb[family]:
            return True
    return False


def _semantically_linked(conn: lb.Connection, a: int, b: int) -> Optional[str]:
    """Return the edge type if SUPERSEDES or EXPLAINS connects a and b.

    These edges are asserted by an agent and mean "these memories are distinct
    and ordered" — a correction and the thing it corrects, a rationale and the
    decision it explains. Merging such a pair destroys the history the edge
    exists to record, which matters because a correction is textually
    near-identical to what it corrects (measured 0.9284 on consecutive
    retry-policy versions) and therefore looks exactly like a duplicate.

    RELATED_TO is deliberately NOT treated as protective: _compute_graph_scores
    creates it automatically between memories sharing topics, so it carries no
    assertion that the two are distinct.
    """
    try:
        r = conn.execute(
            """MATCH (x:Memory)-[e:SUPERSEDES|EXPLAINS]-(y:Memory)
               WHERE x.id = $a AND y.id = $b RETURN label(e) LIMIT 1;""",
            {"a": a, "b": b},
        )
        rows = _collect_results(r)
        if rows:
            return str(rows[0][0]) if rows[0] and rows[0][0] else "SUPERSEDES/EXPLAINS"
    except Exception as e:
        logger.debug(f"Link check failed for {a}/{b}: {e}")
        # Fall back to a label-free existence check
        try:
            r = conn.execute(
                """MATCH (x:Memory)-[e:SUPERSEDES|EXPLAINS]-(y:Memory)
                   WHERE x.id = $a AND y.id = $b RETURN COUNT(e);""",
                {"a": a, "b": b},
            )
            if r.has_next() and (r.get_next()[0] or 0) > 0:
                return "SUPERSEDES/EXPLAINS"
        except Exception:
            pass
    return None


def _probe_vector_index(conn: lb.Connection, k: int = 1) -> Optional[int]:
    """Return how many rows the HNSW index returns for a synthetic probe.

    None means the query itself failed. Uses a constant vector so this costs
    no embedding-model work — we only care whether the index yields anything,
    not what it matches.
    """
    try:
        result = conn.execute(
            """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $q, $k)
               WITH node AS m, distance RETURN m.id;""",
            {"q": [0.1] * EMBEDDING_DIM, "k": k},
        )
        return len(_collect_results(result))
    except Exception as e:
        logger.debug(f"Vector index probe failed: {e}")
        return None


def _ensure_vector_index(conn: lb.Connection, force_rebuild: bool = False) -> dict:
    """Detect and repair an HNSW index that does not cover existing rows.

    Observed in the field: a database carried across versions and delete
    cycles ended up with memory_vec_idx present but returning zero rows for
    every query, while all 127 memories held valid 384-dim embeddings. The
    effect is severe and silent — hybrid search degrades to keyword-only and
    still returns plausible scores. Dropping and recreating the index
    restores semantic search immediately.

    Runs at connection time: one cheap probe, and a rebuild only when broken.
    """
    total = _count_memories(conn)
    if total == 0:
        return {"status": "empty", "rebuilt": False}

    embedded = 0
    try:
        r = conn.execute("MATCH (m:Memory) WHERE m.embedding IS NOT NULL RETURN COUNT(m);")
        if r.has_next():
            embedded = r.get_next()[0] or 0
    except Exception as e:
        logger.debug(f"Embedded-row count failed: {e}")
        return {"status": "unknown", "rebuilt": False}

    if embedded == 0:
        # Nothing to index; rebuilding cannot help. Callers see this via stats.
        return {"status": "no_embeddings", "rebuilt": False, "memories": total}

    # force_rebuild skips the zero-rows probe: callers use it when they have
    # independent evidence of PARTIAL degradation (the index returns rows but
    # misses specific memories — e.g. dream's self-recall audit), which the
    # probe cannot see.
    if not force_rebuild:
        if (_probe_vector_index(conn) or 0) > 0:
            return {"status": "ok", "rebuilt": False}

        logger.error(
            f"Vector index is stale: {embedded} of {total} memories have embeddings "
            f"but the HNSW index returns nothing. Semantic search would silently "
            f"degrade to keyword-only. Rebuilding memory_vec_idx..."
        )
    try:
        _safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                      expected_errors=("does not exist", "not found",
                                       "doesn't have an index"))
        conn.execute(
            "CALL CREATE_VECTOR_INDEX('Memory', 'memory_vec_idx', 'embedding', "
            "metric := 'cosine');"
        )
    except Exception as e:
        logger.error(f"Vector index rebuild FAILED: {e}")
        return {"status": "rebuild_failed", "rebuilt": False, "error": str(e)[:200]}

    after = _probe_vector_index(conn) or 0
    if after > 0:
        logger.info(f"Vector index rebuilt successfully ({embedded} embeddings indexed)")
        return {"status": "rebuilt", "rebuilt": True, "indexed": embedded}
    logger.error("Vector index still returns nothing after rebuild")
    return {"status": "broken", "rebuilt": True}


def _verify_embedding_dim(conn: lb.Connection):
    """Refuse to start if the existing schema has a different embedding dimension."""
    try:
        result = conn.execute("CALL TABLE_INFO('Memory') RETURN *;")
        for row in _collect_results(result):
            # row: property_id, name, type, default, primary_key
            if len(row) > 2 and row[1] == "embedding":
                type_str = str(row[2])
                # Look for FLOAT[N]
                import re
                match = re.search(r"FLOAT\[(\d+)\]", type_str)
                if match:
                    actual = int(match.group(1))
                    if actual != EMBEDDING_DIM:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: schema has FLOAT[{actual}] but "
                            f"MEMORY_EMBEDDING_DIM={EMBEDDING_DIM}. "
                            f"Either set MEMORY_EMBEDDING_DIM={actual} or migrate the database."
                        )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not verify embedding dimension: {e}")


def _get_schema_version(conn: lb.Connection) -> int:
    """Read the current schema version from the SchemaMeta table."""
    try:
        result = conn.execute(
            "MATCH (s:SchemaMeta {key: 'version'}) RETURN s.value;"
        )
        if result.has_next():
            return int(result.get_next()[0])
    except Exception:
        pass
    return 0


def _set_schema_version(conn: lb.Connection, version: int):
    """Write the schema version to the SchemaMeta table."""
    try:
        conn.execute(
            "MATCH (s:SchemaMeta {key: 'version'}) DETACH DELETE s;"
        )
    except Exception:
        pass
    try:
        conn.execute(
            "CREATE (s:SchemaMeta {key: 'version', value: $v});",
            {"v": str(version)},
        )
    except Exception as e:
        logger.warning(f"Failed to record schema version: {e}")


def _apply_migrations(conn: lb.Connection):
    """Run only migrations newer than the recorded schema version."""
    current = _get_schema_version(conn)
    if current >= SCHEMA_VERSION:
        return

    logger.info(f"Migrating schema from v{current} to v{SCHEMA_VERSION}")

    # v1: add workspace column to Memory
    if current < 1:
        _safe_execute(conn, "ALTER TABLE Memory ADD workspace STRING DEFAULT '';",
                      expected_errors=("already exists", "duplicate", "already has property"))

    # v2: add provenance + confidence to RELATED_TO
    if current < 2:
        _safe_execute(conn, "ALTER TABLE RELATED_TO ADD provenance STRING DEFAULT 'EXTRACTED';",
                      expected_errors=("already exists", "duplicate", "already has property"))
        _safe_execute(conn, "ALTER TABLE RELATED_TO ADD confidence DOUBLE DEFAULT 1.0;",
                      expected_errors=("already exists", "duplicate", "already has property"))

    # v3: memories written with workspace '/' were a bug (cwd of the MCP host,
    # not a real project). Retag them as '' so they stay globally visible.
    if current < 3:
        _safe_execute(conn,
                      "MATCH (m:Memory) WHERE m.workspace = '/' SET m.workspace = '';")

    _set_schema_version(conn, SCHEMA_VERSION)
    logger.info(f"Schema migrated to v{SCHEMA_VERSION}")


def _safe_execute(conn, query, params=None, expected_errors: tuple = ()):
    """Execute a query, swallowing only expected errors. Logs unexpected ones at WARNING.

    Args:
        conn: LadybugDB connection
        query: Cypher query
        params: Query parameters
        expected_errors: Substrings of error messages that should be silently ignored
                         (e.g. "already exists" for idempotent CREATE patterns).
                         Other errors are logged at WARNING.
    """
    try:
        if params:
            conn.execute(query, params)
        else:
            conn.execute(query)
    except Exception as e:
        msg = str(e).lower()
        if any(exp.lower() in msg for exp in expected_errors):
            logger.debug(f"Expected error suppressed: {e}")
        else:
            # Truncate query for log readability
            q_preview = query.strip().replace("\n", " ")[:120]
            logger.warning(f"Query failed: {q_preview}... | {e}")


# --- Helpers ---


def _embed(text: str) -> Optional[list[float]]:
    """Embed a single text. Returns None on failure (callers should handle gracefully)."""
    try:
        return list(get_embed_model().embed([text]))[0].tolist()
    except Exception as e:
        # ERROR, not WARNING: without embeddings the semantic channel is dead
        # and retrieval silently degrades to keyword-only. Hosts commonly set
        # FASTMCP_LOG_LEVEL=ERROR, which would hide a warning entirely.
        logger.error(f"Embedding failed for text (len={len(text)}): {e}")
        return None


def _embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed multiple texts in one model call. Returns None placeholders for failures."""
    if not texts:
        return []
    try:
        return [emb.tolist() for emb in get_embed_model().embed(texts)]
    except Exception as e:
        logger.warning(f"Batch embedding failed (count={len(texts)}): {e}")
        # Fall back to per-text embedding so partial success is possible
        return [_embed(t) for t in texts]


def _truncate(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


# Tag storage uses a non-printable delimiter to round-trip safely.
#
# Why not JSON? LadybugDB parameter-binding interprets `[...]`-shaped strings
# as VECTOR literals; it strips the quotes and stores `[a,b]` instead of
# `["a","b"]`, which then can't be distinguished from a tag literally called
# "[a,b]". The empty list literal `"[]"` triggers an "ANY type vector" runtime
# error during binding.
#
# Why not commas? Tags can legitimately contain commas (e.g. "Smith, John").
# Comma-joined storage is lossy on read.
#
# Solution: a Record-Separator prefix (`\x1e`) marks "v2 tag string", followed
# by Unit-Separator-delimited canonical tags. Legacy comma-only rows still
# parse — see _parse_tags fallback path.
_TAG_FORMAT_PREFIX = "\x1e"
_TAG_DELIM = "\x1f"


def _parse_tags(val) -> list:
    """Safely parse tags from LadybugDB.

    Tag storage formats this code recognizes:
    - "" / None: empty list
    - "\\x1e<tag1>\\x1f<tag2>...": v2 unit-separator format (preferred)
    - "tag1,tag2,...": legacy comma-separated rows (still readable)
    - "[a,b]": legacy malformed JSON-derived rows; tolerated as a comma split
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return []
    s = val
    if not s:
        return []
    if s.startswith(_TAG_FORMAT_PREFIX):
        body = s[len(_TAG_FORMAT_PREFIX):]
        return [t for t in body.split(_TAG_DELIM) if t]
    # Legacy paths
    if s.startswith("[") and s.endswith("]"):
        # LadybugDB-mangled JSON array: "[a,b]". Split on commas as best effort.
        return [t.strip() for t in s[1:-1].split(",") if t.strip()]
    return [t.strip() for t in s.split(",") if t.strip()]


def _canonicalize_tag(tag: str) -> str:
    """Normalize a single tag so 'Kuzu', 'kuzu ', ' KUZU' all become 'kuzu'.

    Rules:
    - Strip leading/trailing whitespace
    - Lowercase
    - Collapse internal whitespace runs to single hyphen
    - Strip surrounding punctuation (including comma) that doesn't add meaning.
      Internal commas are preserved as part of the tag and survive JSON storage.
    """
    if not tag:
        return ""
    t = tag.strip().lower()
    # Collapse internal whitespace to single hyphen for consistency
    t = "-".join(t.split())
    # Strip leading/trailing punctuation. Comma is included so a stray trailing
    # comma doesn't become part of the canonical name. Internal commas survive.
    t = t.strip(".,;:!?\"'`()[]{}")
    return t


def _canonicalize_tags(tags: list[str]) -> list[str]:
    """Normalize a list of tags and remove duplicates while preserving order."""
    seen = set()
    result = []
    for tag in tags or []:
        canon = _canonicalize_tag(tag)
        if canon and canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result


def _format_tags(tags: list[str]) -> str:
    """Format tags for storage using a non-printable delimiter format.
    See _TAG_FORMAT_PREFIX comment for rationale on why neither JSON nor a
    comma-joined string works as a storage format here.
    Canonicalizes input to prevent fragmentation (e.g. 'Kuzu' and 'kuzu' merge).
    """
    canonical = _canonicalize_tags(tags)
    if not canonical:
        return ""
    return _TAG_FORMAT_PREFIX + _TAG_DELIM.join(canonical)


def _normalize(content: str) -> str:
    return " ".join(content.lower().split())


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalize(content).encode()).hexdigest()


def _next_id(conn: lb.Connection) -> int:
    """Return the next monotonically-increasing Memory id.

    Uses a SchemaMeta row to avoid an O(N) MAX scan on every insert (this is
    called inside batch loops, so the prior implementation was O(B*N)). Falls
    back to MAX(m.id) if the meta row is missing (legacy databases).
    """
    try:
        result = conn.execute(
            "MATCH (s:SchemaMeta {key: 'memory_id_seq'}) RETURN s.value;"
        )
        if result.has_next():
            current = int(result.get_next()[0])
            new_id = current + 1
            # Update in place. SET on STRING property is supported.
            conn.execute(
                "MATCH (s:SchemaMeta {key: 'memory_id_seq'}) SET s.value = $v;",
                {"v": str(new_id)},
            )
            return new_id
    except Exception:
        # Fall through to seed/recover path.
        pass

    # Seed the counter from MAX(id) when missing or unreadable.
    try:
        max_result = conn.execute("MATCH (m:Memory) RETURN MAX(m.id);")
        seed = 1
        if max_result.has_next():
            val = max_result.get_next()[0]
            if val is not None:
                seed = val + 1
        # Best-effort upsert: delete any partial row, then create.
        try:
            conn.execute(
                "MATCH (s:SchemaMeta {key: 'memory_id_seq'}) DETACH DELETE s;"
            )
        except Exception:
            pass
        try:
            conn.execute(
                "CREATE (s:SchemaMeta {key: 'memory_id_seq', value: $v});",
                {"v": str(seed)},
            )
        except Exception:
            pass
        return seed
    except Exception:
        return 1


def _count_memories(conn: lb.Connection) -> int:
    result = conn.execute("MATCH (m:Memory) RETURN COUNT(*);")
    if result.has_next():
        return result.get_next()[0]
    return 0


def _ensure_topics(conn: lb.Connection, memory_id: int, tags: list[str]):
    """Create Topic nodes and ABOUT relationships for each tag.
    Canonicalizes tag names (lowercase, trimmed) to prevent topic fragmentation.
    """
    for tag in _canonicalize_tags(tags):
        _safe_execute(conn, "MERGE (t:Topic {name: $name});", {"name": tag})
        _safe_execute(conn, """
            MATCH (m:Memory {id: $mid}), (t:Topic {name: $name})
            MERGE (m)-[:ABOUT]->(t);
        """, {"mid": memory_id, "name": tag})


def _save_memory_relationships(conn: lb.Connection, memory_id: int) -> dict[str, list]:
    """Save all relationships of a Memory node before delete + recreate.

    Returns a dict with keys: rels_out, rels_in, sup_out, sup_in, exp_out, exp_in.
    Each value is a list of LadybugDB rows from the corresponding query.
    """
    return {
        "rels_out": _collect_results(conn.execute(
            """MATCH (m:Memory {id: $id})-[r:RELATED_TO]->(b:Memory)
               RETURN b.id, r.provenance, r.confidence;""",
            {"id": memory_id},
        )),
        "rels_in": _collect_results(conn.execute(
            """MATCH (a:Memory)-[r:RELATED_TO]->(m:Memory {id: $id})
               RETURN a.id, r.provenance, r.confidence;""",
            {"id": memory_id},
        )),
        "sup_out": _collect_results(conn.execute(
            "MATCH (m:Memory {id: $id})-[:SUPERSEDES]->(b:Memory) RETURN b.id;",
            {"id": memory_id},
        )),
        "sup_in": _collect_results(conn.execute(
            "MATCH (a:Memory)-[:SUPERSEDES]->(m:Memory {id: $id}) RETURN a.id;",
            {"id": memory_id},
        )),
        "exp_out": _collect_results(conn.execute(
            """MATCH (m:Memory {id: $id})-[r:EXPLAINS]->(b:Memory)
               RETURN b.id, r.rationale_type;""",
            {"id": memory_id},
        )),
        "exp_in": _collect_results(conn.execute(
            """MATCH (a:Memory)-[r:EXPLAINS]->(m:Memory {id: $id})
               RETURN a.id, r.rationale_type;""",
            {"id": memory_id},
        )),
    }


def _restore_memory_relationships(conn: lb.Connection, memory_id: int, saved: dict):
    """Re-create relationships saved by _save_memory_relationships after recreate.

    Logs at ERROR level if any restoration fails — these are silent data loss events.
    """
    failures = []

    def _restore(query, params, desc):
        try:
            conn.execute(query, params)
        except Exception as e:
            failures.append((desc, str(e)))

    for r_row in saved["rels_out"]:
        _restore(
            """MATCH (a:Memory {id: $from}), (b:Memory {id: $to})
               CREATE (a)-[:RELATED_TO {provenance: $p, confidence: $c}]->(b);""",
            {"from": memory_id, "to": r_row[0],
             "p": r_row[1] or "EXTRACTED", "c": r_row[2] or 1.0},
            f"RELATED_TO {memory_id}->{r_row[0]}",
        )
    for r_row in saved["rels_in"]:
        _restore(
            """MATCH (a:Memory {id: $from}), (b:Memory {id: $to})
               CREATE (a)-[:RELATED_TO {provenance: $p, confidence: $c}]->(b);""",
            {"from": r_row[0], "to": memory_id,
             "p": r_row[1] or "EXTRACTED", "c": r_row[2] or 1.0},
            f"RELATED_TO {r_row[0]}->{memory_id}",
        )
    for r_row in saved["sup_out"]:
        _restore(
            "MATCH (a:Memory {id: $f}), (b:Memory {id: $t}) CREATE (a)-[:SUPERSEDES]->(b);",
            {"f": memory_id, "t": r_row[0]},
            f"SUPERSEDES {memory_id}->{r_row[0]}",
        )
    for r_row in saved["sup_in"]:
        _restore(
            "MATCH (a:Memory {id: $f}), (b:Memory {id: $t}) CREATE (a)-[:SUPERSEDES]->(b);",
            {"f": r_row[0], "t": memory_id},
            f"SUPERSEDES {r_row[0]}->{memory_id}",
        )
    for r_row in saved["exp_out"]:
        _restore(
            """MATCH (a:Memory {id: $f}), (b:Memory {id: $t})
               CREATE (a)-[:EXPLAINS {rationale_type: $rt}]->(b);""",
            {"f": memory_id, "t": r_row[0], "rt": (r_row[1] if len(r_row) > 1 else None) or "why"},
            f"EXPLAINS {memory_id}->{r_row[0]}",
        )
    for r_row in saved["exp_in"]:
        _restore(
            """MATCH (a:Memory {id: $f}), (b:Memory {id: $t})
               CREATE (a)-[:EXPLAINS {rationale_type: $rt}]->(b);""",
            {"f": r_row[0], "t": memory_id, "rt": (r_row[1] if len(r_row) > 1 else None) or "why"},
            f"EXPLAINS {r_row[0]}->{memory_id}",
        )

    if failures:
        logger.error(
            f"Lost {len(failures)} relationship(s) during restore for memory {memory_id}: "
            f"{failures[:5]}"
        )


def _collect_results(result) -> list:
    """Collect all rows from a query result."""
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


# --- MCP Tools ---

def _store_without_embedding(conn, content: str, content_hash: str, category: str,
                              tags: list[str], importance: int, now: float) -> dict:
    """Insert a memory with NULL embedding (used when embedding generation fails).
    The memory is searchable via FTS/Cypher but not via vector index until re-embedded.
    """
    mem_id = _next_id(conn)
    conn.execute(
        """CREATE (m:Memory {
               id: $id, content: $content, content_hash: $hash,
               category: $cat, tags: $tags, workspace: $ws, importance: $imp,
               access_count: 0, created_at: $now, updated_at: $now
           });""",
        {"id": mem_id, "content": content, "hash": content_hash,
         "cat": category, "tags": _format_tags(tags), "ws": WORKSPACE, "imp": importance,
         "now": now},
    )
    _ensure_topics(conn, mem_id, tags)
    return {"status": "stored_new_no_embedding", "id": mem_id,
            "message": "Stored, but embedding generation failed. Vector search will skip this memory."}


DEFAULT_IMPORTANCE = 3


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    """Coerce a caller-supplied int into [lo, hi], falling back to `default`.

    Two-sided on purpose. Several tools previously clamped only the upper
    bound, so a negative slipped through: memory_search(top_k=-3) returned zero
    results plus a `degraded` message blaming the embedding model, and
    preview_chars=-5 turned _truncate into text[:-5], silently corrupting output
    instead of erroring.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def _truncate_content(content: str) -> tuple:
    """Cap stored content length. Returns (content, was_truncated)."""
    if not isinstance(content, str):
        return "", False
    if len(content) <= MAX_STORE_CHARS:
        return content, False
    return content[:MAX_STORE_CHARS], True


def _limit_tags(tags) -> list:
    """Cap tag count and per-tag length before they become Topic primary keys."""
    if not tags:
        return []
    if not isinstance(tags, list):
        return []
    out = []
    for t in tags[:MAX_TAGS_PER_MEMORY]:
        if isinstance(t, str) and t.strip():
            out.append(t[:MAX_TAG_CHARS])
    return out


def _clamp_importance(value: int) -> int:
    """Hold importance inside the documented 1-5 range.

    Out-of-range values are not merely cosmetic: the ranking term is
    (importance - 1) / 4, so an unclamped 9 would contribute double the weight
    the scale allows. memory_update already clamped; store did not.
    """
    try:
        return min(5, max(1, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_IMPORTANCE


def _store_one(conn, content: str, category: str, tags: list[str],
               importance: Optional[int] = None,
               embedding: Optional[list[float]] = None,
               supersedes: Optional[int] = None) -> dict:
    """Core single-memory store logic. Returns a status dict.

    `importance` of None means the caller did not state one. That is kept
    distinct from an explicit 3, because substituting the default early let it
    leak into merges: a memory deliberately stored at importance 2 was raised to
    3 by any restatement that simply omitted the argument. A new memory still
    defaults to DEFAULT_IMPORTANCE; a merge with no stated importance leaves the
    existing value alone.

    If `embedding` is pre-computed (for batch mode), uses it instead of computing again.

    `supersedes` marks this memory as replacing an existing one: it disables
    semantic dedup for this store and creates the SUPERSEDES edge. Both must
    happen together, because a correction is textually near-identical to what it
    corrects — two consecutive retry-policy versions measured 0.9284 similarity,
    above the 0.92 dedup threshold — so dedup would merge them and destroy the
    very history the edge exists to record.
    """
    c_hash = _content_hash(content)
    now = time.time()
    # Value written when this turns out to be a NEW memory. Merges consult
    # `importance` itself so they can tell "unstated" from an explicit 3.
    new_memory_importance = (
        DEFAULT_IMPORTANCE if importance is None else _clamp_importance(importance)
    )

    # Layer 1: Exact hash dedup
    result = conn.execute(
        "MATCH (m:Memory {content_hash: $hash}) RETURN m.id;",
        {"hash": c_hash},
    )
    if result.has_next():
        existing_id = result.get_next()[0]
        out = {"status": "already_exists", "id": existing_id}

        # Restating a fact must not promote it. This path used to raise
        # importance by 1 and reset updated_at, which made restatement a way to
        # climb the rankings: re-learning the same thing across sessions is the
        # NORMAL case, and it says how often something is mentioned, not how
        # much it matters. The bump was monotonic with no decay, so the
        # most-restated mundane facts drifted to the top of every query.
        # Measured: a p99-latency note went 2 -> 4 -> 5 on two restatements and
        # displaced every architecture fact from an unrelated architecture query.
        #
        # But an EXPLICIT value is a caller instruction, not a side effect of
        # restating, so it is still honoured — otherwise importance could be
        # raised by restating a paraphrase yet not the identical text, which is
        # both inconsistent and the case an agent hits when it re-encounters a
        # fact verbatim. Never demotes; use memory_update to lower a value.
        # updated_at is untouched throughout: the content did not change.
        cur = conn.execute(
            "MATCH (m:Memory {id: $id}) RETURN m.importance, m.tags;",
            {"id": existing_id},
        )
        cur_row = cur.get_next() if cur.has_next() else (DEFAULT_IMPORTANCE, "")
        cur_imp = cur_row[0] if cur_row[0] is not None else DEFAULT_IMPORTANCE
        cur_tags = _parse_tags(cur_row[1])

        if importance is not None:
            new_imp = min(5, max(cur_imp, _clamp_importance(importance)))
            if new_imp != cur_imp:
                conn.execute(
                    "MATCH (m:Memory {id: $id}) SET m.importance = $imp;",
                    {"id": existing_id, "imp": new_imp},
                )
                out["importance"] = new_imp

        # Tags are additive for the same reason: the merge path unions them, so
        # dropping them here meant a restatement with a new tag silently lost it.
        added = [t for t in _canonicalize_tags(tags or []) if t not in cur_tags]
        if added:
            merged = list(set(cur_tags + added))
            conn.execute(
                "MATCH (m:Memory {id: $id}) SET m.tags = $tags;",
                {"id": existing_id, "tags": _format_tags(merged)},
            )
            _ensure_topics(conn, existing_id, merged)
            out["tags_added"] = added

        return out

    # Layer 2: Semantic dedup
    if embedding is None:
        embedding = _embed(content)
    if embedding is None:
        # Embedding failed — skip semantic dedup and store with NULL embedding
        # (vector search will skip these; FTS / Cypher still work)
        return _store_without_embedding(conn, content, c_hash, category, tags,
                                        new_memory_importance, now)

    # Set when a near-duplicate was refused a merge because its values disagree.
    # Reported on the stored_new result so the caller learns about the competing
    # fact at write time, without waiting for a search to flag it.
    conflict_with = None
    conflict_sim = None

    if _count_memories(conn) > 0:
        try:
            def _dedup_candidates():
                """Nearest neighbours for the dedup decision.

                Returns None if the probe itself failed (a missing index raises
                rather than returning nothing) so the caller can treat that the
                same as an empty result: both mean the index cannot answer.
                """
                try:
                    # Filter by workspace inside the WITH clause to avoid wasted candidates
                    result = conn.execute(
                        """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                           WITH node AS m, distance
                           WHERE m.workspace IN ['', $ws]
                           RETURN m.id, m.content, m.category, m.tags, m.importance, m.workspace, distance
                           ORDER BY distance LIMIT 5;""",
                        {"query": embedding, "k": 20, "ws": WORKSPACE},
                    )
                    return _collect_results(result)
                except Exception as e:
                    logger.debug(f"Dedup vector probe failed: {e}")
                    return None

            candidates = _dedup_candidates()

            # Repair on use, exactly as the search path does. A healthy HNSW
            # index always returns the k nearest neighbours, because cosine
            # distance is defined for every vector pair — so zero rows while
            # embeddings exist means the index is broken, not that nothing was
            # similar.
            #
            # This matters more here than in search, and was missing: the merge
            # branch below does DETACH DELETE + CREATE to re-embed, which can
            # leave the index returning nothing. Dedup would then see no
            # candidates and silently store every subsequent near-duplicate as
            # new — failing open, with no error and no degraded flag, until some
            # later memory_search happened to trigger a rebuild. Measured: after
            # one merge, a 0.9488 paraphrase pair stopped merging entirely.
            if not candidates and _index_repair_allowed():
                _note_index_repair_attempt()
                state = _ensure_vector_index(conn)
                if state.get("rebuilt"):
                    global _vector_index_state
                    _vector_index_state = state
                    candidates = _dedup_candidates()
                    if candidates:
                        _note_index_repair_success()
                        logger.info("Dedup recovered after automatic index rebuild")

            for row in (candidates or []):
                similarity = round(1.0 - row[6], 4)

                # Warn about a competing fact even when it is too far apart to
                # be a merge candidate. Write-time warning used to be a side
                # effect of the dedup branch below, so it only fired at >=0.92
                # while read-time potential_conflicts reaches down to 0.85 —
                # meaning a contradiction in the 0.85-0.92 band returned a clean
                # stored_new and the agent only discovered it if it later
                # happened to search that topic. Write time is when the fix is
                # cheap (one supersedes= away) and the context is still loaded.
                if (
                    supersedes is None
                    and conflict_with is None
                    and CONFLICT_VALUE_FLOOR <= similarity < DEDUP_THRESHOLD
                    and _same_subject(tags, _parse_tags(row[3]))
                    # Mirror the read path's two triggers exactly, or write time
                    # becomes stricter than read time and the asymmetry this was
                    # built to remove comes back in the other direction:
                    #   >= CONFLICT_THRESHOLD  -> near-duplicate, no marker needed
                    #   below it               -> needs a correction marker,
                    #      because differing values alone describe complementary
                    #      facts as often as contradictions ("depends on Redis
                    #      for caching" / "depends on Kafka for event delivery").
                    and (
                        (similarity >= CONFLICT_THRESHOLD
                         and _values_conflict(content, row[1]))
                        or (_has_correction_marker(content, row[1])
                            and _values_conflict(content, row[1]))
                        or _quantities_disagree(content, row[1])
                    )
                ):
                    conflict_with = row[0]
                    conflict_sim = similarity

                # An explicit supersedes target means the caller is recording a
                # new version, not a duplicate. Never merge those.
                if similarity >= DEDUP_THRESHOLD and supersedes is None:
                    match_id = row[0]
                    match_content = row[1]
                    match_category = row[2]
                    match_tags = _parse_tags(row[3])
                    match_importance = row[4]

                    # Distinct subjects that merely read alike must not merge.
                    if not _same_subject(tags, match_tags):
                        logger.debug(
                            f"Not merging into {match_id} (sim {similarity}): "
                            f"different subject ({tags} vs {match_tags})"
                        )
                        continue

                    # Same subject, disagreeing values = competing facts, not
                    # duplicates. Merging would silently discard one side (and
                    # on equal-length texts the `keep` tie below favours the
                    # OLDER value, so the stale fact would win). Keep both and
                    # let potential_conflicts surface it at read time.
                    if _values_conflict(content, match_content):
                        if conflict_with is None:
                            conflict_with = match_id
                            conflict_sim = similarity
                        _v = _name_vocab(content, match_content)
                        logger.info(
                            f"Not merging into {match_id} (sim {similarity}): "
                            f"conflicting values "
                            f"{sorted(_discriminating_tokens(content, _v) ^ _discriminating_tokens(match_content, _v))}"
                        )
                        continue

                    keep = content if len(content) > len(match_content) else match_content
                    merged_tags = list(set(match_tags + tags))
                    # Take the higher of the two, but do NOT add to it. The old
                    # `+ 1` treated every merge as evidence of importance, so
                    # re-storing a paraphrase twice ratcheted a memory to the
                    # ceiling and distorted ranking for unrelated queries.
                    # Importance is caller-owned metadata; the server should not
                    # editorialise it. This now matches dream's merge, which has
                    # always taken a plain max.
                    #
                    # An unstated importance leaves the existing value untouched.
                    # Applying the default here would mean a restatement that
                    # omits the argument silently raises a deliberate 2 to 3.
                    if importance is None:
                        new_imp = match_importance
                    else:
                        new_imp = min(5, max(match_importance, _clamp_importance(importance)))
                    content_changed = (keep != match_content)

                    if content_changed:
                        # Content is different — embedding must change. Fetch full state,
                        # delete + recreate, then re-link relationships.
                        full_result = conn.execute(
                            """MATCH (m:Memory {id: $id})
                               RETURN m.access_count, m.created_at, m.workspace;""",
                            {"id": match_id},
                        )
                        full_row = full_result.get_next() if full_result.has_next() else (0, now, WORKSPACE)
                        ac, ca, ws = full_row[0] or 0, full_row[1] or now, full_row[2] or WORKSPACE

                        # We're in this branch only because keep != match_content,
                        # so a fresh embedding is required.
                        new_emb = _embed(keep)
                        if new_emb is None:
                            logger.warning(f"Re-embed failed during dedup-merge for memory {match_id}; aborting merge")
                            return {"status": "merge_aborted", "id": match_id,
                                    "message": "Embedding failed; merge aborted to preserve data."}

                        saved_rels = _save_memory_relationships(conn, match_id)
                        conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": match_id})
                        conn.execute(
                            """CREATE (m:Memory {
                                   id: $id, content: $content, content_hash: $hash,
                                   category: $cat, tags: $tags, workspace: $ws, importance: $imp,
                                   access_count: $ac, created_at: $ca, updated_at: $now,
                                   embedding: $emb
                               });""",
                            {"id": match_id, "content": keep, "hash": _content_hash(keep),
                             "cat": match_category, "tags": _format_tags(merged_tags), "ws": ws,
                             "imp": new_imp, "ac": ac, "ca": ca, "now": now, "emb": new_emb},
                        )
                        _ensure_topics(conn, match_id, merged_tags)
                        _restore_memory_relationships(conn, match_id, saved_rels)
                    else:
                        # Content unchanged — absorb tags/importance in place and
                        # keep the existing embedding (no index rebuild needed).
                        #
                        # updated_at is deliberately NOT touched. The ranking
                        # recency term reads updated_at, so refreshing it here
                        # let a restatement reset a memory's decay to ~1.0 and
                        # jump it up the results for queries it does not answer.
                        # updated_at means "when the content last changed".
                        conn.execute(
                            """MATCH (m:Memory {id: $id})
                               SET m.tags = $tags, m.importance = $imp;""",
                            {"id": match_id, "tags": _format_tags(merged_tags),
                             "imp": new_imp},
                        )
                        _ensure_topics(conn, match_id, merged_tags)

                    return {
                        "status": "updated_existing",
                        "id": match_id,
                        "similarity": similarity,
                    }
        except Exception as e:
            logger.warning(f"Vector dedup search failed: {e}")

    # No match — insert new
    mem_id = _next_id(conn)
    conn.execute(
        """CREATE (m:Memory {
               id: $id, content: $content, content_hash: $hash,
               category: $cat, tags: $tags, workspace: $ws, importance: $imp,
               access_count: 0, created_at: $now, updated_at: $now,
               embedding: $emb
           });""",
        {"id": mem_id, "content": content, "hash": c_hash,
         "cat": category, "tags": _format_tags(tags), "ws": WORKSPACE,
         "imp": new_memory_importance,
         "now": now, "emb": embedding},
    )
    _ensure_topics(conn, mem_id, tags)

    out = {"status": "stored_new", "id": mem_id}

    # A near-duplicate was kept instead of merged because the values disagree.
    # Tell the caller now: if this is a correction, one memory_relate call
    # resolves it; if both are true, they need distinguishing context.
    if conflict_with is not None:
        out["potential_conflict_with"] = conflict_with
        out["conflict_similarity"] = conflict_sim
        if conflict_sim is not None and conflict_sim >= DEDUP_THRESHOLD:
            out["hint"] = (
                f"Kept as a separate memory: near-identical to id {conflict_with} "
                f"(similarity {conflict_sim}) but the values differ, so merging would "
                f"have discarded one. If this supersedes it, call "
                f"memory_relate(from_id={mem_id}, to_id={conflict_with}, "
                f"relationship='SUPERSEDES')."
            )
        else:
            # Below the dedup threshold nothing was at risk of being merged —
            # the point is that the two facts disagree and only one can be
            # current, which nothing else would have told the caller.
            out["hint"] = (
                f"Stored, but id {conflict_with} is about the same subject with a "
                f"different value (similarity {conflict_sim}). These are worded too "
                f"differently to look like duplicates, so resolve it now while you "
                f"have the context: if this replaces it, call "
                f"memory_relate(from_id={mem_id}, to_id={conflict_with}, "
                f"relationship='SUPERSEDES'); if both hold in different scopes, make "
                f"that explicit in the content."
            )

    # Wire the correction chain in the same call, so a new version can never be
    # stored without the edge that marks what it replaces.
    if supersedes is not None:
        rel = _relate_one(conn, mem_id, supersedes, relationship="SUPERSEDES")
        out["supersedes"] = supersedes
        if rel.get("status") != "created":
            out["supersedes_error"] = rel.get("error") or rel.get("status")

    return out


@mcp.tool()
@_timed("memory_store")
def memory_store(
    content: Optional[str] = None,
    category: Literal["learning", "preference", "decision", "pattern", "general"] = "general",
    tags: Optional[list[str]] = None,
    importance: Optional[int] = None,
    items: Optional[list[dict]] = None,
    supersedes: Optional[int] = None,
) -> str:
    """Store one or more memories with auto-dedup and topic linking.
    Single: pass content/category/tags/importance.
    Batch: pass items=[{content, category?, tags?, importance?, supersedes?}, ...] (faster — single embed call).

    RECORDING A CORRECTION: pass supersedes=<old_id> (works per-item in batch
    mode too). That creates the SUPERSEDES edge AND disables semantic dedup for
    that store — essential, because a correction is textually near-identical to
    what it corrects, so dedup would otherwise merge the two and erase the
    history. Search then ranks the new version above the old one automatically.

    Importance 1-5, default 3 (neutral) for a NEW memory. Omitting it when the
    store turns out to be a duplicate leaves the existing memory's importance
    untouched, so restating a fact can never change how it ranks. Pass it
    explicitly to set a value.
    """
    conn = get_conn()
    tags = _limit_tags(tags or [])

    # Batch mode
    if items is not None:
        if not isinstance(items, list) or not items:
            return {"status": "error", "message": "items must be a non-empty list."}
        if len(items) > MAX_BATCH_ITEMS:
            return {"status": "error",
                    "message": f"Too many items ({len(items)}, limit {MAX_BATCH_ITEMS}). "
                               f"Split the batch."}

        # Embed all in one call
        contents = [_truncate_content(item.get("content", ""))[0] for item in items]
        embeddings = _embed_batch(contents)

        results = []
        for item, emb, text in zip(items, embeddings, contents):
            try:
                if not text.strip():
                    results.append({"status": "error", "message": "content is required."})
                    continue
                res = _store_one(
                    conn,
                    content=text,
                    category=item.get("category", "general"),
                    tags=_limit_tags(item.get("tags", [])),
                    importance=item.get("importance"),
                    embedding=emb,
                    supersedes=item.get("supersedes"),
                )
                results.append(res)
            except Exception as e:
                results.append({"status": "error", "message": str(e)})

        _bump_dream_ops()
        errors = sum(1 for r in results if r.get("status") == "error")
        out = {"results": results, "count": len(results)}
        # A batch where every item failed used to look like a success envelope,
        # because only per-item statuses were reported.
        if errors:
            out["errors"] = errors
            out["status"] = "error" if errors == len(results) else "partial"
        return out

    # Single mode
    if content is None or not str(content).strip():
        return {"status": "error", "message": "content is required (or pass items=[...])."}

    content, truncated = _truncate_content(content)
    res = _store_one(conn, content, category, tags, importance,
                     supersedes=supersedes)
    if truncated:
        res["content_truncated_to"] = MAX_STORE_CHARS
    _bump_dream_ops()
    return res


@mcp.tool()
@_timed("memory_search")
def memory_search(
    query: str,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    top_k: int = 5,
    global_search: bool = False,
    preview_chars: int = 200,
    include_superseded: bool = True,
    explain: bool = False,
    offset: int = 0,
) -> str:
    """Hybrid semantic + keyword + graph search. Filters by current workspace unless global_search=True.
    top_k max 10 per page. preview_chars caps content length per result (default 200).

    offset skips that many ranked results, so offset=10 with top_k=10 returns
    ranks 11-20. Without it the 10th result was the last reachable one for any
    query. The response carries `offset` and `has_more`.
    TIP: Pass tags=[...] to disambiguate overloaded query words (e.g. "workspace"
    could mean Brazil, Kiro, or ATX — tags narrow it instantly). See memory_stats
    for the list of known topics.

    explain=True attaches a per-result "explain" block (raw channel values,
    weighted contributions, and whether the memory came back from the vector
    index for THIS query) plus a top-level "explain_meta". Use it to diagnose
    ranking: e.g. a memory scoring ~0.3 below expectation with
    in_vector_window=false has lost its semantic contribution for that query.

    Memories that a newer memory SUPERSEDES are demoted and marked
    "superseded": true, so the current answer ranks above the version it
    replaced. Pass include_superseded=False to drop them entirely.

    When the top hits have RELATED_TO / SUPERSEDES / EXPLAINS edges, connected
    memories are returned in a separate "related" list (with "linked_to"
    naming the result they hang off). These are memories similarity alone
    would not surface — the incident caused by a decision, the rationale
    behind a convention — and they are kept out of "results" so graph
    proximity never displaces a direct answer.
    """
    conn = get_conn()
    if not isinstance(query, str) or not query.strip():
        return {"status": "error", "message": "query is required."}
    top_k = _clamp_int(top_k, 1, MAX_SEARCH_RESULTS, 5)
    offset = _clamp_int(offset, 0, 10_000, 0)
    preview_chars = _clamp_int(preview_chars, 1, MAX_PREVIEW_CHARS, 200)
    tags = _limit_tags(tags or []) or None

    # Score accumulators: {memory_id: {vector: float, fts: float, graph: float}}
    raw_scores: dict[int, dict[str, float]] = {}
    memory_data: dict[int, dict] = {}

    # Ids that actually got a hit from the vector index. Needed because
    # _record() pre-seeds every channel to 0.0, so key presence cannot
    # distinguish "no semantic hit" from "semantic score of zero".
    vector_hits: set[int] = set()
    def _record(mid: int, channel: str, score: float):
        if mid not in raw_scores:
            raw_scores[mid] = {"vector": 0.0, "fts": 0.0, "graph": 0.0}
        raw_scores[mid][channel] = max(raw_scores[mid][channel], score)

    # Candidate pool: how many rows each channel retrieves before fusion.
    #
    # This used to be top_k * 3, which made the pool a function of the page
    # size and produced two defects. First, it TRUNCATED the ranking: with
    # top_k=5 the true top-scoring memories could be absent entirely, because
    # they never entered a 15-row window — measured on a 25-memory corpus where
    # top_k=10 revealed two memories (0.7410, 0.7400) that outranked every
    # result top_k=5 returned. Second, it made paging incoherent: a wider pool
    # for page 2 re-ranked the candidates, so pages overlapped.
    #
    # A fixed pool fixes both, and it does NOT change any score: per-memory
    # values are pool-independent (vector similarity is per-pair, FTS rows come
    # back score-ordered so the normalising max is the first row regardless of
    # limit, and recency/importance/graph are per-memory). A larger pool only
    # scores MORE candidates, moving the result closer to the true global
    # ranking. Paging deeper than the pool widens it, at which point pages are
    # no longer guaranteed disjoint — use memory_list for exhaustive scans.
    _pool = max(SEARCH_CANDIDATE_POOL, (top_k + offset) * 3)
    _candidate_k = _pool
    _fts_limit = _pool

    # --- Channel 1: Vector search (HNSW cosine similarity) ---
    embedding = _embed(query)

    def _run_vector_channel() -> int:
        """Query the HNSW index into raw_scores. Returns the number of hits."""
        if embedding is None:
            return 0
        try:
            # The candidate window must cover the requested PAGE, not just the
            # page size, or paging past the first page would ask the index for
            # fewer rows than the offset already consumed.
            k = _candidate_k
            if global_search:
                where_clause = ""
                vec_params: dict = {"query": embedding, "k": k}
            else:
                where_clause = "WHERE m.workspace IN ['', $ws]"
                vec_params = {"query": embedding, "k": k, "ws": WORKSPACE}
            result = conn.execute(
                f"""CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                   WITH node AS m, distance
                   {where_clause}
                   RETURN m.id, m.content, m.category, m.tags, m.importance,
                          m.access_count, m.workspace, m.updated_at, distance
                   ORDER BY distance;""",
                vec_params,
            )
            for row in _collect_results(result):
                mid = row[0]
                # Cosine distance is 0..2, so raw 1-distance can go negative.
                # Clamp so a dissimilar hit can never subtract from relevance.
                similarity = max(0.0, min(1.0, 1.0 - row[8]))
                _record(mid, "vector", similarity)
                vector_hits.add(mid)
                memory_data[mid] = {
                    "id": mid, "content": row[1], "category": row[2],
                    "tags": _parse_tags(row[3]),
                    "importance": row[4], "access_count": row[5],
                    "workspace": row[6] or "", "updated_at": row[7] or 0.0,
                }
        except Exception as e:
            # ERROR, not WARNING: this silently degrades search to keyword-only.
            logger.error(f"Vector search failed (semantic channel disabled): {e}")
        return len(vector_hits)

    if embedding is not None and _count_memories(conn) > 0:
        _run_vector_channel()

        # Repair on use. A healthy HNSW index always returns the k nearest
        # neighbours for any query — cosine distance is defined for every
        # vector pair, so even nonsense queries match something. Zero rows
        # while embeddings exist therefore means the index is broken, not that
        # nothing was similar. That makes this signal safe to act on.
        #
        # The connection-time check alone is not enough: a server can run for
        # days, and an index that goes stale mid-session would otherwise serve
        # keyword-only results until restart.
        if not vector_hits and _index_repair_allowed():
            _note_index_repair_attempt()
            state = _ensure_vector_index(conn)
            if state.get("rebuilt"):
                global _vector_index_state
                _vector_index_state = state
                if _run_vector_channel():
                    _note_index_repair_success()
                    logger.info("Search recovered after automatic index rebuild")

    # --- Channel 2: Full-text search (BM25 via FTS index) ---
    if _count_memories(conn) > 0:
        try:
            # Build FTS query: use the raw query words
            fts_query = query.strip()
            if fts_query:
                if global_search:
                    fts_where = ""
                    fts_params: dict = {"query": fts_query, "limit": _fts_limit}
                else:
                    fts_where = "AND m.workspace IN ['', $ws]"
                    fts_params = {"query": fts_query, "limit": _fts_limit, "ws": WORKSPACE}
                result = conn.execute(
                    f"""CALL QUERY_FTS_INDEX('Memory', 'memory_fts_idx', $query, top := $limit)
                       WITH node AS m, score
                       WHERE score > 0 {fts_where}
                       RETURN m.id, m.content, m.category, m.tags, m.importance,
                              m.access_count, m.workspace, m.updated_at, score
                       ORDER BY score DESC;""",
                    fts_params,
                )
                fts_rows = _collect_results(result)
                if fts_rows:
                    # Normalize FTS scores to 0-1 range
                    max_fts = max(row[8] for row in fts_rows) or 1.0
                    for row in fts_rows:
                        mid = row[0]
                        fts_score = row[8] / max_fts  # normalize to 0-1
                        _record(mid, "fts", fts_score)
                        if mid not in memory_data:
                            memory_data[mid] = {
                                "id": mid, "content": row[1], "category": row[2],
                                "tags": _parse_tags(row[3]),
                                "importance": row[4], "access_count": row[5],
                                "workspace": row[6] or "", "updated_at": row[7] or 0.0,
                            }
        except Exception as e:
            logger.debug(f"FTS search failed (non-fatal): {e}")

    # --- Channel 3: Graph traversal (topic connectivity) ---
    # Boost memories that share topics with the query's tag filter,
    # or that are connected to the same topics as top vector hits.
    if tags:
        for tag in _canonicalize_tags(tags):
            try:
                result = conn.execute(
                    """MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $tag})
                       RETURN m.id;""",
                    {"tag": tag},
                )
                for row in _collect_results(result):
                    _record(row[0], "graph", 0.8)  # Strong signal: explicit tag match
            except Exception:
                pass

    # Also boost by graph PageRank + community membership (pre-computed by memory_dream)
    if raw_scores:
        try:
            top_ids = sorted(raw_scores.keys(),
                             key=lambda x: raw_scores[x]["vector"], reverse=True)[:15]
            if top_ids:
                result = conn.execute(
                    """MATCH (m:Memory)
                       WHERE m.id IN $ids
                       RETURN m.id, m.pagerank, m.community_id, m.k_degree;""",
                    {"ids": top_ids},
                )
                top_communities = set()
                for row in _collect_results(result):
                    mid = row[0]
                    pagerank = row[1] or 0.0
                    community = row[2] if len(row) > 2 else -1
                    k_deg = row[3] if len(row) > 3 else 0
                    if pagerank > 0 or (k_deg and k_deg > 0):
                        # Combined graph score: PageRank + K-Core bonus
                        pr_score = min(pagerank * 5.0, 1.0)
                        kcore_bonus = min((k_deg or 0) / 5.0, 0.3)  # cap at 0.3
                        combined = min(pr_score + kcore_bonus, 1.0)
                        _record(mid, "graph", max(raw_scores.get(mid, {}).get("graph", 0), combined))
                    # Track communities of top results for community expansion
                    if community is not None and community >= 0:
                        top_communities.add(community)

                # Community expansion: find other memories in the same communities
                # as our top results — they're likely relevant too.
                # (Previously guarded by `len(results)`, a variable not bound
                # until ~100 lines later, so this raised UnboundLocalError into
                # the enclosing handler and never actually ran.)
                if top_communities:
                    community_list = list(top_communities)[:3]  # cap at 3 communities
                    try:
                        result = conn.execute(
                            """MATCH (m:Memory)
                               WHERE m.community_id IN $cids AND NOT m.id IN $exclude
                               RETURN m.id, m.content, m.category, m.tags, m.importance,
                                      m.access_count, m.workspace, m.updated_at, m.pagerank;""",
                            {"cids": community_list, "exclude": top_ids},
                        )
                        for row in _collect_results(result):
                            mid = row[0]
                            if mid not in raw_scores:
                                # Add community members with a moderate graph score
                                _record(mid, "graph", 0.4)
                                memory_data[mid] = {
                                    "id": mid, "content": row[1], "category": row[2],
                                    "tags": _parse_tags(row[3]),
                                    "importance": row[4], "access_count": row[5],
                                    "workspace": row[6] or "", "updated_at": row[7] or 0.0,
                                }
                    except Exception:
                        pass
        except Exception:
            # pagerank/community_id columns might not exist yet
            pass

    # --- Score fusion ---
    # Weighted combination: vector 0.4 + fts 0.3 + graph 0.15 + recency 0.1 + importance 0.05
    #
    # FUSION_MODE controls how the vector channel is scaled before weighting:
    #   'legacy'     — raw cosine similarity (this is what the published LOCOMO
    #                  score was measured with, so it stays the default)
    #   'normalized' — min-max across candidates, matching how the FTS channel
    #                  is already scaled. BGE similarities sit in a narrow high
    #                  band (~0.4-0.85 even for unrelated text), so raw values
    #                  give the semantic channel a large constant offset and
    #                  weak discrimination, while normalized FTS always spans
    #                  its full weight. Normalizing makes the effective
    #                  influence match the documented weights.
    # Re-run the LOCOMO benchmark before changing the default.
    if FUSION_MODE == "normalized" and len(vector_hits) > 1:
        vec_vals = [raw_scores[m]["vector"] for m in vector_hits]
        v_lo, v_hi = min(vec_vals), max(vec_vals)
        v_span = v_hi - v_lo
        if v_span > 1e-9:
            for m in vector_hits:
                raw_scores[m]["vector"] = (raw_scores[m]["vector"] - v_lo) / v_span

    now = time.time()
    final_scores: dict[int, float] = {}
    # Per-memory fusion inputs, kept when explain=True so a caller can see WHY
    # a memory scored what it did — added because a real ranking anomaly (a
    # memory losing exactly its vector contribution on one query, on one
    # database) was undiagnosable from outside: every enumerated input favoured
    # the losing memory, and the per-channel values were the only place the
    # difference could live.
    explain_data: dict[int, dict] = {}
    for mid, channels in raw_scores.items():
        mem = memory_data.get(mid, {})
        vec_score = channels.get("vector", 0.0)
        fts_score = channels.get("fts", 0.0)
        graph_score = channels.get("graph", 0.0)

        # Recency: exponential decay, half-life of 30 days
        updated_at = mem.get("updated_at", 0.0)
        age_days = (now - updated_at) / 86400 if updated_at > 0 else 30.0
        recency_score = 0.5 ** (age_days / 30.0)  # half-life 30 days

        # Importance: normalize 1-5 to 0-1
        importance = mem.get("importance", 3)
        importance_score = (importance - 1) / 4.0

        final = (
            vec_score * 0.4
            + fts_score * 0.3
            + graph_score * 0.15
            + recency_score * 0.1
            + importance_score * 0.05
        )
        final_scores[mid] = final

        if explain:
            explain_data[mid] = {
                "vector": round(vec_score, 4),
                "fts": round(fts_score, 4),
                "graph": round(graph_score, 4),
                "recency": round(recency_score, 4),
                "importance_norm": round(importance_score, 4),
                "in_vector_window": mid in vector_hits,
                "weighted": {
                    "vector": round(vec_score * 0.4, 4),
                    "fts": round(fts_score * 0.3, 4),
                    "graph": round(graph_score * 0.15, 4),
                    "recency": round(recency_score * 0.1, 4),
                    "importance": round(importance_score * 0.05, 4),
                },
            }

    # --- Supersession awareness ---
    # A memory that something SUPERSEDES is stale by definition. Pure
    # similarity cannot know that, and the query's wording usually matches the
    # stale fact better than its own correction ("rounds using HALF_UP" beats
    # "Correction: now uses HALF_EVEN" for the query "how does it round?"), so
    # the outdated answer wins on relevance. Demote superseded memories so the
    # current answer surfaces without the caller writing a graph query.
    superseded: set[int] = set()
    if final_scores:
        try:
            r = conn.execute(
                """MATCH (x:Memory)-[:SUPERSEDES]->(m:Memory)
                   WHERE m.id IN $ids RETURN DISTINCT m.id;""",
                {"ids": list(final_scores)},
            )
            superseded = {row[0] for row in _collect_results(r)}
        except Exception as e:
            logger.debug(f"Supersession lookup failed: {e}")

    if superseded:
        for mid in superseded:
            if mid in final_scores:
                final_scores[mid] *= SUPERSEDED_PENALTY

    # Build results
    results = []
    skipped = 0
    more_available = False
    for mid, score in sorted(final_scores.items(), key=lambda x: -x[1]):
        mem = memory_data.get(mid)
        if not mem:
            try:
                r = conn.execute(
                    """MATCH (m:Memory {id: $id})
                       RETURN m.content, m.category, m.tags, m.importance, m.access_count, m.workspace;""",
                    {"id": mid},
                )
                if r.has_next():
                    row = r.get_next()
                    mem = {"id": mid, "content": row[0], "category": row[1],
                           "tags": _parse_tags(row[2]),
                           "importance": row[3], "access_count": row[4],
                           "workspace": row[5] or ""}
            except Exception:
                continue

        if not mem:
            continue
        if category and mem["category"] != category:
            continue
        if not global_search and mem.get("workspace", "") not in ("", WORKSPACE):
            continue

        if mid in superseded and not include_superseded:
            continue

        entry = {
            "id": mid,
            "content": _truncate(mem["content"], preview_chars),
            "tags": mem["tags"],
            "score": round(score, 4),
        }
        if mid in superseded:
            # Flag it so the agent does not present stale info as current
            entry["superseded"] = True
        if explain and mid in explain_data:
            ex = dict(explain_data[mid])
            if mid in superseded:
                ex["superseded_penalty"] = SUPERSEDED_PENALTY
            entry["explain"] = ex

        # Paging is applied AFTER the filters above, so offset counts results
        # the caller would actually have seen rather than pre-filter candidates.
        if skipped < offset:
            skipped += 1
            continue

        results.append(entry)
        if len(results) >= top_k:
            # Look one past the page to answer has_more honestly, instead of the
            # `len(results) >= limit` guess the other paginated tools use (which
            # claims has_more on an exact-boundary final page).
            more_available = True
            break

    # Bump access counts
    if results:
        ids = [r["id"] for r in results]
        _safe_execute(conn,
            "MATCH (m:Memory) WHERE m.id IN $ids "
            "SET m.access_count = m.access_count + 1;",
            {"ids": ids})

    # --- Graph neighbours (query-dependent traversal, reported separately) ---
    # PageRank, K-Core and community membership are query-INDEPENDENT: they say
    # a memory is globally central, not that it answers THIS query. They also
    # need memory_dream to have run and stay uniform until the graph has edges.
    #
    # The useful graph move is to expand one hop from what the query actually
    # matched — the incident caused by a decision, the rationale behind a
    # convention, the correction to a stale fact. Those are deliberately NOT
    # mixed into `results`: raw cosine similarity has a high floor (~0.5 even
    # for unrelated text), so every similarity candidate carries a large
    # constant while a graph-only hit caps at the 0.15 graph weight. Boosting
    # graph enough to compete would let loosely-linked memories crowd out
    # direct answers. Reporting them separately preserves ranking precision
    # and still gives the agent the connection explicitly.
    # --- Unresolved conflict detection ---
    # SUPERSEDES resolves conflicts an agent has already noticed. This catches
    # the ones nobody wired: two returned memories that are near-duplicates of
    # each other with NO edge between them are either competing versions or
    # contradictory facts, and nothing marks which is current. That is the
    # normal shape of accumulated agent memory — a fact learned in one session
    # and a conflicting one learned later.
    #
    # Reported as "potential", not "contradiction": distinguishing a genuine
    # contradiction from two complementary facts needs entailment, which would
    # mean an LLM call in the server. Flagging the ambiguity is honest and free;
    # deciding it is the agent's job.
    conflicts: list[dict] = []
    if CONFLICT_DETECTION and len(results) > 1:
        try:
            ids = [r["id"] for r in results]
            tags_by_id = {r["id"]: set(r.get("tags") or []) for r in results}
            # FULL content, not the truncated preview: the value-disagreement
            # check compares numbers and named values, and a 200-char preview
            # can cut off the very token that differs.
            r = conn.execute(
                "MATCH (m:Memory) WHERE m.id IN $ids RETURN m.id, m.embedding, m.content;",
                {"ids": ids},
            )
            rows = _collect_results(r)
            vecs = {row[0]: row[1] for row in rows if row[1]}
            contents_by_id = {row[0]: (row[2] or "") for row in rows}
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    va, vb = vecs.get(a), vecs.get(b)
                    if not va or not vb:
                        continue
                    dot = sum(x * y for x, y in zip(va, vb))
                    na = sum(x * x for x in va) ** 0.5
                    nb = sum(y * y for y in vb) ** 0.5
                    if not na or not nb:
                        continue
                    sim = dot / (na * nb)

                    # Same-subject check first, since BOTH detection paths need
                    # it. Similarity alone over-reports: templated facts about
                    # DIFFERENT subjects read almost identically ("Checkout is
                    # written in Java 17" vs "Billing is written in Java 17")
                    # without conflicting at all. Tag overlap is the cheap
                    # subject proxy. Skipped when either side is untagged.
                    ta, tb = tags_by_id.get(a) or set(), tags_by_id.get(b) or set()
                    same_subject = True
                    if ta and tb:
                        union = ta | tb
                        same_subject = bool(union) and (len(ta & tb) / len(union)) >= CONFLICT_TAG_OVERLAP
                    if not same_subject:
                        continue

                    # Two independent reasons to flag a pair:
                    #
                    #  near_duplicate — high cosine. Catches competing versions
                    #    phrased alike.
                    #
                    #  value_disagreement — same subject and a disagreeing
                    #    value token, at ANY cosine above a low floor. This
                    #    closes the "middle gap": a correction rewritten from
                    #    scratch rather than edited scores too LOW to look like
                    #    a duplicate, so cosine-only detection missed it
                    #    entirely and the stale version outranked the current
                    #    one with no flag. Measured: "Izar retains audit logs
                    #    for 30 days" (0.7677, ranked #1) vs "retention on Izar
                    #    was extended to a full year" (0.6208) — same subject,
                    #    plainly contradictory, silent. Perversely, the more
                    #    thoroughly an agent rewords a correction, the less
                    #    likely cosine was to notice.
                    near_duplicate = sim >= CONFLICT_THRESHOLD
                    # Below the near-duplicate threshold, differing values are
                    # NOT enough on their own. "X depends on Redis for caching"
                    # and "X depends on Kafka for event delivery" disagree on a
                    # value token while both being true — and that shape (same
                    # entity, same predicate, different object) is the norm in a
                    # dependency or architecture graph, so flagging it buried
                    # the signal in noise. Measured false positives on organic
                    # corpus pairs at 0.73-0.82, including two memories about
                    # DIFFERENT subjects (a search feature and a checkout flow)
                    # that shared only their tags.
                    #
                    # Two independent marker-free-or-not signals, either of
                    # which is enough:
                    #
                    #  a correction marker  — catches NON-quantitative
                    #     corrections ("now uses HALF_EVEN"), where there is no
                    #     magnitude to compare.
                    #  comparable quantities — catches corrections that carry NO
                    #     marker, which is the case that matters most: an agent
                    #     that knows it is correcting would pass supersedes= and
                    #     never need this. "retains audit logs for 30 days" vs
                    #     "are kept for one year" states the same dimension with
                    #     different magnitudes and announces nothing.
                    text_a = contents_by_id.get(a, "")
                    text_b = contents_by_id.get(b, "")
                    value_disagreement = (
                        sim >= CONFLICT_VALUE_FLOOR
                        and (
                            (_has_correction_marker(text_a, text_b)
                             and _values_conflict(text_a, text_b))
                            or _quantities_disagree(text_a, text_b)
                        )
                    )
                    if not (near_duplicate or value_disagreement):
                        continue

                    # Any agent-asserted edge dismisses the flag, RELATED_TO
                    # included — that is what the hint tells the agent to write.
                    if _conflict_dismissed(conn, a, b):
                        continue

                    if near_duplicate:
                        reason = "near_duplicate"
                        hint = (f"Near-identical memories with no edge marking which is "
                                f"current. If one replaces the other, re-store the current "
                                f"version with memory_store(..., supersedes=<old_id>). If both "
                                f"are true, call memory_relate(from_id={a}, to_id={b}, "
                                f"relationship='RELATED_TO') — that dismisses this flag "
                                f"permanently.")
                    else:
                        _v = _name_vocab(text_a, text_b)
                        differing = sorted(
                            str(t) for t in (
                                _discriminating_tokens(text_a, _v)
                                ^ _discriminating_tokens(text_b, _v)
                            )
                        )[:6]
                        qa, qb = _extract_quantities(text_a), _extract_quantities(text_b)
                        shared = [f for f in set(qa) & set(qb) if qa[f] != qb[f]]
                        if shared:
                            # Naming the dimension is more actionable than the raw
                            # tokens: "these both state a duration, and they differ".
                            detail = (f"both state a {', '.join(sorted(shared))} but the "
                                      f"values differ")
                        else:
                            detail = f"they disagree on {differing}"
                        reason = "value_disagreement"
                        hint = (f"These are about the same subject and {detail}. They are not "
                                f"phrased alike, so nothing else would flag them. If one "
                                f"replaces the other, re-store it with "
                                f"memory_store(..., supersedes=<old_id>). If both are true, "
                                f"call memory_relate(from_id={a}, to_id={b}, "
                                f"relationship='RELATED_TO') — that dismisses this flag "
                                f"permanently.")

                    conflicts.append({
                        "ids": [a, b],
                        "similarity": round(sim, 4),
                        "reason": reason,
                        "hint": hint,
                    })
                    if len(conflicts) >= 3:
                        break
                if len(conflicts) >= 3:
                    break
        except Exception as e:
            logger.debug(f"Conflict detection failed (non-fatal): {e}")

    related: list[dict] = []
    if results and GRAPH_EXPAND_SEEDS > 0:
        try:
            # Only expand from STRONG matches. A weak hit is a coincidence, and
            # its neighbours are noise: observed a pricing correction placing
            # rank 3 at 53% of the top score on a billing query, dragging its
            # superseded pair into `related` where it had no business being.
            floor = results[0]["score"] * GRAPH_EXPAND_MIN_RATIO
            seeds = [r["id"] for r in results[:GRAPH_EXPAND_SEEDS]
                     if r["score"] >= floor]
            if not seeds:
                raise StopIteration  # nothing confident enough to expand from
            returned = {r["id"] for r in results}
            r = conn.execute(
                """MATCH (s:Memory)-[e:RELATED_TO|SUPERSEDES|EXPLAINS]-(n:Memory)
                   WHERE s.id IN $seeds
                   RETURN DISTINCT n.id, n.content, n.importance, s.id;""",
                {"seeds": seeds},
            )
            for row in _collect_results(r):
                if row[0] in returned:
                    continue  # already ranked on its own merits
                related.append({
                    "id": row[0],
                    "content": _truncate(row[1], preview_chars),
                    "linked_to": row[3],
                })
                if len(related) >= GRAPH_EXPAND_LIMIT:
                    break
        except StopIteration:
            pass  # no seed cleared the relevance floor
        except Exception as e:
            logger.debug(f"Neighbour expansion failed (non-fatal): {e}")

    # Report a dead semantic channel. Without this, a failed embedding model
    # silently turns hybrid search into keyword-only search that still returns
    # plausible-looking scores — the caller has no way to know retrieval is
    # degraded. See _embed()/vector-search error logging above.
    out: dict = {"results": results}
    if offset or more_available:
        out["offset"] = offset
        out["has_more"] = more_available
    if related:
        out["related"] = related
    if conflicts:
        out["potential_conflicts"] = conflicts
    if explain:
        out["explain_meta"] = {
            "fusion_mode": FUSION_MODE,
            "weights": {"vector": 0.4, "fts": 0.3, "graph": 0.15,
                        "recency": 0.1, "importance": 0.05},
            "embedding_model": EMBEDDING_MODEL,
            "query_embedded": embedding is not None,
            # The vector channel asks the HNSW index for the k nearest
            # neighbours GLOBALLY (k = top_k*3) and workspace-filters after,
            # so these two numbers say how much of the window this query used
            # and how many candidates survived the filter.
            "candidate_pool": _pool,
            "vector_hits": len(vector_hits),
            "candidates_scored": len(raw_scores),
            "superseded_penalty": SUPERSEDED_PENALTY,
        }
    if not vector_hits and _count_memories(conn) > 0:
        out["degraded"] = (
            "Semantic (vector) search returned nothing — results are keyword-only. "
            "Causes: the embedding model failed to load, these memories were stored "
            "without embeddings, or the HNSW index is stale. Call memory_stats() and "
            "check runtime.embeddings / runtime.vector_index, then run "
            "memory_reindex() to rebuild the index."
        )

    # Wrap in a key so TOON can recognize the uniform array
    return out


@mcp.tool()
@_timed("memory_update")
def memory_update(memory_id: Optional[int] = None, content: Optional[str] = None,
                  importance: Optional[int] = None, tags: Optional[list[str]] = None,
                  updates: Optional[list[dict]] = None) -> str:
    """Update one or more memories. Preserves relationships across content changes.
    Single: pass memory_id + any of content/importance/tags.
    Batch: pass updates=[{memory_id, content?, importance?, tags?}, ...].
    """
    conn = get_conn()

    # Batch mode
    if updates is not None:
        if not isinstance(updates, list) or not updates:
            return {"status": "error", "message": "updates must be a non-empty list."}
        if len(updates) > MAX_BATCH_ITEMS:
            return {"status": "error",
                    "message": f"Too many updates ({len(updates)}, limit {MAX_BATCH_ITEMS})."}

        # Pre-compute embeddings for all items that change content
        contents_to_embed = [(i, _truncate_content(u.get("content"))[0])
                             for i, u in enumerate(updates)
                             if u.get("content") is not None]
        embeddings_by_idx = {}
        if contents_to_embed:
            indexes, texts = zip(*contents_to_embed)
            embs = _embed_batch(list(texts))
            embeddings_by_idx = dict(zip(indexes, embs))

        results = []
        for i, u in enumerate(updates):
            try:
                res = _update_one(
                    conn,
                    memory_id=u.get("memory_id"),
                    content=(_truncate_content(u["content"])[0]
                             if u.get("content") is not None else None),
                    importance=u.get("importance"),
                    tags=(_limit_tags(u["tags"]) if u.get("tags") is not None else None),
                    embedding=embeddings_by_idx.get(i),
                )
                results.append(res)
            except Exception as e:
                results.append({"status": "error", "memory_id": u.get("memory_id"),
                                "message": str(e)})

        _bump_dream_ops()
        return {"results": results, "count": len(results)}

    # Single mode
    if memory_id is None:
        return {"status": "error",
                           "message": "memory_id is required (or pass updates=[...])."}
    if content is None and importance is None and tags is None:
        return {"status": "error",
                "message": "Nothing to update: pass at least one of content, importance, tags."}

    if content is not None:
        content, truncated = _truncate_content(content)
    else:
        truncated = False
    if tags is not None:
        tags = _limit_tags(tags)

    res = _update_one(conn, memory_id, content, importance, tags)
    if truncated:
        res["content_truncated_to"] = MAX_STORE_CHARS
    _bump_dream_ops()
    return res


def _update_one(conn, memory_id: int, content: Optional[str] = None,
                importance: Optional[int] = None, tags: Optional[list[str]] = None,
                embedding: Optional[list[float]] = None) -> dict:
    """Core single-memory update logic. If `embedding` is pre-computed, uses it."""
    result = conn.execute(
        """MATCH (m:Memory {id: $id})
           RETURN m.id, m.content, m.content_hash, m.category, m.tags,
                  m.importance, m.access_count, m.created_at, m.updated_at, m.workspace;""",
        {"id": memory_id},
    )
    if not result.has_next():
        return {"status": "not_found", "id": memory_id,
                "message": f"Memory {memory_id} not found."}

    r = result.get_next()
    now = time.time()

    if content is not None:
        new_hash = _content_hash(content)
        new_emb = embedding if embedding is not None else _embed(content)
        if new_emb is None:
            return {"status": "error", "id": memory_id,
                    "message": "Embedding generation failed; update aborted to preserve data."}
        new_tags = _format_tags(tags) if tags is not None else r[4]
        new_imp = min(5, max(1, importance)) if importance is not None else r[5]

        # Save and restore relationships across the delete + recreate
        saved_rels = _save_memory_relationships(conn, memory_id)
        conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": memory_id})
        conn.execute(
            """CREATE (m:Memory {
                   id: $id, content: $content, content_hash: $hash,
                   category: $cat, tags: $tags, workspace: $ws, importance: $imp,
                   access_count: $ac, created_at: $ca, updated_at: $now,
                   embedding: $emb
               });""",
            {"id": memory_id, "content": content, "hash": new_hash,
             "cat": r[3], "tags": new_tags, "ws": r[9] or WORKSPACE, "imp": new_imp,
             "ac": r[6] or 0, "ca": r[7] or now, "now": now, "emb": new_emb},
        )
        parsed_tags = _parse_tags(new_tags)
        _ensure_topics(conn, memory_id, parsed_tags)
        _restore_memory_relationships(conn, memory_id, saved_rels)
    else:
        if importance is not None:
            conn.execute(
                "MATCH (m:Memory {id: $id}) SET m.importance = $imp, m.updated_at = $now;",
                {"id": memory_id, "imp": min(5, max(1, importance)), "now": now},
            )
        if tags is not None:
            conn.execute(
                "MATCH (m:Memory {id: $id}) SET m.tags = $tags, m.updated_at = $now;",
                {"id": memory_id, "tags": _format_tags(tags), "now": now},
            )
            _safe_execute(conn, "MATCH (m:Memory {id: $id})-[r:ABOUT]->() DELETE r;",
                          {"id": memory_id})
            _ensure_topics(conn, memory_id, tags)

    return {"status": "updated", "id": memory_id}


@mcp.tool()
@_timed("memory_delete")
def memory_delete(memory_id: int | list[int]) -> str:
    """Delete one or more memories (and their relationships). Pass int or list of ints.

    To remove only an EDGE and keep both memories, use memory_unrelate.
    """
    conn = get_conn()
    ids = memory_id if isinstance(memory_id, list) else [memory_id]
    if not ids:
        return {"status": "error", "message": "memory_id is required."}
    if len(ids) > MAX_BATCH_ITEMS:
        return {"status": "error",
                "message": f"Too many ids ({len(ids)}, limit {MAX_BATCH_ITEMS})."}

    deleted = []
    not_found = []
    invalid = []

    for mid in ids:
        if not isinstance(mid, int) or isinstance(mid, bool):
            invalid.append(mid)
            continue
        result = conn.execute("MATCH (m:Memory {id: $id}) RETURN m.id;", {"id": mid})
        if not result.has_next():
            not_found.append(mid)
            continue
        conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": mid})
        deleted.append(mid)

    # The status used to be a hardcoded "deleted" even when nothing was deleted
    # and every id landed in not_found, so a caller could not tell the
    # difference from the status alone.
    if deleted and not (not_found or invalid):
        status = "deleted"
    elif deleted:
        status = "partial"
    else:
        status = "not_found" if not invalid else "error"

    out = {"status": status, "deleted": deleted, "not_found": not_found}
    if invalid:
        out["invalid"] = invalid
    return out


# SECURITY: relationship labels are interpolated into Cypher (LadybugDB does
# not support parameterized rel labels), so every path that names one validates
# against this single allowlist. Never accept an arbitrary label.
EDGE_TYPES = ("RELATED_TO", "SUPERSEDES", "EXPLAINS")


def _relate_one(conn, from_id: int, to_id: int, relationship: str = "RELATED_TO",
                confidence: float = 1.0, provenance: str = "EXTRACTED") -> dict:
    """Core single-relationship creation logic."""
    rel = str(relationship or "").upper()
    if rel not in EDGE_TYPES:
        return {"status": "error", "from": from_id, "to": to_id,
                "message": f"Unknown relationship: {rel}. Use {', '.join(EDGE_TYPES)}."}

    if from_id is None or to_id is None:
        return {"status": "error", "from": from_id, "to": to_id,
                "message": "from_id and to_id are required."}

    # Pre-flight existence check. LadybugDB's `MATCH ... CREATE` silently no-ops
    # when either side is missing, so we must verify both nodes exist before
    # claiming success.
    missing = []
    for tag, mid in (("from_id", from_id), ("to_id", to_id)):
        r = conn.execute("MATCH (m:Memory {id: $id}) RETURN m.id;", {"id": mid})
        if not r.has_next():
            missing.append({tag: mid})
    if missing:
        return {"status": "not_found", "from": from_id, "to": to_id,
                "missing": missing}

    # Idempotent: re-asserting an edge must not create a parallel duplicate.
    # dream already has to defend against those (_dedupe_related exists because
    # duplicates "silently inflate the relationship count and confuse traversal
    # queries"), and re-importing an export used to double every edge.
    try:
        existing = conn.execute(
            f"MATCH (a:Memory {{id: $f}})-[r:{rel}]->(b:Memory {{id: $t}}) RETURN COUNT(r);",
            {"f": from_id, "t": to_id},
        )
        if existing.has_next() and (existing.get_next()[0] or 0) > 0:
            return {"status": "exists", "from": from_id, "to": to_id, "type": rel}
    except Exception as e:
        logger.debug(f"Duplicate-edge check failed for {rel} {from_id}->{to_id}: {e}")

    try:
        if rel == "RELATED_TO":
            confidence = max(0.0, min(1.0, confidence))
            prov = provenance.upper()
            if prov not in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
                prov = "EXTRACTED"
            conn.execute(
                """MATCH (a:Memory {id: $from}), (b:Memory {id: $to})
                   CREATE (a)-[:RELATED_TO {provenance: $prov, confidence: $conf}]->(b);""",
                {"from": from_id, "to": to_id, "prov": prov, "conf": confidence},
            )
            return {"status": "created", "from": from_id, "to": to_id, "type": rel}
        else:
            conn.execute(
                f"MATCH (a:Memory {{id: $from}}), (b:Memory {{id: $to}}) CREATE (a)-[:{rel}]->(b);",
                {"from": from_id, "to": to_id},
            )
            return {"status": "created", "from": from_id, "to": to_id, "type": rel}
    except Exception as e:
        return {"status": "error", "from": from_id, "to": to_id, "message": str(e)}


@mcp.tool()
@_timed("memory_relate")
def memory_relate(from_id: Optional[int] = None, to_id: Optional[int] = None,
                  relationship: str = "RELATED_TO",
                  confidence: float = 1.0, provenance: str = "EXTRACTED",
                  relations: Optional[list[dict]] = None) -> str:
    """Create relationships. Types: RELATED_TO, SUPERSEDES, EXPLAINS.
    For RELATED_TO: confidence 0-1, provenance EXTRACTED|INFERRED|AMBIGUOUS.
    Single: pass from_id/to_id. Batch: pass relations=[{from_id, to_id, relationship?, ...}, ...].
    """
    conn = get_conn()

    # Batch mode
    if relations is not None:
        if not isinstance(relations, list) or not relations:
            return {"status": "error", "message": "relations must be a non-empty list."}
        if len(relations) > MAX_BATCH_ITEMS:
            return {"status": "error",
                    "message": f"Too many relations ({len(relations)}, limit {MAX_BATCH_ITEMS})."}
        results = []
        for r in relations:
            res = _relate_one(
                conn,
                from_id=r.get("from_id"),
                to_id=r.get("to_id"),
                relationship=r.get("relationship", "RELATED_TO"),
                confidence=r.get("confidence", 1.0),
                provenance=r.get("provenance", "EXTRACTED"),
            )
            results.append(res)
        return {"results": results, "count": len(results)}

    # Single mode
    if from_id is None or to_id is None:
        return {"status": "error",
                           "message": "from_id and to_id are required (or pass relations=[...])."}

    res = _relate_one(conn, from_id, to_id, relationship, confidence, provenance)
    return res


def _unrelate_one(conn, from_id: int, to_id: int,
                  relationship: Optional[str] = None,
                  both_directions: bool = False) -> dict:
    """Core single-relationship removal logic.

    relationship=None removes every edge type between the pair, which is what a
    caller undoing a mistake usually wants (they know the two memories should
    not be linked, not necessarily which label was written).
    """
    if from_id is None or to_id is None:
        return {"status": "error", "from": from_id, "to": to_id,
                "message": "from_id and to_id are required."}

    if relationship is None:
        types = list(EDGE_TYPES)
    else:
        rel = str(relationship).upper()
        if rel not in EDGE_TYPES:
            return {"status": "error", "from": from_id, "to": to_id,
                    "message": f"Unknown relationship: {rel}. Use {', '.join(EDGE_TYPES)}."}
        types = [rel]

    removed: list[str] = []
    for rel in types:
        # Count first: LadybugDB's DELETE reports nothing, so without this the
        # caller cannot tell "removed" from "there was no such edge".
        pairs = [(from_id, to_id)] + ([(to_id, from_id)] if both_directions else [])
        for f, t in pairs:
            try:
                r = conn.execute(
                    f"MATCH (a:Memory {{id: $f}})-[r:{rel}]->(b:Memory {{id: $t}}) "
                    f"RETURN COUNT(r);",
                    {"f": f, "t": t},
                )
                count = r.get_next()[0] if r.has_next() else 0
                if not count:
                    continue
                conn.execute(
                    f"MATCH (a:Memory {{id: $f}})-[r:{rel}]->(b:Memory {{id: $t}}) DELETE r;",
                    {"f": f, "t": t},
                )
                removed.extend([f"{f}-[{rel}]->{t}"] * count)
            except Exception as e:
                return {"status": "error", "from": from_id, "to": to_id,
                        "message": f"{rel}: {e}"}

    if not removed:
        return {"status": "not_found", "from": from_id, "to": to_id,
                "searched": types,
                "message": "No such edge between these memories."}
    return {"status": "deleted", "from": from_id, "to": to_id, "removed": removed}


@mcp.tool()
@_timed("memory_unrelate")
def memory_unrelate(from_id: Optional[int] = None, to_id: Optional[int] = None,
                    relationship: Optional[str] = None,
                    both_directions: bool = False,
                    relations: Optional[list[dict]] = None) -> str:
    """Remove a relationship between two memories. The inverse of memory_relate.

    Omit `relationship` to remove every edge type between the pair; pass
    RELATED_TO, SUPERSEDES or EXPLAINS to remove just that one. Edges are
    directed, so this removes from_id -> to_id; pass both_directions=True to
    remove the reverse as well.
    Batch: relations=[{from_id, to_id, relationship?}, ...].

    Use this to undo a wrong edge — a SUPERSEDES pointing the wrong way, or a
    correction chain wired to the wrong version. It is also the supported way to
    break a circular SUPERSEDES chain, which memory_dream reports under
    `contradictions`. The memories themselves are untouched; only the edge goes.
    """
    conn = get_conn()

    if relations is not None:
        if not relations:
            return {"status": "error", "message": "relations list is empty."}
        if len(relations) > MAX_BATCH_ITEMS:
            return {"status": "error",
                    "message": f"Too many relations ({len(relations)}, limit {MAX_BATCH_ITEMS})."}
        results = [
            _unrelate_one(
                conn,
                from_id=r.get("from_id"),
                to_id=r.get("to_id"),
                relationship=r.get("relationship"),
                both_directions=bool(r.get("both_directions", False)),
            )
            for r in relations
        ]
        return {"results": results, "count": len(results)}

    if from_id is None or to_id is None:
        return {"status": "error",
                "message": "from_id and to_id are required (or pass relations=[...])."}

    return _unrelate_one(conn, from_id, to_id, relationship, both_directions)


# Configurable: bool to allow destructive queries through memory_query.
# Defaults to FALSE for safety — an MCP server is exposed to LLM agents which can
# (and have) hallucinated DELETE queries. Operators must explicitly opt in.
ALLOW_DESTRUCTIVE_QUERIES = os.environ.get("MEMORY_ALLOW_DESTRUCTIVE", "false").lower() == "true"


# Query classification for memory_query's safety gate.
#
# The previous implementation matched raw substrings against query.upper():
# ("DETACH DELETE", "DELETE ", "DROP ", "TRUNCATE"). Two holes, both verified:
#
#   1. "DELETE " requires a literal trailing SPACE, so any other whitespace
#      slipped through. With MEMORY_ALLOW_DESTRUCTIVE=false,
#      "MATCH (m:Memory)\nDETACH\nDELETE\nm;" reported ordinary success and
#      deleted every memory in the database.
#   2. Only removal was considered. SET/REMOVE/COPY were never checked, so
#      read_only=True permitted overwrites:
#      "MATCH (m:Memory {id: 1}) SET m.content = 'OVERWRITTEN';" succeeded
#      under read_only=True and replaced the content.
#
# Now: strip comments and string literals (so a keyword inside a quoted value
# or a /* */ comment neither triggers nor hides a match), then match keywords on
# word boundaries. Underscore-suffixed procedure names (DROP_VECTOR_INDEX) are
# matched separately because "_" is a word character, so \bDROP\b misses them.
_CYPHER_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_CYPHER_STRING_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")

# Removes or overwrites data that already exists. Gated by
# MEMORY_ALLOW_DESTRUCTIVE. SET and REMOVE are included deliberately: an
# overwrite destroys the previous value just as surely as a DELETE, and an
# agent hallucinating `SET m.content = ...` is the same class of accident.
_DESTRUCTIVE_KEYWORDS = ("DELETE", "DETACH", "DROP", "TRUNCATE", "REMOVE", "SET", "COPY")
# Adds data. Rejected by read_only=True but allowed by default, since a caller
# reaching for memory_query to CREATE something is not destroying anything.
_ADDITIVE_KEYWORDS = ("CREATE", "MERGE", "ALTER")


def _strip_cypher_noise(query: str) -> str:
    """Remove comments and string literals so keyword matching sees only code."""
    q = _CYPHER_COMMENT_RE.sub(" ", query)
    return _CYPHER_STRING_RE.sub("''", q)


def _has_keyword(query: str, keywords: tuple) -> bool:
    stripped = _strip_cypher_noise(query).upper()
    for kw in keywords:
        # \bKW\b for bare keywords, \bKW_ for procedure names like DROP_FTS_INDEX.
        if re.search(rf"\b{kw}\b|\b{kw}_", stripped):
            return True
    return False


def _is_destructive(query: str) -> bool:
    """True if the query can remove or overwrite existing data."""
    return _has_keyword(query, _DESTRUCTIVE_KEYWORDS)


def _is_write(query: str) -> bool:
    """True if the query mutates the database at all, additively or otherwise."""
    return _has_keyword(query, _DESTRUCTIVE_KEYWORDS + _ADDITIVE_KEYWORDS)


def _is_unsafe_embedding_set(query: str) -> bool:
    """Detect SET on m.embedding, which fails silently due to the HNSW index.

    Whitespace-insensitive: the old version only stripped spaces, so a tab or
    newline between SET and the property evaded it.
    """
    collapsed = re.sub(r"\s+", "", _strip_cypher_noise(query).upper())
    return "SETM.EMBEDDING" in collapsed or "SETEMBEDDING" in collapsed


@mcp.tool()
@_timed("memory_query")
def memory_query(cypher_query: str, read_only: bool = False) -> str:
    """Run a Cypher query. Supports traversals, writes, INSTALL/LOAD, CALL (algorithms, scans).
    Call memory_schema() first to get table/column names.
    WARNING: SET on m.embedding fails (vector index). Use memory_update for content changes.

    read_only=True rejects ANY mutation (CREATE/MERGE/SET/DELETE/DROP/...), not
    just removals. MEMORY_ALLOW_DESTRUCTIVE (default false) independently blocks
    anything that removes or overwrites existing data — DELETE, DROP, TRUNCATE,
    REMOVE, SET, COPY — so use memory_update to change a memory and
    memory_unrelate to remove an edge.
    """
    conn = get_conn()

    if not isinstance(cypher_query, str) or not cypher_query.strip():
        return {"status": "error", "message": "cypher_query is required."}
    if len(cypher_query) > MAX_QUERY_CHARS:
        return {
            "status": "error",
            "message": f"Query too long ({len(cypher_query)} chars, limit {MAX_QUERY_CHARS}).",
        }

    # Caller-requested read-only check (more specific — runs first so the error
    # message reflects the caller's intent rather than the global flag).
    # This now covers additive writes too: read_only=True used to permit
    # CREATE/MERGE/SET, which made the parameter's name untrue.
    if read_only and _is_write(cypher_query):
        return {
            "status": "error",
            "message": "Query blocked by read_only=True: it mutates the database.",
        }
    # Server-level kill switch for anything that removes or overwrites data.
    if not ALLOW_DESTRUCTIVE_QUERIES and _is_destructive(cypher_query):
        return {
            "status": "error",
            "message": "Destructive query blocked by server config "
                       "(MEMORY_ALLOW_DESTRUCTIVE=false). This covers DELETE, DROP, "
                       "TRUNCATE, REMOVE, SET and COPY. Use memory_update to change a "
                       "memory, memory_delete to remove one, and memory_unrelate to "
                       "remove an edge.",
        }

    # Block SET on m.embedding — silent failure due to HNSW vector index
    if _is_unsafe_embedding_set(cypher_query):
        return {
            "status": "error",
            "message": "SET on m.embedding fails (HNSW vector index). Use memory_update for content changes.",
        }

    if _is_destructive(cypher_query):
        # Log destructive ops at INFO so they're visible in production logs
        preview = cypher_query.strip().replace("\n", " ")[:200]
        logger.info(f"Destructive query executed: {preview}")

    try:
        result = conn.execute(cypher_query)
        rows = _collect_results(result)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
@_timed("memory_schema")
def memory_schema() -> str:
    """Return live DB schema: tables, columns, types, indexes, extensions.
    Call before writing memory_query Cypher.
    """
    conn = get_conn()

    nodes: dict = {}
    rels: dict = {}
    indexes: list = []
    extensions: list = []

    try:
        tables = _collect_results(conn.execute("CALL SHOW_TABLES() RETURN *;"))
        for t in tables:
            name = t[1] if len(t) > 1 else t[0]
            ttype = t[2] if len(t) > 2 else "UNKNOWN"
            is_rel = "REL" in str(ttype).upper()

            try:
                cols = _collect_results(conn.execute(f"CALL TABLE_INFO('{name}') RETURN *;"))
                # Compact column representation: "name:TYPE" or "name:TYPE!" for primary key
                col_strs = []
                for c in cols:
                    col_name = c[1] if len(c) > 1 else "?"
                    col_type = c[2] if len(c) > 2 else "?"
                    is_pk = (not is_rel) and bool(c[4]) if len(c) > 4 else False
                    col_strs.append(f"{col_name}:{col_type}" + ("!" if is_pk else ""))
                col_summary = ", ".join(col_strs)

                if is_rel:
                    try:
                        connectivity = _collect_results(
                            conn.execute(f"CALL SHOW_CONNECTION('{name}') RETURN *;")
                        )
                        endpoints = ", ".join(f"{r[0]}->{r[1]}" for r in connectivity)
                    except Exception:
                        endpoints = "?"
                    rels[name] = {"endpoints": endpoints, "props": col_summary}
                else:
                    nodes[name] = col_summary
            except Exception as e:
                logger.warning(f"TABLE_INFO failed for {name}: {e}")
    except Exception as e:
        return {"status": "error", "message": f"SHOW_TABLES failed: {e}"}

    try:
        idx_rows = _collect_results(conn.execute("CALL SHOW_INDEXES() RETURN *;"))
        for idx in idx_rows:
            tbl = idx[0] if len(idx) > 0 else "?"
            iname = idx[1] if len(idx) > 1 else "?"
            itype = idx[2] if len(idx) > 2 else "?"
            indexes.append(f"{tbl}.{iname}({itype})")
    except Exception:
        pass

    try:
        exts = _collect_results(conn.execute("CALL SHOW_LOADED_EXTENSIONS() RETURN *;"))
        extensions = [e[0] if len(e) > 0 else str(e) for e in exts]
    except Exception:
        pass

    return {
        "nodes": nodes,
        "rels": rels,
        "indexes": indexes,
        "extensions": extensions,
        "notes": "PK marked with !. Use memory_query for Cypher; cannot SET m.embedding (HNSW index).",
    }


@mcp.tool()
@_timed("memory_topics")
def memory_topics(limit: int = 50, offset: int = 0, min_count: int = 1,
                  global_search: bool = False) -> str:
    """List all topics (tags) with memory counts, sorted by usage.
    Use this to discover available filters for memory_search(tags=[...]).
    Filters by current workspace unless global_search=True. min_count filters out rare topics.
    """
    conn = get_conn()
    limit = _clamp_int(limit, 1, 500, 50)
    offset = _clamp_int(offset, 0, 1_000_000, 0)
    min_count = _clamp_int(min_count, 1, 1_000_000, 1)

    if global_search:
        result = conn.execute(
            """MATCH (t:Topic)<-[:ABOUT]-(m:Memory)
               WITH t.name AS topic, COUNT(m) AS count
               WHERE count >= $min_count
               RETURN topic, count
               ORDER BY count DESC, topic ASC
               SKIP $offset LIMIT $limit;""",
            {"min_count": min_count, "offset": offset, "limit": limit},
        )
    else:
        result = conn.execute(
            """MATCH (t:Topic)<-[:ABOUT]-(m:Memory)
               WHERE m.workspace IN ['', $ws]
               WITH t.name AS topic, COUNT(m) AS count
               WHERE count >= $min_count
               RETURN topic, count
               ORDER BY count DESC, topic ASC
               SKIP $offset LIMIT $limit;""",
            {"ws": WORKSPACE, "min_count": min_count, "offset": offset, "limit": limit},
        )

    topics = [{"topic": row[0], "count": row[1]} for row in _collect_results(result)]
    return {
        "topics": topics,
        "returned": len(topics),
        "offset": offset,
        "has_more": len(topics) >= limit,
    }


@mcp.tool()
@_timed("memory_stats")
def memory_stats() -> str:
    """Database statistics: counts, categories, importance distribution, top topics, god nodes."""
    conn = get_conn()

    total = _count_memories(conn)

    # Categories
    cats = {}
    for row in _collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.category, COUNT(*) ORDER BY COUNT(*) DESC;"
    )):
        cats[row[0]] = row[1]

    # Importance
    imp_dist = {}
    for row in _collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.importance, COUNT(*) ORDER BY m.importance;"
    )):
        imp_dist[str(row[0])] = row[1]

    # Topics — top 10 by usage, plus total count for discoverability
    topics = {}
    for row in _collect_results(conn.execute(
        "MATCH (t:Topic)<-[:ABOUT]-(m:Memory) RETURN t.name, COUNT(m) ORDER BY COUNT(m) DESC LIMIT 10;"
    )):
        topics[row[0]] = row[1]

    # Total distinct topics — useful so the agent knows there's more beyond the top 10
    total_topics = 0
    try:
        r = conn.execute("MATCH (t:Topic) RETURN COUNT(t);")
        if r.has_next():
            total_topics = r.get_next()[0] or 0
    except Exception:
        pass

    # Relationships
    rel_count = 0
    try:
        r = conn.execute("MATCH ()-[r]->() RETURN COUNT(r);")
        if r.has_next():
            rel_count = r.get_next()[0]
    except Exception:
        pass

    # Combined ranking: composite score from access_count + degree
    # Replaces separate "most_accessed" and "god_nodes" (which usually overlapped heavily)
    top_memories = []
    try:
        # Fetch access_count for all memories
        access_by_id: dict[int, int] = {}
        for row in _collect_results(conn.execute(
            "MATCH (m:Memory) RETURN m.id, m.access_count;"
        )):
            access_by_id[row[0]] = row[1] or 0

        # Compute degree
        degree_by_id: dict[int, int] = {}
        for q in (
            "MATCH (m:Memory)-[r]->() RETURN m.id, COUNT(r);",
            "MATCH ()-[r]->(m:Memory) RETURN m.id, COUNT(r);",
        ):
            for row in _collect_results(conn.execute(q)):
                degree_by_id[row[0]] = degree_by_id.get(row[0], 0) + (row[1] or 0)

        # Composite score: access_count + degree (equal weight)
        all_ids = set(access_by_id) | set(degree_by_id)
        scored = [
            (tid, access_by_id.get(tid, 0), degree_by_id.get(tid, 0))
            for tid in all_ids
        ]
        scored.sort(key=lambda x: -(x[1] + x[2]))
        top_5 = scored[:5]

        if top_5:
            ids = [s[0] for s in top_5]
            content_by_id: dict[int, str] = {}
            content_result = conn.execute(
                "MATCH (m:Memory) WHERE m.id IN $ids RETURN m.id, m.content;",
                {"ids": ids},
            )
            for row in _collect_results(content_result):
                content_by_id[row[0]] = row[1]
            for tid, ac, deg in top_5:
                content = content_by_id.get(tid, "")
                top_memories.append({
                    "id": tid,
                    "preview": content[:60] if content else "",
                    "accessed": ac,
                    "degree": deg,
                })
    except Exception as e:
        logger.warning(f"Top memories query failed: {e}")

    # Embedding health: memories stored while the model was unavailable have a
    # NULL embedding and are invisible to semantic search. Surfacing the count
    # makes that recoverable instead of a silent quality loss.
    missing_embeddings = 0
    try:
        r = conn.execute(
            "MATCH (m:Memory) WHERE m.embedding IS NULL RETURN COUNT(m);"
        )
        if r.has_next():
            missing_embeddings = r.get_next()[0] or 0
    except Exception as e:
        logger.debug(f"Embedding health check failed: {e}")

    # Probe the actual query path. Counting non-null embedding columns is not
    # sufficient: a stale HNSW index reports full storage while returning
    # nothing to searches.
    # None means "not applicable" (empty DB). Otherwise False covers both a
    # probe that errored and one that returned no rows — from search's point of
    # view those are the same failure.
    vector_index_live: Optional[bool] = None
    if total > 0:
        probe = _probe_vector_index(conn)
        vector_index_live = probe is not None and probe > 0

    # Dream state — useful for hooks deciding whether to trigger consolidation
    now = time.time()
    hours_since_dream = (now - _dream_last_time) / 3600 if _dream_last_time > 0 else None
    dream_due = (
        _dream_ops_since_last >= DREAM_MIN_OPERATIONS
        and (hours_since_dream is None or hours_since_dream >= DREAM_MIN_INTERVAL_HOURS)
        and total >= DREAM_MIN_MEMORIES
    )

    return {
        "total_memories": total,
        "total_relationships": rel_count,
        "total_topics": total_topics,
        "categories": cats,
        "importance": imp_dist,
        "top_topics": topics,
        "top_memories": top_memories,
        "workspace": WORKSPACE,
        "dream": {
            "ops_since": _dream_ops_since_last,
            "hours_since": round(hours_since_dream, 1) if hours_since_dream is not None else None,
            "due": dream_due,
        },
        "runtime": {
            "version": SERVER_VERSION,
            "db_path": DB_PATH,
            "embeddings": {
                "model": EMBEDDING_MODEL,
                "missing": missing_embeddings,
                # Honest health: 'missing == 0' only proves vectors are STORED.
                # It says nothing about whether the index returns them, which
                # is what search actually depends on — so probe the query path.
                "stored_ok": missing_embeddings == 0,
                "index_returns_rows": vector_index_live,
                "healthy": missing_embeddings == 0 and vector_index_live is not False,
                "queryable": vector_index_live,
            },
            "vector_index": {
                **(_vector_index_state or {"status": "unknown"}),
                "repair_attempts": _index_repair_attempts,
            },
            # answering=True means the index matches a term drawn from real
            # content. CAVEAT: this cannot prove BM25 scoring is sound — the
            # observed field failure passed every such probe and only misfired
            # on full-question queries. False here is definitely broken; True
            # is necessary, not sufficient. memory_reindex() rebuilds it.
            "fts_index": {"answering": _probe_fts_index(conn)},
            "fusion_mode": FUSION_MODE,
            "workspace_source": _workspace_source,
            "client": _client_info,
            "client_supports_roots": _client_supports_roots,
            "roots_adoption": {"done": _roots_done, "attempts": _roots_attempts},
        },
    }


def _rebuild_fts_index(conn: lb.Connection) -> dict:
    """Drop and recreate the BM25 index from stored content.

    FTS was the one channel with no rebuild path and no health signal:
    memory_reindex() rebuilt only the HNSW index, so an FTS index carrying bad
    state (e.g. built by an older library version and carried across upgrades)
    was unrepairable — diagnosed in the field as a memory scoring fts 0.0 on a
    full-question query while the same query returned fts 1.0 on a fresh
    ingest of identical content, with the vector channel bit-identical.
    """
    try:
        _safe_execute(conn, "CALL DROP_FTS_INDEX('Memory', 'memory_fts_idx');",
                      expected_errors=("does not exist", "not found",
                                       "doesn't have an index"))
        conn.execute(
            "CALL CREATE_FTS_INDEX('Memory', 'memory_fts_idx', ['content'], "
            "stemmer := 'english');"
        )
    except Exception as e:
        logger.error(f"FTS index rebuild FAILED: {e}")
        return {"rebuilt": False, "error": str(e)[:200]}
    return {"rebuilt": True}


def _probe_fts_index(conn: lb.Connection) -> Optional[bool]:
    """Whether the FTS index answers a term drawn from actual content.

    Uses the first word (>3 chars) of the newest memory, so a healthy index
    must match it. None = no content to probe with.
    """
    try:
        r = conn.execute(
            "MATCH (m:Memory) RETURN m.content ORDER BY m.updated_at DESC LIMIT 1;")
        if not r.has_next():
            return None
        content = r.get_next()[0] or ""
        term = next((w for w in re.findall(r"[A-Za-z]{4,}", content)), None)
        if term is None:
            return None
        res = conn.execute(
            """CALL QUERY_FTS_INDEX('Memory', 'memory_fts_idx', $q, top := 1)
               WITH node AS m, score RETURN m.id;""",
            {"q": term},
        )
        return len(_collect_results(res)) > 0
    except Exception as e:
        logger.debug(f"FTS index probe failed: {e}")
        return False


@mcp.tool()
@_timed("memory_reindex")
def memory_reindex() -> str:
    """Rebuild the search indexes — vector (HNSW) AND full-text (BM25).

    Use when search results include a `degraded` field, when
    memory_stats().runtime.embeddings shows stored_ok true but
    index_returns_rows false, or when a memory scores anomalously low on
    queries it should win (a broken FTS term contributes 0 of the 0.3 keyword
    weight; use memory_search(explain=True) to see per-channel values).

    Safe to run any time: it only rebuilds indexes, never touches memories.
    Cost scales with memory count.
    """
    conn = get_conn()
    total = _count_memories(conn)
    before = _probe_vector_index(conn)

    embedded = 0
    try:
        r = conn.execute("MATCH (m:Memory) WHERE m.embedding IS NOT NULL RETURN COUNT(m);")
        if r.has_next():
            embedded = r.get_next()[0] or 0
    except Exception:
        pass

    if total == 0:
        return {"status": "empty", "message": "No memories to index."}

    # FTS rebuild is independent of embeddings — content is always present.
    fts = _rebuild_fts_index(conn)
    fts_ok = fts.get("rebuilt", False) and bool(_probe_fts_index(conn))

    if embedded == 0:
        return {
            "status": "no_embeddings",
            "memories": total,
            "fts_rebuilt": fts_ok,
            "message": "FTS index rebuilt, but no memory has an embedding, so there "
                       "is no vector index to build. The embedding model was "
                       "unavailable when these were stored; re-store them once "
                       "memory_stats() reports the model healthy.",
        }

    try:
        _safe_execute(conn, "CALL DROP_VECTOR_INDEX('Memory', 'memory_vec_idx');",
                      expected_errors=("does not exist", "not found"))
        conn.execute(
            "CALL CREATE_VECTOR_INDEX('Memory', 'memory_vec_idx', 'embedding', "
            "metric := 'cosine');"
        )
    except Exception as e:
        return {"status": "error", "error": str(e)[:300], "fts_rebuilt": fts_ok}

    after = _probe_vector_index(conn) or 0
    global _vector_index_state
    _vector_index_state = {"status": "rebuilt" if after > 0 else "broken",
                           "rebuilt": True, "indexed": embedded}
    return {
        "status": "rebuilt" if (after > 0 and fts_ok) else "still_broken",
        "memories": total,
        "embeddings_indexed": embedded,
        "fts_rebuilt": fts_ok,
        # The probe is a k=1 synthetic query — it answers "does the index
        # respond at all", not "how many rows are indexed". The old field names
        # (index_rows_before/after) read as row counts and made a healthy
        # rebuild of 38 embeddings look like it had indexed one row.
        "index_answering_before": bool(before),
        "index_answering_after": bool(after),
    }


EXPORT_FORMAT = "memnest-export"
EXPORT_FORMAT_VERSION = 1


@mcp.tool()
@_timed("memory_export")
def memory_export(path: Optional[str] = None, include_embeddings: bool = False,
                  global_export: bool = False) -> str:
    """Write all memories AND their edges to a portable JSON file.

    There was no backup path at all, which is uncomfortable for a store that is
    the single copy of an agent's long-term memory: the database allows one
    writer, index state has been observed to degrade across library upgrades,
    and memory_set_workspace strands the old file rather than moving it.

    path defaults to <db directory>/memnest-export-<timestamp>.json.
    include_embeddings=False keeps the file small; memory_import re-embeds from
    content when they are absent (slower, but model-version independent).
    global_export=True includes every workspace, not just the current one.
    """
    conn = get_conn()

    if path is None:
        base = os.path.dirname(DB_PATH) if DB_PATH != ":memory:" else os.getcwd()
        path = os.path.join(base, f"memnest-export-{int(time.time())}.json")
    path = os.path.abspath(os.path.expanduser(str(path)))

    where = "" if global_export else "WHERE m.workspace IN ['', $ws]"
    params = {} if global_export else {"ws": WORKSPACE}
    cols = ("m.id, m.content, m.category, m.tags, m.importance, m.access_count, "
            "m.created_at, m.updated_at, m.workspace")
    if include_embeddings:
        cols += ", m.embedding"

    try:
        rows = _collect_results(conn.execute(
            f"MATCH (m:Memory) {where} RETURN {cols} ORDER BY m.id;", params))
    except Exception as e:
        return {"status": "error", "message": f"Export query failed: {e}"}

    memories = []
    ids = set()
    for r in rows:
        ids.add(r[0])
        item = {
            "id": r[0], "content": r[1], "category": r[2],
            "tags": _parse_tags(r[3]), "importance": r[4],
            "access_count": r[5] or 0, "created_at": r[6], "updated_at": r[7],
            "workspace": r[8] or "",
        }
        if include_embeddings and len(r) > 9 and r[9] is not None:
            item["embedding"] = list(r[9])
        memories.append(item)

    # Only edges whose BOTH endpoints are in the export, so an import never
    # references a memory that was filtered out by the workspace scope.
    def _edges(query, keys):
        out = []
        try:
            for row in _collect_results(conn.execute(query)):
                if row[0] in ids and row[1] in ids:
                    out.append(dict(zip(keys, row)))
        except Exception as e:
            logger.debug(f"Edge export failed ({keys}): {e}")
        return out

    edges = {
        "related_to": _edges(
            "MATCH (a:Memory)-[r:RELATED_TO]->(b:Memory) "
            "RETURN a.id, b.id, r.provenance, r.confidence;",
            ("from", "to", "provenance", "confidence")),
        "supersedes": _edges(
            "MATCH (a:Memory)-[:SUPERSEDES]->(b:Memory) RETURN a.id, b.id;",
            ("from", "to")),
        "explains": _edges(
            "MATCH (a:Memory)-[r:EXPLAINS]->(b:Memory) "
            "RETURN a.id, b.id, r.rationale_type;",
            ("from", "to", "rationale_type")),
    }

    payload = {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "exported_at": time.time(),
        "server_version": SERVER_VERSION,
        "workspace": "*" if global_export else WORKSPACE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "includes_embeddings": include_embeddings,
        "memories": memories,
        "edges": edges,
    }

    try:
        from pathlib import Path
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)  # atomic: never leaves a partial export
    except Exception as e:
        return {"status": "error", "message": f"Could not write {path}: {e}"}

    return {
        "status": "exported",
        "path": path,
        "memories": len(memories),
        "edges": {k: len(v) for k, v in edges.items()},
        "includes_embeddings": include_embeddings,
        "bytes": os.path.getsize(path) if os.path.exists(path) else None,
    }


@mcp.tool()
@_timed("memory_import")
def memory_import(path: str, dry_run: bool = False) -> str:
    """Restore memories and edges from a memory_export file.

    Ids are REMAPPED, not preserved: the file's ids are matched to whatever ids
    the import produces, so a file can be merged into a database that already
    has memories without collisions. Edges are rewired to the new ids.

    Imported memories go through normal dedup, so re-importing into the same
    database recognises existing content instead of duplicating it — which also
    means an import can be used to merge two memory sets. Embeddings in the file
    are reused when present; otherwise content is re-embedded.

    dry_run=True validates the file and reports what would happen.
    """
    conn = get_conn()
    path = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.isfile(path):
        return {"status": "error", "message": f"No such file: {path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Could not read {path}: {e}"}

    if not isinstance(payload, dict) or payload.get("format") != EXPORT_FORMAT:
        return {"status": "error",
                "message": f"Not a memnest export (expected format={EXPORT_FORMAT!r})."}
    if payload.get("format_version", 0) > EXPORT_FORMAT_VERSION:
        return {"status": "error",
                "message": f"Export format v{payload.get('format_version')} is newer than "
                           f"this server supports (v{EXPORT_FORMAT_VERSION}). Upgrade memnest."}

    memories = payload.get("memories") or []
    if not isinstance(memories, list):
        return {"status": "error", "message": "Malformed export: 'memories' is not a list."}

    file_dim = payload.get("embedding_dim")
    reuse_embeddings = bool(payload.get("includes_embeddings")) and file_dim == EMBEDDING_DIM
    dim_mismatch = (
        payload.get("includes_embeddings") and file_dim not in (None, EMBEDDING_DIM)
    )

    if dry_run:
        return {
            "status": "preview",
            "path": path,
            "memories": len(memories),
            "edges": {k: len(v or []) for k, v in (payload.get("edges") or {}).items()},
            "would_reuse_embeddings": reuse_embeddings,
            "embedding_dim_mismatch": dim_mismatch,
            "note": ("Embeddings in the file have a different dimension and will be "
                     "recomputed from content." if dim_mismatch else None),
        }

    id_map: dict = {}
    imported = 0
    merged = 0
    failed = []

    for item in memories:
        if not isinstance(item, dict):
            continue
        content = _truncate_content(item.get("content") or "")[0]
        if not content.strip():
            continue
        try:
            emb = item.get("embedding") if reuse_embeddings else None
            if emb is not None and len(emb) != EMBEDDING_DIM:
                emb = None
            res = _store_one(
                conn,
                content=content,
                category=item.get("category") or "general",
                tags=_limit_tags(item.get("tags") or []),
                importance=item.get("importance"),
                embedding=emb,
            )
            new_id = res.get("id")
            if new_id is None:
                failed.append({"id": item.get("id"), "reason": res.get("status")})
                continue
            if item.get("id") is not None:
                id_map[item["id"]] = new_id
            if res.get("status") == "stored_new":
                imported += 1
            else:
                merged += 1
        except Exception as e:
            failed.append({"id": item.get("id"), "reason": str(e)[:120]})

    # Rewire edges onto the new ids.
    edge_payload = payload.get("edges") or {}
    edge_counts = {"related_to": 0, "supersedes": 0, "explains": 0}
    edge_skipped = 0
    for kind, rel in (("related_to", "RELATED_TO"), ("supersedes", "SUPERSEDES"),
                      ("explains", "EXPLAINS")):
        for e in (edge_payload.get(kind) or []):
            a, b = id_map.get(e.get("from")), id_map.get(e.get("to"))
            if a is None or b is None or a == b:
                # a == b happens when dedup merged both endpoints into one
                # memory; a self-edge would be meaningless.
                edge_skipped += 1
                continue
            res = _relate_one(conn, a, b, relationship=rel,
                              confidence=e.get("confidence", 1.0),
                              provenance=e.get("provenance", "EXTRACTED"))
            if res.get("status") == "created":
                edge_counts[kind] += 1
            else:
                # "exists" lands here too: re-importing the same file is a no-op
                # for edges rather than doubling them.
                edge_skipped += 1

    _bump_dream_ops()
    out = {
        "status": "imported" if not failed else "partial",
        "path": path,
        "stored_new": imported,
        "merged_into_existing": merged,
        "edges_created": edge_counts,
        "edges_skipped": edge_skipped,
        "reused_embeddings": reuse_embeddings,
    }
    if failed:
        out["failed"] = failed[:20]
        out["failed_count"] = len(failed)
    if dim_mismatch:
        out["note"] = ("Export embeddings had a different dimension; content was "
                       "re-embedded with the current model.")
    return out


@mcp.tool()
@_timed("memory_set_workspace")
def memory_set_workspace(path: str) -> str:
    """Pin the workspace scope (and database location) to a directory.

    Use when auto-detection failed — memory_stats shows workspace '' or a
    path that doesn't match the current project (check runtime.workspace_source
    there). The agent always knows its workspace; the server sometimes can't
    discover it (client without roots support, launched from '/').

    If the database is already open, it is closed and reopened at
    <path>/.memnest/memory.lbug (unless MEMORY_DB_PATH pins a file, in which
    case only the workspace tag changes). Memories stored before the switch
    stay in the previous database file.
    """
    global WORKSPACE, DB_PATH, _workspace_source, _roots_done, _conn, _db

    env_ws = os.environ.get("MEMORY_WORKSPACE", "")
    if env_ws and not _is_bogus_workspace(env_ws):
        return {
            "status": "error",
            "error": f"MEMORY_WORKSPACE is explicitly set to {env_ws!r} in the "
                     f"server config; that always wins. Change the config to move.",
        }

    if _looks_unsubstituted(path):
        return {"status": "error", "error": f"Unexpanded placeholder in path: {path}"}
    p = os.path.abspath(os.path.expanduser(path))
    if _is_bogus_workspace(p):
        return {
            "status": "error",
            "error": f"Refusing {p!r} as a workspace scope: '/', the home "
                     f"directory, and paths under ~/.kiro are never projects",
        }
    if not os.path.isdir(p):
        return {"status": "error", "error": f"Not a directory: {p}"}

    _roots_done = True  # an explicit instruction outranks further adoption
    if p == WORKSPACE:
        return {"status": "unchanged", "workspace": WORKSPACE, "db_path": DB_PATH}

    previous_db = DB_PATH
    WORKSPACE = p
    _workspace_source = "manual"

    # Relocate the database unless it's in-memory (nothing to relocate —
    # closing would discard data) or pinned by MEMORY_DB_PATH (get_conn
    # re-resolves to the same file; only the workspace tag changes).
    db_reopened = False
    if _conn is not None and DB_PATH != ":memory:":
        try:
            _conn.close()
        except Exception as e:
            logger.debug(f"Connection close during workspace switch: {e}")
        try:
            if _db is not None:
                _db.close()
        except Exception as e:
            logger.debug(f"Database close during workspace switch: {e}")
        _conn = None
        _db = None
        get_conn()  # reopen eagerly so failures surface here, not later
        db_reopened = True

    return {
        "status": "ok",
        "workspace": WORKSPACE,
        "db_path": DB_PATH,
        "previous_db_path": previous_db if db_reopened else None,
        "db_reopened": db_reopened,
    }


def _compute_graph_scores(conn: lb.Connection):
    """Pre-compute PageRank and Louvain communities for all Memory nodes.

    Called after ingest/dream to update graph-based importance scores.
    
    1. Creates RELATED_TO edges between memories sharing 3+ meaningful topics
       (inferred relationships for community detection).
    2. Runs PageRank on Memory+Topic graph for importance scoring.
    3. Runs Louvain on Memory-only graph for community detection.
    
    Stores `pagerank` and `community_id` on each Memory node for use in search.
    """
    try:
        # Add columns if not exists
        _safe_execute(conn, "ALTER TABLE Memory ADD pagerank DOUBLE DEFAULT 0.0;",
                      expected_errors=("already exists", "duplicate", "already has property"))
        _safe_execute(conn, "ALTER TABLE Memory ADD community_id INT64 DEFAULT -1;",
                      expected_errors=("already exists", "duplicate", "already has property"))
        _safe_execute(conn, "ALTER TABLE Memory ADD k_degree INT64 DEFAULT 0;",
                      expected_errors=("already exists", "duplicate", "already has property"))

        # --- Step 1: Create inferred RELATED_TO edges from shared topics ---
        # Only create if none exist yet (avoid duplicates on re-run)
        result = conn.execute(
            "MATCH ()-[r:RELATED_TO {provenance: 'INFERRED'}]->() RETURN COUNT(r);"
        )
        existing_inferred = result.get_next()[0] if result.has_next() else 0
        
        if existing_inferred == 0:
            # Create edges between memories sharing 3+ meaningful topics
            # Exclude conversation-level tags (conv-*) and 'chunk' which are on everything
            _safe_execute(conn, """
                MATCH (a:Memory)-[:ABOUT]->(t:Topic)<-[:ABOUT]-(b:Memory)
                WHERE a.id < b.id 
                  AND NOT t.name STARTS WITH 'conv-'
                  AND t.name <> 'chunk'
                WITH a, b, COUNT(DISTINCT t.name) AS shared_topics
                WHERE shared_topics >= 3
                CREATE (a)-[:RELATED_TO {provenance: 'INFERRED', confidence: 0.7}]->(b);
            """)

        # --- Step 2: PageRank on Memory + Topic graph ---
        try:
            conn.execute("CALL DROP_PROJECTED_GRAPH('memory_pr');")
        except Exception:
            pass

        conn.execute(
            "CALL PROJECT_GRAPH('memory_pr', ['Memory', 'Topic'], ['ABOUT', 'RELATED_TO']);"
        )

        result = conn.execute("CALL PAGE_RANK('memory_pr') RETURN node, rank;")
        pr_updates = 0
        for row in _collect_results(result):
            node = row[0]
            rank = row[1]
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id is not None and node.get("_LABEL") == "Memory":
                _safe_execute(conn,
                    "MATCH (m:Memory {id: $id}) SET m.pagerank = $rank;",
                    {"id": node_id, "rank": rank})
                pr_updates += 1

        try:
            conn.execute("CALL DROP_PROJECTED_GRAPH('memory_pr');")
        except Exception:
            pass

        # --- Step 3: Louvain on Memory-only graph ---
        try:
            conn.execute("CALL DROP_PROJECTED_GRAPH('memory_louvain');")
        except Exception:
            pass

        # Check if we have enough Memory→Memory edges for Louvain
        result = conn.execute("MATCH ()-[r:RELATED_TO]->() RETURN COUNT(r);")
        edge_count = result.get_next()[0] if result.has_next() else 0

        louvain_communities = 0
        if edge_count >= 5:
            try:
                conn.execute(
                    "CALL PROJECT_GRAPH('memory_louvain', ['Memory'], ['RELATED_TO']);"
                )
                result = conn.execute(
                    "CALL LOUVAIN('memory_louvain') RETURN node.id, louvain_id;"
                )
                for row in _collect_results(result):
                    mem_id, community = row[0], row[1]
                    if mem_id is not None:
                        _safe_execute(conn,
                            "MATCH (m:Memory {id: $id}) SET m.community_id = $cid;",
                            {"id": mem_id, "cid": community})
                louvain_communities += 1

                conn.execute("CALL DROP_PROJECTED_GRAPH('memory_louvain');")
            except Exception as e:
                logger.debug(f"Louvain failed (non-fatal): {e}")
                try:
                    conn.execute("CALL DROP_PROJECTED_GRAPH('memory_louvain');")
                except Exception:
                    pass

        # --- Step 4: K-Core Decomposition for structural importance ---
        if edge_count >= 5:
            try:
                # Reuse the memory_louvain projection or create fresh
                try:
                    conn.execute("CALL DROP_PROJECTED_GRAPH('memory_kcore');")
                except Exception:
                    pass
                conn.execute(
                    "CALL PROJECT_GRAPH('memory_kcore', ['Memory'], ['RELATED_TO']);"
                )
                result = conn.execute(
                    "CALL K_CORE_DECOMPOSITION('memory_kcore') RETURN node.id, k_degree;"
                )
                for row in _collect_results(result):
                    mem_id, k_deg = row[0], row[1]
                    if mem_id is not None:
                        _safe_execute(conn,
                            "MATCH (m:Memory {id: $id}) SET m.k_degree = $kd;",
                            {"id": mem_id, "kd": k_deg})
                conn.execute("CALL DROP_PROJECTED_GRAPH('memory_kcore');")
            except Exception as e:
                logger.debug(f"K-Core failed (non-fatal): {e}")
                try:
                    conn.execute("CALL DROP_PROJECTED_GRAPH('memory_kcore');")
                except Exception:
                    pass

        logger.info(f"PageRank computed for {pr_updates} memories, "
                    f"Louvain communities detected, {edge_count} edges")
    except Exception as e:
        logger.debug(f"Graph score computation failed (non-fatal): {e}")


# Module-level lock for dream concurrency
_dream_lock = threading.Lock()


@mcp.tool()
@_timed("memory_dream")
def memory_dream(force: bool = False, dry_run: bool = False) -> str:
    """Periodic memory consolidation. Auto-prunes stale (30+ days, importance<=2),
    auto-merges trivial duplicates (sim>=0.95), surfaces clusters at 0.88-0.95 for agent review.
    Triggers on 10+ ops + 24h elapsed. force=True overrides; dry_run=True previews.
    """
    global _dream_ops_since_last, _dream_last_time

    # Idempotency guard: refuse concurrent dreams
    if not _dream_lock.acquire(blocking=False):
        return {"status": "in_progress"}

    try:
        conn = get_conn()
        now = time.time()

        hours_since_last = (now - _dream_last_time) / 3600 if _dream_last_time > 0 else float("inf")
        ops = _dream_ops_since_last

        # Skip if memory count is too low to be useful
        memories_before = _count_memories(conn)
        if memories_before < DREAM_MIN_MEMORIES and not force:
            return {
                "status": "not_needed",
                "reason": "memory_count_too_low",
                "memory_count": memories_before,
            }

        # Threshold check (skipped if force=True)
        dream_needed = force or (ops >= DREAM_MIN_OPERATIONS and hours_since_last >= DREAM_MIN_INTERVAL_HOURS)

        if not dream_needed:
            return {
                "status": "not_needed",
                "ops_since_last": ops,
                "hours_since_last": round(hours_since_last, 1),
            }

        actions_taken = []
        # Pairs left alone because a SUPERSEDES/EXPLAINS edge marks them as
        # distinct versions rather than duplicates.
        protected_pairs = 0
        # Pairs left alone because their tags indicate different subjects.
        distinct_subject_skips = 0
        # Pairs left alone because they are the same subject but disagree on a
        # value — contradictions, which the subject gate cannot detect.
        value_conflict_skips = 0
        # HNSW self-recall audit: memories the index failed to return for their
        # own embedding (partial recall degradation — invisible to the
        # zero-rows check), and whether that triggered a rebuild.
        index_self_misses = 0
        index_rebuilt_by_audit = False

        # Phase 1: Auto-prune stale low-importance memories
        prune_cutoff = now - (DREAM_AUTO_PRUNE_DAYS * 86400)
        try:
            result = conn.execute(
                """MATCH (m:Memory)
                   WHERE m.updated_at < $cutoff AND m.importance <= $max_imp AND m.access_count <= 1
                   RETURN m.id, m.content;""",
                {"cutoff": prune_cutoff, "max_imp": DREAM_AUTO_PRUNE_MAX_IMPORTANCE},
            )
            pruned_ids = []
            for row in _collect_results(result):
                pruned_ids.append(row[0])
                if not dry_run:
                    conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": row[0]})
            if pruned_ids:
                actions_taken.append({
                    "action": "pruned_stale" + ("_preview" if dry_run else ""),
                    "count": len(pruned_ids),
                    "ids": pruned_ids,
                })
        except Exception as e:
            logger.warning(f"Dream prune failed: {e}")

        # Phase 2: Auto-merge trivial duplicates (similarity > 0.95)
        total = _count_memories(conn)
        trivial_merged = []
        if total >= 2:
            try:
                scan_result = conn.execute(
                    """MATCH (m:Memory)
                       RETURN m.id, m.content, m.tags, m.importance, m.embedding,
                              m.created_at, m.category, m.workspace
                       ORDER BY m.updated_at DESC LIMIT $limit;""",
                    {"limit": MAX_CONSOLIDATE_SCAN},
                )
                all_mems = _collect_results(scan_result)
                visited = set()

                for mem in all_mems:
                    mid = mem[0]
                    content = mem[1]
                    tags = mem[2]
                    importance = mem[3]
                    embedding = mem[4]
                    created_at = mem[5]
                    category = mem[6] or "general"
                    mem_ws = mem[7] or WORKSPACE

                    if mid in visited or embedding is None:
                        continue

                    def _merge_probe(emb_):
                        """Neighbours of one memory. None = the probe itself
                        failed (a MISSING index raises rather than returning
                        empty), which the audit treats the same as a miss."""
                        try:
                            r_ = conn.execute(
                                """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                                   WITH node AS m, distance
                                   RETURN m.id, m.content, m.tags, m.importance, m.created_at,
                                          m.category, m.workspace, distance;""",
                                {"query": list(emb_), "k": 4},
                            )
                            return _collect_results(r_)
                        except Exception as e_:
                            logger.debug(f"Dream merge probe failed: {e_}")
                            return None

                    probe_rows = _merge_probe(embedding)

                    # Self-recall audit. A healthy HNSW index queried with a
                    # memory's OWN embedding must return that memory (distance
                    # ~0 beats every other vector). A miss means the index has
                    # PARTIAL recall degradation: it still returns rows, so the
                    # zero-rows self-heal in search/dedup never fires, yet some
                    # memories are unreachable from some query points. Observed
                    # on a long-lived DB: a memory absent from results for a
                    # query it answered at 0.74, while still reachable via its
                    # own wording — and dedup/conflict-protection silently
                    # blind to the missing partner. Dream probes every memory
                    # anyway, so the audit is free; rebuild once and rescan.
                    if not dry_run and (
                        probe_rows is None or mid not in {r[0] for r in probe_rows}
                    ):
                        index_self_misses += 1
                        if not index_rebuilt_by_audit:
                            logger.warning(
                                f"HNSW self-recall miss: memory {mid} not returned "
                                f"for its own embedding. Rebuilding index..."
                            )
                            state = _ensure_vector_index(conn, force_rebuild=True)
                            index_rebuilt_by_audit = bool(state.get("rebuilt"))
                            if index_rebuilt_by_audit:
                                probe_rows = _merge_probe(embedding)

                    for row in (probe_rows or []):
                        other_id = row[0]
                        other_content = row[1]
                        other_tags = row[2]
                        other_imp = row[3]
                        other_created = row[4]
                        other_cat = row[5] or "general"
                        other_ws = row[6] or WORKSPACE
                        dist = row[7]
                        sim = 1.0 - dist

                        if other_id == mid or other_id in visited:
                            continue
                        # Don't merge across workspaces
                        if mem_ws != other_ws:
                            continue
                        if sim < DREAM_TRIVIAL_MERGE_THRESHOLD:
                            continue
                        # Never merge across a SUPERSEDES/EXPLAINS edge: those
                        # assert the memories are distinct versions, and a
                        # correction looks like a duplicate by construction.
                        link = _semantically_linked(conn, mid, other_id)
                        if link:
                            logger.info(
                                f"Skipping merge of {mid}/{other_id} (sim {sim:.4f}): "
                                f"linked by {link}"
                            )
                            protected_pairs += 1
                            continue
                        # Auto-merge is the only unreviewed destructive path, so
                        # it gets the same subject gate as everything else.
                        if not _same_subject(_parse_tags(tags), _parse_tags(other_tags)):
                            logger.info(
                                f"Skipping merge of {mid}/{other_id} (sim {sim:.4f}): "
                                f"different subject"
                            )
                            distinct_subject_skips += 1
                            continue
                        # Same subject but disagreeing values: a contradiction,
                        # not a duplicate. This is the case the subject gate
                        # cannot catch, because a contradiction is same-subject
                        # by construction.
                        if _values_conflict(content, other_content):
                            _v = _name_vocab(content, other_content)
                            logger.info(
                                f"Skipping merge of {mid}/{other_id} (sim {sim:.4f}): "
                                f"conflicting values "
                                f"{sorted(_discriminating_tokens(content, _v) ^ _discriminating_tokens(other_content, _v))}"
                            )
                            value_conflict_skips += 1
                            continue

                        # Keep the longer content, absorb tags, take max importance
                        if len(content or "") >= len(other_content or ""):
                            keep_id, drop_id = mid, other_id
                            keep_content = content
                            keep_created = created_at
                            keep_category = category
                            keep_workspace = mem_ws
                            drop_created = other_created
                        else:
                            keep_id, drop_id = other_id, mid
                            keep_content = other_content
                            keep_created = other_created
                            keep_category = other_cat
                            keep_workspace = other_ws
                            drop_created = created_at

                        merged_tags = list(set(_parse_tags(tags) + _parse_tags(other_tags)))
                        new_imp = min(5, max(importance or 3, other_imp or 3))
                        # Preserve the older created_at
                        preserved_created = min(keep_created or now, drop_created or now)

                        if not dry_run:
                            # Save BOTH memories' relationships, then merge them
                            saved_keep = _save_memory_relationships(conn, keep_id)
                            saved_drop = _save_memory_relationships(conn, drop_id)

                            # Combine the keeper's and dropped node's edges, then
                            # collapse duplicates so we never end up with two
                            # parallel edges of the same kind between the same
                            # endpoints (which would silently inflate the
                            # relationship count and confuse traversal queries).
                            #
                            # For RELATED_TO we key on the other endpoint and
                            # keep the row with the higher confidence; for
                            # SUPERSEDES/EXPLAINS we key on the other endpoint.
                            def _dedupe_related(rows):
                                by_endpoint: dict = {}
                                for r in rows:
                                    if not r:
                                        continue
                                    other = r[0]
                                    if other == keep_id or other == drop_id:
                                        continue
                                    prev = by_endpoint.get(other)
                                    if prev is None or (r[2] or 0) > (prev[2] or 0):
                                        by_endpoint[other] = r
                                return list(by_endpoint.values())

                            def _dedupe_simple(rows, key_indices=(0,)):
                                seen = {}
                                for r in rows:
                                    if not r:
                                        continue
                                    other = r[0]
                                    if other == keep_id or other == drop_id:
                                        continue
                                    k = tuple(r[i] if i < len(r) else None for i in key_indices)
                                    seen[k] = r
                                return list(seen.values())

                            combined = {
                                "rels_out": _dedupe_related(saved_keep["rels_out"] + saved_drop["rels_out"]),
                                "rels_in": _dedupe_related(saved_keep["rels_in"] + saved_drop["rels_in"]),
                                "sup_out": _dedupe_simple(saved_keep["sup_out"] + saved_drop["sup_out"]),
                                "sup_in": _dedupe_simple(saved_keep["sup_in"] + saved_drop["sup_in"]),
                                # EXPLAINS rows carry rationale_type at index 1 — key on (endpoint, rationale).
                                "exp_out": _dedupe_simple(saved_keep["exp_out"] + saved_drop["exp_out"], (0, 1)),
                                "exp_in": _dedupe_simple(saved_keep["exp_in"] + saved_drop["exp_in"], (0, 1)),
                            }

                            new_emb = _embed(keep_content)
                            if new_emb is None:
                                logger.warning(f"Dream merge skipped: embedding failed for memory {keep_id}")
                                continue
                            conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": keep_id})
                            conn.execute(
                                """CREATE (m:Memory {
                                       id: $id, content: $content, content_hash: $hash,
                                       category: $cat, tags: $tags, workspace: $ws,
                                       importance: $imp, access_count: 0,
                                       created_at: $created, updated_at: $now, embedding: $emb
                                   });""",
                                {"id": keep_id, "content": keep_content, "hash": _content_hash(keep_content),
                                 "cat": keep_category, "tags": _format_tags(merged_tags),
                                 "ws": keep_workspace, "imp": new_imp,
                                 "created": preserved_created, "now": now, "emb": new_emb},
                            )
                            _ensure_topics(conn, keep_id, merged_tags)
                            conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": drop_id})
                            _restore_memory_relationships(conn, keep_id, combined)

                        trivial_merged.append({"kept": keep_id, "dropped": drop_id, "similarity": round(sim, 4)})
                        visited.add(mid)
                        visited.add(other_id)
                        visited.add(keep_id)  # ensure keeper not re-processed (issue #16)
                        break
            except Exception as e:
                logger.warning(f"Dream trivial merge failed: {e}")

        if trivial_merged:
            actions_taken.append({
                "action": "trivial_merge" + ("_preview" if dry_run else ""),
                "count": len(trivial_merged),
                "merges": trivial_merged,
            })

        # Phase 3: Find clusters needing agent reasoning. Window is exclusive
        # of the trivial-merge threshold and bounded below by DREAM_CLUSTER_LOW_THRESHOLD.
        clusters = []
        if total >= 2:
            try:
                scan_result = conn.execute(
                    """MATCH (m:Memory)
                       RETURN m.id, m.content, m.importance, m.embedding, m.workspace
                       ORDER BY m.updated_at DESC LIMIT $limit;""",
                    {"limit": MAX_CONSOLIDATE_SCAN},
                )
                all_mems = _collect_results(scan_result)
                visited_clusters = set()

                for mem in all_mems:
                    mid, content, importance, embedding, anchor_ws = (
                        mem[0], mem[1], mem[2], mem[3], (mem[4] or WORKSPACE),
                    )
                    if mid in visited_clusters or embedding is None:
                        continue

                    result = conn.execute(
                        """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                           WITH node AS m, distance
                           RETURN m.id, m.content, distance, m.workspace;""",
                        {"query": list(embedding), "k": 6},
                    )
                    cluster_members = []
                    for row in _collect_results(result):
                        sim = round(1.0 - row[2], 4)
                        candidate_ws = row[3] or WORKSPACE
                        # Skip cross-workspace look-alikes — surfacing them would
                        # invite the agent to merge unrelated projects' data.
                        if candidate_ws != anchor_ws:
                            continue
                        if (
                            row[0] != mid
                            and row[0] not in visited_clusters
                            and DREAM_CLUSTER_LOW_THRESHOLD <= sim < DREAM_TRIVIAL_MERGE_THRESHOLD
                        ):
                            # Already-linked pairs are resolved history, not
                            # candidates for review — surfacing them invites an
                            # agent to merge a correction chain away.
                            if _semantically_linked(conn, mid, row[0]):
                                protected_pairs += 1
                                visited_clusters.add(row[0])
                                continue
                            cluster_members.append({"id": row[0], "preview": _truncate(row[1], 100), "similarity": sim})
                            visited_clusters.add(row[0])

                    if cluster_members:
                        visited_clusters.add(mid)
                        clusters.append({
                            "anchor": {"id": mid, "preview": _truncate(content, 100), "importance": importance},
                            "similar": cluster_members,
                            # None of these are edge-linked (linked pairs were
                            # filtered above), so each is one of: a true
                            # duplicate to merge, OR competing versions that
                            # need a SUPERSEDES edge, OR distinct facts that
                            # merely read alike. The agent must decide which.
                            "resolution": "merge_duplicate | link_with_supersedes | leave_separate",
                        })
                        if len(clusters) >= MAX_CONSOLIDATE_CLUSTERS:
                            break
            except Exception:
                pass

        # Mark dream complete (skip on dry-run)
        if not dry_run:
            with _dream_ops_lock:
                _dream_ops_since_last = 0
                _dream_last_time = now
                try:
                    _persist_dream_state()
                except Exception:
                    pass

        memories_after = _count_memories(conn)
        pruned_count = sum(a["count"] for a in actions_taken if "pruned" in a["action"])
        merged_count = sum(a["count"] for a in actions_taken if "merge" in a["action"])

        # Re-compute graph scores after consolidation changes the graph structure
        if not dry_run and memories_after > 0:
            _compute_graph_scores(conn)

        # Routine FTS rebuild. Unlike the HNSW audit above, bad FTS state has
        # no cheap detector: the observed failure (a memory scoring fts 0.0 on
        # a full-question query while rare-term and exact-text probes all
        # return 1.0) only shows on query shapes the server cannot enumerate,
        # so a self-recall audit would pass on a broken index. Rebuilding
        # unconditionally is bounded (content-sized, dream runs at most daily)
        # and turns a permanently wedged index into one fixed at next dream.
        fts_rebuilt = False
        if not dry_run and memories_after > 0:
            fts_rebuilt = _rebuild_fts_index(conn).get("rebuilt", False)

        # --- Phase 4: SCC contradiction detection on SUPERSEDES subgraph ---
        contradictions = []
        if not dry_run:
            try:
                # Check if we have SUPERSEDES edges
                sup_result = conn.execute(
                    "MATCH ()-[r:SUPERSEDES]->() RETURN COUNT(r);"
                )
                sup_count = sup_result.get_next()[0] if sup_result.has_next() else 0

                if sup_count >= 2:
                    try:
                        conn.execute("CALL DROP_PROJECTED_GRAPH('memory_scc');")
                    except Exception:
                        pass
                    conn.execute(
                        "CALL PROJECT_GRAPH('memory_scc', ['Memory'], ['SUPERSEDES']);"
                    )
                    result = conn.execute(
                        "CALL STRONGLY_CONNECTED_COMPONENTS('memory_scc') "
                        "RETURN group_id, collect(node.id) AS members;"
                    )
                    for row in _collect_results(result):
                        group_id, members = row[0], row[1]
                        if isinstance(members, list) and len(members) > 1:
                            # SCC with >1 node = contradiction cycle
                            contradictions.append({
                                "group_id": group_id,
                                "memory_ids": members,
                                "issue": "circular SUPERSEDES — these memories contradict each other",
                            })
                    try:
                        conn.execute("CALL DROP_PROJECTED_GRAPH('memory_scc');")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"SCC contradiction check failed (non-fatal): {e}")

        return {
            "status": "preview" if dry_run else "completed",
            "pruned": pruned_count,
            "auto_merged": merged_count,
            "protected_by_edges": protected_pairs,
            "protected_by_subject": distinct_subject_skips,
            "protected_by_value_conflict": value_conflict_skips,
            "index_self_misses": index_self_misses,
            "index_rebuilt": index_rebuilt_by_audit,
            "fts_rebuilt": fts_rebuilt,
            "memories_after": memories_after,
            "clusters_for_review": clusters,
            "contradictions": contradictions,
        }
    finally:
        _dream_lock.release()


@mcp.tool()
@_timed("memory_graph_html")
def memory_graph_html(open_browser: bool = True) -> str:
    """Generate an interactive HTML visualization of the memory graph (vis.js).

    Refuses to render graphs larger than MEMORY_GRAPH_MAX_NODES (default 2000).
    Writes to a fixed `memory_graph.html` plus rotates the 5 most recent snapshots.
    Memory content is rendered safely (HTML-escaped tooltips, DOM textContent in
    the detail panel) so untrusted memory text cannot execute as HTML.
    """
    conn = get_conn()

    # Scale guard. vis.js becomes unusable on large graphs and the file balloons.
    total_nodes_check = _count_memories(conn)
    try:
        topic_check = conn.execute("MATCH (t:Topic) RETURN COUNT(t);")
        if topic_check.has_next():
            total_nodes_check += topic_check.get_next()[0] or 0
    except Exception:
        pass
    if total_nodes_check > GRAPH_HTML_MAX_NODES:
        return {
            "status": "too_large",
            "node_count": total_nodes_check,
            "limit": GRAPH_HTML_MAX_NODES,
            "message": (
                f"Graph has {total_nodes_check} nodes; refusing to render "
                f">{GRAPH_HTML_MAX_NODES}. Set MEMORY_GRAPH_MAX_NODES higher to "
                f"override, or filter the dataset first."
            ),
        }

    # --- Gather all graph data ---
    mem_rows = _collect_results(conn.execute(
        """MATCH (m:Memory)
           RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count
           ORDER BY m.id;"""
    ))
    topic_rows = _collect_results(conn.execute(
        "MATCH (t:Topic) RETURN t.name;"
    ))
    about_rows = _collect_results(conn.execute(
        "MATCH (m:Memory)-[:ABOUT]->(t:Topic) RETURN m.id, t.name;"
    ))
    related_rows = _collect_results(conn.execute(
        "MATCH (a:Memory)-[:RELATED_TO]->(b:Memory) RETURN a.id, b.id;"
    ))
    supersedes_rows = _collect_results(conn.execute(
        "MATCH (a:Memory)-[:SUPERSEDES]->(b:Memory) RETURN a.id, b.id;"
    ))

    # --- Build vis.js data ---
    nodes_js = []
    edges_js = []
    category_colors = {
        "learning": "#4FC3F7",
        "preference": "#AED581",
        "decision": "#FFB74D",
        "pattern": "#CE93D8",
        "general": "#90A4AE",
    }

    # SECURITY: vis.js renders `title` as HTML; the JS detail panel uses
    # textContent through buildField(). Tooltip strings are pre-escaped with
    # html.escape so untrusted memory content cannot break out of HTML context.
    def _h(value: Any) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    for row in mem_rows:
        mid, content, category, tags, importance, access_count = row
        color = category_colors.get(category, "#90A4AE")
        size = 15 + (importance or 3) * 5
        label = f"M{mid}"
        content_str = content or ""
        title_text = content_str[:200] + ("..." if len(content_str) > 200 else "")
        parsed_tags = _parse_tags(tags)
        tag_str = ", ".join(parsed_tags) if parsed_tags else "none"

        tooltip = (
            f"<b>Memory #{mid}</b> [{_h(category)}]<br>"
            f"Importance: {_h(importance)} | Accessed: {_h(access_count)}<br>"
            f"Tags: {_h(tag_str)}<br><br>{_h(title_text)}"
        )

        nodes_js.append({
            "id": f"m{mid}",
            "label": label,
            "title": tooltip,
            "color": {"background": color, "border": "#333", "highlight": {"background": "#FFF176", "border": "#333"}},
            "size": size,
            "shape": "dot",
            "font": {"size": 12, "color": "#333"},
            "nodeType": "memory",
            # Raw values — the JS uses textContent (DOM safe). Do NOT inject as HTML.
            "fullContent": content_str,
            "category": category or "",
            "importance": importance or 0,
            "tags": tag_str,
        })

    for row in topic_rows:
        name = row[0] or ""
        nodes_js.append({
            "id": f"t_{name}",
            "label": name,
            "title": f"<b>Topic:</b> {_h(name)}",
            "color": {"background": "#FFF176", "border": "#F9A825", "highlight": {"background": "#FFEE58", "border": "#F57F17"}},
            "size": 12,
            "shape": "diamond",
            "font": {"size": 11, "color": "#555"},
            "nodeType": "topic",
        })

    for row in about_rows:
        edges_js.append({
            "from": f"m{row[0]}",
            "to": f"t_{row[1]}",
            "color": {"color": "#FBC02D", "opacity": 0.6},
            "width": 1,
            "dashes": True,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "title": "ABOUT",
        })
    for row in related_rows:
        edges_js.append({
            "from": f"m{row[0]}",
            "to": f"m{row[1]}",
            "color": {"color": "#42A5F5", "opacity": 0.8},
            "width": 2,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "title": "RELATED_TO",
        })
    for row in supersedes_rows:
        edges_js.append({
            "from": f"m{row[0]}",
            "to": f"m{row[1]}",
            "color": {"color": "#EF5350", "opacity": 0.8},
            "width": 2,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "title": "SUPERSEDES",
        })

    # JSON-encode safely. ensure_ascii avoids any literal `</script>` slipping
    # through string values; we also break `</` defensively.
    def _safe_json(obj):
        s = json.dumps(obj, ensure_ascii=True, default=str)
        return s.replace("</", "<\\/")

    nodes_json = _safe_json(nodes_js)
    edges_json = _safe_json(edges_js)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memnest Memory Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }}
  #header {{ background: #16213e; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #0f3460; }}
  #header h1 {{ font-size: 18px; color: #e94560; }}
  #header .stats {{ font-size: 13px; color: #aaa; }}
  #controls {{ background: #16213e; padding: 10px 24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #0f3460; }}
  #controls label {{ font-size: 12px; color: #aaa; }}
  #controls select, #controls input {{ background: #1a1a2e; color: #eee; border: 1px solid #0f3460; border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
  #controls button {{ background: #e94560; color: #fff; border: none; border-radius: 4px; padding: 5px 14px; cursor: pointer; font-size: 12px; }}
  #controls button:hover {{ background: #c81e45; }}
  #main {{ display: flex; height: calc(100vh - 95px); }}
  #graph {{ flex: 1; }}
  #detail {{ width: 340px; background: #16213e; border-left: 1px solid #0f3460; padding: 16px; overflow-y: auto; display: none; }}
  #detail h3 {{ color: #e94560; margin-bottom: 8px; font-size: 14px; }}
  #detail .field {{ margin-bottom: 10px; }}
  #detail .field-label {{ font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 2px; }}
  #detail .field-value {{ font-size: 13px; line-height: 1.5; word-wrap: break-word; white-space: pre-wrap; }}
  #detail .close-btn {{ float: right; cursor: pointer; color: #888; font-size: 18px; }}
  #detail .close-btn:hover {{ color: #e94560; }}
  .legend {{ display: flex; gap: 16px; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #aaa; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .legend-diamond {{ width: 10px; height: 10px; transform: rotate(45deg); }}
</style>
</head>
<body>
<div id="header">
  <h1>Memnest Memory Graph</h1>
  <div class="stats" id="statsBar"></div>
</div>
<div id="controls">
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#4FC3F7"></div>Learning</div>
    <div class="legend-item"><div class="legend-dot" style="background:#AED581"></div>Preference</div>
    <div class="legend-item"><div class="legend-dot" style="background:#FFB74D"></div>Decision</div>
    <div class="legend-item"><div class="legend-dot" style="background:#CE93D8"></div>Pattern</div>
    <div class="legend-item"><div class="legend-dot" style="background:#90A4AE"></div>General</div>
    <div class="legend-item"><div class="legend-diamond" style="background:#FFF176"></div>Topic</div>
  </div>
  <label>Filter: <select id="filterCategory">
    <option value="all">All Categories</option>
    <option value="learning">Learning</option>
    <option value="preference">Preference</option>
    <option value="decision">Decision</option>
    <option value="pattern">Pattern</option>
    <option value="general">General</option>
  </select></label>
  <label>Min Importance: <input type="range" id="filterImportance" min="1" max="5" value="1" style="width:80px"><span id="impVal">1</span></label>
  <button onclick="resetView()">Reset View</button>
</div>
<div id="main">
  <div id="graph"></div>
  <div id="detail">
    <span class="close-btn" onclick="closeDetail()">&times;</span>
    <div id="detailContent"></div>
  </div>
</div>
<script>
const allNodes = new vis.DataSet({nodes_json});
const allEdges = new vis.DataSet({edges_json});

const container = document.getElementById('graph');
const options = {{
  physics: {{
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{ gravitationalConstant: -40, centralGravity: 0.005, springLength: 120, springConstant: 0.04, damping: 0.4 }},
    stabilization: {{ iterations: 150 }},
  }},
  interaction: {{ hover: true, tooltipDelay: 150, zoomView: true, dragView: true }},
  edges: {{ smooth: {{ type: 'continuous' }} }},
}};

const network = new vis.Network(container, {{ nodes: allNodes, edges: allEdges }}, options);

const memCount = allNodes.get({{ filter: n => n.nodeType === 'memory' }}).length;
const topicCount = allNodes.get({{ filter: n => n.nodeType === 'topic' }}).length;
document.getElementById('statsBar').textContent = memCount + ' memories \u00b7 ' + topicCount + ' topics \u00b7 ' + allEdges.length + ' relationships';

// Build a label/value field as DOM nodes (textContent — no HTML parsing).
function buildField(label, value) {{
  const wrap = document.createElement('div'); wrap.className = 'field';
  const l = document.createElement('div'); l.className = 'field-label'; l.textContent = label;
  const v = document.createElement('div'); v.className = 'field-value'; v.textContent = value == null ? '' : String(value);
  wrap.appendChild(l); wrap.appendChild(v);
  return wrap;
}}

network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    const nodeId = params.nodes[0];
    const node = allNodes.get(nodeId);
    const dc = document.getElementById('detailContent');
    dc.replaceChildren();
    if (node && node.nodeType === 'memory') {{
      const h = document.createElement('h3'); h.textContent = 'Memory #' + node.label.replace('M','');
      dc.appendChild(h);
      dc.appendChild(buildField('Category', node.category));
      const stars = '\u2605'.repeat(node.importance) + '\u2606'.repeat(Math.max(0, 5 - node.importance));
      dc.appendChild(buildField('Importance', stars + '  (' + node.importance + '/5)'));
      dc.appendChild(buildField('Tags', node.tags));
      dc.appendChild(buildField('Content', node.fullContent));
      document.getElementById('detail').style.display = 'block';
    }} else if (node && node.nodeType === 'topic') {{
      const h = document.createElement('h3'); h.textContent = 'Topic: ' + node.label;
      dc.appendChild(h);
      const conn = allEdges.get({{ filter: e => e.to === nodeId }})
        .map(e => e.from.replace('m','#')).join(', ') || 'none';
      dc.appendChild(buildField('Connected Memories', conn));
      document.getElementById('detail').style.display = 'block';
    }}
  }}
}});

function closeDetail() {{ document.getElementById('detail').style.display = 'none'; }}

document.getElementById('filterCategory').addEventListener('change', applyFilters);
document.getElementById('filterImportance').addEventListener('input', function() {{
  document.getElementById('impVal').textContent = this.value;
  applyFilters();
}});

function applyFilters() {{
  const cat = document.getElementById('filterCategory').value;
  const minImp = parseInt(document.getElementById('filterImportance').value);
  const visibleMemIds = new Set();
  allNodes.forEach(function(node) {{
    if (node.nodeType === 'memory') {{
      const show = (cat === 'all' || node.category === cat) && (node.importance >= minImp);
      allNodes.update({{ id: node.id, hidden: !show }});
      if (show) visibleMemIds.add(node.id);
    }}
  }});
  const connectedTopics = new Set();
  allEdges.forEach(function(edge) {{
    if (visibleMemIds.has(edge.from) && edge.to.startsWith('t_')) connectedTopics.add(edge.to);
  }});
  allNodes.forEach(function(node) {{
    if (node.nodeType === 'topic') allNodes.update({{ id: node.id, hidden: !connectedTopics.has(node.id) }});
  }});
  allEdges.forEach(function(edge) {{
    const fromVis = visibleMemIds.has(edge.from) || connectedTopics.has(edge.from);
    const toVis = visibleMemIds.has(edge.to) || connectedTopics.has(edge.to);
    allEdges.update({{ id: edge.id, hidden: !(fromVis && toVis) }});
  }});
}}

function resetView() {{
  document.getElementById('filterCategory').value = 'all';
  document.getElementById('filterImportance').value = 1;
  document.getElementById('impVal').textContent = '1';
  allNodes.forEach(n => allNodes.update({{ id: n.id, hidden: false }}));
  allEdges.forEach(e => allEdges.update({{ id: e.id, hidden: false }}));
  network.fit();
  closeDetail();
}}
</script>
</body>
</html>"""

    if DB_PATH == ":memory:":
        out_dir = os.path.join(tempfile.gettempdir(), "memnest", "graph")
    else:
        out_dir = os.path.join(os.path.dirname(DB_PATH), "graph")
    os.makedirs(out_dir, exist_ok=True)

    # Stable "latest" path users can bookmark, plus a rotating snapshot.
    latest_path = os.path.join(out_dir, "memory_graph.html")
    timestamp = int(time.time() * 1000)
    snapshot_path = os.path.join(out_dir, f"memory_graph_{timestamp}.html")

    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    # Retention: keep 5 most recent snapshots, prune older ones.
    try:
        snapshots = sorted(
            (p for p in Path(out_dir).glob("memory_graph_*.html")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in snapshots[5:]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.debug(f"Graph HTML rotation failed: {e}")

    browser_opened = False
    if open_browser:
        try:
            browser_opened = bool(webbrowser.open(f"file://{os.path.abspath(latest_path)}"))
        except Exception as e:
            logger.warning(f"webbrowser.open failed: {e}")
            browser_opened = False

    return {
        "status": "generated",
        "path": latest_path,
        "snapshot": snapshot_path,
        "browser_opened": browser_opened,
        "node_count": total_nodes_check,
    }


# ----------------------------------------------------------------------------
# Single-memory read and non-ranked listing.
#
# These began as pre-0.2.0 compatibility aliases and carried a note saying they
# would be removed in 0.3.0. That note was wrong by the time it mattered: two of
# them do things no other tool does, and the project is well past 0.3.0.
#
#   memory_get   — the only way to read ONE memory in full. Search truncates to
#                  a preview, and only memory_get returns the edge block.
#   memory_list  — the only ordered, offset-paged, relevance-free enumeration.
#                  memory_search ranks; sometimes you want "newest 20".
#
# memory_traverse is genuinely redundant: it is memory_query(read_only=True)
# under another name. It stays only because published hooks and steering files
# reference it, and it is now marked deprecated in its own docstring rather than
# by a stale comment here.
# ----------------------------------------------------------------------------

@mcp.tool()
@_timed("memory_get")
def memory_get(memory_id: int, include_edges: bool = True) -> str:
    """Get one memory in full — untruncated content, metadata, and its edges.

    This is the right tool for "show me memory 42 completely". Search returns
    truncated previews; this does not.

    include_edges=True (the default) adds an `edges` block listing the memory's
    RELATED_TO / SUPERSEDES / EXPLAINS links in both directions, plus a
    `superseded_by` convenience field. Without it, answering "what does this
    replace?" or "is this still current?" required hand-written Cypher, even
    though edges are the whole point of storing memory as a graph.
    """
    conn = get_conn()
    if not isinstance(memory_id, int) or isinstance(memory_id, bool):
        return {"status": "error", "message": "memory_id must be an integer."}

    result = conn.execute(
        """MATCH (m:Memory {id: $id})
           RETURN m.id, m.content, m.category, m.tags, m.importance,
                  m.access_count, m.created_at, m.updated_at, m.workspace;""",
        {"id": memory_id},
    )
    if not result.has_next():
        return {"status": "not_found", "id": memory_id}

    row = result.get_next()
    # Bump access_count so retrieval is reflected in stats.
    _safe_execute(
        conn,
        "MATCH (m:Memory {id: $id}) SET m.access_count = m.access_count + 1;",
        {"id": memory_id},
    )

    out = {
        "status": "found",
        "id": row[0],
        "content": row[1],
        "category": row[2],
        "tags": _parse_tags(row[3]),
        "importance": row[4],
        "access_count": (row[5] or 0) + 1,
        "created_at": row[6],
        "updated_at": row[7],
        "workspace": row[8] or "",
    }

    if include_edges:
        try:
            # _save_memory_relationships already queries all six directions for
            # the delete+recreate paths; reuse it rather than a seventh copy.
            saved = _save_memory_relationships(conn, memory_id)
            edges: dict = {}

            def _ids(rows):
                return [r[0] for r in rows if r]

            if saved["rels_out"] or saved["rels_in"]:
                edges["related_to"] = {
                    "out": [{"id": r[0], "provenance": r[1], "confidence": r[2]}
                            for r in saved["rels_out"] if r],
                    "in": [{"id": r[0], "provenance": r[1], "confidence": r[2]}
                           for r in saved["rels_in"] if r],
                }
            if saved["sup_out"] or saved["sup_in"]:
                edges["supersedes"] = _ids(saved["sup_out"])
                edges["superseded_by"] = _ids(saved["sup_in"])
            if saved["exp_out"] or saved["exp_in"]:
                edges["explains"] = [
                    {"id": r[0], "rationale_type": (r[1] if len(r) > 1 else None)}
                    for r in saved["exp_out"] if r
                ]
                edges["explained_by"] = [
                    {"id": r[0], "rationale_type": (r[1] if len(r) > 1 else None)}
                    for r in saved["exp_in"] if r
                ]

            out["edges"] = edges
            # Promoted to the top level because it changes how the memory should
            # be USED: a superseded memory must not be presented as current.
            if saved["sup_in"]:
                out["superseded"] = True
                out["superseded_by"] = _ids(saved["sup_in"])
        except Exception as e:
            logger.debug(f"Edge lookup failed for memory {memory_id}: {e}")
            out["edges_error"] = str(e)[:200]

    return out


@mcp.tool()
@_timed("memory_list")
def memory_list(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    min_importance: Optional[int] = None,
    limit: int = MAX_LIST_RESULTS,
    offset: int = 0,
    sort: Literal["recent", "importance", "accessed"] = "recent",
    global_search: bool = False,
) -> str:
    """Enumerate memories by recency, category, topic, or importance — no ranking.

    Use this when you want an ordered slice rather than an answer: "the 20 newest
    memories", "everything tagged auth", "all importance-5 decisions". Use
    memory_search when you have a question. Unlike search, this pages to
    arbitrary depth via offset.

    Sort: 'recent' (updated_at DESC), 'importance' (importance DESC then
    updated_at DESC), or 'accessed' (access_count DESC).
    """
    conn = get_conn()
    limit = _clamp_int(limit, 1, 200, MAX_LIST_RESULTS)
    offset = _clamp_int(offset, 0, 1_000_000, 0)

    where = []
    params: dict = {"limit": limit, "offset": offset, "ws": WORKSPACE}
    if category:
        where.append("m.category = $cat")
        params["cat"] = category
    if min_importance is not None:
        # int() on a non-numeric string used to raise a raw ValueError out of
        # the tool instead of returning the module's error shape.
        try:
            params["min_imp"] = int(min_importance)
        except (TypeError, ValueError):
            return {"status": "error",
                    "message": f"min_importance must be an integer 1-5, got {min_importance!r}."}
        where.append("m.importance >= $min_imp")
    if not global_search:
        where.append("m.workspace IN ['', $ws]")

    if tag:
        canon = _canonicalize_tag(tag)
        params["tag"] = canon
        match_clause = "MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $tag})"
    else:
        match_clause = "MATCH (m:Memory)"

    where_clause = "WHERE " + " AND ".join(where) if where else ""

    sort_map = {
        "recent": "m.updated_at DESC",
        "importance": "m.importance DESC, m.updated_at DESC",
        "accessed": "m.access_count DESC, m.updated_at DESC",
    }
    order_clause = f"ORDER BY {sort_map.get(sort, sort_map['recent'])}"

    query = (
        f"{match_clause} {where_clause} "
        f"RETURN m.id, m.content, m.category, m.tags, m.importance, "
        f"m.access_count, m.updated_at, m.workspace "
        f"{order_clause} SKIP $offset LIMIT $limit;"
    )
    try:
        rows = _collect_results(conn.execute(query, params))
    except Exception as e:
        return {"status": "error", "message": str(e)}

    items = []
    for row in rows:
        items.append({
            "id": row[0],
            "content": _truncate(row[1] or "", MAX_CONTENT_LENGTH),
            "category": row[2],
            "tags": _parse_tags(row[3]),
            "importance": row[4],
            "access_count": row[5] or 0,
            "updated_at": row[6],
            "workspace": row[7] or "",
        })
    return {
        "items": items,
        "returned": len(items),
        "offset": offset,
        "has_more": len(items) >= limit,
    }


@mcp.tool()
@_timed("memory_traverse")
def memory_traverse(cypher_query: str) -> str:
    """DEPRECATED — use memory_query(cypher_query=..., read_only=True) instead.

    Exactly equivalent to that call and kept only so existing hooks and steering
    files keep working. Any mutation is rejected regardless of
    MEMORY_ALLOW_DESTRUCTIVE.
    """
    # Call the underlying implementation directly (not the decorated wrapper)
    # so we avoid double-timing and keep a single elapsed_ms for memory_traverse.
    return memory_query.__wrapped__(cypher_query=cypher_query, read_only=True)


# --- Entry point ---

def main():
    """Entry point for the MCP server (used by pyproject.toml console_scripts).

    Configures logging at the root (we deliberately did NOT call basicConfig at
    import time — see the logger setup at the top of this module — so that
    importing the package as a library doesn't override the host's logging).
    """
    # Only configure logging when run as the entry point.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    # Log resolved configuration so users see what's active
    ws_default_warning = ""
    if "MEMORY_WORKSPACE" not in os.environ:
        ws_default_warning = " (WARNING: MEMORY_WORKSPACE not set, using cwd)"
    logger.info(f"Memnest MCP v{SERVER_VERSION}")
    logger.info(f"Configuration:")
    logger.info(f"  DB_PATH        = {DB_PATH}")
    logger.info(f"  WORKSPACE      = {WORKSPACE}{ws_default_warning}")
    logger.info(f"  EMBEDDING_DIM  = {EMBEDDING_DIM}")
    logger.info(f"  ALLOW_DESTROY  = {ALLOW_DESTRUCTIVE_QUERIES}")
    if ALLOW_DESTRUCTIVE_QUERIES:
        logger.warning(
            "MEMORY_ALLOW_DESTRUCTIVE=true: memory_query may execute "
            "DELETE/DROP/TRUNCATE. Disable in production unless explicitly required."
        )

    # Pre-warm: load embedding model and DB connection so first call isn't slow.
    # EMBED_TIMEOUT_S bounds how long we'll wait for the model to download/init.
    import threading as _threading
    warm_done = _threading.Event()
    warm_err: list = []

    def _warm():
        try:
            get_embed_model()
            get_conn()
            warm_done.set()
        except Exception as e:  # noqa: BLE001
            warm_err.append(e)
            warm_done.set()

    t = _threading.Thread(target=_warm, daemon=True)
    t.start()
    if warm_done.wait(timeout=EMBED_TIMEOUT_S):
        if warm_err:
            logger.warning(f"Pre-warm failed (will lazy-init on first call): {warm_err[0]}")
        else:
            logger.info("Pre-warm complete: embedding model and DB ready")
    else:
        logger.warning(
            f"Pre-warm timed out after {EMBED_TIMEOUT_S}s; continuing — first "
            f"call will block until the model finishes loading."
        )

    mcp.run()


if __name__ == "__main__":
    main()
