"""
Memnest Memory MCP Server
===========================
AI agent memory powered by LadybugDB — graph + vector search in one embedded database.

Graph data model:
  - Memory nodes (content, embeddings, metadata, workspace)
  - Topic nodes (auto-linked from tags)
  - Relationships: ABOUT, RELATED_TO, SUPERSEDES, EXPLAINS

Workspace namespacing:
  - MEMORY_WORKSPACE env var scopes memories to a project
  - Defaults to cwd; pass global_search=True to bypass

Response format:
  - MEMORY_RESPONSE_FORMAT=toon (default if installed) for compact LLM-friendly output
  - MEMORY_RESPONSE_FORMAT=json for backward-compatible JSON
  - TOON typically reduces tokens by 30-60% vs JSON

Tools (see individual docstrings for params):
  memory_store, memory_search, memory_update, memory_delete, memory_relate,
  memory_query, memory_schema, memory_topics, memory_stats, memory_dream,
  memory_graph_html
  Compatibility aliases: memory_get, memory_list, memory_traverse
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

import real_ladybug as lb
from fastembed import TextEmbedding
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
DB_PATH = os.environ.get("MEMORY_DB_PATH", os.path.expanduser("~/.memnest/memory.lbug"))
EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("MEMORY_EMBEDDING_DIM", "384"))
DEDUP_THRESHOLD = float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92"))
LATENCY_WARN_MS = int(os.environ.get("MEMORY_LATENCY_WARN_MS", "200"))
WORKSPACE = os.environ.get("MEMORY_WORKSPACE", os.getcwd())

# Response format: 'json' (default) or 'toon' (compact for LLM context)
RESPONSE_FORMAT = os.environ.get("MEMORY_RESPONSE_FORMAT", "toon" if _TOON_AVAILABLE else "json").lower()

MAX_CONTENT_LENGTH = int(os.environ.get("MEMORY_MAX_CONTENT", "500"))
MAX_SEARCH_RESULTS = int(os.environ.get("MEMORY_SEARCH_LIMIT", "10"))
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
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
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
    global _conn, _db
    if _conn is not None:
        return _conn

    from pathlib import Path
    if DB_PATH == ":memory:":
        _db = lb.Database(":memory:")
    else:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db = lb.Database(DB_PATH)

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
    _load_dream_state()
    return _conn


# Schema version — bump when adding migrations to _apply_migrations().
# v1: workspace column on Memory.
# v2: provenance + confidence on RELATED_TO.
SCHEMA_VERSION = 2


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
        logger.warning(f"Embedding failed for text (len={len(text)}): {e}")
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


def _store_one(conn, content: str, category: str, tags: list[str], importance: int,
               embedding: Optional[list[float]] = None) -> dict:
    """Core single-memory store logic. Returns a status dict.

    If `embedding` is pre-computed (for batch mode), uses it instead of computing again.
    """
    c_hash = _content_hash(content)
    now = time.time()

    # Layer 1: Exact hash dedup
    result = conn.execute(
        "MATCH (m:Memory {content_hash: $hash}) RETURN m.id;",
        {"hash": c_hash},
    )
    if result.has_next():
        existing_id = result.get_next()[0]
        conn.execute(
            """MATCH (m:Memory {id: $id})
               SET m.importance = CASE WHEN m.importance < 5 THEN m.importance + 1 ELSE 5 END,
                   m.updated_at = $now;""",
            {"id": existing_id, "now": now},
        )
        return {"status": "already_exists", "id": existing_id}

    # Layer 2: Semantic dedup
    if embedding is None:
        embedding = _embed(content)
    if embedding is None:
        # Embedding failed — skip semantic dedup and store with NULL embedding
        # (vector search will skip these; FTS / Cypher still work)
        return _store_without_embedding(conn, content, c_hash, category, tags, importance, now)

    if _count_memories(conn) > 0:
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
            for row in _collect_results(result):
                similarity = round(1.0 - row[6], 4)
                if similarity >= DEDUP_THRESHOLD:
                    match_id = row[0]
                    match_content = row[1]
                    match_category = row[2]
                    match_tags = _parse_tags(row[3])
                    match_importance = row[4]

                    keep = content if len(content) > len(match_content) else match_content
                    merged_tags = list(set(match_tags + tags))
                    new_imp = min(5, max(match_importance, importance) + 1)
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
                        # Content unchanged — just update tags/importance/timestamp in place.
                        # Keep the existing embedding (no index rebuild needed).
                        conn.execute(
                            """MATCH (m:Memory {id: $id})
                               SET m.tags = $tags, m.importance = $imp, m.updated_at = $now;""",
                            {"id": match_id, "tags": _format_tags(merged_tags),
                             "imp": new_imp, "now": now},
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
         "cat": category, "tags": _format_tags(tags), "ws": WORKSPACE, "imp": importance,
         "now": now, "emb": embedding},
    )
    _ensure_topics(conn, mem_id, tags)

    return {"status": "stored_new", "id": mem_id}


@mcp.tool()
@_timed("memory_store")
def memory_store(
    content: Optional[str] = None,
    category: Literal["learning", "preference", "decision", "pattern", "general"] = "general",
    tags: Optional[list[str]] = None,
    importance: int = 3,
    items: Optional[list[dict]] = None,
) -> str:
    """Store one or more memories with auto-dedup and topic linking.
    Single: pass content/category/tags/importance.
    Batch: pass items=[{content, category?, tags?, importance?}, ...] (faster — single embed call).
    Importance 1-5; default 3 (neutral).
    """
    conn = get_conn()
    tags = tags or []

    # Batch mode
    if items is not None:
        if not items:
            return {"status": "error", "message": "items list is empty."}

        # Embed all in one call
        contents = [item.get("content", "") for item in items]
        embeddings = _embed_batch(contents)

        results = []
        for item, emb in zip(items, embeddings):
            try:
                res = _store_one(
                    conn,
                    content=item.get("content", ""),
                    category=item.get("category", "general"),
                    tags=item.get("tags", []),
                    importance=item.get("importance", 3),
                    embedding=emb,
                )
                results.append(res)
            except Exception as e:
                results.append({"status": "error", "message": str(e)})

        _bump_dream_ops()
        return {"results": results, "count": len(results)}

    # Single mode
    if content is None:
        return {"status": "error", "message": "content is required (or pass items=[...])."}

    res = _store_one(conn, content, category, tags, importance)
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
) -> str:
    """Hybrid semantic + keyword + graph search. Filters by current workspace unless global_search=True.
    top_k max 10. preview_chars caps content length per result (default 200).
    TIP: Pass tags=[...] to disambiguate overloaded query words (e.g. "workspace"
    could mean Brazil, Kiro, or ATX — tags narrow it instantly). See memory_stats
    for the list of known topics.
    """
    conn = get_conn()
    top_k = min(top_k, MAX_SEARCH_RESULTS)

    # Score accumulators: {memory_id: {vector: float, fts: float, graph: float}}
    raw_scores: dict[int, dict[str, float]] = {}
    memory_data: dict[int, dict] = {}

    def _record(mid: int, channel: str, score: float):
        if mid not in raw_scores:
            raw_scores[mid] = {"vector": 0.0, "fts": 0.0, "graph": 0.0}
        raw_scores[mid][channel] = max(raw_scores[mid][channel], score)

    # --- Channel 1: Vector search (HNSW cosine similarity) ---
    embedding = _embed(query)
    if embedding is not None and _count_memories(conn) > 0:
        try:
            if global_search:
                where_clause = ""
                vec_params: dict = {"query": embedding, "k": top_k * 3}
            else:
                where_clause = "WHERE m.workspace IN ['', $ws]"
                vec_params = {"query": embedding, "k": top_k * 3, "ws": WORKSPACE}
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
                similarity = 1.0 - row[8]
                _record(mid, "vector", similarity)
                memory_data[mid] = {
                    "id": mid, "content": row[1], "category": row[2],
                    "tags": _parse_tags(row[3]),
                    "importance": row[4], "access_count": row[5],
                    "workspace": row[6] or "", "updated_at": row[7] or 0.0,
                }
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

    # --- Channel 2: Full-text search (BM25 via FTS index) ---
    if _count_memories(conn) > 0:
        try:
            # Build FTS query: use the raw query words
            fts_query = query.strip()
            if fts_query:
                if global_search:
                    fts_where = ""
                    fts_params: dict = {"query": fts_query, "limit": top_k * 2}
                else:
                    fts_where = "AND m.workspace IN ['', $ws]"
                    fts_params = {"query": fts_query, "limit": top_k * 2, "ws": WORKSPACE}
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
                # as our top results — they're likely relevant too
                if top_communities and len(results) < top_k:
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
    now = time.time()
    final_scores: dict[int, float] = {}
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

    # Build results
    results = []
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

        results.append({
            "id": mid,
            "content": _truncate(mem["content"], preview_chars),
            "tags": mem["tags"],
            "score": round(score, 4),
        })
        if len(results) >= top_k:
            break

    # Bump access counts
    if results:
        ids = [r["id"] for r in results]
        _safe_execute(conn,
            "MATCH (m:Memory) WHERE m.id IN $ids "
            "SET m.access_count = m.access_count + 1;",
            {"ids": ids})

    # Wrap in a key so TOON can recognize the uniform array
    return {"results": results}


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
        if not updates:
            return {"status": "error", "message": "updates list is empty."}

        # Pre-compute embeddings for all items that change content
        contents_to_embed = [(i, u.get("content")) for i, u in enumerate(updates)
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
                    content=u.get("content"),
                    importance=u.get("importance"),
                    tags=u.get("tags"),
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

    res = _update_one(conn, memory_id, content, importance, tags)
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
    """Delete one or more memories (and their relationships). Pass int or list of ints."""
    conn = get_conn()
    ids = memory_id if isinstance(memory_id, list) else [memory_id]
    deleted = []
    not_found = []

    for mid in ids:
        result = conn.execute("MATCH (m:Memory {id: $id}) RETURN m.id;", {"id": mid})
        if not result.has_next():
            not_found.append(mid)
            continue
        conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": mid})
        deleted.append(mid)

    return {
        "status": "deleted",
        "deleted": deleted,
        "not_found": not_found,
    }


def _relate_one(conn, from_id: int, to_id: int, relationship: str = "RELATED_TO",
                confidence: float = 1.0, provenance: str = "EXTRACTED") -> dict:
    """Core single-relationship creation logic."""
    rel = relationship.upper()
    # SECURITY: rel is interpolated into Cypher (LadybugDB does not support
    # parameterized rel labels). Validate against an allowlist; never accept
    # arbitrary input here.
    if rel not in ("RELATED_TO", "SUPERSEDES", "EXPLAINS"):
        return {"status": "error", "from": from_id, "to": to_id,
                "message": f"Unknown relationship: {rel}. Use RELATED_TO, SUPERSEDES, or EXPLAINS."}

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
        if not relations:
            return {"status": "error", "message": "relations list is empty."}
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


# Configurable: bool to allow destructive queries through memory_query.
# Defaults to FALSE for safety — an MCP server is exposed to LLM agents which can
# (and have) hallucinated DELETE queries. Operators must explicitly opt in.
ALLOW_DESTRUCTIVE_QUERIES = os.environ.get("MEMORY_ALLOW_DESTRUCTIVE", "false").lower() == "true"


_DESTRUCTIVE_PATTERNS = (
    "DETACH DELETE",
    "DELETE ",
    "DROP ",
    "TRUNCATE",
)


def _is_destructive(query: str) -> bool:
    upper = query.upper()
    return any(p in upper for p in _DESTRUCTIVE_PATTERNS)


def _is_unsafe_embedding_set(query: str) -> bool:
    """Detect SET on m.embedding which fails silently due to HNSW index."""
    upper = query.upper().replace(" ", "")
    return "SETM.EMBEDDING" in upper or "SETEMBEDDING" in upper


@mcp.tool()
@_timed("memory_query")
def memory_query(cypher_query: str, read_only: bool = False) -> str:
    """Run a Cypher query. Supports traversals, writes, INSTALL/LOAD, CALL (algorithms, scans).
    Call memory_schema() first to get table/column names.
    WARNING: SET on m.embedding fails (vector index). Use memory_update for content changes.
    read_only=True rejects DELETE/DROP/TRUNCATE.
    """
    conn = get_conn()

    # Caller-requested read-only check (more specific — runs first so the error
    # message reflects the caller's intent rather than the global flag).
    if read_only and _is_destructive(cypher_query):
        return {
            "status": "error",
            "message": "Destructive query blocked by read_only=True.",
        }
    # Server-level kill switch for destructive ops
    if not ALLOW_DESTRUCTIVE_QUERIES and _is_destructive(cypher_query):
        return {
            "status": "error",
            "message": "Destructive query blocked by server config (MEMORY_ALLOW_DESTRUCTIVE=false).",
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
    limit = max(1, min(limit, 500))

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

                    result = conn.execute(
                        """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                           WITH node AS m, distance
                           RETURN m.id, m.content, m.tags, m.importance, m.created_at,
                                  m.category, m.workspace, distance;""",
                        {"query": list(embedding), "k": 4},
                    )
                    for row in _collect_results(result):
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
                            cluster_members.append({"id": row[0], "preview": _truncate(row[1], 100), "similarity": sim})
                            visited_clusters.add(row[0])

                    if cluster_members:
                        visited_clusters.add(mid)
                        clusters.append({
                            "anchor": {"id": mid, "preview": _truncate(content, 100), "importance": importance},
                            "similar": cluster_members,
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
# Compatibility aliases for the pre-0.2.0 tool surface.
#
# These wrap the new tools but keep the old names so existing hooks/steering
# files continue to work. They are thin shims and will be removed in 0.3.0.
# ----------------------------------------------------------------------------

@mcp.tool()
@_timed("memory_get")
def memory_get(memory_id: int) -> str:
    """Get full untruncated content of a memory by ID, including metadata.

    Compatibility alias retained from 0.1.x. Prefer `memory_query` for richer
    Cypher access; this remains for hooks and steering files that referenced
    the older name.
    """
    conn = get_conn()
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

    return {
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
    """List memories filtered by recency, category, topic, or importance.

    Compatibility alias retained from 0.1.x. For arbitrary queries use
    `memory_query`. Sort: 'recent' (updated_at DESC), 'importance' (importance
    DESC then updated_at DESC), or 'accessed' (access_count DESC).
    """
    conn = get_conn()
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    where = []
    params: dict = {"limit": limit, "offset": offset, "ws": WORKSPACE}
    if category:
        where.append("m.category = $cat")
        params["cat"] = category
    if min_importance is not None:
        where.append("m.importance >= $min_imp")
        params["min_imp"] = int(min_importance)
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
    """Run a READ-ONLY Cypher query (compatibility alias for memory_query).

    This shim forwards to memory_query with read_only=True. Destructive
    operations are always rejected here, regardless of MEMORY_ALLOW_DESTRUCTIVE.
    For writes, use memory_query directly.
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
