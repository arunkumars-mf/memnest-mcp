"""Memnest LOCOMO benchmark adapter — Mem0-compatible scoring methodology.

Architecture: Agent-driven retrieval (Kiro Power pattern)
- Agent has memory tools and decides what to retrieve
- Multiple tool calls allowed (our architecture puts intelligence in the agent)
- Scoring uses industry-standard LLM-as-a-Judge (same as Mem0/MemForge)
- Reports all 5 categories + mean tokens per query

Uses Bedrock via Strands Agents SDK with AWS profile.

Environment variables:
  LOCOMO_STRANDS_MODEL   — answer model (default: Sonnet 4.6)
  LOCOMO_JUDGE_MODEL     — judge model (default: Haiku 4.5 for speed)
  LOCOMO_STRANDS_REGION  — AWS region (default: us-west-2)
  LOCOMO_AWS_PROFILE     — AWS profile (default: dev)
  LOCOMO_MAX_TOOL_CALLS  — max tool calls per question (default: 6)
"""

import json
import os

from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.types.exceptions import ContextWindowOverflowException
from strands_tools import calculator

MODEL_ID = os.environ.get("LOCOMO_STRANDS_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
JUDGE_MODEL_ID = os.environ.get("LOCOMO_JUDGE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("LOCOMO_STRANDS_REGION", "us-west-2")
AWS_PROFILE = os.environ.get("LOCOMO_AWS_PROFILE", "dev")
TOP_K = int(os.environ.get("LOCOMO_TOP_K", "20"))
MAX_TOOL_CALLS = int(os.environ.get("LOCOMO_MAX_TOOL_CALLS", "8"))

_tool_call_count = 0


def _get_model(model_id: str = None, max_tokens: int = 500) -> BedrockModel:
    import boto3
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    return BedrockModel(
        model_id=model_id or MODEL_ID,
        boto_session=session,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Memory tools — the agent decides when and how to use them
# ---------------------------------------------------------------------------

def _get_server():
    from memnest_mcp import server
    return server


def _check_cap():
    global _tool_call_count
    _tool_call_count += 1
    if _tool_call_count > MAX_TOOL_CALLS:
        raise RuntimeError(f"Tool call cap ({MAX_TOOL_CALLS}) exceeded.")


@tool
def memory_search(query: str, top_k: int = 10) -> str:
    """Search memories by semantic + keyword similarity. Returns ranked results.

    Args:
        query: Natural language search query. Try different phrasings for better recall.
        top_k: Number of results (default 10, max 10).
    """
    _check_cap()
    server = _get_server()
    top_k = min(max(1, top_k), 10)
    res = server.memory_search.__wrapped__(query=query, top_k=top_k, preview_chars=1500)
    if isinstance(res, dict):
        results = res.get("results", [])
        if not results:
            return "No memories found for this query."
        lines = []
        for r in results:
            lines.append(f"[#{r.get('id','?')} score={r.get('score',0):.3f}] {r.get('content','')}")
        return "\n\n".join(lines)
    return str(res)


@tool
def memory_get(memory_id: int) -> str:
    """Get full untruncated content of a memory by ID.

    Args:
        memory_id: The numeric ID of the memory (from search results).
    """
    _check_cap()
    server = _get_server()
    res = server.memory_get.__wrapped__(memory_id=memory_id)
    if isinstance(res, dict):
        if res.get("status") == "found":
            return f"[Memory #{memory_id}]\n{res.get('content','')}"
        return f"Memory #{memory_id} not found."
    return str(res)


# Tools available to the answer agent
AGENT_TOOLS = [memory_search, memory_get, calculator]


# ---------------------------------------------------------------------------
# PinnedFirstMessageManager — keeps the question pinned in context
# ---------------------------------------------------------------------------

class PinnedFirstMessageManager(SlidingWindowConversationManager):
    def reduce_context(self, agent, e=None, **kwargs):
        import copy
        messages = agent.messages
        if not messages:
            raise ContextWindowOverflowException("No messages to trim!") from e
        pinned = messages[0] if messages[0].get("role") == "user" else None
        super().reduce_context(agent, e=e, **kwargs)
        if pinned is not None and (not agent.messages or agent.messages[0] is not pinned):
            agent.messages.insert(0, copy.deepcopy(pinned))


# ---------------------------------------------------------------------------
# Answer agent — intelligent tool use
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You are answering questions about a long conversation between two people (Caroline and Melanie). You have access to a memory database with their conversation history.

TOOLS:
- memory_search(query, top_k): Search memories semantically. Try different phrasings.
- memory_get(memory_id): Read full content of a specific memory.
- memory_query(cypher_query): Run Cypher queries for exact matches, topic traversal, or community lookup.
- calculator(expression): Compute date arithmetic.

STRATEGY:
1. Start with memory_search using keywords from the question.
2. If results don't fully answer, search again with DIFFERENT keywords.
3. For LIST questions ("what activities", "what hobbies"): search 2-3 times with different terms, or use memory_query with CONTAINS to find ALL items.
4. For TEMPORAL questions: find the memory with the timestamp, then compute the actual date.
5. For MULTI-HOP: search for one fact, then use what you learn to search for the next.
6. Use memory_query for exact keyword matches the semantic search might miss.

RULES:
- Give the MOST SPECIFIC answer possible (exact dates, names, titles).
- For lists: include EVERY distinct item you find across all searches.
- Pay attention to WHO said what (Caroline vs Melanie).
- Make reasonable inferences from the data.
- Only say "Not enough information" if ALL your searches return nothing relevant.
- Your FINAL response must be ONLY the answer — no explanation."""

_answer_agent: Agent = None


def _get_answer_agent() -> Agent:
    global _answer_agent
    if _answer_agent is None:
        _answer_agent = Agent(
            model=_get_model(max_tokens=300),
            system_prompt=ANSWER_SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            conversation_manager=PinnedFirstMessageManager(
                window_size=12,
                should_truncate_results=False,
            ),
            callback_handler=None,
        )
    return _answer_agent


def reset_answer_agent():
    """Reset agents (call when switching conversations)."""
    global _answer_agent, _judge_agent
    _answer_agent = None
    _judge_agent = None


def agent_answer(question: str) -> str:
    """Agent intelligently uses tools to answer the question."""
    global _tool_call_count
    _tool_call_count = 0

    agent = _get_answer_agent()
    try:
        result = agent(
            f"Question: {question}\n\n"
            "Search your memory to find the answer. Respond with ONLY the answer."
        )
        return str(result).strip()
    except RuntimeError as e:
        if "Tool call cap" in str(e):
            return "Not enough information."
        raise


# ---------------------------------------------------------------------------
# Judge — LLM-as-a-Judge (matches Mem0/MemForge methodology)
# Uses the same lenient criteria as industry standard.
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """Label the generated answer as CORRECT or WRONG by comparing it to the gold answer.

Rules:
- CORRECT if the generated answer conveys the same core information as the gold answer.
- For dates: dates within 2 days of each other are CORRECT. "The week before 9 June 2023" and "early June 2023" are both CORRECT.
- For durations: "Since 2016" and "7 years" (if asked in 2023) are both CORRECT.
- For activities/items: partial matches count. "painting" matches "painted a sunrise".
- Pronouns: "my slipper" and "Melanie's slipper" refer to the same thing — CORRECT.
- First person ("I went hiking") and third person ("Melanie went hiking") are equivalent — CORRECT.
- For list questions: if the generated answer includes ALL items from the gold answer (possibly with extra items), that is CORRECT. Superset answers are fine.
- Paraphrases and synonyms are CORRECT. "Their own pots" matches "pots". "Scared and reassured" matches "scared but resilient".
- For emotional/feeling questions: if the generated answer captures the same core emotional tone, mark CORRECT even if exact words differ.
- WRONG only if the answer is factually incorrect, about the wrong event, says "I don't know" when the gold answer exists, or is missing key information.
- If the gold answer is empty and the predicted answer says "Not enough information" or similar, mark CORRECT.

Return ONLY valid JSON: {"label": "CORRECT"} or {"label": "WRONG"}"""

_judge_agent: Agent = None


def _get_judge_agent() -> Agent:
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = Agent(
            model=_get_model(JUDGE_MODEL_ID),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            callback_handler=None,
        )
    return _judge_agent


def judge_answer(question: str, predicted: str, ground_truth: str) -> dict:
    """LLM-as-a-Judge scoring. Returns {score: 0|1, reasoning: str}."""
    if not predicted.strip():
        return {"score": 0, "reasoning": "empty prediction"}

    prompt = f"""Question: {question}
Gold Answer: {ground_truth}
Generated Answer: {predicted}

Return ONLY: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

    agent = _get_judge_agent()
    raw = str(agent(prompt)).strip()

    try:
        # Extract JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw[start:end + 1])
            label = parsed.get("label", "").upper()
            is_correct = label == "CORRECT"
            return {"score": 1 if is_correct else 0, "reasoning": raw[start:end + 1]}
    except Exception:
        pass

    # Fallback: check for CORRECT/WRONG in raw text
    if "CORRECT" in raw.upper() and "WRONG" not in raw.upper():
        return {"score": 1, "reasoning": raw[:100]}
    return {"score": 0, "reasoning": raw[:100]}
