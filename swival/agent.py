import argparse
from contextlib import nullcontext
import copy
from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from importlib import metadata

import tiktoken

from . import fmt
from .config import _UNSET
from .report import AgentError, ConfigError, ReportCollector
from .snapshot import SNAPSHOT_HISTORY_SENTINEL, SnapshotState, READ_ONLY_TOOLS
from .thinking import ThinkingState
from .todo import TodoState
from .tracker import FileAccessTracker
from .mcp_client import McpShutdownError
from .tools import (
    TOOLS,
    RUN_COMMAND_TOOL,
    USE_SKILL_TOOL,
    dispatch,
    cleanup_old_cmd_outputs,
)

DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
MAX_ARG_LOG = 1000
MAX_INSTRUCTIONS_CHARS = 10_000

_encoder = tiktoken.get_encoding("cl100k_base")

MAX_HISTORY_SIZE = 500 * 1024  # 500KB
TODO_REMINDER_INTERVAL = 3  # remind after N turns of no todo usage

INIT_PROMPT = (
    "Scan this project to find its conventions — patterns applied consistently "
    "across the whole codebase that an AI agent wouldn't know without reading "
    "the source. Look across all areas: naming schemes, file and directory "
    "structure, error handling, return value formats, test organisation, "
    "documentation style, CLI behaviour, exit codes, and API design. Read "
    "source files, tests, docs, and config. Use think to separate genuine "
    "project-wide patterns (appear in many independent places) from one-off "
    "choices."
)

INIT_ENRICH_PROMPT = (
    "Review your list. Cut anything that: (1) only appears in one file or "
    "module, (2) is standard Python/Unix/web practice that any competent agent "
    "would already know, or (3) wouldn't affect how an agent writes correct "
    "code or makes tool calls. Keep only conventions that cross module "
    "boundaries and would surprise a capable agent new to this project. Check "
    "tests, docs, and config files for anything missed."
)

INIT_WRITE_PROMPT = (
    "Write the findings to AGENTS.md as a concise bulleted list. "
    "Two sentences maximum per item. "
    "The file is injected into every future agent context, so brevity is essential."
)

_CONTEXT_OVERFLOW_RE = re.compile(
    r"context.{0,10}(length|window|limit)"
    r"|maximum.{0,10}(context|token)"
    r"|token.{0,10}limit"
    r"|exceed.{0,10}(context|token|max)",
    re.IGNORECASE,
)

_EMPTY_ASSISTANT_RE = re.compile(
    r"must have either content or tool_calls"
    r"|must have either 'content' or 'tool_calls'"
    r"|must have non-null content or tool_calls",
    re.IGNORECASE,
)


class ContextOverflowError(Exception):
    """Raised when the LLM call fails due to context window overflow."""

    pass


# ---------------------------------------------------------------------------
# Message accessor helpers — abstract over dict vs object messages
# ---------------------------------------------------------------------------


def _msg_get(msg, key, default=None):
    return (
        msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)
    )


def _msg_role(msg) -> str | None:
    return _msg_get(msg, "role")


def _msg_content(msg) -> str:
    return _msg_get(msg, "content", "") or ""


def _msg_tool_calls(msg):
    return _msg_get(msg, "tool_calls")


def _msg_tool_call_id(msg) -> str | None:
    return _msg_get(msg, "tool_call_id")


def _msg_name(msg) -> str:
    return _msg_get(msg, "name", "") or ""


def _set_msg_content(msg, value: str) -> None:
    if isinstance(msg, dict):
        msg["content"] = value
    else:
        msg.content = value


def _sanitize_assistant_messages(messages: list) -> bool:
    """Fix assistant messages that have neither content nor tool_calls.

    Some providers (e.g. Mistral via OpenRouter) reject conversations containing
    assistant messages with both content and tool_calls absent.  Setting content
    to an empty string satisfies validation.

    Returns True if any messages were fixed.
    """
    fixed = False
    for msg in messages:
        if _msg_role(msg) != "assistant":
            continue
        has_content = bool(_msg_content(msg))
        has_tools = bool(_msg_tool_calls(msg))
        if not has_content and not has_tools:
            _set_msg_content(msg, "")
            fixed = True
    return fixed


def _safe_history_path(base_dir: str) -> Path:
    """Build history path, verify it resolves inside base_dir."""
    base = Path(base_dir).resolve()
    history_path = (Path(base_dir) / ".swival" / "HISTORY.md").resolve()
    if not history_path.is_relative_to(base):
        raise ValueError(f"history path {history_path} escapes base directory {base}")
    return history_path


def append_history(
    base_dir: str, question: str, answer: str, *, diagnostics: bool = True
) -> None:
    """Append a timestamped Q&A entry to .swival/HISTORY.md."""
    if not answer or not answer.strip():
        return

    try:
        history_path = _safe_history_path(base_dir)
    except ValueError:
        if diagnostics:
            fmt.warning("history path escapes base directory, skipping write")
        return

    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)

        current_size = history_path.stat().st_size if history_path.exists() else 0
        if current_size >= MAX_HISTORY_SIZE:
            if diagnostics:
                fmt.warning("history file at capacity, skipping write")
            return

        # Truncate question for the header
        q_display = question[:200] + "..." if len(question) > 200 else question
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"---\n\n**{timestamp}** — *{q_display}*\n\n{answer}\n\n"

        with history_path.open("a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        if diagnostics:
            fmt.warning("failed to write history entry")


def _canonical_error(error: str) -> str:
    """Extract a stable error fingerprint for repeat detection."""
    return error.split("\n", 1)[0]


def estimate_tokens(messages: list, tools: list | None = None) -> int:
    """Count tokens across all messages using tiktoken."""
    total = 0
    for m in messages:
        content = _msg_content(m)
        tool_calls = _msg_tool_calls(m)
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    content += tc.function.name + (tc.function.arguments or "")
                elif isinstance(tc, dict):
                    fn = tc.get("function", {})
                    content += fn.get("name", "") + (fn.get("arguments", "") or "")
        total += len(_encoder.encode(content))
    if tools:
        total += len(_encoder.encode(json.dumps(tools)))
    # Per-message overhead (role, separators) — ~4 tokens each
    total += 4 * len(messages)
    return total


def _estimate_tool_tokens(tools: list) -> int:
    """Estimate token cost of the tool schemas alone."""
    if not tools:
        return 0
    return len(_encoder.encode(json.dumps(tools)))


def enforce_mcp_token_budget(
    tools: list,
    mcp_manager,
    context_length: int | None,
    verbose: bool = False,
) -> list:
    """Check MCP tool token usage against context budget.

    Iteratively drops the most expensive MCP server until under 50% of context.
    Returns the (possibly trimmed) tools list.
    """
    if context_length is None or mcp_manager is None:
        return tools

    tool_tokens = _estimate_tool_tokens(tools)
    threshold_warn = int(context_length * 0.3)
    threshold_drop = int(context_length * 0.5)

    if tool_tokens <= threshold_warn:
        return tools

    # Compute per-server token costs
    tool_info = mcp_manager.get_tool_info()
    if not tool_info:
        return tools

    if tool_tokens > threshold_warn:
        # Always warn (not gated on verbose) — this is operationally important
        lines = []
        for server_name in tool_info:
            server_schemas = [
                t
                for t in tools
                if t.get("function", {})
                .get("name", "")
                .startswith(f"mcp__{server_name}__")
            ]
            st = _estimate_tool_tokens(server_schemas)
            lines.append(f"  {server_name}: ~{st} tokens ({len(server_schemas)} tools)")
        fmt.warning(
            f"MCP tool schemas use ~{tool_tokens} tokens "
            f"({tool_tokens * 100 // context_length}% of context):\n" + "\n".join(lines)
        )

    # Iterative drop loop
    while tool_tokens > threshold_drop and tool_info:
        # Find server with most token cost
        worst_server = None
        worst_tokens = 0
        for server_name in tool_info:
            server_schemas = [
                t
                for t in tools
                if t.get("function", {})
                .get("name", "")
                .startswith(f"mcp__{server_name}__")
            ]
            st = _estimate_tool_tokens(server_schemas)
            if st > worst_tokens:
                worst_tokens = st
                worst_server = server_name

        if worst_server is None:
            break

        # Drop this server's tools from the tools list and manager state
        prefix = f"mcp__{worst_server}__"
        tools = [
            t
            for t in tools
            if not t.get("function", {}).get("name", "").startswith(prefix)
        ]
        del tool_info[worst_server]

        # Update manager internals so get_tool_info() reflects the drop
        mcp_manager._tool_schemas.pop(worst_server, None)
        for key in list(mcp_manager._tool_map):
            if key.startswith(prefix):
                del mcp_manager._tool_map[key]

        tool_tokens = _estimate_tool_tokens(tools)
        fmt.error(
            f"Dropped MCP server {worst_server!r} tools (~{worst_tokens} tokens) "
            f"to stay under 50% context budget. "
            f"Remaining: ~{tool_tokens} tokens."
        )

    return tools


def group_into_turns(messages: list) -> list[list]:
    """Group messages into atomic turns.

    A turn is one of:
    - A single message (system, user, or assistant without tool_calls)
    - An assistant message with tool_calls + all its matching tool results
    """
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = _msg_role(msg)
        tool_calls = _msg_tool_calls(msg)

        if role == "assistant" and tool_calls:
            # Collect this assistant msg + all following tool results
            turn = [msg]
            tc_ids = {tc.id if hasattr(tc, "id") else tc["id"] for tc in tool_calls}
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                next_role = _msg_role(next_msg)
                tc_id = _msg_tool_call_id(next_msg)
                if next_role == "tool" and tc_id in tc_ids:
                    turn.append(next_msg)
                    j += 1
                else:
                    break
            turns.append(turn)
            i = j
        else:
            turns.append([msg])
            i += 1
    return turns


def compact_tool_result(name: str, args: dict | None, content: str) -> str:
    """Produce a structured summary for a large tool result.

    Returns the original *content* unchanged when it is short enough (<=1000
    chars).  For larger results the summary preserves the tool name, key
    arguments, and output metadata so the model still knows what happened.
    """
    if len(content) <= 1000:
        return content

    args = args or {}

    if name == "read_file":
        path = args.get("file_path", "?")
        lines = content.count("\n")
        return f"[read_file: {path}, {lines} lines — content compacted]"

    if name == "grep":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        matches = content.count("\n")
        return f"[grep: '{pattern}' in {path}, ~{matches} matches — compacted]"

    if name == "list_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        count = content.count("\n")
        return f"[list_files: '{pattern}' in {path}, ~{count} entries — compacted]"

    if name == "run_command":
        cmd = args.get("command", ["?"])
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        head = content[:200]
        tail = content[-200:]
        return (
            f"[run_command: `{cmd}` — first 200 chars:\n{head}\n"
            f"... last 200 chars:\n{tail}]"
        )

    if name == "fetch_url":
        url = args.get("url", "?")
        return f"[fetch_url: {url}, {len(content)} chars — content compacted]"

    if name.startswith("mcp__"):
        head = content[:300]
        return f"[{name}: {len(content)} chars — compacted]\nFirst 300 chars:\n{head}"

    # Unknown tool — generic structured fallback
    return f"[{name}: compacted — originally {len(content)} chars]"


def _tool_call_index(turn: list) -> dict[str, tuple[str, dict | None]]:
    """Build a mapping from tool_call_id → (tool_name, parsed_args) for a turn.

    The first message in a tool-call turn is the assistant message whose
    ``tool_calls`` list carries the function name and arguments.
    """
    index: dict[str, tuple[str, dict | None]] = {}
    first = turn[0]
    tool_calls = _msg_tool_calls(first)
    if not tool_calls:
        return index
    for tc in tool_calls:
        tc_id = tc.id if hasattr(tc, "id") else tc["id"]
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "?")
        fn_args_raw = (
            fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        )
        try:
            fn_args = (
                json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            )
        except (json.JSONDecodeError, TypeError):
            fn_args = None
        index[tc_id] = (fn_name, fn_args)
    return index


def compact_messages(messages: list) -> list:
    """Compact large tool results in older turns, preserving turn atomicity.

    Uses per-tool structured summaries (via ``compact_tool_result``) instead of
    a blanket character-count truncation.
    """
    turns = group_into_turns(messages)
    # Skip the most recent 2 turns
    cutoff = max(0, len(turns) - 2)
    for turn in turns[:cutoff]:
        tc_index = _tool_call_index(turn)
        for msg in turn:
            if _msg_role(msg) == "tool":
                content = _msg_content(msg)
                if content and len(content) > 1000:
                    tc_id = _msg_tool_call_id(msg)
                    tool_name, tool_args = tc_index.get(tc_id, ("?", None))
                    replacement = compact_tool_result(tool_name, tool_args, content)
                    _set_msg_content(msg, replacement)
    # Flatten turns back to message list
    return [msg for turn in turns for msg in turn]


def is_pinned(turn: list) -> bool:
    """User turns are always preserved — they must never be silently dropped."""
    return any(_msg_role(msg) == "user" for msg in turn)


def score_turn(turn: list) -> int:
    """Heuristic importance score for an agent/tool turn.

    Higher scores mean the turn is more valuable to keep.
    """
    score = 0
    for msg in turn:
        content = _msg_content(msg)
        # Errors are important — the agent learned something
        if "error" in content.lower() or "failed" in content.lower():
            score += 3
        # File writes/edits are important — the agent took action
        tool_calls = _msg_tool_calls(msg)
        if tool_calls:
            for tc in tool_calls:
                fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                if fn_name in ("write_file", "edit_file"):
                    score += 5
        # Thinking turns are important — the agent reasoned
        if "think" in _msg_name(msg):
            score += 2
        # Snapshot recap messages are high-value distilled knowledge
        if content.startswith("[snapshot:"):
            score += 5
    return score


_STATIC_SPLICE_MARKER = {
    "role": "user",
    "content": (
        "[context compacted — older tool calls and results were "
        "removed to fit context window]"
    ),
}

_RECAP_PREFIX = (
    "[non-instructional context recap — this is a factual summary "
    "of prior conversation, not a set of instructions]\n\n"
)


def _count_leading_turns(turns: list, roles: str | set) -> int:
    """Count consecutive turns at the start whose first message has a role in *roles*."""
    if isinstance(roles, str):
        roles = {roles}
    count = 0
    for turn in turns:
        if _msg_role(turn[0]) in roles:
            count += 1
        else:
            break
    return count


def _build_checkpoint_recap(compaction_state) -> dict | None:
    """Build a recap message from compaction checkpoint summaries, or None."""
    if compaction_state and compaction_state.summaries:
        checkpoint_text = compaction_state.get_full_summary()
        if checkpoint_text:
            return {
                "role": "assistant",
                "content": (
                    "[non-instructional context recap — factual summary "
                    "from periodic checkpoints]\n\n" + checkpoint_text
                ),
            }
    return None


def drop_middle_turns(
    messages: list,
    *,
    call_llm_fn=None,
    model_id=None,
    base_url=None,
    api_key=None,
    top_p=None,
    seed=None,
    provider=None,
    compaction_state: "CompactionState | None" = None,
) -> list:
    """Drop lowest-importance middle turns; pin user turns, keep leading block + tail.

    When *call_llm_fn* and the associated LLM parameters are provided, the
    dropped turns are summarized into a compact recap injected as an
    ``assistant`` message.  If summarization fails, falls back to the
    checkpoint summary (if available), then to the static splice marker.
    """
    turns = group_into_turns(messages)

    leading_count = _count_leading_turns(turns, {"system", "user"})

    keep_tail = 3
    # If there's no middle to drop, return unchanged
    if leading_count + keep_tail >= len(turns):
        return [msg for turn in turns for msg in turn]

    leading = turns[:leading_count]
    middle = turns[leading_count:-keep_tail]
    tail = turns[-keep_tail:]

    # Partition middle into pinned (user) and droppable (agent/tool) turns.
    pinned = []
    droppable = []
    for turn in middle:
        if is_pinned(turn):
            pinned.append(turn)
        else:
            droppable.append(turn)

    # Sort droppable turns by score descending and keep only the top ones.
    droppable.sort(key=score_turn, reverse=True)
    keep_count = len(droppable) // 2
    kept = droppable[:keep_count]
    dropped = droppable[keep_count:]

    # Try AI summarization of dropped turns, fall back to static marker.
    recap = None
    if call_llm_fn and dropped:
        summary = summarize_turns(
            dropped,
            call_llm_fn,
            model_id,
            base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )
        if summary:
            recap = {
                "role": "assistant",
                "content": _RECAP_PREFIX + summary,
            }

    if recap is None:
        recap = _build_checkpoint_recap(compaction_state)

    if recap is None:
        recap = dict(_STATIC_SPLICE_MARKER)

    result = []
    for turn in leading:
        result.extend(turn)
    result.append(recap)
    # Reassemble kept middle turns in original order
    kept_set = set(id(t) for t in kept) | set(id(t) for t in pinned)
    for turn in middle:
        if id(turn) in kept_set:
            for msg in turn:
                result.append(msg)
    for turn in tail:
        result.extend(turn)
    return result


def aggressive_drop_turns(
    messages: list,
    *,
    call_llm_fn=None,
    model_id=None,
    base_url=None,
    api_key=None,
    top_p=None,
    seed=None,
    provider=None,
    compaction_state: "CompactionState | None" = None,
) -> list:
    """Aggressive compaction: keep only system prompt + recap + last 2 turns.

    This is the last resort before giving up. All middle content is dropped
    and replaced with a summary (or static marker if summarization fails).
    """
    turns = group_into_turns(messages)

    leading_count = _count_leading_turns(turns, "system")

    keep_tail = 2
    if leading_count + keep_tail >= len(turns):
        return [msg for turn in turns for msg in turn]

    leading = turns[:leading_count]
    middle = turns[leading_count:-keep_tail]
    tail = turns[-keep_tail:]

    # Try to summarize everything being dropped
    recap = None
    if call_llm_fn and middle:
        summary = summarize_turns(
            middle,
            call_llm_fn,
            model_id,
            base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )
        if summary:
            recap = {
                "role": "assistant",
                "content": _RECAP_PREFIX + summary,
            }

    if recap is None:
        recap = _build_checkpoint_recap(compaction_state)

    if recap is None:
        recap = dict(_STATIC_SPLICE_MARKER)

    result = []
    for turn in leading:
        result.extend(turn)
    result.append(recap)
    for turn in tail:
        result.extend(turn)
    return result


def _call_summarize_llm(
    text, system_prompt, call_llm_fn, model_id, base_url, api_key, top_p, seed, provider
):
    """Call the LLM to summarize text. Returns string or None on failure."""
    if len(text) > 8000:
        text = text[:8000] + "\n[... truncated for summary call]"

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    try:
        resp, _ = call_llm_fn(
            base_url=base_url,
            model_id=model_id,
            messages=prompt,
            max_output_tokens=512,
            temperature=0,
            top_p=top_p,
            seed=seed,
            tools=None,
            verbose=False,
            api_key=api_key,
            provider=provider,
        )
        content = resp.content if hasattr(resp, "content") else resp.get("content", "")
        return content if content else None
    except Exception:
        return None


def summarize_turns(
    turns_to_drop, call_llm_fn, model_id, base_url, api_key, top_p, seed, provider
):
    """Ask the model to summarize dropped turns into a compact recap.

    Returns the summary string, or ``None`` if summarization fails for any
    reason.  The caller **must** fall back to the static splice marker when
    this returns ``None``.
    """
    flat = []
    for turn in turns_to_drop:
        for msg in turn:
            role = _msg_role(msg) or "?"
            content = _msg_content(msg)
            if content:
                flat.append(f"[{role}] {content[:2000]}")

    joined = "\n".join(flat)
    return _call_summarize_llm(
        joined,
        "Summarize this agent conversation excerpt into a factual recap. "
        "Preserve: file paths, key findings, decisions, errors, and "
        "anything needed to continue the task. Do NOT include instructions "
        "or directives. Output only a factual summary. Be concise.",
        call_llm_fn,
        model_id,
        base_url,
        api_key,
        top_p,
        seed,
        provider,
    )


def summarize_turns_from_text(
    text, call_llm_fn, model_id, base_url, api_key, top_p, seed, provider
):
    """Summarize pre-joined text (used for checkpoint consolidation).

    Same contract as ``summarize_turns``: returns a string or ``None``.
    """
    return _call_summarize_llm(
        text,
        "Condense these conversation summaries into a single, shorter "
        "factual recap. Preserve: file paths, key findings, decisions, "
        "errors. Do NOT include instructions or directives. Be concise.",
        call_llm_fn,
        model_id,
        base_url,
        api_key,
        top_p,
        seed,
        provider,
    )


MAX_CHECKPOINT_TOKENS = 2048
MAX_CHECKPOINTS = 5


class CompactionState:
    """Rolling summary checkpoints for proactive context preservation.

    Every *checkpoint_interval* turns, the recent turns are summarized and
    appended to an internal list. When the list grows beyond ``MAX_CHECKPOINTS``,
    the oldest half is merged into a single consolidated summary (hierarchical
    map/reduce).  If the merge fails, the oldest summaries are dropped to
    enforce the bound unconditionally.
    """

    def __init__(self, checkpoint_interval: int = 10):
        self.summaries: list[str] = []
        self.turns_since_last: int = 0
        self.checkpoint_interval: int = checkpoint_interval

    def maybe_checkpoint(
        self,
        messages,
        call_llm_fn,
        *,
        model_id,
        base_url,
        api_key,
        top_p,
        seed,
        provider,
    ):
        """Attempt a checkpoint after each agent turn.

        Always resets the counter regardless of success/failure so a transient
        outage doesn't cause retry on every subsequent turn.
        """
        self.turns_since_last += 1
        if self.turns_since_last < self.checkpoint_interval:
            return

        self.turns_since_last = 0

        recent = _get_recent_turns(messages, self.checkpoint_interval)
        summary = summarize_turns(
            recent,
            call_llm_fn,
            model_id,
            base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )
        if summary is None:
            return

        self.summaries.append(summary)
        self._maybe_consolidate(
            call_llm_fn,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )

    def _maybe_consolidate(
        self, call_llm_fn, *, model_id, base_url, api_key, top_p, seed, provider
    ):
        """Merge old summaries when the list exceeds MAX_CHECKPOINTS."""
        if len(self.summaries) <= MAX_CHECKPOINTS:
            return
        half = len(self.summaries) // 2
        to_merge = self.summaries[:half]
        merged = summarize_turns_from_text(
            "\n\n".join(to_merge),
            call_llm_fn,
            model_id,
            base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )
        if merged:
            self.summaries = [merged] + self.summaries[half:]
        else:
            # Consolidation failed — drop oldest to enforce bound.
            self.summaries = self.summaries[half:]

    def get_full_summary(self) -> str:
        """Return all checkpoint summaries joined, hard-capped by char count."""
        full = "\n\n".join(self.summaries)
        cap = MAX_CHECKPOINT_TOKENS * 4  # ~4 chars/token estimate
        if len(full) > cap:
            full = full[:cap] + "\n[... older checkpoints truncated]"
        return full


def _get_recent_turns(messages: list, n: int) -> list[list]:
    """Return the last *n* turns from *messages*."""
    turns = group_into_turns(messages)
    return turns[-n:] if len(turns) > n else turns


MIN_OUTPUT_TOKENS = 16  # Minimum accepted by most LLM APIs


def clamp_output_tokens(
    messages: list,
    tools: list | None,
    context_length: int | None,
    requested_max_output: int,
) -> int:
    """Reduce max_output_tokens if prompt + output would exceed context.

    Raises ContextOverflowError if there isn't enough room for even the
    minimum output budget — the caller should compact and retry.
    """
    if context_length is None:
        return requested_max_output
    prompt_tokens = estimate_tokens(messages, tools)
    available = context_length - prompt_tokens
    if available < MIN_OUTPUT_TOKENS:
        raise ContextOverflowError(
            f"Prompt (~{prompt_tokens} tokens) leaves only {available} tokens "
            f"for output (need >= {MIN_OUTPUT_TOKENS}); context_length={context_length}"
        )
    return min(requested_max_output, available)


def load_instructions(
    base_dir: str,
    config_dir: "Path | None" = None,
    *,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    """Load CLAUDE.md and/or AGENTS.md, if present.

    User-level AGENTS.md (from *config_dir*) is prepended to the
    project-level AGENTS.md (from *base_dir*) inside a single
    ``<agent-instructions>`` block.  Both share a combined budget of
    ``MAX_INSTRUCTIONS_CHARS``.

    Returns (combined_text, filenames_loaded) where combined_text is
    XML-tagged sections (or "" if none found) and filenames_loaded lists
    the absolute paths of files that were actually loaded.
    """
    sections: list[str] = []
    loaded: list[str] = []

    # --- CLAUDE.md (project-level only) ---
    claude_path = Path(base_dir).resolve() / "CLAUDE.md"
    if claude_path.is_file():
        try:
            file_size = claude_path.stat().st_size
            with claude_path.open(encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_INSTRUCTIONS_CHARS + 1)
        except OSError:
            content = None
        else:
            if len(content) > MAX_INSTRUCTIONS_CHARS:
                content = (
                    content[:MAX_INSTRUCTIONS_CHARS]
                    + f"\n[truncated — CLAUDE.md exceeds {MAX_INSTRUCTIONS_CHARS} character limit]"
                )
            if verbose:
                fmt.info(
                    f"Loaded CLAUDE.md ({file_size} bytes) from {claude_path.parent}"
                )
            sections.append(
                f"<project-instructions>\n{content}\n</project-instructions>"
            )
            loaded.append(str(claude_path))

    # --- AGENTS.md (user-level + project-level, shared budget) ---
    agent_parts: list[str] = []
    budget = MAX_INSTRUCTIONS_CHARS

    # User-level AGENTS.md
    if config_dir is not None:
        user_agents_path = Path(config_dir) / "AGENTS.md"
        if user_agents_path.is_file():
            try:
                file_size = user_agents_path.stat().st_size
                with user_agents_path.open(encoding="utf-8", errors="replace") as f:
                    user_content = f.read(budget + 1)
            except OSError:
                if verbose:
                    fmt.info(f"Skipped unreadable {user_agents_path}")
            else:
                if len(user_content) > budget:
                    user_content = (
                        user_content[:budget]
                        + f"\n[truncated — user AGENTS.md exceeds {budget} character limit]"
                    )
                budget -= len(user_content)
                if verbose:
                    fmt.info(
                        f"Loaded AGENTS.md ({file_size} bytes) from {user_agents_path.parent}"
                    )
                agent_parts.append(f"<!-- user: {user_agents_path} -->\n{user_content}")
                loaded.append(str(user_agents_path))

    # Project-level AGENTS.md
    proj_agents_path = Path(base_dir).resolve() / "AGENTS.md"
    if proj_agents_path.is_file() and budget > 0:
        try:
            file_size = proj_agents_path.stat().st_size
            with proj_agents_path.open(encoding="utf-8", errors="replace") as f:
                proj_content = f.read(budget + 1)
        except OSError:
            pass
        else:
            if len(proj_content) > budget:
                proj_content = (
                    proj_content[:budget]
                    + f"\n[truncated — AGENTS.md exceeds {budget} character limit]"
                )
            if verbose:
                fmt.info(
                    f"Loaded AGENTS.md ({file_size} bytes) from {proj_agents_path.parent}"
                )
            agent_parts.append(f"<!-- project: {proj_agents_path} -->\n{proj_content}")
            loaded.append(str(proj_agents_path))

    if agent_parts:
        inner = "\n\n".join(agent_parts)
        sections.append(f"<agent-instructions>\n{inner}\n</agent-instructions>")

    return "\n\n".join(sections), loaded


def handle_tool_call(
    tool_call,
    base_dir,
    thinking_state,
    verbose,
    resolved_commands=None,
    skills_catalog=None,
    skill_read_roots=None,
    extra_write_roots=None,
    yolo=False,
    file_tracker=None,
    todo_state=None,
    snapshot_state=None,
    mcp_manager=None,
    messages=None,
):
    """Execute a single tool call and return (tool_msg, metadata).

    tool_msg is the message dict for the LLM conversation.
    metadata has stable keys: name, arguments, elapsed, succeeded.
    """
    name = tool_call.function.name
    raw_args = tool_call.function.arguments

    try:
        parsed_args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError) as e:
        if verbose:
            fmt.tool_error(name, f"invalid JSON: {e}")
        error_content = f"error: invalid JSON in tool arguments: {e}"
        return (
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": error_content,
            },
            {"name": name, "arguments": None, "elapsed": 0.0, "succeeded": False},
        )

    _skip_generic_log = name in ("think", "todo", "snapshot")
    if not _skip_generic_log and verbose:
        pretty = json.dumps(parsed_args, indent=2)
        if len(pretty) > MAX_ARG_LOG:
            pretty = pretty[:MAX_ARG_LOG] + "\n... (truncated)"
        fmt.tool_call(name, pretty)

    t0 = time.monotonic()
    try:
        result = dispatch(
            name,
            parsed_args,
            base_dir,
            thinking_state=thinking_state,
            todo_state=todo_state,
            snapshot_state=snapshot_state,
            resolved_commands=resolved_commands or {},
            skills_catalog=skills_catalog or {},
            skill_read_roots=skill_read_roots if skill_read_roots is not None else [],
            extra_write_roots=extra_write_roots
            if extra_write_roots is not None
            else [],
            yolo=yolo,
            file_tracker=file_tracker,
            tool_call_id=tool_call.id,
            mcp_manager=mcp_manager,
            messages=messages,
        )
    except McpShutdownError:
        result = "error: MCP server is shutting down"
    except Exception as e:
        result = f"error: {e}"
    elapsed = time.monotonic() - t0

    succeeded = not result.startswith("error:")
    if not _skip_generic_log and verbose:
        if not succeeded:
            fmt.tool_error(name, result)
        else:
            fmt.tool_result(name, elapsed, result[:500])

    return (
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        },
        {
            "name": name,
            "arguments": parsed_args,
            "elapsed": elapsed,
            "succeeded": succeeded,
        },
    )


def discover_model(base_url, verbose):
    """Query LM Studio's native API to find the currently loaded LLM."""
    url = f"{base_url}/api/v1/models"
    if verbose:
        fmt.model_info(f"Querying {url} for loaded models...")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise AgentError(f"could not connect to LM Studio at {base_url}: {e}")
    except json.JSONDecodeError as e:
        raise AgentError(f"invalid JSON from {url}: {e}")

    # Find first entry with type=="llm" and non-empty loaded_instances
    # LM Studio uses "data" (OpenAI-compat) or "models" (native API) as the top-level key
    entries = data.get("data") or data.get("models") or []
    for entry in entries:
        if entry.get("type") == "llm" and entry.get("loaded_instances"):
            instance = entry["loaded_instances"][0]
            context_length = instance.get("config", {}).get("context_length")
            model_key = entry.get("id", entry.get("key"))
            vision = False
            try:
                import litellm

                vision = litellm.supports_vision(model=f"openai/{model_key}")
            except Exception:
                pass
            if verbose:
                vision_tag = " (vision enabled)" if vision else ""
                fmt.model_info(
                    f"Discovered loaded model: {model_key} (context={context_length}){vision_tag}"
                )
            return model_key, context_length

    return None, None


def configure_context(base_url, model_key, requested_context, current_context, verbose):
    """Reload the model with a different context size if needed."""
    if requested_context == current_context:
        if verbose:
            fmt.model_info(
                f"Requested context {requested_context} matches current context, no reload needed."
            )
        return

    url = f"{base_url}/api/v1/models/load"
    payload = json.dumps(
        {"model": model_key, "context_length": requested_context}
    ).encode()
    if verbose:
        fmt.model_info(
            f"Reloading model {model_key} with context_length={requested_context}..."
        )
        fmt.model_info("Note: this may take a while as the model reloads.")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except urllib.error.URLError as e:
        raise AgentError(f"failed to reload model with new context size: {e}")

    if verbose:
        fmt.model_info("Model reloaded successfully.")


def call_llm(
    base_url,
    model_id,
    messages,
    max_output_tokens,
    temperature,
    top_p,
    seed,
    tools,
    verbose,
    *,
    provider="lmstudio",
    api_key=None,
    extra_body=None,
):
    """Call LiteLLM with the appropriate provider. Returns (message, finish_reason)."""
    import litellm

    litellm.suppress_debug_info = True

    if provider == "lmstudio":
        model_str = f"openai/{model_id}"
        kwargs = {"api_base": f"{base_url}/v1", "api_key": "lm-studio"}
    elif provider == "huggingface":
        bare_id = model_id.removeprefix("huggingface/")
        model_str = f"huggingface/{bare_id}"
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["api_base"] = base_url
    elif provider == "openrouter":
        # Only strip the prefix if the user already included the LiteLLM
        # "openrouter/" prefix (i.e. "openrouter/openrouter/free"). Don't strip
        # org names like "openrouter" in "openrouter/free".
        bare_id = (
            model_id[len("openrouter/") :]
            if model_id.startswith("openrouter/openrouter/")
            else model_id
        )
        model_str = f"openrouter/{bare_id}"
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["api_base"] = base_url
    elif provider == "generic":
        model_str = f"openai/{model_id}"
        stripped = base_url.rstrip("/")
        api_base = stripped if stripped.endswith("/v1") else f"{stripped}/v1"
        kwargs = {"api_base": api_base, "api_key": api_key or "none"}
    else:
        raise AgentError(f"unknown provider {provider!r}")

    if verbose:
        extras = []
        if temperature is not None:
            extras.append(f"temperature={temperature}")
        if top_p is not None:
            extras.append(f"top_p={top_p}")
        if seed is not None:
            extras.append(f"seed={seed}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        fmt.model_info(
            f"Calling model {model_str} with max_tokens={max_output_tokens}{extra_str}"
        )

    completion_kwargs = dict(
        model=model_str,
        messages=messages,
        max_tokens=max_output_tokens,
        **kwargs,
    )
    if tools is not None:
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = "auto"
    for key, val in [("temperature", temperature), ("top_p", top_p), ("seed", seed)]:
        if val is not None:
            completion_kwargs[key] = val
    if extra_body is not None:
        completion_kwargs["extra_body"] = extra_body

    try:
        response = litellm.completion(**completion_kwargs)
    except litellm.ContextWindowExceededError:
        raise ContextOverflowError("context window exceeded (typed)")
    except litellm.BadRequestError as e:
        msg_text = str(e)
        if _CONTEXT_OVERFLOW_RE.search(msg_text):
            raise ContextOverflowError(f"context window exceeded (inferred): {e}")
        if _EMPTY_ASSISTANT_RE.search(msg_text):
            # Provider rejected an assistant message with no content and no
            # tool_calls (common with Mistral via OpenRouter).  Fix the
            # messages in place and retry once.
            if _sanitize_assistant_messages(messages):
                if verbose:
                    fmt.warning("Fixed empty assistant message in history, retrying...")
                try:
                    response = litellm.completion(**completion_kwargs)
                except Exception as e2:
                    raise AgentError(
                        f"LLM call failed after message sanitization: {e2}"
                    )
                choice = response.choices[0]
                return choice.message, choice.finish_reason
        raise AgentError(f"LLM call failed: {e}")
    except Exception as e:
        raise AgentError(f"LLM call failed: {e}")

    choice = response.choices[0]
    return choice.message, choice.finish_reason


def run_reviewer(
    reviewer_cmd: str,
    base_dir: str,
    answer: str,
    verbose: bool,
    timeout: int = 3600,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run the reviewer executable.

    Returns (exit_code, stdout_text, stderr_text).
    Never raises — all failures return (2, "", "") with a warning on stderr.
    """
    import shlex

    argv = shlex.split(reviewer_cmd) + [base_dir]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            argv,
            input=answer.encode(),
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        if verbose:
            fmt.warning(f"reviewer timed out after {timeout}s, accepting answer as-is")
        return 2, "", ""
    except OSError as e:
        if verbose:
            fmt.warning(f"reviewer failed to run: {e}")
        return 2, "", ""
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if stderr and verbose:
        fmt.warning(f"reviewer stderr: {stderr.rstrip()}")
    return proc.returncode, stdout, stderr


def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="swival",
        usage="%(prog)s [options] <question>\n       %(prog)s --repl [options] [question]",
        description="A CLI coding agent with tool-calling, sandboxed file access, and multi-provider LLM support.",
    )
    parser.add_argument(
        "question", nargs="?", default=None, help="The question or task for the model."
    )
    parser.add_argument(
        "--add-dir",
        type=str,
        action="append",
        default=None,
        help="Grant read/write access to an extra directory (repeatable).",
    )
    parser.add_argument(
        "--add-dir-ro",
        type=str,
        action="append",
        default=None,
        help="Grant read-only access to an extra directory (repeatable).",
    )
    parser.add_argument(
        "--allowed-commands",
        type=str,
        default=_UNSET,
        help='Comma-separated list of allowed command basenames (e.g. "ls,git,python3").',
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=_UNSET,
        help="API key for the provider (overrides env var).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for file tools (default: current directory).",
    )
    parser.add_argument(
        "--base-url",
        default=_UNSET,
        help="Server base URL (default: http://127.0.0.1:1234 for lmstudio).",
    )

    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        action="store_true",
        default=_UNSET,
        help="Force ANSI color even when stderr is not a TTY.",
    )
    color_group.add_argument(
        "--no-color",
        action="store_true",
        default=_UNSET,
        help="Disable ANSI color even when stderr is a TTY.",
    )

    def _parse_extra_body(value):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("--extra-body must be a JSON object")
        return parsed

    parser.add_argument(
        "--extra-body",
        type=_parse_extra_body,
        default=_UNSET,
        metavar="JSON",
        help='Extra parameters to pass to the LLM API as JSON (e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\').',
    )

    parser.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Generate a config file template and exit.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=_UNSET,
        help="Requested context length for the model (may trigger a reload).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=_UNSET,
        help="Maximum output tokens (default: 32768).",
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=_UNSET,
        help="Maximum number of reviewer retry rounds (default: 5). 0 disables retries.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_UNSET,
        help="Maximum agent loop iterations (default: 100).",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to an MCP JSON config file (replaces .mcp.json default lookup).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_UNSET,
        help="Override auto-discovered model with a specific model identifier.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        default=_UNSET,
        help="Don't write responses to .swival/HISTORY.md",
    )
    parser.add_argument(
        "--no-instructions",
        action="store_true",
        default=_UNSET,
        help="Don't load CLAUDE.md or AGENTS.md from the base directory or user config directory.",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        default=_UNSET,
        help="Disable MCP server connections entirely.",
    )
    parser.add_argument(
        "--no-read-guard",
        action="store_true",
        default=_UNSET,
        help="Disable read-before-write guard (allow writing files without reading them first).",
    )
    parser.add_argument(
        "--no-skills",
        action="store_true",
        default=_UNSET,
        help="Don't load or discover any skills.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=_UNSET,
        metavar="FILE",
        help="Read the task description from FILE instead of SWIVAL_TASK env var (reviewer mode).",
    )
    parser.add_argument(
        "--proactive-summaries",
        action="store_true",
        default=_UNSET,
        help="Periodically summarize conversation to preserve context across compaction events.",
    )
    parser.add_argument(
        "--project",
        action="store_true",
        default=False,
        help="With --init-config, write to <base-dir>/swival.toml instead of global config.",
    )
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "huggingface", "openrouter", "generic"],
        default=_UNSET,
        help="LLM provider: lmstudio (local), huggingface (HF API), openrouter (multi-provider API), generic (any OpenAI-compatible server).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=_UNSET,
        help="Suppress all diagnostics; only print the final result.",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start an interactive session instead of answering a single question.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="FILE",
        help="Write a JSON evaluation report to FILE. Incompatible with --repl.",
    )
    parser.add_argument(
        "--review-prompt",
        type=str,
        default=_UNSET,
        help="Custom instructions appended to the built-in review prompt (reviewer mode).",
    )
    parser.add_argument(
        "--reviewer",
        metavar="COMMAND",
        default=_UNSET,
        help="Reviewer command (shell-split). Called after each answer with base_dir as argument "
        "and answer on stdin. Exit 0=accept, 1=retry with stdout as feedback, 2=reviewer error.",
    )
    parser.add_argument(
        "--reviewer-mode",
        action="store_true",
        default=False,
        help="Run as a reviewer: read base_dir from positional arg, answer from stdin, "
        "call LLM to judge, exit 0/1/2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_UNSET,
        help="Random seed for reproducible outputs (optional, model support varies).",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=None,
        help="Additional directory to scan for skills (can be repeated).",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system-prompt",
        type=str,
        default=_UNSET,
        help="System prompt to include.",
    )
    prompt_group.add_argument(
        "--no-system-prompt",
        action="store_true",
        default=_UNSET,
        help="Omit the system message entirely.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=_UNSET,
        help="Sampling temperature (default: provider default).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=_UNSET,
        help="Top-p (nucleus) sampling (default: 1.0).",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=_UNSET,
        metavar="FILE",
        help="Read verification/acceptance criteria from FILE (reviewer mode).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=_UNSET,
        help="Disable filesystem sandbox and command whitelist (unrestricted mode).",
    )

    return parser


def _handle_init_config(args):
    """Generate a config file template and write it."""
    from .config import generate_config, global_config_dir

    project = getattr(args, "project", False)
    if project:
        base_dir = Path(getattr(args, "base_dir", ".")).resolve()
        dest = base_dir / "swival.toml"
    else:
        dest = global_config_dir() / "config.toml"

    if dest.exists():
        print(
            f"Error: {dest} already exists. Remove it first to regenerate.",
            file=sys.stderr,
        )
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(generate_config(project=project), encoding="utf-8")
    print(f"Created {dest}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle --version first
    if args.version:
        try:
            version = metadata.version("swival")
        except metadata.PackageNotFoundError:
            version = "unknown"
        print(version)
        sys.exit(0)

    # Handle --init-config before anything else
    if getattr(args, "init_config", False):
        _handle_init_config(args)
        sys.exit(0)

    # Load config files, apply to args, resolve sentinels to defaults
    from .config import load_config, apply_config_to_args
    from .config import ConfigError as _ConfigError

    # --- Reviewer mode: reinterpret positional arg as base_dir ---
    if args.reviewer_mode:
        if args.repl:
            parser.error("--reviewer-mode is incompatible with --repl")
        if args.question is None:
            parser.error("--reviewer-mode requires a positional argument (base_dir)")

        # Snapshot whether --reviewer was explicitly on CLI (before config merge)
        reviewer_from_cli = args.reviewer is not _UNSET

        base_dir = Path(args.question).resolve()
        try:
            file_config = load_config(base_dir)
        except _ConfigError as e:
            parser.error(str(e))
        apply_config_to_args(args, file_config)

        # Config inheritance hazard: clear the reviewer key
        if reviewer_from_cli:
            parser.error("--reviewer-mode and --reviewer cannot be used together")
        args.reviewer = None

        args.verbose = not args.quiet
        fmt.init(color=args.color, no_color=args.no_color)

        from .reviewer import run_as_reviewer

        sys.exit(run_as_reviewer(args, str(base_dir)))

    base_dir = Path(args.base_dir).resolve()
    try:
        file_config = load_config(base_dir)
    except _ConfigError as e:
        parser.error(str(e))

    # Stash MCP servers from TOML config before apply_config_to_args strips them
    args._mcp_servers_toml = file_config.pop("mcp_servers", None)

    apply_config_to_args(args, file_config)

    # Derived values (after all sentinels are resolved)
    args.verbose = not args.quiet

    if not args.repl and args.question is None:
        parser.error("question is required (or use --repl)")
    if args.report and args.repl:
        parser.error("--report is incompatible with --repl")
    if args.reviewer and args.repl:
        parser.error("--reviewer is incompatible with --repl")

    fmt.init(color=args.color, no_color=args.no_color)

    # Validation: max_review_rounds >= 0
    if args.max_review_rounds < 0:
        parser.error("--max-review-rounds must be >= 0")

    # Validation: max_output_tokens <= max_context_tokens
    if (
        args.max_context_tokens is not None
        and args.max_output_tokens > args.max_context_tokens
    ):
        parser.error(
            "--max-output-tokens must be <= --max-context-tokens when both are specified."
        )

    report = ReportCollector() if args.report else None

    # Helper to build the settings dict for the report
    def _report_settings(
        model_id="unknown", skills_catalog=None, instructions_loaded=None
    ):
        return {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "context_length": getattr(
                args, "_resolved_context_length", args.max_context_tokens
            ),
            "yolo": args.yolo,
            "allowed_commands": (
                sorted(args.allowed_commands)
                if isinstance(args.allowed_commands, list)
                else sorted(
                    c.strip()
                    for c in (args.allowed_commands or "").split(",")
                    if c.strip()
                )
            ),
            "max_review_rounds": args.max_review_rounds,
            "skills_discovered": sorted(skills_catalog or {}),
            "instructions_loaded": instructions_loaded or [],
        }

    def _write_report(
        outcome,
        answer=None,
        exit_code=0,
        turns=None,
        error_message=None,
        model_id="unknown",
        skills_catalog=None,
        instructions_loaded=None,
        review_rounds=0,
        todo_state=None,
        snapshot_state=None,
    ):
        if not report:
            return
        effective_turns = turns if turns is not None else report.max_turn_seen
        todo_stats = None
        if todo_state is not None and todo_state._total_actions > 0:
            remaining = todo_state.remaining_count
            todo_stats = {
                "added": todo_state.add_count,
                "completed": todo_state.done_count,
                "remaining": remaining,
            }
        snapshot_stats = None
        if snapshot_state is not None:
            total = snapshot_state.stats["restores"] + snapshot_state.stats["saves"]
            if total > 0:
                snapshot_stats = dict(snapshot_state.stats)
        report.finalize(
            task=args.question or "",
            model=model_id,
            provider=args.provider,
            settings=_report_settings(model_id, skills_catalog, instructions_loaded),
            outcome=outcome,
            answer=answer,
            exit_code=exit_code,
            turns=effective_turns,
            error_message=error_message,
            review_rounds=review_rounds,
            todo_stats=todo_stats,
            snapshot_stats=snapshot_stats,
        )
        try:
            report.write(args.report)
        except OSError as e:
            fmt.error(f"Failed to write report to {args.report}: {e}")
            return
        if args.verbose:
            fmt.info(f"Report written to {args.report}")

    try:
        _run_main(args, report, _write_report, parser)
    except AgentError as e:
        fmt.error(str(e))
        _write_report(
            "error",
            exit_code=1,
            error_message=str(e),
            model_id=getattr(args, "_resolved_model_id", args.model or "unknown"),
            skills_catalog=getattr(args, "_resolved_skills", None),
            instructions_loaded=getattr(args, "_resolved_instructions", None),
            review_rounds=getattr(args, "_review_rounds", 0),
        )
        sys.exit(1)


def resolve_provider(
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    max_context_tokens: int | None,
    verbose: bool,
) -> tuple[str, str | None, str | None, int | None, dict]:
    """Validate provider args, discover model (LM Studio), return resolved config.

    Returns (model_id, api_base, api_key, context_length, llm_kwargs).
    Raises ConfigError for invalid configuration.
    """
    if provider == "lmstudio":
        api_base = base_url or "http://127.0.0.1:1234"
        if model:
            model_id = model
            current_context = None
            if verbose:
                fmt.model_info(f"Using user-specified model: {model_id}")
        else:
            model_id, current_context = discover_model(api_base, verbose)
            if not model_id:
                raise AgentError(
                    "no loaded LLM found in LM Studio. "
                    "Load a model in LM Studio or use --model to specify one."
                )
        if max_context_tokens is not None:
            configure_context(
                api_base,
                model_id,
                max_context_tokens,
                current_context,
                verbose,
            )
        context_length = max_context_tokens or current_context
        resolved_key = None

    elif provider == "huggingface":
        if not model:
            raise ConfigError("--model is required when --provider is huggingface")
        bare_model = model.removeprefix("huggingface/")
        if "/" not in bare_model:
            raise ConfigError(
                "HuggingFace model must be in org/model format (e.g. zai-org/GLM-5)"
            )
        api_base = base_url
        model_id = model
        context_length = max_context_tokens
        resolved_key = api_key or os.environ.get("HF_TOKEN")
        if not resolved_key:
            raise ConfigError(
                "--api-key or HF_TOKEN env var required for huggingface provider"
            )

    elif provider == "openrouter":
        if not model:
            raise ConfigError("--model is required when --provider is openrouter")
        api_base = base_url
        model_id = model
        context_length = max_context_tokens
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ConfigError(
                "--api-key or OPENROUTER_API_KEY env var required for openrouter provider"
            )
    elif provider == "generic":
        if not model:
            raise ConfigError("--model is required when --provider is generic")
        if not base_url:
            raise ConfigError("--base-url is required when --provider is generic")
        api_base = base_url
        model_id = model
        context_length = max_context_tokens
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")

    else:
        raise ConfigError(f"unknown provider: {provider!r}")

    llm_kwargs = {
        "provider": provider,
        "api_key": resolved_key,
    }
    return model_id, api_base, resolved_key, context_length, llm_kwargs


def resolve_commands(
    allowed_commands: str | list[str] | None,
    yolo: bool,
    base_dir: str,
) -> dict[str, str]:
    """Validate commands against PATH, reject commands inside workspace.

    Returns resolved_commands dict mapping name -> absolute path.
    In yolo mode, returns empty dict (any command can run).
    Raises ConfigError/AgentError for invalid commands.
    """
    if yolo:
        return {}

    if isinstance(allowed_commands, list):
        allowed_names = {c.strip() for c in allowed_commands if c.strip()}
    elif allowed_commands:
        allowed_names = {c.strip() for c in allowed_commands.split(",") if c.strip()}
    else:
        allowed_names = set()
    resolved_commands: dict[str, str] = {}
    base_resolved = Path(base_dir).resolve()
    for name in sorted(allowed_names):
        cmd_path = shutil.which(name)
        if cmd_path is None:
            raise ConfigError(f"allowed command {name!r} not found on PATH")
        abs_path = Path(cmd_path).resolve()
        if abs_path.is_relative_to(base_resolved):
            raise ConfigError(
                f"allowed command {name!r} resolves to {abs_path}, "
                f"which is inside base directory {base_resolved}. "
                f"Commands inside the workspace can be modified by the model."
            )
        resolved_commands[name] = str(abs_path)
    return resolved_commands


def build_tools(
    resolved_commands: dict[str, str],
    skills_catalog: dict,
    yolo: bool,
) -> list:
    """Construct the tools list from base + conditionals."""
    tools = list(TOOLS)
    if skills_catalog:
        tools.append(USE_SKILL_TOOL)
    if yolo:
        tool = copy.deepcopy(RUN_COMMAND_TOOL)
        tool["function"]["description"] = "Run any command and return its output."
        tool["function"]["parameters"]["properties"]["command"] = {
            "oneOf": [
                {
                    "type": "string",
                    "description": "Shell command string (executed via sh -c). Supports pipes, redirects, &&, etc.",
                },
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command as array of strings. Each argument is a separate element.",
                },
            ],
            "description": 'Command to run. Can be a shell string (e.g. "ls -la | head") or an array of strings (e.g. ["ls", "-la"]).',
        }
        tools.append(tool)
    elif resolved_commands:
        tool = copy.deepcopy(RUN_COMMAND_TOOL)
        tool["function"]["description"] = (
            f"Run a command and return its output. Allowed commands: {', '.join(sorted(resolved_commands))}."
        )
        tools.append(tool)
    return tools


def build_system_prompt(
    base_dir: str,
    system_prompt: str | None,
    no_system_prompt: bool,
    no_instructions: bool,
    skills_catalog: dict,
    yolo: bool,
    resolved_commands: dict[str, str],
    verbose: bool,
    config_dir: "Path | None" = None,
    mcp_tool_info: dict | None = None,
) -> tuple[str | None, list[str]]:
    """Assemble full system prompt with instructions, date, skills.

    Returns (system_prompt_text, instructions_loaded).
    system_prompt_text is None if no_system_prompt is True.
    """
    if no_system_prompt:
        return None, []

    instructions_loaded: list[str] = []
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        if not no_instructions:
            instructions, instructions_loaded = load_instructions(
                base_dir,
                config_dir,
                verbose=verbose,
            )
            if instructions:
                system_content += "\n\n" + instructions

    now = datetime.now().astimezone()
    system_content += f"\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M %Z')}"

    if skills_catalog and not system_prompt:
        from .skills import format_skill_catalog

        catalog_text = format_skill_catalog(skills_catalog)
        if catalog_text:
            system_content += "\n\n" + catalog_text
    if yolo:
        system_content += (
            "\n\n**Command execution tool:**\n"
            "- `run_command`: Run any command and return its output. "
            'Pass a shell string (e.g. `"ls -la | grep foo"`) or an array (e.g. `["ls", "-la"]`). '
            "Shell strings support pipes, redirects, `&&`, etc. "
            "Optional `timeout` (1-120s, default 30)."
        )
    elif resolved_commands:
        cmd_list = ", ".join(sorted(resolved_commands))
        system_content += (
            "\n\n**Command execution tool:**\n"
            f"- `run_command`: Run a whitelisted command and return its output. "
            f'Pass the command and arguments as a list (e.g. `["ls", "-la"]`). '
            f"Optional `timeout` (1-120s, default 30). "
            f"Allowed commands: {cmd_list}."
        )

    if mcp_tool_info and not system_prompt:
        system_content += "\n\n" + _format_mcp_tool_info(mcp_tool_info)

    return system_content, instructions_loaded


def _format_mcp_tool_info(tool_info: dict[str, list[tuple[str, str]]]) -> str:
    """Format MCP tool info for the system prompt."""
    lines = ["## MCP Tools", "", "Tools provided by external MCP servers:", ""]
    for server_name, tools in sorted(tool_info.items()):
        lines.append(f"**{server_name}**:")
        for namespaced_name, description in tools:
            desc = f": {description}" if description else ""
            lines.append(f"- `{namespaced_name}`{desc}")
        lines.append("")
    return "\n".join(lines)


def _run_main(args, report, _write_report, parser):
    # Provider-specific model discovery and context configuration
    try:
        model_id, api_base, api_key, context_length, llm_kwargs = resolve_provider(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_context_tokens=args.max_context_tokens,
            verbose=args.verbose,
        )
    except ConfigError as e:
        parser.error(str(e))
    if args.extra_body is not None:
        llm_kwargs["extra_body"] = args.extra_body

    # Stash resolved model_id for error reporting
    args._resolved_model_id = model_id

    # Resolve --add-dir paths
    allowed_dirs: list[Path] = []
    for d in getattr(args, "add_dir", []):
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            raise AgentError(f"--add-dir path is not a directory: {d}")
        if p == Path(p.anchor):
            raise AgentError(f"--add-dir cannot be the filesystem root: {d}")
        allowed_dirs.append(p)

    # Resolve --add-dir-ro paths
    allowed_dirs_ro: list[Path] = []
    for d in getattr(args, "add_dir_ro", []):
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            raise AgentError(f"--add-dir-ro path is not a directory: {d}")
        if p == Path(p.anchor):
            raise AgentError(f"--add-dir-ro cannot be the filesystem root: {d}")
        allowed_dirs_ro.append(p)

    base_dir = args.base_dir
    yolo = args.yolo

    resolved_commands = resolve_commands(args.allowed_commands, yolo, base_dir)

    # Discover skills
    from .skills import discover_skills

    skills_catalog: dict = {}
    skill_read_roots: list[Path] = list(allowed_dirs_ro)
    if not args.no_skills:
        skills_catalog = discover_skills(base_dir, args.skills_dir, args.verbose)
    args._resolved_skills = skills_catalog

    tools = build_tools(resolved_commands, skills_catalog, yolo)

    # Initialize MCP servers
    mcp_manager = None
    mcp_tool_info = {}
    if not getattr(args, "no_mcp", False):
        from .config import load_mcp_json, merge_mcp_configs
        from .mcp_client import McpManager

        toml_servers = getattr(args, "_mcp_servers_toml", None)
        json_servers = None

        mcp_config_path = getattr(args, "mcp_config", None)
        if mcp_config_path:
            p = Path(mcp_config_path)
            if not p.is_file():
                raise ConfigError(f"--mcp-config file not found: {mcp_config_path}")
            json_servers = load_mcp_json(p)
        else:
            # Default: look for .mcp.json in project root
            default_mcp = Path(base_dir).resolve() / ".mcp.json"
            if default_mcp.is_file():
                json_servers = load_mcp_json(default_mcp)

        mcp_servers = merge_mcp_configs(toml_servers, json_servers)
        if mcp_servers:
            mcp_manager = McpManager(mcp_servers, verbose=args.verbose)
            # start() connects to servers; individual connection failures
            # are logged and skipped (non-fatal), but ConfigError from
            # validation (bad names, collisions) propagates as fatal.
            mcp_manager.start()
            mcp_tools = mcp_manager.list_tools()
            if mcp_tools:
                tools.extend(mcp_tools)

            # Enforce token budget (may remove tools/servers)
            tools = enforce_mcp_token_budget(
                tools, mcp_manager, context_length, verbose=args.verbose
            )

            # Capture tool info AFTER pruning so prompt matches reality
            mcp_tool_info = mcp_manager.get_tool_info()

    system_content, instructions_loaded = build_system_prompt(
        base_dir=base_dir,
        system_prompt=args.system_prompt,
        no_system_prompt=args.no_system_prompt,
        no_instructions=args.no_instructions,
        skills_catalog=skills_catalog,
        yolo=yolo,
        resolved_commands=resolved_commands,
        verbose=args.verbose,
        config_dir=getattr(args, "config_dir", None),
        mcp_tool_info=mcp_tool_info,
    )
    messages = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    args._resolved_instructions = instructions_loaded
    args._resolved_context_length = context_length

    # Clean up stale cmd_output files from previous sessions
    removed = cleanup_old_cmd_outputs(base_dir)
    if removed and args.verbose:
        fmt.info(f"Cleaned up {removed} stale cmd_output file(s) from .swival/")

    import atexit

    atexit.register(cleanup_old_cmd_outputs, base_dir)

    thinking_state = ThinkingState(verbose=args.verbose)
    todo_state = TodoState(notes_dir=base_dir, verbose=args.verbose)
    snapshot_state = SnapshotState(verbose=args.verbose)
    file_tracker = (
        None if getattr(args, "no_read_guard", False) else FileAccessTracker()
    )

    loop_kwargs = dict(
        api_base=api_base,
        model_id=model_id,
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        context_length=context_length,
        base_dir=base_dir,
        thinking_state=thinking_state,
        todo_state=todo_state,
        snapshot_state=snapshot_state,
        resolved_commands=resolved_commands,
        skills_catalog=skills_catalog,
        skill_read_roots=skill_read_roots,
        extra_write_roots=allowed_dirs,
        yolo=yolo,
        verbose=args.verbose,
        llm_kwargs=llm_kwargs,
        file_tracker=file_tracker,
        mcp_manager=mcp_manager,
    )

    if getattr(args, "proactive_summaries", False):
        loop_kwargs["compaction_state"] = CompactionState()

    no_history = getattr(args, "no_history", False)

    # Validate reviewer executable at startup
    reviewer_cmd = None
    if args.reviewer:
        import shlex

        try:
            parts = shlex.split(args.reviewer)
        except ValueError as e:
            raise AgentError(f"malformed reviewer command: {e}")
        if not parts:
            raise AgentError("reviewer command is empty")
        exe = parts[0]
        resolved = shutil.which(exe)
        if resolved:
            reviewer_cmd = args.reviewer
        else:
            p = Path(exe).resolve()
            if p.is_file() and os.access(p, os.X_OK):
                reviewer_cmd = args.reviewer
            else:
                raise AgentError(
                    f"reviewer executable not found or not executable: {exe}"
                )

    if not args.repl:
        # Single-shot path
        messages.append({"role": "user", "content": args.question})
        review_round = 0
        turn_offset = 0

        # Build env vars for reviewer subprocess
        reviewer_env: dict[str, str] | None = None
        if reviewer_cmd:
            reviewer_env = {"SWIVAL_TASK": args.question}
            model_id = getattr(args, "_resolved_model_id", None)
            if model_id:
                reviewer_env["SWIVAL_MODEL"] = model_id

        while True:
            answer, exhausted = run_agent_loop(
                messages,
                tools,
                **loop_kwargs,
                report=report,
                turn_offset=turn_offset,
            )

            if not reviewer_cmd or answer is None or exhausted:
                break

            review_round += 1
            args._review_rounds = review_round
            if args.verbose:
                fmt.info(f"Review round {review_round}: sending answer to reviewer")

            reviewer_env["SWIVAL_REVIEW_ROUND"] = str(review_round)
            exit_code, review_text, review_stderr = run_reviewer(
                reviewer_cmd,
                base_dir,
                answer,
                args.verbose,
                env_extra=reviewer_env,
            )

            if report:
                report.record_review(
                    review_round, exit_code, review_text, stderr=review_stderr
                )

            if exit_code == 0:
                if args.verbose:
                    fmt.review_accepted(review_round)
                break
            elif exit_code == 1:
                if review_round >= args.max_review_rounds:
                    if args.verbose:
                        fmt.warning(
                            f"Max review rounds ({args.max_review_rounds}) reached, accepting answer"
                        )
                    break
                if args.verbose:
                    fmt.review_feedback(review_round, review_text)
                messages.append({"role": "user", "content": review_text})
                if report:
                    turn_offset = report.max_turn_seen
                loop_kwargs["max_turns"] = args.max_turns
                continue
            else:
                if args.verbose:
                    fmt.warning(
                        f"Reviewer exited with code {exit_code}, accepting answer as-is"
                    )
                break

        if not no_history and answer:
            append_history(base_dir, args.question, answer, diagnostics=args.verbose)
        if answer is not None:
            print(answer)
        if report:
            _write_report(
                "exhausted" if exhausted else "success",
                answer=answer,
                exit_code=2 if exhausted else 0,
                turns=report.max_turn_seen,
                model_id=model_id,
                skills_catalog=skills_catalog,
                instructions_loaded=instructions_loaded,
                review_rounds=review_round,
                todo_state=todo_state,
                snapshot_state=snapshot_state,
            )
        if exhausted:
            if args.verbose:
                fmt.warning("max turns reached, agent stopped.")
            sys.exit(2)
        return

    # REPL path
    if args.question:
        messages.append({"role": "user", "content": args.question})
        answer, exhausted = run_agent_loop(messages, tools, **loop_kwargs)
        if not no_history and answer:
            append_history(base_dir, args.question, answer, diagnostics=args.verbose)
        if answer is not None:
            print(answer)
        if exhausted and args.verbose:
            fmt.warning(
                "max turns reached for initial question. Use /continue to resume."
            )

    repl_loop(messages, tools, **loop_kwargs, no_history=no_history)


def run_agent_loop(
    messages: list,
    tools: list,
    *,
    api_base: str,
    model_id: str,
    max_turns: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    context_length: int | None,
    base_dir: str,
    thinking_state: ThinkingState,
    todo_state: TodoState,
    snapshot_state: SnapshotState | None = None,
    resolved_commands: dict,
    skills_catalog: dict,
    skill_read_roots: list,
    extra_write_roots: list,
    yolo: bool,
    verbose: bool,
    llm_kwargs: dict,
    file_tracker: FileAccessTracker | None = None,
    report: ReportCollector | None = None,
    turn_offset: int = 0,
    compaction_state: "CompactionState | None" = None,
    mcp_manager=None,
) -> tuple[str | None, bool]:
    """Run the tool-calling loop until a final answer or max turns.

    Mutates `messages` in place (appends assistant/tool messages,
    in-place compaction on overflow).
    Returns (final_answer, exhausted). final_answer is the last
    assistant text (may be None). exhausted is True if max_turns hit.
    """
    consecutive_errors: dict[str, tuple[str, int]] = {}
    turns = 0
    think_used = False
    think_nudge_fired = False
    todo_last_used = 0
    snapshot_read_streak = 0
    snapshot_nudge_fired = False

    # Reset dirty state only if the last message is a user message
    # (new scope boundary). Skip on /continue where the last message
    # is an assistant or tool message from the previous run.
    if snapshot_state is not None:
        last_role = _msg_role(messages[-1]) if messages else ""
        if last_role == "user":
            snapshot_state.reset_dirty()

    _snapshot_strip_marker = "\n\n" + SNAPSHOT_HISTORY_SENTINEL

    while turns < max_turns:
        turns += 1

        # Inject snapshot history into system message so the LLM
        # can see prior investigation summaries even after compaction.
        # Always strip any prior injection first (handles re-entry
        # via /continue or repeated run_agent_loop calls).
        if snapshot_state is not None and messages:
            sys_msg = messages[0] if _msg_role(messages[0]) == "system" else None
            if sys_msg is not None and isinstance(sys_msg, dict):
                base = sys_msg["content"]
                idx = base.find(_snapshot_strip_marker)
                if idx != -1:
                    base = base[:idx]
                history_text = snapshot_state.inject_into_prompt()
                if history_text:
                    sys_msg["content"] = base + "\n\n" + history_text
                else:
                    sys_msg["content"] = base

        token_est = estimate_tokens(messages, tools)
        if verbose:
            fmt.turn_header(turns, max_turns, token_est)

        t0 = time.monotonic()
        try:
            effective_max_output = clamp_output_tokens(
                messages, tools, context_length, max_output_tokens
            )
            if effective_max_output != max_output_tokens and verbose:
                fmt.info(
                    f"Output tokens: {effective_max_output} (clamped, context_length={context_length}, prompt=~{token_est})"
                )

            _llm_args = (
                api_base,
                model_id,
                messages,
                effective_max_output,
                temperature,
                top_p,
                seed,
                tools,
                verbose,
            )

            with (
                fmt.llm_spinner(f"Waiting for LLM (turn {turns}/{max_turns})")
                if verbose
                else nullcontext()
            ):
                msg, finish_reason = call_llm(*_llm_args, **llm_kwargs)
        except ContextOverflowError:
            elapsed = time.monotonic() - t0
            if report:
                report.record_llm_call(
                    turns + turn_offset, elapsed, token_est, "context_overflow"
                )

            # --- Graduated compaction levels ---
            # Each level is tried in order. If the LLM call succeeds after
            # a compaction step, we break out. If it still overflows, we
            # try the next level. If all levels fail, raise AgentError.
            _llm_summary_kwargs = dict(
                call_llm_fn=call_llm,
                model_id=model_id,
                base_url=api_base,
                api_key=llm_kwargs.get("api_key"),
                top_p=top_p,
                seed=seed,
                provider=llm_kwargs.get("provider"),
                compaction_state=compaction_state,
            )
            compaction_levels = [
                (
                    "compact_messages",
                    "compacting tool results...",
                    lambda: compact_messages(messages),
                ),
                (
                    "drop_middle_turns",
                    "dropping low-importance turns...",
                    lambda: drop_middle_turns(messages, **_llm_summary_kwargs),
                ),
                (
                    "aggressive_drop",
                    "aggressive compaction (last resort)...",
                    lambda: aggressive_drop_turns(messages, **_llm_summary_kwargs),
                ),
            ]

            for level_name, level_desc, compact_fn in compaction_levels:
                if verbose:
                    fmt.warning(f"context window exceeded, {level_desc}")
                tokens_before = estimate_tokens(messages, tools)
                messages[:] = compact_fn()
                effective_max_output = clamp_output_tokens(
                    messages, tools, context_length, max_output_tokens
                )
                tokens_after = estimate_tokens(messages, tools)
                if report:
                    report.record_compaction(
                        turns + turn_offset, level_name, tokens_before, tokens_after
                    )
                if verbose:
                    fmt.context_stats(f"Context after {level_name}", tokens_after)

                _llm_args = (
                    api_base,
                    model_id,
                    messages,
                    effective_max_output,
                    temperature,
                    top_p,
                    seed,
                    tools,
                    verbose,
                )
                t0 = time.monotonic()
                try:
                    with (
                        fmt.llm_spinner(
                            f"Waiting for LLM (turn {turns}/{max_turns}, retry after compaction)"
                        )
                        if verbose
                        else nullcontext()
                    ):
                        msg, finish_reason = call_llm(*_llm_args, **llm_kwargs)
                except ContextOverflowError:
                    elapsed = time.monotonic() - t0
                    if report:
                        report.record_llm_call(
                            turns + turn_offset,
                            elapsed,
                            tokens_after,
                            "context_overflow",
                            is_retry=True,
                            retry_reason=level_name,
                        )
                    continue  # try next level
                except AgentError:
                    elapsed = time.monotonic() - t0
                    if report:
                        report.record_llm_call(
                            turns + turn_offset,
                            elapsed,
                            tokens_after,
                            "error",
                            is_retry=True,
                            retry_reason=level_name,
                        )
                    raise
                else:
                    elapsed = time.monotonic() - t0
                    if verbose:
                        fmt.llm_timing(elapsed, finish_reason)
                    if report:
                        report.record_llm_call(
                            turns + turn_offset,
                            elapsed,
                            tokens_after,
                            finish_reason,
                            is_retry=True,
                            retry_reason=level_name,
                        )
                    break  # success
            else:
                # All compaction levels exhausted
                raise AgentError("context window exceeded even after compaction")

        except AgentError:
            elapsed = time.monotonic() - t0
            if report:
                report.record_llm_call(turns + turn_offset, elapsed, token_est, "error")
            raise
        else:
            elapsed = time.monotonic() - t0
            if verbose:
                fmt.llm_timing(elapsed, finish_reason)
            if report:
                report.record_llm_call(
                    turns + turn_offset, elapsed, token_est, finish_reason
                )
        # Handle empty assistant response (no content, no tool_calls).
        # Some providers return these occasionally; appending them as-is
        # would poison the history and cause BadRequestError on the next call.
        if not getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
            if verbose:
                fmt.warning("LLM returned empty response, requesting continuation...")
            # Give the message minimal content so it's valid in history
            msg.content = ""
            messages.append(msg)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response was empty. Please continue working on "
                        "the task using the available tools."
                    ),
                }
            )
            continue

        messages.append(msg)

        # Log intermediate assistant text (reasoning before tool calls, or truncated responses)
        if msg.content and (msg.tool_calls or finish_reason == "length") and verbose:
            fmt.assistant_text(msg.content)

        if not msg.tool_calls:
            if finish_reason == "length":
                # Output was truncated before the model could finish;
                # nudge it to continue using tools instead of quitting.
                if report:
                    report.record_truncated_response(turns + turn_offset)
                if verbose:
                    fmt.info(
                        "Response truncated (finish_reason=length), prompting continuation."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "Your response was cut off. Please use the provided tools to complete the task step by step.",
                    }
                )
                continue
            # Model produced a final text answer
            if verbose:
                fmt.completion(turns, "ok")
            if verbose:
                summary = thinking_state.summary_line()
                if summary:
                    fmt.think_summary(summary)
                if todo_state:
                    summary = todo_state.summary_line()
                    if summary:
                        fmt.todo_summary(summary)
                if snapshot_state:
                    summary = snapshot_state.summary_line()
                    if summary:
                        fmt.info(summary)
            return msg.content or "", False

        interventions: list[str] = []
        all_tools_readonly = True
        for tool_call in msg.tool_calls:
            tool_msg, tool_meta = handle_tool_call(
                tool_call,
                base_dir,
                thinking_state,
                verbose,
                resolved_commands=resolved_commands,
                skills_catalog=skills_catalog,
                skill_read_roots=skill_read_roots,
                extra_write_roots=extra_write_roots,
                yolo=yolo,
                file_tracker=file_tracker,
                todo_state=todo_state,
                snapshot_state=snapshot_state,
                mcp_manager=mcp_manager,
                messages=messages,
            )
            messages.append(tool_msg)

            if report:
                report.record_tool_call(
                    turns + turn_offset,
                    tool_meta["name"],
                    tool_meta["arguments"],
                    tool_meta["succeeded"],
                    tool_meta["elapsed"],
                    len(tool_msg["content"]),
                    error=tool_msg["content"] if not tool_meta["succeeded"] else None,
                )

            tool_name = tool_meta["name"]
            if tool_name == "think":
                think_used = True
            if tool_name == "todo":
                todo_last_used = turns

            if snapshot_state is not None:
                if tool_name not in READ_ONLY_TOOLS:
                    snapshot_state.mark_dirty(tool_name)
                    all_tools_readonly = False

            result = tool_msg["content"]
            if result.startswith("error:"):
                canonical_error = _canonical_error(result)
                prev_error, prev_count = consecutive_errors.get(tool_name, ("", 0))
                if canonical_error == prev_error:
                    count = prev_count + 1
                else:
                    count = 1
                consecutive_errors[tool_name] = (canonical_error, count)

                if count >= 2:
                    if count >= 3:
                        level = "stop"
                        interventions.append(
                            f"STOP: You have failed to use `{tool_name}` correctly {count} times in a row "
                            "with the same error. Do NOT call "
                            f"`{tool_name}` again with the same arguments. "
                            "Either fix the arguments or use a completely different approach to accomplish your task."
                        )
                    else:
                        level = "nudge"
                        interventions.append(
                            f"IMPORTANT: You have called `{tool_name}` {count} times with the same error. "
                            f"The error is: {canonical_error}\n"
                            "Please carefully re-read the error message and fix your tool call. "
                            "If you cannot use this tool correctly, use a different approach."
                        )
                    if report:
                        report.record_guardrail(turns + turn_offset, tool_name, level)
                    if verbose:
                        fmt.guardrail(tool_name, count, canonical_error)
            else:
                consecutive_errors.pop(tool_name, None)
        # Think nudge: if model used edit_file/write_file without thinking first
        if not think_used and not think_nudge_fired:
            has_mutating = any(
                tc.function.name in ("edit_file", "write_file", "delete_file")
                for tc in msg.tool_calls
            )
            if has_mutating:
                think_nudge_fired = True
                interventions.append(
                    "Tip: Consider using the `think` tool before making edits. "
                    "Planning your approach first leads to better outcomes."
                )

        # Todo reminder: nudge when items remain and todo hasn't been used recently.
        if todo_state is not None:
            remaining = todo_state.remaining_count
            if remaining > 0 and (turns - todo_last_used) >= TODO_REMINDER_INTERVAL:
                todo_last_used = turns  # reset so we don't nag every turn
                items_preview = "; ".join(
                    i.text[:60] for i in todo_state.items if not i.done
                )[:200]
                interventions.append(
                    f"Reminder: You have {remaining} unfinished todo item(s): {items_preview}. "
                    "Use the `todo` tool to review and work through them."
                )

        if snapshot_state is not None:
            if all_tools_readonly:
                snapshot_read_streak += 1
                if snapshot_read_streak >= 5 and not snapshot_nudge_fired:
                    snapshot_nudge_fired = True
                    interventions.append(
                        "Tip: You've done a lot of reading. Consider calling "
                        '`snapshot restore summary="..."` to collapse your '
                        "investigation into a summary and free context."
                    )
            else:
                snapshot_read_streak = 0
                snapshot_nudge_fired = False

        if interventions:
            messages.append({"role": "user", "content": "\n\n".join(interventions)})
        if verbose:
            fmt.context_stats(
                f"Context after turn {turns}", estimate_tokens(messages, tools)
            )

        # Proactive checkpoint (if enabled)
        if compaction_state is not None:
            compaction_state.maybe_checkpoint(
                messages,
                call_llm,
                model_id=model_id,
                base_url=api_base,
                api_key=llm_kwargs.get("api_key"),
                top_p=top_p,
                seed=seed,
                provider=llm_kwargs.get("provider"),
            )

    # max_turns exhausted — extract last assistant text
    if verbose:
        fmt.completion(turns, "max_turns")
    last_text = None
    for m in reversed(messages):
        if _msg_role(m) == "assistant":
            content = _msg_content(m)
            if content:
                last_text = content
                break
    if verbose:
        summary = thinking_state.summary_line()
        if summary:
            fmt.think_summary(summary)
        if todo_state:
            summary = todo_state.summary_line()
            if summary:
                fmt.todo_summary(summary)
        if snapshot_state:
            summary = snapshot_state.summary_line()
            if summary:
                fmt.info(summary)
    return last_text, True


# ---------------------------------------------------------------------------
# REPL command helpers
# ---------------------------------------------------------------------------


def _repl_help() -> None:
    """Print available REPL commands."""
    fmt.info(
        "Available commands:\n"
        "  /help              Show this help message\n"
        "  /clear             Reset conversation to initial state\n"
        "  /compact [--drop]  Compress context (--drop removes middle turns)\n"
        "  /add-dir <path>    Grant read+write access to a directory\n"
        "  /add-dir-ro <path> Grant read-only access to a directory\n"
        "  /extend [N]        Double max turns, or set to N\n"
        "  /continue          Reset turn counter and continue the agent loop\n"
        "  /init              Generate AGENTS.md for the current project\n"
        "  /exit, /quit       Exit the REPL"
    )


def _repl_clear(
    messages: list,
    thinking_state: ThinkingState,
    file_tracker: FileAccessTracker | None = None,
    todo_state: TodoState | None = None,
    snapshot_state: SnapshotState | None = None,
) -> None:
    """Clear conversation history, keeping only the leading system messages."""
    leading = []
    for msg in messages:
        if _msg_role(msg) == "system":
            leading.append(msg)
        else:
            break

    dropped = len(messages) - len(leading)
    messages[:] = leading

    # Fully reset ThinkingState
    thinking_state.history.clear()
    thinking_state.branches.clear()
    thinking_state.think_calls = 0

    if file_tracker is not None:
        file_tracker.reset()

    if todo_state is not None:
        todo_state.reset()

    if snapshot_state is not None:
        snapshot_state.reset()

    fmt.info(f"context cleared ({dropped} messages removed)")


def _repl_add_dir(path_str: str, extra_write_roots: list) -> None:
    """Add a directory to the write-access whitelist."""
    path_str = path_str.strip()
    if not path_str:
        fmt.warning("/add-dir requires a path argument")
        return

    p = Path(path_str).expanduser().resolve()
    if not p.is_dir():
        fmt.warning(f"not a directory: {path_str}")
        return
    if p == Path(p.anchor):
        fmt.warning("cannot add filesystem root")
        return
    if p in extra_write_roots:
        fmt.info(f"already in whitelist: {p}")
        return

    extra_write_roots.append(p)
    fmt.info(f"added to whitelist: {p}")


def _repl_add_dir_ro(path_str: str, skill_read_roots: list) -> None:
    """Add a directory to the read-only whitelist."""
    path_str = path_str.strip()
    if not path_str:
        fmt.warning("/add-dir-ro requires a path argument")
        return

    p = Path(path_str).expanduser().resolve()
    if not p.is_dir():
        fmt.warning(f"not a directory: {path_str}")
        return
    if p == Path(p.anchor):
        fmt.warning("cannot add filesystem root")
        return
    if p in skill_read_roots:
        fmt.info(f"already in read-only whitelist: {p}")
        return

    skill_read_roots.append(p)
    fmt.info(f"added to read-only whitelist: {p}")


def _repl_compact(
    messages: list, tools: list, context_length: int | None, arg: str
) -> None:
    """Manually compact conversation context."""
    before = estimate_tokens(messages, tools)

    messages[:] = compact_messages(messages)
    if arg.strip() == "--drop":
        messages[:] = drop_middle_turns(messages)

    after = estimate_tokens(messages, tools)
    saved = before - after
    fmt.info(f"compacted: {before} -> {after} tokens ({saved} saved)")


def _repl_extend(arg: str, state: dict) -> None:
    """Double max turns (default) or set to a specific value."""
    arg = arg.strip()
    if arg:
        try:
            n = int(arg)
        except ValueError:
            fmt.warning(f"invalid number: {arg}")
            return
        if n < 1:
            fmt.warning("max turns must be at least 1")
            return
        state["max_turns"] = n
        fmt.info(f"max turns set to {n}")
    else:
        old = state["max_turns"]
        state["max_turns"] = old * 2
        fmt.info(f"max turns doubled: {old} -> {old * 2}")


def repl_loop(
    messages: list,
    tools: list,
    *,
    api_base: str,
    model_id: str,
    max_turns: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    context_length: int | None,
    base_dir: str,
    thinking_state: ThinkingState,
    todo_state: TodoState,
    snapshot_state: SnapshotState | None = None,
    resolved_commands: dict,
    skills_catalog: dict,
    skill_read_roots: list,
    extra_write_roots: list,
    yolo: bool,
    verbose: bool,
    llm_kwargs: dict,
    file_tracker: FileAccessTracker | None = None,
    no_history: bool = False,
    compaction_state: "CompactionState | None" = None,
    mcp_manager=None,
) -> None:
    """Interactive read-eval-print loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory

    history_path = os.path.join(base_dir, ".swival", "repl_history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(
        history=FileHistory(history_path),
        enable_history_search=True,
    )
    prompt_text = FormattedText([("bold fg:ansigreen", "swival> ")])

    if verbose:
        fmt.repl_banner()

    turn_state = {"max_turns": max_turns}
    _repl_loop_kwargs = dict(
        api_base=api_base,
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        context_length=context_length,
        base_dir=base_dir,
        thinking_state=thinking_state,
        todo_state=todo_state,
        snapshot_state=snapshot_state,
        resolved_commands=resolved_commands,
        skills_catalog=skills_catalog,
        skill_read_roots=skill_read_roots,
        extra_write_roots=extra_write_roots,
        yolo=yolo,
        verbose=verbose,
        llm_kwargs=llm_kwargs,
        file_tracker=file_tracker,
        compaction_state=compaction_state,
        mcp_manager=mcp_manager,
    )

    while True:
        try:
            print(file=sys.stderr)  # blank line before prompt
            line = session.prompt(prompt_text)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)  # newline after ^D / ^C
            break

        line = line.strip()
        if not line:
            continue

        # REPL commands — only intercept known commands; unknown /foo passes through
        if line in ("/exit", "/quit"):
            break

        cmd_parts = line.split(None, 1)
        cmd = cmd_parts[0].lower()
        cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd == "/help":
            _repl_help()
            continue
        elif cmd == "/clear":
            _repl_clear(
                messages,
                thinking_state,
                file_tracker=file_tracker,
                todo_state=todo_state,
                snapshot_state=snapshot_state,
            )
            continue
        elif cmd == "/add-dir":
            _repl_add_dir(cmd_arg, extra_write_roots)
            continue
        elif cmd == "/add-dir-ro":
            _repl_add_dir_ro(cmd_arg, skill_read_roots)
            continue
        elif cmd == "/compact":
            _repl_compact(messages, tools, context_length, cmd_arg)
            continue
        elif cmd == "/extend":
            _repl_extend(cmd_arg, turn_state)
            continue
        elif cmd == "/init":
            if cmd_arg:
                fmt.warning(f"/init takes no arguments, ignoring {cmd_arg!r}")
            # Clear conversation history first to start with a clean context
            _repl_clear(
                messages,
                thinking_state,
                file_tracker=file_tracker,
                todo_state=todo_state,
                snapshot_state=snapshot_state,
            )
            # Three-pass init: explore, enrich, then write to file
            for _pass, prompt in enumerate(
                (INIT_PROMPT, INIT_ENRICH_PROMPT, INIT_WRITE_PROMPT), 1
            ):
                messages.append({"role": "user", "content": prompt})
                try:
                    answer, exhausted = run_agent_loop(
                        messages,
                        tools,
                        max_turns=turn_state["max_turns"],
                        **_repl_loop_kwargs,
                    )
                except KeyboardInterrupt:
                    fmt.warning("interrupted, /init aborted.")
                    break
                if not no_history and answer:
                    append_history(
                        base_dir,
                        f"/init pass {_pass}",
                        answer,
                        diagnostics=verbose,
                    )
                if answer is not None:
                    print(answer)
                if exhausted and verbose:
                    fmt.warning(f"max turns reached during /init pass {_pass}.")
            continue
        elif cmd == "/continue":
            fmt.info("continuing agent loop...")
            try:
                answer, exhausted = run_agent_loop(
                    messages,
                    tools,
                    max_turns=turn_state["max_turns"],
                    **_repl_loop_kwargs,
                )
            except KeyboardInterrupt:
                fmt.warning("interrupted, continuation aborted.")
                continue
            if not no_history and answer:
                append_history(base_dir, "(continued)", answer, diagnostics=verbose)
            if answer is not None:
                print(answer)
            if exhausted and verbose:
                fmt.warning(
                    "max turns reached for this question. Use /continue to resume."
                )
            continue

        messages.append({"role": "user", "content": line})
        try:
            answer, exhausted = run_agent_loop(
                messages,
                tools,
                max_turns=turn_state["max_turns"],
                **_repl_loop_kwargs,
            )
        except KeyboardInterrupt:
            fmt.warning("interrupted, question aborted.")
            continue

        if not no_history and answer:
            append_history(base_dir, line, answer, diagnostics=verbose)
        if answer is not None:
            print(answer)
        if exhausted and verbose:
            fmt.warning("max turns reached for this question. Use /continue to resume.")


if __name__ == "__main__":
    main()
