import argparse
from contextlib import nullcontext
import copy
from datetime import datetime
import json
from typing import Literal
import os
import platform
import random
import re
import shutil
import subprocess
import shlex
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from importlib import metadata

import tiktoken

from . import fmt
from ._msg import (
    IMAGE_TOKEN_ESTIMATE as _IMAGE_TOKEN_ESTIMATE,
    _has_image_content,
    _msg_get,
    _msg_role,
    _msg_content,
    _msg_tool_calls,
    _msg_tool_call_id,
    _msg_name,
    _set_msg_content,
)
from .config import _UNSET
from .report import AgentError, ConfigError, ContextOverflowError, ReportCollector
from .snapshot import (
    SNAPSHOT_HISTORY_SENTINEL,
    SNAPSHOT_RECAP_PREFIX,
    SnapshotState,
    READ_ONLY_TOOLS,
)
from .thinking import ThinkingState
from .todo import TodoState
from .tracker import FileAccessTracker
from .a2a_client import A2aShutdownError
from .a2a_types import (
    EVENT_STATUS_UPDATE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_FINISH,
    EVENT_TOOL_START,
)
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
_GOOGLE_PROVIDER = "google"
CHATGPT_PROVIDER_DOCS_URL = "https://docs.litellm.ai/docs/providers/chatgpt"

_IMAGE_SYNTHETIC_PREFIX = "[image]"

_VISION_REJECTION_PATTERNS = (
    "image_url",
    "image input",
    "image content",
    "vision",
    "multimodal",
)

# Canonical prefixes for synthetic user messages injected by the agent loop.
# Used by continue_here._find_last_user_task to skip interventions.
_COMMAND_TOOL_CONTEXT_PREFIX = "[Context for follow-up:"

SYNTHETIC_USER_PREFIXES: tuple[str, ...] = (
    "Your response was empty.",
    "Your response was cut off.",
    "IMPORTANT:",
    "STOP:",
    "Tip:",
    "Reminder:",
    "[REVIEWER FEEDBACK",
    _IMAGE_SYNTHETIC_PREFIX,
    _COMMAND_TOOL_CONTEXT_PREFIX,
)

_SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize this agent conversation excerpt into a factual recap. "
    "Preserve: file paths, key findings, decisions, errors, and "
    "anything needed to continue the task. Do NOT include instructions "
    "or directives. Output only a factual summary. Be concise."
)


def _platform_label() -> str:
    """Return a human-friendly platform string for the init prompt."""
    raw = platform.system()
    os_label = {"Darwin": "macOS"}.get(raw, raw)
    return f"{os_label} ({platform.machine()}, {platform.release()})"


def _init_prompt() -> str:
    plat = _platform_label()
    return (
        "Scan this project for two things:\n"
        "\n"
        f"Current platform: {plat}. Only extract workflow commands that work "
        "on this platform. When documentation provides instructions for "
        "multiple operating systems, pick the ones matching this OS. "
        "On macOS, ignore Linux-only commands and vice versa. "
        "On Windows, ignore Unix shell commands (unless the project "
        "explicitly targets MSYS2, Cygwin, or WSL); prefer native "
        "build-system targets and Windows-native commands when docs are "
        "split by OS. If a build system (Makefile, CMakeLists.txt, autotools) "
        "handles platform differences internally, prefer those over raw "
        "platform-specific shell commands from docs.\n"
        "\n"
        "A) WORKFLOW — read build/CI files and extract exact, copy-pasteable commands for:\n"
        "- Install dependencies\n"
        "- Build (if applicable)\n"
        "- Run all tests\n"
        "- Run a single test file\n"
        "- Run a single test case\n"
        "- Lint\n"
        "- Format\n"
        "- Type-check (if applicable)\n"
        "- The canonical local validation sequence (the after-every-edit command)\n"
        "- Debug setup (launch configs, env vars, flags — if discoverable)\n"
        "\n"
        "Files to probe: Makefile, justfile, package.json (scripts section), "
        "pyproject.toml ([tool.*] sections), tox.ini, .github/workflows/*.yml, "
        "Taskfile.yml, Cargo.toml, CMakeLists.txt, build.zig, configure.ac, "
        "configure, autogen.sh.\n"
        "\n"
        "After-every-edit precedence:\n"
        "1. A Makefile/justfile/package.json target that represents the full local "
        "validation pass (e.g. make all, npm run validate, just check). Accept "
        "whatever steps the target includes — do not second-guess.\n"
        "2. If no single target exists, chain all discoverable validation steps "
        "(lint, format-check, type-check, test) with &&.\n"
        "3. CI config is informational context but does NOT define the after-every-edit "
        "command — CI often runs a subset or superset of local validation. "
        "Prefer local build-system targets over CI steps.\n"
        "\n"
        "B) CONVENTIONS — cross-cutting patterns applied consistently across the "
        "codebase that an AI agent wouldn't know without reading the source. "
        "Look at: naming schemes, file/directory structure, error handling, return "
        "value formats, test organisation, API design. Read source files, tests, "
        "docs, and config. Use think to separate genuine project-wide patterns "
        "(appear in many independent places) from one-off choices."
    )


INIT_ENRICH_PROMPT = (
    "Review your findings. Never cut workflow commands (build, test, lint, "
    "format, type-check, debug, after-every-edit). These are always actionable.\n"
    "\n"
    "For conventions, cut anything that: (1) only appears in one file or module, "
    "(2) is standard practice any competent agent already knows, or (3) would not "
    "cause an agent to produce incorrect code or miss a required step. Keep only "
    "conventions that cross module boundaries and would surprise a capable agent "
    "new to this project. Check tests, docs, and config for anything missed."
)

_INIT_AGENTS_MD_BUDGET = 3000

INIT_WRITE_PROMPT = (
    "Write findings to AGENTS.md. Use exactly this structure:\n"
    "\n"
    "## Workflow\n"
    "\n"
    "- install: `<command>`\n"
    "- build: `<command>` (omit line if N/A)\n"
    "- test all: `<command>`\n"
    "- test file: `<command with placeholder>`\n"
    "- test case: `<command with placeholder>`\n"
    "- lint: `<command>`\n"
    "- format: `<command>`\n"
    "- typecheck: `<command>` (omit line if N/A)\n"
    "- after every edit: `<command or sequence>`\n"
    "- debug: `<notes>` (omit line if nothing discoverable)\n"
    "\n"
    "## Conventions\n"
    "\n"
    "- <terse convention bullets, 2 sentences max each>\n"
    "\n"
    "Rules:\n"
    f"- Total output must not exceed {_INIT_AGENTS_MD_BUDGET} characters. "
    "Workflow section takes priority. Cut convention bullets before workflow lines.\n"
    "- ## Workflow must be the first section.\n"
    "- Every command must be exact and copy-pasteable. No descriptions of what "
    "commands do.\n"
    "- The file is injected into every future agent context, so brevity is essential."
)

INIT_RETRY_PROMPT = (
    "The previous write failed validation: {reason}. "
    "Rewrite AGENTS.md with ## Workflow as the first section, followed by "
    "## Conventions. Follow the exact structure from the previous instructions."
)

_WORKFLOW_HEADING_RE = re.compile(r"^## Workflow\s*$", re.MULTILINE)
_CONVENTIONS_HEADING_RE = re.compile(r"^## Conventions\s*$", re.MULTILINE)
_ANY_H2_RE = re.compile(r"^## .+", re.MULTILINE)


def validate_agents_md(path: Path) -> tuple[str | None, str | None]:
    """Check AGENTS.md structure.

    Returns ``(reason, content)`` — *reason* is ``None`` when valid.
    *content* is the file text (``None`` when the file doesn't exist).
    """
    if not path.is_file():
        return "AGENTS.md was not created", None
    content = path.read_text(encoding="utf-8", errors="replace")
    if not _WORKFLOW_HEADING_RE.search(content):
        return "missing ## Workflow section", content
    if not _CONVENTIONS_HEADING_RE.search(content):
        return "missing ## Conventions section", content
    first_h2 = _ANY_H2_RE.search(content)
    if first_h2 and not _WORKFLOW_HEADING_RE.match(first_h2.group()):
        return "## Workflow is not the first section", content
    return None, content


LEARN_PROMPT = (
    "Review this session for concrete mistakes, confusions, or surprises you "
    "encountered with tools, commands, APIs, or syntax. Persist concise notes "
    "to `.swival/memory/MEMORY.md` for any durable lessons that will help in "
    "future sessions. If you were confused by something, add a note so you do "
    "not repeat the mistake. Do not store transient workspace state that may "
    "change soon, such as whether a file currently exists, current branch "
    "contents, or one-off task status. Keep MEMORY.md short (bulleted notes). "
    "For detailed topics, create separate files in `.swival/memory/` and "
    "reference them from MEMORY.md. If there is nothing worth noting, say so."
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

_TRANSIENT_PATTERNS = re.compile(
    r"Connection reset by peer|Connection refused|timed out"
    r"|RemoteDisconnected|Temporary failure in name resolution"
    r"|SSLError|EOF occurred|BrokenPipeError",
    re.IGNORECASE,
)


def _is_transient(exc):
    """Return True if the exception looks like a transient network/server error."""
    import litellm as _lt

    if isinstance(
        exc,
        (
            _lt.BadRequestError,
            _lt.AuthenticationError,
            _lt.NotFoundError,
            _lt.ContextWindowExceededError,
        ),
    ):
        return False
    if isinstance(
        exc,
        (
            _lt.InternalServerError,
            _lt.APIConnectionError,
            _lt.Timeout,
            _lt.RateLimitError,
        ),
    ):
        return True
    if isinstance(exc, _lt.APIError):
        status = getattr(exc, "status_code", None)
        if status is not None and 500 <= status < 600:
            return True
    return bool(_TRANSIENT_PATTERNS.search(str(exc)))


def _retries_from_exc(exc):
    """Extract provider retry count from an exception, if attached."""
    return getattr(exc, "_provider_retries", 0)


def _raise_with_retries(exc):
    """Attach _provider_retries default to an exception before raising."""
    if not hasattr(exc, "_provider_retries"):
        exc._provider_retries = 0
    raise exc


# Heuristics for open-weight backends that leak hidden-reasoning or tokenizer
# control markers into assistant content. These patterns intentionally prefer
# stripping standalone think markers over preserving literal tag discussions,
# because leaked reasoning is far more common in practice.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_THINK_BLOCK_PREFIX_RE = re.compile(
    r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL
)
_THINK_LINE_PREFIX_RE = re.compile(
    r"^.*?\n\s*</think>\s*\n*", re.IGNORECASE | re.DOTALL
)
_THINK_TAG_LINE_RE = re.compile(r"(?mi)^\s*</?think>\s*$\n?")


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


def _sanitize_assistant_content(text: str) -> str:
    """Strip leaked hidden-reasoning markers from assistant content."""
    if not text:
        return text

    cleaned = _SPECIAL_TOKEN_RE.sub("", text)
    while True:
        updated = _THINK_BLOCK_PREFIX_RE.sub("", cleaned, count=1)
        if "</think>" in updated.lower():
            updated = _THINK_LINE_PREFIX_RE.sub("", updated, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _THINK_TAG_LINE_RE.sub("", cleaned)
    return cleaned.strip()


def _sanitize_assistant_message(msg) -> None:
    """Normalize assistant content in-place for dict-or-namespace messages."""
    content = _msg_get(msg, "content")
    if isinstance(content, str):
        _set_msg_content(msg, _sanitize_assistant_content(content))


def _safe_history_path(base_dir: str) -> Path:
    """Build history path, verify it resolves inside base_dir."""
    base = Path(base_dir).resolve()
    history_path = (Path(base_dir) / ".swival" / "HISTORY.md").resolve()
    if not history_path.is_relative_to(base):
        raise ValueError(f"history path {history_path} escapes base directory {base}")
    return history_path


def _safe_memory_path(base_dir: str) -> Path:
    """Build memory path, verify it resolves inside base_dir."""
    base = Path(base_dir).resolve()
    memory_path = (Path(base_dir) / ".swival" / "memory" / "MEMORY.md").resolve()
    if not memory_path.is_relative_to(base):
        raise ValueError(f"memory path {memory_path} escapes base directory {base}")
    return memory_path


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

        # File lock makes the size check + append atomic across contexts.
        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]  # Windows

        lock_fd = None
        if fcntl is not None:
            lock_path = history_path.parent / "HISTORY.md.lock"
            lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            if fcntl is not None and lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

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
        finally:
            if fcntl is not None and lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
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
        content_raw = _msg_get(m, "content", "")
        if isinstance(content_raw, list):
            # Multimodal content array — estimate text and image parts separately
            text_parts = []
            for part in content_raw:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        total += _IMAGE_TOKEN_ESTIMATE
            content = " ".join(text_parts)
        else:
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

    if name == "read_multiple_files":
        files = args.get("files", [])
        paths = [f.get("file_path", "?") for f in files] if files else ["?"]
        return f"[read_multiple_files: {', '.join(paths)}, {len(content)} chars — compacted]"

    if name == "grep":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        # Extract match count from the "Found N match(es)" header
        m = re.match(r"Found (\d+) match", content)
        matches = int(m.group(1)) if m else content.count("\n")
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

    if name.startswith("a2a__"):
        # Preserve contextId/taskId for input-required tasks
        if "[input-required]" in content:
            # Extract the header line with IDs
            for line in content.splitlines():
                if line.startswith("[input-required]"):
                    return f"[{name}: {line} — compacted]"
            return f"[{name}: input-required — compacted]"
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


def _replace_last_image_message(messages: list, fallback_text: str) -> bool:
    """Find the last message with image_url content and replace it in place.

    Returns True if a replacement was made, False otherwise.
    """
    for i in range(len(messages) - 1, -1, -1):
        if (
            isinstance(messages[i], dict)
            and isinstance(messages[i].get("content"), list)
            and any(
                p.get("type") == "image_url"
                for p in messages[i]["content"]
                if isinstance(p, dict)
            )
        ):
            messages[i] = {"role": "user", "content": fallback_text}
            return True
    return False


def _strip_image_content(messages: list) -> None:
    """Replace list-valued content (multimodal image messages) with text-only."""
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            text = _msg_content(msg)  # extracts text parts
            msg["content"] = text + " [image data removed during compaction]"


def compact_messages(messages: list) -> list:
    """Compact large tool results in older turns, preserving turn atomicity.

    Uses per-tool structured summaries (via ``compact_tool_result``) instead of
    a blanket character-count truncation.
    """
    _strip_image_content(messages)
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


_DROPPABLE_USER_PREFIXES = (_IMAGE_SYNTHETIC_PREFIX, _COMMAND_TOOL_CONTEXT_PREFIX)


def is_pinned(turn: list) -> bool:
    """User turns are always preserved — except synthetic injections."""
    for msg in turn:
        if _msg_role(msg) == "user":
            content = _msg_content(msg)
            if content.startswith(_DROPPABLE_USER_PREFIXES):
                return False
            return True
    return False


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
        if content.startswith(SNAPSHOT_RECAP_PREFIX):
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
        _result = call_llm_fn(
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
        resp = _result[0]
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
        _SUMMARIZE_SYSTEM_PROMPT,
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
_CUSTOM_CMD_OUTPUT_CAP = 100_000  # Byte cap when context_length is unknown


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


def _global_agents_md_path() -> Path:
    """Return the cross-agent global AGENTS.md path (testable seam)."""
    return Path.home() / ".agents" / "AGENTS.md"


def load_instructions(
    base_dir: str,
    config_dir: "Path | None" = None,
    *,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    """Load CLAUDE.md and/or AGENTS.md, if present.

    AGENTS.md is loaded from up to three locations (user-level from
    *config_dir*, global cross-agent from ``~/.agents/``, and project-level
    from *base_dir*) inside a single ``<agent-instructions>`` block.  All
    three share a combined budget of ``MAX_INSTRUCTIONS_CHARS``.

    Returns (combined_text, filenames_loaded) where combined_text is
    XML-tagged sections (or "" if none found) and filenames_loaded lists
    the absolute paths of files that were actually loaded.
    """
    from .skills import strip_markdown_comments

    # Read up to 10x the output budget so comment stripping has room to work,
    # while still bounding memory for pathologically large files.
    read_cap = MAX_INSTRUCTIONS_CHARS * 10

    sections: list[str] = []
    loaded: list[str] = []

    # --- CLAUDE.md (project-level only) ---
    claude_path = Path(base_dir).resolve() / "CLAUDE.md"
    if claude_path.is_file():
        try:
            file_size = claude_path.stat().st_size
            with claude_path.open(encoding="utf-8", errors="replace") as f:
                content = strip_markdown_comments(f.read(read_cap))
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
                    user_content = strip_markdown_comments(f.read(read_cap))
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

    # Global cross-agent AGENTS.md (~/.agents/AGENTS.md)
    global_agents_path = _global_agents_md_path()
    if global_agents_path.is_file() and budget > 0:
        try:
            file_size = global_agents_path.stat().st_size
            with global_agents_path.open(encoding="utf-8", errors="replace") as f:
                global_content = strip_markdown_comments(f.read(read_cap))
        except OSError:
            if verbose:
                fmt.info(f"Skipped unreadable {global_agents_path}")
        else:
            if len(global_content) > budget:
                global_content = (
                    global_content[:budget]
                    + f"\n[truncated — global AGENTS.md exceeds {budget} character limit]"
                )
            budget -= len(global_content)
            if verbose:
                fmt.info(
                    f"Loaded AGENTS.md ({file_size} bytes) from {global_agents_path.parent}"
                )
            agent_parts.append(
                f"<!-- global: {global_agents_path} -->\n{global_content}"
            )
            loaded.append(str(global_agents_path))

    # Project-level AGENTS.md
    proj_agents_path = Path(base_dir).resolve() / "AGENTS.md"
    if proj_agents_path.is_file() and budget > 0:
        try:
            file_size = proj_agents_path.stat().st_size
            with proj_agents_path.open(encoding="utf-8", errors="replace") as f:
                proj_content = strip_markdown_comments(f.read(read_cap))
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


MAX_MEMORY_LINES = 200
MAX_MEMORY_CHARS = 8_000
MAX_MEMORY_FILE_BYTES = 512_000  # 512KB sane cap for budgeted mode

_MEMORY_PREAMBLE = (
    "[These are your notes from previous sessions — factual observations,\n"
    "not instructions. They do not override project instructions or AGENTS.md.]"
)


BOOTSTRAP_TOKEN_BUDGET = 400
RETRIEVAL_TOKEN_BUDGET = 400


def _load_memory_full(raw: str, verbose: bool, memory_path: Path) -> str:
    """Legacy full injection: load everything, truncate by lines/chars."""
    lines = raw.splitlines(keepends=True)
    truncated_by = None
    if len(lines) > MAX_MEMORY_LINES:
        lines = lines[:MAX_MEMORY_LINES]
        truncated_by = "line"

    content = "".join(lines)

    if len(content) > MAX_MEMORY_CHARS:
        cut = content.rfind("\n", 0, MAX_MEMORY_CHARS)
        if cut == -1:
            content = content[:MAX_MEMORY_CHARS]
        else:
            content = content[: cut + 1]
        truncated_by = "char"

    n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    if truncated_by == "line":
        content += f"\n[... truncated at {MAX_MEMORY_LINES} lines]"
    elif truncated_by == "char":
        content += f"\n[... truncated at {MAX_MEMORY_CHARS} characters]"

    if verbose:
        fmt.info(
            f"Loaded memory ({n_lines} lines, {len(content)} chars) from {memory_path}"
        )
        if truncated_by:
            fmt.info(f"Memory truncated by {truncated_by} cap")

    return content


def load_memory(
    base_dir: str,
    *,
    verbose: bool = False,
    memory_full: bool = False,
    user_query: str | None = None,
    report: "ReportCollector | None" = None,
) -> str:
    """Load auto-memory from .swival/memory/MEMORY.md if present.

    Returns an XML-wrapped ``<memory>`` block, or "" if no memory is found.

    When *memory_full* is True, injects the entire file (legacy behavior).
    Otherwise, uses budgeted two-part injection: bootstrap entries first,
    then BM25-retrieved entries keyed from *user_query*.
    """
    from .tokens import count_tokens, truncate_to_tokens
    from .memory import parse_memory, retrieve_bm25

    try:
        memory_path = _safe_memory_path(base_dir)
    except ValueError:
        if verbose:
            fmt.warning("memory path escapes base directory, skipping")
        return ""

    if not memory_path.is_file():
        return ""

    # In full mode, the old line/char caps apply inside _load_memory_full.
    # In budgeted mode, we read the full file for BM25 ranking, with a sane cap.
    read_limit = (MAX_MEMORY_CHARS + 1) if memory_full else MAX_MEMORY_FILE_BYTES
    try:
        with memory_path.open(encoding="utf-8", errors="replace") as f:
            raw = f.read(read_limit)
    except OSError:
        if verbose:
            fmt.warning(f"failed to read memory from {memory_path}")
        return ""

    if not raw or not raw.strip():
        return ""

    # Legacy full injection mode
    if memory_full:
        content = _load_memory_full(raw, verbose, memory_path)
        if report:
            report.record_memory(
                total_entries=0,
                bootstrap_entries=0,
                retrievable_entries=0,
                bootstrap_tokens=count_tokens(content),
                retrieval_tokens=0,
                retrieved_ids=[],
                mode="full",
            )
        return f"<memory>\n{_MEMORY_PREAMBLE}\n\n{content}\n</memory>"

    # Budgeted injection
    entries = parse_memory(raw)
    if not entries:
        return ""

    bootstrap = [e for e in entries if e.is_bootstrap]
    retrievable = [e for e in entries if not e.is_bootstrap]

    def _pack_entries(
        entry_list: list, budget: int
    ) -> tuple[list[str], int, list[str]]:
        """Pack entries into a budget, truncating the last if needed."""
        parts: list[str] = []
        tokens_used = 0
        ids: list[str] = []
        for entry in entry_list:
            entry_tokens = entry.tokens
            if tokens_used + entry_tokens > budget:
                remaining = budget - tokens_used
                if remaining > 20:
                    parts.append(truncate_to_tokens(entry.content, remaining))
                    tokens_used += remaining
                    ids.append(entry.id)
                break
            parts.append(entry.content)
            tokens_used += entry_tokens
            ids.append(entry.id)
        return parts, tokens_used, ids

    # Part 1: bootstrap block (always included, within budget)
    bootstrap_parts, bootstrap_tokens, _ = _pack_entries(
        bootstrap, BOOTSTRAP_TOKEN_BUDGET
    )

    # Part 2: retrieval block (BM25-ranked, within budget)
    retrieved_ids: list[str] = []
    retrieval_parts: list[str] = []
    retrieval_tokens = 0

    if retrievable:
        if user_query:
            results = retrieve_bm25(
                user_query,
                retrievable,
                top_k=5,
                token_budget=RETRIEVAL_TOKEN_BUDGET,
            )
            for entry, _score in results:
                retrieval_parts.append(entry.content)
                retrieval_tokens += entry.tokens
                retrieved_ids.append(entry.id)
        else:
            # No query available — take first entries that fit
            retrieval_parts, retrieval_tokens, retrieved_ids = _pack_entries(
                retrievable, RETRIEVAL_TOKEN_BUDGET
            )

    # Assemble
    sections: list[str] = []
    if bootstrap_parts:
        sections.extend(bootstrap_parts)
    if retrieval_parts:
        sections.extend(retrieval_parts)

    if verbose:
        fmt.info(
            f"Memory: {len(entries)} entries "
            f"({len(bootstrap)} bootstrap, {len(retrievable)} retrievable), "
            f"injecting {bootstrap_tokens}+{retrieval_tokens} tokens"
        )
        if retrieved_ids:
            fmt.info(f"Retrieved memory entries: {', '.join(retrieved_ids)}")

    if report:
        report.record_memory(
            total_entries=len(entries),
            bootstrap_entries=len(bootstrap),
            retrievable_entries=len(retrievable),
            bootstrap_tokens=bootstrap_tokens,
            retrieval_tokens=retrieval_tokens,
            retrieved_ids=retrieved_ids,
            mode="budgeted",
        )

    if not sections:
        return ""

    content = "\n\n".join(sections)

    return f"<memory>\n{_MEMORY_PREAMBLE}\n\n{content}\n</memory>"


def _show_state_summaries(thinking_state, todo_state, snapshot_state) -> None:
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


def _post_tool_bookkeeping(
    tool_msg,
    tool_meta,
    turn,
    turn_offset,
    report,
    snapshot_state,
    consecutive_errors,
    verbose,
    _emit,
):
    """Post-tool-call bookkeeping shared by run_agent_loop() and command provider.

    Handles: post-call event emission, report logging, snapshot dirty tracking,
    error guardrail tracking.

    EVENT_TOOL_START is NOT included — callers emit it before execution.

    Returns list of intervention strings.
    """
    interventions = []
    name = tool_meta["name"]

    if tool_meta["succeeded"]:
        _emit(
            EVENT_TOOL_FINISH,
            {"name": name, "turn": turn, "elapsed": tool_meta["elapsed"]},
        )
    else:
        _emit(
            EVENT_TOOL_ERROR,
            {"name": name, "turn": turn, "error": tool_msg["content"][:500]},
        )

    if report:
        report.record_tool_call(
            turn + turn_offset,
            name,
            tool_meta["arguments"],
            tool_meta["succeeded"],
            tool_meta["elapsed"],
            len(tool_msg["content"]),
            error=tool_msg["content"] if not tool_meta["succeeded"] else None,
        )

    if snapshot_state is not None:
        snapshot_state.mark_dirty(name)

    result = tool_msg["content"]
    if result.startswith("error:"):
        canonical = _canonical_error(result)
        prev_err, prev_count = consecutive_errors.get(name, ("", 0))
        count = prev_count + 1 if canonical == prev_err else 1
        consecutive_errors[name] = (canonical, count)

        if count >= 2:
            if count >= 3:
                level = "stop"
                interventions.append(
                    f"STOP: You have failed to use `{name}` correctly {count} times in a row "
                    "with the same error. Do NOT call "
                    f"`{name}` again with the same arguments. "
                    "Either fix the arguments or use a completely different approach to accomplish your task."
                )
            else:
                level = "nudge"
                interventions.append(
                    f"IMPORTANT: You have called `{name}` {count} times with the same error. "
                    f"The error is: {canonical}\n"
                    "Please carefully re-read the error message and fix your tool call. "
                    "If you cannot use this tool correctly, use a different approach."
                )
            if report:
                report.record_guardrail(turn + turn_offset, name, level)
            if verbose:
                fmt.guardrail(name, count, canonical)
    else:
        consecutive_errors.pop(name, None)

    return interventions


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
    a2a_manager=None,
    messages=None,
    image_stash=None,
    scratch_dir=None,
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
            a2a_manager=a2a_manager,
            messages=messages,
            verbose=verbose,
            image_stash=image_stash,
            scratch_dir=scratch_dir,
        )
    except McpShutdownError:
        result = "error: MCP server is shutting down"
    except A2aShutdownError:
        result = "error: A2A agent is shutting down"
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


def _pick_best_choice(choices):
    """Select the most actionable choice from a multi-choice response.

    The Responses-API bridge in litellm may split a single LLM turn into
    multiple choices: one for text output (finish_reason='stop') and another
    for tool calls (finish_reason='tool_calls').  When both exist, tool calls
    take priority — the text is merged into the tool-call choice so it isn't
    lost.
    """
    if not choices:
        raise AgentError("LLM returned an empty choices list")
    if len(choices) == 1:
        return choices[0]

    tool_choice = None
    text_parts = []
    for c in choices:
        if getattr(c.message, "tool_calls", None):
            tool_choice = c
        elif getattr(c.message, "content", None):
            text_parts.append(c.message.content)

    if tool_choice is not None:
        if text_parts:
            tool_choice.message.content = "\n\n".join(text_parts)
        return tool_choice

    return choices[0]


def _resolve_model_str(provider: str, model_id: str) -> str:
    """Map (provider, model_id) to the litellm model string."""
    if provider == "lmstudio":
        return f"openai/{model_id}"
    elif provider == "huggingface":
        return f"huggingface/{model_id.removeprefix('huggingface/')}"
    elif provider == "openrouter":
        bare = (
            model_id[len("openrouter/") :]
            if model_id.startswith("openrouter/openrouter/")
            else model_id
        )
        return f"openrouter/{bare}"
    elif provider == "generic":
        return f"openai/{model_id}"
    elif provider == "chatgpt":
        bare = model_id.removeprefix("chatgpt/").removeprefix("chatgpt/")
        return f"chatgpt/{bare}"
    elif provider == "command":
        return model_id
    else:
        return model_id


def _render_transcript(messages):
    """Render a messages list as a plain-text transcript for command provider."""
    from ._msg import _msg_get, _msg_role, _msg_tool_calls, _msg_tool_call_id

    # First pass: index tool_call_id → function name from assistant messages
    tc_names = {}
    for m in messages:
        tool_calls = _msg_tool_calls(m)
        if tool_calls:
            for tc in tool_calls:
                tc_id = _msg_get(tc, "id", "")
                fn = _msg_get(tc, "function")
                name = _msg_get(fn, "name", "tool") if fn else "tool"
                tc_names[tc_id] = name

    # Second pass: render
    lines = []
    for m in messages:
        role = _msg_role(m) or "unknown"
        content = _msg_get(m, "content", "")

        # Image-aware content extraction (differs from _msg_content which
        # silently drops images — here we insert placeholders)
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") in ("image_url", "image"):
                        parts.append("[image omitted]")
            content = "\n".join(parts)

        if not content:
            continue

        if role == "tool":
            tool_call_id = _msg_tool_call_id(m)
            msg_name = _msg_get(m, "name", "")
            if msg_name and (
                msg_name.startswith("mcp__")
                or msg_name.startswith("a2a__")
                or msg_name == "use_skill"
            ):
                lines.append(
                    f'[swival_result id="{tool_call_id}" name="{msg_name}"]\n{content}'
                )
            else:
                tool_name = tc_names.get(tool_call_id, "tool")
                lines.append(f"[tool:{tool_name}]\n{content}")
        else:
            lines.append(f"[{role}]\n{content}")

    return "\n\n".join(lines)


_SWIVAL_BLOCK_RE = re.compile(
    r"<swival:call\s([^>]+)>\s*(\{.*?\})\s*</swival:call>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_swival_calls(text):
    """Extract (call_id, tool_name, args_dict) tuples from agent output.

    Attribute order in the opening tag does not matter. Unknown attributes
    are ignored. Both id and name are required; blocks missing either are
    skipped.

    Malformed JSON in a block produces an entry with {"_parse_error": "..."}
    so the caller can feed an error result back to the agent.
    """
    results = []
    for m in _SWIVAL_BLOCK_RE.finditer(text):
        attr_str, args_json = m.group(1), m.group(2)
        attrs = dict(_ATTR_RE.findall(attr_str))

        call_id = attrs.get("id")
        name = attrs.get("name")
        if not call_id or not name:
            continue

        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            results.append((call_id, name, {"_parse_error": str(e)}))
            continue
        results.append((call_id, name, args))
    return results


def _render_swival_tool_catalog(tool_schemas):
    """Render tool schemas as a text catalog for the command provider system prompt."""
    lines = []
    for schema in tool_schemas:
        func = schema.get("function", schema)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        param_parts = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            opt = "" if pname in required else "?"
            param_parts.append(f'"{pname}{opt}": {ptype}')
        param_str = "{" + ", ".join(param_parts) + "}" if param_parts else "{}"

        lines.append(f"- {name}: {desc}")
        lines.append(f"  Parameters: {param_str}")
    return "\n".join(lines)


def _filter_command_tool_schemas(tools):
    """Filter tool schemas to those exposable to command provider (MCP/A2A/skills)."""
    return [
        t
        for t in tools
        if t.get("function", {}).get("name", "").startswith(("mcp__", "a2a__"))
        or t.get("function", {}).get("name") == "use_skill"
    ]


def _make_tool_call_obj(call_id, name, args_dict):
    """Build a synthetic tool_call matching the shape handle_tool_call() expects."""
    from types import SimpleNamespace

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args_dict),
        ),
    )


class _SyntheticMessage:
    """Lightweight message object compatible with the agent loop.

    Supports: msg.content, msg.tool_calls, msg.role,
    getattr(msg, ...), msg.model_dump(exclude_none=True).
    """

    __slots__ = ("role", "content", "tool_calls")

    def __init__(self, content):
        self.role = "assistant"
        self.content = content
        self.tool_calls = None

    def model_dump(self, **kwargs):
        d = {"role": self.role, "content": self.content}
        if kwargs.get("exclude_none"):
            return {k: v for k, v in d.items() if v is not None}
        d["tool_calls"] = self.tool_calls
        return d


def _make_synthetic_message(text):
    """Build a synthetic message object compatible with the agent loop."""
    return _SyntheticMessage(text)


def _call_command(command_str, messages, verbose, max_output_tokens=None):
    """Run an external command as the LLM, passing the conversation on stdin."""
    parts = shlex.split(command_str)
    transcript = _render_transcript(messages)

    if verbose:
        fmt.model_info(f"Running command: {command_str}")

    response_text = _run_command_once(parts, transcript, verbose, command_str)

    if max_output_tokens and max_output_tokens > 0:
        from .tokens import truncate_to_tokens

        response_text = truncate_to_tokens(response_text, max_output_tokens)

    msg = _make_synthetic_message(response_text)
    return msg, "stop"


_COMMAND_TOOL_MAX_ROUNDS = 20


def _run_command_once(parts, transcript, verbose, command_str):
    """Run command subprocess and return (response_text, stderr_text).

    Raises AgentError on failure.
    """
    try:
        proc = subprocess.run(
            parts,
            input=transcript,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise AgentError(f"command timed out after 300s: {command_str}") from e
    except OSError as e:
        raise AgentError(f"command failed to start: {e}") from e

    if proc.returncode != 0:
        error_text = (
            proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        )
        raise AgentError(f"command provider failed: {error_text}")

    if proc.stderr.strip() and verbose:
        print(proc.stderr, end="", file=sys.stderr)

    response_text = proc.stdout.strip()
    if not response_text:
        raise AgentError("command provider returned empty output")

    return response_text


def _call_command_with_tools(
    command_str,
    messages,
    handle_tool_call_kwargs,
    outer_turn,
    outer_turn_offset,
    report,
    snapshot_state,
    verbose,
    _emit,
    max_output_tokens=None,
):
    """Run command provider with Swival tool-calling support.

    The external agent uses <swival:call> XML blocks to request tool execution.
    Swival parses them, dispatches via handle_tool_call(), and re-invokes the
    command with updated transcript until the agent responds without tool calls.

    Returns (synthetic_message, "stop", activity_summary).
    activity_summary is a list of {"name": str, "succeeded": bool} dicts.
    """
    parts = shlex.split(command_str)
    transcript_messages = list(messages)
    consecutive_errors: dict[str, tuple[str, int]] = {}
    tool_activity: list[dict] = []
    response_text = ""

    for _ in range(_COMMAND_TOOL_MAX_ROUNDS):
        transcript = _render_transcript(transcript_messages)

        if verbose:
            fmt.model_info(f"Running command: {command_str}")

        response_text = _run_command_once(parts, transcript, verbose, command_str)

        calls = _parse_swival_calls(response_text)
        if not calls:
            break

        transcript_messages.append({"role": "assistant", "content": response_text})

        round_interventions: list[str] = []
        for call_id, name, args in calls:
            if "_parse_error" in args:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"error: invalid JSON in tool arguments: {args['_parse_error']}",
                }
                tool_meta = {
                    "name": name,
                    "arguments": None,
                    "elapsed": 0.0,
                    "succeeded": False,
                }
            else:
                tc = _make_tool_call_obj(call_id, name, args)
                _emit(EVENT_TOOL_START, {"name": name, "turn": outer_turn})
                tool_msg, tool_meta = handle_tool_call(tc, **handle_tool_call_kwargs)

            intv = _post_tool_bookkeeping(
                tool_msg,
                tool_meta,
                outer_turn,
                outer_turn_offset,
                report,
                snapshot_state,
                consecutive_errors,
                verbose,
                _emit,
            )
            round_interventions.extend(intv)

            tool_activity.append({"name": name, "succeeded": tool_meta["succeeded"]})

            transcript_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": tool_msg["content"],
                }
            )

        if round_interventions:
            transcript_messages.append(
                {"role": "user", "content": "\n\n".join(round_interventions)}
            )

    if max_output_tokens and max_output_tokens > 0:
        from .tokens import truncate_to_tokens

        response_text = truncate_to_tokens(response_text, max_output_tokens)

    return _make_synthetic_message(response_text), "stop", tool_activity


def _model_supports_vision(model_str: str) -> bool | None:
    """Check if the resolved model supports vision via litellm.

    Returns True, False, or None (unknown / not in registry).
    litellm.supports_vision() returns False for models not in its registry,
    so we first check if the model is known at all via get_model_info().
    """
    try:
        import litellm

        try:
            litellm.get_model_info(model=model_str)
        except Exception:
            return None  # model not in registry — try optimistically
        return litellm.supports_vision(model=model_str)
    except Exception:
        return None


def _is_vision_rejection(error: "AgentError") -> bool:
    """Heuristic: does this error look like a vision/multimodal rejection?"""
    msg = str(error).lower()
    return any(pattern in msg for pattern in _VISION_REJECTION_PATTERNS)


def _completion_with_retry(completion_kwargs, *, max_retries, verbose):
    """Call litellm.completion() with retry on transient errors.

    Returns (response, provider_retries) where provider_retries is the number
    of retries performed (0 = first attempt succeeded).

    On failure, attaches ``_provider_retries`` to the raised exception so
    callers can record how many attempts were made before the error.

    Raises ContextOverflowError, litellm.BadRequestError, or the original
    exception for non-transient errors.
    """
    import litellm

    if max_retries < 1:
        max_retries = 1

    for attempt in range(max_retries):
        try:
            return litellm.completion(**completion_kwargs), attempt
        except litellm.ContextWindowExceededError:
            coe = ContextOverflowError("context window exceeded (typed)")
            coe._provider_retries = attempt
            raise coe
        except litellm.BadRequestError as e:
            e._provider_retries = attempt
            raise
        except Exception as e:
            if _CONTEXT_OVERFLOW_RE.search(str(e)):
                coe = ContextOverflowError(f"context window exceeded (inferred): {e}")
                coe._provider_retries = attempt
                raise coe
            if not _is_transient(e) or attempt == max_retries - 1:
                e._provider_retries = attempt
                raise
            delay = min(2 * (2**attempt), 30)
            delay *= 0.75 + 0.5 * random.random()
            if verbose:
                fmt.warning(
                    f"Network error: {e} — retrying in {delay:.0f}s "
                    f"(attempt {attempt + 2}/{max_retries})"
                )
            time.sleep(delay)


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
    reasoning_effort=None,
    sanitize_thinking=None,
    cache=None,
    secret_shield=None,
    command_tool_kwargs=None,
    max_retries=5,
    llm_filter=None,
    call_kind="agent",
):
    """Call LiteLLM with the appropriate provider.

    Returns (message, finish_reason, cmd_activity, provider_retries).
    cmd_activity is a list of {"name": str, "succeeded": bool} dicts
    (non-empty only for command provider with tool calls).
    provider_retries is the number of transient-error retries (0 = first attempt ok).
    """
    # --- Outbound: user-defined filter ---
    if llm_filter is not None:
        from .filter import run_llm_filter, FilterError

        try:
            messages = run_llm_filter(
                llm_filter,
                messages,
                model=model_id,
                provider=provider,
                tools=tools,
                call_kind=call_kind,
            )
        except FilterError as e:
            raise AgentError(f"LLM filter blocked request: {e}") from e
        _sanitize_assistant_messages(messages)
        cache = None  # filter script is an external mutable dependency; cached responses may be stale

    if provider == "command":
        if command_tool_kwargs is not None:
            return (
                *_call_command_with_tools(
                    model_id,
                    messages,
                    verbose=verbose,
                    max_output_tokens=max_output_tokens,
                    **command_tool_kwargs,
                ),
                0,
            )
        msg, stop = _call_command(model_id, messages, verbose, max_output_tokens)
        return msg, stop, [], 0

    # --- Outbound: encrypt secrets ---
    if secret_shield is not None:
        # Sanitize on canonical list before making the encryption copy
        _sanitize_assistant_messages(messages)
        messages = secret_shield.encrypt_messages(messages)
        cache = None  # disable cache when encryption is active

    import litellm

    litellm.suppress_debug_info = True

    # Resolve sanitize_thinking: opt-in only.
    if sanitize_thinking is None:
        sanitize_thinking = False

    _skip_params: set[str] = set()
    _skip_tool_choice = False

    model_str = _resolve_model_str(provider, model_id)

    if provider == "lmstudio":
        kwargs = {"api_base": f"{base_url}/v1", "api_key": "lm-studio"}
    elif provider == "huggingface":
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["api_base"] = base_url
    elif provider == "openrouter":
        kwargs = {
            "api_key": api_key,
            "extra_headers": {
                "HTTP-Referer": "https://swival.dev",
                "X-Title": "swival",
            },
        }
        if base_url:
            kwargs["api_base"] = base_url
    elif provider == "generic":
        kwargs = {"api_base": base_url, "api_key": api_key or "none"}
    elif provider == "chatgpt":
        kwargs = {}
        _skip_params = {"top_p", "seed"}
        _skip_tool_choice = True
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
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
        if reasoning_effort is not None:
            extras.append(f"reasoning_effort={reasoning_effort}")
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
        if not _skip_tool_choice:
            completion_kwargs["tool_choice"] = "auto"
    for key, val in [("temperature", temperature), ("top_p", top_p), ("seed", seed)]:
        if val is not None and key not in _skip_params:
            completion_kwargs[key] = val
    if extra_body is not None:
        completion_kwargs["extra_body"] = extra_body
    if reasoning_effort is not None and reasoning_effort != "default":
        completion_kwargs["reasoning_effort"] = reasoning_effort

    # --- Cache lookup ---
    # Skip cache for vision requests — base64 payloads would bloat the DB
    if cache is not None and _has_image_content(messages):
        cache = None
    cache_kwargs = None
    if cache is not None:
        api_base_for_key = kwargs.get("api_base", "")
        cache_kwargs = {
            **completion_kwargs,
            "_provider": provider,
            "_api_base": api_base_for_key,
        }
        hit = cache.get(cache_kwargs)
        if hit is not None:
            from .cache import _reconstruct_message

            msg_dict, finish_reason = hit
            if verbose:
                fmt.info("Cache hit")
            msg = _reconstruct_message(msg_dict)
            if sanitize_thinking:
                _sanitize_assistant_message(msg)
            # Note: cache is disabled when secret_shield is active, so no
            # decrypt needed here.  But guard defensively in case the logic
            # changes.
            return msg, finish_reason, [], 0

    def _cache_store(choice):
        if cache is not None:
            msg_d = (
                choice.message.model_dump(exclude_none=True)
                if hasattr(choice.message, "model_dump")
                else dict(vars(choice.message))
            )
            cache.put(cache_kwargs, msg_d, choice.finish_reason)

    def _decrypt_msg(msg):
        """Reverse known encrypted tokens in response content and tool args."""
        if secret_shield is None:
            return msg
        if msg.content:
            msg.content = secret_shield.reverse_known(msg.content)
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args_str = tc.function.arguments
                tc.function.arguments = secret_shield.reverse_known(args_str)
        return msg

    retries = 0
    try:
        response, retries = _completion_with_retry(
            completion_kwargs, max_retries=max_retries, verbose=verbose
        )
    except ContextOverflowError:
        raise  # already has _provider_retries from _completion_with_retry
    except litellm.BadRequestError as e:
        msg_text = str(e)
        if _CONTEXT_OVERFLOW_RE.search(msg_text):
            coe = ContextOverflowError(f"context window exceeded (inferred): {e}")
            coe._provider_retries = _retries_from_exc(e)
            raise coe
        if _EMPTY_ASSISTANT_RE.search(msg_text):
            # Provider rejected an assistant message with no content and no
            # tool_calls (common with Mistral via OpenRouter).  Fix the
            # messages in place and retry once.
            first_retries = _retries_from_exc(e)
            if _sanitize_assistant_messages(messages):
                if verbose:
                    fmt.warning("Fixed empty assistant message in history, retrying...")
                try:
                    response, retries = _completion_with_retry(
                        completion_kwargs,
                        max_retries=max_retries,
                        verbose=verbose,
                    )
                except ContextOverflowError as coe2:
                    coe2._provider_retries = first_retries + getattr(
                        coe2, "_provider_retries", 0
                    )
                    raise
                except Exception as e2:
                    combined = first_retries + _retries_from_exc(e2)
                    if _CONTEXT_OVERFLOW_RE.search(str(e2)):
                        coe = ContextOverflowError(
                            f"context window exceeded (inferred, post-sanitization): {e2}"
                        )
                        coe._provider_retries = combined
                        raise coe
                    ae = AgentError(f"LLM call failed after message sanitization: {e2}")
                    ae._provider_retries = combined
                    _raise_with_retries(ae)
                retries += first_retries
                choice = _pick_best_choice(response.choices)
                if sanitize_thinking:
                    _sanitize_assistant_message(choice.message)
                _cache_store(choice)
                return (
                    _decrypt_msg(choice.message),
                    choice.finish_reason,
                    [],
                    retries,
                )
        ae = AgentError(f"LLM call failed: {e}")
        ae._provider_retries = _retries_from_exc(e)
        _raise_with_retries(ae)
    except Exception as e:
        ae = AgentError(f"LLM call failed: {e}")
        ae._provider_retries = _retries_from_exc(e)
        _raise_with_retries(ae)

    choice = _pick_best_choice(response.choices)
    if sanitize_thinking:
        _sanitize_assistant_message(choice.message)
    _cache_store(choice)
    return _decrypt_msg(choice.message), choice.finish_reason, [], retries


# Provider → env var that resolve_provider() checks for that provider
_PROVIDER_KEY_ENV: dict[str, str] = {
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
    "generic": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "chatgpt": "CHATGPT_API_KEY",
}


def _build_self_review_cmd(args: argparse.Namespace) -> str:
    """Build a reviewer command that mirrors the current invocation's settings."""
    import shlex

    parts = [sys.executable, "-m", "swival.agent", "--reviewer-mode", "--quiet"]

    if args.yolo:
        parts.append("--yolo")
    if args.provider and args.provider != "lmstudio":
        parts.extend(["--provider", args.provider])
    if args.model:
        parts.extend(["--model", str(args.model)])
    if args.base_url:
        parts.extend(["--base-url", str(args.base_url)])
    if args.skills_dir:
        for d in args.skills_dir:
            parts.extend(["--skills-dir", d])
    if args.max_context_tokens:
        parts.extend(["--max-context-tokens", str(args.max_context_tokens)])
    if args.max_output_tokens and args.max_output_tokens != 32768:
        parts.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if getattr(args, "encrypt_secrets", False):
        parts.append("--encrypt-secrets")
    if getattr(args, "retries", 5) != 5:
        parts.extend(["--retries", str(args.retries)])

    return shlex.join(parts)


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
    if stderr and verbose and proc.returncode == 2:
        fmt.warning(f"reviewer stderr: {stderr.rstrip()}")
    return proc.returncode, stdout, stderr


def build_parser():
    """Build and return the argument parser."""
    help_examples = (
        "Examples:\n"
        '  swival --yolo "Refactor the auth module"\n'
        "  swival --yolo --repl\n"
        '  swival --provider huggingface --model zai-org/GLM-5 "Write parser tests"\n'
        '  swival --yolo --self-review "Add input validation"\n'
        "  swival -q < task.md"
    )
    parser = argparse.ArgumentParser(
        prog="swival",
        usage=(
            "%(prog)s [options] <task>\n"
            "       %(prog)s [options] < task.md\n"
            "       %(prog)s --repl [options] [task]"
        ),
        description=(
            "A CLI coding agent with tool-calling, sandboxed file access, and "
            "multi-provider LLM support.\n"
            "Pass a task as a positional argument, or omit it and pipe the task on stdin."
        ),
        epilog=help_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser._positionals.title = "Task input"
    parser._optionals.title = "General"
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        metavar="TASK",
        help="Task to run. If omitted and stdin is piped, Swival reads the task from stdin.",
    )

    modes = parser.add_argument_group("Modes")
    provider_group = parser.add_argument_group("Provider and model")
    behavior_group = parser.add_argument_group("Agent behavior")
    access_group = parser.add_argument_group("Filesystem and command access")
    prompt_group = parser.add_argument_group("Prompt, instructions, memory, and skills")
    integrations_group = parser.add_argument_group("Integrations")
    review_group = parser.add_argument_group("Review and reporting")
    server_group = parser.add_argument_group("A2A server")
    output_group = parser.add_argument_group("Output and setup")

    access_group.add_argument(
        "--add-dir",
        type=str,
        action="append",
        default=None,
        help="Grant read/write access to an extra directory (repeatable).",
    )
    access_group.add_argument(
        "--add-dir-ro",
        type=str,
        action="append",
        default=None,
        help="Grant read-only access to an extra directory (repeatable).",
    )
    access_group.add_argument(
        "--allowed-commands",
        type=str,
        default=_UNSET,
        help='Comma-separated list of allowed command basenames (e.g. "ls,git,python3").',
    )
    provider_group.add_argument(
        "--api-key",
        type=str,
        default=_UNSET,
        help="API key for the provider (overrides env var: HF_TOKEN, "
        "OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or CHATGPT_API_KEY).",
    )
    access_group.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for file tools (default: current directory).",
    )
    provider_group.add_argument(
        "--base-url",
        default=_UNSET,
        help="Server base URL (default: http://127.0.0.1:1234 for lmstudio).",
    )

    color_group = output_group.add_mutually_exclusive_group()
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

    provider_group.add_argument(
        "--extra-body",
        type=_parse_extra_body,
        default=_UNSET,
        metavar="JSON",
        help='Extra parameters to pass to the LLM API as JSON (e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\').',
    )

    _REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "default")
    provider_group.add_argument(
        "--reasoning-effort",
        choices=_REASONING_LEVELS,
        default=_UNSET,
        metavar="LEVEL",
        help="Reasoning effort level for models that support it (e.g. gpt-5.4). "
        f"One of: {', '.join(_REASONING_LEVELS)}.",
    )

    provider_group.add_argument(
        "--sanitize-thinking",
        action="store_true",
        default=_UNSET,
        help="Strip leaked <think> tags from assistant responses.",
    )

    behavior_group.add_argument(
        "--cache",
        action="store_true",
        default=_UNSET,
        help="Enable LLM response caching (.swival/cache.db).",
    )
    behavior_group.add_argument(
        "--cache-dir",
        type=str,
        default=_UNSET,
        metavar="PATH",
        help="Custom cache database directory (default: .swival).",
    )

    behavior_group.add_argument(
        "--llm-filter",
        metavar="COMMAND",
        dest="llm_filter",
        default=_UNSET,
        help="Filter command (shell-split) run before each outbound LLM request. "
        "Receives JSON on stdin, writes filtered messages JSON to stdout. "
        'Non-zero exit or {"allow": false} blocks the request.',
    )

    encrypt_group = behavior_group.add_mutually_exclusive_group()
    encrypt_group.add_argument(
        "--encrypt-secrets",
        action="store_true",
        default=_UNSET,
        help="Encrypt recognized credential tokens before sending to LLM provider.",
    )
    encrypt_group.add_argument(
        "--no-encrypt-secrets",
        action="store_true",
        default=_UNSET,
        help="Disable secret encryption (default).",
    )
    behavior_group.add_argument(
        "--encrypt-secrets-key",
        type=str,
        default=_UNSET,
        metavar="HEX",
        help="Hex-encoded 32-byte key for secret encryption (default: random per session).",
    )
    output_group.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Generate a config file template and exit.",
    )
    provider_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=_UNSET,
        help="Requested context length for the model (may trigger a reload).",
    )
    behavior_group.add_argument(
        "--max-output-tokens",
        type=int,
        default=_UNSET,
        help="Maximum output tokens (default: 32768).",
    )
    review_group.add_argument(
        "--max-review-rounds",
        type=int,
        default=_UNSET,
        help="Maximum number of reviewer retry rounds (default: 15). 0 disables retries.",
    )
    behavior_group.add_argument(
        "--max-turns",
        type=int,
        default=_UNSET,
        help="Maximum agent loop iterations (default: 100).",
    )
    behavior_group.add_argument(
        "--retries",
        type=int,
        default=_UNSET,
        help="Max provider retries on transient network errors (default: 5, 1 = no retry).",
    )
    integrations_group.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to an MCP JSON config file (replaces .mcp.json default lookup).",
    )
    provider_group.add_argument(
        "--model",
        type=str,
        default=_UNSET,
        help="Override auto-discovered model with a specific model identifier.",
    )
    prompt_group.add_argument(
        "--no-history",
        action="store_true",
        default=_UNSET,
        help="Don't write responses to .swival/HISTORY.md",
    )
    prompt_group.add_argument(
        "--no-memory",
        action="store_true",
        default=_UNSET,
        help="Don't load auto-memory from .swival/memory/.",
    )
    prompt_group.add_argument(
        "--memory-full",
        action="store_true",
        default=_UNSET,
        help="Inject all of MEMORY.md into the prompt (skip budgeted retrieval).",
    )
    prompt_group.add_argument(
        "--no-continue",
        action="store_true",
        default=_UNSET,
        help="Don't write or read .swival/continue.md on session interruption.",
    )
    prompt_group.add_argument(
        "--no-instructions",
        action="store_true",
        default=_UNSET,
        help="Don't load CLAUDE.md or AGENTS.md from the base directory, user config directory, or ~/.agents/.",
    )
    integrations_group.add_argument(
        "--no-mcp",
        action="store_true",
        default=_UNSET,
        help="Disable MCP server connections entirely.",
    )
    integrations_group.add_argument(
        "--a2a-config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to an A2A TOML config file with [a2a_servers.*] tables.",
    )
    integrations_group.add_argument(
        "--no-a2a",
        action="store_true",
        default=_UNSET,
        help="Disable A2A agent connections entirely.",
    )
    integrations_group.add_argument(
        "--lifecycle-command",
        metavar="COMMAND",
        default=_UNSET,
        help="Command invoked at startup and exit as: <command> startup|exit <base_dir>. "
        "Receives SWIVAL_* env vars with Git and project metadata.",
    )
    integrations_group.add_argument(
        "--lifecycle-timeout",
        type=int,
        default=_UNSET,
        metavar="SECONDS",
        help="Timeout for lifecycle hook execution (default: 300).",
    )
    integrations_group.add_argument(
        "--lifecycle-fail-closed",
        action="store_true",
        default=_UNSET,
        help="Abort the run if a lifecycle hook fails (default: fail-open, log warning).",
    )
    integrations_group.add_argument(
        "--no-lifecycle",
        action="store_true",
        default=_UNSET,
        help="Disable lifecycle hooks entirely (useful for nested or automated invocations).",
    )
    access_group.add_argument(
        "--no-read-guard",
        action="store_true",
        default=_UNSET,
        help="Disable read-before-write guard (allow writing files without reading them first).",
    )
    access_group.add_argument(
        "--sandbox",
        choices=["builtin", "agentfs"],
        default=_UNSET,
        help='Sandbox backend: "builtin" (app-layer path guards) or "agentfs" (OS-enforced via AgentFS). Default: builtin.',
    )
    access_group.add_argument(
        "--sandbox-session",
        type=str,
        default=_UNSET,
        help="AgentFS session ID for persistent sandbox state across runs (only with --sandbox agentfs).",
    )
    access_group.add_argument(
        "--sandbox-strict-read",
        action="store_true",
        default=_UNSET,
        help="Enable strict read isolation in AgentFS sandbox (requires agentfs with strict read support).",
    )
    access_group.add_argument(
        "--no-sandbox-auto-session",
        action="store_true",
        default=_UNSET,
        help="Disable automatic session ID generation for AgentFS sandbox.",
    )
    prompt_group.add_argument(
        "--no-skills",
        action="store_true",
        default=_UNSET,
        help="Don't load or discover any skills.",
    )
    review_group.add_argument(
        "--objective",
        type=str,
        default=_UNSET,
        metavar="FILE",
        help="Read the task description from FILE instead of SWIVAL_TASK env var (reviewer mode).",
    )
    behavior_group.add_argument(
        "--proactive-summaries",
        action="store_true",
        default=_UNSET,
        help="Periodically summarize conversation to preserve context across compaction events.",
    )
    output_group.add_argument(
        "--project",
        action="store_true",
        default=False,
        help="With --init-config, write to <base-dir>/swival.toml instead of global config.",
    )
    provider_group.add_argument(
        "--provider",
        choices=[
            "lmstudio",
            "huggingface",
            "openrouter",
            "generic",
            "google",
            "chatgpt",
            "command",
        ],
        default=_UNSET,
        help="LLM provider: lmstudio (local), huggingface (HF API), openrouter (multi-provider API), generic (any OpenAI-compatible server), google (Gemini via OpenAI-compatible endpoint), chatgpt (ChatGPT Plus/Pro subscription via OAuth), command (external command as LLM, --model is the command to run).",
    )
    output_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=_UNSET,
        help="Suppress all diagnostics; only print the final result.",
    )
    modes.add_argument(
        "--repl",
        action="store_true",
        help="Start an interactive session instead of answering a single question.",
    )
    review_group.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="FILE",
        help="Write a JSON evaluation report to FILE. Incompatible with --repl.",
    )
    review_group.add_argument(
        "--review-prompt",
        type=str,
        default=_UNSET,
        help="Custom instructions appended to the built-in review prompt (reviewer mode).",
    )
    review_group.add_argument(
        "--reviewer",
        metavar="COMMAND",
        default=_UNSET,
        help="Reviewer command (shell-split). Called after each answer with base_dir as argument "
        "and answer on stdin. Exit 0=accept, 1=retry with stdout as feedback, 2=reviewer error.",
    )
    review_group.add_argument(
        "--reviewer-mode",
        action="store_true",
        default=False,
        help="Run as a reviewer: read base_dir from positional arg, answer from stdin, "
        "call LLM to judge, exit 0/1/2.",
    )
    review_group.add_argument(
        "--self-review",
        action="store_true",
        default=_UNSET,
        help="Use a second swival instance as reviewer, inheriting provider, model, "
        "skills-dir, and yolo settings from the current invocation.",
    )
    provider_group.add_argument(
        "--seed",
        type=int,
        default=_UNSET,
        help="Random seed for reproducible outputs (optional, model support varies).",
    )
    prompt_group.add_argument(
        "--skills-dir",
        action="append",
        default=None,
        help="Additional directory to scan for skills (can be repeated).",
    )

    system_prompt_group = prompt_group.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system-prompt",
        type=str,
        default=_UNSET,
        help="System prompt to include.",
    )
    system_prompt_group.add_argument(
        "--no-system-prompt",
        action="store_true",
        default=_UNSET,
        help="Omit the system message entirely.",
    )

    provider_group.add_argument(
        "--temperature",
        type=float,
        default=_UNSET,
        help="Sampling temperature (default: provider default).",
    )
    provider_group.add_argument(
        "--top-p",
        type=float,
        default=_UNSET,
        help="Top-p (nucleus) sampling (default: 1.0).",
    )
    server_group.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help="Start an A2A server exposing this agent as an endpoint.",
    )
    server_group.add_argument(
        "--serve-host",
        type=str,
        default="0.0.0.0",
        help="Host for the A2A server (default: 0.0.0.0). Only used with --serve.",
    )
    server_group.add_argument(
        "--serve-port",
        type=int,
        default=8080,
        help="Port for the A2A server (default: 8080). Only used with --serve.",
    )
    server_group.add_argument(
        "--serve-auth-token",
        type=str,
        default=None,
        help="Bearer token for A2A server auth. Only used with --serve.",
    )
    server_group.add_argument(
        "--serve-name",
        type=str,
        default=_UNSET,
        help="Custom agent name for the A2A agent card. Only used with --serve.",
    )
    server_group.add_argument(
        "--serve-description",
        type=str,
        default=_UNSET,
        help="Custom agent description for the A2A agent card. Only used with --serve.",
    )
    review_group.add_argument(
        "--verify",
        type=str,
        default=_UNSET,
        metavar="FILE",
        help="Read verification/acceptance criteria from FILE (reviewer mode).",
    )
    output_group.add_argument(
        "--version",
        action="store_true",
        help="Print the version and exit.",
    )
    access_group.add_argument(
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
    import signal

    def _sigterm_handler(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _sigterm_handler)

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

        # Snapshot whether these were explicitly on CLI (before config merge)
        reviewer_from_cli = args.reviewer is not _UNSET
        self_review_from_cli = args.self_review is not _UNSET and args.self_review

        base_dir = Path(args.question).resolve()
        try:
            file_config = load_config(base_dir)
        except _ConfigError as e:
            parser.error(str(e))
        apply_config_to_args(args, file_config)

        # Config inheritance hazard: clear keys that don't apply in reviewer mode
        if reviewer_from_cli:
            parser.error("--reviewer-mode and --reviewer cannot be used together")
        args.reviewer = None

        if self_review_from_cli:
            parser.error("--self-review is incompatible with --reviewer-mode")
        args.self_review = False

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

    # Stash A2A servers from TOML config before apply_config_to_args strips them
    args._a2a_servers_toml = file_config.pop("a2a_servers", None)

    # Stash serve_skills from TOML config before apply_config_to_args strips them
    args._serve_skills_config = file_config.pop("serve_skills", None)

    apply_config_to_args(args, file_config)

    # Derived values (after all sentinels are resolved)
    args.verbose = not args.quiet

    # Synthesize reviewer command from current args when --self-review is set
    if args.self_review:
        if args.reviewer:
            parser.error("--self-review and --reviewer cannot be used together")
        args.reviewer = _build_self_review_cmd(args)

    # --- A2A serve mode ---
    _is_serve = getattr(args, "serve", False)

    # Read question from stdin if not provided and stdin is piped
    if (
        not args.repl
        and not _is_serve
        and args.question is None
        and not sys.stdin.isatty()
    ):
        args.question = sys.stdin.read().strip()
        if not args.question:
            parser.error("question is required (stdin was empty)")

    if not args.repl and not _is_serve and args.question is None:
        parser.error("question is required (or use --repl)")
    if args.report and args.repl:
        parser.error("--report is incompatible with --repl")
    if args.reviewer and args.repl:
        parser.error("--reviewer is incompatible with --repl")
    if args.self_review and args.repl:
        parser.error("--self-review is incompatible with --repl")

    fmt.init(color=args.color, no_color=args.no_color)

    # Validation: --sandbox-session requires --sandbox agentfs
    if args.sandbox_session is not None and args.sandbox != "agentfs":
        parser.error("--sandbox-session requires --sandbox agentfs")

    # Validation: --sandbox-strict-read requires --sandbox agentfs
    if args.sandbox_strict_read and args.sandbox != "agentfs":
        parser.error("--sandbox-strict-read requires --sandbox agentfs")

    # Validation: max_review_rounds >= 0
    if args.max_review_rounds < 0:
        parser.error("--max-review-rounds must be >= 0")

    # Validation: retries >= 1
    if args.retries < 1:
        parser.error("--retries must be >= 1")

    # Validation: max_output_tokens <= max_context_tokens
    if (
        args.max_context_tokens is not None
        and args.max_output_tokens > args.max_context_tokens
    ):
        parser.error(
            "--max-output-tokens must be <= --max-context-tokens when both are specified."
        )

    # AgentFS sandbox: re-exec inside agentfs if requested.
    # This replaces the current process on success (does not return).
    from .sandbox_agentfs import (
        maybe_reexec,
        is_sandboxed,
        get_agentfs_version,
        get_agentfs_session,
        diff_hint,
    )

    maybe_reexec(
        sandbox=args.sandbox,
        sandbox_session=args.sandbox_session,
        base_dir=str(Path(args.base_dir).resolve()),
        add_dirs=getattr(args, "add_dir", []) or [],
        sandbox_strict_read=args.sandbox_strict_read,
        sandbox_auto_session=not args.no_sandbox_auto_session,
    )

    if args.sandbox == "agentfs" and is_sandboxed() and args.verbose:
        session = get_agentfs_session()
        parts = ["Sandbox: agentfs"]
        if session:
            parts.append(f"(session: {session})")
        fmt.info(" ".join(parts))
        if session:
            fmt.info(
                f"Resume: swival --sandbox agentfs --sandbox-session {session} ..."
            )

    # --- A2A serve mode ---
    # Placed after validations and AgentFS re-exec so all CLI checks apply.
    if _is_serve:
        from .config import args_to_session_kwargs

        session_kwargs = args_to_session_kwargs(args, str(base_dir))

        # MCP servers
        if not getattr(args, "no_mcp", False):
            mcp_servers = _resolve_mcp_servers(args, base_dir)
            if mcp_servers:
                session_kwargs["mcp_servers"] = mcp_servers

        # A2A client servers (outbound, for the served agent to call)
        if not getattr(args, "no_a2a", False):
            a2a_servers = _resolve_a2a_servers(args)
            if a2a_servers:
                session_kwargs["a2a_servers"] = a2a_servers

        from .a2a_server import A2aServer

        serve_skills = getattr(args, "_serve_skills_config", None)

        server = A2aServer(
            session_kwargs=session_kwargs,
            host=args.serve_host,
            port=args.serve_port,
            auth_token=args.serve_auth_token,
            name=args.serve_name if args.serve_name is not _UNSET else None,
            description=args.serve_description
            if args.serve_description is not _UNSET
            else None,
            skills=serve_skills,
        )
        server.serve()
        sys.exit(0)

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
        _session = get_agentfs_session()
        _diff = diff_hint(_session)
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
            sandbox_mode=args.sandbox,
            sandbox_session=_session or args.sandbox_session,
            sandbox_strict_read=args.sandbox_strict_read,
            agentfs_version=get_agentfs_version(),
            diff_hint=_diff,
        )
        try:
            report.write(args.report)
        except OSError as e:
            fmt.error(f"Failed to write report to {args.report}: {e}")
            return
        if args.verbose:
            fmt.info(f"Report written to {args.report}")

    _run_outcome = "success"
    _run_exit_code = 0
    try:
        _run_main(args, report, _write_report, parser)
    except AgentError as e:
        _run_outcome = "error"
        _run_exit_code = 1
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
    except SystemExit as e:
        _run_exit_code = e.code if isinstance(e.code, int) else 1
        _run_outcome = {
            0: "success",
            2: "exhausted",
            130: "interrupted",
            143: "interrupted",
        }.get(_run_exit_code, "error")
        raise
    finally:
        # --- Lifecycle exit hook ---
        _lc_cmd = getattr(args, "_lifecycle_cmd", None)
        _lc_no = getattr(args, "_no_lifecycle", False)
        _lc_error = None
        if _lc_cmd and not _lc_no:
            from .lifecycle import run_lifecycle_hook, LifecycleError

            _lc_report_path = getattr(args, "report", None)
            try:
                _lc_result = run_lifecycle_hook(
                    _lc_cmd,
                    "exit",
                    str(Path(args.base_dir).resolve()),
                    timeout=getattr(args, "_lifecycle_timeout", 300),
                    fail_closed=getattr(args, "_lifecycle_fail_closed", False),
                    provider=args.provider,
                    model=getattr(args, "_resolved_model_id", None),
                    git_meta=getattr(args, "_lifecycle_git_meta", None),
                    report_path=_lc_report_path
                    if isinstance(_lc_report_path, str)
                    else None,
                    outcome=_run_outcome,
                    exit_code=_run_exit_code,
                    verbose=args.verbose,
                )
                if args.verbose and _lc_result:
                    fmt.info(
                        f"Lifecycle exit hook completed in "
                        f"{_lc_result['duration']:.1f}s"
                    )
                # Record exit hook in report and re-persist the file
                if report and _lc_result:
                    report.record_lifecycle(_lc_result)
                    if _lc_report_path and report._last_report is not None:
                        report._last_report["timeline"] = report.events
                        report._last_report["stats"]["lifecycle"] = (
                            report.lifecycle_events
                        )
                        try:
                            report.write(_lc_report_path)
                        except OSError:
                            pass
            except LifecycleError as e:
                _lc_error = e
                fmt.error(f"lifecycle exit hook failed (fail-closed): {e}")
                if report:
                    report.record_lifecycle(
                        {
                            "event": "exit",
                            "exit_code": None,
                            "duration": 0,
                            "error": str(e),
                        }
                    )

            # If fail-closed exit hook failed, amend the report to reflect it
            if _lc_error is not None and report and _lc_report_path:
                _write_report(
                    "error",
                    exit_code=1,
                    error_message=f"lifecycle exit hook failed: {_lc_error}",
                    model_id=getattr(
                        args, "_resolved_model_id", args.model or "unknown"
                    ),
                    skills_catalog=getattr(args, "_resolved_skills", None),
                    instructions_loaded=getattr(args, "_resolved_instructions", None),
                    review_rounds=getattr(args, "_review_rounds", 0),
                )

        _cache = getattr(args, "_llm_cache", None)
        if _cache is not None:
            _cache.close()
        _shield = getattr(args, "_secret_shield", None)
        if _shield is not None:
            _shield.destroy()
        _mcp = getattr(args, "_mcp_manager", None)
        if _mcp is not None:
            _mcp.close()
        _a2a = getattr(args, "_a2a_manager", None)
        if _a2a is not None:
            _a2a.close()

        if _lc_error is not None:
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
    provider_name = provider
    llm_provider = provider
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
            raise ConfigError(f"--model is required when --provider is {provider_name}")
        if not base_url:
            raise ConfigError(
                f"--base-url is required when --provider is {provider_name}"
            )
        stripped = base_url.rstrip("/")
        api_base = stripped if stripped.endswith("/v1") else f"{stripped}/v1"
        model_id = model
        context_length = max_context_tokens
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")

    elif provider == _GOOGLE_PROVIDER:
        if not model:
            raise ConfigError("--model is required when --provider is google")
        # Route through Google's OpenAI-compatible endpoint instead of
        # LiteLLM's native gemini adapter, which is unstable with newer
        # models (empty choices, 500s).  See GitHub issue #6.
        _GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
        llm_provider = "generic"
        api_base = base_url or _GOOGLE_OPENAI_BASE
        model_id = model
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise ConfigError(
                "--api-key, GEMINI_API_KEY, or OPENAI_API_KEY env var required for google provider"
            )
        context_length = max_context_tokens
        if context_length is None:
            try:
                import litellm

                _bare = model_id.removeprefix("gemini/")
                _model_str = f"gemini/{_bare}"
                info = litellm.get_model_info(_model_str)
                context_length = info.get("max_input_tokens")
            except Exception:
                pass

    elif provider == "chatgpt":
        if not model:
            raise ConfigError(
                "--model is required when --provider is chatgpt. "
                f"See {CHATGPT_PROVIDER_DOCS_URL} for the current supported model names."
            )
        api_base = base_url
        model_id = model
        resolved_key = api_key or os.environ.get("CHATGPT_API_KEY")
        context_length = max_context_tokens
        if context_length is None:
            try:
                import litellm

                _bare = model_id.removeprefix("chatgpt/").removeprefix("chatgpt/")
                _model_str = f"chatgpt/{_bare}"
                info = litellm.get_model_info(_model_str)
                context_length = info.get("max_input_tokens")
            except Exception:
                pass

    elif provider == "command":
        if not model or not model.strip():
            raise ConfigError(
                "--model is required for 'command' provider (the command to run)"
            )
        parts = shlex.split(model)
        if not parts:
            raise ConfigError("--model is empty for 'command' provider")
        if not shutil.which(parts[0]):
            raise ConfigError(f"command not found: {parts[0]}")
        model_id = model
        api_base = None
        resolved_key = None
        context_length = max_context_tokens
        llm_provider = "command"

    else:
        raise ConfigError(f"unknown provider: {provider!r}")

    llm_kwargs = {
        "provider": llm_provider,
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
        skill_tool = copy.deepcopy(USE_SKILL_TOOL)
        names_list = sorted(skills_catalog)
        # Machine-readable constraint — always set.
        skill_tool["function"]["parameters"]["properties"]["name"]["enum"] = names_list
        # Human-readable hint in description — keep short for large catalogs.
        names_str = ", ".join(names_list)
        if len(names_str) <= 200:
            skill_tool["function"]["description"] = (
                f"Activate a skill to get detailed instructions. "
                f"Available skills: {names_str}. "
                f"Use this instead of searching for SKILL.md files."
            )
        else:
            skill_tool["function"]["description"] = (
                f"Activate a skill to get detailed instructions. "
                f"{len(names_list)} skills available (see enum). "
                f"Use this instead of searching for SKILL.md files."
            )
        tools.append(skill_tool)
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


_COMMAND_PROVIDER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and concisely."
)

# ---------------------------------------------------------------------------
# Interaction-policy directives
# ---------------------------------------------------------------------------
# Substituted into system_prompt.txt placeholders {{AUTONOMY_DIRECTIVE}} and
# {{AMBIGUITY_DIRECTIVE}}.  The sentinels use double-brace + SCREAMING_SNAKE
# to minimise accidental collision with user instructions or memory text.
# If they *do* appear in injected content the .replace() will still fire —
# an acknowledged edge case considered acceptable given the sentinel style.

_InteractionPolicy = Literal["autonomous", "interactive"]

_AUTONOMY_DIRECTIVES: dict[_InteractionPolicy, str] = {
    "autonomous": (
        "You solve tasks autonomously using the tools provided. "
        "Keep going until the task is fully complete \u2014 do not stop to ask for "
        'confirmation or clarification. Never ask "should I continue?" \u2014 just '
        "continue. If the task is ambiguous, use `think` to reason through the "
        "possible interpretations against the codebase context, pick the most "
        "likely intent, and briefly state your choice before acting."
    ),
    "interactive": (
        "You solve tasks using the tools provided. Keep going until the task is "
        "fully complete. If a request is genuinely ambiguous and you cannot "
        "determine the intent from codebase context, briefly ask the user to "
        "clarify before acting. For straightforward tasks, act without asking. "
        'Never ask "should I continue?" mid-task \u2014 just continue.'
    ),
}

_AMBIGUITY_DIRECTIVES: dict[_InteractionPolicy, str] = {
    "autonomous": (
        "- If the task is ambiguous, use `think` to reason through the possible "
        "interpretations against the codebase context, pick the most likely intent, "
        "and briefly state your choice before acting."
    ),
    "interactive": (
        "- If the task is genuinely ambiguous, ask the user a brief clarifying "
        "question. For minor ambiguities, pick the most likely intent and state "
        "your choice."
    ),
}


def _apply_interaction_policy(
    system_content: str,
    policy: _InteractionPolicy,
) -> str:
    """Replace autonomy placeholders with policy-specific directives."""
    return system_content.replace(
        "{{AUTONOMY_DIRECTIVE}}", _AUTONOMY_DIRECTIVES[policy]
    ).replace("{{AMBIGUITY_DIRECTIVE}}", _AMBIGUITY_DIRECTIVES[policy])


def build_system_prompt(
    base_dir: str,
    system_prompt: str | None,
    no_system_prompt: bool,
    no_instructions: bool,
    no_memory: bool,
    skills_catalog: dict,
    yolo: bool,
    resolved_commands: dict[str, str],
    verbose: bool,
    config_dir: "Path | None" = None,
    mcp_tool_info: dict | None = None,
    a2a_tool_info: dict | None = None,
    no_continue: bool = False,
    memory_full: bool = False,
    user_query: str | None = None,
    report: "ReportCollector | None" = None,
    provider: str | None = None,
    command_tool_schemas: list | None = None,
) -> tuple[str | None, list[str]]:
    """Assemble full system prompt with instructions, date, skills, memory.

    Returns (system_prompt_text, instructions_loaded).
    system_prompt_text is None if no_system_prompt is True.
    """
    if no_system_prompt:
        return None, []

    instructions_loaded: list[str] = []
    if system_prompt:
        system_content = system_prompt
    elif provider == "command":
        system_content = _COMMAND_PROVIDER_SYSTEM_PROMPT
        if command_tool_schemas:
            catalog = _render_swival_tool_catalog(command_tool_schemas)
            system_content += (
                "\n\n"
                "In addition to your own tools, you have access to external tools "
                "provided by the orchestrator. To call one, emit a block in your "
                "response:\n\n"
                '<swival:call id="UNIQUE_ID" name="tool_name">\n'
                '{"param": "value"}\n'
                "</swival:call>\n\n"
                "Each call must have a unique id (e.g. c1, c2, c3). Do NOT use "
                "your own tool-calling mechanism for these — they must appear as "
                "literal text in your response. The orchestrator will execute them "
                "and provide results in [swival_result] sections.\n\n"
                "Continue working until you can give a final answer with no "
                "<swival:call> blocks.\n\n"
                "Available external tools:\n\n" + catalog
            )
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

        if not no_memory:
            memory_text = load_memory(
                base_dir,
                verbose=verbose,
                memory_full=memory_full,
                user_query=user_query,
                report=report,
            )
            if memory_text:
                system_content += "\n\n" + memory_text

    now = datetime.now().astimezone()
    system_content += f"\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M %Z')}"

    # Tool-related prompt sections are skipped for the command provider,
    # which disables tool calling entirely.
    if provider != "command":
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

        if a2a_tool_info and not system_prompt:
            system_content += "\n\n" + _format_a2a_tool_info(a2a_tool_info)

    # Load continue-here file from a previous interrupted session
    if not no_continue:
        from .continue_here import load_continue_file, format_continue_prompt

        continue_content = load_continue_file(base_dir)
        if continue_content:
            system_content += "\n\n" + format_continue_prompt(continue_content)
            if verbose:
                fmt.info("Loaded continue file from previous session")

    return system_content, instructions_loaded


def _format_external_tool_info(
    heading: str, preamble: str, tool_info: dict[str, list[tuple[str, str]]]
) -> str:
    """Format external tool info (MCP or A2A) for the system prompt."""
    lines = [f"## {heading}", "", preamble, ""]
    for server_name, tools in sorted(tool_info.items()):
        lines.append(f"**{server_name}**:")
        for namespaced_name, description in tools:
            desc = f": {description}" if description else ""
            lines.append(f"- `{namespaced_name}`{desc}")
        lines.append("")
    return "\n".join(lines)


def _format_mcp_tool_info(tool_info: dict[str, list[tuple[str, str]]]) -> str:
    return _format_external_tool_info(
        "MCP Tools", "Tools provided by external MCP servers:", tool_info
    )


def _format_a2a_tool_info(tool_info: dict[str, list[tuple[str, str]]]) -> str:
    return _format_external_tool_info(
        "A2A Tools",
        "Tools provided by remote A2A agents. Each tool accepts a natural-language\n"
        "message. For multi-turn conversations, pass back the contextId from a\n"
        "previous result. For input-required resumption, pass both contextId and taskId.",
        tool_info,
    )


def _show_agentfs_diff_hint(args) -> None:
    """Show agentfs diff command hint on exit (verbose mode only)."""
    if args.sandbox != "agentfs" or not args.verbose:
        return
    from .sandbox_agentfs import is_sandboxed, get_agentfs_session, diff_hint

    if not is_sandboxed():
        return
    hint = diff_hint(get_agentfs_session())
    if hint:
        fmt.sandbox_hint(f"Review changes: {hint}")


def _resolve_mcp_servers(args, base_dir) -> dict | None:
    """Resolve MCP server configs from TOML + JSON sources. Returns merged dict or None."""
    from .config import load_mcp_json, merge_mcp_configs

    toml_servers = getattr(args, "_mcp_servers_toml", None)
    json_servers = None

    mcp_config_path = getattr(args, "mcp_config", None)
    if mcp_config_path:
        p = Path(mcp_config_path)
        if not p.is_file():
            raise ConfigError(f"--mcp-config file not found: {mcp_config_path}")
        json_servers = load_mcp_json(p)
    else:
        default_mcp = Path(base_dir).resolve() / ".mcp.json"
        if default_mcp.is_file():
            json_servers = load_mcp_json(default_mcp)

    return merge_mcp_configs(toml_servers, json_servers) or None


def _resolve_a2a_servers(args) -> dict | None:
    """Resolve A2A server configs from TOML + config file. Returns merged dict or None."""
    a2a_servers = getattr(args, "_a2a_servers_toml", None) or {}

    a2a_config_path = getattr(args, "a2a_config", None)
    if a2a_config_path:
        from .config import load_a2a_config

        p = Path(a2a_config_path)
        if not p.is_file():
            raise ConfigError(f"--a2a-config file not found: {a2a_config_path}")
        file_servers = load_a2a_config(p)
        file_servers.update(a2a_servers)
        a2a_servers = file_servers

    return a2a_servers or None


def _validate_external_command(cmd_string: str, label: str) -> None:
    """Validate that a shell command string is well-formed and the executable exists."""
    import shlex

    try:
        parts = shlex.split(cmd_string)
    except ValueError as e:
        raise AgentError(f"malformed {label} command: {e}")
    if not parts:
        raise AgentError(f"{label} command is empty")
    exe = parts[0]
    if not shutil.which(exe):
        p = Path(exe).resolve()
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise AgentError(f"{label} executable not found or not executable: {exe}")


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
    if getattr(args, "reasoning_effort", None) is not None:
        llm_kwargs["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "sanitize_thinking", False):
        llm_kwargs["sanitize_thinking"] = True
    llm_kwargs["max_retries"] = args.retries

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
        # Auto-grant read access to external skill directories
        for skill in skills_catalog.values():
            if not skill.is_local and skill.path not in skill_read_roots:
                skill_read_roots.append(skill.path)
    args._resolved_skills = skills_catalog

    tools = build_tools(resolved_commands, skills_catalog, yolo)

    # Initialize MCP servers
    mcp_manager = None
    mcp_tool_info = {}
    if not getattr(args, "no_mcp", False):
        from .mcp_client import McpManager

        mcp_servers = _resolve_mcp_servers(args, base_dir)
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
    args._mcp_manager = mcp_manager

    # Initialize A2A agents
    a2a_manager = None
    a2a_tool_info = {}
    if not getattr(args, "no_a2a", False):
        from .a2a_client import A2aManager

        a2a_servers = _resolve_a2a_servers(args)
        if a2a_servers:
            a2a_manager = A2aManager(a2a_servers, verbose=args.verbose)
            a2a_manager.start()
            a2a_tools = a2a_manager.list_tools()
            if a2a_tools:
                tools.extend(a2a_tools)
            a2a_tool_info = a2a_manager.get_tool_info()
    args._a2a_manager = a2a_manager

    # --- Secret encryption lifecycle ---
    secret_shield = None
    if getattr(args, "encrypt_secrets", False):
        from .secrets import SecretShield

        secret_shield = SecretShield.from_config(
            key_hex=getattr(args, "encrypt_secrets_key", None),
            tweak_str=getattr(args, "encrypt_secrets_tweak", None),
            extra_patterns=getattr(args, "encrypt_secrets_patterns", None),
        )
        args._secret_shield = secret_shield  # stash for cleanup

    # --- Cache lifecycle ---
    llm_cache = None
    if getattr(args, "cache", False):
        from .cache import open_cache

        llm_cache = open_cache(base_dir, getattr(args, "cache_dir", None))
        args._llm_cache = llm_cache  # stash for cleanup in outer handler
        if args.verbose:
            stats = llm_cache.stats()
            fmt.info(f"Cache: {llm_cache.db_path} ({stats['entries']} entries)")

    # --- Lifecycle startup hook ---
    lifecycle_cmd = getattr(args, "lifecycle_command", None)
    lifecycle_timeout = getattr(args, "lifecycle_timeout", 300)
    lifecycle_fail_closed = getattr(args, "lifecycle_fail_closed", False)
    no_lifecycle = getattr(args, "no_lifecycle", False)
    lifecycle_startup_result = None
    lifecycle_git_meta = None

    if lifecycle_cmd and not no_lifecycle:
        _validate_external_command(lifecycle_cmd, "lifecycle_command")
        from .lifecycle import run_lifecycle_hook, _git_metadata

        lifecycle_git_meta = _git_metadata(base_dir)
        lifecycle_startup_result = run_lifecycle_hook(
            lifecycle_cmd,
            "startup",
            base_dir,
            timeout=lifecycle_timeout,
            fail_closed=lifecycle_fail_closed,
            provider=args.provider,
            model=model_id,
            git_meta=lifecycle_git_meta,
            verbose=args.verbose,
        )
        if args.verbose and lifecycle_startup_result:
            fmt.info(
                f"Lifecycle startup hook completed in "
                f"{lifecycle_startup_result['duration']:.1f}s"
            )
        if report and lifecycle_startup_result:
            report.record_lifecycle(lifecycle_startup_result)

    # Stash lifecycle state on args for exit hook
    args._lifecycle_cmd = lifecycle_cmd
    args._lifecycle_timeout = lifecycle_timeout
    args._lifecycle_fail_closed = lifecycle_fail_closed
    args._no_lifecycle = no_lifecycle
    args._lifecycle_git_meta = lifecycle_git_meta
    args._lifecycle_startup_result = lifecycle_startup_result

    # Build list of tool schemas exposable to command provider (MCP/A2A/skills).
    _command_tool_schemas = (
        _filter_command_tool_schemas(tools) or None
        if llm_kwargs.get("provider") == "command"
        else None
    )

    system_content, instructions_loaded = build_system_prompt(
        base_dir=base_dir,
        system_prompt=args.system_prompt,
        no_system_prompt=args.no_system_prompt,
        no_instructions=args.no_instructions,
        no_memory=getattr(args, "no_memory", False),
        memory_full=getattr(args, "memory_full", False),
        skills_catalog=skills_catalog,
        yolo=yolo,
        resolved_commands=resolved_commands,
        verbose=args.verbose,
        config_dir=getattr(args, "config_dir", None),
        mcp_tool_info=mcp_tool_info,
        a2a_tool_info=a2a_tool_info,
        no_continue=getattr(args, "no_continue", False),
        user_query=getattr(args, "question", None),
        report=report,
        provider=llm_kwargs.get("provider"),
        command_tool_schemas=_command_tool_schemas,
    )
    policy: _InteractionPolicy = "interactive" if args.repl else "autonomous"
    if system_content is not None:
        system_content = _apply_interaction_policy(system_content, policy)
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
        a2a_manager=a2a_manager,
        cache=llm_cache,
        secret_shield=secret_shield,
    )

    # Validate and thread llm_filter
    llm_filter_cmd = getattr(args, "llm_filter", None)
    if llm_filter_cmd:
        _validate_external_command(llm_filter_cmd, "llm_filter")
        loop_kwargs["llm_filter"] = llm_filter_cmd

    if getattr(args, "proactive_summaries", False):
        loop_kwargs["compaction_state"] = CompactionState()

    no_history = getattr(args, "no_history", False)
    no_continue = getattr(args, "no_continue", False)
    _continue_here = not no_continue
    loop_kwargs["continue_here"] = _continue_here

    # Validate reviewer executable at startup
    reviewer_cmd = None
    if args.reviewer:
        _validate_external_command(args.reviewer, "reviewer")
        reviewer_cmd = args.reviewer

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
            # Pass API key via provider-specific env var (avoid CLI exposure)
            if args.self_review and args.api_key:
                env_var = _PROVIDER_KEY_ENV.get(args.provider)
                if env_var:
                    reviewer_env[env_var] = args.api_key
            # Pass encryption key via env var (avoid ps exposure)
            if getattr(args, "encrypt_secrets", False):
                key_hex = getattr(args, "encrypt_secrets_key", None)
                if key_hex:
                    from .secrets import ENCRYPT_KEY_ENV

                    reviewer_env[ENCRYPT_KEY_ENV] = key_hex

        while True:
            try:
                answer, exhausted = run_agent_loop(
                    messages,
                    tools,
                    **loop_kwargs,
                    report=report,
                    turn_offset=turn_offset,
                )
            except (KeyboardInterrupt, SystemExit) as exc:
                is_term = isinstance(exc, SystemExit)
                exit_code = exc.code if is_term else 130
                fmt.warning("terminated." if is_term else "interrupted.")
                if _continue_here:
                    from .continue_here import write_continue_file

                    write_continue_file(
                        base_dir,
                        messages,
                        todo_state=todo_state,
                        snapshot_state=snapshot_state,
                        thinking_state=thinking_state,
                    )
                sys.exit(exit_code)

            if not reviewer_cmd or answer is None or exhausted:
                break

            review_round += 1
            args._review_rounds = review_round
            if args.verbose:
                fmt.review_sending(review_round)

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
                retry_msg = (
                    f"[REVIEWER FEEDBACK — Round {review_round}]\n"
                    "A reviewer has evaluated your answer and requested changes. "
                    "You MUST address the feedback below by taking concrete "
                    "tool-call actions — do not simply rewrite your previous "
                    "answer. If the task cannot be completed as requested, use "
                    "tools to gather evidence, then report the failure clearly.\n\n"
                    f"{review_text}"
                )
                messages.append({"role": "user", "content": retry_msg})
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
        _show_agentfs_diff_hint(args)
        if exhausted:
            if args.verbose:
                fmt.warning("max turns reached, agent stopped.")
            sys.exit(2)
        return

    # REPL path
    if args.question:
        messages.append({"role": "user", "content": args.question})
        try:
            answer, exhausted = run_agent_loop(messages, tools, **loop_kwargs)
        except KeyboardInterrupt:
            fmt.warning("interrupted during initial question.")
            if _continue_here:
                from .continue_here import write_continue_file

                write_continue_file(
                    base_dir,
                    messages,
                    todo_state=todo_state,
                    snapshot_state=snapshot_state,
                    thinking_state=thinking_state,
                )
            answer, exhausted = None, False
        except SystemExit as exc:
            fmt.warning("terminated during initial question.")
            if _continue_here:
                from .continue_here import write_continue_file

                write_continue_file(
                    base_dir,
                    messages,
                    todo_state=todo_state,
                    snapshot_state=snapshot_state,
                    thinking_state=thinking_state,
                )
            raise SystemExit(exc.code)
        if not no_history and answer:
            append_history(base_dir, args.question, answer, diagnostics=args.verbose)
        if answer is not None:
            print(answer)
        if exhausted and args.verbose:
            fmt.warning(
                "max turns reached for initial question. Use /continue to resume."
            )

    repl_loop(messages, tools, **loop_kwargs, no_history=no_history)
    _show_agentfs_diff_hint(args)


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
    scratch_dir: str | None = None,
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
    a2a_manager=None,
    continue_here: bool = True,
    cache=None,
    secret_shield=None,
    llm_filter=None,
    event_callback=None,
    cancel_flag=None,
) -> tuple[str | None, bool]:
    """Run the tool-calling loop until a final answer or max turns.

    Mutates `messages` in place (appends assistant/tool messages,
    in-place compaction on overflow).
    Returns (final_answer, exhausted). final_answer is the last
    assistant text (may be None). exhausted is True if max_turns hit.
    """
    # Thread cache, secret_shield, and llm_filter into llm_kwargs (for main
    # loop calls via **llm_kwargs) and create a wrapper for secondary call
    # sites that pass call_llm as a function reference (compaction summaries,
    # proactive checkpoints, continue-file enrichment).
    _call_llm_for_secondary = call_llm
    if llm_filter is not None:
        llm_kwargs = {**llm_kwargs, "llm_filter": llm_filter}
    if secret_shield is not None:
        llm_kwargs = {**llm_kwargs, "secret_shield": secret_shield}
    if cache is not None:
        llm_kwargs = {**llm_kwargs, "cache": cache}

    if cache is not None or secret_shield is not None or llm_filter is not None:

        def _call_llm_for_secondary(*args, **kwargs):
            if llm_filter is not None:
                kwargs.setdefault("llm_filter", llm_filter)
                kwargs.setdefault("call_kind", "summary")
            if cache is not None:
                kwargs.setdefault("cache", cache)
            if secret_shield is not None:
                kwargs.setdefault("secret_shield", secret_shield)
            return call_llm(*args, **kwargs)

    consecutive_errors: dict[str, tuple[str, int]] = {}
    turns = 0
    think_used = False
    think_nudge_fired = False
    todo_last_used = 0
    snapshot_read_streak = 0
    snapshot_nudge_fired = False
    _vision_pending = False
    _provider_retries = 0
    loop_start = time.monotonic()

    def _emit(kind: str, data: dict) -> None:
        if event_callback is not None:
            try:
                event_callback(kind, data)
            except Exception:
                pass

    # Reset dirty state only if the last message is a user message
    # (new scope boundary). Skip on /continue where the last message
    # is an assistant or tool message from the previous run.
    if snapshot_state is not None:
        last_role = _msg_role(messages[-1]) if messages else ""
        if last_role == "user":
            snapshot_state.reset_dirty()

    _snapshot_strip_marker = "\n\n" + SNAPSHOT_HISTORY_SENTINEL

    # Strip view_image from tools if the model is known to lack vision support
    provider = llm_kwargs.get("provider", "lmstudio")
    if provider != "command":
        model_str = _resolve_model_str(provider, model_id)
        if _model_supports_vision(model_str) is False:
            tools = [
                t for t in tools if t.get("function", {}).get("name") != "view_image"
            ]
    effective_tools = None if provider == "command" else tools

    # Build command_tool_kwargs for command provider tool-calling support
    _command_tool_schemas = (
        _filter_command_tool_schemas(tools) if provider == "command" else []
    )
    if _command_tool_schemas:
        _handle_tc_kwargs = dict(
            base_dir=base_dir,
            thinking_state=thinking_state,
            verbose=verbose,
            resolved_commands=resolved_commands,
            skills_catalog=skills_catalog,
            skill_read_roots=skill_read_roots,
            extra_write_roots=extra_write_roots,
            yolo=yolo,
            file_tracker=file_tracker,
            todo_state=todo_state,
            snapshot_state=snapshot_state,
            mcp_manager=mcp_manager,
            a2a_manager=a2a_manager,
            messages=None,  # inner loop manages its own transcript
            image_stash=None,
            scratch_dir=scratch_dir,
        )
        llm_kwargs = {
            **llm_kwargs,
            "command_tool_kwargs": {
                "handle_tool_call_kwargs": _handle_tc_kwargs,
                "outer_turn": 0,  # updated per-turn below
                "outer_turn_offset": turn_offset,
                "report": report,
                "snapshot_state": snapshot_state,
                "_emit": _emit,
            },
        }

    # Auto-inject skills when user mentions $skill-name.
    # Injected as a synthetic assistant tool_call + tool result pair so that
    # compaction can trim the skill body like any other tool output.
    if skills_catalog and messages:
        last_msg = messages[-1]
        if _msg_role(last_msg) == "user":
            user_text = _msg_content(last_msg) or ""
            if "$" in user_text:
                from .skills import inject_skill_mentions

                activations = inject_skill_mentions(
                    user_text, skills_catalog, skill_read_roots
                )
                if activations:
                    import uuid as _uuid

                    tool_calls = []
                    _uid = _uuid.uuid4().hex[:8]
                    for name, _result in activations:
                        tc_id = f"auto_skill_{name}_{_uid}"
                        tool_calls.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": "use_skill",
                                    "arguments": json.dumps({"name": name}),
                                },
                            }
                        )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                    )
                    for name, result in activations:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"auto_skill_{name}_{_uid}",
                                "content": result,
                            }
                        )
                        if report:
                            succeeded = not result.startswith("error:")
                            report.record_tool_call(
                                turn=0,
                                name="use_skill",
                                arguments={"name": name},
                                succeeded=succeeded,
                                duration=0.0,
                                result_length=len(result),
                                error=result if not succeeded else None,
                            )
                    if verbose:
                        names = [n for n, _ in activations]
                        fmt.info(f"Auto-activated skill(s): {', '.join(names)}")

    while turns < max_turns:
        turns += 1

        # Check cancellation flag
        if cancel_flag is not None and cancel_flag.is_set():
            if verbose:
                fmt.info("Task cancelled by external request.")
            _emit(EVENT_STATUS_UPDATE, {"turn": turns, "cancelled": True})
            return None, True

        _emit(
            EVENT_STATUS_UPDATE,
            {
                "turn": turns,
                "max_turns": max_turns,
                "elapsed": time.monotonic() - loop_start,
            },
        )

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

        token_est = estimate_tokens(messages, effective_tools)
        if verbose:
            fmt.turn_header(turns, max_turns, token_est)

        t0 = time.monotonic()
        try:
            effective_max_output = clamp_output_tokens(
                messages, effective_tools, context_length, max_output_tokens
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
                effective_tools,
                verbose,
            )

            if "command_tool_kwargs" in llm_kwargs:
                llm_kwargs["command_tool_kwargs"]["outer_turn"] = turns
            with (
                fmt.llm_spinner(f"Waiting for LLM (turn {turns}/{max_turns})")
                if verbose
                else nullcontext()
            ):
                _llm_result = call_llm(*_llm_args, **llm_kwargs)
                msg, finish_reason = _llm_result[0], _llm_result[1]
                cmd_activity = _llm_result[2] if len(_llm_result) > 2 else []
                _provider_retries = _llm_result[3] if len(_llm_result) > 3 else 0
        except ContextOverflowError as _coe:
            elapsed = time.monotonic() - t0
            if report:
                report.record_llm_call(
                    turns + turn_offset,
                    elapsed,
                    token_est,
                    "context_overflow",
                    provider_retries=getattr(_coe, "_provider_retries", 0),
                )

            # --- Graduated compaction levels ---
            # Each level is tried in order. If the LLM call succeeds after
            # a compaction step, we break out. If it still overflows, we
            # try the next level. If all levels fail, raise AgentError.
            _llm_summary_kwargs = dict(
                call_llm_fn=_call_llm_for_secondary,
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
                # If an image was just injected, replace it with an
                # explanatory fallback before compaction strips the data
                # silently.  This way the model knows analysis was dropped.
                if _vision_pending:
                    _replace_last_image_message(
                        messages,
                        _IMAGE_SYNTHETIC_PREFIX
                        + " The image was dropped during context compaction "
                        "and could not be analyzed. Inform the user that the "
                        "image could not be processed due to context limits.",
                    )
                    _vision_pending = False
                tokens_before = estimate_tokens(messages, effective_tools)
                messages[:] = compact_fn()
                if snapshot_state is not None:
                    snapshot_state.invalidate_index_checkpoint()
                try:
                    effective_max_output = clamp_output_tokens(
                        messages, effective_tools, context_length, max_output_tokens
                    )
                except ContextOverflowError:
                    tokens_after = estimate_tokens(messages, effective_tools)
                    if report:
                        report.record_compaction(
                            turns + turn_offset, level_name, tokens_before, tokens_after
                        )
                    continue  # try next compaction level
                tokens_after = estimate_tokens(messages, effective_tools)
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
                    effective_tools,
                    verbose,
                )
                t0 = time.monotonic()
                if "command_tool_kwargs" in llm_kwargs:
                    llm_kwargs["command_tool_kwargs"]["outer_turn"] = turns
                try:
                    with (
                        fmt.llm_spinner(
                            f"Waiting for LLM (turn {turns}/{max_turns}, retry after compaction)"
                        )
                        if verbose
                        else nullcontext()
                    ):
                        _llm_result = call_llm(*_llm_args, **llm_kwargs)
                        msg, finish_reason = _llm_result[0], _llm_result[1]
                        cmd_activity = _llm_result[2] if len(_llm_result) > 2 else []
                        _provider_retries = (
                            _llm_result[3] if len(_llm_result) > 3 else 0
                        )
                except ContextOverflowError as _coe:
                    elapsed = time.monotonic() - t0
                    if report:
                        report.record_llm_call(
                            turns + turn_offset,
                            elapsed,
                            tokens_after,
                            "context_overflow",
                            is_retry=True,
                            retry_reason=level_name,
                            provider_retries=getattr(_coe, "_provider_retries", 0),
                        )
                    continue  # try next level
                except AgentError as _ae:
                    elapsed = time.monotonic() - t0
                    if report:
                        report.record_llm_call(
                            turns + turn_offset,
                            elapsed,
                            tokens_after,
                            "error",
                            is_retry=True,
                            retry_reason=level_name,
                            provider_retries=getattr(_ae, "_provider_retries", 0),
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
                            provider_retries=_provider_retries,
                        )
                    break  # success
            else:
                # All compaction levels exhausted.  Last resort: if we still
                # have tools attached, drop them entirely and retry as a plain
                # chat completion.  The model loses all tool-calling ability
                # but can at least produce a text answer.
                _drop_tools_ok = False
                if effective_tools is not None:
                    fmt.warning(
                        "context window exceeded even after compaction — "
                        "dropping all tools and retrying as plain chat"
                    )
                    effective_tools = None
                    # Truncate a bloated system prompt so the user's
                    # actual question can fit in the remaining context.
                    if (
                        context_length
                        and messages
                        and _msg_role(messages[0]) == "system"
                    ):
                        sys_content = _msg_content(messages[0]) or ""
                        max_sys_chars = context_length  # ~1 token/char, generous
                        if len(sys_content) > max_sys_chars:
                            _set_msg_content(
                                messages[0],
                                sys_content[:max_sys_chars]
                                + "\n\n[system prompt truncated to fit context window]",
                            )
                    try:
                        effective_max_output = clamp_output_tokens(
                            messages, None, context_length, max_output_tokens
                        )
                    except ContextOverflowError:
                        pass
                    else:
                        _llm_args = (
                            api_base,
                            model_id,
                            messages,
                            effective_max_output,
                            temperature,
                            top_p,
                            seed,
                            None,
                            verbose,
                        )
                        t0 = time.monotonic()
                        try:
                            with (
                                fmt.llm_spinner(
                                    f"Waiting for LLM (turn {turns}/{max_turns}, no tools)"
                                )
                                if verbose
                                else nullcontext()
                            ):
                                _llm_result = call_llm(*_llm_args, **llm_kwargs)
                                msg, finish_reason = _llm_result[0], _llm_result[1]
                                cmd_activity = (
                                    _llm_result[2] if len(_llm_result) > 2 else []
                                )
                                _provider_retries = (
                                    _llm_result[3] if len(_llm_result) > 3 else 0
                                )
                        except ContextOverflowError:
                            pass
                        else:
                            elapsed = time.monotonic() - t0
                            if verbose:
                                fmt.llm_timing(elapsed, finish_reason)
                            if report:
                                report.record_llm_call(
                                    turns + turn_offset,
                                    elapsed,
                                    estimate_tokens(messages, None),
                                    finish_reason,
                                    is_retry=True,
                                    retry_reason="drop_tools",
                                    provider_retries=_provider_retries,
                                )
                            _drop_tools_ok = True

                if not _drop_tools_ok:
                    if continue_here:
                        from .continue_here import write_continue_file

                        write_continue_file(
                            base_dir,
                            messages,
                            todo_state=todo_state,
                            snapshot_state=snapshot_state,
                            thinking_state=thinking_state,
                        )
                    raise ContextOverflowError(
                        "context window exceeded even after compaction"
                    )

        except AgentError as e:
            if _vision_pending and _is_vision_rejection(e):
                _vision_pending = False
                _replace_last_image_message(
                    messages,
                    _IMAGE_SYNTHETIC_PREFIX + " The image could not be sent "
                    "to the model — it does not support image analysis. "
                    "Please inform the user and suggest a vision-capable model.",
                )
                if verbose:
                    fmt.warning(
                        "Model rejected image content, retrying without image..."
                    )
                continue  # retry the LLM call with text-only
            elapsed = time.monotonic() - t0
            if report:
                report.record_llm_call(
                    turns + turn_offset,
                    elapsed,
                    token_est,
                    "error",
                    provider_retries=getattr(e, "_provider_retries", 0),
                )
            raise
        else:
            _vision_pending = False  # success — clear the flag
            elapsed = time.monotonic() - t0
            if verbose:
                fmt.llm_timing(elapsed, finish_reason)
            if report:
                report.record_llm_call(
                    turns + turn_offset,
                    elapsed,
                    token_est,
                    finish_reason,
                    provider_retries=_provider_retries,
                )
        # Handle empty assistant response (no content, no tool_calls).
        # Some providers return these occasionally; appending them as-is
        # would poison the history and cause BadRequestError on the next call.
        if not getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
            if verbose:
                fmt.warning("LLM returned empty response, requesting continuation...")
            # Give the message minimal content so it's valid in history
            msg.content = ""
            messages.append(
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else vars(msg)
            )
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

        messages.append(
            msg.model_dump(exclude_none=True)
            if hasattr(msg, "model_dump")
            else vars(msg)
        )

        # Emit events for streaming consumers: text_chunk for final answers only,
        # status_update for intermediate reasoning (before tool calls).
        if msg.content and not msg.tool_calls and finish_reason != "length":
            _emit(EVENT_TEXT_CHUNK, {"text": msg.content, "turn": turns})
        elif msg.content and msg.tool_calls:
            _emit(
                EVENT_STATUS_UPDATE,
                {
                    "turn": turns,
                    "type": "reasoning",
                    "text_length": len(msg.content),
                },
            )

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
            if cmd_activity:
                lines = [
                    f"  - {a['name']}: {'ok' if a['succeeded'] else 'error'}"
                    for a in cmd_activity
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            _COMMAND_TOOL_CONTEXT_PREFIX
                            + " external tool calls made during "
                            "the previous response:\n" + "\n".join(lines) + "\n]"
                        ),
                    }
                )
            if verbose:
                fmt.completion(turns, "ok")
                _show_state_summaries(thinking_state, todo_state, snapshot_state)
            return msg.content or "", False

        interventions: list[str] = []
        all_tools_readonly = True
        image_stash: list[dict] = []
        for tool_call in msg.tool_calls:
            # Check cancellation before each tool call
            if cancel_flag is not None and cancel_flag.is_set():
                if verbose:
                    fmt.info("Task cancelled by external request.")
                _emit(EVENT_STATUS_UPDATE, {"turn": turns, "cancelled": True})
                return None, True

            _tc_name = tool_call.function.name
            _emit(EVENT_TOOL_START, {"name": _tc_name, "turn": turns})

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
                a2a_manager=a2a_manager,
                messages=messages,
                image_stash=image_stash,
                scratch_dir=scratch_dir,
            )
            messages.append(tool_msg)

            bk_interventions = _post_tool_bookkeeping(
                tool_msg,
                tool_meta,
                turns,
                turn_offset,
                report,
                snapshot_state,
                consecutive_errors,
                verbose,
                _emit,
            )
            interventions.extend(bk_interventions)

            tool_name = tool_meta["name"]
            if tool_name == "think":
                think_used = True
            if tool_name == "todo":
                todo_last_used = turns

            if snapshot_state is not None:
                if tool_name not in READ_ONLY_TOOLS:
                    all_tools_readonly = False
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

        # Inject image data into conversation after all tool calls are processed
        if image_stash:
            provider = llm_kwargs.get("provider", "lmstudio")
            if provider == "command":
                vision_support = None
            else:
                model_str = _resolve_model_str(provider, model_id)
                vision_support = _model_supports_vision(model_str)

            if vision_support is False:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            _IMAGE_SYNTHETIC_PREFIX
                            + " The current model does not support "
                            "vision/image analysis. The image could not be displayed. "
                            "Please inform the user and suggest they use a vision-capable model."
                        ),
                    }
                )
            else:
                parts = []
                questions = [img["question"] for img in image_stash if img["question"]]
                text = (
                    _IMAGE_SYNTHETIC_PREFIX
                    + " "
                    + (
                        " ".join(questions)
                        if questions
                        else "Describe and analyze the attached image(s)."
                    )
                )
                parts.append({"type": "text", "text": text})
                for img in image_stash:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img["data_url"]},
                        }
                    )
                messages.append({"role": "user", "content": parts})
                _vision_pending = True
            image_stash.clear()

        if interventions:
            messages.append({"role": "user", "content": "\n\n".join(interventions)})
        if verbose:
            fmt.context_stats(
                f"Context after turn {turns}",
                estimate_tokens(messages, effective_tools),
            )

        # Proactive checkpoint (if enabled)
        if compaction_state is not None:
            compaction_state.maybe_checkpoint(
                messages,
                _call_llm_for_secondary,
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
        _show_state_summaries(thinking_state, todo_state, snapshot_state)

    # Save continue file (with LLM enhancement since we're not in a hurry)
    if continue_here:
        from .continue_here import write_continue_file

        write_continue_file(
            base_dir,
            messages,
            todo_state=todo_state,
            snapshot_state=snapshot_state,
            thinking_state=thinking_state,
            call_llm_fn=_call_llm_for_secondary,
            model_id=model_id,
            base_url=api_base,
            api_key=llm_kwargs.get("api_key"),
            top_p=top_p,
            seed=seed,
            provider=llm_kwargs.get("provider"),
        )

    return last_text, True


# ---------------------------------------------------------------------------
# REPL command helpers
# ---------------------------------------------------------------------------


def _repl_run_custom_command(
    line: str, base_dir: str, *, model_id: str = ""
) -> tuple[str, str] | None:
    """Look up and run a custom command from the user's commands directory.

    *line* starts with ``!``.  Returns ``(cmd_name, stdout_text)`` on success,
    or ``None`` if the command could not be run (errors printed to stderr).
    """
    from .config import global_config_dir

    raw = line[1:].lstrip()

    # Split command name from the raw argument string
    parts = raw.split(None, 1)  # split on whitespace, max 1 split
    if not parts:
        return None
    cmd_name = parts[0]
    arg_string = parts[1].strip() if len(parts) > 1 else ""

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", cmd_name):
        fmt.error(f"invalid command name: {cmd_name!r}")
        return None

    commands_dir = global_config_dir() / "commands"
    if not commands_dir.is_dir():
        fmt.error(f"no commands directory at {commands_dir}")
        return None

    cmd_path = commands_dir / cmd_name
    if not cmd_path.is_file():
        _ci = sys.platform == "win32"
        _key = cmd_name.lower() if _ci else cmd_name
        candidates = [
            f
            for f in commands_dir.iterdir()
            if (f.stem.lower() if _ci else f.stem) == _key
            and f.is_file()
            and os.access(f, os.X_OK)
        ]
        if len(candidates) == 1:
            cmd_path = candidates[0]
        elif len(candidates) > 1:
            names = ", ".join(f.name for f in sorted(candidates))
            fmt.error(f"ambiguous command {cmd_name}: {names}")
            return None
        else:
            fmt.error(f"command not found: {cmd_name}")
            return None
    if not os.access(cmd_path, os.X_OK):
        fmt.error(f"command not executable: {cmd_name}")
        return None

    env = None
    if model_id:
        env = {**os.environ, "SWIVAL_MODEL": model_id}

    try:
        proc = subprocess.run(
            [str(cmd_path), base_dir] + ([arg_string] if arg_string else []),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=base_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        fmt.error(f"command timed out after 30s: {cmd_name}")
        return None
    except OSError as exc:
        fmt.error(f"failed to start command {cmd_name}: {exc}")
        return None

    if proc.returncode != 0:
        error_text = (
            proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        )
        fmt.error(f"command failed: {error_text}")
        return None

    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    stdout = proc.stdout.strip()
    if not stdout:
        fmt.info("command produced no output, skipping.")
        return None

    return cmd_name, stdout


def _truncate_for_context(
    text: str,
    messages: list,
    tools: list,
    context_length: int | None,
) -> str | None:
    """Truncate *text* to fit in remaining context, or return None to skip."""
    from .tokens import count_tokens, truncate_to_tokens

    if context_length is None:
        encoded = text.encode()
        if len(encoded) > _CUSTOM_CMD_OUTPUT_CAP:
            text = encoded[:_CUSTOM_CMD_OUTPUT_CAP].decode(errors="ignore")
            fmt.warning("command output truncated to 100KB (unknown context length).")
        return text

    current_cost = estimate_tokens(messages, tools)
    budget = (
        context_length - current_cost - MIN_OUTPUT_TOKENS - 4
    )  # 4 = per-message overhead
    if budget <= 0:
        fmt.warning("not enough context headroom to inject command output.")
        return None

    tok_count = count_tokens(text)
    if tok_count > budget:
        text = truncate_to_tokens(text, budget)
        fmt.warning("command output truncated to fit context window.")

    return text


def _repl_help() -> None:
    """Print available REPL commands."""
    fmt.info(
        "Available commands:\n"
        "  /help              Show this help message\n"
        "  /clear, /new       Reset conversation to initial state\n"
        "  /compact [--drop]  Compress context (--drop removes middle turns)\n"
        "  /save [label]      Set a context checkpoint\n"
        "  /restore           Summarize & collapse since checkpoint\n"
        "  /unsave            Cancel active checkpoint\n"
        "  /add-dir <path>    Grant read+write access to a directory\n"
        "  /add-dir-ro <path> Grant read-only access to a directory\n"
        "  /extend [N]        Double max turns, or set to N\n"
        "  /continue          Reset turn counter and continue the agent loop\n"
        "  /continue-status   Show if a continue file exists from a prior session\n"
        "  /learn             Review session for mistakes and persist to memory\n"
        "  /tools             List all available tools\n"
        "  /init              Scan project for build/test/lint workflow and conventions, write AGENTS.md\n"
        "  /exit, /quit       Exit the REPL\n"
        "\n"
        "  !command [args]    Run <config_dir>/commands/command; output becomes your next prompt"
    )


def _repl_tools(tools: list, mcp_manager=None, a2a_manager=None) -> None:
    """Print all available tools grouped by source."""
    # Collect MCP/A2A tool info from managers for classification.
    mcp_info = mcp_manager.get_tool_info() if mcp_manager is not None else {}
    a2a_info = a2a_manager.get_tool_info() if a2a_manager is not None else {}
    external_names: set[str] = set()
    for entries in (*mcp_info.values(), *a2a_info.values()):
        external_names.update(name for name, _ in entries)

    # Built-in: everything not claimed by MCP/A2A.
    builtin: list[tuple[str, str]] = []
    for t in tools:
        name = t["function"]["name"]
        if name not in external_names:
            builtin.append((name, t["function"].get("description", "")))
    builtin.sort()

    def _format_entries(entries: list[tuple[str, str]], indent: str) -> list[str]:
        if not entries:
            return []
        col = max(len(name) for name, _ in entries) + 2
        lines: list[str] = []
        for name, desc in entries:
            padding = " " * (col - len(name))
            # Normalize embedded newlines to hanging-indent continuation.
            desc_lines = desc.split("\n")
            first = f"{indent}{name}{padding}{desc_lines[0]}"
            lines.append(first)
            if len(desc_lines) > 1:
                hang = " " * (len(indent) + col)
                for cont in desc_lines[1:]:
                    lines.append(f"{hang}{cont}")
        return lines

    parts: list[str] = []

    if builtin:
        parts.append("Built-in tools:")
        parts.extend(_format_entries(builtin, "  "))

    for source_info, kind, noun in [
        (mcp_info, "MCP tools", "server"),
        (a2a_info, "A2A tools", "agent"),
    ]:
        if not source_info:
            continue
        n = len(source_info)
        label = noun if n == 1 else f"{noun}s"
        if parts:
            parts.append("")
        parts.append(f"{kind} ({n} {label}):")
        for group in sorted(source_info):
            entries = sorted(source_info[group])
            parts.append(f"  {group}:")
            parts.extend(_format_entries(entries, "    "))

    if parts:
        fmt.info("\n".join(parts))
    else:
        fmt.info("No tools available.")


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

    fmt.reset_state()
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
    messages: list,
    tools: list,
    context_length: int | None,
    arg: str,
    snapshot_state: "SnapshotState | None" = None,
) -> None:
    """Manually compact conversation context."""
    before = estimate_tokens(messages, tools)

    messages[:] = compact_messages(messages)
    if arg.strip() == "--drop":
        messages[:] = drop_middle_turns(messages)

    if snapshot_state is not None:
        snapshot_state.invalidate_index_checkpoint()

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


def _repl_snapshot_save(
    label: str, messages: list, snapshot_state: "SnapshotState | None"
) -> None:
    if snapshot_state is None:
        fmt.warning("snapshot not available")
        return
    result = snapshot_state.save_at_index(label, len(messages))
    if result.startswith("error:"):
        fmt.warning(
            f"checkpoint already active (label={snapshot_state.explicit_label!r}). Cancel it first with /unsave."
        )
    else:
        fmt.info(f"checkpoint saved: {label}")


def _repl_snapshot_restore(
    messages: list,
    snapshot_state: "SnapshotState | None",
    *,
    model_id: str,
    api_base: str,
    api_key: str | None,
    top_p: float,
    seed: int | None,
    provider: str | None,
) -> None:
    if snapshot_state is None:
        fmt.warning("snapshot not available")
        return
    if len(messages) <= 1:
        fmt.warning("nothing to collapse")
        return

    def summarize_fn(text):
        return _call_summarize_llm(
            text,
            _SUMMARIZE_SYSTEM_PROMPT,
            call_llm,
            model_id,
            api_base,
            api_key,
            top_p,
            seed,
            provider,
        )

    result = snapshot_state.restore_with_autosummary(messages, summarize_fn)
    if result.startswith("error:"):
        fmt.warning(result)
    else:
        fmt.info(result)


def _repl_snapshot_unsave(snapshot_state: "SnapshotState | None") -> None:
    if snapshot_state is None:
        fmt.warning("snapshot not available")
        return
    result = snapshot_state.cancel()
    try:
        data = json.loads(result)
        if data.get("status") == "no_checkpoint":
            fmt.warning("no active checkpoint to cancel")
        else:
            fmt.info(f"checkpoint cancelled: {data.get('label', '?')}")
    except (json.JSONDecodeError, TypeError):
        fmt.info(result)


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
    a2a_manager=None,
    continue_here: bool = True,
    cache=None,
    secret_shield=None,
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

    fmt.reset_state()
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
        a2a_manager=a2a_manager,
        cache=cache,
        secret_shield=secret_shield,
    )

    while True:
        try:
            print(file=sys.stderr)  # blank line before prompt
            line = session.prompt(prompt_text)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)  # newline after ^D / ^C
            if continue_here and any(_msg_role(m) != "system" for m in messages):
                from .continue_here import write_continue_file

                write_continue_file(
                    base_dir,
                    messages,
                    todo_state=todo_state,
                    snapshot_state=snapshot_state,
                    thinking_state=thinking_state,
                )
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("!") and len(line) > 1 and not line[1:].startswith(" "):
            result = _repl_run_custom_command(line, base_dir, model_id=model_id)
            if result is None:
                continue
            cmd_name, prompt_content = result
            prompt_content = _truncate_for_context(
                prompt_content, messages, tools, context_length
            )
            if prompt_content is None:
                continue

            fmt.info(f"[!{cmd_name}] output:\n{prompt_content}")
            messages.append({"role": "user", "content": prompt_content})
            try:
                answer, exhausted = run_agent_loop(
                    messages,
                    tools,
                    max_turns=turn_state["max_turns"],
                    **_repl_loop_kwargs,
                )
            except KeyboardInterrupt:
                fmt.warning("interrupted, question aborted.")
                if continue_here:
                    from .continue_here import write_continue_file

                    write_continue_file(
                        base_dir,
                        messages,
                        todo_state=todo_state,
                        snapshot_state=snapshot_state,
                        thinking_state=thinking_state,
                    )
                continue

            history_label = f"[!{cmd_name}] {line}"
            if not no_history and answer:
                append_history(base_dir, history_label, answer, diagnostics=verbose)
            if answer is not None:
                print(answer)
            if exhausted and verbose:
                fmt.warning(
                    "max turns reached for this question. Use /continue to resume."
                )
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
        elif cmd == "/tools":
            _repl_tools(tools, mcp_manager, a2a_manager)
            continue
        elif cmd in ("/clear", "/new"):
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
            _repl_compact(messages, tools, context_length, cmd_arg, snapshot_state)
            continue
        elif cmd == "/save":
            label = cmd_arg.strip() or "user-checkpoint"
            _repl_snapshot_save(label, messages, snapshot_state)
            continue
        elif cmd == "/restore":
            _repl_snapshot_restore(
                messages,
                snapshot_state,
                model_id=model_id,
                api_base=api_base,
                api_key=llm_kwargs.get("api_key"),
                top_p=top_p,
                seed=seed,
                provider=llm_kwargs.get("provider"),
            )
            continue
        elif cmd == "/unsave":
            _repl_snapshot_unsave(snapshot_state)
            continue
        elif cmd == "/extend":
            _repl_extend(cmd_arg, turn_state)
            continue
        elif cmd == "/continue-status":
            from .continue_here import load_continue_file

            content = load_continue_file(base_dir, delete=False)
            if content:
                preview = content[:500]
                if len(content) > 500:
                    preview += "\n... (truncated)"
                fmt.info(f"Continue file exists ({len(content)} chars):\n{preview}")
            else:
                fmt.info("No continue file found.")
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
            # Three-pass init: explore, enrich, write — then validate
            _init_aborted = False
            for _pass, prompt in enumerate(
                (_init_prompt(), INIT_ENRICH_PROMPT, INIT_WRITE_PROMPT), 1
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
                    if continue_here:
                        from .continue_here import write_continue_file

                        write_continue_file(
                            base_dir,
                            messages,
                            todo_state=todo_state,
                            snapshot_state=snapshot_state,
                            thinking_state=thinking_state,
                        )
                    _init_aborted = True
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
            # Post-write validation and conditional retry
            if not _init_aborted:
                agents_path = Path(base_dir).resolve() / "AGENTS.md"
                reason, content = validate_agents_md(agents_path)
                if reason is not None:
                    retry_prompt = INIT_RETRY_PROMPT.format(reason=reason)
                    messages.append({"role": "user", "content": retry_prompt})
                    try:
                        answer, exhausted = run_agent_loop(
                            messages,
                            tools,
                            max_turns=turn_state["max_turns"],
                            **_repl_loop_kwargs,
                        )
                    except KeyboardInterrupt:
                        fmt.warning("interrupted, /init retry aborted.")
                        if continue_here:
                            from .continue_here import write_continue_file

                            write_continue_file(
                                base_dir,
                                messages,
                                todo_state=todo_state,
                                snapshot_state=snapshot_state,
                                thinking_state=thinking_state,
                            )
                        _init_aborted = True
                    else:
                        if not no_history and answer:
                            append_history(
                                base_dir,
                                "/init pass 4 (retry)",
                                answer,
                                diagnostics=verbose,
                            )
                        if answer is not None:
                            print(answer)
                        retry_reason, content = validate_agents_md(agents_path)
                        if retry_reason is not None:
                            fmt.warning(
                                f"AGENTS.md still invalid after retry: {retry_reason}"
                            )
                # Budget check using already-read content
                if not _init_aborted and content is not None:
                    if len(content) > _INIT_AGENTS_MD_BUDGET:
                        fmt.warning(
                            f"AGENTS.md is {len(content)} chars, "
                            f"exceeds {_INIT_AGENTS_MD_BUDGET} target."
                        )
            continue
        elif cmd == "/learn":
            messages.append({"role": "user", "content": LEARN_PROMPT})
            try:
                answer, exhausted = run_agent_loop(
                    messages,
                    tools,
                    max_turns=turn_state["max_turns"],
                    **_repl_loop_kwargs,
                )
            except KeyboardInterrupt:
                fmt.warning("interrupted, /learn aborted.")
                if continue_here:
                    from .continue_here import write_continue_file

                    write_continue_file(
                        base_dir,
                        messages,
                        todo_state=todo_state,
                        snapshot_state=snapshot_state,
                        thinking_state=thinking_state,
                    )
                continue
            if not no_history and answer:
                append_history(base_dir, "/learn", answer, diagnostics=verbose)
            if answer is not None:
                print(answer)
            if exhausted and verbose:
                fmt.warning("max turns reached for /learn. Use /continue to resume.")
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
                if continue_here:
                    from .continue_here import write_continue_file

                    write_continue_file(
                        base_dir,
                        messages,
                        todo_state=todo_state,
                        snapshot_state=snapshot_state,
                        thinking_state=thinking_state,
                    )
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
            if continue_here:
                from .continue_here import write_continue_file

                write_continue_file(
                    base_dir,
                    messages,
                    todo_state=todo_state,
                    snapshot_state=snapshot_state,
                    thinking_state=thinking_state,
                )
            continue

        if not no_history and answer:
            append_history(base_dir, line, answer, diagnostics=verbose)
        if answer is not None:
            print(answer)
        if exhausted and verbose:
            fmt.warning("max turns reached for this question. Use /continue to resume.")


if __name__ == "__main__":
    main()
