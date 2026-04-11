"""Continue-here files: save session state on interruption, resume on next start."""

import time
from pathlib import Path

from . import fmt
from ._msg import _msg_role, _msg_content, _msg_tool_calls, _is_synthetic

CONTINUE_PATH = ".swival/continue.md"
MAX_CONTINUE_CHARS = 4000
STALENESS_SECONDS = 24 * 60 * 60  # 24 hours


_CONTINUE_SUMMARY_PROMPT = (
    "You are summarizing an interrupted AI coding session so it can be resumed.\n"
    "The session was working on a task and was interrupted before completion.\n\n"
    "Extract from the conversation:\n"
    "1. What was completed (files read, edits made, tests run, findings)\n"
    "2. What remains to be done\n"
    "3. Key decisions that were made (so they aren't re-debated)\n"
    "4. Anything tricky or surprising encountered\n"
    "5. The exact next step to take when resuming\n\n"
    "Be specific: include file paths, line numbers, function names, error messages.\n"
    "Write in concise bullet points under markdown headings. No preamble."
)


def _safe_continue_path(base_dir: str) -> Path:
    """Build the continue file path and verify it resolves inside base_dir."""
    base = Path(base_dir).resolve()
    p = (Path(base_dir) / CONTINUE_PATH).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"continue path {p} escapes base directory {base}")
    return p


def _find_last_user_task(messages: list) -> str | None:
    """Find the most recent substantive user message (not a synthetic intervention)."""
    for msg in reversed(messages):
        if _msg_role(msg) != "user":
            continue
        content = _msg_content(msg)
        if content and not _is_synthetic(msg):
            return content
    return None


def _find_first_user_task(messages: list) -> str | None:
    """Find the first substantive user message."""
    for msg in messages:
        if _msg_role(msg) != "user":
            continue
        content = _msg_content(msg)
        if content and not _is_synthetic(msg):
            return content
    return None


def _extract_recent_tool_activity(messages: list, max_entries: int = 8) -> list[str]:
    """Extract the last N tool call names and brief results from messages."""
    entries = []
    for msg in reversed(messages):
        if len(entries) >= max_entries:
            break
        tc = _msg_tool_calls(msg)
        if tc:
            for call in tc:
                fn = (
                    call.function
                    if hasattr(call, "function")
                    else call.get("function", {})
                )
                name = fn.name if hasattr(fn, "name") else fn.get("name", "?")
                entries.append(f"called `{name}`")
        elif _msg_role(msg) == "tool":
            content = _msg_content(msg)
            if content:
                # Truncate long tool results
                preview = content[:120].replace("\n", " ")
                if len(content) > 120:
                    preview += "..."
                entries.append(f"  → {preview}")
    entries.reverse()
    return entries


def _build_deterministic_continue(
    messages: list,
    todo_state=None,
    snapshot_state=None,
    thinking_state=None,
) -> str:
    """Build continue-file content from structured state, no LLM needed."""
    sections: list[str] = ["# Continue Here\n"]

    # Current task
    last_task = _find_last_user_task(messages)
    first_task = _find_first_user_task(messages)

    if last_task:
        sections.append("## Current task")
        task_preview = last_task[:500]
        if len(last_task) > 500:
            task_preview += "..."
        sections.append(task_preview)

        if first_task and first_task != last_task:
            sections.append("\n## Original task")
            orig_preview = first_task[:300]
            if len(first_task) > 300:
                orig_preview += "..."
            sections.append(orig_preview)

    # Todo state
    if todo_state is not None:
        remaining = [i for i in todo_state.items if not i.done]
        done = [i for i in todo_state.items if i.done]
        if remaining or done:
            sections.append("\n## Task checklist")
            for item in done:
                sections.append(f"- [x] {item.text}")
            for item in remaining:
                sections.append(f"- [ ] {item.text}")

    # Snapshot history (investigation summaries)
    if snapshot_state is not None and snapshot_state.history:
        sections.append("\n## Prior investigation summaries")
        for entry in snapshot_state.history[-5:]:
            label = entry.get("label", "investigation")
            summary = entry.get("summary", "")
            if summary:
                summary_preview = summary[:400]
                if len(summary) > 400:
                    summary_preview += "..."
                sections.append(f"- **{label}**: {summary_preview}")

    # Thinking history (last few reasoning steps)
    if thinking_state is not None and thinking_state.history:
        recent = thinking_state.history[-5:]
        sections.append("\n## Key reasoning")
        for entry in recent:
            text = entry.thought[:200].replace("\n", " ")
            if len(entry.thought) > 200:
                text += "..."
            sections.append(f"- {text}")

    # Recent tool activity
    activity = _extract_recent_tool_activity(messages)
    if activity:
        sections.append("\n## Recent activity")
        for line in activity:
            sections.append(f"- {line}")

    content = "\n".join(sections)
    if len(content) > MAX_CONTINUE_CHARS:
        content = content[:MAX_CONTINUE_CHARS]
    return content


def write_continue_file(
    base_dir: str,
    messages: list,
    *,
    todo_state=None,
    snapshot_state=None,
    thinking_state=None,
    call_llm_fn=None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    provider: str | None = None,
) -> bool:
    """Write .swival/continue.md from current session state.

    Always writes a deterministic version first. If call_llm_fn is
    provided and succeeds, overwrites with the richer LLM version.
    Returns True if file was written.
    """
    try:
        path = _safe_continue_path(base_dir)
    except ValueError:
        return False

    # Build and write deterministic version first
    det_content = _build_deterministic_continue(
        messages, todo_state, snapshot_state, thinking_state
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(det_content, encoding="utf-8")
    except OSError:
        return False

    # Try LLM enhancement if available
    if call_llm_fn is not None:
        llm_content = _try_llm_summary(
            messages,
            call_llm_fn,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            top_p=top_p,
            seed=seed,
            provider=provider,
        )
        if llm_content:
            try:
                final = llm_content
                if len(final) > MAX_CONTINUE_CHARS:
                    final = final[:MAX_CONTINUE_CHARS]
                path.write_text(final, encoding="utf-8")
            except OSError:
                pass  # deterministic version already written

    return True


def _try_llm_summary(
    messages: list,
    call_llm_fn,
    *,
    model_id: str | None,
    base_url: str | None,
    api_key: str | None,
    top_p: float | None,
    seed: int | None,
    provider: str | None,
) -> str | None:
    """Call the LLM to generate a richer continue summary. Returns None on failure."""
    # Build a condensed view of the conversation
    lines = []
    for msg in messages:
        role = _msg_role(msg)
        content = _msg_content(msg)
        if role == "system":
            continue  # skip system prompt
        if content:
            lines.append(f"[{role}] {content[:2000]}")

    text = "\n".join(lines)
    if len(text) > 8000:
        text = text[:8000] + "\n[... truncated]"

    prompt = [
        {"role": "system", "content": _CONTINUE_SUMMARY_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        _result = call_llm_fn(
            base_url=base_url,
            model_id=model_id,
            messages=prompt,
            max_output_tokens=1024,
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


def load_continue_file(base_dir: str, *, delete: bool = True) -> str | None:
    """Read and optionally delete .swival/continue.md. Returns content or None."""
    try:
        path = _safe_continue_path(base_dir)
    except ValueError:
        return None

    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not content or not content.strip():
        if delete:
            try:
                path.unlink()
            except OSError:
                pass
        return None

    # Staleness warning
    try:
        mtime = path.stat().st_mtime
        age = time.time() - mtime
        if age > STALENESS_SECONDS:
            hours = int(age / 3600)
            fmt.warning(f"continue file is {hours}h old — loading anyway")
    except OSError:
        pass

    if delete:
        try:
            path.unlink()
        except OSError:
            pass

    # Cap size on read
    if len(content) > MAX_CONTINUE_CHARS:
        content = content[:MAX_CONTINUE_CHARS]

    return content


def format_continue_prompt(content: str) -> str:
    """Wrap continue-file content in <continue-here> tags for prompt injection."""
    return (
        "<continue-here>\n"
        "[This session is resuming interrupted work. The previous session "
        "was interrupted before completion. Review the state below and "
        "continue from where it left off.]\n\n"
        f"{content}\n"
        "</continue-here>"
    )
