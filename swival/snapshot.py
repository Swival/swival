"""Snapshot tool for proactive context collapse."""

import json

from . import fmt

MAX_LABEL_LENGTH = 100
MAX_SUMMARY_LENGTH = 4000
MAX_HISTORY = 10
MAX_SUMMARY_DISPLAY = 1200

VALID_ACTIONS = {"save", "restore", "cancel", "status"}

SNAPSHOT_HISTORY_SENTINEL = "<!-- swival:snapshot-history-39a7c -->"

READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "grep",
        "glob",
        "fetch_url",
        "think",
        "todo",
        "snapshot",
    }
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without importing tiktoken."""
    return len(text) // 4


class SnapshotState:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Explicit scope (set by save, cleared by restore/cancel)
        self.explicit_active: bool = False
        self.explicit_label: str | None = None
        self.explicit_begin_tool_call_id: str | None = None

        # Implicit scope resolved at restore-time via backward scan
        self.last_restore_tool_call_id: str | None = None

        # Dirty tracking (resets at every scope boundary)
        self.dirty: bool = False
        self.dirty_tools: set[str] = set()

        # Completed history (survives compaction via prompt injection)
        self.history: list[dict] = []

        # Metrics
        self.stats: dict = {
            "saves": 0,
            "restores": 0,
            "cancels": 0,
            "blocked": 0,
            "force_restores": 0,
            "tokens_saved": 0,
        }

    def process(
        self,
        args: dict,
        *,
        messages: list | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        action = args.get("action", "")
        if action not in VALID_ACTIONS:
            return f"error: invalid action {action!r}, expected one of: {', '.join(sorted(VALID_ACTIONS))}"

        if action == "save":
            label = args.get("label", "")
            return self._save(label, tool_call_id)
        elif action == "restore":
            summary = args.get("summary", "")
            force = args.get("force", False)
            if messages is None:
                return "error: restore requires access to the message list"
            return self._restore(summary, messages, force, tool_call_id)
        elif action == "cancel":
            return self._cancel()
        elif action == "status":
            return self._status(messages)

        return f"error: unhandled action {action!r}"

    def _save(self, label: str, tool_call_id: str | None) -> str:
        if not label:
            return "error: save requires a non-empty 'label' parameter"
        if len(label) > MAX_LABEL_LENGTH:
            return f"error: label exceeds {MAX_LABEL_LENGTH} character limit"
        if self.explicit_active:
            return f"error: explicit checkpoint already active (label={self.explicit_label!r}). Call cancel first."

        self.explicit_active = True
        self.explicit_label = label
        self.explicit_begin_tool_call_id = tool_call_id
        self.reset_dirty()
        self.stats["saves"] += 1

        if self.verbose:
            fmt.info(f"snapshot: checkpoint saved — {label}")

        return json.dumps(
            {"action": "save", "label": label, "status": "checkpoint_set"}
        )

    def _restore(
        self, summary: str, messages: list, force: bool, tool_call_id: str | None
    ) -> str:
        if not summary:
            return "error: restore requires a non-empty 'summary' parameter"
        if len(summary) > MAX_SUMMARY_LENGTH:
            return f"error: summary exceeds {MAX_SUMMARY_LENGTH} character limit"

        if self.dirty and not force:
            tools_list = ", ".join(sorted(self.dirty_tools))
            self.stats["blocked"] += 1
            return (
                f"error: scope is dirty ({tools_list}). "
                "Call `snapshot restore force=true` to override, "
                "or `snapshot cancel` to keep context."
            )

        start_idx = self._resolve_start(messages)
        if isinstance(start_idx, str):
            return start_idx

        # Find the end of the collapsible scope.  The current turn's
        # assistant message (which issued this restore call) and any
        # tool-result messages already appended for earlier tool calls
        # in the same batch must be excluded — collapsing them would
        # orphan tool_call_ids.  Scan backwards for the last assistant
        # message with tool_calls; that marks the current turn boundary.
        end_idx = len(messages)
        for i in range(len(messages) - 1, max(start_idx - 1, -1), -1):
            msg_i = messages[i]
            role = _msg_role(msg_i)
            tc = (
                msg_i.get("tool_calls")
                if isinstance(msg_i, dict)
                else getattr(msg_i, "tool_calls", None)
            )
            if role == "assistant" and tc:
                end_idx = i
                break

        if end_idx - start_idx <= 0:
            return json.dumps(
                {
                    "action": "restore",
                    "status": "warning",
                    "message": "empty scope — nothing to collapse",
                }
            )

        # Calculate stats before collapsing
        scope_messages = messages[start_idx:end_idx]
        turns_collapsed = len(scope_messages)
        tokens_before = sum(_estimate_tokens(_msg_content(m)) for m in scope_messages)
        tokens_after = _estimate_tokens(summary)
        tokens_saved = max(0, tokens_before - tokens_after)

        scope_type = "explicit" if self.explicit_active else "implicit"
        label = self.explicit_label or "investigation"

        # Build synthetic recap message (no tool_calls field — avoids orphaned IDs)
        recap = {
            "role": "assistant",
            "content": (
                f"[snapshot: {label}]\n"
                f"{summary}\n"
                f"(collapsed {turns_collapsed} turns, saved ~{tokens_saved} tokens)"
            ),
        }
        if tool_call_id:
            recap["_snapshot_restore_id"] = tool_call_id

        # Replace the scope with the recap
        messages[start_idx:end_idx] = [recap]

        # Record history
        entry = {
            "label": label,
            "summary": summary[:MAX_SUMMARY_DISPLAY],
            "scope_type": scope_type,
            "turns_collapsed": turns_collapsed,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_saved,
            "dirty_at_restore": self.dirty,
            "forced_restore": force and self.dirty,
        }
        self.history.append(entry)
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # Update state
        self.stats["restores"] += 1
        self.stats["tokens_saved"] += tokens_saved
        if force and self.dirty:
            self.stats["force_restores"] += 1

        # Clear explicit checkpoint and reset dirty
        self.explicit_active = False
        self.explicit_label = None
        self.explicit_begin_tool_call_id = None
        self.last_restore_tool_call_id = tool_call_id
        self.reset_dirty()

        if self.verbose:
            fmt.info(
                f"snapshot: restored — collapsed {turns_collapsed} turns, "
                f"saved ~{tokens_saved} tokens"
            )

        return json.dumps(
            {
                "action": "restore",
                "status": "collapsed",
                "turns_collapsed": turns_collapsed,
                "tokens_saved": tokens_saved,
            }
        )

    def _resolve_start(self, messages: list) -> int | str:
        """Find the start index of the scope to collapse.

        Returns the message index or an error string.
        """
        # Explicit checkpoint: find the tool_call_id marker
        if self.explicit_active and self.explicit_begin_tool_call_id:
            for i, msg in enumerate(messages):
                tc_id = _msg_tool_call_id(msg)
                if tc_id == self.explicit_begin_tool_call_id:
                    # Start after the save response message
                    return i + 1
            return (
                "error: explicit checkpoint marker was removed (likely by compaction). "
                "Call `snapshot cancel` and try again."
            )

        # Implicit checkpoint: scan backwards for the most recent user message
        # or last restore boundary, whichever is newer
        last_user_idx = None
        last_restore_idx = None

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if last_user_idx is None and _msg_role(msg) == "user":
                last_user_idx = i
            if last_restore_idx is None and self.last_restore_tool_call_id:
                # Check both tool_call_id (for tool responses) and
                # _snapshot_restore_id (for recap messages)
                tc_id = _msg_tool_call_id(msg)
                restore_marker = (
                    msg.get("_snapshot_restore_id")
                    if isinstance(msg, dict)
                    else getattr(msg, "_snapshot_restore_id", None)
                )
                if (
                    tc_id == self.last_restore_tool_call_id
                    or restore_marker == self.last_restore_tool_call_id
                ):
                    last_restore_idx = i

            # Stop early if we found both
            if last_user_idx is not None and (
                last_restore_idx is not None or self.last_restore_tool_call_id is None
            ):
                break

        candidates = []
        if last_user_idx is not None:
            candidates.append(last_user_idx)
        if last_restore_idx is not None:
            candidates.append(last_restore_idx)

        if not candidates:
            return "error: no implicit checkpoint found (no user message in history)"

        # Use the most recent boundary, start after it
        boundary = max(candidates)
        return boundary + 1

    def _cancel(self) -> str:
        if not self.explicit_active:
            return json.dumps(
                {
                    "action": "cancel",
                    "status": "no_checkpoint",
                    "message": "no explicit checkpoint to cancel",
                }
            )

        label = self.explicit_label
        self.explicit_active = False
        self.explicit_label = None
        self.explicit_begin_tool_call_id = None
        self.stats["cancels"] += 1

        if self.verbose:
            fmt.info(f"snapshot: cancelled checkpoint — {label}")

        return json.dumps({"action": "cancel", "status": "cleared", "label": label})

    def _status(self, messages: list | None) -> str:
        info: dict = {
            "action": "status",
            "explicit_active": self.explicit_active,
            "explicit_label": self.explicit_label,
            "dirty": self.dirty,
            "dirty_tools": sorted(self.dirty_tools),
            "history_count": len(self.history),
            "stats": dict(self.stats),
        }
        return json.dumps(info)

    def mark_dirty(self, tool_name: str) -> None:
        if tool_name not in READ_ONLY_TOOLS:
            self.dirty = True
            self.dirty_tools.add(tool_name)

    def reset_dirty(self) -> None:
        self.dirty = False
        self.dirty_tools.clear()

    def inject_into_prompt(self) -> str | None:
        """Render history for system prompt injection."""
        if not self.history:
            return None

        lines = [SNAPSHOT_HISTORY_SENTINEL + "\n[Snapshot history — prior investigation summaries]"]
        total_chars = 0
        budget = 6000  # ~1500 tokens
        for entry in self.history:
            summary = entry["summary"]
            if len(summary) > MAX_SUMMARY_DISPLAY:
                summary = summary[:MAX_SUMMARY_DISPLAY]
            label = entry.get("label", "investigation")
            line = f"\n- [{label}] {summary}"
            if total_chars + len(line) > budget:
                break
            lines.append(line)
            total_chars += len(line)

        return "".join(lines) if len(lines) > 1 else None

    def reset(self) -> None:
        """Full reset for /clear."""
        self.explicit_active = False
        self.explicit_label = None
        self.explicit_begin_tool_call_id = None
        self.last_restore_tool_call_id = None
        self.dirty = False
        self.dirty_tools.clear()
        self.history.clear()
        self.stats = {
            "saves": 0,
            "restores": 0,
            "cancels": 0,
            "blocked": 0,
            "force_restores": 0,
            "tokens_saved": 0,
        }

    def summary_line(self) -> str | None:
        total = self.stats["restores"] + self.stats["saves"]
        if total == 0:
            return None
        saved = self.stats["tokens_saved"]
        return f"snapshot: {self.stats['restores']} restore(s), ~{saved} tokens saved"


def _msg_content(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def _msg_role(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def _msg_tool_call_id(msg) -> str | None:
    if isinstance(msg, dict):
        return msg.get("tool_call_id")
    return getattr(msg, "tool_call_id", None)
