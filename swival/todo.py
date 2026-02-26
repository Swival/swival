"""Todo list tool for tracking work items across an agent session."""

import json
from dataclasses import dataclass
from pathlib import Path

from . import fmt

MAX_ITEMS = 50
MAX_ITEM_TEXT = 500
VALID_ACTIONS = {"add", "done", "remove", "clear", "list"}


@dataclass
class TodoItem:
    text: str
    done: bool = False


def _safe_todo_path(notes_dir: str) -> Path:
    """Build the todo file path and verify it resolves inside notes_dir."""
    base = Path(notes_dir).resolve()
    todo_path = (Path(notes_dir) / ".swival" / "todo.md").resolve()
    if not todo_path.is_relative_to(base):
        raise ValueError(f"todo path {todo_path} escapes base directory {base}")
    return todo_path


class TodoState:
    def __init__(self, notes_dir: str | None = None, verbose: bool = False):
        self.items: list[TodoItem] = []
        self.notes_dir = notes_dir
        self.verbose = verbose
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0

        # Session isolation: delete stale todo file from a prior run.
        if notes_dir is not None:
            try:
                todo_path = _safe_todo_path(notes_dir)
                todo_path.unlink()
            except FileNotFoundError:
                pass
            except ValueError:
                # .swival is a symlink escaping base_dir — disable persistence.
                self.notes_dir = None

    @property
    def remaining_count(self) -> int:
        return sum(1 for i in self.items if not i.done)

    def process(self, args: dict) -> str:
        """Handle a todo action. Returns JSON with the current list or an error string."""
        action = args.get("action", "")
        if action not in VALID_ACTIONS:
            return f"error: invalid action {action!r}, expected one of: {', '.join(sorted(VALID_ACTIONS))}"

        self._total_actions += 1

        if action == "list":
            return self._response("list")

        if action == "clear":
            count = len(self.items)
            self.items.clear()
            self._save()
            if self.verbose:
                fmt.todo_update("cleared", f"{count} items removed")
            return self._response("clear")

        task = args.get("task", "").strip()
        if not task:
            return f"error: '{action}' requires a non-empty 'task' parameter"

        if action == "add":
            return self._add(task)
        elif action == "done":
            return self._done(task)
        elif action == "remove":
            return self._remove(task)

        return f"error: unhandled action {action!r}"

    def _add(self, task: str) -> str:
        if len(task) > MAX_ITEM_TEXT:
            return f"error: task text exceeds {MAX_ITEM_TEXT} character limit, please shorten it"
        task_key = self._task_key(task)
        if any(self._task_key(i.text) == task_key for i in self.items):
            if self.verbose:
                remaining = self.remaining_count
                fmt.todo_update(
                    "add", f"Already listed: {task[:80]} ({remaining} remaining)"
                )
            return self._response(
                "add",
                note=f"'{task}' already in list — not added. Continue with your next action.",
            )
        if len(self.items) >= MAX_ITEMS:
            return f"error: todo list full ({MAX_ITEMS} items max)"
        self.items.append(TodoItem(text=task))
        self.add_count += 1
        self._save()
        remaining = self.remaining_count
        if self.verbose:
            fmt.todo_update("add", f"{task[:80]} ({remaining} remaining)")
        return self._response("add")

    def _done(self, task: str) -> str:
        match = self._match_item(task, include_done=True)
        if isinstance(match, str):
            return match  # error string
        # No-op if already done
        if not match.done:
            match.done = True
            self.done_count += 1
            self._save()
        remaining = self.remaining_count
        if self.verbose:
            fmt.todo_update("done", f"{match.text[:80]} ({remaining} remaining)")
        return self._response("done")

    def _remove(self, task: str) -> str:
        match = self._match_item(task, include_done=True)
        if isinstance(match, str):
            return match  # error string
        self.items.remove(match)
        self._save()
        remaining = self.remaining_count
        if self.verbose:
            fmt.todo_update(
                "remove", f"Removed: {match.text[:80]} ({remaining} remaining)"
            )
        return self._response("remove")

    def _match_item(self, task: str, include_done: bool = False) -> TodoItem | str:
        """Find a matching item. Returns the item or an error string."""
        candidates = (
            self.items if include_done else [i for i in self.items if not i.done]
        )
        lower = self._task_key(task)

        # 1. Exact match (case-insensitive)
        exact = [i for i in candidates if self._task_key(i.text) == lower]
        if len(exact) == 1:
            return exact[0]

        # 2. Prefix match
        prefix = [i for i in candidates if self._task_key(i.text).startswith(lower)]
        if len(prefix) == 1:
            return prefix[0]

        # 3. Substring match
        sub = [i for i in candidates if lower in self._task_key(i.text)]
        if len(sub) == 1:
            return sub[0]

        if not exact and not prefix and not sub:
            return f"error: no task matching '{task}'"

        # Ambiguous — report the conflicting items
        ambiguous = exact or prefix or sub
        items_str = "; ".join(f"'{i.text}'" for i in ambiguous[:5])
        return f"error: '{task}' matches multiple items — be more specific: {items_str}"

    @staticmethod
    def _task_key(task: str) -> str:
        return task.casefold()

    def _response(self, action: str, note: str | None = None) -> str:
        items = [{"task": i.text, "done": i.done} for i in self.items]
        remaining = self.remaining_count
        resp: dict = {
            "action": action,
            "total": len(self.items),
            "remaining": remaining,
            "items": items,
        }
        if note:
            resp["note"] = note
        return json.dumps(resp)

    def _save(self) -> None:
        if self.notes_dir is None:
            return
        try:
            todo_path = _safe_todo_path(self.notes_dir)
        except ValueError:
            return
        try:
            todo_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for item in self.items:
                marker = "x" if item.done else " "
                lines.append(f"- [{marker}] {item.text}")
            todo_path.write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
        except OSError:
            pass

    def reset(self) -> None:
        """Reset all state. Used by REPL /clear."""
        self.items.clear()
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0
        if self.notes_dir is not None:
            try:
                todo_path = _safe_todo_path(self.notes_dir)
                todo_path.unlink(missing_ok=True)
            except ValueError:
                pass

    def summary_line(self) -> str | None:
        """One-line usage summary, or None if todo was never called."""
        if self._total_actions == 0:
            return None
        remaining = self.remaining_count
        return f"todo: {self.add_count} added, {self.done_count} done, {remaining} remaining"
