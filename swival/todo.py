"""Todo list tool for tracking work items across an agent session."""

import json
from dataclasses import dataclass

from . import fmt

MAX_ITEMS = 50
MAX_ITEM_TEXT = 500
VALID_ACTIONS = {"add", "done", "remove", "clear", "list"}
_REASON_LIST_FULL = "todo list full"


@dataclass(slots=True)
class TodoItem:
    text: str
    done: bool = False


def _task_key(text: str) -> str:
    return text.casefold()


def _to_stripped_list(raw) -> list[str]:
    """Coerce a string, list, or other value into a list of stripped strings."""
    if isinstance(raw, str):
        stripped = raw.strip()
        # LLMs sometimes JSON-encode the array as a string — unwrap it.
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [t.strip() if isinstance(t, str) else str(t) for t in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        return [stripped]
    if isinstance(raw, list):
        return [t.strip() if isinstance(t, str) else str(t) for t in raw]
    return [str(raw).strip()]


def _normalize_tasks(args: dict) -> list[str] | str:
    """Extract and normalize the task list from args.

    Returns a list of stripped task strings, or an error string.
    """
    has_tasks = "tasks" in args
    has_task = "task" in args

    if has_tasks and has_task:
        if _to_stripped_list(args["tasks"]) != _to_stripped_list(args["task"]):
            return "error: provide either 'tasks' or legacy alias 'task', not conflicting values"

    if not has_tasks and not has_task:
        return "error: action requires a 'tasks' parameter"

    raw = args["tasks"] if has_tasks else args["task"]

    items = _to_stripped_list(raw)
    if not items:
        return "error: 'tasks' must not be empty"

    return items


class TodoState:
    def __init__(self, verbose: bool = False):
        self.items: list[TodoItem] = []
        self.verbose = verbose
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0

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
            if self.verbose:
                fmt.todo_list(self.items, action="clear", note=f"{count} items removed")
            return self._response("clear")

        # Normalize input for add/done/remove
        normalized = _normalize_tasks(args)
        if isinstance(normalized, str):
            return normalized  # error string

        # Validate per-item: empty/whitespace and length
        valid_tasks: list[str] = []
        errors: list[dict] = []
        for task in normalized:
            if not task:
                errors.append({"task": "", "reason": "empty or whitespace-only task"})
            elif len(task) > MAX_ITEM_TEXT:
                errors.append(
                    {
                        "task": task[:80],
                        "reason": f"exceeds {MAX_ITEM_TEXT} character limit",
                    }
                )
            else:
                valid_tasks.append(task)

        if not valid_tasks:
            if len(normalized) == 1:
                if not normalized[0]:
                    return f"error: '{action}' requires a non-empty 'tasks' parameter"
                return f"error: task text exceeds {MAX_ITEM_TEXT} character limit, please shorten it"
            return (
                f"error: all {len(normalized)} items failed — no valid tasks provided"
            )

        if action == "add":
            return self._batch_add(valid_tasks, errors)
        if action == "done":
            return self._batch_done(valid_tasks, errors)
        if action == "remove":
            return self._batch_remove(valid_tasks, errors)
        return f"error: unhandled action {action!r}"

    def _batch_add(self, tasks: list[str], errors: list[dict]) -> str:
        skipped: list[str] = []
        added = 0
        existing_keys = {_task_key(i.text) for i in self.items}
        for task in tasks:
            task_key = _task_key(task)
            if task_key in existing_keys:
                skipped.append(task)
                continue
            if len(self.items) >= MAX_ITEMS:
                errors.append({"task": task[:80], "reason": _REASON_LIST_FULL})
                continue
            self.items.append(TodoItem(text=task))
            existing_keys.add(task_key)
            self.add_count += 1
            added += 1

        succeeded = added + len(skipped)
        if succeeded == 0 and errors:
            if all(e["reason"] == _REASON_LIST_FULL for e in errors):
                return f"error: todo list full ({MAX_ITEMS} items max per session)"
            return self._all_failed_error(errors, tasks)

        if self.verbose:
            note = self._batch_note("added", added, len(skipped), len(errors))
            fmt.todo_list(self.items, action="add", note=note)
        return self._response("add", skipped=skipped or None, errors=errors or None)

    def _batch_done(self, tasks: list[str], errors: list[dict]) -> str:
        done_count = 0
        for task in tasks:
            match = self._match_item(task, include_done=True)
            if isinstance(match, str):
                errors.append({"task": task, "reason": match.removeprefix("error: ")})
                continue
            if not match.done:
                match.done = True
                self.done_count += 1
            done_count += 1

        if done_count == 0 and errors:
            return self._all_failed_error(errors, tasks)

        return self._finalize_batch("done", "marked done", done_count, errors)

    def _batch_remove(self, tasks: list[str], errors: list[dict]) -> str:
        removed = 0
        for task in tasks:
            match = self._match_item(task, include_done=True)
            if isinstance(match, str):
                errors.append({"task": task, "reason": match.removeprefix("error: ")})
                continue
            self.items.remove(match)
            removed += 1

        if removed == 0 and errors:
            return self._all_failed_error(errors, tasks)

        return self._finalize_batch("remove", "removed", removed, errors)

    def _finalize_batch(
        self,
        action: str,
        verb: str,
        count: int,
        errors: list[dict],
    ) -> str:
        if self.verbose:
            note = self._batch_note(verb, count, 0, len(errors))
            fmt.todo_list(self.items, action=action, note=note)
        return self._response(action, errors=errors or None)

    @staticmethod
    def _all_failed_error(errors: list[dict], tasks: list[str]) -> str:
        if len(tasks) == 1:
            return f"error: {errors[0]['reason']}"
        return f"error: all {len(errors)} items failed — {errors[0]['reason']}"

    @staticmethod
    def _batch_note(verb: str, count: int, skipped: int = 0, failed: int = 0) -> str:
        note = f"{verb} {count} item{'s' if count != 1 else ''}"
        extras = ", ".join(
            text
            for text in (
                f"{skipped} skipped" if skipped else "",
                f"{failed} failed" if failed else "",
            )
            if text
        )
        if extras:
            note += f" ({extras})"
        return note

    def _match_item(self, task: str, include_done: bool = False) -> TodoItem | str:
        """Find a matching item. Returns the item or an error string."""
        candidates = (
            self.items if include_done else [i for i in self.items if not i.done]
        )
        lower = _task_key(task)

        exact = [i for i in candidates if _task_key(i.text) == lower]
        if len(exact) == 1:
            return exact[0]

        prefix = [i for i in candidates if _task_key(i.text).startswith(lower)]
        if len(prefix) == 1:
            return prefix[0]

        sub = [i for i in candidates if lower in _task_key(i.text)]
        if len(sub) == 1:
            return sub[0]

        ambiguous = exact or prefix or sub
        if not ambiguous:
            return f"error: no task matching '{task}'"

        items_str = "; ".join(f"'{i.text}'" for i in ambiguous[:5])
        return f"error: '{task}' matches multiple items — be more specific: {items_str}"

    def _response(
        self,
        action: str,
        skipped: list[str] | None = None,
        errors: list[dict] | None = None,
    ) -> str:
        items = [{"task": i.text, "done": i.done} for i in self.items]
        remaining = self.remaining_count
        resp: dict = {
            "action": action,
            "total": len(self.items),
            "remaining": remaining,
            "items": items,
        }
        if skipped:
            resp["skipped"] = skipped
        if errors:
            resp["errors"] = errors
        return json.dumps(resp)

    def reset(self) -> None:
        """Reset all state. Used by REPL /clear."""
        self.items.clear()
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0

    def summary_line(self) -> str | None:
        """One-line usage summary, or None if todo was never called."""
        if self._total_actions == 0:
            return None
        return (
            f"todo: {self.add_count} added, {self.done_count} done, "
            f"{self.remaining_count} remaining"
        )
