"""Todo list tool for tracking work items across an agent session."""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]  # Windows

from . import fmt

# Per-session add cap.  The shared todo.md file may temporarily exceed
# this when concurrent sessions each add items up to the cap, because the
# merge preserves all successfully-added items rather than silently
# discarding them.  The cap is only enforced at add-time against the
# local session's item count.
MAX_ITEMS = 50
MAX_ITEM_TEXT = 500
VALID_ACTIONS = {"add", "done", "remove", "clear", "list"}
_REASON_LIST_FULL = "todo list full"


@dataclass
class TodoItem:
    text: str
    done: bool = False


def _task_key(text: str) -> str:
    return text.casefold()


def _parse_todo_file(path: Path) -> list[TodoItem]:
    """Parse a todo.md file into a list of TodoItems."""
    items: list[TodoItem] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return items
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x] "):
            items.append(TodoItem(text=stripped[6:], done=True))
        elif stripped.startswith("- [ ] "):
            items.append(TodoItem(text=stripped[6:], done=False))
    return items


@contextmanager
def _todo_lock(todo_path: Path):
    """Acquire an exclusive flock on the todo.md sidecar lock file."""
    if fcntl is None:
        yield
        return
    lock_path = todo_path.parent / "todo.md.lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


def _safe_todo_path(notes_dir: str, todo_dir: str | None = None) -> Path:
    """Build the todo file path and verify it resolves inside notes_dir.

    When *todo_dir* is set (A2A serve mode), use ``todo_dir/todo.md``
    directly instead of deriving ``notes_dir/.swival/todo.md``.
    """
    base = Path(notes_dir).resolve()
    if todo_dir is not None:
        todo_path = (Path(todo_dir) / "todo.md").resolve()
    else:
        todo_path = (Path(notes_dir) / ".swival" / "todo.md").resolve()
    if not todo_path.is_relative_to(base):
        raise ValueError(f"todo path {todo_path} escapes base directory {base}")
    return todo_path


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

    if has_tasks:
        raw = args["tasks"]
    elif has_task:
        raw = args["task"]
    else:
        return "error: action requires a 'tasks' parameter"

    items = _to_stripped_list(raw)
    if not items:
        return "error: 'tasks' must not be empty"

    return items


class TodoState:
    def __init__(
        self,
        notes_dir: str | None = None,
        verbose: bool = False,
        todo_dir: str | None = None,
    ):
        self.items: list[TodoItem] = []
        self.notes_dir = notes_dir
        self.todo_dir = todo_dir
        self.verbose = verbose
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0
        self._removed_keys: set[str] = set()
        self._baseline_keys: set[str] = set()

        # Session isolation: record which items already exist on disk so
        # this session's merges preserve them for concurrent sessions but
        # don't absorb them into self.items.  _baseline_keys is kept
        # separate from _removed_keys so baseline items are still written
        # back to disk (a live peer session may own them).
        if notes_dir is not None:
            try:
                todo_path = _safe_todo_path(notes_dir, todo_dir)
                todo_path.parent.mkdir(parents=True, exist_ok=True)
                with _todo_lock(todo_path):
                    for item in _parse_todo_file(todo_path):
                        self._baseline_keys.add(_task_key(item.text))
            except ValueError:
                # .swival is a symlink escaping base_dir — disable persistence.
                self.notes_dir = None
            except OSError:
                pass

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
            self._removed_keys.update(_task_key(i.text) for i in self.items)
            self._removed_keys.update(self._baseline_keys)
            self._baseline_keys.clear()
            self.items.clear()
            self._save()
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
                # Single-item failure — match legacy error format
                if not normalized[0]:
                    return f"error: '{action}' requires a non-empty 'tasks' parameter"
                return f"error: task text exceeds {MAX_ITEM_TEXT} character limit, please shorten it"
            return (
                f"error: all {len(normalized)} items failed — no valid tasks provided"
            )

        # Dispatch to batch handler
        if action == "add":
            return self._batch_add(valid_tasks, errors)
        elif action == "done":
            return self._batch_done(valid_tasks, errors)
        elif action == "remove":
            return self._batch_remove(valid_tasks, errors)

        return f"error: unhandled action {action!r}"

    def _batch_add(self, tasks: list[str], errors: list[dict]) -> str:
        skipped: list[str] = []
        added = 0
        for task in tasks:
            task_key = _task_key(task)
            if any(_task_key(i.text) == task_key for i in self.items):
                skipped.append(task)
                continue
            if len(self.items) >= MAX_ITEMS:
                errors.append({"task": task[:80], "reason": _REASON_LIST_FULL})
                continue
            self.items.append(TodoItem(text=task))
            self.add_count += 1
            added += 1

        succeeded = added + len(skipped)
        if succeeded == 0 and errors:
            if all(e["reason"] == _REASON_LIST_FULL for e in errors):
                return f"error: todo list full ({MAX_ITEMS} items max per session)"
            return self._all_failed_error(errors, tasks)

        self._save()
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
            self._removed_keys.add(_task_key(match.text))
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
        self._save()
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
        parts = [f"{verb} {count} item{'s' if count != 1 else ''}"]
        extras = []
        if skipped:
            extras.append(f"{skipped} skipped")
        if failed:
            extras.append(f"{failed} failed")
        if extras:
            parts.append(f"({', '.join(extras)})")
        return " ".join(parts)

    def _match_item(self, task: str, include_done: bool = False) -> TodoItem | str:
        """Find a matching item. Returns the item or an error string."""
        candidates = (
            self.items if include_done else [i for i in self.items if not i.done]
        )
        lower = _task_key(task)

        # 1. Exact match (case-insensitive)
        exact = [i for i in candidates if _task_key(i.text) == lower]
        if len(exact) == 1:
            return exact[0]

        # 2. Prefix match
        prefix = [i for i in candidates if _task_key(i.text).startswith(lower)]
        if len(prefix) == 1:
            return prefix[0]

        # 3. Substring match
        sub = [i for i in candidates if lower in _task_key(i.text)]
        if len(sub) == 1:
            return sub[0]

        if not exact and not prefix and not sub:
            return f"error: no task matching '{task}'"

        # Ambiguous — report the conflicting items
        ambiguous = exact or prefix or sub
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

    def _save(self) -> None:
        if self.notes_dir is None:
            return
        try:
            todo_path = _safe_todo_path(self.notes_dir, self.todo_dir)
        except ValueError:
            return
        try:
            todo_path.parent.mkdir(parents=True, exist_ok=True)
            with _todo_lock(todo_path):
                on_disk = {_task_key(i.text): i for i in _parse_todo_file(todo_path)}

                to_write: dict[str, TodoItem] = {}
                own_keys: set[str] = set()
                for item in self.items:
                    key = _task_key(item.text)
                    own_keys.add(key)
                    disk_item = on_disk.get(key)
                    if disk_item is not None and disk_item.done and not item.done:
                        item.done = True
                    to_write[key] = item

                for key, item in on_disk.items():
                    if key not in to_write and key not in self._removed_keys:
                        to_write[key] = item
                        # Absorb genuinely new concurrent items into self.items.
                        if key not in own_keys and key not in self._baseline_keys:
                            self.items.append(item)

                lines = []
                for item in to_write.values():
                    marker = "x" if item.done else " "
                    lines.append(f"- [{marker}] {item.text}")
                content = ("\n".join(lines) + "\n") if lines else ""
                todo_path.write_text(content, encoding="utf-8")
        except OSError:
            pass

    def reset(self) -> None:
        """Reset all state. Used by REPL /clear.

        Moves all known keys (in-memory + on-disk) to _baseline_keys so
        they are preserved on disk for concurrent sessions but not absorbed
        back into this session.  _removed_keys is cleared so that future
        saves from this session don't keep deleting items that a concurrent
        session may have restored.
        """
        baseline: set[str] = set()
        baseline.update(_task_key(i.text) for i in self.items)
        self.items.clear()
        self.add_count = 0
        self.done_count = 0
        self._total_actions = 0
        self._removed_keys.clear()
        if self.notes_dir is not None:
            try:
                todo_path = _safe_todo_path(self.notes_dir, self.todo_dir)
                todo_path.parent.mkdir(parents=True, exist_ok=True)
                with _todo_lock(todo_path):
                    for item in _parse_todo_file(todo_path):
                        baseline.add(_task_key(item.text))
                    todo_path.write_text("", encoding="utf-8")
            except (ValueError, OSError):
                pass
        self._baseline_keys = baseline

    def summary_line(self) -> str | None:
        """One-line usage summary, or None if todo was never called."""
        if self._total_actions == 0:
            return None
        remaining = self.remaining_count
        return f"todo: {self.add_count} added, {self.done_count} done, {remaining} remaining"
