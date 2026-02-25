"""Structured thinking tool for multi-step reasoning."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import fmt


@dataclass
class ThoughtEntry:
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: bool = False
    revises_thought: int | None = None
    branch_from_thought: int | None = None
    branch_id: str | None = None


MAX_THOUGHT_LENGTH = 10000
MAX_HISTORY = 200
MAX_BRANCHES = 20
MAX_BRANCH_ID_LENGTH = 50
MAX_NOTES = 50
MAX_NOTE_LENGTH = 5000


def _safe_notes_path(notes_dir: str) -> Path:
    """Build the notes file path and verify it resolves inside notes_dir.

    Prevents symlink-based escapes (e.g. .swival -> /etc).
    """
    base = Path(notes_dir).resolve()
    notes_path = (Path(notes_dir) / ".swival" / "notes.md").resolve()
    if not notes_path.is_relative_to(base):
        raise ValueError(f"notes path {notes_path} escapes base directory {base}")
    return notes_path


class ThinkingState:
    def __init__(self, verbose: bool = False, notes_dir: str | None = None):
        self.history: list[ThoughtEntry] = []
        self.branches: dict[str, list[ThoughtEntry]] = {}
        self.verbose = verbose
        self.notes_dir = notes_dir
        self.note_count = 0

        # Usage counters (unconditional, not gated on verbose)
        self.think_calls = 0
        self.note_attempts = 0
        self.note_saves = 0

        # Session isolation: delete any stale notes file from a prior run.
        if notes_dir is not None:
            notes_path = _safe_notes_path(notes_dir)
            try:
                notes_path.unlink()
            except FileNotFoundError:
                pass  # No stale file — fine.

    def process(self, args: dict) -> str:
        """Validate and record a thinking step. Returns a JSON summary or error string."""
        # History cap
        if len(self.history) >= MAX_HISTORY:
            return f"error: thinking history full ({MAX_HISTORY} steps max)"

        self.think_calls += 1

        thought = args.get("thought", "")

        # Auto-default optional numbering params
        if "thought_number" in args:
            thought_number = args["thought_number"]
        else:
            thought_number = len(self.history) + 1

        if "total_thoughts" in args:
            total_thoughts = args["total_thoughts"]
        elif self.history:
            total_thoughts = self.history[-1].total_thoughts
        else:
            total_thoughts = 3

        next_thought_needed = args.get("next_thought_needed", True)
        is_revision = args.get("is_revision", False)
        revises_thought = args.get("revises_thought")
        branch_from_thought = args.get("branch_from_thought")
        branch_id = args.get("branch_id")

        # Truncate thought text
        if len(thought) > MAX_THOUGHT_LENGTH:
            thought = thought[:MAX_THOUGHT_LENGTH]

        # Build set of recorded thought numbers for reference validation
        recorded_numbers = {e.thought_number for e in self.history}

        # Revision validation
        if is_revision and revises_thought is None:
            return "error: is_revision requires revises_thought"
        if revises_thought is not None and not is_revision:
            return "error: revises_thought requires is_revision=true"
        if revises_thought is not None:
            if revises_thought not in recorded_numbers:
                return f"error: revises_thought={revises_thought} not found in history"

        # Branch validation
        if branch_from_thought is not None and branch_id is None:
            return "error: branch_from_thought requires branch_id"
        if branch_id is not None and branch_from_thought is None:
            return "error: branch_id requires branch_from_thought"
        if branch_id is not None:
            branch_id = branch_id.strip()
            if not branch_id:
                return "error: branch_id must not be blank"
            if len(branch_id) > MAX_BRANCH_ID_LENGTH:
                return (
                    f"error: branch_id exceeds {MAX_BRANCH_ID_LENGTH} character limit"
                )
        if branch_from_thought is not None:
            if branch_from_thought not in recorded_numbers:
                return f"error: branch_from_thought={branch_from_thought} not found in history"
            if branch_id not in self.branches and len(self.branches) >= MAX_BRANCHES:
                return f"error: too many branches ({MAX_BRANCHES} max)"

        # Auto-adjust total_thoughts
        if thought_number > total_thoughts:
            total_thoughts = thought_number

        # Record
        entry = ThoughtEntry(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from_thought=branch_from_thought,
            branch_id=branch_id,
        )
        self.history.append(entry)

        if branch_id is not None:
            self.branches.setdefault(branch_id, []).append(entry)

        # Logging
        if self.verbose:
            self._log(entry)

        # Build response
        response = {
            "thought_number": thought_number,
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "branches": list(self.branches.keys()),
            "history_length": len(self.history),
        }

        # Handle persistent note
        note_text = args.get("note", "").strip()
        if note_text:
            self._handle_note(note_text, entry, response)

        return json.dumps(response)

    def _handle_note(self, note_text: str, entry: ThoughtEntry, response: dict) -> None:
        """Persist a note to disk and update the response dict."""
        self.note_attempts += 1

        if self.notes_dir is None:
            response["note_saved"] = False
            response["note_error"] = "no_notes_dir"
            return

        if self.note_count >= MAX_NOTES:
            response["note_saved"] = False
            response["note_error"] = "cap_exceeded"
            return

        if len(note_text) > MAX_NOTE_LENGTH:
            note_text = note_text[:MAX_NOTE_LENGTH]

        try:
            notes_path = _safe_notes_path(self.notes_dir)
        except ValueError:
            response["note_saved"] = False
            response["note_error"] = "write_failed"
            return

        self.note_count += 1

        # Build header
        header = f"## Note {self.note_count} (thought {entry.thought_number}"
        if entry.branch_id is not None:
            header += f", branch: {entry.branch_id}"
        header += ")"

        try:
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            with notes_path.open("a", encoding="utf-8") as f:
                f.write(f"{header}\n{note_text}\n\n")
        except (OSError, UnicodeEncodeError):
            self.note_count -= 1
            response["note_saved"] = False
            response["note_error"] = "write_failed"
            return

        self.note_saves += 1
        response["note_saved"] = True
        response["notes_file"] = ".swival/notes.md"

    def summary_line(self) -> str | None:
        """Return a one-line usage summary, or None if think was never called."""
        if self.think_calls == 0:
            return None
        parts = [f"think: {self.think_calls} call{'s' if self.think_calls != 1 else ''}"]
        if self.note_saves:
            parts.append(f"{self.note_saves} note{'s' if self.note_saves != 1 else ''} saved")
        return ", ".join(parts)

    def _log(self, entry: ThoughtEntry) -> None:
        """Write a formatted log line to stderr."""
        # Normalize: newlines -> spaces, collapse whitespace, truncate
        text = re.sub(r"\s+", " ", entry.thought).strip()
        if len(text) > 200:
            text = text[:200]

        fmt.think_step(
            entry.thought_number,
            entry.total_thoughts,
            text,
            is_revision=entry.is_revision,
            revises_thought=entry.revises_thought,
            branch_id=entry.branch_id,
            branch_from_thought=entry.branch_from_thought,
        )
