"""Structured thinking tool for multi-step reasoning."""

import json
import re
from dataclasses import dataclass

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


class ThinkingState:
    def __init__(self, verbose: bool = False):
        self.history: list[ThoughtEntry] = []
        self.branches: dict[str, list[ThoughtEntry]] = {}
        self.verbose = verbose

        # Usage counters (unconditional, not gated on verbose)
        self.think_calls = 0

        # One-shot sanitizer: track last error to auto-retry on repeats
        self._last_error: str | None = None

    def _sanitize(self, args: dict) -> dict:
        """Infer mode from fields, strip incompatible fields, coerce mismatches.

        Handles the common "full template" case where models send every field
        with default values (is_revision=false, revises_thought=1, etc.).
        """
        args = dict(args)  # shallow copy

        # Determine mode — explicit mode takes priority
        mode = args.pop("mode", None)
        is_rev = args.get("is_revision")
        has_revises = "revises_thought" in args
        has_branch = "branch_from_thought" in args or "branch_id" in args

        if mode is None:
            if is_rev is True:
                mode = "revision"
            elif has_branch and is_rev is not True:
                mode = "branch"
            elif has_revises and is_rev is not False:
                # revises_thought present, is_revision not explicitly false → coerce
                mode = "revision"
            else:
                mode = "new"

        # Downgrade impossible modes when there's no history to reference
        if mode == "revision" and not self.history:
            mode = "new"
        if mode == "branch" and not self.history:
            mode = "new"

        # Apply mode constraints — strip incompatible fields silently
        if mode == "new":
            args.pop("revises_thought", None)
            args.pop("branch_from_thought", None)
            args.pop("branch_id", None)
            args["is_revision"] = False
        elif mode == "revision":
            args.pop("branch_from_thought", None)
            args.pop("branch_id", None)
            args["is_revision"] = True
        elif mode == "branch":
            args.pop("revises_thought", None)
            args["is_revision"] = False

        # Remove legacy field from args (internal only)
        args.pop("is_revision", None)
        # Re-add as clean boolean for _validate_and_record
        args["is_revision"] = mode == "revision"

        return args

    def process(self, args: dict) -> str:
        """Validate and record a thinking step. Returns a JSON summary or error string."""
        # History cap
        if len(self.history) >= MAX_HISTORY:
            return f"error: thinking history full ({MAX_HISTORY} steps max)"

        self.think_calls += 1
        args = self._sanitize(args)

        result = self._validate_and_record(args)

        if result.startswith("error:"):
            # One-shot sanitizer: if the same error repeats, strip everything
            # and retry as a minimal thought to break the loop
            if self._last_error == result:
                self._last_error = None
                return self._validate_and_record({
                    "thought": args.get("thought", ""),
                    "is_revision": False,
                })
            self._last_error = result
            return self._add_correction(result)

        self._last_error = None
        return result

    def _validate_and_record(self, args: dict) -> str:
        """Core validation and recording. Returns JSON on success, error string on failure."""
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
            return "error: revision mode requires revises_thought"
        if revises_thought is not None:
            if revises_thought not in recorded_numbers:
                return f"error: revises_thought={revises_thought} not found in history"

        # Branch validation
        if branch_from_thought is not None and branch_id is None:
            return "error: branch mode requires branch_id"
        if branch_id is not None and branch_from_thought is None:
            return "error: branch mode requires branch_from_thought"
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

        return json.dumps(response)

    def _add_correction(self, error: str) -> str:
        """Append a corrective suggestion to an error message."""
        valid = sorted({e.thought_number for e in self.history})

        if "revises_thought" in error and "not found" in error:
            if valid:
                return (
                    f"{error}; valid thought numbers: {valid}. "
                    f"Omit revises_thought for a normal thought, "
                    f"or use revises_thought={valid[-1]}"
                )
            return f"{error}; no thoughts recorded yet. Omit revises_thought for a normal thought"

        if "revision mode requires revises_thought" in error:
            if valid:
                return (
                    f"{error}; valid thought numbers: {valid}. "
                    f'Use mode="revision" with revises_thought={valid[-1]}, '
                    f'or omit mode for a normal thought'
                )
            return f"{error}; no thoughts recorded yet. Omit mode for a normal thought"

        if "branch_from_thought" in error and "not found" in error:
            if valid:
                return (
                    f"{error}; valid thought numbers: {valid}. "
                    f"Use branch_from_thought={valid[-1]}"
                )
            return f"{error}; no thoughts recorded yet"

        if "branch mode requires" in error:
            return f"{error}. Both branch_from_thought and branch_id must be set together"

        return error

    def summary_line(self) -> str | None:
        """Return a one-line usage summary, or None if think was never called."""
        if self.think_calls == 0:
            return None
        return f"think: {self.think_calls} call{'s' if self.think_calls != 1 else ''}"

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
