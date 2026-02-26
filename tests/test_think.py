"""Tests for the think tool: ThinkingState, dispatch integration, and agent logging."""

import json
import tempfile

from swival.thinking import ThinkingState
from swival.tools import dispatch


# ---------------------------------------------------------------------------
# ThinkingState.process() unit tests
# ---------------------------------------------------------------------------


class TestLinearFlow:
    def test_sequential_thoughts(self):
        state = ThinkingState()
        for i in range(1, 4):
            result = json.loads(
                state.process(
                    {
                        "thought": f"Step {i}",
                        "thought_number": i,
                        "total_thoughts": 3,
                        "next_thought_needed": i < 3,
                    }
                )
            )
            assert result["thought_number"] == i
            assert result["history_length"] == i
            assert result["total_thoughts"] == 3
        assert result["next_thought_needed"] is False


class TestRevision:
    def test_valid_revision_with_mode(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First thought",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Correcting step 1",
                    "thought_number": 2,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "mode": "revision",
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2

    def test_valid_revision_with_legacy_is_revision(self):
        """is_revision=True still works (backward compat in sanitizer)."""
        state = ThinkingState()
        state.process(
            {
                "thought": "First thought",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Correcting step 1",
                    "thought_number": 2,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "is_revision": True,
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2

    def test_revision_mode_without_revises_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad revision",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "revision",
            }
        )
        assert "revision mode requires revises_thought" in result

    def test_legacy_is_revision_without_revises_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad revision",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "is_revision": True,
            }
        )
        assert "revision mode requires revises_thought" in result

    def test_revises_thought_without_mode_coerced(self):
        """revises_thought without is_revision/mode → coerced to revision."""
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Coerced revision",
                    "thought_number": 2,
                    "total_thoughts": 2,
                    "next_thought_needed": True,
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2

    def test_revises_nonexistent_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "revision",
                "revises_thought": 5,
            }
        )
        assert "revises_thought=5 not found in history" in result

    def test_revision_with_nonsequential_numbers(self):
        """thought_number=5 exists in history; revising it should work."""
        state = ThinkingState()
        state.process(
            {
                "thought": "Jump to 5",
                "thought_number": 5,
                "total_thoughts": 10,
                "next_thought_needed": True,
            }
        )
        # Revising thought 5 should succeed
        result = json.loads(
            state.process(
                {
                    "thought": "Fix thought 5",
                    "thought_number": 6,
                    "total_thoughts": 10,
                    "next_thought_needed": True,
                    "mode": "revision",
                    "revises_thought": 5,
                }
            )
        )
        assert result["history_length"] == 2
        # Revising thought 1 (never recorded) should fail
        result = state.process(
            {
                "thought": "Bad ref",
                "thought_number": 7,
                "total_thoughts": 10,
                "next_thought_needed": True,
                "mode": "revision",
                "revises_thought": 1,
            }
        )
        assert "revises_thought=1 not found in history" in result


class TestBranching:
    def test_valid_branch(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Alternative approach",
                    "thought_number": 2,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "mode": "branch",
                    "branch_from_thought": 1,
                    "branch_id": "approach-b",
                }
            )
        )
        assert "approach-b" in result["branches"]

    def test_branch_mode_without_branch_id(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
            }
        )
        assert "branch mode requires branch_id" in result

    def test_branch_mode_without_branch_from(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_id": "orphan",
            }
        )
        assert "branch mode requires branch_from_thought" in result

    def test_branch_from_nonexistent_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 99,
                "branch_id": "bad",
            }
        )
        assert "branch_from_thought=99 not found in history" in result

    def test_branch_with_nonsequential_numbers(self):
        """branch_from_thought should match actual thought_number, not index."""
        state = ThinkingState()
        state.process(
            {
                "thought": "Jump to 10",
                "thought_number": 10,
                "total_thoughts": 20,
                "next_thought_needed": True,
            }
        )
        # Branching from thought 10 should work
        result = json.loads(
            state.process(
                {
                    "thought": "Alt from 10",
                    "thought_number": 11,
                    "total_thoughts": 20,
                    "next_thought_needed": True,
                    "mode": "branch",
                    "branch_from_thought": 10,
                    "branch_id": "alt",
                }
            )
        )
        assert "alt" in result["branches"]
        # Branching from thought 1 (never recorded) should fail
        result = state.process(
            {
                "thought": "Bad branch",
                "thought_number": 12,
                "total_thoughts": 20,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
                "branch_id": "bad",
            }
        )
        assert "branch_from_thought=1 not found in history" in result

    def test_branch_id_too_long(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
                "branch_id": "x" * 51,
            }
        )
        assert "branch_id exceeds 50 character limit" in result

    def test_blank_branch_id(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
                "branch_id": "   ",
            }
        )
        assert result == "error: branch_id must not be blank"

    def test_too_many_branches(self):
        state = ThinkingState()
        # Create 20 branches
        for i in range(1, 22):
            state.process(
                {
                    "thought": f"Thought {i}",
                    "thought_number": i,
                    "total_thoughts": 25,
                    "next_thought_needed": True,
                }
            )
        # Now branch from existing thoughts
        for i in range(20):
            result = state.process(
                {
                    "thought": f"Branch {i}",
                    "thought_number": 22 + i,
                    "total_thoughts": 50,
                    "next_thought_needed": True,
                    "mode": "branch",
                    "branch_from_thought": 1,
                    "branch_id": f"branch-{i}",
                }
            )
            assert not result.startswith("error"), (
                f"Branch {i} should succeed: {result}"
            )
        # 21st branch should fail
        result = state.process(
            {
                "thought": "One too many",
                "thought_number": 42,
                "total_thoughts": 50,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
                "branch_id": "branch-20",
            }
        )
        assert result == "error: too many branches (20 max)"


class TestAutoAdjust:
    def test_thought_number_exceeds_total(self):
        state = ThinkingState()
        result = json.loads(
            state.process(
                {
                    "thought": "Overshot",
                    "thought_number": 7,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                }
            )
        )
        assert result["total_thoughts"] == 7


class TestDefaulting:
    """Tests for optional thought_number/total_thoughts/next_thought_needed."""

    def test_thought_only(self):
        """Calling with just 'thought' auto-increments and defaults correctly."""
        state = ThinkingState()
        r1 = json.loads(state.process({"thought": "First"}))
        assert r1["thought_number"] == 1
        assert r1["total_thoughts"] == 3  # default for first call
        assert r1["next_thought_needed"] is True  # default

        r2 = json.loads(state.process({"thought": "Second"}))
        assert r2["thought_number"] == 2
        assert r2["total_thoughts"] == 3  # carried forward

        r3 = json.loads(state.process({"thought": "Third"}))
        assert r3["thought_number"] == 3
        assert r3["total_thoughts"] == 3

    def test_explicit_params_still_work(self):
        """Passing all four fields works identically to before (no regression)."""
        state = ThinkingState()
        r = json.loads(
            state.process(
                {
                    "thought": "Step 1",
                    "thought_number": 1,
                    "total_thoughts": 5,
                    "next_thought_needed": True,
                }
            )
        )
        assert r["thought_number"] == 1
        assert r["total_thoughts"] == 5
        assert r["next_thought_needed"] is True

    def test_mixed_explicit_and_default(self):
        """Some calls provide explicit numbers, some rely on defaults."""
        state = ThinkingState()
        # Explicit first call
        r1 = json.loads(
            state.process(
                {
                    "thought": "Explicit",
                    "thought_number": 1,
                    "total_thoughts": 5,
                    "next_thought_needed": True,
                }
            )
        )
        assert r1["thought_number"] == 1
        assert r1["total_thoughts"] == 5

        # Default second call — should auto-increment, carry forward total
        r2 = json.loads(state.process({"thought": "Defaulted"}))
        assert r2["thought_number"] == 2
        assert r2["total_thoughts"] == 5  # carried from explicit call

        # Explicit third call with different total
        r3 = json.loads(
            state.process(
                {
                    "thought": "Explicit again",
                    "thought_number": 10,
                    "total_thoughts": 10,
                    "next_thought_needed": False,
                }
            )
        )
        assert r3["thought_number"] == 10
        assert r3["total_thoughts"] == 10
        assert r3["next_thought_needed"] is False

    def test_default_with_revision(self):
        """Revision still works when thought_number is auto-incremented."""
        state = ThinkingState()
        state.process({"thought": "First"})  # auto: thought_number=1
        r = json.loads(
            state.process(
                {
                    "thought": "Revise first",
                    "mode": "revision",
                    "revises_thought": 1,
                }
            )
        )
        assert r["thought_number"] == 2
        assert r["history_length"] == 2

    def test_default_with_branch(self):
        """Branching works when thought_number is auto-incremented."""
        state = ThinkingState()
        state.process({"thought": "Main"})  # auto: thought_number=1
        r = json.loads(
            state.process(
                {
                    "thought": "Alt approach",
                    "mode": "branch",
                    "branch_from_thought": 1,
                    "branch_id": "alt",
                }
            )
        )
        assert r["thought_number"] == 2
        assert "alt" in r["branches"]

    def test_dispatch_thought_only(self):
        """dispatch('think', {'thought': '...'}) returns valid JSON."""
        state = ThinkingState()
        result = dispatch(
            "think",
            {"thought": "Quick plan"},
            tempfile.mkdtemp(),
            thinking_state=state,
        )
        parsed = json.loads(result)
        assert parsed["thought_number"] == 1
        assert parsed["total_thoughts"] == 3
        assert parsed["history_length"] == 1


class TestTruncation:
    def test_long_thought_truncated(self):
        state = ThinkingState()
        long_text = "x" * 15000
        result = json.loads(
            state.process(
                {
                    "thought": long_text,
                    "thought_number": 1,
                    "total_thoughts": 1,
                    "next_thought_needed": False,
                }
            )
        )
        assert result["history_length"] == 1
        assert len(state.history[0].thought) == 10000


class TestHistoryCap:
    def test_201st_thought_rejected(self):
        state = ThinkingState()
        for i in range(1, 201):
            result = state.process(
                {
                    "thought": f"Step {i}",
                    "thought_number": i,
                    "total_thoughts": 250,
                    "next_thought_needed": True,
                }
            )
            assert not result.startswith("error"), f"Step {i} should succeed"
        result = state.process(
            {
                "thought": "One too many",
                "thought_number": 201,
                "total_thoughts": 250,
                "next_thought_needed": True,
            }
        )
        assert result == "error: thinking history full (200 steps max)"


# ---------------------------------------------------------------------------
# Sanitizer / coercion tests
# ---------------------------------------------------------------------------


class TestSanitizer:
    def test_template_payload_first_thought(self):
        """Full template payload on first call — should succeed as normal thought."""
        state = ThinkingState()
        result = json.loads(
            state.process(
                {
                    "thought": "Planning step",
                    "thought_number": 1,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "is_revision": False,
                    "revises_thought": 1,
                    "branch_from_thought": 1,
                    "branch_id": "main",
                }
            )
        )
        assert result["history_length"] == 1
        assert result["thought_number"] == 1
        # Should NOT be recorded as revision or branch
        entry = state.history[0]
        assert entry.is_revision is False
        assert entry.revises_thought is None
        assert entry.branch_from_thought is None
        assert entry.branch_id is None

    def test_is_revision_false_strips_revision_fields(self):
        """is_revision=false with revises_thought → stripped to normal thought."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Not a revision despite field",
                    "is_revision": False,
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2
        entry = state.history[1]
        assert entry.is_revision is False
        assert entry.revises_thought is None

    def test_revises_thought_without_flag_coerced(self):
        """revises_thought present, is_revision absent → coerced to revision."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Implicit revision",
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2
        entry = state.history[1]
        assert entry.is_revision is True
        assert entry.revises_thought == 1

    def test_branch_fields_without_mode_coerced(self):
        """branch_from_thought + branch_id without mode → coerced to branch."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Implicit branch",
                    "branch_from_thought": 1,
                    "branch_id": "alt",
                }
            )
        )
        assert result["history_length"] == 2
        assert "alt" in result["branches"]

    def test_mode_new_strips_all_optional(self):
        """Explicit mode=new strips revision and branch fields."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Forced new",
                    "mode": "new",
                    "revises_thought": 1,
                    "branch_from_thought": 1,
                    "branch_id": "x",
                }
            )
        )
        assert result["history_length"] == 2
        entry = state.history[1]
        assert entry.is_revision is False
        assert entry.revises_thought is None
        assert entry.branch_from_thought is None

    def test_mode_revision_strips_branch_fields(self):
        """mode=revision strips branch fields."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Revision ignoring branch",
                    "mode": "revision",
                    "revises_thought": 1,
                    "branch_from_thought": 1,
                    "branch_id": "stray",
                }
            )
        )
        assert result["history_length"] == 2
        entry = state.history[1]
        assert entry.is_revision is True
        assert entry.revises_thought == 1
        assert entry.branch_from_thought is None

    def test_mode_branch_strips_revision_fields(self):
        """mode=branch strips revision fields."""
        state = ThinkingState()
        state.process({"thought": "First"})
        result = json.loads(
            state.process(
                {
                    "thought": "Branch ignoring revision",
                    "mode": "branch",
                    "branch_from_thought": 1,
                    "branch_id": "alt",
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2
        entry = state.history[1]
        assert entry.is_revision is False
        assert entry.revises_thought is None
        assert "alt" in result["branches"]

    def test_revision_downgraded_when_no_history(self):
        """mode=revision on first call (no history) → downgraded to new."""
        state = ThinkingState()
        result = json.loads(
            state.process(
                {
                    "thought": "Can't revise nothing",
                    "mode": "revision",
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 1
        entry = state.history[0]
        assert entry.is_revision is False
        assert entry.revises_thought is None

    def test_branch_downgraded_when_no_history(self):
        """mode=branch on first call (no history) → downgraded to new."""
        state = ThinkingState()
        result = json.loads(
            state.process(
                {
                    "thought": "Can't branch from nothing",
                    "mode": "branch",
                    "branch_from_thought": 1,
                    "branch_id": "alt",
                }
            )
        )
        assert result["history_length"] == 1
        entry = state.history[0]
        assert entry.is_revision is False
        assert entry.branch_from_thought is None


# ---------------------------------------------------------------------------
# Corrective error payload tests
# ---------------------------------------------------------------------------


class TestCorrectiveErrors:
    def test_nonexistent_revises_thought_suggests_valid(self):
        state = ThinkingState()
        state.process({"thought": "First"})
        state.process({"thought": "Second"})
        result = state.process(
            {
                "thought": "Bad ref",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        assert "revises_thought=99 not found" in result
        assert "valid thought numbers: [1, 2]" in result

    def test_revision_without_revises_suggests_valid(self):
        state = ThinkingState()
        state.process({"thought": "First"})
        result = state.process(
            {
                "thought": "Missing target",
                "mode": "revision",
            }
        )
        assert "revision mode requires revises_thought" in result
        assert "valid thought numbers: [1]" in result

    def test_nonexistent_branch_from_suggests_valid(self):
        state = ThinkingState()
        state.process({"thought": "First"})
        result = state.process(
            {
                "thought": "Bad branch",
                "mode": "branch",
                "branch_from_thought": 99,
                "branch_id": "x",
            }
        )
        assert "branch_from_thought=99 not found" in result
        assert "valid thought numbers: [1]" in result

    def test_branch_missing_id_suggests_both(self):
        state = ThinkingState()
        state.process({"thought": "First"})
        result = state.process(
            {
                "thought": "Incomplete branch",
                "mode": "branch",
                "branch_from_thought": 1,
            }
        )
        assert "branch mode requires branch_id" in result
        assert "branch_from_thought and branch_id must be set together" in result


# ---------------------------------------------------------------------------
# One-shot sanitizer tests
# ---------------------------------------------------------------------------


class TestOneShotSanitizer:
    def test_repeated_error_auto_retries_as_minimal(self):
        """Same validation error twice → auto-retry as minimal thought."""
        state = ThinkingState()
        state.process({"thought": "First"})

        # First call: revision pointing to nonexistent thought
        r1 = state.process(
            {
                "thought": "Bad revision",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        assert r1.startswith("error:")

        # Second call: exact same payload → auto-sanitized, succeeds
        r2 = state.process(
            {
                "thought": "Bad revision",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        parsed = json.loads(r2)
        assert parsed["history_length"] == 2  # first + auto-sanitized

    def test_different_error_does_not_trigger_sanitizer(self):
        """Different errors don't trigger auto-retry."""
        state = ThinkingState()
        state.process({"thought": "First"})

        # First error: bad revision target
        r1 = state.process(
            {
                "thought": "Bad",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        assert r1.startswith("error:")

        # Different error: missing branch_id
        r2 = state.process(
            {
                "thought": "Also bad",
                "mode": "branch",
                "branch_from_thought": 1,
            }
        )
        assert r2.startswith("error:")

    def test_success_resets_error_tracking(self):
        """A successful call clears the error tracker."""
        state = ThinkingState()
        state.process({"thought": "First"})

        # Error
        r1 = state.process(
            {
                "thought": "Bad",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        assert r1.startswith("error:")

        # Success (resets tracker)
        r2 = json.loads(state.process({"thought": "Good"}))
        assert r2["history_length"] == 2

        # Same error again — NOT auto-retried (tracker was reset)
        r3 = state.process(
            {
                "thought": "Bad again",
                "mode": "revision",
                "revises_thought": 99,
            }
        )
        assert r3.startswith("error:")


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestLogging:
    def _reinit_console(self):
        """Reinitialize fmt console so it uses the current (capsys-patched) stderr."""
        from swival import fmt

        fmt.init(color=False, no_color=False)

    def test_verbose_logs_to_stderr(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Checking edge cases",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        captured = capsys.readouterr()
        assert "[think 1/3]" in captured.err
        assert "Checking edge cases" in captured.err

    def test_quiet_no_stderr(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=False)
        state.process(
            {
                "thought": "Silent thought",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            }
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_newline_normalization(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Line one\nLine two\n\nLine four",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            }
        )
        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")
        assert len(lines) == 1
        assert "Line one Line two Line four" in lines[0]

    def test_revision_log_format(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        state.process(
            {
                "thought": "Revised first",
                "thought_number": 2,
                "total_thoughts": 3,
                "next_thought_needed": True,
                "mode": "revision",
                "revises_thought": 1,
            }
        )
        captured = capsys.readouterr()
        assert "rev:1" in captured.err

    def test_branch_log_format(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Main line",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        state.process(
            {
                "thought": "Alternative",
                "thought_number": 2,
                "total_thoughts": 3,
                "next_thought_needed": True,
                "mode": "branch",
                "branch_from_thought": 1,
                "branch_id": "alt",
            }
        )
        captured = capsys.readouterr()
        assert "branch:alt" in captured.err
        assert "from:1" in captured.err


# ---------------------------------------------------------------------------
# Dispatch integration / regression tests
# ---------------------------------------------------------------------------


class TestDispatchRegression:
    """Verify existing tools still work with the new **kwargs signature."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a test file
        import os

        with open(os.path.join(self.tmpdir, "hello.txt"), "w") as f:
            f.write("Hello world\n")

    def test_read_file_no_kwargs(self):
        result = dispatch("read_file", {"file_path": "hello.txt"}, self.tmpdir)
        assert "Hello world" in result

    def test_write_file_no_kwargs(self):
        result = dispatch(
            "write_file", {"file_path": "new.txt", "content": "test"}, self.tmpdir
        )
        assert "Wrote" in result

    def test_grep_no_kwargs(self):
        result = dispatch("grep", {"pattern": "Hello"}, self.tmpdir)
        assert "Hello" in result

    def test_think_without_state(self):
        result = dispatch(
            "think",
            {
                "thought": "test",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            },
            self.tmpdir,
        )
        assert result == "error: think tool is not available"

    def test_think_with_state(self):
        state = ThinkingState()
        result = dispatch(
            "think",
            {
                "thought": "test",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            },
            self.tmpdir,
            thinking_state=state,
        )
        parsed = json.loads(result)
        assert parsed["history_length"] == 1


# ---------------------------------------------------------------------------
# Agent log-skip integration test
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments_json):
        self.id = "call_test"
        self.function = _FakeFunction(name, arguments_json)


class TestAgentLogSkip:
    """Verify that agent.py's handle_tool_call skips generic logging for think.

    Monkeypatches fmt.tool_call / fmt.tool_result / fmt.tool_error to track
    which tool names are logged. Think calls should be skipped.
    """

    def test_think_skips_generic_log(self, monkeypatch):
        from swival import agent
        from swival import fmt

        calls = []
        monkeypatch.setattr(
            fmt, "tool_call", lambda name, args: calls.append(("tool_call", name))
        )
        monkeypatch.setattr(
            fmt,
            "tool_result",
            lambda name, elapsed, preview: calls.append(("tool_result", name)),
        )
        monkeypatch.setattr(
            fmt, "tool_error", lambda name, msg: calls.append(("tool_error", name))
        )

        thinking_state = ThinkingState(verbose=False)
        base_dir = tempfile.mkdtemp()

        # Call the real handle_tool_call for a "think" tool call
        tool_call = _FakeToolCall(
            "think",
            json.dumps(
                {
                    "thought": "Planning step",
                    "thought_number": 1,
                    "total_thoughts": 2,
                    "next_thought_needed": True,
                }
            ),
        )
        result_msg, _meta = agent.handle_tool_call(
            tool_call, base_dir, thinking_state, verbose=True
        )
        assert result_msg["role"] == "tool"

        # No fmt.tool_* calls should have been made for think
        assert not calls, f"unexpected fmt calls for think: {calls}"

        # Now call handle_tool_call for "read_file" — generic logging should appear
        calls.clear()
        import os

        with open(os.path.join(base_dir, "test.txt"), "w") as f:
            f.write("hello\n")

        tool_call = _FakeToolCall("read_file", json.dumps({"file_path": "test.txt"}))
        result_msg, _meta = agent.handle_tool_call(
            tool_call, base_dir, thinking_state, verbose=True
        )
        assert result_msg["role"] == "tool"

        tool_call_names = [name for action, name in calls if action == "tool_call"]
        tool_result_names = [name for action, name in calls if action == "tool_result"]
        assert "read_file" in tool_call_names, f"missing tool_call: {calls}"
        assert "read_file" in tool_result_names, f"missing tool_result: {calls}"


# ---------------------------------------------------------------------------
# Usage counters and summary tests
# ---------------------------------------------------------------------------


class TestUsageCounters:
    def test_think_calls_counter(self):
        state = ThinkingState()
        assert state.think_calls == 0
        state.process({"thought": "First"})
        assert state.think_calls == 1
        state.process({"thought": "Second"})
        assert state.think_calls == 2

    def test_summary_line_no_calls(self):
        state = ThinkingState()
        assert state.summary_line() is None

    def test_summary_line_calls_only(self):
        state = ThinkingState()
        state.process({"thought": "A"})
        state.process({"thought": "B"})
        assert state.summary_line() == "think: 2 calls"

    def test_summary_line_single_call(self):
        state = ThinkingState()
        state.process({"thought": "Only one"})
        assert state.summary_line() == "think: 1 call"
