"""Tests for swival/audit.py — scope, record parsing, triage, verification, artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from swival.audit import (
    AUDIT_PROVENANCE_URL,
    AuditRunState,
    AuditScope,
    DeepReviewResult,
    FindingRecord,
    PhaseSchema,
    RecordSchema,
    TriageRecord,
    VerificationResult,
    VerifiedFinding,
    _TransientVerifierError,
    _build_context_indices,
    _canonicalize_finding,
    _extract_exports,
    _extract_imports,
    _finding_key,
    _git_show_many,
    _is_auditable,
    _make_slug,
    _load_file_contents,
    _order_by_attack_surface,
    _parse_records,
    _parse_records_with_repair,
    _score_attack_surface,
    _verify_one_finding,
    _verify_single_finding,
)
from swival.input_commands import INPUT_COMMANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git(tmp_path: Path) -> None:
    """Create a minimal git repo with one committed file."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )


def _commit_file(tmp_path: Path, rel_path: str, content: str) -> None:
    """Write and commit a file."""
    fp = tmp_path / rel_path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    subprocess.run(
        ["git", "add", rel_path], cwd=tmp_path, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", f"add {rel_path}"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    def test_audit_in_input_commands(self):
        assert "/audit" in INPUT_COMMANDS

    def test_audit_is_agent_turn(self):
        assert INPUT_COMMANDS["/audit"].kind == "agent_turn"

    def test_audit_modes(self):
        assert INPUT_COMMANDS["/audit"].modes == ("repl", "oneshot")


class TestAuditOneshotDispatch:
    """Verify /audit dispatches through execute_input in oneshot mode."""

    def test_audit_dispatches_in_oneshot(self, monkeypatch):
        import types as _types

        from swival.input_dispatch import InputContext, parse_input_line
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        ctx = InputContext(
            messages=[],
            tools=[],
            base_dir="/tmp",
            turn_state={"max_turns": 10, "turns_used": 0},
            thinking_state=ThinkingState(),
            todo_state=TodoState(),
            snapshot_state=None,
            file_tracker=None,
            no_history=True,
            continue_here=False,
            verbose=False,
            loop_kwargs={
                "model_id": "test",
                "api_base": "http://test",
                "context_length": 128000,
                "files_mode": "some",
                "compaction_state": None,
                "command_policy": _types.SimpleNamespace(mode="allowlist"),
                "top_p": None,
                "seed": None,
                "llm_kwargs": {},
            },
        )

        called = {}

        def fake_run_audit(cmd_arg, ctx_arg):
            called["cmd_arg"] = cmd_arg
            called["ctx"] = ctx_arg
            return "audit done"

        monkeypatch.setattr("swival.audit.run_audit_command", fake_run_audit)

        from swival.agent import execute_input

        parsed = parse_input_line("/audit")
        result = execute_input(parsed, ctx, mode="oneshot")

        assert "not available" not in (result.text or "")
        assert "cmd_arg" in called


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    def test_auditable_extensions(self):
        assert _is_auditable("foo.py")
        assert _is_auditable("bar.js")
        assert _is_auditable("config.toml")
        assert not _is_auditable("image.png")
        assert not _is_auditable("readme.md")
        assert not _is_auditable("data.csv")

    def test_scope_from_git(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "main.py", "print('hello')")
        _commit_file(tmp_path, "readme.md", "# Hello")
        _commit_file(tmp_path, "lib.js", "console.log('hi')")

        from swival.audit import _resolve_scope

        scope = _resolve_scope(str(tmp_path), [])
        assert "main.py" in scope.tracked_files
        assert "readme.md" in scope.tracked_files
        assert "main.py" in scope.mandatory_files
        assert "lib.js" in scope.mandatory_files
        assert "readme.md" not in scope.mandatory_files

    def test_scope_focus_restricts(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "src/a.py", "pass")
        _commit_file(tmp_path, "src/b.py", "pass")
        _commit_file(tmp_path, "lib/c.py", "pass")

        from swival.audit import _resolve_scope

        scope = _resolve_scope(str(tmp_path), ["src/"])
        assert "src/a.py" in scope.mandatory_files
        assert "src/b.py" in scope.mandatory_files
        assert "lib/c.py" not in scope.mandatory_files

    def test_scope_uses_committed_not_dirty(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "committed")
        # Dirty the working tree
        (tmp_path / "a.py").write_text("dirty")
        (tmp_path / "untracked.py").write_text("new")

        from swival.audit import _resolve_scope, _git_show

        scope = _resolve_scope(str(tmp_path), [])
        assert "untracked.py" not in scope.tracked_files
        content = _git_show("a.py", str(tmp_path))
        assert content == "committed"


# ---------------------------------------------------------------------------
# Attack-surface scoring
# ---------------------------------------------------------------------------


class TestAttackSurface:
    def test_high_score_for_dangerous_code(self):
        code = "subprocess.run(cmd)\nos.path.join(user_input)\neval(data)"
        assert _score_attack_surface(code) >= 9

    def test_zero_for_benign_code(self):
        code = "x = 1 + 2\nresult = x * 3"
        assert _score_attack_surface(code) == 0

    def test_ordering(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "safe.py", "x = 1")
        _commit_file(tmp_path, "danger.py", "subprocess.run(cmd)\neval(data)")

        cache = _load_file_contents(["safe.py", "danger.py"], str(tmp_path))
        ordered = _order_by_attack_surface(["safe.py", "danger.py"], cache)
        assert ordered[0] == "danger.py"


# ---------------------------------------------------------------------------
# Import / export extraction
# ---------------------------------------------------------------------------


class TestImportExport:
    def test_python_imports(self):
        code = "import os\nfrom pathlib import Path\nimport json"
        imports = _extract_imports(code)
        assert "os" in imports
        assert "pathlib" in imports
        assert "json" in imports

    def test_js_imports(self):
        code = "import express from 'express'\nconst fs = require('fs')"
        imports = _extract_imports(code)
        assert "express" in imports
        assert "fs" in imports

    def test_python_exports(self):
        code = "def handle_request():\n    pass\n\nclass UserModel:\n    pass\n\ndef _private():\n    pass"
        exports = _extract_exports(code)
        assert "handle_request" in exports
        assert "UserModel" in exports
        assert "_private" not in exports


# ---------------------------------------------------------------------------
# Batched git read (_git_show_many)
# ---------------------------------------------------------------------------


class TestGitShowMany:
    def test_basic_multiple_files(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "alpha\n")
        _commit_file(tmp_path, "b.py", "beta beta beta\n")
        _commit_file(tmp_path, "c.py", "gamma")

        out = _git_show_many(["a.py", "b.py", "c.py"], str(tmp_path))
        assert out == {"a.py": "alpha\n", "b.py": "beta beta beta\n", "c.py": "gamma"}

    def test_path_with_spaces(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "with space.py", "ok")
        out = _git_show_many(["with space.py"], str(tmp_path))
        assert out == {"with space.py": "ok"}

    def test_missing_path_skipped_without_desync(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "first.py", "FIRST")
        _commit_file(tmp_path, "third.py", "THIRD")

        out = _git_show_many(["first.py", "ghost.py", "third.py"], str(tmp_path))
        assert out == {"first.py": "FIRST", "third.py": "THIRD"}

    def test_non_blob_object_does_not_desync(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "subdir/inner.py", "INNER")
        _commit_file(tmp_path, "after.py", "AFTER")

        out = _git_show_many(["subdir", "after.py"], str(tmp_path))
        assert out == {"after.py": "AFTER"}

    def test_empty_file(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "empty.py", "")
        out = _git_show_many(["empty.py"], str(tmp_path))
        assert out == {"empty.py": ""}

    def test_no_trailing_newline(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "no_nl.py", "no newline at end")
        out = _git_show_many(["no_nl.py"], str(tmp_path))
        assert out == {"no_nl.py": "no newline at end"}

    def test_varied_sizes(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "small.py", "x")
        _commit_file(tmp_path, "medium.py", "y" * 1024)
        _commit_file(tmp_path, "large.py", "z" * (256 * 1024))

        out = _git_show_many(["small.py", "medium.py", "large.py"], str(tmp_path))
        assert out["small.py"] == "x"
        assert out["medium.py"] == "y" * 1024
        assert out["large.py"] == "z" * (256 * 1024)

    def test_matches_git_show_for_non_utf8(self, tmp_path):
        from swival.audit import _git_show

        _init_git(tmp_path)
        fp = tmp_path / "binary.py"
        fp.write_bytes(b"prefix\x80\x81\xfesuffix")
        subprocess.run(
            ["git", "add", "binary.py"], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "add binary"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )

        batch = _git_show_many(["binary.py"], str(tmp_path))
        single = _git_show("binary.py", str(tmp_path))
        assert batch["binary.py"] == single

    def test_rejects_path_with_newline(self, tmp_path):
        _init_git(tmp_path)
        with pytest.raises(RuntimeError):
            _git_show_many(["evil\nname.py"], str(tmp_path))

    def test_load_file_contents_falls_back_per_path(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "kept.py", "real")

        cache = _load_file_contents(["kept.py", "ghost.py"], str(tmp_path))
        assert cache == {"kept.py": "real"}


# ---------------------------------------------------------------------------
# Context indices (tokenization equivalence)
# ---------------------------------------------------------------------------


class TestBuildContextIndices:
    def test_basic_caller_index(self):
        cache = {
            "lib.py": "def handle_request():\n    pass\n",
            "app.py": "from lib import handle_request\nhandle_request()\n",
        }
        imp_idx, call_idx = _build_context_indices(["lib.py", "app.py"], cache)

        assert "lib" in imp_idx["app.py"]
        assert call_idx["app.py"] == ["lib.py"]
        assert "lib.py" not in call_idx

    def test_self_file_excluded(self):
        cache = {
            "lib.py": "def handle_request():\n    handle_request()\n",
        }
        _imp_idx, call_idx = _build_context_indices(["lib.py"], cache)
        assert "lib.py" not in call_idx

    def test_symbol_exported_from_multiple_files(self):
        cache = {
            "a.py": "def shared():\n    pass\n",
            "b.py": "def shared():\n    pass\n",
            "c.py": "shared()\n",
        }
        _imp_idx, call_idx = _build_context_indices(["a.py", "b.py", "c.py"], cache)
        assert call_idx["c.py"] == ["a.py", "b.py"]

    def test_substring_does_not_match(self):
        cache = {
            "lib.py": "def run():\n    pass\n",
            "app.py": "rerun()\nprerun()\nrunner()\n",
        }
        _imp_idx, call_idx = _build_context_indices(["lib.py", "app.py"], cache)
        assert "app.py" not in call_idx

    def test_underscore_and_digit_identifiers(self):
        cache = {
            "lib.py": "def handle_v2():\n    pass\n\ndef _private():\n    pass\n",
            "app.py": "handle_v2()\n_private()\n",
        }
        _imp_idx, call_idx = _build_context_indices(["lib.py", "app.py"], cache)
        assert call_idx["app.py"] == ["lib.py"]

    def test_no_overlap_no_entry(self):
        cache = {
            "lib.py": "def alpha():\n    pass\n",
            "app.py": "print('hello world')\n",
        }
        _imp_idx, call_idx = _build_context_indices(["lib.py", "app.py"], cache)
        assert call_idx == {}


# ---------------------------------------------------------------------------
# Structured-text record parsing
# ---------------------------------------------------------------------------


class TestParseRecords:
    """Unit tests for _parse_records and its strict validation pass."""

    _FINDING_SCHEMA = PhaseSchema(
        record=RecordSchema(
            name="finding",
            required=("title", "severity", "location", "claim"),
            enums={"severity": ("low", "medium", "high", "critical")},
        ),
        cardinality="zero_or_more",
        allow_none=True,
    )

    _FINDING_NO_NONE = PhaseSchema(
        record=RecordSchema(
            name="finding",
            required=("title", "severity", "location", "claim"),
            enums={"severity": ("low", "medium", "high", "critical")},
        ),
        cardinality="zero_or_more",
        allow_none=False,
    )

    _PROFILE_SCHEMA = PhaseSchema(
        record=RecordSchema(
            name="profile",
            required=("language", "summary"),
            optional=("framework", "entry_point"),
            repeated={
                "language": "languages",
                "framework": "frameworks",
                "entry_point": "entry_points",
            },
        ),
        cardinality="one",
    )

    _TRIAGE_SCHEMA = PhaseSchema(
        record=RecordSchema(
            name="triage",
            required=("priority", "summary"),
            enums={"priority": ("ESCALATE_HIGH", "ESCALATE_MEDIUM", "SKIP")},
            booleans=("needs_followup",),
        ),
        cardinality="one",
    )

    _EXPANSION_SCHEMA = PhaseSchema(
        record=RecordSchema(
            name="expansion",
            required=("type", "proof"),
            multiline=("proof",),
        ),
        cardinality="one",
    )

    # -- Happy path ---------------------------------------------------------

    def test_single_record_with_all_keys(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert result[0]["title"] == "a bug"
        assert result[0]["severity"] == "high"
        assert result[0]["location"] == "x.py:1"
        assert result[0]["claim"] == "it crashes"

    def test_multiple_records_of_same_type(self):
        text = (
            "@@ finding @@\n"
            "title: bug A\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: claim A\n"
            "\n"
            "@@ finding @@\n"
            "title: bug B\n"
            "severity: low\n"
            "location: y.py:2\n"
            "claim: claim B\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 2
        assert result[0]["title"] == "bug A"
        assert result[1]["title"] == "bug B"

    def test_multiline_continuation_joins(self):
        text = (
            "@@ expansion @@\n"
            "type: vulnerability\n"
            "proof:\n"
            "  user input arrives at line 10\n"
            "  flows to eval at line 20\n"
            "  reachable from public handler\n"
        )
        result = _parse_records(text, self._EXPANSION_SCHEMA)
        assert (
            result[0]["proof"]
            == "user input arrives at line 10 flows to eval at line 20 "
            "reachable from public handler"
        )

    def test_preamble_before_first_header_is_ignored(self):
        text = (
            "Here is my analysis. I will produce one finding.\n"
            "\n"
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert result[0]["title"] == "a bug"

    def test_fenced_code_block_unwrapped(self):
        text = (
            "```\n"
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
            "```"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert result[0]["title"] == "a bug"

    def test_key_casing_and_separators_accepted(self):
        text = (
            "@@ FINDING @@\n"
            "Title: a bug\n"
            "SEVERITY = high\n"
            "location: x.py:1\n"
            "Claim = it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert result[0]["title"] == "a bug"
        assert result[0]["severity"] == "high"
        assert result[0]["claim"] == "it crashes"

    def test_header_extra_whitespace_matches(self):
        text = (
            "@@   finding   @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1

    def test_repeated_keys_collect_to_plural(self):
        text = (
            "@@ profile @@\n"
            "language: python\n"
            "language: rust\n"
            "framework: pytest\n"
            "summary: a tiny tool\n"
        )
        result = _parse_records(text, self._PROFILE_SCHEMA)
        assert result[0]["languages"] == ["python", "rust"]
        assert result[0]["frameworks"] == ["pytest"]
        assert result[0]["entry_points"] == []
        assert result[0]["summary"] == "a tiny tool"

    def test_none_sentinel_returns_empty_when_allowed(self):
        result = _parse_records("@@ none @@", self._FINDING_SCHEMA)
        assert result == []

    def test_none_sentinel_with_preamble(self):
        result = _parse_records(
            "I found nothing.\n\n@@ none @@\n", self._FINDING_SCHEMA
        )
        assert result == []

    # -- Strict-after-parse -------------------------------------------------

    def test_missing_required_key_raises_with_field_and_index(self):
        text = "@@ finding @@\ntitle: a bug\nseverity: high\nlocation: x.py:1\n"
        with pytest.raises(
            ValueError, match="missing required key 'claim' in record 0"
        ):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_finding_missing_claim_fails_whole_response(self):
        text = (
            "@@ finding @@\n"
            "title: bug A\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: claim A\n"
            "\n"
            "@@ finding @@\n"
            "title: bug B\n"
            "severity: low\n"
            "location: y.py:2\n"
        )
        with pytest.raises(
            ValueError, match="missing required key 'claim' in record 1"
        ):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_none_sentinel_rejected_when_not_allowed(self):
        with pytest.raises(ValueError, match="not permitted"):
            _parse_records("@@ none @@", self._FINDING_NO_NONE)

    def test_cardinality_one_rejects_zero_records(self):
        with pytest.raises(ValueError, match="exactly one"):
            _parse_records("just prose, nothing else", self._PROFILE_SCHEMA)

    def test_cardinality_one_rejects_two_records(self):
        text = (
            "@@ profile @@\n"
            "language: python\n"
            "summary: tool A\n"
            "\n"
            "@@ profile @@\n"
            "language: rust\n"
            "summary: tool B\n"
        )
        with pytest.raises(ValueError, match="exactly one"):
            _parse_records(text, self._PROFILE_SCHEMA)

    def test_enum_out_of_set_raises(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: catastrophic\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        with pytest.raises(ValueError, match="invalid enum value 'catastrophic'"):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_zero_records_no_allow_none_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _parse_records("just preamble", self._FINDING_NO_NONE)

    def test_prose_only_with_allow_none_still_raises(self):
        """allow_none means the @@ none @@ sentinel is permitted; it does NOT
        mean the parser silently accepts a prose-only response."""
        with pytest.raises(
            ValueError, match="at least one .* or the '@@ none @@' sentinel"
        ):
            _parse_records(
                "Here is a critical issue but no proper @@ block format.",
                self._FINDING_SCHEMA,
            )

    def test_empty_value_with_allow_none_still_raises(self):
        """An empty response under allow_none must error rather than coerce
        to []. The model has to either emit a record or the sentinel."""
        with pytest.raises(ValueError, match="empty"):
            _parse_records("", self._FINDING_SCHEMA)

    def test_empty_repeated_value_is_dropped(self):
        text = (
            "@@ profile @@\nlanguage: python\nlanguage:\nlanguage: rust\nsummary: ok\n"
        )
        result = _parse_records(text, self._PROFILE_SCHEMA)
        assert result[0]["languages"] == ["python", "rust"]

    def test_required_repeated_with_only_empty_values_raises_missing(self):
        text = "@@ profile @@\nlanguage:\nlanguage:\nsummary: ok\n"
        with pytest.raises(ValueError, match="missing required key 'language'"):
            _parse_records(text, self._PROFILE_SCHEMA)

    def test_optional_repeated_with_only_empty_values_normalizes_to_empty(self):
        text = "@@ profile @@\nlanguage: python\nframework:\nframework:\nsummary: ok\n"
        result = _parse_records(text, self._PROFILE_SCHEMA)
        assert result[0]["frameworks"] == []

    # -- Adversarial --------------------------------------------------------

    def test_value_with_commas_preserved_verbatim(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: foo, bar, baz\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert result[0]["claim"] == "foo, bar, baz"

    def test_trailer_text_does_not_merge(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
            "\n"
            "This is some trailing prose. It mentions a key: value somewhere.\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert "trailing" not in result[0]["claim"]

    def test_duplicate_scalar_key_raises(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "severity: medium\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        with pytest.raises(ValueError, match="duplicate key 'severity' in record 0"):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_key_known_to_other_schema_terminates_record(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
            "priority: ESCALATE_HIGH\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert "priority" not in result[0]

    def test_continuation_before_any_key_rejected(self):
        text = "@@ expansion @@\n  some stray continuation\ntype: bug\nproof: short\n"
        with pytest.raises(ValueError, match="continuation line before any key"):
            _parse_records(text, self._EXPANSION_SCHEMA)

    def test_continuation_on_non_multiline_key_rejected(self):
        text = (
            "@@ expansion @@\n"
            "type: bug\n"
            "  this should not continue type\n"
            "proof: short\n"
        )
        with pytest.raises(ValueError, match="not multiline"):
            _parse_records(text, self._EXPANSION_SCHEMA)

    def test_mixed_record_types_raises(self):
        text = (
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
            "\n"
            "@@ profile @@\n"
            "language: python\n"
            "summary: x\n"
        )
        with pytest.raises(ValueError, match="unexpected record type"):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_mixed_case_header_accepted(self):
        text = (
            "@@ Finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1

    def test_empty_value_counts_as_missing(self):
        text = (
            "@@ finding @@\n"
            "title:\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        with pytest.raises(ValueError, match="missing required key 'title'"):
            _parse_records(text, self._FINDING_SCHEMA)

    def test_at_at_substring_not_treated_as_header(self):
        text = (
            "@@ finding @@\n"
            "title: bug with @@ marker @@ in title\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        result = _parse_records(text, self._FINDING_SCHEMA)
        assert len(result) == 1
        assert "@@ marker @@" in result[0]["title"]

    def test_boolean_true_coercion(self):
        text = "@@ triage @@\npriority: SKIP\nsummary: tiny\nneeds_followup: yes\n"
        result = _parse_records(text, self._TRIAGE_SCHEMA)
        assert result[0]["needs_followup"] is True

    def test_boolean_false_coercion(self):
        text = "@@ triage @@\npriority: SKIP\nsummary: tiny\nneeds_followup: false\n"
        result = _parse_records(text, self._TRIAGE_SCHEMA)
        assert result[0]["needs_followup"] is False

    def test_invalid_boolean_raises(self):
        text = "@@ triage @@\npriority: SKIP\nsummary: tiny\nneeds_followup: maybe\n"
        with pytest.raises(ValueError, match="invalid boolean"):
            _parse_records(text, self._TRIAGE_SCHEMA)

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_records("", self._FINDING_SCHEMA)

    def test_none_sentinel_mixed_with_records_raises(self):
        text = (
            "@@ none @@\n"
            "\n"
            "@@ finding @@\n"
            "title: a bug\n"
            "severity: high\n"
            "location: x.py:1\n"
            "claim: it crashes\n"
        )
        with pytest.raises(ValueError, match="mixed"):
            _parse_records(text, self._FINDING_SCHEMA)


class TestParseRecordsWithRepair:
    """Integration of _parse_records_with_repair with metric tracking."""

    _SCHEMA = PhaseSchema(
        record=RecordSchema(
            name="finding",
            required=("title", "severity", "location", "claim"),
            enums={"severity": ("low", "medium", "high", "critical")},
        ),
        cardinality="zero_or_more",
        allow_none=True,
    )

    _EXAMPLE = (
        "@@ finding @@\n"
        "title: example\n"
        "severity: low\n"
        "location: x.py:1\n"
        "claim: example claim\n"
    )

    def test_clean_response_no_repair(self):
        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}
        result = _parse_records_with_repair(
            ctx=None,
            raw="@@ none @@",
            schema=self._SCHEMA,
            worked_example=self._EXAMPLE,
            metrics=metrics,
        )
        assert result == []
        assert metrics["parse_failures"] == 0

    def test_repair_succeeds(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: "@@ none @@",
        )
        result = _parse_records_with_repair(
            ctx=SimpleNamespace(),
            raw="@@ finding @@\ntitle: incomplete\n",
            schema=self._SCHEMA,
            worked_example=self._EXAMPLE,
            metrics=metrics,
        )
        assert result == []
        assert metrics["parse_failures"] == 1
        assert metrics["repair_successes"] == 1

    def test_repair_failure_propagates(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ finding @@\ntitle: still incomplete\n"
            ),
        )
        with pytest.raises(ValueError):
            _parse_records_with_repair(
                ctx=SimpleNamespace(),
                raw="@@ finding @@\ntitle: incomplete\n",
                schema=self._SCHEMA,
                worked_example=self._EXAMPLE,
                metrics=metrics,
            )
        assert metrics["parse_failures"] == 1
        assert metrics["repair_failures"] == 1
        assert metrics["repair_successes"] == 0


# ---------------------------------------------------------------------------
# Phase 3b expansion contract
# ---------------------------------------------------------------------------


class TestPhase3bExpansion:
    """Phase 3b structured-text contract: schema, multiline proof, repair."""

    def _schema(self):
        from swival.audit import _PHASE3B_EXPANSION_SCHEMA

        return _PHASE3B_EXPANSION_SCHEMA

    def _example(self):
        from swival.audit import _PHASE3B_WORKED_EXAMPLE

        return _PHASE3B_WORKED_EXAMPLE

    def test_happy_path_with_multiline_proof(self):
        text = (
            "@@ expansion @@\n"
            "type: code execution\n"
            "attacker: remote client\n"
            "trigger: crafted request parameter reaches eval\n"
            "impact: arbitrary code execution as server user\n"
            "preconditions: caller passes user input\n"
            "proof:\n"
            "  user input reaches eval at line 10.\n"
            "  no validation between origin and sink.\n"
            "  reachable from a request handler.\n"
            "fix_outline: validate input before eval\n"
        )
        result = _parse_records(text, self._schema())
        assert len(result) == 1
        proof = result[0]["proof"]
        assert "user input reaches eval at line 10." in proof
        assert "no validation between origin and sink." in proof
        assert "reachable from a request handler." in proof

    def test_multiline_continuations_metric_increments(self):
        text = (
            "@@ expansion @@\n"
            "type: code execution\n"
            "attacker: remote client\n"
            "trigger: crafted request parameter reaches eval\n"
            "impact: arbitrary code execution as server user\n"
            "preconditions: x\n"
            "proof:\n"
            "  line 1\n"
            "  line 2\n"
            "fix_outline: y\n"
        )
        metrics = {"multiline_continuations": 0}
        _parse_records(text, self._schema(), metrics=metrics)
        assert metrics["multiline_continuations"] == 1

    def test_multiline_metric_not_incremented_on_failure(self):
        # Schema requires fix_outline; missing → validation failure.
        text = (
            "@@ expansion @@\n"
            "type: code execution\n"
            "attacker: remote client\n"
            "trigger: crafted request parameter reaches eval\n"
            "impact: arbitrary code execution as server user\n"
            "preconditions: x\n"
            "proof:\n"
            "  line 1\n"
            "  line 2\n"
        )
        metrics = {"multiline_continuations": 0}
        with pytest.raises(ValueError, match="fix_outline"):
            _parse_records(text, self._schema(), metrics=metrics)
        assert metrics["multiline_continuations"] == 0

    def test_bad_enum_fails(self):
        text = (
            "@@ expansion @@\n"
            "type: not-in-list\n"
            "attacker: remote client\n"
            "trigger: crafted request parameter reaches sink\n"
            "impact: arbitrary code execution as server user\n"
            "preconditions: x\n"
            "proof: y\n"
            "fix_outline: z\n"
        )
        with pytest.raises(ValueError, match="invalid enum value 'not-in-list'"):
            _parse_records(text, self._schema())

    def test_bad_enum_then_repair(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {
            "parse_failures": 0,
            "repair_successes": 0,
            "repair_failures": 0,
            "multiline_continuations": 0,
        }
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: crafted request parameter reaches sink\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: x\n"
                "proof: y\n"
                "fix_outline: z\n"
            ),
        )
        result = _parse_records_with_repair(
            ctx=SimpleNamespace(),
            raw=(
                "@@ expansion @@\n"
                "type: speculation\n"
                "attacker: remote client\n"
                "trigger: crafted request parameter reaches sink\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: x\n"
                "proof: y\n"
                "fix_outline: z\n"
            ),
            schema=self._schema(),
            worked_example=self._example(),
            metrics=metrics,
        )
        assert len(result) == 1
        assert result[0]["type"] == "code execution"
        assert metrics["parse_failures"] == 1
        assert metrics["repair_successes"] == 1

    def test_duplicate_scalar_key_fails(self):
        text = (
            "@@ expansion @@\n"
            "type: code execution\n"
            "type: logic error\n"
            "attacker: remote client\n"
            "trigger: crafted request parameter reaches sink\n"
            "impact: arbitrary code execution as server user\n"
            "preconditions: x\n"
            "proof: y\n"
            "fix_outline: z\n"
        )
        with pytest.raises(ValueError, match="duplicate key 'type'"):
            _parse_records(text, self._schema())

    def test_missing_field_repairs_successfully(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {
            "parse_failures": 0,
            "repair_successes": 0,
            "repair_failures": 0,
        }
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: crafted request parameter reaches sink\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: caller passes input\n"
                "proof: input reaches sink\n"
                "fix_outline: validate it\n"
            ),
        )
        result = _parse_records_with_repair(
            ctx=SimpleNamespace(),
            raw="@@ expansion @@\ntype: code execution\n",
            schema=self._schema(),
            worked_example=self._example(),
            metrics=metrics,
        )
        assert len(result) == 1
        assert result[0]["fix_outline"] == "validate it"
        assert metrics["parse_failures"] == 1
        assert metrics["repair_successes"] == 1

    def test_repair_prompt_forbids_new_claims(self, monkeypatch):
        from types import SimpleNamespace

        captured: list = []

        def fake(ctx, messages, temperature=0.0, trace_task=None):
            captured.append(messages)
            return (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: crafted request parameter reaches sink\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: x\n"
                "proof: y\n"
                "fix_outline: z\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake)
        metrics = {
            "parse_failures": 0,
            "repair_successes": 0,
            "repair_failures": 0,
        }
        _parse_records_with_repair(
            ctx=SimpleNamespace(),
            raw="@@ expansion @@\ntype: code execution\n",
            schema=self._schema(),
            worked_example=self._example(),
            metrics=metrics,
        )
        assert len(captured) == 1
        repair_user = captured[0][1]["content"].lower()
        # User-message guard
        assert "do not introduce new" in repair_user
        assert "do not add new claims" in repair_user
        # System-message guard
        repair_system = captured[0][0]["content"].lower()
        assert "do not invent" in repair_system

    def test_end_to_end_phase3_records_produce_finding_record(
        self, monkeypatch, tmp_path
    ):
        """Phase 3a + 3b in @@ format should produce the same FindingRecord
        shape as the JSON-era pipeline."""
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="e2e",
            scope=scope,
            queued_files=["a.py"],
            triage_records={
                "a.py": TriageRecord(
                    path="a.py",
                    priority="ESCALATE_HIGH",
                    confidence="high",
                    bug_classes=["unsafe_data_flow"],
                    summary="x",
                    relevant_symbols=[],
                    suspicious_flows=[],
                    needs_followup=True,
                )
            },
            repo_profile={"summary": "tiny repo"},
            import_index={},
            caller_index={},
            state_dir=tmp_path,
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr(
            "swival.audit._git_show", lambda path, base_dir: "eval(input())"
        )

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: eval injection\n"
                    "severity: critical\n"
                    "location: a.py:1\n"
                    "attacker: remote client\n"
                    "trigger: request body reaches eval\n"
                    "impact: arbitrary code execution as server user\n"
                    "claim: user input flows directly into eval\n"
                )
            return (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: request body reaches eval\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: caller invokes the handler with attacker input\n"
                "proof:\n"
                "  user input arrives at line 1 via input() call.\n"
                "  flows directly to eval() with no sanitization.\n"
                "  reachable from any HTTP request handler.\n"
                "fix_outline: replace eval with ast.literal_eval or remove entirely\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f.title == "eval injection"
        assert f.severity == "critical"
        assert f.finding_type == "code execution"
        assert f.locations == ["a.py:1"]
        assert f.source_file == "a.py"
        assert "user input arrives at line 1" in f.proof[0]
        assert "reachable from any HTTP request handler" in f.proof[0]
        assert f.fix_outline.startswith("replace eval")
        assert state.metrics["multiline_continuations"] == 1
        assert state.metrics["parse_failures"] == 0


# ---------------------------------------------------------------------------
# Phase 2 triage contract (fail-closed, no repair)
# ---------------------------------------------------------------------------


class TestPhase2Triage:
    """Phase 2 structured-text contract: parse only, fail-closed to SKIP."""

    def _state(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        return AuditRunState(
            run_id="t",
            scope=scope,
            queued_files=["a.py"],
            repo_profile={"summary": "tiny repo"},
            import_index={},
            caller_index={},
            state_dir=tmp_path,
        )

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

    def _patch_git(self, monkeypatch):
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")

    def _patch_llm(self, monkeypatch, response):
        calls = {"n": 0}

        def fake(ctx, messages, temperature=None, trace_task=None):
            calls["n"] += 1
            return response

        monkeypatch.setattr("swival.audit._call_audit_llm", fake)
        return calls

    def test_happy_path_full_record(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\n"
            "priority: ESCALATE_HIGH\n"
            "confidence: high\n"
            "summary: parses untrusted url before authentication\n"
            "bug_class: input_validation\n"
            "bug_class: trust_boundary_breaks\n"
            "relevant_symbol: parse_url\n"
            "relevant_symbol: authenticate\n"
            "suspicious_flow: external request body reaches parse_url\n"
            "needs_followup: true\n",
        )

        state = self._state(tmp_path)
        rec = _phase2_triage_one("a.py", state, self._ctx(tmp_path))
        assert rec.priority == "ESCALATE_HIGH"
        assert rec.confidence == "high"
        assert rec.bug_classes == ["input_validation", "trust_boundary_breaks"]
        assert rec.relevant_symbols == ["parse_url", "authenticate"]
        assert rec.suspicious_flows == ["external request body reaches parse_url"]
        assert rec.needs_followup is True
        assert state.metrics["parse_failures"] == 0

    def test_missing_repeated_keys_normalize_to_empty(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\npriority: SKIP\nconfidence: high\nsummary: ok\n",
        )

        rec = _phase2_triage_one("a.py", self._state(tmp_path), self._ctx(tmp_path))
        assert rec.priority == "SKIP"
        assert rec.bug_classes == []
        assert rec.relevant_symbols == []
        assert rec.suspicious_flows == []

    def test_needs_followup_defaults_to_false_when_omitted(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\n"
            "priority: SKIP\n"
            "confidence: low\n"
            "summary: nothing of interest\n",
        )

        rec = _phase2_triage_one("a.py", self._state(tmp_path), self._ctx(tmp_path))
        assert rec.needs_followup is False

    def test_priority_case_normalized_to_upper(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\n"
            "priority: escalate_medium\n"
            "confidence: MEDIUM\n"
            "summary: fits\n",
        )

        rec = _phase2_triage_one("a.py", self._state(tmp_path), self._ctx(tmp_path))
        assert rec.priority == "ESCALATE_MEDIUM"
        assert rec.confidence == "medium"

    def test_invalid_priority_falls_back_to_skip(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\npriority: ESCALATE_LATER\nconfidence: high\nsummary: ok\n",
        )

        state = self._state(tmp_path)
        rec = _phase2_triage_one("a.py", state, self._ctx(tmp_path))
        assert rec.priority == "SKIP"
        assert rec.summary == "triage failed (unparseable LLM response)"
        assert state.metrics["parse_failures"] == 1

    def test_invalid_confidence_falls_back_to_skip(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(
            monkeypatch,
            "@@ triage @@\npriority: SKIP\nconfidence: somewhat\nsummary: ok\n",
        )

        state = self._state(tmp_path)
        rec = _phase2_triage_one("a.py", state, self._ctx(tmp_path))
        assert rec.priority == "SKIP"
        assert rec.summary == "triage failed (unparseable LLM response)"
        assert state.metrics["parse_failures"] == 1

    def test_malformed_response_falls_back_to_skip(self, monkeypatch, tmp_path):
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        self._patch_llm(monkeypatch, "I am not following the format at all.")

        state = self._state(tmp_path)
        rec = _phase2_triage_one("a.py", state, self._ctx(tmp_path))
        assert rec.priority == "SKIP"
        assert rec.summary == "triage failed (unparseable LLM response)"
        assert state.metrics["parse_failures"] == 1

    def test_phase2_does_not_call_repair(self, monkeypatch, tmp_path):
        """Phase 2 must fail-closed without invoking the repair path —
        triage volume makes per-failure repair the wrong cost profile."""
        from swival.audit import _phase2_triage_one

        self._patch_git(monkeypatch)
        calls = self._patch_llm(monkeypatch, "totally broken response")

        state = self._state(tmp_path)
        rec = _phase2_triage_one("a.py", state, self._ctx(tmp_path))
        assert rec.priority == "SKIP"
        # Exactly one LLM call: the initial triage. No repair.
        assert calls["n"] == 1
        assert state.metrics["parse_failures"] == 1
        assert state.metrics.get("repair_successes", 0) == 0
        assert state.metrics.get("repair_failures", 0) == 0


# ---------------------------------------------------------------------------
# Phase 1 repo profile contract
# ---------------------------------------------------------------------------


class TestPhase1Profile:
    """Phase 1 structured-text contract: schema, repair, plural-key canonicalization."""

    def _state(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        return AuditRunState(
            run_id="p1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path,
        )

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

    def _schema(self):
        from swival.audit import _PHASE1_PROFILE_SCHEMA

        return _PHASE1_PROFILE_SCHEMA

    def _example(self):
        from swival.audit import _PHASE1_WORKED_EXAMPLE

        return _PHASE1_WORKED_EXAMPLE

    def test_happy_path_repeated_keys_to_plural(self):
        text = (
            "@@ profile @@\n"
            "language: python\n"
            "language: rust\n"
            "framework: pytest\n"
            "framework: uv\n"
            "entry_point: swival/agent.py\n"
            "entry_point: swival/audit.py\n"
            "trust_boundary: cli args\n"
            "trust_boundary: mcp servers\n"
            "persistence_layer: .swival/HISTORY.md\n"
            "auth_surface: chatgpt oauth device flow\n"
            "dangerous_operation: subprocess\n"
            "dangerous_operation: file write\n"
            "summary: a python cli coding agent with mcp client and audit pipeline.\n"
        )
        result = _parse_records(text, self._schema())
        assert len(result) == 1
        profile = result[0]
        assert profile["languages"] == ["python", "rust"]
        assert profile["frameworks"] == ["pytest", "uv"]
        assert profile["entry_points"] == ["swival/agent.py", "swival/audit.py"]
        assert profile["trust_boundaries"] == ["cli args", "mcp servers"]
        assert profile["persistence_layers"] == [".swival/HISTORY.md"]
        assert profile["auth_surfaces"] == ["chatgpt oauth device flow"]
        assert profile["dangerous_operations"] == ["subprocess", "file write"]
        assert profile["summary"].startswith("a python cli")

    def test_missing_optional_repeated_keys_default_to_empty(self):
        text = "@@ profile @@\nlanguage: python\nsummary: minimal profile\n"
        result = _parse_records(text, self._schema())
        profile = result[0]
        assert profile["languages"] == ["python"]
        assert profile["summary"] == "minimal profile"
        assert profile["frameworks"] == []
        assert profile["entry_points"] == []
        assert profile["trust_boundaries"] == []
        assert profile["persistence_layers"] == []
        assert profile["auth_surfaces"] == []
        assert profile["dangerous_operations"] == []

    def test_missing_summary_fails(self):
        text = "@@ profile @@\nlanguage: python\n"
        with pytest.raises(ValueError, match="missing required key 'summary'"):
            _parse_records(text, self._schema())

    def test_missing_language_fails(self):
        text = "@@ profile @@\nsummary: nothing to see\n"
        with pytest.raises(ValueError, match="missing required key 'language'"):
            _parse_records(text, self._schema())

    def test_two_profile_records_rejected(self):
        text = (
            "@@ profile @@\n"
            "language: python\n"
            "summary: first\n"
            "\n"
            "@@ profile @@\n"
            "language: rust\n"
            "summary: second\n"
        )
        with pytest.raises(ValueError, match="exactly one"):
            _parse_records(text, self._schema())

    def test_repo_profile_returns_canonicalized_dict(self, monkeypatch, tmp_path):
        from swival.audit import _phase1_repo_profile

        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ profile @@\n"
                "language: python\n"
                "framework: pytest\n"
                "entry_point: swival/agent.py\n"
                "summary: a tiny tool\n"
            ),
        )

        state = self._state(tmp_path)
        profile = _phase1_repo_profile(state, self._ctx(tmp_path))
        assert profile["languages"] == ["python"]
        assert profile["frameworks"] == ["pytest"]
        assert profile["entry_points"] == ["swival/agent.py"]
        assert profile["summary"] == "a tiny tool"
        # Downstream phases JSON-encode this dict for prompt context.
        import json as _json

        encoded = _json.dumps(profile)
        assert "languages" in encoded
        assert "summary" in encoded

    def test_repo_profile_repairs_missing_field(self, monkeypatch, tmp_path):
        from swival.audit import _phase1_repo_profile

        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")
        calls = {"n": 0}

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Initial response: missing required summary
                return "@@ profile @@\nlanguage: python\n"
            return "@@ profile @@\nlanguage: python\nsummary: repaired summary\n"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        state = self._state(tmp_path)
        profile = _phase1_repo_profile(state, self._ctx(tmp_path))
        assert profile["summary"] == "repaired summary"
        assert state.metrics["parse_failures"] == 1
        assert state.metrics["repair_successes"] == 1


# ---------------------------------------------------------------------------
# Per-record-type parse-failure breakdown
# ---------------------------------------------------------------------------


class TestParseFailureBreakdown:
    """Typed parse_failures_<type> counters increment alongside the aggregate,
    and the formatter renders the breakdown when at least one is populated."""

    def _new_metrics(self):
        # Mirror AuditRunState.metrics defaults so the assertions cover the
        # actual on-disk shape, not a hand-rolled dict.
        scope = AuditScope(
            branch="main",
            commit="abc",
            tracked_files=[],
            mandatory_files=[],
            focus=[],
        )
        state = AuditRunState(run_id="m", scope=scope, queued_files=[])
        return state.metrics

    def test_default_metrics_include_typed_counters(self):
        m = self._new_metrics()
        for key in (
            "parse_failures",
            "parse_failures_profile",
            "parse_failures_triage",
            "parse_failures_finding",
            "parse_failures_expansion",
        ):
            assert m[key] == 0

    def test_records_with_repair_increments_typed_finding(self, monkeypatch):
        from types import SimpleNamespace
        from swival.audit import _PHASE3A_FINDING_SCHEMA, _PHASE3A_WORKED_EXAMPLE

        metrics = self._new_metrics()
        # Repair returns a malformed response too — drives parse failure
        # without succeeding, so we can pin the typed counter to one event.
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ finding @@\ntitle: bad\n"
            ),
        )
        with pytest.raises(ValueError):
            _parse_records_with_repair(
                ctx=SimpleNamespace(),
                raw="@@ finding @@\ntitle: incomplete\n",
                schema=_PHASE3A_FINDING_SCHEMA,
                worked_example=_PHASE3A_WORKED_EXAMPLE,
                metrics=metrics,
            )
        assert metrics["parse_failures"] == 1
        assert metrics["parse_failures_finding"] == 1
        assert metrics["parse_failures_triage"] == 0
        assert metrics["parse_failures_expansion"] == 0
        assert metrics["parse_failures_profile"] == 0

    def test_records_with_repair_increments_typed_expansion(self, monkeypatch):
        from types import SimpleNamespace
        from swival.audit import (
            _PHASE3B_EXPANSION_SCHEMA,
            _PHASE3B_WORKED_EXAMPLE,
        )

        metrics = self._new_metrics()
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ expansion @@\ntype: code execution\n"
            ),
        )
        with pytest.raises(ValueError):
            _parse_records_with_repair(
                ctx=SimpleNamespace(),
                raw="@@ expansion @@\ntype: code execution\n",
                schema=_PHASE3B_EXPANSION_SCHEMA,
                worked_example=_PHASE3B_WORKED_EXAMPLE,
                metrics=metrics,
            )
        assert metrics["parse_failures"] == 1
        assert metrics["parse_failures_expansion"] == 1
        assert metrics["parse_failures_finding"] == 0

    def test_records_with_repair_increments_typed_profile(self, monkeypatch):
        from types import SimpleNamespace
        from swival.audit import _PHASE1_PROFILE_SCHEMA, _PHASE1_WORKED_EXAMPLE

        metrics = self._new_metrics()
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ profile @@\nlanguage: python\n"
            ),
        )
        with pytest.raises(ValueError):
            _parse_records_with_repair(
                ctx=SimpleNamespace(),
                raw="@@ profile @@\nlanguage: python\n",
                schema=_PHASE1_PROFILE_SCHEMA,
                worked_example=_PHASE1_WORKED_EXAMPLE,
                metrics=metrics,
            )
        assert metrics["parse_failures"] == 1
        assert metrics["parse_failures_profile"] == 1

    def test_phase2_fail_closed_increments_typed_triage(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _phase2_triage_one

        scope = AuditScope(
            branch="main",
            commit="abc",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="t",
            scope=scope,
            queued_files=["a.py"],
            repo_profile={"summary": "tiny"},
            state_dir=tmp_path,
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: "garbage",
        )

        _phase2_triage_one("a.py", state, ctx)
        assert state.metrics["parse_failures"] == 1
        assert state.metrics["parse_failures_triage"] == 1
        assert state.metrics["parse_failures_finding"] == 0


class TestFormatAuditMetrics:
    """Phase 3 summary line rendering — aggregate plus parenthesized breakdown."""

    def _import(self):
        from swival.audit import _format_audit_metrics

        return _format_audit_metrics

    def test_no_metrics_returns_empty(self):
        f = self._import()
        assert f({}) == ""
        assert f({"parse_failures": 0, "repair_successes": 0}) == ""

    def test_aggregate_only_renders_without_breakdown(self):
        f = self._import()
        out = f({"parse_failures": 3})
        assert out == "3 parse failures"

    def test_breakdown_appears_in_parens(self):
        f = self._import()
        out = f(
            {
                "parse_failures": 5,
                "parse_failures_triage": 2,
                "parse_failures_finding": 3,
            }
        )
        assert out == "5 parse failures (2 triage, 3 finding)"

    def test_breakdown_skips_zero_typed_counters(self):
        f = self._import()
        out = f(
            {
                "parse_failures": 2,
                "parse_failures_triage": 0,
                "parse_failures_finding": 2,
                "parse_failures_expansion": 0,
            }
        )
        assert out == "2 parse failures (2 finding)"

    def test_other_metrics_appended_after_parse_failures(self):
        f = self._import()
        out = f(
            {
                "parse_failures": 1,
                "parse_failures_triage": 1,
                "repair_successes": 4,
                "multiline_continuations": 2,
            }
        )
        assert out == (
            "1 parse failures (1 triage), 4 repairs succeeded, "
            "2 multiline continuations"
        )

    def test_only_other_metrics_no_parse_failures(self):
        f = self._import()
        out = f({"repair_successes": 1, "analytical_retries": 2})
        assert out == "1 repairs succeeded, 2 analytical retries"


# ---------------------------------------------------------------------------
# State persistence and resume
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def _make_state(self, tmp_path: Path) -> AuditRunState:
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py", "b.py"],
            mandatory_files=["a.py", "b.py"],
            focus=[],
        )
        return AuditRunState(
            run_id="test-run",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py"},
            deep_reviewed_files={"a.py"},
            triage_records={
                "a.py": TriageRecord(
                    path="a.py",
                    priority="ESCALATE_HIGH",
                    confidence="high",
                    bug_classes=["command_execution"],
                    summary="dangerous exec call",
                    relevant_symbols=["run"],
                    suspicious_flows=["input->exec"],
                    needs_followup=True,
                )
            },
            proposed_findings=[
                FindingRecord(
                    title="Command injection",
                    finding_type="vulnerability",
                    severity="critical",
                    locations=["a.py:10"],
                    preconditions=["user input reaches exec"],
                    proof=["1. user input", "2. flows to exec"],
                    fix_outline="sanitize input",
                    source_file="a.py",
                )
            ],
            state_dir=tmp_path / ".swival" / "audit",
            next_index=1,
            phase="verification",
        )

    def test_save_and_load(self, tmp_path):
        state = self._make_state(tmp_path)
        state.save()

        loaded = AuditRunState.load(state.state_dir, "test-run")
        assert loaded.run_id == "test-run"
        assert loaded.scope.commit == "abc123"
        assert "a.py" in loaded.reviewed_files
        assert "a.py" in loaded.triage_records
        assert loaded.triage_records["a.py"].priority == "ESCALATE_HIGH"
        assert "a.py" in loaded.deep_reviewed_files
        assert len(loaded.proposed_findings) == 1
        assert loaded.proposed_findings[0].title == "Command injection"
        assert loaded.phase == "verification"

    def test_resume_matches_commit_and_focus(self, tmp_path):
        state = self._make_state(tmp_path)
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", None)
        assert found is not None
        assert found.run_id == "test-run"

        not_found = AuditRunState.find_resumable(state.state_dir, "different", None)
        assert not_found is None

        not_found = AuditRunState.find_resumable(state.state_dir, "abc123", ["src/"])
        assert not_found is None

    def test_resume_without_focus_matches_focused_run(self, tmp_path):
        state = self._make_state(tmp_path)
        scope = state.scope
        state.scope = AuditScope(
            branch=scope.branch,
            commit=scope.commit,
            tracked_files=scope.tracked_files,
            mandatory_files=scope.mandatory_files,
            focus=["subdir/"],
        )
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", None)
        assert found is not None
        assert found.run_id == "test-run"

        found = AuditRunState.find_resumable(state.state_dir, "abc123", ["subdir/"])
        assert found is not None

        not_found = AuditRunState.find_resumable(state.state_dir, "abc123", ["other/"])
        assert not_found is None

    def test_done_state_not_resumable(self, tmp_path):
        state = self._make_state(tmp_path)
        state.phase = "done"
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", None)
        assert found is None

    def test_incomplete_coverage_blocks_no_findings_message(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py", "b.py"],
            mandatory_files=["a.py", "b.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="x",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py"},  # b.py not reviewed
            state_dir=tmp_path,
        )
        unreviewed = [
            f for f in state.scope.mandatory_files if f not in state.reviewed_files
        ]
        assert len(unreviewed) == 1
        assert "b.py" in unreviewed

    def test_incomplete_deep_review_blocks_completion(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py", "b.py"],
            mandatory_files=["a.py", "b.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="x",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py", "b.py"},
            candidate_files=["a.py", "b.py"],
            deep_reviewed_files={"a.py"},
            state_dir=tmp_path,
        )
        undeep_reviewed = [
            f for f in state.candidate_files if f not in state.deep_reviewed_files
        ]
        assert len(undeep_reviewed) == 1
        assert "b.py" in undeep_reviewed


# ---------------------------------------------------------------------------
# Verification gates
# ---------------------------------------------------------------------------


class TestDeepReviewRecovery:
    def test_deep_review_repairs_malformed_inventory_records(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="x",
            scope=scope,
            queued_files=["a.py"],
            triage_records={
                "a.py": TriageRecord(
                    path="a.py",
                    priority="ESCALATE_HIGH",
                    confidence="high",
                    bug_classes=["unsafe_data_flow"],
                    summary="x",
                    relevant_symbols=[],
                    suspicious_flows=[],
                    needs_followup=True,
                )
            },
            repo_profile={"summary": "tiny repo"},
            import_index={},
            caller_index={},
            state_dir=tmp_path,
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr(
            "swival.audit._git_show", lambda path, base_dir: "print('x')"
        )

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "@@ finding @@\ntitle: incomplete\n"
            return "@@ none @@"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []
        assert calls["n"] == 2
        assert state.metrics["parse_failures"] == 1
        assert state.metrics["repair_successes"] == 1


class TestVerificationGates:
    def _make_state(self, tmp_path: Path) -> AuditRunState:
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["main.c"],
            mandatory_files=["main.c"],
            focus=[],
        )
        return AuditRunState(
            run_id="verify-run",
            scope=scope,
            queued_files=["main.c"],
            reviewed_files={"main.c"},
            state_dir=tmp_path,
        )

    def _make_finding(self, **overrides) -> FindingRecord:
        finding = FindingRecord(
            title="Fixed-size stack buffer can be overflowed by argv data and suffix append",
            finding_type="vulnerability",
            severity="high",
            locations=["main.c:7"],
            preconditions=["program receives a command-line argument"],
            proof=[
                "argv-controlled data reaches unsafe string operations",
                "the bug is demonstrable with a small proof of concept",
            ],
            fix_outline="Use bounded copies and validate argument presence before use.",
            source_file="main.c",
        )
        for key, value in overrides.items():
            setattr(finding, key, value)
        return finding

    def test_no_reproduction_discards(self, monkeypatch, tmp_path):
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        monkeypatch.setattr(
            "swival.audit._phase4c_reproduce",
            lambda finding, state, ctx, work_dir: None,
        )

        verified = _verify_single_finding(
            finding, state, ctx=None, work_dir=tmp_path / "work"
        )
        assert verified is None

    def test_reproduced_finding_is_verified(self, monkeypatch, tmp_path):
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        monkeypatch.setattr(
            "swival.audit._phase4c_reproduce",
            lambda finding, state, ctx, work_dir: {
                "reproduced": True,
                "summary": "crash observed\nREPRODUCED",
            },
        )

        verified = _verify_single_finding(
            finding, state, ctx=None, work_dir=tmp_path / "work"
        )
        assert verified is not None
        assert verified.finding.title == finding.title
        assert (
            verified.correctness_reason == "verified by proof-of-concept reproduction"
        )
        assert verified.rebuttal_reason == "not used; PoC verifier is authoritative"
        assert verified.reproducer == {
            "reproduced": True,
            "summary": "crash observed\nREPRODUCED",
        }

    def test_phase4_verifier_uses_fallback_max_turns(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival import audit

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        captured = {}

        class DummyWorktree:
            def __init__(self, work_dir):
                self.work_dir = work_dir

            def __enter__(self):
                return self.work_dir

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            "swival.audit._gather_evidence",
            lambda finding, ctx: ("--- main.c ---\ncode", 1),
        )
        monkeypatch.setattr(
            "swival.audit._worktree", lambda base_dir, work_dir: DummyWorktree(work_dir)
        )

        def fake_run(messages, tools, **kw):
            captured.update(kw)
            return "proof\nREPRODUCED", False

        monkeypatch.setattr("swival.agent.run_agent_loop", fake_run)

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            loop_kwargs={
                "api_base": "x",
                "model_id": "m",
                "max_output_tokens": 100,
                "temperature": 0.0,
                "top_p": None,
                "seed": None,
                "context_length": None,
                "resolved_commands": {},
                "llm_kwargs": {},
            },
        )
        work_dir = (
            tmp_path / ".swival" / "audit" / state.run_id / "verify" / "0" / "work"
        )
        result = audit._phase4c_reproduce(finding, state, ctx, work_dir)
        assert result is not None
        assert captured["max_turns"] == 100


# ---------------------------------------------------------------------------
# Artifact naming
# ---------------------------------------------------------------------------


class TestPromptSemantics:
    def test_phase3a_prefers_narrow_directly_proven_bug(self):
        from swival.audit import _PHASE3A_SYSTEM

        assert (
            "Prefer the narrowest bug that the evidence directly proves."
            in _PHASE3A_SYSTEM
        )
        assert "undefined behavior or uninitialized-state bugs" in _PHASE3A_SYSTEM

    def test_phase3b_expansion_prompt_exists(self):
        from swival.audit import _PHASE3B_SYSTEM

        assert "expanding one security finding" in _PHASE3B_SYSTEM.lower()

    def test_phase4_verifier_allows_source_or_runtime_proof(self):
        from swival.audit import _PHASE4_VERIFY_SYSTEM

        assert (
            "you may compile/run small proof-of-concept code"
            in _PHASE4_VERIFY_SYSTEM.lower()
        )
        assert "or demonstrate equivalent runtime evidence" in _PHASE4_VERIFY_SYSTEM
        assert "narrower directly source-grounded local bug" in _PHASE4_VERIFY_SYSTEM
        assert "NOTREPRODUCED" in _PHASE4_VERIFY_SYSTEM


class TestArtifacts:
    def test_slug_generation(self):
        assert (
            _make_slug("Command Injection in Parser") == "command-injection-in-parser"
        )
        assert _make_slug("SQL   injection!!") == "sql-injection"
        assert _make_slug("") == "finding"

    def test_sequential_numbering(self):
        """Artifact numbers should be sequential 001, 002, ..."""
        for i, expected in [(1, "001"), (2, "002"), (10, "010")]:
            assert f"{i:03d}" == expected

    def test_no_findings_exact_message(self):
        expected = "No provable security bugs found in Git-tracked files."
        assert expected == "No provable security bugs found in Git-tracked files."

    def test_report_provenance_url(self):
        assert AUDIT_PROVENANCE_URL == "https://swival.dev"


# ---------------------------------------------------------------------------
# Triage ordering
# ---------------------------------------------------------------------------


class TestTriageOrdering:
    def test_triage_prompt_ends_with_file_path(self):
        """The triage prompt variable suffix must end with 'The file is: <path>'."""
        from swival.audit import _PHASE2_SYSTEM

        assert "The file is:" not in _PHASE2_SYSTEM

    def test_deep_review_includes_bug_classes(self):
        """Phase 3a bug classes are passed via user message, not system prompt."""
        from swival.audit import _PHASE3A_SYSTEM

        assert "bug classes" not in _PHASE3A_SYSTEM.lower()


class TestMessageLayout:
    """Verify that variable data lands in user messages (not system) and
    that the ordering within user messages is cache-friendly."""

    def _make_state(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        return AuditRunState(
            run_id="x",
            scope=scope,
            queued_files=["a.py"],
            triage_records={
                "a.py": TriageRecord(
                    path="a.py",
                    priority="ESCALATE_HIGH",
                    confidence="high",
                    bug_classes=["unsafe_data_flow", "injection"],
                    summary="x",
                    relevant_symbols=[],
                    suspicious_flows=[],
                    needs_followup=True,
                )
            },
            repo_profile={"summary": "tiny repo", "languages": ["Python"]},
            import_index={},
            caller_index={},
            state_dir=tmp_path,
        )

    def test_phase2_repo_profile_in_user_not_system(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _phase2_triage_one, _PHASE2_SYSTEM

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        captured = {}

        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")

        def fake_call(ctx, messages, temperature=None, trace_task=None):
            captured["messages"] = messages
            return (
                "@@ triage @@\n"
                "priority: SKIP\n"
                "confidence: high\n"
                "summary: ok\n"
                "needs_followup: false\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        _phase2_triage_one("a.py", state, ctx)

        system_content = captured["messages"][0]["content"]
        user_content = captured["messages"][1]["content"]
        assert system_content == _PHASE2_SYSTEM
        assert "Repository profile:" in user_content
        assert "tiny repo" in user_content

    def test_phase3a_bug_classes_in_user_not_system(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _phase3a_inventory, _PHASE3A_SYSTEM

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        captured = {}

        def fake_call(ctx, messages, temperature=None, trace_task=None):
            captured["messages"] = messages
            return "@@ none @@"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        _phase3a_inventory("a.py", state, ctx, "x = 1")

        system_content = captured["messages"][0]["content"]
        user_content = captured["messages"][1]["content"]
        assert system_content == _PHASE3A_SYSTEM
        assert "unsafe_data_flow" in user_content
        assert "injection" in user_content

    def test_phase3b_evidence_before_finding_metadata(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _phase3b_expand_one, _PHASE3B_SYSTEM

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        captured = {}

        def fake_call(ctx, messages, temperature=None, trace_task=None):
            captured["messages"] = messages
            return (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: request body reaches eval\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: none\n"
                "proof: direct\n"
                "fix_outline: fix it\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        stub = {
            "title": "eval injection",
            "severity": "high",
            "location": "a.py:1",
            "attacker": "remote client",
            "trigger": "request body reaches eval",
            "impact": "arbitrary code execution as server user",
            "claim": "user input reaches eval",
        }
        _phase3b_expand_one((stub, "a.py", "eval(input())", state, ctx))

        system_content = captured["messages"][0]["content"]
        user_content = captured["messages"][1]["content"]
        assert system_content == _PHASE3B_SYSTEM
        evidence_pos = user_content.index("Committed evidence")
        finding_pos = user_content.index("Finding to expand:")
        assert evidence_pos < finding_pos, (
            "evidence must come before finding metadata for prefix caching"
        )
        assert "eval injection" in user_content
        assert "user input reaches eval" in user_content


# ---------------------------------------------------------------------------
# Scope serialization round-trip
# ---------------------------------------------------------------------------


class TestScopeRoundTrip:
    def test_scope_to_dict_and_back(self):
        scope = AuditScope(
            branch="main",
            commit="abc",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=["src"],
        )
        d = scope.to_dict()
        restored = AuditScope.from_dict(d)
        assert restored == scope

    def test_scope_frozen(self):
        scope = AuditScope(
            branch="main",
            commit="abc",
            tracked_files=[],
            mandatory_files=[],
            focus=[],
        )
        with pytest.raises(AttributeError):
            scope.branch = "other"


# ---------------------------------------------------------------------------
# Phase 4 parallelism
# ---------------------------------------------------------------------------


class TestPhase4Parallelism:
    def _make_scope(self):
        return AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["main.c"],
            mandatory_files=["main.c"],
            focus=[],
        )

    def _make_state(self, tmp_path):
        return AuditRunState(
            run_id="p4-run",
            scope=self._make_scope(),
            queued_files=["main.c"],
            state_dir=tmp_path / ".swival" / "audit",
        )

    def _make_finding(self, title="Bug", source_file="main.c"):
        return FindingRecord(
            title=title,
            finding_type="vulnerability",
            severity="high",
            locations=["main.c:1"],
            preconditions=["none"],
            proof=["step 1"],
            fix_outline="fix it",
            source_file=source_file,
        )

    def test_verification_result_verified(self):
        f = self._make_finding()
        vf = VerifiedFinding(finding=f, correctness_reason="ok", rebuttal_reason="n/a")
        r = VerificationResult(finding_key="0", verified_finding=vf)
        assert r.verified_finding is not None
        assert not r.discarded
        assert r.error is None

    def test_verification_result_discarded(self):
        r = VerificationResult(finding_key="0", discarded=True)
        assert r.discarded
        assert r.verified_finding is None
        assert r.error is None

    def test_verification_result_error(self):
        r = VerificationResult(finding_key="0", error="provider timeout")
        assert r.error == "provider timeout"
        assert not r.discarded
        assert r.verified_finding is None

    def test_verify_one_finding_verified(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        vf = VerifiedFinding(
            finding=finding,
            correctness_reason="ok",
            rebuttal_reason="n/a",
            reproducer={"reproduced": True, "summary": "ok"},
        )

        monkeypatch.setattr(
            "swival.audit._verify_single_finding",
            lambda f, s, c, work_dir: vf,
        )

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.finding_key == key
        assert result.verified_finding is vf
        assert not result.discarded
        assert result.error is None

    def test_verify_one_finding_discarded(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)

        monkeypatch.setattr(
            "swival.audit._verify_single_finding",
            lambda f, s, c, work_dir: None,
        )

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.finding_key == key
        assert result.discarded
        assert result.verified_finding is None

    def test_verify_one_finding_retries_transient_error(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        vf = VerifiedFinding(
            finding=finding, correctness_reason="ok", rebuttal_reason="n/a"
        )
        calls = {"n": 0}

        def mock_verify(f, s, c, work_dir):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _TransientVerifierError("provider timeout")
            return vf

        monkeypatch.setattr("swival.audit._verify_single_finding", mock_verify)

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.verified_finding is vf
        assert calls["n"] == 2

    def test_verify_one_finding_no_retry_on_runtime_error(self, monkeypatch, tmp_path):
        """Non-transient RuntimeError (e.g. worktree failure) must not be retried."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        calls = {"n": 0}

        def mock_verify(f, s, c, work_dir):
            calls["n"] += 1
            raise RuntimeError("worktree add failed")

        monkeypatch.setattr("swival.audit._verify_single_finding", mock_verify)

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.error == "worktree add failed"
        assert calls["n"] == 1

    def _loop_kwargs(self):
        return {
            "api_base": "x",
            "model_id": "m",
            "max_output_tokens": 100,
            "temperature": 0.0,
            "top_p": None,
            "seed": None,
            "context_length": None,
            "resolved_commands": {},
            "llm_kwargs": {},
        }

    def test_worktree_failure_is_error_not_discard(self, monkeypatch, tmp_path):
        """Worktree setup crash must propagate as 'failed', not 'discarded'."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)

        class FailingWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                raise RuntimeError("worktree add failed")

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", FailingWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.error is not None
        assert not result.discarded

    def test_worktree_failure_not_retried(self, monkeypatch, tmp_path):
        """Worktree failure is deterministic and must not trigger a retry."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        calls = {"n": 0}

        class FailingWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                calls["n"] += 1
                raise RuntimeError("worktree add failed")

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", FailingWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        _verify_one_finding((key, finding), state, ctx)
        assert calls["n"] == 1

    def test_agent_loop_crash_is_error_not_discard(self, monkeypatch, tmp_path):
        """Agent loop crash must propagate as 'failed', not 'discarded'."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)

        class DummyWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                return tmp_path / "wt"

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", DummyWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )

        def crash_loop(msgs, tools, **kw):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr("swival.agent.run_agent_loop", crash_loop)

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.error is not None
        assert not result.discarded

    def test_agent_loop_transport_error_is_retried(self, monkeypatch, tmp_path):
        """Transport errors (ConnectionError etc.) get one retry."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        calls = {"n": 0}

        class DummyWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                return tmp_path / "wt"

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", DummyWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )

        def crash_loop(msgs, tools, **kw):
            calls["n"] += 1
            raise ConnectionError("network unreachable")

        monkeypatch.setattr("swival.agent.run_agent_loop", crash_loop)

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.error is not None
        assert calls["n"] == 2  # original + one retry

    def test_agent_loop_logic_error_not_retried(self, monkeypatch, tmp_path):
        """Non-transport agent loop errors must not be retried."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        calls = {"n": 0}

        class DummyWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                return tmp_path / "wt"

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", DummyWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )

        def crash_loop(msgs, tools, **kw):
            calls["n"] += 1
            raise RuntimeError("context overflow")

        monkeypatch.setattr("swival.agent.run_agent_loop", crash_loop)

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.error is not None
        assert calls["n"] == 1  # no retry

    def test_notreproduced_is_discard_not_error(self, monkeypatch, tmp_path):
        """Legitimate NOTREPRODUCED must be 'discarded', not 'error'."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)

        class DummyWorktree:
            def __init__(self, base_dir, work_dir):
                pass

            def __enter__(self):
                return tmp_path / "wt"

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", DummyWorktree)
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("evidence", 1)
        )
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: ("could not confirm\nNOTREPRODUCED", False),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path), tools=[], loop_kwargs=self._loop_kwargs()
        )
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.discarded
        assert result.error is None

    def test_stale_running_reset_to_pending(self, tmp_path):
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        state.proposed_findings = [finding]
        state.verification_state = {
            key: {
                "status": "running",
                "attempts": 1,
                "last_error": None,
                "summary": None,
            },
        }
        for vs in state.verification_state.values():
            if vs["status"] == "running":
                vs["status"] = "pending"
        assert state.verification_state[key]["status"] == "pending"

    def test_resume_only_requeues_non_terminal(self, tmp_path):
        state = self._make_state(tmp_path)
        findings = [
            self._make_finding(title="A"),
            self._make_finding(title="B"),
            self._make_finding(title="C"),
        ]
        keys = [_finding_key(f) for f in findings]
        state.proposed_findings = findings
        state.verification_state = {
            keys[0]: {
                "status": "verified",
                "attempts": 1,
                "last_error": None,
                "summary": None,
            },
            keys[1]: {
                "status": "discarded",
                "attempts": 1,
                "last_error": None,
                "summary": None,
            },
            keys[2]: {
                "status": "failed",
                "attempts": 1,
                "last_error": "timeout",
                "summary": None,
            },
        }
        pending = []
        for f in state.proposed_findings:
            k = _finding_key(f)
            if state.verification_state[k]["status"] in ("pending", "failed"):
                pending.append((k, f))
        assert len(pending) == 1
        assert pending[0][0] == keys[2]

    def test_unique_worktree_paths(self, tmp_path):
        state = self._make_state(tmp_path)
        findings = [
            self._make_finding(title="A"),
            self._make_finding(title="B"),
            self._make_finding(title="C"),
        ]
        paths = set()
        for f in findings:
            key = _finding_key(f)
            work_dir = (
                tmp_path / state.state_dir / state.run_id / "verify" / key / "work"
            )
            paths.add(str(work_dir))
        assert len(paths) == 3

    def test_finding_key_is_content_stable(self):
        """Key must be the same for identical findings regardless of list position."""
        f1 = self._make_finding(title="Bug A")
        f2 = self._make_finding(title="Bug A")
        assert _finding_key(f1) == _finding_key(f2)

        f3 = self._make_finding(title="Bug B")
        assert _finding_key(f1) != _finding_key(f3)

    def test_incomplete_verification_blocks_artifacts(self, tmp_path):
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        state.proposed_findings = [finding]
        state.verification_state = {
            key: {
                "status": "failed",
                "attempts": 1,
                "last_error": "err",
                "summary": None,
            },
        }
        non_terminal = [
            k
            for k, vs in state.verification_state.items()
            if vs["status"] not in ("verified", "discarded")
        ]
        assert len(non_terminal) == 1

    def test_all_failed_produces_incomplete(self, tmp_path):
        state = self._make_state(tmp_path)
        findings = [
            self._make_finding(title="A"),
            self._make_finding(title="B"),
        ]
        keys = [_finding_key(f) for f in findings]
        state.proposed_findings = findings
        state.verification_state = {
            keys[0]: {
                "status": "failed",
                "attempts": 1,
                "last_error": "err",
                "summary": None,
            },
            keys[1]: {
                "status": "failed",
                "attempts": 1,
                "last_error": "err",
                "summary": None,
            },
        }
        non_terminal = [
            k
            for k, vs in state.verification_state.items()
            if vs["status"] not in ("verified", "discarded")
        ]
        n_failed = sum(
            1 for k in non_terminal if state.verification_state[k]["status"] == "failed"
        )
        assert len(non_terminal) == 2
        assert n_failed == 2

    def test_verification_state_persists(self, tmp_path):
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        state.proposed_findings = [finding]
        state.verification_state = {
            key: {
                "status": "verified",
                "attempts": 1,
                "last_error": None,
                "summary": "ok",
            },
        }
        state.save()

        loaded = AuditRunState.load(state.state_dir, "p4-run")
        assert loaded.verification_state == state.verification_state

    def test_duplicate_findings_deduplicated(self, tmp_path):
        """Identical findings from phase 3 must collapse to one verification slot."""
        f1 = self._make_finding(title="Same Bug")
        f2 = self._make_finding(title="Same Bug")
        assert _finding_key(f1) == _finding_key(f2)

        seen_keys: set[str] = set()
        deduped = []
        for f in [f1, f2]:
            key = _finding_key(f)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(f)
        assert len(deduped) == 1

    def test_stale_numeric_keys_pruned(self, tmp_path):
        """Old numeric keys from a previous key scheme must not block the final gate."""
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        state.proposed_findings = [finding]
        state.verification_state = {
            "0": {
                "status": "failed",
                "attempts": 1,
                "last_error": "old",
                "summary": None,
            },
        }

        current_keys = {_finding_key(f) for f in state.proposed_findings}
        stale = [k for k in state.verification_state if k not in current_keys]
        for k in stale:
            del state.verification_state[k]

        assert "0" not in state.verification_state
        assert len(state.verification_state) == 0

    def test_migration_reconciles_verified_findings(self, tmp_path):
        """Findings already in verified_findings must not be re-queued after migration."""
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        vf = VerifiedFinding(
            finding=finding, correctness_reason="ok", rebuttal_reason="n/a"
        )
        state.proposed_findings = [finding]
        state.verified_findings = [vf]
        # Old numeric key gets pruned, but finding is already verified
        state.verification_state = {
            "0": {
                "status": "verified",
                "attempts": 1,
                "last_error": None,
                "summary": None,
            },
        }

        # Simulate phase 4 entry: prune + reconcile
        current_keys = {_finding_key(f) for f in state.proposed_findings}
        stale = [k for k in state.verification_state if k not in current_keys]
        for k in stale:
            del state.verification_state[k]

        already_verified_keys = {
            _finding_key(vf.finding) for vf in state.verified_findings
        }
        for f in state.proposed_findings:
            k = _finding_key(f)
            if k not in state.verification_state:
                state.verification_state[k] = {
                    "status": "verified" if k in already_verified_keys else "pending",
                    "attempts": 0,
                    "last_error": None,
                    "summary": None,
                }

        assert key in state.verification_state
        assert state.verification_state[key]["status"] == "verified"

    def test_attempts_counts_retries(self, monkeypatch, tmp_path):
        """VerificationResult.attempts must reflect actual tries including retries."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)
        calls = {"n": 0}
        vf = VerifiedFinding(
            finding=finding, correctness_reason="ok", rebuttal_reason="n/a"
        )

        def mock_verify(f, s, c, work_dir):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _TransientVerifierError("timeout")
            return vf

        monkeypatch.setattr("swival.audit._verify_single_finding", mock_verify)

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.attempts == 2
        assert result.verified_finding is vf

    def test_attempts_one_on_first_success(self, monkeypatch, tmp_path):
        """Single successful verification must report attempts=1."""
        from types import SimpleNamespace

        state = self._make_state(tmp_path)
        finding = self._make_finding()
        key = _finding_key(finding)

        monkeypatch.setattr(
            "swival.audit._verify_single_finding",
            lambda f, s, c, work_dir: None,
        )

        ctx = SimpleNamespace(base_dir=str(tmp_path))
        result = _verify_one_finding((key, finding), state, ctx)
        assert result.attempts == 1
        assert result.discarded

    def test_verified_findings_deduplicated_before_artifacts(self, tmp_path):
        """Duplicate verified_findings from pre-migration state must not produce duplicate artifacts."""
        state = self._make_state(tmp_path)
        finding = self._make_finding()
        vf = VerifiedFinding(
            finding=finding, correctness_reason="ok", rebuttal_reason="n/a"
        )
        state.verified_findings = [vf, vf]

        seen_vf_keys: set[str] = set()
        deduped_vf = []
        for v in state.verified_findings:
            vf_key = _finding_key(v.finding)
            if vf_key not in seen_vf_keys:
                seen_vf_keys.add(vf_key)
                deduped_vf.append(v)
        state.verified_findings = deduped_vf

        assert len(state.verified_findings) == 1


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalization:
    def test_basic_canonicalization(self):
        inventory = {
            "title": "Buffer overflow",
            "severity": "HIGH",
            "location": "main.c:17",
            "claim": "strcpy overflows stack buffer",
        }
        expansion = {
            "type": "code execution",
            "attacker": "local user",
            "trigger": "argv[1] reaches strcpy",
            "impact": "arbitrary code execution",
            "preconditions": "attacker controls argv[1]",
            "proof": "input reaches strcpy without bounds check",
            "fix_outline": "use strncpy with bounds",
        }
        f = _canonicalize_finding(inventory, expansion, "main.c")
        assert f.title == "Buffer overflow"
        assert f.finding_type == "code execution"
        assert f.severity == "high"
        assert f.locations == ["main.c:17"]
        assert f.preconditions == ["attacker controls argv[1]"]
        assert f.proof == [
            "attacker: local user trigger: argv[1] reaches strcpy "
            "impact: arbitrary code execution input reaches strcpy without bounds check"
        ]
        assert f.fix_outline == "use strncpy with bounds"
        assert f.source_file == "main.c"

    def test_invalid_severity_defaults_to_low(self):
        inventory = {"severity": "EXTREME"}
        expansion = {"type": "unknown"}
        f = _canonicalize_finding(inventory, expansion, "x.py")
        assert f.severity == "low"

    def test_missing_severity_defaults_to_low(self):
        inventory = {}
        expansion = {"type": "unknown"}
        f = _canonicalize_finding(inventory, expansion, "x.py")
        assert f.severity == "low"

    def test_empty_preconditions_and_proof(self):
        inventory = {"location": "a.py:1"}
        expansion = {"type": "bug", "preconditions": "", "proof": ""}
        f = _canonicalize_finding(inventory, expansion, "a.py")
        assert f.preconditions == []
        assert f.proof == []


# ---------------------------------------------------------------------------
# Phase 3 inventory + expansion
# ---------------------------------------------------------------------------


class TestPhase3Split:
    def _make_state(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        return AuditRunState(
            run_id="x",
            scope=scope,
            queued_files=["a.py"],
            triage_records={
                "a.py": TriageRecord(
                    path="a.py",
                    priority="ESCALATE_HIGH",
                    confidence="high",
                    bug_classes=["unsafe_data_flow"],
                    summary="x",
                    relevant_symbols=[],
                    suspicious_flows=[],
                    needs_followup=True,
                )
            },
            repo_profile={"summary": "tiny repo"},
            import_index={},
            caller_index={},
            state_dir=tmp_path,
        )

    def test_zero_findings_inventory(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "x = 1")
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: "@@ none @@",
        )

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []

    def test_inventory_plus_expansion_produces_finding(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr(
            "swival.audit._git_show", lambda path, base_dir: "eval(input())"
        )

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: eval injection\n"
                    "severity: high\n"
                    "location: a.py:1\n"
                    "attacker: remote client\n"
                    "trigger: request body reaches eval\n"
                    "impact: arbitrary code execution as server user\n"
                    "claim: user input reaches eval\n"
                )
            return (
                "@@ expansion @@\n"
                "type: code execution\n"
                "attacker: remote client\n"
                "trigger: request body reaches eval\n"
                "impact: arbitrary code execution as server user\n"
                "preconditions: user provides input\n"
                "proof: input flows to eval without sanitization\n"
                "fix_outline: remove eval\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f.title == "eval injection"
        assert f.finding_type == "code execution"
        assert f.severity == "high"
        assert f.locations == ["a.py:1"]
        assert f.source_file == "a.py"

    def test_all_expansions_fail_triggers_retry(self, monkeypatch, tmp_path):
        """When all expansion attempts fail, the file should not silently
        succeed with zero findings — it must trigger the analytical retry path."""
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        inventory_response = (
            "@@ finding @@\n"
            "title: bug A\n"
            "severity: high\n"
            "location: a.py:1\n"
            "attacker: remote client\n"
            "trigger: crafted request reaches bug A\n"
            "impact: denial of service\n"
            "claim: claim A\n"
        )

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                inventory_response
                if "phase 3" in (messages[0].get("content", "") or "").lower()
                else "totally broken output {{{"
            ),
        )

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is not None
        assert state.metrics["analytical_retries"] >= 1

    def test_partial_expansion_failure_keeps_successes(self, monkeypatch, tmp_path):
        """When some expansions succeed and some fail, keep the successful ones."""
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: bug A\n"
                    "severity: high\n"
                    "location: a.py:1\n"
                    "attacker: remote client\n"
                    "trigger: crafted request reaches bug A\n"
                    "impact: denial of service\n"
                    "claim: claim A\n"
                    "\n"
                    "@@ finding @@\n"
                    "title: bug B\n"
                    "severity: medium\n"
                    "location: a.py:2\n"
                    "attacker: remote client\n"
                    "trigger: crafted request reaches bug B\n"
                    "impact: denial of service\n"
                    "claim: claim B\n"
                )
            if calls["n"] == 2:
                return (
                    "@@ expansion @@\n"
                    "type: denial of service\n"
                    "attacker: remote client\n"
                    "trigger: crafted request reaches bug A\n"
                    "impact: denial of service\n"
                    "preconditions: none\n"
                    "proof: proven\n"
                    "fix_outline: fix\n"
                )
            return "broken {{{"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        assert result.findings[0].title == "bug A"

    def test_out_of_scope_expansion_is_discarded_without_retry(
        self, monkeypatch, tmp_path
    ):
        """A real non-security bug should be dropped, not treated as a failed
        expansion that drives analytical retries."""
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: teardown waiters are not woken\n"
                    "severity: medium\n"
                    "location: a.py:1\n"
                    "attacker: missing\n"
                    "trigger: shutdown path only\n"
                    "impact: no attacker-controlled security outcome\n"
                    "claim: shutdown can leave waiters asleep\n"
                )
            return (
                "@@ expansion @@\n"
                "type: out-of-scope\n"
                "attacker: missing\n"
                "trigger: missing\n"
                "impact: missing\n"
                "preconditions: out-of-scope\n"
                "proof: out-of-scope because only shutdown sequencing is shown\n"
                "fix_outline: no security fix\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []
        assert state.metrics["analytical_retries"] == 0

    def test_security_control_failure_is_accepted(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr(
            "swival.audit._git_show", lambda path, base_dir: "verify_sig(...)"
        )

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: signature verifier accepts forged signatures\n"
                    "severity: critical\n"
                    "location: a.py:42\n"
                    "attacker: any caller of the signature verifier\n"
                    "trigger: signature buffer whose final byte is zero\n"
                    "impact: Ed25519 signature verifier fails open\n"
                    "claim: early return short-circuits constant-time compare\n"
                )
            return (
                "@@ expansion @@\n"
                "type: security_control_failure\n"
                "attacker: any caller of the signature verifier\n"
                "trigger: signature buffer whose final byte is zero\n"
                "impact: Ed25519 signature verifier fails open: forged sigs accepted\n"
                "preconditions: caller invokes verify_sig with a 64-byte buffer\n"
                "proof: verify_sig is the Ed25519 signature decision point and "
                "returns accept on sig[63]==0 short-circuit\n"
                "fix_outline: remove early return and complete the compare\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f.finding_type == "security_control_failure"
        assert f.severity == "critical"

    def test_security_control_failure_low_severity_is_dropped(
        self, monkeypatch, tmp_path
    ):
        # Regression guard: SCF must not be usable to smuggle medium-severity
        # generic logic bugs back into the audit output.
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: parser accepts trailing garbage\n"
                    "severity: medium\n"
                    "location: a.py:9\n"
                    "attacker: any caller of the parser\n"
                    "trigger: input with trailing bytes after the structure\n"
                    "impact: parser accepts malformed input\n"
                    "claim: bounds check is off by one\n"
                )
            return (
                "@@ expansion @@\n"
                "type: security_control_failure\n"
                "attacker: any caller of the parser\n"
                "trigger: input with trailing bytes\n"
                "impact: parser fails open: trailing bytes accepted\n"
                "preconditions: caller passes attacker-shaped input\n"
                "proof: bounds check at line 9 admits one extra byte\n"
                "fix_outline: tighten the bounds check\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []
        assert state.metrics["analytical_retries"] == 0

    def test_helper_contract_violation_stays_out_of_scope(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    "@@ finding @@\n"
                    "title: helper returns success when it should not\n"
                    "severity: medium\n"
                    "location: a.py:5\n"
                    "attacker: missing\n"
                    "trigger: internal call from a sibling module\n"
                    "impact: helper contract violated, no attacker gain proven\n"
                    "claim: helper returns 0 on a path that should return -1\n"
                )
            return (
                "@@ expansion @@\n"
                "type: out-of-scope\n"
                "attacker: missing\n"
                "trigger: missing\n"
                "impact: missing\n"
                "preconditions: out-of-scope\n"
                "proof: out-of-scope because the helper is not itself a "
                "named security control and no attacker gain is proven\n"
                "fix_outline: no security fix\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []
        assert state.metrics["analytical_retries"] == 0

    def test_analytical_retry_on_inventory_failure(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        calls = {"n": 0}

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                return "@@ finding @@\ntitle: bad\n"
            return "@@ none @@"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert result.findings == []
        assert state.metrics["analytical_retries"] == 1

    def test_both_attempts_fail_returns_error(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        state = self._make_state(tmp_path)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        monkeypatch.setattr("swival.audit._git_show", lambda path, base_dir: "code")
        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                "@@ finding @@\ntitle: incomplete\n"
            ),
        )

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is not None

    def test_metrics_persist_in_state(self, tmp_path):
        state = self._make_state(tmp_path)
        state.metrics["parse_failures"] = 3
        state.metrics["repair_successes"] = 2
        state.save()

        loaded = AuditRunState.load(state.state_dir, "x")
        assert loaded.metrics["parse_failures"] == 3
        assert loaded.metrics["repair_successes"] == 2


# ---------------------------------------------------------------------------
# Auto-retry and resumability
# ---------------------------------------------------------------------------


class TestAutoRetry:
    """Tests for automatic retry loops in phases 2, 3, and 4, and the
    done-but-incomplete resumability fix."""

    def _make_scope(self, commit="abc123", files=None):
        files = files or ["a.py"]
        return AuditScope(
            branch="main",
            commit=commit,
            tracked_files=files,
            mandatory_files=files,
            focus=[],
        )

    def _make_finding(self, title="Bug", source_file="a.py"):
        return FindingRecord(
            title=title,
            finding_type="vulnerability",
            severity="high",
            locations=[f"{source_file}:1"],
            preconditions=["none"],
            proof=["step 1"],
            fix_outline="fix it",
            source_file=source_file,
        )

    @staticmethod
    def _triage_escalate(path):
        return TriageRecord(
            path=path,
            priority="ESCALATE_HIGH",
            confidence="high",
            bug_classes=["eval"],
            summary="dangerous",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=True,
        )

    @staticmethod
    def _triage_skip(path):
        return TriageRecord(
            path=path,
            priority="SKIP",
            confidence="high",
            bug_classes=[],
            summary="ok",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=False,
        )

    # -- Phase 2: triage retry -----------------------------------------------

    def test_triage_retries_failed_files(self, monkeypatch, tmp_path):
        """Files that return None from the triage worker are retried."""
        from types import SimpleNamespace

        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "import os")
        _commit_file(tmp_path, "b.py", "import sys")

        calls = {"n": 0}

        def fake_triage_one(path, state, ctx):
            calls["n"] += 1
            if path == "b.py" and calls["n"] <= 2:
                return None
            return self._triage_skip(path)

        monkeypatch.setattr("swival.audit._phase2_triage_one", fake_triage_one)
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("", ctx)
        assert "Audit incomplete" not in result or "not reviewed" not in result
        # b.py should have eventually been reviewed via retry
        assert calls["n"] >= 3

    # -- Phase 3: deep-review retry -------------------------------------------

    def test_deep_review_retries_failed_files(self, monkeypatch, tmp_path):
        """Files that return an error from deep review are retried."""
        from types import SimpleNamespace
        from swival.audit import DeepReviewResult, run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "eval(input())")

        calls = {"n": 0}

        def fake_deep_review(path, state, ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                return DeepReviewResult(path=path, error="transient failure")
            return DeepReviewResult(path=path, findings=[])

        monkeypatch.setattr("swival.audit._deep_review_one", fake_deep_review)
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )
        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_escalate(path),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("", ctx)
        assert "failed deep review" not in result
        assert calls["n"] >= 2

    def test_deep_review_exhausted_retries_returns_incomplete(
        self, monkeypatch, tmp_path
    ):
        """When deep review always fails, the result says incomplete and state
        stays at deep_review (not done)."""
        from types import SimpleNamespace
        from swival.audit import DeepReviewResult, run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "eval(input())")

        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, error="always fails"),
        )
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )
        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_escalate(path),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("", ctx)
        assert "failed deep review after retries" in result

        # State should stay at deep_review, not done
        state_dir = Path(tmp_path) / ".swival" / "audit"
        import json

        for entry in state_dir.iterdir():
            sf = entry / "state.json"
            if sf.exists():
                blob = json.loads(sf.read_text())
                assert blob["phase"] == "deep_review"

    # -- Phase 4: verification retry ------------------------------------------

    def test_verification_retries_failed_findings(self, monkeypatch, tmp_path):
        """Failed verifier findings are retried within the same run."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "eval(input())")
        finding = self._make_finding()
        calls = {"n": 0}

        def fake_verify(item, state, ctx):
            calls["n"] += 1
            _key, _finding = item
            if calls["n"] == 1:
                return VerificationResult(
                    finding_key=_key, error="provider timeout", attempts=1
                )
            vf = VerifiedFinding(
                finding=_finding,
                correctness_reason="ok",
                rebuttal_reason="n/a",
                reproducer={"reproduced": True, "summary": "ok"},
            )
            return VerificationResult(finding_key=_key, verified_finding=vf, attempts=1)

        monkeypatch.setattr("swival.audit._verify_one_finding", fake_verify)
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )
        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_escalate(path),
        )
        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, findings=[finding]),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state: "--- patch ---",
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# Report",
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("", ctx)
        assert "Audit incomplete" not in result
        assert calls["n"] >= 2

    def test_verification_exhausted_retries_returns_incomplete(
        self, monkeypatch, tmp_path
    ):
        """When verification always fails, the result mentions attempt count."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "eval(input())")
        finding = self._make_finding()

        monkeypatch.setattr(
            "swival.audit._verify_one_finding",
            lambda item, state, ctx: VerificationResult(
                finding_key=item[0], error="always fails", attempts=1
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )
        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_escalate(path),
        )
        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, findings=[finding]),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("", ctx)
        assert "after 3 attempts" in result
        assert "Use /audit --resume to retry" in result

    def test_verification_attempts_additive_across_retries(self, monkeypatch, tmp_path):
        """verification_state attempts must accumulate across outer retry
        iterations, including inner retry counts from _verify_one_finding."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "eval(input())")
        finding = self._make_finding()

        monkeypatch.setattr(
            "swival.audit._verify_one_finding",
            lambda item, state, ctx: VerificationResult(
                finding_key=item[0], error="fail", attempts=2
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "test"},
        )
        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_escalate(path),
        )
        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, findings=[finding]),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        run_audit_command("", ctx)

        # Load the state and check that attempts accumulated: 3 outer rounds × 2 inner = 6
        import json

        state_dir = Path(tmp_path) / ".swival" / "audit"
        for entry in state_dir.iterdir():
            sf = entry / "state.json"
            if sf.exists():
                blob = json.loads(sf.read_text())
                for vs in blob["verification_state"].values():
                    assert vs["attempts"] == 6

    # -- Done-but-incomplete resumability fix ----------------------------------

    def test_artifacts_phase_triage_gap_rewinds_to_triage(self, monkeypatch, tmp_path):
        """When the artifacts phase detects unreviewed files, state must rewind
        to 'triage' so /audit --resume re-enters the triage phase and can fill
        the gap."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")
        _commit_file(tmp_path, "b.py", "y = 2")

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = self._make_scope(commit=commit, files=["a.py", "b.py"])
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="gap-test",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py"},  # b.py missing
            candidate_files=[],
            deep_reviewed_files=set(),
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("--resume", ctx)
        assert "Audit incomplete" in result
        assert "not reviewed" in result

        # State must be rewound to "triage", not stuck at "artifacts" or "done"
        found = AuditRunState.find_resumable(state_dir, commit, None)
        assert found is not None
        assert found.phase == "triage"

    def test_artifacts_phase_deep_review_gap_rewinds_to_deep_review(
        self, monkeypatch, tmp_path
    ):
        """When artifacts phase detects deep-review gaps, state must rewind
        to 'deep_review' so /audit --resume can fill them."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")
        _commit_file(tmp_path, "b.py", "y = 2")

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = self._make_scope(commit=commit, files=["a.py", "b.py"])
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="gap-dr-test",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py", "b.py"},
            candidate_files=["a.py", "b.py"],
            deep_reviewed_files={"a.py"},  # b.py not deep-reviewed
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )
        result = run_audit_command("--resume", ctx)
        assert "Audit incomplete" in result
        assert "deep review" in result

        found = AuditRunState.find_resumable(state_dir, commit, None)
        assert found is not None
        assert found.phase == "deep_review"

    def test_triage_gap_resume_recovers_and_completes(self, monkeypatch, tmp_path):
        """End-to-end: a run stuck at artifacts with a triage gap should
        complete after two resumes — first rewinds to triage, second finishes."""
        from types import SimpleNamespace
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")
        _commit_file(tmp_path, "b.py", "y = 2")

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = self._make_scope(commit=commit, files=["a.py", "b.py"])
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="recover-test",
            scope=scope,
            queued_files=["a.py", "b.py"],
            reviewed_files={"a.py"},  # b.py missing
            candidate_files=[],
            deep_reviewed_files=set(),
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        monkeypatch.setattr(
            "swival.audit._phase2_triage_one",
            lambda path, state, ctx: self._triage_skip(path),
        )

        ctx = SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )

        # First resume: rewinds to triage, returns incomplete
        result1 = run_audit_command("--resume", ctx)
        assert "not reviewed" in result1

        # Second resume: triage fills b.py, no findings, completes
        result2 = run_audit_command("--resume", ctx)
        assert "No provable security bugs" in result2


class TestCallAuditLlmOverflowRetry:
    """Tests for _call_audit_llm context-overflow truncation retry."""

    def _make_ctx(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            base_dir="/tmp",
            trace_dir=None,
            loop_kwargs={
                "api_base": "http://localhost",
                "model_id": "test",
                "max_output_tokens": 1024,
                "llm_kwargs": {"provider": "lmstudio"},
            },
        )

    def test_no_overflow_returns_full_content(self, monkeypatch):
        from types import SimpleNamespace

        from swival.audit import _call_audit_llm

        def fake_call_llm(*args, **kwargs):
            msg = SimpleNamespace(content="ok", role="assistant")
            return msg, "stop", None, 0, None

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 1000},
        ]
        result = _call_audit_llm(ctx, msgs)
        assert result == "ok"

    def test_overflow_retries_with_truncated_content(self, monkeypatch):
        from types import SimpleNamespace

        from swival.agent import ContextOverflowError
        from swival.audit import _call_audit_llm

        seen_texts = []

        def fake_call_llm(*args, **kwargs):
            messages = args[2]
            user_text = messages[-1]["content"]
            seen_texts.append(user_text)
            if "[truncated" not in user_text:
                raise ContextOverflowError("too big")
            msg = SimpleNamespace(content="truncated-ok", role="assistant")
            return msg, "stop", None, 0, None

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 1000},
        ]
        result = _call_audit_llm(ctx, msgs)
        assert result == "truncated-ok"
        assert len(seen_texts) == 2
        assert len(seen_texts[0]) == 1000
        assert "[truncated" in seen_texts[1]
        assert len(seen_texts[1]) < 1000

    def test_adaptive_truncation_multiple_halvings(self, monkeypatch):
        from types import SimpleNamespace

        from swival.agent import ContextOverflowError
        from swival.audit import _call_audit_llm

        calls = []

        def fake_call_llm(*args, **kwargs):
            messages = args[2]
            user_text = messages[-1]["content"]
            calls.append(len(user_text))
            if len(user_text) > 400:
                raise ContextOverflowError("too big")
            msg = SimpleNamespace(content="ok-after-two", role="assistant")
            return msg, "stop", None, 0, None

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "y" * 2000},
        ]
        result = _call_audit_llm(ctx, msgs)
        assert result == "ok-after-two"
        assert len(calls) >= 3
        assert calls[0] == 2000
        for c in calls[1:]:
            assert c < calls[0]

    def test_overflow_raises_when_floor_reached(self, monkeypatch):
        from swival.agent import ContextOverflowError
        from swival.audit import _call_audit_llm

        def fake_call_llm(*args, **kwargs):
            raise ContextOverflowError("always too big")

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "z" * 500},
        ]
        with pytest.raises(ContextOverflowError):
            _call_audit_llm(ctx, msgs)

    def test_overflow_trace_records_original_attempt(self, monkeypatch):
        from types import SimpleNamespace

        from swival.agent import ContextOverflowError
        from swival.audit import _call_audit_llm

        traces = []

        def fake_trace(ctx, messages, task=None):
            traces.append(task)

        def fake_call_llm(*args, **kwargs):
            messages = args[2]
            user_text = messages[-1]["content"]
            if len(user_text) > 600:
                raise ContextOverflowError("too big")
            msg = SimpleNamespace(content="ok", role="assistant")
            return msg, "stop", None, 0, None

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        monkeypatch.setattr("swival.audit._write_audit_trace", fake_trace)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "w" * 1000},
        ]
        _call_audit_llm(ctx, msgs, trace_task="triage foo.py")
        assert any("overflow" in (t or "") for t in traces)
        assert any(t == "triage foo.py" for t in traces)

    def test_empty_response_retries_with_truncation(self, monkeypatch):
        from types import SimpleNamespace

        from swival.audit import _call_audit_llm

        calls = []

        def fake_call_llm(*args, **kwargs):
            messages = args[2]
            user_text = messages[-1]["content"]
            calls.append(len(user_text))
            if "[truncated" not in user_text:
                msg = SimpleNamespace(content="", role="assistant")
                return msg, "stop", None, 0, None
            msg = SimpleNamespace(content="ok-after-truncation", role="assistant")
            return msg, "stop", None, 0, None

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        ctx = self._make_ctx()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 2000},
        ]
        result = _call_audit_llm(ctx, msgs)
        assert result == "ok-after-truncation"
        assert len(calls) == 2
        assert calls[0] == 2000
        assert calls[1] < 2000


class TestMultiFocusPaths:
    def test_normalize_focus_strips_trailing_slash_and_dedupes(self):
        from swival.audit import _normalize_focus

        assert _normalize_focus(["src/a/", "src/b", "src/a"]) == ["src/a", "src/b"]
        assert _normalize_focus([]) == []
        assert _normalize_focus(["/"]) == ["/"]

    def test_normalize_focus_preserves_trailing_slash_on_globs(self):
        from swival.audit import _normalize_focus

        assert _normalize_focus(["src/*/"]) == ["src/*/"]
        assert _normalize_focus(["src/?/"]) == ["src/?/"]
        assert _normalize_focus(["src/[ab]/"]) == ["src/[ab]/"]
        assert _normalize_focus(["src/*.py"]) == ["src/*.py"]

    def test_resolve_scope_unions_two_paths(self, tmp_path):
        from swival.audit import _resolve_scope

        _init_git(tmp_path)
        _commit_file(tmp_path, "src/a/x.py", "pass")
        _commit_file(tmp_path, "src/b/y.py", "pass")
        _commit_file(tmp_path, "lib/c.py", "pass")

        scope = _resolve_scope(str(tmp_path), ["src/a", "src/b"])
        assert "src/a/x.py" in scope.mandatory_files
        assert "src/b/y.py" in scope.mandatory_files
        assert "lib/c.py" not in scope.mandatory_files

    def test_resolve_scope_normalizes_trailing_slash(self, tmp_path):
        from swival.audit import _resolve_scope

        _init_git(tmp_path)
        _commit_file(tmp_path, "src/a.py", "pass")

        scope = _resolve_scope(str(tmp_path), ["src/"])
        assert scope.focus == ["src"]
        assert "src/a.py" in scope.mandatory_files

    def test_from_dict_coerces_legacy_string_focus(self):
        scope = AuditScope.from_dict(
            {
                "branch": "main",
                "commit": "abc",
                "tracked_files": ["a.py"],
                "mandatory_files": ["a.py"],
                "focus": "src/auth",
            }
        )
        assert scope.focus == ["src/auth"]

    def test_from_dict_coerces_legacy_null_focus(self):
        scope = AuditScope.from_dict(
            {
                "branch": "main",
                "commit": "abc",
                "tracked_files": ["a.py"],
                "mandatory_files": ["a.py"],
                "focus": None,
            }
        )
        assert scope.focus == []

    def test_from_dict_normalizes_list_focus(self):
        scope = AuditScope.from_dict(
            {
                "branch": "main",
                "commit": "abc",
                "tracked_files": [],
                "mandatory_files": [],
                "focus": ["src/a/", "src/a", "src/b/"],
            }
        )
        assert scope.focus == ["src/a", "src/b"]

    def _make_focused_state(self, tmp_path: Path, focus: list[str]) -> AuditRunState:
        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["src/a/x.py", "src/b/y.py"],
            mandatory_files=["src/a/x.py", "src/b/y.py"],
            focus=focus,
        )
        return AuditRunState(
            run_id="multi-run",
            scope=scope,
            queued_files=list(scope.mandatory_files),
            state_dir=tmp_path / ".swival" / "audit",
            phase="triage",
        )

    def test_find_resumable_set_equality_ignores_order(self, tmp_path):
        state = self._make_focused_state(tmp_path, ["src/a", "src/b"])
        state.save()

        found = AuditRunState.find_resumable(
            state.state_dir, "abc123", ["src/b", "src/a"]
        )
        assert found is not None
        assert found.run_id == "multi-run"

    def test_find_resumable_normalizes_trailing_slash(self, tmp_path):
        state = self._make_focused_state(tmp_path, ["src/a"])
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", ["src/a/"])
        assert found is not None

    def test_find_resumable_none_wildcards_focused_run(self, tmp_path):
        state = self._make_focused_state(tmp_path, ["src/a"])
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", None)
        assert found is not None
        assert found.run_id == "multi-run"

    def test_find_resumable_empty_list_does_not_match_focused_run(self, tmp_path):
        state = self._make_focused_state(tmp_path, ["src/a"])
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", [])
        assert found is None

    def test_find_resumable_empty_list_matches_whole_repo_run(self, tmp_path):
        state = self._make_focused_state(tmp_path, [])
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", [])
        assert found is not None

    def test_find_resumable_matches_persisted_legacy_string_focus(self, tmp_path):
        import json

        state_dir = tmp_path / ".swival" / "audit"
        run_dir = state_dir / "legacy-run"
        run_dir.mkdir(parents=True)
        blob = {
            "run_id": "legacy-run",
            "scope": {
                "branch": "main",
                "commit": "abc123",
                "tracked_files": ["src/auth/x.py"],
                "mandatory_files": ["src/auth/x.py"],
                "focus": "src/auth",
            },
            "queued_files": ["src/auth/x.py"],
            "phase": "triage",
        }
        (run_dir / "state.json").write_text(json.dumps(blob))

        found = AuditRunState.find_resumable(state_dir, "abc123", ["src/auth"])
        assert found is not None
        assert found.scope.focus == ["src/auth"]


class TestAuditCommandParser:
    """Parser-level tests: capture kwargs passed to _run_audit_phases."""

    def _capture(self, monkeypatch):
        import inspect

        from swival.audit import _run_audit_phases

        sig = inspect.signature(_run_audit_phases)
        captured: dict = {}

        def fake_phases(*args, **kwargs):
            captured.update(sig.bind(*args, **kwargs).arguments)
            return "captured"

        monkeypatch.setattr("swival.audit._run_audit_phases", fake_phases)
        return captured

    def _make_ctx(self, tmp_path: Path):
        from types import SimpleNamespace

        return SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )

    def test_no_args_passes_focus_none(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("", self._make_ctx(tmp_path))
        assert captured["focus"] is None
        assert captured["workers"] == 4
        assert captured["resume"] is False
        assert captured["regen"] is False

    def test_two_paths_collected_as_list(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("src/a src/b --workers 2 --resume", self._make_ctx(tmp_path))
        assert captured["focus"] == ["src/a", "src/b"]
        assert captured["workers"] == 2
        assert captured["resume"] is True

    def test_dedupes_repeated_paths(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("src/a src/a/", self._make_ctx(tmp_path))
        assert captured["focus"] == ["src/a"]

    def test_flag_intermixing(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("--resume src/a --workers 4 src/b", self._make_ctx(tmp_path))
        assert captured["focus"] == ["src/a", "src/b"]
        assert captured["workers"] == 4
        assert captured["resume"] is True
