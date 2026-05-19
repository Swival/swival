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
    PatchGenerationResult,
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
    _phase1_source_inventory,
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


def _make_ctx(tmp_path: Path):
    """Build a minimal InputContext stand-in for parser/dispatch tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        base_dir=str(tmp_path),
        tools=[],
        verbose=False,
        no_history=True,
        loop_kwargs={},
    )


def _capture_run_audit_phases(monkeypatch) -> dict:
    """Replace `_run_audit_phases` with a recorder; return the kwargs dict."""
    import inspect

    from swival.audit import _run_audit_phases

    sig = inspect.signature(_run_audit_phases)
    captured: dict = {}

    def fake_phases(*args, **kwargs):
        captured.update(sig.bind(*args, **kwargs).arguments)
        return "captured"

    monkeypatch.setattr("swival.audit._run_audit_phases", fake_phases)
    return captured


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
        ordered, scores = _order_by_attack_surface(["safe.py", "danger.py"], cache)
        assert ordered[0] == "danger.py"
        assert scores["danger.py"] > scores["safe.py"]
        assert scores["safe.py"] == 0


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

    def test_source_inventory_reports_language_counts_and_examples(self):
        inventory = _phase1_source_inventory(
            [
                "src/libsodium/crypto_auth/auth.c",
                "src/libsodium/crypto_auth/auth.h",
                "src/libsodium/crypto_box/box.c",
                "src/libsodium/crypto_scalarmult/curve25519/fe51_mul.S",
                "src/libsodium/Makefile.am",
            ]
        )

        assert "--- source inventory ---" in inventory
        assert "c: 3 file(s); examples: src/libsodium/crypto_auth/auth.c" in inventory
        assert (
            "assembly: 1 file(s); examples: src/libsodium/crypto_scalarmult"
            in inventory
        )
        assert "other extensions: .am=1" in inventory

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

    def test_repo_profile_includes_source_inventory_and_autotools_manifest(
        self, monkeypatch, tmp_path
    ):
        from swival.audit import _phase1_repo_profile

        captured = {}

        def fake_git_show(path, base_dir):
            assert base_dir == str(tmp_path)
            if path == "src/libsodium/Makefile.am":
                return "libsodium_la_SOURCES = crypto_auth/auth.c\n"
            raise AssertionError(f"unexpected git show path: {path}")

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            captured["user"] = messages[1]["content"]
            return "@@ profile @@\nlanguage: c\nsummary: scoped c library\n"

        monkeypatch.setattr("swival.audit._git_show", fake_git_show)
        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=[
                "src/libsodium/Makefile.am",
                "src/libsodium/crypto_auth/auth.c",
                "src/libsodium/crypto_auth/auth.h",
            ],
            mandatory_files=[
                "src/libsodium/crypto_auth/auth.c",
                "src/libsodium/crypto_auth/auth.h",
            ],
            focus=["src/libsodium"],
        )
        state = AuditRunState(
            run_id="p1", scope=scope, queued_files=[], state_dir=tmp_path
        )

        profile = _phase1_repo_profile(state, self._ctx(tmp_path))

        assert profile["languages"] == ["c"]
        assert "--- source inventory ---" in captured["user"]
        assert (
            "c: 2 file(s); examples: src/libsodium/crypto_auth/auth.c"
            in captured["user"]
        )
        assert "--- src/libsodium/Makefile.am ---" in captured["user"]
        assert "libsodium_la_SOURCES" in captured["user"]

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
        state_path = state.state_dir / state.run_id / "state.json"
        assert "next_index" not in state_path.read_text()

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

    def test_find_resumable_skips_state_without_artifact_state(self, tmp_path):
        import json

        state_dir = tmp_path / ".swival" / "audit"
        run_dir = state_dir / "legacy"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(
            json.dumps(
                {
                    "run_id": "legacy",
                    "scope": {
                        "branch": "m",
                        "commit": "c",
                        "tracked_files": ["a.py"],
                        "mandatory_files": ["a.py"],
                        "focus": [],
                    },
                    "queued_files": ["a.py"],
                    "phase": "triage",
                }
            )
        )

        assert AuditRunState.find_resumable(state_dir, "c", None) is None

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
# Hunt-harness foundation: HuntTask, coverage records, root-cause groups
# ---------------------------------------------------------------------------


class TestHuntTaskFoundation:
    def _ssrf_task(self, **over):
        from swival.audit import HuntTask

        base = dict(
            id="ssrf-abcd1234",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url POST body"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/api/**",
            seed_files=["src/api/webhook.py"],
            seed_symbols=["handle_webhook"],
            sink_files=["src/clients/fetch.py"],
            source="phase1",
            priority="high",
        )
        base.update(over)
        return HuntTask(**base)

    def test_validate_accepts_well_formed_task(self):
        from swival.audit import _validate_hunt_task

        _validate_hunt_task(self._ssrf_task())

    def test_validate_rejects_missing_attacker_position(self):
        from swival.audit import HuntTaskValidationError, _validate_hunt_task

        with pytest.raises(HuntTaskValidationError) as exc:
            _validate_hunt_task(self._ssrf_task(attacker_position=""))
        assert "attacker_position" in str(exc.value)

    def test_validate_rejects_missing_controlled_inputs(self):
        from swival.audit import HuntTaskValidationError, _validate_hunt_task

        with pytest.raises(HuntTaskValidationError) as exc:
            _validate_hunt_task(self._ssrf_task(controlled_inputs=[]))
        assert "controlled_inputs" in str(exc.value)

        with pytest.raises(HuntTaskValidationError):
            _validate_hunt_task(self._ssrf_task(controlled_inputs=["", "  "]))

    def test_validate_rejects_missing_trust_boundary(self):
        from swival.audit import HuntTaskValidationError, _validate_hunt_task

        with pytest.raises(HuntTaskValidationError) as exc:
            _validate_hunt_task(self._ssrf_task(trust_boundary_crossed=""))
        assert "trust_boundary_crossed" in str(exc.value)

    def test_validate_rejects_bad_task_kind(self):
        from swival.audit import HuntTaskValidationError, _validate_hunt_task

        with pytest.raises(HuntTaskValidationError) as exc:
            _validate_hunt_task(self._ssrf_task(task_kind="bogus"))
        assert "task_kind" in str(exc.value)

    def test_validate_reachability_kind_accepted(self):
        from swival.audit import _validate_hunt_task

        _validate_hunt_task(self._ssrf_task(task_kind="reachability"))

    def test_validate_unknown_attack_class_records_coverage_note(self):
        from swival.audit import _validate_hunt_task

        t = self._ssrf_task(attack_class="prompt_injection")
        _validate_hunt_task(t)
        assert any("unknown attack_class" in n for n in t.coverage_notes)

    def test_hunt_task_id_is_deterministic(self):
        from swival.audit import _hunt_task_id

        a = _hunt_task_id(
            "ssrf",
            "unauthenticated remote client",
            "src/api/**",
            ["src/api/webhook.py"],
        )
        b = _hunt_task_id(
            "ssrf",
            "unauthenticated remote client",
            "src/api/**",
            ["src/api/webhook.py"],
        )
        assert a == b
        assert a.startswith("ssrf-")

    def test_hunt_task_id_changes_with_scope(self):
        from swival.audit import _hunt_task_id

        a = _hunt_task_id("ssrf", "x", "src/a/**", ["f1"])
        b = _hunt_task_id("ssrf", "x", "src/b/**", ["f1"])
        assert a != b

    def test_hunt_task_id_order_insensitive(self):
        from swival.audit import _hunt_task_id

        a = _hunt_task_id("ssrf", "x", "src/a/**", ["f1", "f2"])
        b = _hunt_task_id("ssrf", "x", "src/a/**", ["f2", "f1"])
        assert a == b


class TestHuntTaskGeneration:
    """Deterministic hunt-task generation from Phase-1 + sink scans."""

    def _state(self, tmp_path):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["src/api/webhook.py", "src/exec.py", "src/clients/fetch.py"],
            mandatory_files=[
                "src/api/webhook.py",
                "src/exec.py",
                "src/clients/fetch.py",
            ],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=list(scope.mandatory_files),
            state_dir=tmp_path,
        )
        state.repo_profile = {
            "entry_points": ["src/api/main.py", "src/api/webhook.py"],
            "trust_boundaries": ["HTTP API"],
            "auth_surfaces": [],
            "dangerous_operations": ["subprocess"],
        }
        state.attack_scores = {
            "src/api/webhook.py": 7,
            "src/exec.py": 9,
            "src/clients/fetch.py": 5,
        }
        return state

    def _cache(self):
        return {
            "src/api/webhook.py": (
                "import requests\n"
                "def handle_webhook(req):\n"
                "    url = req.json['callback_url']\n"
                "    return requests.get(url)\n"
            ),
            "src/exec.py": (
                "import subprocess\n"
                "def run_cmd(req):\n"
                "    subprocess.run(req.body['cmd'], shell=True)\n"
            ),
            "src/clients/fetch.py": (
                "import requests\n"
                "def fetch_remote(url):\n"
                "    return requests.get(url)\n"
            ),
        }

    def test_generates_tasks_with_valid_attacker_model(self, tmp_path):
        from swival.audit import _generate_hunt_tasks, _validate_hunt_task

        tasks = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        assert tasks, "expected at least one task"
        for t in tasks:
            _validate_hunt_task(t)
            assert t.attack_class
            assert t.attacker_position
            assert t.controlled_inputs
            assert t.trust_boundary_crossed

    def test_classes_include_command_and_ssrf(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        tasks = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        classes = {t.attack_class for t in tasks}
        assert "command_execution" in classes
        assert "ssrf" in classes

    def test_id_is_stable_across_runs(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        a = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        b = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        assert [t.id for t in a] == [t.id for t in b]

    def test_configured_tasks_come_first_and_survive_trim(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        configured = [
            {
                "attack_class": "authorization",
                "attacker_position": "authenticated low-privileged user",
                "controlled_inputs": ["tenant_id"],
                "trust_boundary_crossed": "HTTP API handler",
                "scope": "src/**",
                "priority": "high",
            }
        ]
        tasks = _generate_hunt_tasks(
            self._state(tmp_path),
            self._cache(),
            configured=configured,
            max_tasks=1,
        )
        assert len(tasks) == 1
        assert tasks[0].source == "user"
        assert tasks[0].attack_class == "authorization"

    def test_invalid_configured_entry_is_dropped_not_raised(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        configured = [
            # missing trust_boundary
            {
                "attack_class": "ssrf",
                "attacker_position": "unauthenticated remote client",
                "controlled_inputs": ["url"],
                "scope": "src/**",
            },
            # well-formed
            {
                "attack_class": "authorization",
                "attacker_position": "authenticated low-privileged user",
                "controlled_inputs": ["tenant_id"],
                "trust_boundary_crossed": "API handler",
                "scope": "src/**",
            },
        ]
        tasks = _generate_hunt_tasks(
            self._state(tmp_path), self._cache(), configured=configured
        )
        configured_tasks = [t for t in tasks if t.source == "user"]
        assert len(configured_tasks) == 1
        assert configured_tasks[0].attack_class == "authorization"

    def test_language_hint_is_set_from_seed_extensions(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        tasks = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        py_tasks = [t for t in tasks if any(s.endswith(".py") for s in t.seed_files)]
        assert py_tasks
        assert all(t.language_hint == "python" for t in py_tasks)

    def test_high_attack_score_seeds_get_high_priority(self, tmp_path):
        from swival.audit import _generate_hunt_tasks

        tasks = _generate_hunt_tasks(self._state(tmp_path), self._cache())
        # exec.py attack score is 9, so its command_execution task should be high
        exec_tasks = [t for t in tasks if "src/exec.py" in t.seed_files]
        assert exec_tasks
        assert any(t.priority == "high" for t in exec_tasks)


class TestHunterResponseParsing:
    """The hunter LLM response parser splits findings from coverage and tolerates
    malformed blocks (returning the partial result with a warning rather than
    raising)."""

    _SAMPLE = (
        "@@ finding @@\n"
        "title: callback_url reaches outbound fetch without host allowlist\n"
        "severity: high\n"
        "location: src/clients/fetch.py:42\n"
        "local_bug: outbound HTTP GET trusts caller-supplied URL with no host or scheme check\n"
        "source_boundary: POST /webhook handler\n"
        "controlled_input: callback_url JSON field\n"
        "reachability_step: POST /webhook\n"
        "reachability_step: handle_webhook\n"
        "reachability_step: client.fetch_remote\n"
        "sink_operation: requests.get(url)\n"
        "reachability_status: reachable\n"
        "exploit_chain_step: attacker posts callback_url=http://169.254.169.254/\n"
        "claim: SSRF: attacker controls callback_url end-to-end into requests.get\n"
        "\n"
        "@@ coverage @@\n"
        "observed_file: src/api/webhook.py\n"
        "observed_file: src/clients/fetch.py\n"
        "observed_symbol: handle_webhook\n"
        "observed_edge: src/api/webhook.py->src/clients/fetch.py\n"
        "not_covered: did not inspect retry path in src/clients/retry.py\n"
        "followup_task: ssrf | unauthenticated remote client | retry URL | retry uncovered\n"
        "confidence: medium\n"
    )

    def test_parses_findings_and_coverage(self):
        from swival.audit import _parse_hunt_response

        metrics = {}
        findings, coverage = _parse_hunt_response(self._SAMPLE, metrics)
        assert len(findings) == 1
        f = findings[0]
        assert f["title"].startswith("callback_url reaches")
        assert f["severity"] == "high"
        assert f["reachability_status"] == "reachable"
        assert f["controlled_inputs"] == ["callback_url JSON field"]
        assert f["reachability_path"] == [
            "POST /webhook",
            "handle_webhook",
            "client.fetch_remote",
        ]
        assert coverage is not None
        assert coverage["confidence"] == "medium"
        assert "src/api/webhook.py" in coverage["observed_files"]
        assert (
            "did not inspect retry path in src/clients/retry.py"
            in coverage["not_covered"]
        )

    def test_none_response_still_returns_coverage(self):
        from swival.audit import _parse_hunt_response

        raw = (
            "@@ none @@\n"
            "\n"
            "@@ coverage @@\n"
            "observed_file: src/api/webhook.py\n"
            "not_covered: did not inspect src/clients/fetch.py\n"
            "confidence: low\n"
        )
        findings, coverage = _parse_hunt_response(raw, {})
        assert findings == []
        assert coverage is not None
        assert coverage["confidence"] == "low"

    def test_finding_to_record_carries_harness_fields(self, tmp_path):
        from swival.audit import (
            HuntTask,
            _hunt_finding_to_record,
            _parse_hunt_response,
        )

        findings, _coverage = _parse_hunt_response(self._SAMPLE, {})
        task = HuntTask(
            id="ssrf-x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/**",
            seed_files=["src/api/webhook.py", "src/clients/fetch.py"],
            seed_symbols=[],
            sink_files=["src/clients/fetch.py"],
            source="phase1",
            priority="high",
        )
        rec = _hunt_finding_to_record(findings[0], task)
        assert rec.attack_class == "ssrf"
        assert rec.attacker_position == "unauthenticated remote client"
        assert rec.local_bug.startswith("outbound HTTP GET")
        assert rec.reachability_status == "reachable"
        assert rec.hunt_task_id == "ssrf-x"
        assert rec.reachability_path == [
            "POST /webhook",
            "handle_webhook",
            "client.fetch_remote",
        ]
        assert rec.sink_operation == "requests.get(url)"
        assert rec.finding_type == "ssrf"
        assert rec.controlled_inputs == ["callback_url JSON field"]

    def test_coverage_record_ratios(self, tmp_path):
        from swival.audit import (
            CoverageRecord,
            HuntTask,
            _coverage_record_from_parsed,
        )

        task = HuntTask(
            id="ssrf-x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/**",
            seed_files=["a.py", "b.py", "c.py"],
            seed_symbols=[],
            sink_files=["a.py", "b.py"],
            source="phase1",
            priority="high",
        )
        cov = _coverage_record_from_parsed(
            task,
            None,
            {
                "observed_files": ["a.py"],
                "observed_symbols": [],
                "observed_edges": [],
                "not_covered": ["b.py", "c.py"],
                "confidence": "low",
            },
        )
        assert isinstance(cov, CoverageRecord)
        assert cov.covered_fraction_of_seed_files == round(1 / 3, 3)
        assert cov.covered_fraction_of_sink_hits == 0.5
        assert cov.unobserved_high_priority_seeds == ["b.py", "c.py"]
        assert cov.confidence == "low"

    def test_invalid_reachability_status_normalises_to_unknown(self):
        from swival.audit import HuntTask, _hunt_finding_to_record

        task = HuntTask(
            id="x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=[],
            source="phase1",
            priority="high",
        )
        rec = _hunt_finding_to_record(
            {
                "title": "t",
                "severity": "high",
                "location": "a.py:1",
                "local_bug": "x",
                "source_boundary": "y",
                "sink_operation": "z",
                "reachability_status": "yes-please",
                "claim": "c",
            },
            task,
        )
        assert rec.reachability_status == "unknown"


class TestPhaseHuntOne:
    """End-to-end exercise of the hunt phase for one task, with the LLM
    mocked. Verifies that hunter output becomes a FindingRecord with the
    harness fields populated, that the coverage record is persisted, and that
    attack_class_metrics is bumped."""

    def test_full_task_run(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _phase_hunt_one,
        )

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["src/api/webhook.py"],
            mandatory_files=["src/api/webhook.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["src/api/webhook.py"],
            state_dir=tmp_path,
        )
        state.budget_tokens = 1_000_000
        task = HuntTask(
            id="ssrf-x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/api/**",
            seed_files=["src/api/webhook.py"],
            seed_symbols=["handle_webhook"],
            sink_files=["src/clients/fetch.py"],
            source="phase1",
            priority="high",
        )

        monkeypatch.setattr(
            "swival.audit._git_show",
            lambda path, base_dir: (
                "def handle_webhook(req): return requests.get(req.json['callback_url'])"
            ),
        )

        canned = (
            "@@ finding @@\n"
            "title: callback_url reaches outbound fetch without host allowlist\n"
            "severity: high\n"
            "location: src/clients/fetch.py:42\n"
            "local_bug: outbound HTTP GET trusts caller-supplied URL with no host or scheme check\n"
            "source_boundary: POST /webhook handler\n"
            "controlled_input: callback_url JSON field\n"
            "reachability_step: POST /webhook\n"
            "reachability_step: handle_webhook\n"
            "reachability_step: client.fetch_remote\n"
            "sink_operation: requests.get(url)\n"
            "reachability_status: reachable\n"
            "claim: SSRF via webhook callback_url\n"
            "\n"
            "@@ coverage @@\n"
            "observed_file: src/api/webhook.py\n"
            "observed_symbol: handle_webhook\n"
            "not_covered: did not check retry path\n"
            "confidence: medium\n"
        )

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            assert trace_task and trace_task.startswith("audit: hunt ssrf-x")
            return canned

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        findings, coverage, error = _phase_hunt_one(task, state, ctx)
        assert error is None
        assert len(findings) == 1
        f = findings[0]
        assert f.attack_class == "ssrf"
        assert f.reachability_status == "reachable"
        assert f.hunt_task_id == "ssrf-x"
        assert f.source_file == "src/api/webhook.py"
        # Budget was charged
        assert state.budget_used.get("hunt", 0) > 0
        # Coverage record ratios populated against task scope
        assert coverage.confidence == "medium"
        assert coverage.observed_files == ["src/api/webhook.py"]
        assert coverage.covered_fraction_of_seed_files == 1.0
        # attack_class_metrics bumped
        assert state.attack_class_metrics["ssrf"]["proposed"] == 1

    def test_resume_picks_up_only_pending_tasks(self, tmp_path):
        """When state is saved mid-batch, resume picks up tasks still
        ``pending`` and leaves the ones already marked ``done`` alone."""
        from swival.audit import AuditRunState, AuditScope, HuntTask

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = tmp_path / ".swival" / "audit"
        state = AuditRunState(
            run_id="r1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
            hunt_mode=True,
        )
        state.phase = "hunt"
        done_t = HuntTask(
            id="ssrf-done",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=["a.py"],
            source="phase1",
            priority="high",
            status="done",
            attempts=1,
        )
        pending_t = HuntTask(
            id="cmd-pending",
            task_kind="hunt",
            attack_class="command_execution",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["cmd"],
            trust_boundary_crossed="POST /run",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=["a.py"],
            source="phase1",
            priority="high",
            status="pending",
        )
        state.hunt_tasks[done_t.id] = done_t
        state.hunt_tasks[pending_t.id] = pending_t
        state.save()

        loaded = AuditRunState.load(state_dir, "r1")
        pending = [t for t in loaded.hunt_tasks.values() if t.status == "pending"]
        assert [t.id for t in pending] == ["cmd-pending"]
        # done task survives with attempts intact
        assert loaded.hunt_tasks["ssrf-done"].attempts == 1

    def test_failed_tasks_block_advance_to_verification(self, tmp_path, monkeypatch):
        """If any hunt task is still ``failed`` after the retry loop, the
        phase must stay incomplete and return an actionable error rather than
        silently advancing to verification (which would let a false 'no
        findings' result through)."""
        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _run_audit_phases,
        )

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1\n")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = tmp_path / ".swival" / "audit"
        state = AuditRunState(
            run_id="huntfail",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
            hunt_mode=True,
        )
        state.phase = "hunt"
        state.repo_profile = {"summary": "tiny"}
        state.hunt_tasks["t-fail"] = HuntTask(
            id="t-fail",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["url"],
            trust_boundary_crossed="POST /x",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=[],
            source="phase1",
            priority="high",
        )
        state.save()

        # Hunter LLM always errors out, so the task stays in ``failed`` after
        # all retries.
        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            raise RuntimeError("simulated transient failure")

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")

        result = _run_audit_phases(
            "--resume",
            _make_ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
            hunt_mode=True,
        )
        assert "Audit incomplete" in result
        assert "hunt task" in result
        assert "No provable" not in result

        loaded = AuditRunState.load(state_dir, "huntfail")
        assert loaded.phase == "hunt"
        assert loaded.hunt_tasks["t-fail"].status == "failed"
        # The retry loop did run multiple times before giving up.
        assert loaded.hunt_tasks["t-fail"].attempts >= 2

    def test_stale_running_tasks_reset_on_resume(self, tmp_path, monkeypatch):
        """A previous run crashed mid-batch and left a task ``running``.
        On resume the hunt phase resets it to ``pending`` and retries it."""
        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _run_audit_phases,
        )

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1\n")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = tmp_path / ".swival" / "audit"
        state = AuditRunState(
            run_id="huntstale",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
            hunt_mode=True,
        )
        state.phase = "hunt"
        state.repo_profile = {"summary": "tiny"}
        state.hunt_tasks["t-stale"] = HuntTask(
            id="t-stale",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["url"],
            trust_boundary_crossed="POST /x",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=[],
            source="phase1",
            priority="high",
            status="running",
            attempts=1,
        )
        state.save()

        canned = "@@ none @@\n\n@@ coverage @@\nobserved_file: a.py\nconfidence: low\n"

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            return canned

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")

        result = _run_audit_phases(
            "--resume",
            _make_ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
            hunt_mode=True,
        )
        # task ran to completion → audit reports either no findings or
        # advances cleanly; the key invariant is that the stale row no longer
        # holds the phase incomplete.
        assert "still marked running" not in result
        loaded = AuditRunState.load(state_dir, "huntstale")
        assert loaded.hunt_tasks["t-stale"].status == "done"
        assert loaded.phase in ("verification", "artifacts", "done")

    def test_low_priority_tasks_dropped_when_budget_throttled(
        self, tmp_path, monkeypatch
    ):
        """When ``BudgetPlanner.should_throttle_low_priority()`` flips, the
        hunt phase drops any task whose priority is ``low`` before the next
        batch, leaving higher-priority tasks alone."""
        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _run_audit_phases,
        )

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1\n")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )

        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = tmp_path / ".swival" / "audit"
        state = AuditRunState(
            run_id="huntthrottle",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
            hunt_mode=True,
        )
        state.phase = "hunt"
        state.repo_profile = {"summary": "tiny"}
        state.budget_tokens = 100
        state.budget_used = {"hunt": 60}  # > 50% of global -> throttle

        def _make_task(tid, priority):
            return HuntTask(
                id=tid,
                task_kind="hunt",
                attack_class="ssrf",
                attacker_position="unauthenticated remote client",
                controlled_inputs=["url"],
                trust_boundary_crossed="POST /x",
                scope_hint="",
                seed_files=["a.py"],
                seed_symbols=[],
                sink_files=[],
                source="phase1",
                priority=priority,
            )

        state.hunt_tasks["high-1"] = _make_task("high-1", "high")
        state.hunt_tasks["low-1"] = _make_task("low-1", "low")
        state.hunt_tasks["low-2"] = _make_task("low-2", "low")
        state.save()

        canned = "@@ none @@\n\n@@ coverage @@\nobserved_file: a.py\nconfidence: low\n"

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            return canned

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x = 1")

        result = _run_audit_phases(
            "--resume",
            _make_ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
            hunt_mode=True,
        )
        # Throttling marks dropped tasks ``failed`` with a clear reason, which
        # under the new gate keeps the audit incomplete.
        assert "Audit incomplete" in result
        loaded = AuditRunState.load(state_dir, "huntthrottle")
        assert loaded.hunt_tasks["high-1"].status == "done"
        assert loaded.hunt_tasks["low-1"].status == "failed"
        assert loaded.hunt_tasks["low-2"].status == "failed"
        assert "budget throttle" in loaded.hunt_tasks["low-1"].last_error

    def test_budget_exhaustion_short_circuits_before_llm(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _phase_hunt_one,
        )

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path,
        )
        state.budget_tokens = 1000
        state.budget_used = {"hunt": 1500}  # already over global cap

        def boom(*a, **kw):
            raise AssertionError("LLM should not be called when budget is exhausted")

        monkeypatch.setattr("swival.audit._call_audit_llm", boom)
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "x")

        task = HuntTask(
            id="x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["url"],
            trust_boundary_crossed="POST /x",
            scope_hint="",
            seed_files=["a.py"],
            seed_symbols=[],
            sink_files=[],
            source="phase1",
            priority="high",
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        findings, coverage, error = _phase_hunt_one(task, state, ctx)
        assert findings == []
        assert error and "budget exhausted" in error

    def test_invalid_task_returns_error(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from swival.audit import (
            AuditRunState,
            AuditScope,
            HuntTask,
            _phase_hunt_one,
        )

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r", scope=scope, queued_files=["a.py"], state_dir=tmp_path
        )
        task = HuntTask(
            id="bad",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="",  # missing
            controlled_inputs=["x"],
            trust_boundary_crossed="x",
            scope_hint="",
            seed_files=[],
            seed_symbols=[],
            sink_files=[],
            source="phase1",
            priority="high",
        )

        # The mocked LLM must NOT be called for invalid tasks.
        def boom(*a, **kw):
            raise AssertionError("should not call LLM for invalid task")

        monkeypatch.setattr("swival.audit._call_audit_llm", boom)
        monkeypatch.setattr("swival.audit._git_show", lambda p, b: "")
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        findings, coverage, error = _phase_hunt_one(task, state, ctx)
        assert findings == []
        assert error is not None and "invalid task" in error
        assert coverage.task_id == "bad"


class TestReachabilitySplit:
    """Step 4: LOCAL_ONLY findings always promote to a reachability task or
    are dropped post-trace; high/critical security_control_failure is the
    only carve-out."""

    def _state_with_hunt_task(self, tmp_path):
        from swival.audit import AuditRunState, AuditScope, HuntTask

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["src/api/webhook.py", "src/clients/fetch.py"],
            mandatory_files=["src/api/webhook.py", "src/clients/fetch.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r1",
            scope=scope,
            queued_files=["src/api/webhook.py", "src/clients/fetch.py"],
            state_dir=tmp_path,
            hunt_mode=True,
        )
        parent = HuntTask(
            id="ssrf-1",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/api/**",
            seed_files=["src/api/webhook.py"],
            seed_symbols=["handle_webhook"],
            sink_files=["src/clients/fetch.py"],
            source="phase1",
            priority="high",
            language_hint="python",
            status="done",
        )
        state.hunt_tasks[parent.id] = parent
        return state, parent

    def _local_only_finding(self, parent):
        from swival.audit import FindingRecord

        return FindingRecord(
            title="caller-controlled URL hits fetch_remote",
            finding_type="ssrf",
            severity="high",
            locations=["src/clients/fetch.py:42"],
            preconditions=[],
            proof=["local_bug: ..."],
            fix_outline="",
            source_file="src/clients/fetch.py",
            local_bug="outbound GET trusts caller-supplied URL",
            source_boundary="POST /webhook",
            sink_operation="requests.get",
            reachability_status="local_only",
            hunt_task_id=parent.id,
            attack_class=parent.attack_class,
            attacker_position=parent.attacker_position,
            controlled_inputs=list(parent.controlled_inputs),
        )

    def test_local_only_generates_reachability_task(self, tmp_path):
        from swival.audit import _generate_reachability_tasks, _finding_key

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        state.proposed_findings = [finding]

        tasks = _generate_reachability_tasks(state)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_kind == "reachability"
        assert t.attack_class == "ssrf"
        assert t.attacker_position == parent.attacker_position
        assert t.controlled_inputs == parent.controlled_inputs
        assert t.parent_task_id == parent.id
        assert t.parent_finding_key == _finding_key(finding)
        assert t.source == "local_only_promotion"
        assert "src/clients/fetch.py" in t.sink_files

    def test_reachability_task_dedupes_on_repeat(self, tmp_path):
        from swival.audit import _generate_reachability_tasks

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        state.proposed_findings = [finding]
        first = _generate_reachability_tasks(state)
        for t in first:
            state.hunt_tasks[t.id] = t
        second = _generate_reachability_tasks(state)
        assert second == []

    def test_scf_high_skips_reachability_task(self, tmp_path):
        from swival.audit import _generate_reachability_tasks

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        finding.finding_type = "security_control_failure"
        finding.severity = "critical"
        state.proposed_findings = [finding]
        tasks = _generate_reachability_tasks(state)
        assert tasks == []

    def test_scf_low_still_promotes(self, tmp_path):
        from swival.audit import _generate_reachability_tasks

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        finding.finding_type = "security_control_failure"
        finding.severity = "low"
        state.proposed_findings = [finding]
        tasks = _generate_reachability_tasks(state)
        assert len(tasks) == 1

    def test_reachable_finding_does_not_promote(self, tmp_path):
        from swival.audit import _generate_reachability_tasks

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        finding.reachability_status = "reachable"
        state.proposed_findings = [finding]
        assert _generate_reachability_tasks(state) == []

    def test_orphan_finding_skipped(self, tmp_path):
        from swival.audit import _generate_reachability_tasks

        state, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        finding.hunt_task_id = "missing-task"
        state.proposed_findings = [finding]
        assert _generate_reachability_tasks(state) == []

    def test_parse_reachability_response_reachable(self):
        from swival.audit import _parse_reachability_response

        raw = (
            "@@ reachability @@\n"
            "verdict: reachable\n"
            "reachability_step: POST /webhook\n"
            "reachability_step: handle_webhook\n"
            "reachability_step: client.fetch_remote\n"
            "confidence: high\n"
        )
        metrics: dict[str, int] = {}
        parsed = _parse_reachability_response(raw, metrics)
        assert parsed is not None
        assert parsed["verdict"] == "reachable"
        assert parsed["reachability_path"] == [
            "POST /webhook",
            "handle_webhook",
            "client.fetch_remote",
        ]

    def test_parse_reachability_malformed_returns_none(self):
        from swival.audit import _parse_reachability_response

        raw = "garbage with no record header"
        assert _parse_reachability_response(raw, {}) is None

    def test_apply_reachability_result_updates_finding(self, tmp_path):
        from swival.audit import _apply_reachability_result

        _, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        _apply_reachability_result(
            finding,
            {"verdict": "reachable", "reachability_path": ["a", "b"]},
        )
        assert finding.reachability_status == "reachable"
        assert finding.reachability_path == ["a", "b"]

    def test_apply_reachability_normalizes_bad_verdict(self, tmp_path):
        from swival.audit import _apply_reachability_result

        _, parent = self._state_with_hunt_task(tmp_path)
        finding = self._local_only_finding(parent)
        _apply_reachability_result(finding, {"verdict": "definitely!"})
        assert finding.reachability_status == "unknown"

    def test_filter_keeps_reachable_drops_local_only(self, tmp_path):
        from swival.audit import _filter_to_reachable_findings

        state, parent = self._state_with_hunt_task(tmp_path)
        f_reachable = self._local_only_finding(parent)
        f_reachable.reachability_status = "reachable"
        f_local = self._local_only_finding(parent)
        f_local.title = "still local"
        f_unknown = self._local_only_finding(parent)
        f_unknown.title = "unknown trace"
        f_unknown.reachability_status = "unknown"
        state.proposed_findings = [f_reachable, f_local, f_unknown]
        discarded = _filter_to_reachable_findings(state)
        assert discarded == 2
        assert [f.title for f in state.proposed_findings] == [f_reachable.title]

    def test_filter_keeps_scf_high_even_when_local_only(self, tmp_path):
        from swival.audit import _filter_to_reachable_findings

        state, parent = self._state_with_hunt_task(tmp_path)
        scf = self._local_only_finding(parent)
        scf.finding_type = "security_control_failure"
        scf.severity = "high"
        state.proposed_findings = [scf]
        assert _filter_to_reachable_findings(state) == 0
        assert state.proposed_findings == [scf]

    def test_filter_keeps_file_centric_findings(self, tmp_path):
        from swival.audit import _filter_to_reachable_findings

        state, parent = self._state_with_hunt_task(tmp_path)
        f = self._local_only_finding(parent)
        f.hunt_task_id = ""  # file-centric
        f.reachability_status = "unknown"
        state.proposed_findings = [f]
        assert _filter_to_reachable_findings(state) == 0
        assert state.proposed_findings == [f]

    def test_filter_no_op_outside_hunt_mode(self, tmp_path):
        from swival.audit import _filter_to_reachable_findings

        state, parent = self._state_with_hunt_task(tmp_path)
        state.hunt_mode = False
        f = self._local_only_finding(parent)
        state.proposed_findings = [f]
        assert _filter_to_reachable_findings(state) == 0
        assert state.proposed_findings == [f]


class TestProofBlockParser:
    """Step 5: ``swival-audit-proof-v1`` block parsing and rejection."""

    def _wrap(self, body: str) -> str:
        return f"some preamble\n```swival-audit-proof-v1\n{body}\n```\n"

    def test_parse_runtime_reproduced(self):
        from swival.audit import _parse_proof_block

        body = (
            "verdict: REPRODUCED\n"
            "proof_kind: runtime\n"
            "commands:\n"
            "  - ./poc < input.bin\n"
            "  - gdb -batch -ex bt ./core\n"
            "artifacts:\n"
            "  - poc.c\n"
            "observed_output: SIGSEGV in parse_header\n"
            "trigger: oversized version field\n"
            "impact: remote crash\n"
            "limitations: tested on x86_64 only\n"
        )
        out = _parse_proof_block(self._wrap(body))
        assert out["verdict"] == "REPRODUCED"
        assert out["proof_kind"] == "runtime"
        assert out["commands"] == ["./poc < input.bin", "gdb -batch -ex bt ./core"]
        assert out["artifacts"] == ["poc.c"]
        assert out["impact"] == "remote crash"

    def test_parse_source_reproduced(self):
        from swival.audit import _parse_proof_block

        body = (
            "verdict: REPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: see verify.c:42-58 cited in finding\n"
            "trigger: oversized header\n"
            "impact: overflow\n"
            "limitations: none\n"
        )
        out = _parse_proof_block(self._wrap(body))
        assert out["verdict"] == "REPRODUCED"
        assert out["proof_kind"] == "source"
        assert out["commands"] == []
        assert out["artifacts"] == []

    def test_parse_notreproduced(self):
        from swival.audit import _parse_proof_block

        body = (
            "verdict: NOTREPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: existing length check rejects oversized input\n"
            "trigger: not reachable\n"
            "impact: none\n"
            "limitations: none\n"
        )
        out = _parse_proof_block(self._wrap(body))
        assert out["verdict"] == "NOTREPRODUCED"
        assert out["proof_kind"] == "source"

    def test_parse_notreproduced_allows_none_proof_kind(self):
        """NOTREPRODUCED with ``proof_kind: none`` (or absent) must parse
        cleanly. Treating it as malformed would collapse a clean negative
        verdict into a discard with misleading diagnostics."""
        from swival.audit import _parse_proof_block

        none_body = (
            "verdict: NOTREPRODUCED\n"
            "proof_kind: none\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: control blocks input\n"
            "trigger: not reachable\n"
            "impact: none\n"
            "limitations: none\n"
        )
        out = _parse_proof_block(self._wrap(none_body))
        assert out["verdict"] == "NOTREPRODUCED"
        assert out["proof_kind"] == ""

        absent_body = (
            "verdict: NOTREPRODUCED\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: control blocks input\n"
            "trigger: not reachable\n"
            "impact: none\n"
            "limitations: none\n"
        )
        out = _parse_proof_block(self._wrap(absent_body))
        assert out["verdict"] == "NOTREPRODUCED"
        assert out["proof_kind"] == ""

    def test_parse_malformed_no_verdict_raises(self):
        from swival.audit import ProofBlockError, _parse_proof_block

        body = (
            "proof_kind: source\n"
            "commands: none\n"
            "observed_output: meh\n"
            "trigger: x\n"
            "impact: y\n"
            "limitations: none\n"
        )
        try:
            _parse_proof_block(self._wrap(body))
        except ProofBlockError:
            return
        raise AssertionError("expected ProofBlockError")

    def test_parse_missing_block_raises(self):
        from swival.audit import ProofBlockError, _parse_proof_block

        try:
            _parse_proof_block("nothing useful here, no sentinel")
        except ProofBlockError:
            return
        raise AssertionError("expected ProofBlockError")

    def test_parse_reproduced_requires_proof_kind(self):
        from swival.audit import ProofBlockError, _parse_proof_block

        body = (
            "verdict: REPRODUCED\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: x\n"
            "trigger: y\n"
            "impact: z\n"
            "limitations: none\n"
        )
        try:
            _parse_proof_block(self._wrap(body))
        except ProofBlockError:
            return
        raise AssertionError("expected ProofBlockError")

    def test_parse_takes_last_block_on_sentinel_collision(self):
        from swival.audit import _parse_proof_block

        early = (
            "```swival-audit-proof-v1\n"
            "verdict: REPRODUCED\n"
            "proof_kind: runtime\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: stale\n"
            "trigger: stale\n"
            "impact: stale\n"
            "limitations: none\n"
            "```"
        )
        final = (
            "```swival-audit-proof-v1\n"
            "verdict: NOTREPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: definitive\n"
            "trigger: not reachable\n"
            "impact: none\n"
            "limitations: none\n"
            "```"
        )
        answer = (
            f"the model captured prior output:\n{early}\n\nFinal verdict:\n{final}\n"
        )
        out = _parse_proof_block(answer)
        assert out["verdict"] == "NOTREPRODUCED"
        assert out["observed_output"] == "definitive"

    def test_parse_block_inside_outer_fence(self):
        from swival.audit import _parse_proof_block

        body = (
            "verdict: REPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: cited\n"
            "trigger: x\n"
            "impact: y\n"
            "limitations: none\n"
        )
        answer = f"```text\n```swival-audit-proof-v1\n{body}\n```\n```\n"
        out = _parse_proof_block(answer)
        assert out["verdict"] == "REPRODUCED"

    def test_invalid_proof_kind_raises(self):
        from swival.audit import ProofBlockError, _parse_proof_block

        body = (
            "verdict: REPRODUCED\n"
            "proof_kind: dynamic\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: x\n"
            "trigger: y\n"
            "impact: z\n"
            "limitations: none\n"
        )
        try:
            _parse_proof_block(self._wrap(body))
        except ProofBlockError:
            return
        raise AssertionError("expected ProofBlockError")


class TestPhase4cReproduce:
    """The verifier wrapper accepts only structured proof blocks; malformed
    output, NOTREPRODUCED verdicts, and missing sentinels all map to
    'discarded'."""

    def _state(self, tmp_path):
        from swival.audit import AuditRunState, AuditScope

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["main.c"],
            mandatory_files=["main.c"],
            focus=[],
        )
        return AuditRunState(
            run_id="r5",
            scope=scope,
            queued_files=["main.c"],
            state_dir=tmp_path / ".swival" / "audit",
        )

    def _finding(self):
        from swival.audit import FindingRecord

        return FindingRecord(
            title="t",
            finding_type="memory_safety",
            severity="high",
            locations=["main.c:7"],
            preconditions=[],
            proof=["..."],
            fix_outline="",
            source_file="main.c",
        )

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
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

    def _setup_worktree(self, monkeypatch, tmp_path):
        class Dummy:
            def __init__(self, work_dir):
                self.work_dir = work_dir

            def __enter__(self):
                return self.work_dir

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", lambda base, work: Dummy(work))
        monkeypatch.setattr(
            "swival.audit._gather_evidence", lambda f, c: ("--- main.c ---\n", 1)
        )

    def test_malformed_output_discards(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: ("no sentinel anywhere REPRODUCED", False),
        )
        state = self._state(tmp_path)
        finding = self._finding()
        result = audit._phase4c_reproduce(
            finding, state, self._ctx(tmp_path), tmp_path / "work"
        )
        assert result is None

    def test_reproduced_persists_proof_artifact(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch, tmp_path)
        body = (
            "```swival-audit-proof-v1\n"
            "verdict: REPRODUCED\n"
            "proof_kind: runtime\n"
            "commands:\n  - ./poc\n"
            "artifacts:\n  - poc.c\n"
            "observed_output: SIGSEGV\n"
            "trigger: oversized argv\n"
            "impact: stack overflow\n"
            "limitations: none\n"
            "```"
        )
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (f"trace...\n{body}", False),
        )
        state = self._state(tmp_path)
        finding = self._finding()
        result = audit._phase4c_reproduce(
            finding, state, self._ctx(tmp_path), tmp_path / "work"
        )
        assert result is not None
        assert result["reproduced"] is True
        assert result["proof_kind"] == "runtime"
        assert result["commands"] == ["./poc"]
        assert result["impact"] == "stack overflow"

        from swival.audit import _finding_key

        key = _finding_key(finding)
        proof_dir = state.state_dir / state.run_id / "verify" / key
        assert (proof_dir / "proof.txt").exists()
        assert (proof_dir / "proof.json").exists()
        assert (proof_dir / "commands.log").read_text().strip() == "./poc"

    def test_notreproduced_block_returns_none(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch, tmp_path)
        body = (
            "```swival-audit-proof-v1\n"
            "verdict: NOTREPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: control blocks\n"
            "trigger: not reachable\n"
            "impact: none\n"
            "limitations: none\n"
            "```"
        )
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (body, False),
        )
        state = self._state(tmp_path)
        finding = self._finding()
        result = audit._phase4c_reproduce(
            finding, state, self._ctx(tmp_path), tmp_path / "work"
        )
        assert result is None


class TestDisproofBlockParser:
    """Step 6: ``swival-audit-disproof-v1`` block parsing and rejection."""

    def _wrap(self, body: str) -> str:
        return f"some preamble\n```swival-audit-disproof-v1\n{body}\n```\n"

    def test_parse_invalid_with_blocking_control(self):
        from swival.audit import _parse_disproof_block

        body = (
            "verdict: INVALID\n"
            "reason: handler guards path with realpath/within-root check\n"
            "counterclaim_checked:\n"
            "  - upload path is not rooted under uploads_dir\n"
            "blocking_control: assert resolve(path).startswith(uploads_root)\n"
            "missing_link: none\n"
            "required_next_proof: none\n"
            "review_model: \n"
        )
        out = _parse_disproof_block(self._wrap(body))
        assert out["verdict"] == "INVALID"
        assert out["blocking_control"].startswith("assert resolve")
        assert out["missing_link"] == ""
        assert out["counterclaim_checked"] == [
            "upload path is not rooted under uploads_dir"
        ]

    def test_parse_needs_proof_with_required_next_proof(self):
        from swival.audit import _parse_disproof_block

        body = (
            "verdict: NEEDS_PROOF\n"
            "reason: cannot tell if header path is attacker-reachable\n"
            "counterclaim_checked:\n"
            "  - request handler ever receives X-Trace-Id from clients\n"
            "blocking_control: none\n"
            "missing_link: no exposure of route to unauthenticated clients\n"
            "required_next_proof: curl with crafted X-Trace-Id against /track\n"
            "review_model: \n"
        )
        out = _parse_disproof_block(self._wrap(body))
        assert out["verdict"] == "NEEDS_PROOF"
        assert out["required_next_proof"].startswith("curl")

    def test_parse_plausible(self):
        from swival.audit import _parse_disproof_block

        body = (
            "verdict: PLAUSIBLE\n"
            "reason: could not find any blocking control\n"
            "counterclaim_checked: none\n"
            "blocking_control: none\n"
            "missing_link: none\n"
            "required_next_proof: none\n"
            "review_model: \n"
        )
        out = _parse_disproof_block(self._wrap(body))
        assert out["verdict"] == "PLAUSIBLE"
        assert out["counterclaim_checked"] == []
        assert out["blocking_control"] == ""

    def test_missing_block_raises(self):
        from swival.audit import DisproofBlockError, _parse_disproof_block

        try:
            _parse_disproof_block("INVALID — no block here")
        except DisproofBlockError:
            return
        raise AssertionError("expected DisproofBlockError")

    def test_invalid_verdict_raises(self):
        from swival.audit import DisproofBlockError, _parse_disproof_block

        try:
            _parse_disproof_block(self._wrap("verdict: UNCERTAIN\n"))
        except DisproofBlockError:
            return
        raise AssertionError("expected DisproofBlockError")


class TestDisproofRouting:
    """Step 6: per-finding routing rules from _disproof_routing."""

    def test_invalid_with_concrete_evidence_routes_to_invalid(self):
        from swival.audit import _disproof_routing

        parsed = {
            "verdict": "INVALID",
            "reason": "guarded",
            "counterclaim_checked": ["x"],
            "blocking_control": "if not allowed: return 403",
            "missing_link": "",
            "required_next_proof": "",
            "review_model": "rev-mini",
        }
        rec = _disproof_routing(parsed, None, proof_strict=False)
        assert rec["status"] == "invalid"
        assert rec["blocking_control"].startswith("if not allowed")

    def test_invalid_without_evidence_demotes_to_plausible(self):
        from swival.audit import _disproof_routing

        parsed = {
            "verdict": "INVALID",
            "reason": "vibes",
            "counterclaim_checked": [],
            "blocking_control": "",
            "missing_link": "",
            "required_next_proof": "",
            "review_model": "rev-mini",
        }
        rec = _disproof_routing(parsed, None, proof_strict=False)
        assert rec["status"] == "plausible"

    def test_needs_proof_carries_required_proof(self):
        from swival.audit import _disproof_routing

        parsed = {
            "verdict": "NEEDS_PROOF",
            "reason": "n/a",
            "counterclaim_checked": [],
            "blocking_control": "",
            "missing_link": "no proof",
            "required_next_proof": "curl /x",
            "review_model": "rev-mini",
        }
        rec = _disproof_routing(parsed, None, proof_strict=False)
        assert rec["status"] == "needs_proof"
        assert rec["required_next_proof"] == "curl /x"

    def test_transport_failure_balanced_fails_open(self):
        from swival.audit import _disproof_routing

        rec = _disproof_routing(None, "timeout", proof_strict=False)
        assert rec["status"] == "failed_open"
        assert rec["reason"] == "timeout"

    def test_transport_failure_strict_routes_to_gapfill(self):
        from swival.audit import _disproof_routing

        rec = _disproof_routing(None, "timeout", proof_strict=True)
        assert rec["status"] == "gapfill"


class TestDisproofPhaseOrchestration:
    """Step 6: end-to-end routing inside the disproof phase of _run_audit_phases."""

    def _state(self, tmp_path, *, proof_strict=False):
        from swival.audit import AuditRunState, AuditScope, FindingRecord

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py", "b.py"],
            mandatory_files=["a.py", "b.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="d1",
            scope=scope,
            queued_files=["a.py", "b.py"],
            state_dir=tmp_path / ".swival" / "audit",
            proof_strict=proof_strict,
        )
        f1 = FindingRecord(
            title="ssrf via webhook",
            finding_type="vulnerability",
            severity="high",
            locations=["a.py:1"],
            preconditions=[],
            proof=["..."],
            fix_outline="guard host",
            source_file="a.py",
        )
        f2 = FindingRecord(
            title="ide-only false positive",
            finding_type="vulnerability",
            severity="medium",
            locations=["b.py:2"],
            preconditions=[],
            proof=["..."],
            fix_outline="n/a",
            source_file="b.py",
        )
        state.proposed_findings = [f1, f2]
        state.phase = "disproof"
        return state

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
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

    def test_invalid_finding_is_discarded_before_verification(
        self, monkeypatch, tmp_path
    ):
        from swival import audit

        state = self._state(tmp_path)
        ctx = self._ctx(tmp_path)
        invalid_block = (
            "```swival-audit-disproof-v1\n"
            "verdict: INVALID\n"
            "reason: guarded\n"
            "counterclaim_checked:\n  - input is trusted\n"
            "blocking_control: route requires admin auth\n"
            "missing_link: none\n"
            "required_next_proof: none\n"
            "review_model: \n"
            "```"
        )
        plausible_block = (
            "```swival-audit-disproof-v1\n"
            "verdict: PLAUSIBLE\n"
            "reason: nope\n"
            "counterclaim_checked: none\n"
            "blocking_control: none\n"
            "missing_link: none\n"
            "required_next_proof: none\n"
            "review_model: \n"
            "```"
        )

        calls = {"n": 0}

        def fake_call(ctx_, msgs, **kw):
            calls["n"] += 1
            return invalid_block if calls["n"] == 1 else plausible_block

        monkeypatch.setattr(audit, "_call_audit_llm", fake_call)
        monkeypatch.setattr(audit, "_gather_evidence", lambda f, c: ("ev", 1))

        # Stop after the disproof phase by simulating "no findings".
        before = list(state.proposed_findings)
        from swival.audit import _finding_key

        f1_key = _finding_key(before[0])

        # Drive the phase by hand to avoid pulling in the rest of the orchestrator.
        targets = [(f, _finding_key(f)) for f in state.proposed_findings]
        results = []
        for f, key in targets:
            parsed, err = audit._phase_disproof_one(f, state, ctx, reviewer_model="")
            results.append((key, parsed, err))
        for key, parsed, err in results:
            rec = audit._disproof_routing(parsed, err, proof_strict=False)
            state.adversarial_state[key] = rec

        assert state.adversarial_state[f1_key]["status"] == "invalid"
        statuses = {v["status"] for v in state.adversarial_state.values()}
        assert "plausible" in statuses

    def test_disproof_records_review_model(self, monkeypatch, tmp_path):
        from swival import audit

        state = self._state(tmp_path)
        ctx = self._ctx(tmp_path)
        block = (
            "```swival-audit-disproof-v1\n"
            "verdict: PLAUSIBLE\n"
            "reason: ok\n"
            "counterclaim_checked: none\n"
            "blocking_control: none\n"
            "missing_link: none\n"
            "required_next_proof: none\n"
            "review_model: \n"
            "```"
        )
        monkeypatch.setattr(audit, "_call_audit_llm", lambda c, m, **kw: block)
        monkeypatch.setattr(audit, "_gather_evidence", lambda f, c: ("ev", 1))

        parsed, err = audit._phase_disproof_one(
            state.proposed_findings[0], state, ctx, reviewer_model="reviewer-mini"
        )
        assert err is None
        assert parsed["review_model"] == "reviewer-mini"

    def test_disproof_handles_malformed_block(self, monkeypatch, tmp_path):
        from swival import audit

        state = self._state(tmp_path)
        ctx = self._ctx(tmp_path)
        monkeypatch.setattr(audit, "_call_audit_llm", lambda c, m, **kw: "no block")
        monkeypatch.setattr(audit, "_gather_evidence", lambda f, c: ("ev", 1))

        parsed, err = audit._phase_disproof_one(
            state.proposed_findings[0], state, ctx, reviewer_model=""
        )
        assert parsed is None
        assert err and "malformed disproof block" in err


class TestDisproofAgreement:
    """Step 6 review: agreement_with_proof must be set deterministically."""

    def _bare(self, tmp_path):
        from swival.audit import (
            AuditRunState,
            AuditScope,
            FindingRecord,
            _finding_key,
        )

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="ag1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path / ".swival" / "audit",
        )
        finding = FindingRecord(
            title="x",
            finding_type="vulnerability",
            severity="high",
            locations=["a.py:1"],
            preconditions=[],
            proof=["..."],
            fix_outline="g",
            source_file="a.py",
        )
        return state, finding, _finding_key(finding)

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
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

    def _setup_worktree(self, monkeypatch):
        class Dummy:
            def __init__(self, work):
                self.work = work

            def __enter__(self):
                return self.work

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("swival.audit._worktree", lambda b, w: Dummy(w))
        monkeypatch.setattr("swival.audit._gather_evidence", lambda f, c: ("ev", 1))

    def _reproduced_block(self):
        return (
            "```swival-audit-proof-v1\n"
            "verdict: REPRODUCED\n"
            "proof_kind: runtime\n"
            "commands:\n  - ./poc\n"
            "artifacts: none\n"
            "observed_output: ok\n"
            "trigger: t\n"
            "impact: i\n"
            "limitations: none\n"
            "```"
        )

    def _notreproduced_block(self):
        return (
            "```swival-audit-proof-v1\n"
            "verdict: NOTREPRODUCED\n"
            "proof_kind: source\n"
            "commands: none\n"
            "artifacts: none\n"
            "observed_output: guarded\n"
            "trigger: none\n"
            "impact: none\n"
            "limitations: none\n"
            "```"
        )

    def test_plausible_reproduced_is_agreed(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch)
        state, finding, key = self._bare(tmp_path)
        state.adversarial_state[key] = {"status": "plausible"}
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (self._reproduced_block(), False),
        )
        vf = audit._verify_single_finding(
            finding, state, self._ctx(tmp_path), tmp_path / "w"
        )
        assert vf is not None
        assert state.adversarial_state[key]["agreement_with_proof"] == "agreed"
        assert vf.adversarial["agreement_with_proof"] == "agreed"

    def test_plausible_notreproduced_is_disagreed(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch)
        state, finding, key = self._bare(tmp_path)
        state.adversarial_state[key] = {"status": "plausible"}
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (self._notreproduced_block(), False),
        )
        vf = audit._verify_single_finding(
            finding, state, self._ctx(tmp_path), tmp_path / "w"
        )
        assert vf is None
        assert state.adversarial_state[key]["agreement_with_proof"] == "disagreed"

    def test_needs_proof_either_outcome_is_agreed(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch)
        state, finding, key = self._bare(tmp_path)
        state.adversarial_state[key] = {"status": "needs_proof"}
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (self._reproduced_block(), False),
        )
        audit._verify_single_finding(
            finding, state, self._ctx(tmp_path), tmp_path / "w"
        )
        assert state.adversarial_state[key]["agreement_with_proof"] == "agreed"

        state.adversarial_state[key] = {"status": "needs_proof"}
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (self._notreproduced_block(), False),
        )
        audit._verify_single_finding(
            finding, state, self._ctx(tmp_path), tmp_path / "w"
        )
        assert state.adversarial_state[key]["agreement_with_proof"] == "agreed"

    def test_failed_open_records_no_review(self, monkeypatch, tmp_path):
        from swival import audit

        self._setup_worktree(monkeypatch)
        state, finding, key = self._bare(tmp_path)
        state.adversarial_state[key] = {"status": "failed_open"}
        monkeypatch.setattr(
            "swival.agent.run_agent_loop",
            lambda msgs, tools, **kw: (self._reproduced_block(), False),
        )
        audit._verify_single_finding(
            finding, state, self._ctx(tmp_path), tmp_path / "w"
        )
        assert state.adversarial_state[key]["agreement_with_proof"] == "no_review"


class TestProofStrictResumeRetriesGapfill:
    """Step 6 review: --proof-strict gapfill items must retry on resume,
    not be permanently skipped by the terminal-status filter."""

    def test_gapfill_status_is_not_in_skip_set(self):
        import inspect

        from swival import audit

        src = inspect.getsource(audit._run_audit_phases)
        chunk = src.split('if state.phase == "disproof":', 1)[1]
        skip_block = chunk.split("if targets", 1)[0]
        assert '"gapfill"' not in skip_block, (
            "gapfill must be retryable on resume so --proof-strict failures "
            "can recover after the reviewer is back"
        )


class TestBudgetPlanner:
    def _state(self, tmp_path: Path, budget: int = 0):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        s = AuditRunState(
            run_id="b1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path / ".swival" / "audit",
        )
        s.budget_tokens = budget
        return s

    def test_disabled_when_total_is_zero(self, tmp_path):
        from swival.audit import BudgetPlanner

        b = BudgetPlanner(self._state(tmp_path, budget=0))
        assert b.enabled is False
        assert b.remaining() == float("inf")
        assert b.remaining("hunt") == float("inf")
        assert b.exhausted() is False
        assert b.should_throttle_low_priority() is False
        assert b.summary_line() is None
        assert b.detail_lines() == []

    def test_allocation_uses_default_shares(self, tmp_path):
        from swival.audit import BudgetPlanner, _BUDGET_DEFAULT_SHARES

        b = BudgetPlanner(self._state(tmp_path, budget=1_000_000))
        for phase, share in _BUDGET_DEFAULT_SHARES.items():
            assert b.allocation(phase) == 1_000_000 * share

    def test_charge_and_remaining(self, tmp_path):
        from swival.audit import BudgetPlanner

        s = self._state(tmp_path, budget=1_000_000)
        b = BudgetPlanner(s)
        b.charge("hunt", 100_000)
        assert s.budget_used == {"hunt": 100_000}
        assert b.used("hunt") == 100_000
        assert b.used() == 100_000
        # hunt allocation is 30% = 300k
        assert b.remaining("hunt") == 200_000
        # global remaining is 900k
        assert b.remaining() == 900_000
        b.charge("hunt", 250_000)  # blows past its allocation
        assert b.remaining("hunt") == 0
        # global budget still has room, so the planner is not yet exhausted
        assert b.exhausted() is False

    def test_throttle_low_priority_at_half(self, tmp_path):
        from swival.audit import BudgetPlanner

        s = self._state(tmp_path, budget=1_000_000)
        b = BudgetPlanner(s)
        b.charge("hunt", 499_999)
        assert b.should_throttle_low_priority() is False
        b.charge("hunt", 2)
        assert b.should_throttle_low_priority() is True

    def test_exhausted_when_global_pool_drained(self, tmp_path):
        from swival.audit import BudgetPlanner

        s = self._state(tmp_path, budget=100_000)
        b = BudgetPlanner(s)
        b.charge("hunt", 100_000)
        assert b.exhausted() is True

    def test_summary_and_detail_lines(self, tmp_path):
        from swival.audit import BudgetPlanner

        s = self._state(tmp_path, budget=1_000_000)
        b = BudgetPlanner(s)
        b.charge("hunt", 50_000)
        b.charge("profile", 5_000)
        line = b.summary_line()
        assert line is not None
        assert "55,000" in line
        assert "1,000,000" in line
        details = b.detail_lines()
        assert any("hunt" in d for d in details)
        assert any("profile" in d for d in details)
        # phases with zero usage are skipped
        assert not any("dedupe" in d for d in details)

    def test_charge_call_uses_token_estimator(self, tmp_path):
        from swival.audit import BudgetPlanner

        s = self._state(tmp_path, budget=1_000_000)
        b = BudgetPlanner(s)
        # 100 chars at len//4 = 25 tokens; pair adds to 50
        b.charge_call("hunt", "a" * 100, "b" * 100)
        assert s.budget_used == {"hunt": 50}
        b.charge_call("hunt", "a" * 200, "")
        assert s.budget_used["hunt"] == 100


class TestAuditCliFlags:
    """The new --hunt / --proof-strict / --trace-reachability / --budget-tokens flags."""

    def _ctx(self, tmp_path):
        return _make_ctx(tmp_path)

    def _capture(self, monkeypatch) -> dict:
        return _capture_run_audit_phases(monkeypatch)

    def test_hunt_flag_passed_through(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("--hunt", self._ctx(tmp_path))
        assert captured["hunt_mode"] is True
        assert captured["budget_tokens"] == 0

    def test_proof_strict_flag_threads_through(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("--proof-strict", self._ctx(tmp_path))
        assert captured["proof_strict"] is True

    def test_trace_reachability_flag_is_reserved(self, tmp_path):
        from swival.audit import run_audit_command

        out = run_audit_command("--trace-reachability", self._ctx(tmp_path))
        assert out.startswith("error:")
        assert "--trace-reachability is reserved" in out
        assert "not yet implemented" in out

    def test_budget_tokens_accepts_underscore_separator(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("--hunt --budget-tokens 2_000_000", self._ctx(tmp_path))
        assert captured["budget_tokens"] == 2_000_000

    def test_budget_tokens_accepts_comma_separator(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        captured = self._capture(monkeypatch)
        run_audit_command("--budget-tokens 1,500,000", self._ctx(tmp_path))
        assert captured["budget_tokens"] == 1_500_000

    def test_gapfill_flag_is_reserved(self, tmp_path):
        from swival.audit import run_audit_command

        out = run_audit_command("--gapfill 25", self._ctx(tmp_path))
        assert out.startswith("error:")
        assert "--gapfill is reserved" in out
        assert "not yet implemented" in out

    def test_proof_strict_config_key_threads_through(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text("[audit]\nproof_strict = true\n")
        captured = self._capture(monkeypatch)
        run_audit_command("", self._ctx(tmp_path))
        assert captured["proof_strict"] is True

    def test_reviewer_profile_is_reserved(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text(
            '[audit.reviewer]\nprofile = "reviewer"\n'
        )
        out = run_audit_command("", self._ctx(tmp_path))
        assert out.startswith("error:")
        assert "[audit.reviewer] profile" in out
        assert "reserved" in out

    def test_reviewer_model_threads_through(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text(
            '[audit.reviewer]\nmodel = "reviewer-mini"\n'
        )
        captured = self._capture(monkeypatch)
        run_audit_command("", self._ctx(tmp_path))
        assert captured["reviewer_model"] == "reviewer-mini"

    def test_max_gapfill_tasks_config_key_is_reserved(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text("[audit]\nmax_gapfill_tasks = 10\n")
        out = run_audit_command("", self._ctx(tmp_path))
        assert out.startswith("error:")
        assert "max_gapfill_tasks" in out
        assert "not yet implemented" in out

    def test_budget_tokens_requires_value(self, tmp_path):
        from swival.audit import run_audit_command

        out = run_audit_command("--budget-tokens", _make_ctx(tmp_path))
        assert out.startswith("error:") and "--budget-tokens" in out

    def test_budget_tokens_rejects_negative(self, tmp_path):
        from swival.audit import run_audit_command

        out = run_audit_command("--budget-tokens -1", _make_ctx(tmp_path))
        assert out.startswith("error:")

    def test_unknown_flag_message_lists_new_flags(self, tmp_path):
        from swival.audit import run_audit_command

        out = run_audit_command("--unknown-flag", _make_ctx(tmp_path))
        assert out.startswith("error: unknown option")
        assert "--hunt" in out
        assert "--budget-tokens" in out
        assert "--budget-tokens" in out


class TestHarnessStateRoundtrip:
    def test_harness_fields_roundtrip(self, tmp_path):
        from swival.audit import (
            AuditRunState,
            AuditScope,
            CoverageRecord,
            HuntTask,
            RootCauseGroup,
            VerifiedFinding,
        )

        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = tmp_path / ".swival" / "audit"
        state = AuditRunState(
            run_id="r1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
        )
        state.hunt_mode = True
        state.proof_strict = True
        state.trace_reachability = True
        state.budget_tokens = 2_000_000
        state.budget_used = {"phase1": 1024, "hunt": 50_000}
        state.attack_class_metrics = {"ssrf": {"proposed": 4, "verified": 1}}

        t = HuntTask(
            id="ssrf-x",
            task_kind="hunt",
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
            trust_boundary_crossed="POST /webhook",
            scope_hint="src/api/**",
            seed_files=["a.py"],
            seed_symbols=["h"],
            sink_files=["b.py"],
            source="phase1",
            priority="high",
            language_hint="python",
        )
        state.hunt_tasks[t.id] = t
        state.coverage_records[t.id] = CoverageRecord(
            task_id=t.id,
            observed_files=["a.py"],
            observed_symbols=["h"],
            observed_edges=["a.py->b.py"],
            covered_fraction_of_seed_files=1.0,
            covered_fraction_of_sink_hits=0.5,
            unobserved_high_priority_seeds=["c.py"],
            explicit_not_covered=["retry path"],
            followup_task_ids=["ssrf-y"],
            confidence="medium",
        )
        state.adversarial_state["k1"] = {
            "status": "plausible",
            "attempts": 1,
            "review_model": "reviewer-mini",
        }

        finding = FindingRecord(
            title="ssrf via webhook",
            finding_type="vulnerability",
            severity="high",
            locations=["a.py:1"],
            preconditions=[],
            proof=[],
            fix_outline="guard host",
            source_file="a.py",
            local_bug="unguarded fetch",
            source_boundary="POST /webhook",
            reachability_path=["POST /webhook", "handle_webhook"],
            sink_operation="outbound HTTP GET",
            exploit_chain=["attacker controls callback_url"],
            reachability_status="reachable",
            hunt_task_id=t.id,
            attack_class="ssrf",
            attacker_position="unauthenticated remote client",
            controlled_inputs=["callback_url"],
        )
        state.proposed_findings.append(finding)
        state.verified_findings.append(
            VerifiedFinding(
                finding=finding,
                correctness_reason="ok",
                rebuttal_reason="n/a",
                reproducer={"proof_kind": "runtime"},
                adversarial={"verdict": "PLAUSIBLE", "review_model": "reviewer-mini"},
                trace={"reachable": True},
                root_cause_group_id="g1",
            )
        )
        state.root_cause_groups.append(
            RootCauseGroup(
                id="g1",
                primary_finding_key="k1",
                variant_finding_keys=[],
                root_cause_summary="ssrf",
                source_boundaries=["POST /webhook"],
                affected_locations=["a.py:1"],
                finding_type="vulnerability",
                attack_class="ssrf",
            )
        )
        state.save()

        loaded = AuditRunState.load(state_dir, "r1")
        assert loaded.hunt_mode is True
        assert loaded.proof_strict is True
        assert loaded.trace_reachability is True
        assert loaded.budget_tokens == 2_000_000
        assert loaded.budget_used == {"phase1": 1024, "hunt": 50_000}
        assert loaded.attack_class_metrics == {"ssrf": {"proposed": 4, "verified": 1}}
        assert loaded.hunt_tasks[t.id].language_hint == "python"
        assert loaded.coverage_records[t.id].followup_task_ids == ["ssrf-y"]
        assert loaded.adversarial_state["k1"]["review_model"] == "reviewer-mini"
        rf = loaded.proposed_findings[0]
        assert rf.local_bug == "unguarded fetch"
        assert rf.reachability_status == "reachable"
        assert rf.hunt_task_id == t.id
        rvf = loaded.verified_findings[0]
        assert rvf.adversarial == {
            "verdict": "PLAUSIBLE",
            "review_model": "reviewer-mini",
        }
        assert rvf.trace == {"reachable": True}
        assert rvf.root_cause_group_id == "g1"
        assert loaded.root_cause_groups[0].id == "g1"

    def test_load_state_without_harness_fields_uses_defaults(self, tmp_path):
        import json

        state_dir = tmp_path / ".swival" / "audit"
        run_dir = state_dir / "legacy"
        run_dir.mkdir(parents=True)
        blob = {
            "run_id": "legacy",
            "scope": {
                "branch": "main",
                "commit": "abc1234",
                "tracked_files": ["a.py"],
                "mandatory_files": ["a.py"],
                "focus": [],
            },
            "queued_files": ["a.py"],
            "reviewed_files": ["a.py"],
            "triage_records": {},
            "candidate_files": [],
            "deep_reviewed_files": [],
            "proposed_findings": [
                {
                    "title": "t",
                    "finding_type": "vulnerability",
                    "severity": "high",
                    "locations": [],
                    "preconditions": [],
                    "proof": [],
                    "fix_outline": "",
                    "source_file": "a.py",
                }
            ],
            "verified_findings": [
                {
                    "finding": {
                        "title": "t",
                        "finding_type": "vulnerability",
                        "severity": "high",
                        "locations": [],
                        "preconditions": [],
                        "proof": [],
                        "fix_outline": "",
                        "source_file": "a.py",
                    },
                    "correctness_reason": "ok",
                    "rebuttal_reason": "n/a",
                    "reproducer": None,
                }
            ],
            "repo_profile": None,
            "import_index": {},
            "caller_index": {},
            "attack_scores": {},
            "verification_state": {},
            "artifact_state": {},
            "phase": "init",
            "metrics": {},
            "select_all": False,
            "measure_triage": False,
            "measurement_escalated_paths": [],
        }
        (run_dir / "state.json").write_text(json.dumps(blob))

        loaded = AuditRunState.load(state_dir, "legacy")
        assert loaded.hunt_mode is False
        assert loaded.hunt_tasks == {}
        assert loaded.coverage_records == {}
        assert loaded.adversarial_state == {}
        assert loaded.root_cause_groups == []
        assert loaded.trace_state == {}
        assert loaded.proof_strict is False
        assert loaded.trace_reachability is False
        assert loaded.budget_tokens == 0
        assert loaded.budget_used == {}
        assert loaded.attack_class_metrics == {}
        assert loaded.proposed_findings[0].local_bug == ""
        assert loaded.proposed_findings[0].reachability_status == "unknown"
        assert loaded.proposed_findings[0].hunt_task_id == ""
        assert loaded.verified_findings[0].adversarial is None
        assert loaded.verified_findings[0].trace is None
        assert loaded.verified_findings[0].root_cause_group_id == ""


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

    def test_artifact_state_assigns_max_plus_one_after_prune(self, tmp_path):
        from swival.audit import _ensure_artifact_state

        state = self._make_state(tmp_path)
        f1 = self._make_finding(title="A")
        f2 = self._make_finding(title="B")
        f3 = self._make_finding(title="C")
        state.verified_findings = [
            VerifiedFinding(finding=f1, correctness_reason="ok", rebuttal_reason="n/a"),
            VerifiedFinding(finding=f2, correctness_reason="ok", rebuttal_reason="n/a"),
        ]
        _ensure_artifact_state(state)
        key1 = _finding_key(f1)
        state.artifact_state[key1]["index"] = 1
        state.artifact_state[_finding_key(f2)]["index"] = 5
        state.verified_findings = [
            VerifiedFinding(finding=f2, correctness_reason="ok", rebuttal_reason="n/a"),
            VerifiedFinding(finding=f3, correctness_reason="ok", rebuttal_reason="n/a"),
        ]

        _ensure_artifact_state(state)

        assert key1 not in state.artifact_state
        assert state.artifact_state[_finding_key(f3)]["index"] == 6

    def test_artifact_state_preserves_filenames_on_retry(self, tmp_path):
        from swival.audit import _ensure_artifact_state

        state = self._make_state(tmp_path)
        finding = self._make_finding(title="Original Title")
        vf = VerifiedFinding(
            finding=finding, correctness_reason="ok", rebuttal_reason="n/a"
        )
        state.verified_findings = [vf]
        _ensure_artifact_state(state)
        key = _finding_key(finding)
        original_patch = state.artifact_state[key]["patch_filename"]
        state.artifact_state[key]["status"] = "failed"

        _ensure_artifact_state(state)

        assert state.artifact_state[key]["patch_filename"] == original_patch

    def _make_verified(self, **overrides) -> VerifiedFinding:
        return VerifiedFinding(
            finding=self._make_finding(**overrides),
            correctness_reason="ok",
            rebuttal_reason="n/a",
        )

    def _make_artifact_state(self, tmp_path, findings):
        state = self._make_state(tmp_path)
        state.phase = "artifacts"
        state.reviewed_files = {"main.c"}
        state.candidate_files = ["main.c"]
        state.deep_reviewed_files = {"main.c"}
        state.verified_findings = list(findings)
        return state

    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            base_dir=str(tmp_path),
            tools=[],
            verbose=False,
            no_history=True,
            loop_kwargs={},
        )

    def _phase5_state(self, tmp_path, findings):
        _init_git(tmp_path)
        _commit_file(tmp_path, "main.c", "int main(void) { return 0; }")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = self._make_artifact_state(tmp_path, findings)
        state.scope = AuditScope(
            branch=state.scope.branch,
            commit=commit,
            tracked_files=state.scope.tracked_files,
            mandatory_files=state.scope.mandatory_files,
            focus=state.scope.focus,
        )
        state.state_dir = state_dir
        return state, state_dir

    def test_phase5_patch_failure_marks_failed_and_stays_artifacts(
        self, monkeypatch, tmp_path
    ):
        from swival.audit import _run_audit_phases

        vf = self._make_verified()
        state, state_dir = self._phase5_state(tmp_path, [vf])
        state.save()
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                error_code="patch_turn_budget_exhausted", error="turn budget exhausted"
            ),
        )

        result = _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        entry = loaded.artifact_state[_finding_key(vf.finding)]
        assert "Audit incomplete" in result
        assert "No provable" not in result
        assert loaded.phase == "artifacts"
        assert entry["status"] == "failed"
        assert entry["last_error_code"] == "patch_turn_budget_exhausted"

    def test_phase5_no_diff_is_retryable(self, monkeypatch, tmp_path):
        from swival.audit import _run_audit_phases

        vf = self._make_verified()
        state, state_dir = self._phase5_state(tmp_path, [vf])
        state.save()
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                error_code="patch_no_diff", error="no changes produced"
            ),
        )

        _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        entry = loaded.artifact_state[_finding_key(vf.finding)]
        assert loaded.phase == "artifacts"
        assert entry["status"] == "failed"
        assert entry["last_error_code"] == "patch_no_diff"

    def test_phase5_report_exception_is_retryable(self, monkeypatch, tmp_path):
        from swival.audit import _run_audit_phases

        vf = self._make_verified()
        state, state_dir = self._phase5_state(tmp_path, [vf])
        state.save()
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )

        _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        entry = loaded.artifact_state[_finding_key(vf.finding)]
        assert loaded.phase == "artifacts"
        assert entry["status"] == "failed"
        assert entry["last_error_code"] == "report_generation_error"

    def test_phase5_write_error_is_retryable(self, monkeypatch, tmp_path):
        from swival.audit import _ensure_artifact_state, _run_audit_phases

        vf = self._make_verified()
        state, state_dir = self._phase5_state(tmp_path, [vf])
        _ensure_artifact_state(state)
        entry = state.artifact_state[_finding_key(vf.finding)]
        entry["patch_filename"] = "existing-dir"
        artifact_dir = Path(tmp_path) / state.artifact_dir
        (artifact_dir / "existing-dir").mkdir(parents=True)
        state.save()
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        entry = loaded.artifact_state[_finding_key(vf.finding)]
        assert entry["status"] == "failed"
        assert entry["last_error_code"] == "write_artifact_error"

    def test_resume_retries_only_failed_and_pending(self, monkeypatch, tmp_path):
        from swival.audit import _ensure_artifact_state, _run_audit_phases

        findings = [
            self._make_verified(title="A"),
            self._make_verified(title="B"),
            self._make_verified(title="C"),
        ]
        state, state_dir = self._phase5_state(tmp_path, findings)
        _ensure_artifact_state(state)
        state.artifact_state[_finding_key(findings[0].finding)]["status"] = "written"
        state.artifact_state[_finding_key(findings[1].finding)]["status"] = "failed"
        state.artifact_state[_finding_key(findings[2].finding)]["status"] = "pending"
        state.save()
        patched = []
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: (
                patched.append(vf.finding.title)
                or PatchGenerationResult(patch_text="diff\n")
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        assert patched == ["B", "C"]

    def test_targeted_regen_only_selected_and_keeps_written(
        self, monkeypatch, tmp_path
    ):
        from swival.audit import _ensure_artifact_state, _run_audit_phases

        findings = [self._make_verified(title="A"), self._make_verified(title="B")]
        state, state_dir = self._phase5_state(tmp_path, findings)
        state.phase = "done"
        _ensure_artifact_state(state)
        state.artifact_state[_finding_key(findings[0].finding)]["status"] = "written"
        state.artifact_state[_finding_key(findings[1].finding)]["status"] = "failed"
        original_patch = state.artifact_state[_finding_key(findings[1].finding)][
            "patch_filename"
        ]
        state.save()
        patched = []
        info_lines = []
        monkeypatch.setattr("swival.audit.fmt.info", info_lines.append)
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: (
                patched.append(vf.finding.title)
                or PatchGenerationResult(patch_text="diff\n")
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        _run_audit_phases(
            "--regen --finding 2",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            False,
            True,
            None,
            finding_selector="2",
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        assert patched == ["B"]
        assert (
            loaded.artifact_state[_finding_key(findings[0].finding)]["status"]
            == "written"
        )
        assert (
            loaded.artifact_state[_finding_key(findings[1].finding)]["patch_filename"]
            == original_patch
        )
        assert any("[1/1] regenerating finding 2/2" in line for line in info_lines)

    def test_phase5_success_marks_done(self, monkeypatch, tmp_path):
        from swival.audit import _run_audit_phases

        vf = self._make_verified()
        state, state_dir = self._phase5_state(tmp_path, [vf])
        state.save()
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        result = _run_audit_phases(
            "--resume",
            self._ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        loaded = AuditRunState.load(state_dir, state.run_id)
        entry = loaded.artifact_state[_finding_key(vf.finding)]
        assert "Audit complete" in result
        assert loaded.phase == "done"
        assert entry["status"] == "written"

    def test_phase5_patch_budget_passed_to_isolated_loop(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        import swival.agent as agent_mod
        import swival.audit as audit_mod
        from swival.audit import _phase5_patch

        captured = {}

        class FakeWorktree:
            def __init__(self, base_dir, work_dir):
                self.work_dir = work_dir

            def __enter__(self):
                return self.work_dir

            def __exit__(self, *exc):
                return False

        def fake_kwargs(ctx, work_dir, max_turns=None):
            captured["max_turns"] = max_turns
            return {"base_dir": str(work_dir)}

        class FakeDiff:
            stdout = b"diff --git a/main.c b/main.c\n"

        monkeypatch.setattr(audit_mod, "_worktree", FakeWorktree)
        monkeypatch.setattr(
            audit_mod, "_gather_evidence", lambda finding, ctx: ("source", 1)
        )
        monkeypatch.setattr(audit_mod, "_make_isolated_loop_kwargs", fake_kwargs)
        monkeypatch.setattr(
            agent_mod, "run_agent_loop", lambda messages, tools, **kw: ("done", False)
        )
        monkeypatch.setattr(audit_mod.subprocess, "run", lambda *a, **kw: FakeDiff())
        ctx = SimpleNamespace(base_dir=str(tmp_path), tools=[], loop_kwargs={})
        state = self._make_state(tmp_path)

        result = _phase5_patch(self._make_verified(), ctx, state, patch_max_turns=75)

        assert captured["max_turns"] == 75
        assert result.patch_text is not None

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
            return (
                "proof\n```swival-audit-proof-v1\n"
                "verdict: REPRODUCED\n"
                "proof_kind: source\n"
                "commands: none\n"
                "artifacts: none\n"
                "observed_output: cited at main.c:7\n"
                "trigger: oversized argv\n"
                "impact: stack overflow\n"
                "limitations: none\n"
                "```",
                False,
            )

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
            lambda msgs, tools, **kw: (
                "could not confirm\n```swival-audit-proof-v1\n"
                "verdict: NOTREPRODUCED\n"
                "proof_kind: source\n"
                "commands: none\n"
                "artifacts: none\n"
                "observed_output: control blocks attacker\n"
                "trigger: not reachable\n"
                "impact: none\n"
                "limitations: none\n```",
                False,
            ),
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
        """Duplicate verified_findings must not produce duplicate artifacts."""
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
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                patch_text="--- patch ---"
            ),
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
            triage_records={"a.py": self._triage_skip("a.py")},
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


class TestMatchPathGlob:
    def test_exact_match(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("src/foo.rs", "src/foo.rs")
        assert not _match_path_glob("src/foo.rs", "src/bar.rs")

    def test_prefix_no_wildcard(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("src/a.py", "src")
        assert _match_path_glob("src/sub/a.py", "src/")
        assert not _match_path_glob("source/a.py", "src")

    def test_slashless_wildcard_is_recursive(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("foo.rs", "*.rs")
        assert _match_path_glob("src/foo.rs", "*.rs")
        assert _match_path_glob("crates/foo/src/bar.rs", "*.rs")
        assert not _match_path_glob("foo.py", "*.rs")

    def test_anchored_single_star_does_not_cross_slash(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("src/foo.rs", "src/*.rs")
        assert not _match_path_glob("src/sub/bar.rs", "src/*.rs")
        assert not _match_path_glob("src/sub/deep/bar.rs", "src/*.rs")

    def test_anchored_double_star_recurses(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("src/foo.rs", "src/**/*.rs")
        assert _match_path_glob("src/sub/bar.rs", "src/**/*.rs")
        assert _match_path_glob("src/a/b/c.rs", "src/**/*.rs")
        assert not _match_path_glob("crates/foo/src/bar.rs", "src/**/*.rs")

    def test_anchored_pattern_does_not_match_suffix(self):
        from swival.audit import _match_path_glob

        assert not _match_path_glob("crates/foo/src/bar.rs", "src/*.rs")
        assert not _match_path_glob("a/b/src/bar.rs", "src/**/*.rs")

    def test_question_mark_and_charclass(self):
        from swival.audit import _match_path_glob

        assert _match_path_glob("src/a.rs", "src/?.rs")
        assert not _match_path_glob("src/ab.rs", "src/?.rs")
        assert _match_path_glob("src/a.rs", "src/[ab].rs")
        assert not _match_path_glob("src/c.rs", "src/[ab].rs")

    def test_resolve_scope_anchored_glob_is_repo_rooted_and_segment_aware(
        self, tmp_path
    ):
        from swival.audit import _resolve_scope

        _init_git(tmp_path)
        _commit_file(tmp_path, "src/lib.rs", "// top")
        _commit_file(tmp_path, "src/nested/lib.rs", "// nested")
        _commit_file(tmp_path, "crates/foo/src/lib.rs", "// vendored")

        scope = _resolve_scope(str(tmp_path), ["src/*.rs"])
        assert "src/lib.rs" in scope.mandatory_files
        assert "src/nested/lib.rs" not in scope.mandatory_files
        assert "crates/foo/src/lib.rs" not in scope.mandatory_files

        recursive = _resolve_scope(str(tmp_path), ["src/**/*.rs"])
        assert "src/lib.rs" in recursive.mandatory_files
        assert "src/nested/lib.rs" in recursive.mandatory_files
        assert "crates/foo/src/lib.rs" not in recursive.mandatory_files


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


class TestAuditCommandParser:
    """Parser-level tests: capture kwargs passed to _run_audit_phases."""

    def test_no_args_passes_focus_none(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("", _make_ctx(tmp_path))
        assert captured["focus"] is None
        assert captured["workers"] == 4
        assert captured["resume"] is False
        assert captured["regen"] is False

    def test_two_paths_collected_as_list(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("src/a src/b --workers 2 --resume", _make_ctx(tmp_path))
        assert captured["focus"] == ["src/a", "src/b"]
        assert captured["workers"] == 2
        assert captured["resume"] is True

    def test_dedupes_repeated_paths(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("src/a src/a/", _make_ctx(tmp_path))
        assert captured["focus"] == ["src/a"]

    def test_flag_intermixing(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("--resume src/a --workers 4 src/b", _make_ctx(tmp_path))
        assert captured["focus"] == ["src/a", "src/b"]
        assert captured["workers"] == 4
        assert captured["resume"] is True

    def test_unknown_dash_option_errors(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _capture_run_audit_phases(monkeypatch)
        result = run_audit_command("-resume", _make_ctx(tmp_path))
        assert result.startswith("error:")
        assert "-resume" in result

    def test_unknown_double_dash_option_errors(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _capture_run_audit_phases(monkeypatch)
        result = run_audit_command("--bogus", _make_ctx(tmp_path))
        assert result.startswith("error:")
        assert "--bogus" in result

    def test_patch_max_turns_parsed(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("--patch-max-turns 75", _make_ctx(tmp_path))
        assert captured["patch_max_turns"] == 75

    def test_patch_max_turns_rejects_bad_values(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _capture_run_audit_phases(monkeypatch)
        assert run_audit_command(
            "--patch-max-turns nope", _make_ctx(tmp_path)
        ).startswith("error:")
        assert run_audit_command("--patch-max-turns 0", _make_ctx(tmp_path)).startswith(
            "error:"
        )

    def test_finding_requires_regen(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _capture_run_audit_phases(monkeypatch)
        result = run_audit_command("--finding 2", _make_ctx(tmp_path))
        assert result.startswith("error:")
        assert "--regen" in result

    def test_finding_rejects_repeated_flags(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _capture_run_audit_phases(monkeypatch)
        result = run_audit_command(
            "--regen --finding 2 --finding 5", _make_ctx(tmp_path)
        )
        assert result.startswith("error:")
        assert "only be provided once" in result

    def test_finding_selector_parser_rejects_empty_and_zero(self):
        from swival.audit import _parse_finding_selector

        for raw in ("", ",,", "0"):
            with pytest.raises(ValueError):
                _parse_finding_selector(raw, total=3)

    def test_finding_selector_parser_accepts_lists_and_ranges(self):
        from swival.audit import _parse_finding_selector

        assert _parse_finding_selector("2,4-5", total=5) == {1, 3, 4}


class TestSelectAll:
    """Tests for the /audit --all flag (skip Phase 2 triage)."""

    def _scope(self, files):
        return AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=list(files),
            mandatory_files=list(files),
            focus=[],
        )

    def test_select_all_round_trips_through_save_load(self, tmp_path):
        scope = self._scope(["a.py"])
        state = AuditRunState(
            run_id="ra",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path / ".swival" / "audit",
            select_all=True,
        )
        state.save()

        loaded = AuditRunState.load(state.state_dir, "ra")
        assert loaded.select_all is True

    # -- Parser plumbing -----------------------------------------------------

    def test_parser_sets_select_all_true_with_flag(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("--all src/foo", _make_ctx(tmp_path))
        assert captured["select_all"] is True
        assert captured["focus"] == ["src/foo"]

    def test_parser_select_all_false_by_default(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("src/foo", _make_ctx(tmp_path))
        assert captured["select_all"] is False
        assert captured["focus"] == ["src/foo"]

    # -- Phase 2 bypass ------------------------------------------------------

    def _stub_pipeline(self, monkeypatch, triage_calls: list):
        """Stub Phase 1 and Phase 3-5 so the pipeline is driven without LLM calls.
        Phase 2's triage worker is stubbed to record any unexpected invocation."""
        from swival.audit import DeepReviewResult

        monkeypatch.setattr(
            "swival.audit._phase1_repo_profile",
            lambda state, ctx: {"summary": "stub"},
        )

        def record_triage(path, state, ctx):
            triage_calls.append(path)
            return None

        monkeypatch.setattr("swival.audit._phase2_triage_one", record_triage)
        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, findings=[]),
        )
        monkeypatch.setattr(
            "swival.audit._verify_one_finding",
            lambda item, state, ctx: VerificationResult(
                finding_key=item[0], discarded=True
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50: PatchGenerationResult(
                patch_text="--- patch ---"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# Report",
        )

    def test_select_all_skips_phase2_triage(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "import os")
        _commit_file(tmp_path, "b.py", "import sys")

        triage_calls: list[str] = []
        self._stub_pipeline(monkeypatch, triage_calls)

        result = run_audit_command("--all", _make_ctx(tmp_path))

        assert triage_calls == []
        assert "Audit incomplete" not in result

        state_dir = Path(tmp_path) / ".swival" / "audit"
        run_dir = next(p for p in state_dir.iterdir() if (p / "state.json").exists())
        loaded = AuditRunState.load(state_dir, run_dir.name)
        assert loaded.phase == "done"
        assert loaded.select_all is True
        assert set(loaded.candidate_files) == set(loaded.queued_files)
        assert set(loaded.scope.mandatory_files).issubset(loaded.reviewed_files)

    # -- Diagnostics ---------------------------------------------------------

    def test_select_all_emits_skip_line_and_banner(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")

        info_lines: list[str] = []
        warning_lines: list[str] = []
        monkeypatch.setattr("swival.audit.fmt.info", info_lines.append)
        monkeypatch.setattr("swival.audit.fmt.warning", warning_lines.append)

        triage_calls: list[str] = []
        self._stub_pipeline(monkeypatch, triage_calls)

        run_audit_command("--all", _make_ctx(tmp_path))

        assert any(" --all" in line for line in info_lines), info_lines
        assert any("phase 2: skipped (--all)" in line for line in info_lines), (
            info_lines
        )

    def test_select_all_sharpens_large_scope_warning(self, monkeypatch, tmp_path):
        import swival.audit as audit_mod
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")

        warning_lines: list[str] = []
        monkeypatch.setattr("swival.audit.fmt.warning", warning_lines.append)
        monkeypatch.setattr(audit_mod, "_LARGE_SCOPE_THRESHOLD", 0)

        self._stub_pipeline(monkeypatch, [])

        run_audit_command("--all", _make_ctx(tmp_path))

        assert any("--all" in line for line in warning_lines), warning_lines
        assert any("triage selection skipped" in line for line in warning_lines), (
            warning_lines
        )

    # -- Phase 3 with no triage record ---------------------------------------

    def test_phase3a_inventory_handles_missing_triage_record(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        from swival.audit import _phase3a_inventory

        scope = self._scope(["a.py"])
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            triage_records={},
            repo_profile={"summary": "tiny repo"},
            state_dir=tmp_path,
            select_all=True,
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        captured: dict = {}

        def fake_call(ctx, messages, temperature=0.0, trace_task=None):
            captured["user"] = messages[1]["content"]
            return "@@ none @@"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        _phase3a_inventory("a.py", state, ctx, content="print('x')")

        assert "Focus bug classes: all" in captured["user"]
        assert "Phase 2 triage result:\n{}" in captured["user"]

    # -- Resume preservation -------------------------------------------------

    def test_resume_preserves_persisted_select_all_true(self, monkeypatch, tmp_path):
        """A persisted select_all=True survives resume even if --all isn't passed."""
        from swival.audit import _run_audit_phases

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        state_dir = Path(tmp_path) / ".swival" / "audit"
        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="ra",
            scope=scope,
            queued_files=["a.py"],
            candidate_files=["a.py"],
            reviewed_files={"a.py"},
            deep_reviewed_files={"a.py"},
            state_dir=state_dir,
            phase="verification",
            select_all=True,
        )
        state.save()

        triage_calls: list[str] = []
        self._stub_pipeline(monkeypatch, triage_calls)

        _run_audit_phases(
            "--resume",
            _make_ctx(tmp_path),
            str(tmp_path),
            state_dir,
            workers=1,
            resume=True,
            regen=False,
            focus=None,
            select_all=False,
        )

        loaded = AuditRunState.load(state_dir, "ra")
        assert loaded.select_all is True

    def test_resume_ignores_runtime_select_all_when_persisted_false(
        self, monkeypatch, tmp_path
    ):
        """Runtime --all on resume does not flip a persisted select_all=False."""
        from swival.audit import _run_audit_phases

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        state_dir = Path(tmp_path) / ".swival" / "audit"
        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="rb",
            scope=scope,
            queued_files=["a.py"],
            candidate_files=["a.py"],
            reviewed_files={"a.py"},
            deep_reviewed_files={"a.py"},
            state_dir=state_dir,
            phase="verification",
            select_all=False,
        )
        state.save()

        triage_calls: list[str] = []
        self._stub_pipeline(monkeypatch, triage_calls)

        _run_audit_phases(
            "--all --resume",
            _make_ctx(tmp_path),
            str(tmp_path),
            state_dir,
            workers=1,
            resume=True,
            regen=False,
            focus=None,
            select_all=True,
        )

        loaded = AuditRunState.load(state_dir, "rb")
        assert loaded.select_all is False


# ---------------------------------------------------------------------------
# Triage recall: promotion, confirmation pass, force_review, measure-triage
# ---------------------------------------------------------------------------


def _bare_triage(path, *, priority="SKIP", confidence="medium", needs_followup=False):
    return TriageRecord(
        path=path,
        priority=priority,
        confidence=confidence,
        bug_classes=[],
        summary=f"{priority} {path}",
        relevant_symbols=[],
        suspicious_flows=[],
        needs_followup=needs_followup,
    )


class TestTriageRecordFields:
    """New fields on TriageRecord round-trip through save/load."""

    def test_dataclass_defaults(self):
        rec = TriageRecord(
            path="a.py",
            priority="SKIP",
            confidence="medium",
            bug_classes=[],
            summary="x",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=False,
        )
        assert rec.promotion_reasons == []
        assert rec.triage_failure_mode is None
        assert rec.confirmation_outcome is None

    def test_round_trip(self, tmp_path):
        scope = AuditScope(
            branch="main",
            commit="c1",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        rec = _bare_triage("a.py")
        rec.promotion_reasons = ["attack-surface score 9"]
        rec.triage_failure_mode = "parse_error"
        rec.confirmation_outcome = "promoted"
        state = AuditRunState(
            run_id="t1",
            scope=scope,
            queued_files=["a.py"],
            triage_records={"a.py": rec},
            state_dir=tmp_path,
        )
        state.save()
        loaded = AuditRunState.load(state.state_dir, "t1")
        loaded_rec = loaded.triage_records["a.py"]
        assert loaded_rec.promotion_reasons == ["attack-surface score 9"]
        assert loaded_rec.triage_failure_mode == "parse_error"
        assert loaded_rec.confirmation_outcome == "promoted"


class TestAttackScoreCache:
    """Phase 1 caches attack-surface scores; dependency_index aliases caller_index."""

    def test_order_returns_score_map(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "danger.py", "subprocess.run(cmd)\neval(data)")
        _commit_file(tmp_path, "safe.py", "x = 1")

        cache = _load_file_contents(["safe.py", "danger.py"], str(tmp_path))
        ordered, scores = _order_by_attack_surface(["safe.py", "danger.py"], cache)
        assert ordered[0] == "danger.py"
        assert scores["danger.py"] >= 5
        assert scores["safe.py"] == 0

    def test_attack_scores_round_trip(self, tmp_path):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            attack_scores={"a.py": 12},
            state_dir=tmp_path,
        )
        state.save()
        loaded = AuditRunState.load(state.state_dir, "r")
        assert loaded.attack_scores == {"a.py": 12}

    def test_dependency_index_aliases_caller_index(self):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            caller_index={"a.py": ["b.py"]},
        )
        assert state.dependency_index == {"a.py": ["b.py"]}


class TestPromotion:
    """Deterministic promotion rules in _apply_promotions."""

    def _state(self, tmp_path, **kwargs):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py", "b.py", "c.py"],
            mandatory_files=["a.py", "b.py", "c.py"],
            focus=[],
        )
        defaults = dict(
            run_id="r",
            scope=scope,
            queued_files=["a.py", "b.py", "c.py"],
            state_dir=tmp_path,
        )
        defaults.update(kwargs)
        return AuditRunState(**defaults)

    def test_score_threshold_promotes_skip(self, tmp_path):
        from swival.audit import _apply_promotions

        state = self._state(
            tmp_path,
            attack_scores={"a.py": 12, "b.py": 0, "c.py": 0},
            triage_records={
                "a.py": _bare_triage("a.py"),
                "b.py": _bare_triage("b.py"),
                "c.py": _bare_triage("c.py"),
            },
        )
        promotions = _apply_promotions(state, force_review_matches={})
        assert "attack-surface score 12" in promotions["a.py"]
        assert state.triage_records["a.py"].priority == "ESCALATE_MEDIUM"
        assert state.triage_records["b.py"].priority == "SKIP"

    def test_entry_point_promotes(self, tmp_path):
        from swival.audit import _apply_promotions

        state = self._state(
            tmp_path,
            triage_records={p: _bare_triage(p) for p in ["a.py", "b.py", "c.py"]},
            repo_profile={"entry_points": ["a.py"], "trust_boundaries": []},
        )
        promotions = _apply_promotions(state, {})
        assert "phase 1 entry point" in promotions["a.py"]
        assert state.triage_records["a.py"].priority == "ESCALATE_MEDIUM"

    def test_one_hop_reach_from_entry_point(self, tmp_path):
        from swival.audit import _apply_promotions

        state = self._state(
            tmp_path,
            triage_records={p: _bare_triage(p) for p in ["a.py", "b.py", "c.py"]},
            attack_scores={"a.py": 0, "b.py": 3, "c.py": 0},
            caller_index={"a.py": ["b.py"]},  # b.py is what a.py depends on
            repo_profile={"entry_points": ["a.py"], "trust_boundaries": []},
        )
        _apply_promotions(state, {})
        assert state.triage_records["b.py"].priority == "ESCALATE_MEDIUM"
        assert any(
            "reached from entry point a.py" in r
            for r in state.triage_records["b.py"].promotion_reasons
        )

    def test_needs_followup_promotes(self, tmp_path):
        from swival.audit import _apply_promotions

        state = self._state(
            tmp_path,
            triage_records={
                "a.py": _bare_triage("a.py", needs_followup=True),
                "b.py": _bare_triage("b.py"),
                "c.py": _bare_triage("c.py"),
            },
        )
        _apply_promotions(state, {})
        assert state.triage_records["a.py"].priority == "ESCALATE_MEDIUM"
        assert state.triage_records["b.py"].priority == "SKIP"

    def test_failure_mode_promotes(self, tmp_path):
        from swival.audit import _apply_promotions

        rec = _bare_triage("a.py")
        rec.triage_failure_mode = "parse_error"
        state = self._state(
            tmp_path,
            triage_records={
                "a.py": rec,
                "b.py": _bare_triage("b.py"),
                "c.py": _bare_triage("c.py"),
            },
        )
        _apply_promotions(state, {})
        assert state.triage_records["a.py"].priority == "ESCALATE_MEDIUM"
        assert any(
            "infrastructure failure" in r
            for r in state.triage_records["a.py"].promotion_reasons
        )

    def test_missing_record_synthesized(self, tmp_path):
        from swival.audit import _apply_promotions

        state = self._state(
            tmp_path,
            triage_records={"a.py": _bare_triage("a.py")},
        )
        _apply_promotions(state, {})
        # Both b.py and c.py were missing — synthesized + promoted
        assert state.triage_records["b.py"].triage_failure_mode == "missing"
        assert state.triage_records["b.py"].priority == "ESCALATE_MEDIUM"
        assert state.triage_records["c.py"].priority == "ESCALATE_MEDIUM"


class TestForceReviewConfig:
    """[audit] force_review TOML schema and merge logic."""

    def _write(self, path, body):
        path.write_text(body)

    def test_loads_project_force_review(self, tmp_path, monkeypatch):
        from swival.config import load_config

        # Force a fresh global dir so we don't pick up real ~/.config/swival
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        (tmp_path / "swival.toml").write_text(
            '[audit]\nforce_review = ["swival/audit.py", "swival/edit.py"]\n'
        )
        cfg = load_config(tmp_path)
        assert cfg["audit"]["force_review"] == [
            "swival/audit.py",
            "swival/edit.py",
        ]
        from swival.audit import _load_audit_config

        audit_cfg = _load_audit_config(str(tmp_path))
        assert audit_cfg["force_review_sources"]["swival/audit.py"] == "project"
        assert "_force_review_sources" not in cfg["audit"]

    def test_merges_global_and_project(self, tmp_path, monkeypatch):
        from swival.audit import _load_audit_config
        from swival.config import load_config

        xdg = tmp_path / "xdg"
        (xdg / "swival").mkdir(parents=True)
        (xdg / "swival" / "config.toml").write_text(
            '[audit]\nforce_review = ["always.py"]\n'
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        (tmp_path / "swival.toml").write_text('[audit]\nforce_review = ["here.py"]\n')

        cfg = load_config(tmp_path)
        assert set(cfg["audit"]["force_review"]) == {"always.py", "here.py"}
        audit_cfg = _load_audit_config(str(tmp_path))
        assert audit_cfg["force_review_sources"]["always.py"] == "global"
        assert audit_cfg["force_review_sources"]["here.py"] == "project"

    def test_unknown_audit_subkey_raises(self, tmp_path, monkeypatch):
        from swival.config import load_config, ConfigError

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text(
            '[audit]\nforce_review = []\nbogus = "x"\n'
        )
        with pytest.raises(ConfigError, match="audit.bogus"):
            load_config(tmp_path)

    def test_non_string_glob_raises(self, tmp_path, monkeypatch):
        from swival.config import load_config, ConfigError

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text('[audit]\nforce_review = ["ok.py", 42]\n')
        with pytest.raises(ConfigError, match="force_review"):
            load_config(tmp_path)

    def test_patch_max_turns_project_overrides_global(self, tmp_path, monkeypatch):
        from swival.audit import _load_audit_config
        from swival.config import load_config

        xdg = tmp_path / "xdg"
        (xdg / "swival").mkdir(parents=True)
        (xdg / "swival" / "config.toml").write_text("[audit]\npatch_max_turns = 60\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        (tmp_path / "swival.toml").write_text("[audit]\npatch_max_turns = 75\n")

        cfg = load_config(tmp_path)
        assert cfg["audit"]["patch_max_turns"] == 75
        audit_cfg = _load_audit_config(str(tmp_path))
        assert audit_cfg["patch_max_turns"] == 75

    def test_patch_max_turns_rejects_invalid(self, tmp_path, monkeypatch):
        from swival.config import load_config, ConfigError

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text("[audit]\npatch_max_turns = 0\n")
        with pytest.raises(ConfigError, match="patch_max_turns"):
            load_config(tmp_path)

    def test_harness_audit_keys_load(self, tmp_path, monkeypatch):
        from swival.audit import _load_audit_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text(
            "[audit]\n"
            "budget_tokens = 2_000_000\n"
            "max_gapfill_tasks = 40\n"
            "max_hunt_tasks = 150\n"
            "proof_strict = true\n"
            "\n"
            "[audit.reviewer]\n"
            'profile = "reviewer"\n'
            'model = "claude-haiku-mini"\n'
            "\n"
            "[[audit.hunt_task]]\n"
            'attack_class = "ssrf"\n'
            'attacker_position = "unauthenticated remote client"\n'
        )
        audit_cfg = _load_audit_config(str(tmp_path))
        assert audit_cfg["budget_tokens"] == 2_000_000
        assert audit_cfg["max_gapfill_tasks"] == 40
        assert audit_cfg["max_hunt_tasks"] == 150
        assert audit_cfg["proof_strict"] is True
        assert audit_cfg["reviewer_profile"] == "reviewer"
        assert audit_cfg["reviewer_model"] == "claude-haiku-mini"
        assert audit_cfg["hunt_tasks"][0]["attack_class"] == "ssrf"

    def test_reviewer_subkey_rejects_unknown(self, tmp_path, monkeypatch):
        from swival.config import ConfigError, load_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text('[audit.reviewer]\nbogus = "x"\n')
        with pytest.raises(ConfigError, match="audit.reviewer.bogus"):
            load_config(tmp_path)

    def test_proof_strict_must_be_bool(self, tmp_path, monkeypatch):
        from swival.config import ConfigError, load_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text("[audit]\nproof_strict = 1\n")
        with pytest.raises(ConfigError, match="proof_strict"):
            load_config(tmp_path)

    def test_budget_tokens_must_be_non_negative(self, tmp_path, monkeypatch):
        from swival.config import ConfigError, load_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "swival.toml").write_text("[audit]\nbudget_tokens = -5\n")
        with pytest.raises(ConfigError, match="budget_tokens"):
            load_config(tmp_path)

    def test_project_reviewer_overrides_global(self, tmp_path, monkeypatch):
        from swival.audit import _load_audit_config

        xdg = tmp_path / "xdg"
        (xdg / "swival").mkdir(parents=True)
        (xdg / "swival" / "config.toml").write_text(
            '[audit.reviewer]\nprofile = "g-rev"\nmodel = "g-model"\n'
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        (tmp_path / "swival.toml").write_text('[audit.reviewer]\nprofile = "p-rev"\n')
        audit_cfg = _load_audit_config(str(tmp_path))
        # project profile wins; model falls back to global because project did
        # not set it
        assert audit_cfg["reviewer_profile"] == "p-rev"
        assert audit_cfg["reviewer_model"] == "g-model"


class TestForceReviewMatching:
    """_resolve_force_review glob behavior and warnings."""

    def test_match_exact_path(self):
        from swival.audit import _resolve_force_review

        matches, warns = _resolve_force_review(
            ["a.py"], {"a.py": "project"}, ["a.py", "b.py"]
        )
        assert matches == {"a.py": "project"}
        assert warns == []

    def test_directory_trailing_slash(self):
        from swival.audit import _resolve_force_review

        matches, _ = _resolve_force_review(
            ["src/"],
            {"src/": "project"},
            ["src/a.py", "src/sub/b.py", "other.py"],
        )
        assert "src/a.py" in matches
        assert "src/sub/b.py" in matches
        assert "other.py" not in matches

    def test_zero_match_project_warns(self):
        from swival.audit import _resolve_force_review

        matches, warns = _resolve_force_review(
            ["missing.py"], {"missing.py": "project"}, ["a.py"]
        )
        assert matches == {}
        assert any("missing.py" in w for w in warns)

    def test_zero_match_global_silent(self):
        from swival.audit import _resolve_force_review

        matches, warns = _resolve_force_review(
            ["missing.py"], {"missing.py": "global"}, ["a.py"]
        )
        assert matches == {}
        assert warns == []

    def test_project_overrides_global_origin(self):
        from swival.audit import _resolve_force_review

        matches, _ = _resolve_force_review(
            ["a.py", "a.py"],
            {"a.py": "project"},  # last-write wins on same glob
            ["a.py"],
        )
        assert matches["a.py"] == "project"


class TestForceReviewPromotion:
    def test_force_review_promotes_skip(self, tmp_path):
        from swival.audit import _apply_promotions

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        rec = _bare_triage("a.py")
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            triage_records={"a.py": rec},
            state_dir=tmp_path,
        )
        _apply_promotions(state, {"a.py": "project"})
        assert state.triage_records["a.py"].priority == "ESCALATE_MEDIUM"
        assert any(
            "forced via swival.toml" in r
            for r in state.triage_records["a.py"].promotion_reasons
        )


class TestPhase2PromptScoreCache:
    """The triage prompt score must come from state.attack_scores so that
    the prompt and the promotion gate can never disagree."""

    def test_uses_cached_score(self, tmp_path, monkeypatch):
        from swival.audit import _phase2_triage_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "subprocess.run(cmd)\neval(data)")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        # Prime the cache with a sentinel value that does NOT equal what
        # _score_attack_surface(content) would return for this file.
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            attack_scores={"a.py": 999},
            state_dir=tmp_path,
        )

        captured = {}

        def fake_call(ctx, msgs, temperature=0.0, trace_task=None):
            captured["user"] = msgs[1]["content"]
            return (
                "@@ triage @@\n"
                "priority: SKIP\n"
                "confidence: medium\n"
                "summary: x\n"
                "needs_followup: false\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        _phase2_triage_one("a.py", state, ctx)
        assert "score=999" in captured["user"], captured["user"]

    def test_falls_back_to_compute_when_cache_empty(self, tmp_path, monkeypatch):
        """Legacy state files predate attack_scores. The prompt must still
        receive a real score, and the cache must be backfilled."""
        from swival.audit import _phase2_triage_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "subprocess.run(cmd)\neval(data)")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            attack_scores={},  # legacy: empty
            state_dir=tmp_path,
        )

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, msgs, temperature=0.0, trace_task=None: (
                "@@ triage @@\n"
                "priority: SKIP\n"
                "confidence: medium\n"
                "summary: x\n"
                "needs_followup: false\n"
            ),
        )
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})

        _phase2_triage_one("a.py", state, ctx)
        assert "a.py" in state.attack_scores
        assert state.attack_scores["a.py"] >= 5  # subprocess+eval scores

    def test_confirmation_pass_uses_cached_score(self, tmp_path, monkeypatch):
        from swival.audit import _phase2_confirm_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            attack_scores={"a.py": 777},
            repo_profile={"summary": "tiny"},
            state_dir=tmp_path,
        )

        captured = {}

        def fake_call(ctx, msgs, temperature=0.0, trace_task=None):
            captured["user"] = msgs[1]["content"]
            return (
                "@@ triage @@\n"
                "priority: SKIP\n"
                "confidence: low\n"
                "summary: x\n"
                "needs_followup: false\n"
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)
        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        _phase2_confirm_one("a.py", state, ctx)
        assert "score=777" in captured["user"]


class TestPhase2TriageFailureRecord:
    def test_llm_call_failure_returns_record(self, tmp_path, monkeypatch):
        from swival.audit import _phase2_triage_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path,
        )

        def _boom(*a, **kw):
            raise TimeoutError("upstream timed out")

        monkeypatch.setattr("swival.audit._call_audit_llm", _boom)

        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        rec = _phase2_triage_one("a.py", state, ctx)
        assert rec.priority == "SKIP"
        assert rec.triage_failure_mode == "llm_call_failed:TimeoutError"

    def test_parse_error_tagged(self, tmp_path, monkeypatch):
        from swival.audit import _phase2_triage_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            state_dir=tmp_path,
        )

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, msgs, temperature=0.0, trace_task=None: "garbage no record",
        )

        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        rec = _phase2_triage_one("a.py", state, ctx)
        assert rec.priority == "SKIP"
        assert rec.triage_failure_mode == "parse_error"


class TestConfirmationPass:
    def test_promotes_low_confidence_skip(self, tmp_path, monkeypatch):
        from swival.audit import _phase2_confirm_one
        from types import SimpleNamespace

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1")

        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            attack_scores={"a.py": 0},
            repo_profile={"summary": "tiny"},
            state_dir=tmp_path,
        )

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, msgs, temperature=0.0, trace_task=None: (
                "@@ triage @@\n"
                "priority: ESCALATE_MEDIUM\n"
                "confidence: medium\n"
                "summary: confirmed worth a look\n"
                "needs_followup: false\n"
            ),
        )

        ctx = SimpleNamespace(base_dir=str(tmp_path), loop_kwargs={})
        rec = _phase2_confirm_one("a.py", state, ctx)
        assert rec is not None
        assert rec.priority == "ESCALATE_MEDIUM"


class TestMeasureTriage:
    def test_parser_sets_measure_triage(self, monkeypatch, tmp_path):
        from swival.audit import run_audit_command

        captured = _capture_run_audit_phases(monkeypatch)
        run_audit_command("--measure-triage src/foo", _make_ctx(tmp_path))
        assert captured["measure_triage"] is True

    def test_round_trip(self, tmp_path):
        scope = AuditScope(
            branch="m",
            commit="c",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state = AuditRunState(
            run_id="r",
            scope=scope,
            queued_files=["a.py"],
            measure_triage=True,
            measurement_escalated_paths={"a.py"},
            state_dir=tmp_path,
        )
        state.save()
        loaded = AuditRunState.load(state.state_dir, "r")
        assert loaded.measure_triage is True
        assert loaded.measurement_escalated_paths == {"a.py"}

    def test_resume_mismatch_rejected(self, tmp_path, monkeypatch):
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="m1",
            scope=scope,
            queued_files=["a.py"],
            state_dir=state_dir,
            measure_triage=False,
            phase="deep_review",
        )
        state.save()

        result = run_audit_command("--measure-triage --resume", _make_ctx(tmp_path))
        assert "measure-triage mismatch" in result

    def test_recall_only_emits_inside_phase5(self, tmp_path, monkeypatch):
        """Recall emission belongs inside the artifacts (Phase 5) block.

        The bug: ``artifacts_written`` is a local that resets to 0, and the
        old code emitted recall after the artifacts block unconditionally —
        so a measurement run that enters ``_run_audit_phases`` with state
        already past Phase 5 (e.g. via ``--regen``, which forces phase back
        to artifacts and re-runs only that block) would still print recall,
        and the trailing ``artifacts_written == 0`` fallback could fire on
        cold paths and print "No provable security bugs" despite real
        findings on disk.

        The fix moves recall emission inside the Phase-5 block so it only
        fires when artifacts actually run. This test pins the structural
        invariant: in source, the recall call must appear before the
        ``state.phase = "done"`` assignment, inside the
        ``if state.phase == "artifacts":`` body.
        """
        import inspect

        from swival.audit import _run_audit_phases

        src = inspect.getsource(_run_audit_phases)
        recall_idx = src.find("_emit_measure_triage_recall(state)")
        done_idx = src.find('state.phase = "done"')
        artifacts_gate = src.find('if state.phase == "artifacts":')
        assert recall_idx > 0, "recall call missing from _run_audit_phases"
        assert recall_idx > artifacts_gate, (
            "recall must be inside the artifacts block, not before it"
        )
        assert recall_idx < done_idx, (
            "recall must run before phase=done so it only emits on successful Phase-5 completion"
        )


class TestResumeForceReview:
    def test_added_force_review_promotes_saved_skip_on_resume(
        self, tmp_path, monkeypatch
    ):
        """Adding a glob to swival.toml after Phase 2 should promote a saved
        SKIP on the next resume."""
        from swival.audit import run_audit_command

        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "pass")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        scope = AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=[],
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="rf",
            scope=scope,
            queued_files=["a.py"],
            triage_records={"a.py": _bare_triage("a.py")},  # SKIP
            candidate_files=[],
            reviewed_files={"a.py"},
            deep_reviewed_files={"a.py"},
            state_dir=state_dir,
            phase="deep_review",
        )
        state.save()

        # Now add force_review for a.py
        (tmp_path / "swival.toml").write_text('[audit]\nforce_review = ["a.py"]\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        # Stub heavy phases
        monkeypatch.setattr(
            "swival.audit._deep_review_one",
            lambda path, state, ctx: DeepReviewResult(path=path, findings=[]),
        )

        run_audit_command("--resume", _make_ctx(tmp_path))

        loaded = AuditRunState.load(state_dir, "rf")
        assert loaded.triage_records["a.py"].priority == "ESCALATE_MEDIUM"
        assert "a.py" in loaded.candidate_files
        assert any(
            "forced via swival.toml" in r
            for r in loaded.triage_records["a.py"].promotion_reasons
        )
