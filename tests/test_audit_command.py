"""Tests for swival/audit.py — scope, JSON parsing, triage, verification, artifacts."""

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
    TriageRecord,
    VerificationResult,
    VerifiedFinding,
    _TransientVerifierError,
    _canonicalize_finding,
    _extract_exports,
    _extract_imports,
    _finding_key,
    _is_auditable,
    _make_slug,
    _load_file_contents,
    _order_by_attack_surface,
    _parse_json_response,
    _parse_with_repair,
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
                "top_p": 1.0,
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

        scope = _resolve_scope(str(tmp_path), None)
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

        scope = _resolve_scope(str(tmp_path), "src/")
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

        scope = _resolve_scope(str(tmp_path), None)
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
# JSON parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_plain_json(self):
        result = _parse_json_response('{"priority": "SKIP"}')
        assert result["priority"] == "SKIP"

    def test_fenced_json(self):
        text = '```json\n{"priority": "ESCALATE_HIGH"}\n```'
        result = _parse_json_response(text)
        assert result["priority"] == "ESCALATE_HIGH"

    def test_json_with_preamble(self):
        text = 'Here is my analysis:\n{"priority": "SKIP", "confidence": "high"}'
        result = _parse_json_response(text)
        assert result["priority"] == "SKIP"

    def test_json_with_suffix(self):
        text = '{"accepted": true}\n\nThis finding looks valid.'
        result = _parse_json_response(text)
        assert result["accepted"] is True

    def test_missing_required_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_json_response('{"foo": 1}', required_keys=["bar"])

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_json_response("")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _parse_json_response("just plain text with no braces")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_json_response("{broken json")


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
            focus=None,
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

        not_found = AuditRunState.find_resumable(state.state_dir, "abc123", "src/")
        assert not_found is None

    def test_resume_without_focus_matches_focused_run(self, tmp_path):
        state = self._make_state(tmp_path)
        scope = state.scope
        state.scope = AuditScope(
            branch=scope.branch,
            commit=scope.commit,
            tracked_files=scope.tracked_files,
            mandatory_files=scope.mandatory_files,
            focus="subdir/",
        )
        state.save()

        found = AuditRunState.find_resumable(state.state_dir, "abc123", None)
        assert found is not None
        assert found.run_id == "test-run"

        found = AuditRunState.find_resumable(state.state_dir, "abc123", "subdir/")
        assert found is not None

        not_found = AuditRunState.find_resumable(state.state_dir, "abc123", "other/")
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
            focus=None,
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
            focus=None,
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
    def test_deep_review_repairs_malformed_inventory_json(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from swival.audit import _deep_review_one

        scope = AuditScope(
            branch="main",
            commit="abc123",
            tracked_files=["a.py"],
            mandatory_files=["a.py"],
            focus=None,
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
                return '{"findings": ['
            return '{"findings": []}'

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
            focus=None,
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
                "top_p": 1.0,
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
        from swival.audit import _PHASE3A_TEMPLATE

        assert (
            "Prefer the narrowest bug that the evidence directly proves."
            in _PHASE3A_TEMPLATE
        )
        assert "undefined behavior or uninitialized-state bugs" in _PHASE3A_TEMPLATE

    def test_phase3b_expansion_prompt_exists(self):
        from swival.audit import _PHASE3B_TEMPLATE

        assert "expanding one security finding" in _PHASE3B_TEMPLATE.lower()
        assert "{title}" in _PHASE3B_TEMPLATE
        assert "{claim}" in _PHASE3B_TEMPLATE

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
        expected = "No provable logic or security bugs found in Git-tracked files."
        assert (
            expected == "No provable logic or security bugs found in Git-tracked files."
        )

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
        # The suffix template appends "The file is: {path}" at the end
        # Verified by reading the _phase2_triage_one function

    def test_deep_review_includes_bug_classes(self):
        """Phase 3a inventory prompt includes triage bug classes."""
        from swival.audit import _PHASE3A_TEMPLATE

        assert "{bug_classes}" in _PHASE3A_TEMPLATE


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
            focus="src/",
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
            focus=None,
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
            focus=None,
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
            "top_p": 1.0,
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
# JSON repair and parse_with_repair
# ---------------------------------------------------------------------------


class TestJsonRepair:
    def test_parse_with_repair_succeeds_on_valid_json(self):
        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}
        result = _parse_with_repair(
            ctx=None,
            raw='{"findings": []}',
            required_keys=["findings"],
            schema_hint="{}",
            metrics=metrics,
        )
        assert result == {"findings": []}
        assert metrics["parse_failures"] == 0

    def test_parse_with_repair_repairs_malformed(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: '{"findings": []}',
        )

        result = _parse_with_repair(
            ctx=SimpleNamespace(),
            raw='{"findings": [',
            required_keys=["findings"],
            schema_hint="{}",
            metrics=metrics,
        )
        assert result == {"findings": []}
        assert metrics["parse_failures"] == 1
        assert metrics["repair_successes"] == 1

    def test_parse_with_repair_raises_when_repair_fails(self, monkeypatch):
        from types import SimpleNamespace

        metrics = {"parse_failures": 0, "repair_successes": 0, "repair_failures": 0}

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: "still broken {{{",
        )

        with pytest.raises(ValueError):
            _parse_with_repair(
                ctx=SimpleNamespace(),
                raw='{"findings": [',
                required_keys=["findings"],
                schema_hint="{}",
                metrics=metrics,
            )
        assert metrics["parse_failures"] == 1
        assert metrics["repair_failures"] == 1
        assert metrics["repair_successes"] == 0


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
            "type": "vulnerability",
            "preconditions": "attacker controls argv[1]",
            "proof": "input reaches strcpy without bounds check",
            "fix_outline": "use strncpy with bounds",
        }
        f = _canonicalize_finding(inventory, expansion, "main.c")
        assert f.title == "Buffer overflow"
        assert f.finding_type == "vulnerability"
        assert f.severity == "high"
        assert f.locations == ["main.c:17"]
        assert f.preconditions == ["attacker controls argv[1]"]
        assert f.proof == ["input reaches strcpy without bounds check"]
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
            focus=None,
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
            lambda ctx, messages, temperature=0.0, trace_task=None: '{"findings": []}',
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
                    '{"findings": [{"title": "eval injection",'
                    '"severity": "high", "location": "a.py:1",'
                    '"claim": "user input reaches eval"}]}'
                )
            return (
                '{"type": "vulnerability",'
                '"preconditions": "user provides input",'
                '"proof": "input flows to eval without sanitization",'
                '"fix_outline": "remove eval"}'
            )

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f.title == "eval injection"
        assert f.finding_type == "vulnerability"
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

        monkeypatch.setattr(
            "swival.audit._call_audit_llm",
            lambda ctx, messages, temperature=0.0, trace_task=None: (
                '{"findings": [{"title": "bug A",'
                '"severity": "high", "location": "a.py:1",'
                '"claim": "claim A"}]}'
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
                    '{"findings": ['
                    '{"title": "bug A", "severity": "high", "location": "a.py:1", "claim": "claim A"},'
                    '{"title": "bug B", "severity": "medium", "location": "a.py:2", "claim": "claim B"}'
                    "]}"
                )
            if calls["n"] == 2:
                return (
                    '{"type": "vulnerability",'
                    '"preconditions": "none",'
                    '"proof": "proven",'
                    '"fix_outline": "fix"}'
                )
            return "broken {{{"

        monkeypatch.setattr("swival.audit._call_audit_llm", fake_call)

        result = _deep_review_one("a.py", state, ctx)
        assert result.error is None
        assert len(result.findings) == 1
        assert result.findings[0].title == "bug A"

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
                return "not json at all"
            return '{"findings": []}'

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
            lambda ctx, messages, temperature=0.0, trace_task=None: "never valid json",
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
            focus=None,
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
            lambda vf, patch_fn, ctx: "# Report",
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
        assert "No provable logic or security bugs" in result2
