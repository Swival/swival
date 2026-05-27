"""Tests for the audit-findings README index."""

from __future__ import annotations

import io
import re
import subprocess
from pathlib import Path

from rich.console import Console

from swival import audit_ui, fmt
from swival.audit import (
    AuditRunState,
    AuditScope,
    FindingRecord,
    PatchGenerationResult,
    VerifiedFinding,
    _artifact_key,
    _ensure_artifact_state,
    _finding_key,
    _render_findings_readme,
    _run_audit_phases,
    _write_findings_readme,
)


def _init_git(tmp_path: Path) -> None:
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


def _make_finding(**overrides) -> FindingRecord:
    f = FindingRecord(
        title="Buffer overflow in parser",
        finding_type="memory_safety",
        severity="high",
        locations=["main.c:7"],
        preconditions=["program receives a command-line argument"],
        proof=["argv data reaches unbounded copy"],
        fix_outline="bounds-check the copy",
        source_file="main.c",
    )
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


def _make_verified(**overrides) -> VerifiedFinding:
    return VerifiedFinding(
        finding=_make_finding(**overrides),
        correctness_reason="reproduced",
        rebuttal_reason="n/a",
    )


def _make_state(tmp_path: Path, **scope_overrides) -> AuditRunState:
    scope = AuditScope(
        branch=scope_overrides.get("branch", "main"),
        commit=scope_overrides.get(
            "commit", "deadbeefcafe1234567890abcdef0123456789ab"
        ),
        tracked_files=scope_overrides.get("tracked_files", ["main.c", "lib.c"]),
        mandatory_files=scope_overrides.get("mandatory_files", ["main.c", "lib.c"]),
        focus=scope_overrides.get("focus", []),
    )
    return AuditRunState(
        run_id="r1",
        scope=scope,
        queued_files=list(scope.mandatory_files),
        reviewed_files=set(scope.mandatory_files),
        triage_records={},
        candidate_files=list(scope.mandatory_files),
        deep_reviewed_files=set(scope.mandatory_files),
        state_dir=tmp_path / ".swival" / "audit",
    )


def _make_two_finding_state(tmp_path: Path) -> AuditRunState:
    state = _make_state(tmp_path)
    high = _make_verified(title="High-severity bug", severity="high")
    crit = _make_verified(
        title="Critical-severity bug", severity="critical", source_file="lib.c"
    )
    state.verified_findings = [high, crit]
    _ensure_artifact_state(state)
    for vf in state.verified_findings:
        entry = state.artifact_state[_artifact_key(vf)]
        entry["status"] = "written"
    return state


# ---------------------------------------------------------------------------
# 1. Deterministic render
# ---------------------------------------------------------------------------


class TestDeterministicRender:
    def test_render_is_byte_identical_across_calls(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        a = _render_findings_readme(state)
        b = _render_findings_readme(state)
        assert a == b

    def test_metadata_contains_commit_and_branch(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        text = _render_findings_readme(state)
        assert "deadbeefcafe" in text
        assert "- branch: `main`" in text
        assert "- commit: `deadbeefcafe1234567890abcdef0123456789ab`" in text

    def test_summary_table_links_to_each_artifact(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        text = _render_findings_readme(state)
        for vf in state.verified_findings:
            entry = state.artifact_state[_artifact_key(vf)]
            report = entry["report_filename"]
            patch = entry["patch_filename"]
            assert f"[{report}]({report})" in text
            assert f"[{patch}]({patch})" in text

    def test_severity_grouping_orders_critical_before_high(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        text = _render_findings_readme(state)
        i_crit = text.index("### critical")
        i_high = text.index("### high")
        assert i_crit < i_high

    def test_failed_section_absent_when_all_written(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        text = _render_findings_readme(state)
        assert "Failed or pending artifacts" not in text

    def test_totals_block_matches_state(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        # _audit_totals reads len() of triage_records; the value type does not
        # matter — only the count does.
        state.triage_records = {"main.c": object(), "lib.c": object()}  # type: ignore[dict-item]
        text = _render_findings_readme(state)
        assert "- findings verified: 2" in text
        assert "- files triaged: 2" in text
        assert "- artifacts: 2 written, 0 failed, 0 pending" in text


# ---------------------------------------------------------------------------
# 2. Table escaping
# ---------------------------------------------------------------------------


class TestTableEscaping:
    def test_pipe_and_newline_are_escaped(self, tmp_path):
        state = _make_state(tmp_path)
        vf = _make_verified(title="a | b\nc")
        state.verified_findings = [vf]
        _ensure_artifact_state(state)
        state.artifact_state[_artifact_key(vf)]["status"] = "written"

        text = _render_findings_readme(state)
        assert "a \\| b c" in text

        # The escaped row must remain a parseable GFM row with 8 cells.
        matching = [
            line
            for line in text.splitlines()
            if line.startswith("|") and "a \\| b" in line
        ]
        assert len(matching) == 1, matching
        row = matching[0]
        # Split on unescaped pipes only — `\|` is a literal pipe inside a cell
        # per GFM and must not introduce a new column.
        cells = re.split(r"(?<!\\)\|", row)
        assert cells[0] == "" and cells[-1] == ""
        assert len(cells) == 10, row


# ---------------------------------------------------------------------------
# 3. Failed-artifact disclosure
# ---------------------------------------------------------------------------


class TestFailedArtifactDisclosure:
    def test_failed_entry_listed_with_error_and_retry_command(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        vf_failed = state.verified_findings[1]
        entry = state.artifact_state[_artifact_key(vf_failed)]
        entry["status"] = "failed"
        entry["last_error_code"] = "patch_turn_budget_exhausted"
        entry["last_error"] = "ran out of turns generating the patch"
        idx = entry["index"]

        text = _render_findings_readme(state)
        assert "## Failed or pending artifacts" in text
        assert "patch_turn_budget_exhausted" in text
        assert "ran out of turns generating the patch" in text
        assert f"/audit --regen --finding {idx} --patch-max-turns 75" in text
        assert "/audit --resume --patch-max-turns 75" in text


# ---------------------------------------------------------------------------
# 4. End-to-end via the phase machine, clean exit
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        base_dir=str(tmp_path),
        tools=[],
        verbose=False,
        no_history=True,
        loop_kwargs={},
    )


def _phase5_setup(tmp_path: Path, findings: list[VerifiedFinding]):
    _init_git(tmp_path)
    _commit_file(tmp_path, "main.c", "int main(void) { return 0; }")
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
        .decode()
        .strip()
    )
    state_dir = Path(tmp_path) / ".swival" / "audit"
    state = AuditRunState(
        run_id="e2e",
        scope=AuditScope(
            branch="main",
            commit=commit,
            tracked_files=["main.c"],
            mandatory_files=["main.c"],
            focus=[],
        ),
        queued_files=["main.c"],
        reviewed_files={"main.c"},
        triage_records={},
        candidate_files=["main.c"],
        deep_reviewed_files={"main.c"},
        verified_findings=list(findings),
        state_dir=state_dir,
        phase="artifacts",
    )
    return state, state_dir, commit


class TestEndToEndCleanExit:
    def test_readme_is_written_at_end_of_artifact_phase(self, monkeypatch, tmp_path):
        vf = _make_verified()
        state, state_dir, _ = _phase5_setup(tmp_path, [vf])
        state.save()

        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50, ui=None: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        result = _run_audit_phases(
            "--resume",
            _ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        readme = tmp_path / "audit-findings" / "README.md"
        assert readme.exists()
        body = readme.read_text(encoding="utf-8")
        loaded = AuditRunState.load(state_dir, "e2e")
        entry = loaded.artifact_state[_artifact_key(vf)]
        assert entry["report_filename"] in body
        assert entry["patch_filename"] in body
        assert "Open audit-findings/README.md to review." in result


# ---------------------------------------------------------------------------
# 5. End-to-end, failed-artifact rewind
# ---------------------------------------------------------------------------


class TestEndToEndFailedRewind:
    def test_readme_written_before_failed_artifact_rewind(self, monkeypatch, tmp_path):
        ok = _make_verified(title="Will succeed")
        bad = _make_verified(title="Will fail")
        state, state_dir, _ = _phase5_setup(tmp_path, [ok, bad])
        state.save()

        def fake_patch(vf, ctx, state, patch_max_turns=50, ui=None):
            if vf.finding.title == "Will succeed":
                return PatchGenerationResult(patch_text="diff\n")
            return PatchGenerationResult(
                error_code="patch_turn_budget_exhausted",
                error="turn budget exhausted",
            )

        monkeypatch.setattr("swival.audit._phase5_patch", fake_patch)
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        result = _run_audit_phases(
            "--resume",
            _ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        assert "Audit incomplete" in result
        assert "artifact generation has" in result
        readme = tmp_path / "audit-findings" / "README.md"
        assert readme.exists()
        body = readme.read_text(encoding="utf-8")
        assert "## Failed or pending artifacts" in body

        loaded = AuditRunState.load(state_dir, "e2e")
        ok_entry = loaded.artifact_state[_finding_key(ok.finding)]
        bad_entry = loaded.artifact_state[_finding_key(bad.finding)]
        assert ok_entry["status"] == "written"
        assert bad_entry["status"] == "failed"

        failed_patch = bad_entry["patch_filename"]
        assert f"[{failed_patch}]({failed_patch})" not in body
        assert failed_patch in body


# ---------------------------------------------------------------------------
# 5a. End-to-end, unreviewed rewind
# ---------------------------------------------------------------------------


class TestEndToEndUnreviewedRewind:
    def test_readme_written_when_unreviewed_files_remain(self, monkeypatch, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "main.c", "int main(void) { return 0; }")
        _commit_file(tmp_path, "other.c", "int other(void) { return 0; }")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        vf = _make_verified()
        state = AuditRunState(
            run_id="unrev",
            scope=AuditScope(
                branch="main",
                commit=commit,
                tracked_files=["main.c", "other.c"],
                mandatory_files=["main.c", "other.c"],
                focus=[],
            ),
            queued_files=["main.c", "other.c"],
            reviewed_files={"main.c"},
            triage_records={},
            candidate_files=["main.c"],
            deep_reviewed_files={"main.c"},
            verified_findings=[vf],
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50, ui=None: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        result = _run_audit_phases(
            "--resume",
            _ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        assert "files were not reviewed" in result
        readme = tmp_path / "audit-findings" / "README.md"
        assert readme.exists()
        body = readme.read_text(encoding="utf-8")
        assert "Buffer overflow in parser" in body


# ---------------------------------------------------------------------------
# 5b. End-to-end, undeep-reviewed rewind
# ---------------------------------------------------------------------------


class TestEndToEndUndeepReviewedRewind:
    def test_readme_written_when_undeep_reviewed_remain(self, monkeypatch, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "main.c", "int main(void) { return 0; }")
        _commit_file(tmp_path, "other.c", "int other(void) { return 0; }")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        vf = _make_verified()
        state = AuditRunState(
            run_id="undeep",
            scope=AuditScope(
                branch="main",
                commit=commit,
                tracked_files=["main.c", "other.c"],
                mandatory_files=["main.c", "other.c"],
                focus=[],
            ),
            queued_files=["main.c", "other.c"],
            reviewed_files={"main.c", "other.c"},
            triage_records={},
            candidate_files=["main.c", "other.c"],
            deep_reviewed_files={"main.c"},
            verified_findings=[vf],
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        monkeypatch.setattr(
            "swival.audit._phase5_patch",
            lambda vf, ctx, state, patch_max_turns=50, ui=None: PatchGenerationResult(
                patch_text="diff\n"
            ),
        )
        monkeypatch.setattr(
            "swival.audit._phase5_report",
            lambda vf, patch_fn, patch_text, ctx: "# report",
        )

        result = _run_audit_phases(
            "--resume",
            _ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        assert "escalated files failed" in result
        readme = tmp_path / "audit-findings" / "README.md"
        assert readme.exists()


# ---------------------------------------------------------------------------
# 6. No README on zero findings
# ---------------------------------------------------------------------------


class TestNoReadmeOnZeroFindings:
    def test_no_readme_when_zero_findings(self, monkeypatch, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "main.c", "int main(void) { return 0; }")
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
            .decode()
            .strip()
        )
        state_dir = Path(tmp_path) / ".swival" / "audit"
        state = AuditRunState(
            run_id="zero",
            scope=AuditScope(
                branch="main",
                commit=commit,
                tracked_files=["main.c"],
                mandatory_files=["main.c"],
                focus=[],
            ),
            queued_files=["main.c"],
            reviewed_files={"main.c"},
            triage_records={},
            candidate_files=["main.c"],
            deep_reviewed_files={"main.c"},
            verified_findings=[],
            state_dir=state_dir,
            phase="artifacts",
        )
        state.save()

        result = _run_audit_phases(
            "--resume",
            _ctx(tmp_path),
            str(tmp_path),
            state_dir,
            1,
            True,
            False,
            None,
        )

        assert "No provable security bugs" in result
        readme = tmp_path / "audit-findings" / "README.md"
        assert not readme.exists()


# ---------------------------------------------------------------------------
# 7. Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_write_replaces_existing_file_and_leaves_no_tmp(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        artifact_dir = tmp_path / state.artifact_dir
        artifact_dir.mkdir(parents=True)
        canary = artifact_dir / "README.md"
        canary.write_text("CANARY CONTENTS\n", encoding="utf-8")

        result = _write_findings_readme(state, str(tmp_path))
        assert result is True
        body = canary.read_text(encoding="utf-8")
        assert "CANARY CONTENTS" not in body
        assert "# Security audit" in body
        assert not (artifact_dir / "README.tmp").exists()


# ---------------------------------------------------------------------------
# Stale README cleanup — gate failure must not leave an old README around.
# ---------------------------------------------------------------------------


class TestStaleReadmeCleanup:
    def test_failed_regen_removes_existing_readme(self, tmp_path):
        """After a regen flips every entry back to pending and the retry also
        fails, the gate must reject and the previous README must be removed
        so it cannot keep advertising stale links."""
        from swival.audit import _reset_artifact_targets_for_regen

        state = _make_two_finding_state(tmp_path)
        assert _write_findings_readme(state, str(tmp_path)) is True
        readme = tmp_path / state.artifact_dir / "README.md"
        assert readme.exists()

        _reset_artifact_targets_for_regen(state, None)
        result = _write_findings_readme(state, str(tmp_path))
        assert result is False
        assert not readme.exists()

    def test_zero_findings_with_existing_readme_removes_it(self, tmp_path):
        state = _make_state(tmp_path)
        artifact_dir = tmp_path / state.artifact_dir
        artifact_dir.mkdir(parents=True)
        readme = artifact_dir / "README.md"
        readme.write_text("STALE\n", encoding="utf-8")
        result = _write_findings_readme(state, str(tmp_path))
        assert result is False
        assert not readme.exists()


# ---------------------------------------------------------------------------
# Repo basename in the README title.
# ---------------------------------------------------------------------------


class TestRepoBasenameInTitle:
    def test_title_uses_repo_basename_arg(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        text = _render_findings_readme(state, repo_name="my-cool-repo")
        first_line = text.splitlines()[0]
        assert first_line.startswith("# Security audit — my-cool-repo @ deadbeefcafe")

    def test_write_helper_passes_base_dir_basename(self, tmp_path):
        state = _make_two_finding_state(tmp_path)
        result = _write_findings_readme(state, str(tmp_path))
        assert result is True
        readme = tmp_path / state.artifact_dir / "README.md"
        body = readme.read_text(encoding="utf-8")
        expected_repo = tmp_path.resolve().name
        first_line = body.splitlines()[0]
        assert first_line.startswith(f"# Security audit — {expected_repo} @ ")


# ---------------------------------------------------------------------------
# 8. TTY summary text
# ---------------------------------------------------------------------------


def _swap_console(monkeypatch, *, force_terminal: bool):
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=force_terminal,
        no_color=True,
        width=140,
        record=True,
    )
    monkeypatch.setattr(fmt, "_console", console)
    return buf, console


class TestTtySummaryReadmeRow:
    def test_summary_with_readme_written_true(self, monkeypatch):
        _, console = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c0ffee",
            workers=1,
            total_files=1,
        ) as ui:
            ui.tally(verified=2, severity="high")
            ui.summary(
                artifact_dir="audit-findings",
                written=2,
                readme_written=True,
            )
        out = console.export_text(clear=False)
        assert "see README.md" in out

    def test_summary_with_readme_written_false(self, monkeypatch):
        _, console = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c0ffee",
            workers=1,
            total_files=1,
        ) as ui:
            ui.tally(verified=1, severity="high")
            ui.summary(artifact_dir="audit-findings", written=1)
        out = console.export_text(clear=False)
        assert "see README.md" not in out
        assert "1 written to audit-findings/" in out


# ---------------------------------------------------------------------------
# 9. Regen does not break stale links
# ---------------------------------------------------------------------------


class TestRegenStaleLinks:
    def test_regen_pending_finding_renders_as_plain_text(self, tmp_path):
        from swival.audit import _reset_artifact_targets_for_regen

        state = _make_two_finding_state(tmp_path)
        vf2 = state.verified_findings[1]
        idx2 = state.artifact_state[_artifact_key(vf2)]["index"]
        # _reset_artifact_targets_for_regen takes a 0-based index set, even
        # though /audit --finding is 1-based at the CLI surface.
        _reset_artifact_targets_for_regen(state, {1})

        text = _render_findings_readme(state)
        report2 = state.artifact_state[_artifact_key(vf2)]["report_filename"]
        patch2 = state.artifact_state[_artifact_key(vf2)]["patch_filename"]
        assert f"[{report2}]({report2})" not in text
        assert f"[{patch2}]({patch2})" not in text
        assert report2 in text
        assert patch2 in text
        row_marker = f"| {idx2} | pending |"
        assert row_marker in text
