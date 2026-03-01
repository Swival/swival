"""Tests for swival.sandbox_agentfs — re-exec bootstrap, config, argv construction."""

import argparse
import os
import stat
import sys

import pytest

from swival import agent
from swival.config import _UNSET, ConfigError, apply_config_to_args
from swival.report import ReportCollector
from swival.sandbox_agentfs import (
    _AGENTFS_ENV,
    _ENV_MARKER,
    _VERSION_ENV,
    _absolutize_argv,
    _find_agentfs,
    build_agentfs_argv,
    check_sandbox_available,
    get_agentfs_version,
    is_inside_agentfs,
    is_sandboxed,
    maybe_reexec,
    probe_agentfs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a namespace mimicking build_parser() defaults."""
    defaults = {
        "provider": _UNSET,
        "model": _UNSET,
        "api_key": _UNSET,
        "base_url": _UNSET,
        "max_output_tokens": _UNSET,
        "max_context_tokens": _UNSET,
        "temperature": _UNSET,
        "top_p": _UNSET,
        "seed": _UNSET,
        "max_turns": _UNSET,
        "system_prompt": _UNSET,
        "no_system_prompt": _UNSET,
        "allowed_commands": _UNSET,
        "yolo": _UNSET,
        "add_dir": None,
        "add_dir_ro": None,
        "sandbox": _UNSET,
        "sandbox_session": _UNSET,
        "sandbox_strict_read": _UNSET,
        "no_read_guard": _UNSET,
        "no_instructions": _UNSET,
        "no_skills": _UNSET,
        "skills_dir": None,
        "no_history": _UNSET,
        "color": _UNSET,
        "no_color": _UNSET,
        "quiet": _UNSET,
        "reviewer": _UNSET,
        "review_prompt": _UNSET,
        "objective": _UNSET,
        "verify": _UNSET,
        "max_review_rounds": _UNSET,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_agentfs_script(tmp_path, *, version_output="agentfs v0.6.2"):
    """Write a dummy agentfs script that prints its args or version."""
    script = tmp_path / "agentfs"
    script.write_text(
        f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "{version_output}"; exit 0; fi\necho $@\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _set_sandboxed(monkeypatch):
    """Set both env markers to simulate Swival-initiated re-exec inside agentfs."""
    monkeypatch.setenv(_ENV_MARKER, "1")
    monkeypatch.setenv(_AGENTFS_ENV, "1")


def _set_external_agentfs(monkeypatch):
    """Set only AGENTFS=1 to simulate external agentfs run wrapping."""
    monkeypatch.delenv(_ENV_MARKER, raising=False)
    monkeypatch.setenv(_AGENTFS_ENV, "1")


def _clear_sandboxed(monkeypatch):
    """Clear both env markers."""
    monkeypatch.delenv(_ENV_MARKER, raising=False)
    monkeypatch.delenv(_AGENTFS_ENV, raising=False)


# ===========================================================================
# is_sandboxed() — requires both markers (Swival re-exec path)
# ===========================================================================


class TestIsSandboxed:
    def test_not_sandboxed_by_default(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        assert is_sandboxed() is False

    def test_sandboxed_when_both_markers_set(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        assert is_sandboxed() is True

    def test_not_sandboxed_with_only_swival_marker(self, monkeypatch):
        """Setting SWIVAL_AGENTFS_ACTIVE alone must not bypass the check."""
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_AGENTFS_ENV, raising=False)
        assert is_sandboxed() is False

    def test_not_sandboxed_with_only_agentfs_marker(self, monkeypatch):
        _set_external_agentfs(monkeypatch)
        assert is_sandboxed() is False

    def test_not_sandboxed_for_other_values(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "0")
        monkeypatch.setenv(_AGENTFS_ENV, "1")
        assert is_sandboxed() is False

    def test_not_sandboxed_for_empty_string(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "")
        monkeypatch.setenv(_AGENTFS_ENV, "1")
        assert is_sandboxed() is False


# ===========================================================================
# is_inside_agentfs() — accepts both re-exec and external wrapping
# ===========================================================================


class TestIsInsideAgentfs:
    def test_false_by_default(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        assert is_inside_agentfs() is False

    def test_true_with_both_markers(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        assert is_inside_agentfs() is True

    def test_true_with_only_agentfs_env(self, monkeypatch):
        """External agentfs run wrapping sets only AGENTFS=1."""
        _set_external_agentfs(monkeypatch)
        assert is_inside_agentfs() is True

    def test_false_with_only_swival_marker(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_AGENTFS_ENV, raising=False)
        assert is_inside_agentfs() is False


# ===========================================================================
# _find_agentfs()
# ===========================================================================


class TestFindAgentfs:
    def test_found_on_path(self, tmp_path, monkeypatch):
        script = _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert _find_agentfs() == script

    def test_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent-dir-for-test")
        with pytest.raises(ConfigError, match="agentfs binary not found"):
            _find_agentfs()

    def test_error_message_has_correct_url(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent-dir-for-test")
        with pytest.raises(ConfigError, match="tursodatabase/agentfs"):
            _find_agentfs()


# ===========================================================================
# _absolutize_argv()
# ===========================================================================


class TestAbsolutizeArgv:
    def test_resolves_base_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(["swival", "--base-dir", "subdir", "question"])
        assert result[2] == str((tmp_path / "subdir").resolve())

    def test_resolves_add_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(["swival", "--add-dir", "rel/path", "question"])
        assert result[2] == str((tmp_path / "rel/path").resolve())

    def test_resolves_multiple_path_flags(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(
            ["swival", "--base-dir", "proj", "--add-dir", "extra", "--add-dir-ro", "ro"]
        )
        assert result[2] == str((tmp_path / "proj").resolve())
        assert result[4] == str((tmp_path / "extra").resolve())
        assert result[6] == str((tmp_path / "ro").resolve())

    def test_preserves_absolute_paths(self):
        result = _absolutize_argv(["swival", "--base-dir", "/absolute/path", "q"])
        assert result[2] == "/absolute/path"

    def test_preserves_non_path_flags(self):
        result = _absolutize_argv(
            ["swival", "--model", "gpt-4", "--sandbox", "agentfs", "question"]
        )
        assert result == [
            "swival",
            "--model",
            "gpt-4",
            "--sandbox",
            "agentfs",
            "question",
        ]

    def test_expands_tilde(self, monkeypatch):
        result = _absolutize_argv(["swival", "--base-dir", "~/myproj"])
        from pathlib import Path

        assert result[2] == str(Path("~/myproj").expanduser().resolve())

    def test_equals_form_resolved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(["swival", "--base-dir=subdir", "question"])
        assert result[1] == "--base-dir=" + str((tmp_path / "subdir").resolve())

    def test_equals_form_multiple_flags(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(
            ["swival", "--base-dir=proj", "--add-dir=extra", "--add-dir-ro=ro"]
        )
        assert result[0] == "swival"
        assert result[1] == "--base-dir=" + str((tmp_path / "proj").resolve())
        assert result[2] == "--add-dir=" + str((tmp_path / "extra").resolve())
        assert result[3] == "--add-dir-ro=" + str((tmp_path / "ro").resolve())

    def test_equals_form_absolute_preserved(self):
        result = _absolutize_argv(["swival", "--base-dir=/abs/path", "q"])
        assert result[1] == "--base-dir=/abs/path"

    def test_equals_form_mixed_with_split_form(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(
            ["swival", "--base-dir=proj", "--add-dir", "extra", "question"]
        )
        assert result[1] == "--base-dir=" + str((tmp_path / "proj").resolve())
        assert result[3] == str((tmp_path / "extra").resolve())

    def test_equals_form_non_path_flag_unchanged(self):
        result = _absolutize_argv(["swival", "--model=gpt-4", "question"])
        assert result[1] == "--model=gpt-4"

    def test_stops_at_double_dash_equals_form(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(
            ["swival", "--base-dir", "proj", "--", "--base-dir=subdir"]
        )
        assert result[2] == str((tmp_path / "proj").resolve())
        assert result[4] == "--base-dir=subdir"  # untouched

    def test_stops_at_double_dash_split_form(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _absolutize_argv(
            ["swival", "--add-dir", "proj", "--", "--add-dir", "rel"]
        )
        assert result[2] == str((tmp_path / "proj").resolve())
        assert result[4] == "--add-dir"  # untouched
        assert result[5] == "rel"  # untouched


# ===========================================================================
# build_agentfs_argv()
# ===========================================================================


class TestBuildArgv:
    def test_basic_argv(self, tmp_path):
        argv = build_agentfs_argv(
            agentfs_bin="/usr/local/bin/agentfs",
            base_dir=str(tmp_path),
            add_dirs=[],
            session=None,
            swival_argv=["swival", "--repl"],
        )
        assert argv[0] == "/usr/local/bin/agentfs"
        assert argv[1] == "run"
        assert "--no-default-allows" in argv
        assert "--allow" in argv
        resolved_base = str(tmp_path.resolve())
        allow_idx = argv.index("--allow")
        assert argv[allow_idx + 1] == resolved_base
        assert "--" in argv
        dash_idx = argv.index("--")
        assert argv[dash_idx + 1 :] == ["swival", "--repl"]

    def test_with_session(self, tmp_path):
        argv = build_agentfs_argv(
            agentfs_bin="agentfs",
            base_dir=str(tmp_path),
            add_dirs=[],
            session="my-session",
            swival_argv=["swival", "hello"],
        )
        session_idx = argv.index("--session")
        assert argv[session_idx + 1] == "my-session"

    def test_without_session(self, tmp_path):
        argv = build_agentfs_argv(
            agentfs_bin="agentfs",
            base_dir=str(tmp_path),
            add_dirs=[],
            session=None,
            swival_argv=["swival", "hello"],
        )
        assert "--session" not in argv

    def test_add_dirs_become_allow(self, tmp_path):
        extra1 = tmp_path / "extra1"
        extra2 = tmp_path / "extra2"
        extra1.mkdir()
        extra2.mkdir()
        argv = build_agentfs_argv(
            agentfs_bin="agentfs",
            base_dir=str(tmp_path),
            add_dirs=[str(extra1), str(extra2)],
            session=None,
            swival_argv=["swival", "q"],
        )
        allow_indices = [i for i, v in enumerate(argv) if v == "--allow"]
        assert len(allow_indices) == 3  # base_dir + 2 extras
        allow_paths = [argv[i + 1] for i in allow_indices]
        assert str(tmp_path.resolve()) in allow_paths
        assert str(extra1.resolve()) in allow_paths
        assert str(extra2.resolve()) in allow_paths

    def test_no_default_allows_always_present(self, tmp_path):
        argv = build_agentfs_argv(
            agentfs_bin="agentfs",
            base_dir=str(tmp_path),
            add_dirs=[],
            session=None,
            swival_argv=["swival"],
        )
        assert "--no-default-allows" in argv

    def test_swival_argv_preserved_exactly(self, tmp_path):
        original = ["swival", "--sandbox", "agentfs", "--model", "test", "question"]
        argv = build_agentfs_argv(
            agentfs_bin="agentfs",
            base_dir=str(tmp_path),
            add_dirs=[],
            session=None,
            swival_argv=original,
        )
        dash_idx = argv.index("--")
        assert argv[dash_idx + 1 :] == original


# ===========================================================================
# maybe_reexec()
# ===========================================================================


class TestMaybeReexec:
    def test_noop_for_builtin_mode(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        maybe_reexec(
            sandbox="builtin",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

    def test_noop_when_already_sandboxed(self, tmp_path, monkeypatch):
        _set_sandboxed(monkeypatch)
        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

    def test_raises_when_agentfs_missing(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        monkeypatch.setenv("PATH", "/nonexistent-dir-for-test")
        with pytest.raises(ConfigError, match="agentfs binary not found"):
            maybe_reexec(
                sandbox="agentfs",
                sandbox_session=None,
                base_dir=str(tmp_path),
                add_dirs=[],
            )

    def test_calls_execvpe(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "--repl"])

        captured = {}

        def fake_execvpe(file, args, env):
            captured["file"] = file
            captured["args"] = args
            captured["env"] = env

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session="test-session",
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert captured["file"].endswith("agentfs")
        assert captured["args"][1] == "run"
        assert "--no-default-allows" in captured["args"]
        assert "--session" in captured["args"]
        session_idx = captured["args"].index("--session")
        assert captured["args"][session_idx + 1] == "test-session"
        assert captured["env"][_ENV_MARKER] == "1"
        dash_idx = captured["args"].index("--")
        assert captured["args"][dash_idx + 1 :] == ["swival", "--repl"]

    def test_passes_add_dirs_as_allow(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        extra = tmp_path / "extra"
        extra.mkdir()

        captured = {}

        def fake_execvpe(file, args, env):
            captured["args"] = args

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[str(extra)],
        )

        allow_indices = [i for i, v in enumerate(captured["args"]) if v == "--allow"]
        assert len(allow_indices) == 2  # base_dir + extra
        allow_paths = [captured["args"][i + 1] for i in allow_indices]
        assert str(extra.resolve()) in allow_paths

    def test_chdir_to_base_dir_before_exec(self, tmp_path, monkeypatch):
        """Verify CWD is changed to base_dir so agentfs overlays the right directory."""
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        chdir_calls = []
        monkeypatch.setattr(os, "chdir", lambda p: chdir_calls.append(p))
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert len(chdir_calls) == 1
        assert chdir_calls[0] == str(tmp_path.resolve())

    def test_swival_marker_alone_does_not_skip_reexec(self, tmp_path, monkeypatch):
        """Setting only SWIVAL_AGENTFS_ACTIVE=1 (without AGENTFS=1) must not skip re-exec."""
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_AGENTFS_ENV, raising=False)
        monkeypatch.setenv("PATH", "/nonexistent-dir-for-test")

        with pytest.raises(ConfigError, match="agentfs binary not found"):
            maybe_reexec(
                sandbox="agentfs",
                sandbox_session=None,
                base_dir=str(tmp_path),
                add_dirs=[],
            )

    def test_relative_paths_absolutized_before_reexec(self, tmp_path, monkeypatch):
        """Relative --base-dir in argv must be resolved before chdir + re-exec."""
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))

        subdir = tmp_path / "project"
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["swival", "--base-dir", "project", "--add-dir", "project", "task"],
        )

        captured = {}

        def fake_execvpe(file, args, env):
            captured["args"] = args

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(subdir),
            add_dirs=[str(subdir)],
        )

        # The swival argv (after --) should have absolute paths
        dash_idx = captured["args"].index("--")
        child_argv = captured["args"][dash_idx + 1 :]
        bd_idx = child_argv.index("--base-dir")
        assert child_argv[bd_idx + 1] == str(subdir.resolve())
        ad_idx = child_argv.index("--add-dir")
        assert child_argv[ad_idx + 1] == str(subdir.resolve())


# ===========================================================================
# check_sandbox_available()
# ===========================================================================


class TestCheckSandboxAvailable:
    def test_raises_when_not_sandboxed(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        with pytest.raises(ConfigError, match="requires running inside an AgentFS"):
            check_sandbox_available()

    def test_ok_when_sandboxed_via_reexec(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        check_sandbox_available()  # should not raise

    def test_ok_when_wrapped_externally(self, monkeypatch):
        """External agentfs run sets only AGENTFS=1 — must be accepted."""
        _set_external_agentfs(monkeypatch)
        check_sandbox_available()  # should not raise

    def test_raises_with_only_swival_marker(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_AGENTFS_ENV, raising=False)
        with pytest.raises(ConfigError, match="requires running inside an AgentFS"):
            check_sandbox_available()


# ===========================================================================
# Session fail-fast
# ===========================================================================


class TestSessionSandboxFailFast:
    def test_session_agentfs_outside_sandbox_raises(self, tmp_path, monkeypatch):
        from swival.session import Session

        _clear_sandboxed(monkeypatch)
        sess = Session(base_dir=str(tmp_path), sandbox="agentfs", history=False)
        with pytest.raises(ConfigError, match="requires running inside an AgentFS"):
            sess._setup()

    def test_session_agentfs_with_external_wrapping_ok(self, tmp_path, monkeypatch):
        """External agentfs run wrapping must be accepted by Session."""
        from swival.session import Session

        _set_external_agentfs(monkeypatch)
        sess = Session(base_dir=str(tmp_path), sandbox="agentfs", history=False)
        # _setup will fail later (no provider), but the sandbox check should pass.
        with pytest.raises(Exception) as exc_info:
            sess._setup()
        assert "requires running inside" not in str(exc_info.value)

    def test_session_builtin_does_not_check(self, tmp_path, monkeypatch):
        """sandbox='builtin' should not trigger the agentfs check."""
        from swival.session import Session

        _clear_sandboxed(monkeypatch)
        sess = Session(base_dir=str(tmp_path), sandbox="builtin", history=False)
        with pytest.raises(Exception) as exc_info:
            sess._setup()
        assert "requires running inside" not in str(exc_info.value)


# ===========================================================================
# Config parsing integration
# ===========================================================================


class TestConfigParsing:
    def test_sandbox_default_is_builtin(self):
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.sandbox == "builtin"

    def test_sandbox_from_config(self):
        args = _make_args()
        apply_config_to_args(args, {"sandbox": "agentfs"})
        assert args.sandbox == "agentfs"

    def test_sandbox_cli_overrides_config(self):
        args = _make_args(sandbox="builtin")
        apply_config_to_args(args, {"sandbox": "agentfs"})
        assert args.sandbox == "builtin"  # CLI wins

    def test_sandbox_session_from_config(self):
        args = _make_args()
        apply_config_to_args(args, {"sandbox_session": "my-id"})
        assert args.sandbox_session == "my-id"

    def test_sandbox_session_default_is_none(self):
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.sandbox_session is None

    def test_invalid_sandbox_value_in_config(self, tmp_path, monkeypatch):
        from swival.config import _validate_config

        with pytest.raises(ConfigError, match="must be one of"):
            _validate_config({"sandbox": "docker"}, "test")


# ===========================================================================
# Report integration
# ===========================================================================


class TestReportSandboxMetadata:
    def test_report_includes_sandbox_builtin_by_default(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
        )
        assert r["sandbox"] == {"mode": "builtin"}

    def test_report_includes_sandbox_agentfs_with_session(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
            sandbox_session="abc123",
        )
        assert r["sandbox"] == {
            "mode": "agentfs",
            "session": "abc123",
            "strict_read": False,
        }

    def test_report_sandbox_no_session_key_when_none(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
        )
        assert r["sandbox"] == {"mode": "agentfs", "strict_read": False}
        assert "session" not in r["sandbox"]

    def test_finalize_passes_sandbox_through(self, tmp_path):
        rc = ReportCollector()
        r = rc.finalize(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
            sandbox_session="s1",
        )
        assert r["sandbox"]["mode"] == "agentfs"
        assert r["sandbox"]["session"] == "s1"


# ===========================================================================
# CLI integration tests (using real build_parser)
# ===========================================================================


class TestCLIIntegration:
    def test_sandbox_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--sandbox", "agentfs", "question"])
        assert args.sandbox == "agentfs"

    def test_sandbox_default_unset(self):
        parser = agent.build_parser()
        args = parser.parse_args(["question"])
        assert args.sandbox is _UNSET

    def test_sandbox_session_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(
            ["--sandbox", "agentfs", "--sandbox-session", "my-id", "question"]
        )
        assert args.sandbox_session == "my-id"

    def test_sandbox_invalid_choice_rejected(self):
        parser = agent.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--sandbox", "docker", "question"])

    def test_sandbox_builtin_explicit(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--sandbox", "builtin", "question"])
        assert args.sandbox == "builtin"


# ===========================================================================
# Integration: mock agentfs binary receives correct invocation
# ===========================================================================


class TestMockAgentfsBinary:
    def test_mock_agentfs_sees_correct_args(self, tmp_path, monkeypatch):
        """Write a mock agentfs that records its invocation, then verify."""
        log_file = tmp_path / "agentfs_log.txt"
        mock_script = tmp_path / "bin" / "agentfs"
        mock_script.parent.mkdir()
        mock_script.write_text(
            f'#!/bin/sh\necho "$@" > {log_file}\n',
            encoding="utf-8",
        )
        mock_script.chmod(mock_script.stat().st_mode | stat.S_IEXEC)

        _clear_sandboxed(monkeypatch)
        monkeypatch.setenv("PATH", str(mock_script.parent))
        monkeypatch.setattr(sys, "argv", ["swival", "--sandbox", "agentfs", "task"])

        captured = {}

        def fake_execvpe(file, args, env):
            captured["file"] = file
            captured["args"] = args
            captured["env"] = env

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session="test-session",
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert _ENV_MARKER in captured["env"]
        assert captured["env"][_ENV_MARKER] == "1"
        args = captured["args"]
        assert args[0].endswith("agentfs")
        assert args[1] == "run"
        assert "--no-default-allows" in args
        assert "--session" in args
        idx = args.index("--session")
        assert args[idx + 1] == "test-session"
        assert "--" in args
        dash = args.index("--")
        assert args[dash + 1 :] == ["swival", "--sandbox", "agentfs", "task"]

    def test_builtin_mode_run_command_unchanged(self, tmp_path, monkeypatch):
        """Verify that builtin mode doesn't invoke agentfs at all."""
        call_count = {"n": 0}

        def should_not_be_called(file, args, env):
            call_count["n"] += 1

        monkeypatch.setattr(os, "execvpe", should_not_be_called)
        _clear_sandboxed(monkeypatch)

        maybe_reexec(
            sandbox="builtin",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert call_count["n"] == 0


# ===========================================================================
# probe_agentfs()
# ===========================================================================


class TestProbeAgentfs:
    def test_parses_standard_version(self, tmp_path):
        script = _mock_agentfs_script(tmp_path, version_output="agentfs v0.6.2")
        result = probe_agentfs(script)
        assert result["version"] == "0.6.2"
        assert result["supports_strict_read"] is False

    def test_parses_version_without_v_prefix(self, tmp_path):
        script = _mock_agentfs_script(tmp_path, version_output="agentfs 0.6.2")
        result = probe_agentfs(script)
        assert result["version"] == "0.6.2"

    def test_parses_dirty_version(self, tmp_path):
        script = _mock_agentfs_script(
            tmp_path, version_output="agentfs 0.6.2-3-gabcdef-dirty"
        )
        result = probe_agentfs(script)
        assert result["version"] == "0.6.2"

    def test_returns_unknown_on_unparsable_output(self, tmp_path):
        script = _mock_agentfs_script(tmp_path, version_output="something weird")
        result = probe_agentfs(script)
        assert result["version"] == "unknown"
        assert result["supports_strict_read"] is False

    def test_returns_unknown_on_missing_binary(self):
        result = probe_agentfs("/nonexistent/binary/agentfs")
        assert result["version"] == "unknown"
        assert result["supports_strict_read"] is False

    def test_supports_strict_read_always_false_today(self, tmp_path):
        script = _mock_agentfs_script(tmp_path, version_output="agentfs v99.99.99")
        result = probe_agentfs(script)
        assert result["supports_strict_read"] is False


# ===========================================================================
# get_agentfs_version()
# ===========================================================================


class TestGetAgentfsVersion:
    def test_returns_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv(_VERSION_ENV, "0.6.2")
        assert get_agentfs_version() == "0.6.2"

    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv(_VERSION_ENV, raising=False)
        assert get_agentfs_version() is None


# ===========================================================================
# Strict read validation in maybe_reexec()
# ===========================================================================


class TestStrictReadValidation:
    def test_strict_read_unsupported_raises(self, tmp_path, monkeypatch):
        """--sandbox-strict-read with no agentfs support -> ConfigError."""
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))

        with pytest.raises(ConfigError, match="strict read support"):
            maybe_reexec(
                sandbox="agentfs",
                sandbox_session=None,
                base_dir=str(tmp_path),
                add_dirs=[],
                sandbox_strict_read=True,
            )

    def test_strict_read_false_does_not_raise(self, tmp_path, monkeypatch):
        """sandbox_strict_read=False should not trigger the check."""
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: None)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        # Should not raise
        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
            sandbox_strict_read=False,
        )

    def test_strict_read_error_includes_version(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path, version_output="agentfs v0.7.0")
        monkeypatch.setenv("PATH", str(tmp_path))

        with pytest.raises(ConfigError, match="0.7.0"):
            maybe_reexec(
                sandbox="agentfs",
                sandbox_session=None,
                base_dir=str(tmp_path),
                add_dirs=[],
                sandbox_strict_read=True,
            )

    def test_strict_read_noop_for_builtin(self, tmp_path, monkeypatch):
        """sandbox_strict_read is ignored when sandbox is not agentfs."""
        _clear_sandboxed(monkeypatch)
        # Should not raise even with strict_read=True because sandbox != agentfs
        maybe_reexec(
            sandbox="builtin",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
            sandbox_strict_read=True,
        )


# ===========================================================================
# Version env var propagation during re-exec
# ===========================================================================


class TestVersionPropagation:
    def test_version_env_set_during_reexec(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path, version_output="agentfs v0.8.1")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        captured = {}

        def fake_execvpe(file, args, env):
            captured["env"] = env

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert captured["env"][_VERSION_ENV] == "0.8.1"

    def test_unknown_version_propagated(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_agentfs_script(tmp_path, version_output="garbage")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        captured = {}

        def fake_execvpe(file, args, env):
            captured["env"] = env

        monkeypatch.setattr(os, "execvpe", fake_execvpe)
        monkeypatch.setattr(os, "chdir", lambda p: None)

        maybe_reexec(
            sandbox="agentfs",
            sandbox_session=None,
            base_dir=str(tmp_path),
            add_dirs=[],
        )

        assert captured["env"][_VERSION_ENV] == "unknown"


# ===========================================================================
# Report: strict_read and agentfs_version
# ===========================================================================


class TestStrictReadReport:
    def test_agentfs_report_includes_strict_read(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
            sandbox_strict_read=True,
            agentfs_version="0.6.2",
        )
        assert r["sandbox"]["strict_read"] is True
        assert r["sandbox"]["agentfs_version"] == "0.6.2"

    def test_builtin_report_omits_strict_read(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="builtin",
        )
        assert "strict_read" not in r["sandbox"]
        assert "agentfs_version" not in r["sandbox"]

    def test_agentfs_version_omitted_when_none(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
        )
        assert r["sandbox"]["strict_read"] is False
        assert "agentfs_version" not in r["sandbox"]

    def test_finalize_passes_strict_read_through(self):
        rc = ReportCollector()
        r = rc.finalize(
            task="test",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            sandbox_mode="agentfs",
            sandbox_strict_read=True,
            agentfs_version="0.7.0",
        )
        assert r["sandbox"]["strict_read"] is True
        assert r["sandbox"]["agentfs_version"] == "0.7.0"


# ===========================================================================
# CLI: --sandbox-strict-read
# ===========================================================================


class TestStrictReadCLI:
    def test_flag_parsed_as_store_true(self):
        parser = agent.build_parser()
        args = parser.parse_args(
            ["--sandbox", "agentfs", "--sandbox-strict-read", "question"]
        )
        assert args.sandbox_strict_read is True

    def test_default_is_unset(self):
        parser = agent.build_parser()
        args = parser.parse_args(["question"])
        assert args.sandbox_strict_read is _UNSET

    def test_config_default_is_false(self):
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.sandbox_strict_read is False

    def test_config_sets_value(self):
        args = _make_args()
        apply_config_to_args(args, {"sandbox_strict_read": True})
        assert args.sandbox_strict_read is True
