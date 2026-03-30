"""Tests for the lifecycle hook system."""

import os
import stat
import subprocess
import types

import pytest

from swival.lifecycle import (
    LifecycleError,
    _git_metadata,
    _normalize_remote,
    build_hook_env,
    run_lifecycle_hook,
)


class TestNormalizeRemote:
    def test_ssh(self):
        assert _normalize_remote("git@github.com:org/repo.git") == "github.com/org/repo"

    def test_https(self):
        assert (
            _normalize_remote("https://github.com/org/repo.git")
            == "github.com/org/repo"
        )

    def test_http(self):
        assert _normalize_remote("http://github.com/org/repo") == "github.com/org/repo"

    def test_ssh_url(self):
        assert (
            _normalize_remote("ssh://git@github.com/org/repo.git")
            == "github.com/org/repo"
        )

    def test_no_dot_git(self):
        assert _normalize_remote("git@github.com:org/repo") == "github.com/org/repo"

    def test_passthrough(self):
        assert _normalize_remote("some-custom-thing") == "some-custom-thing"


class TestGitMetadata:
    def test_inside_git_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )

        meta = _git_metadata(str(tmp_path))
        assert meta["git_present"] == "1"
        assert meta["repo_root"] == str(tmp_path)
        assert len(meta["git_head"]) == 40
        assert meta["git_dirty"] == "0"
        assert meta["project_rel"] == ""

    def test_outside_git_repo(self, tmp_path):
        # Create a dir outside any git repo
        non_git = tmp_path / "not-a-repo"
        non_git.mkdir()
        meta = _git_metadata(str(non_git))
        assert meta["git_present"] == "0"
        assert "repo_root" not in meta

    def test_dirty_staged(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )
        (tmp_path / "dirty.txt").write_text("dirty")
        subprocess.run(["git", "add", "dirty.txt"], cwd=tmp_path, capture_output=True)

        meta = _git_metadata(str(tmp_path))
        assert meta["git_dirty"] == "1"

    def test_dirty_untracked(self, tmp_path):
        """Untracked files should also make the repo dirty."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )
        (tmp_path / "new_file.txt").write_text("untracked")

        meta = _git_metadata(str(tmp_path))
        assert meta["git_dirty"] == "1"

    def test_nested_project(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )
        nested = tmp_path / "services" / "api"
        nested.mkdir(parents=True)

        meta = _git_metadata(str(nested))
        assert meta["project_rel"] == "services/api"

    def test_with_origin(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:org/repo.git"],
            cwd=tmp_path,
            capture_output=True,
        )

        meta = _git_metadata(str(tmp_path))
        assert meta["git_remote"] == "git@github.com:org/repo.git"
        assert "repo_hash" in meta
        assert "project_hash" in meta
        assert len(meta["repo_hash"]) == 48


class TestBuildHookEnv:
    def test_startup_env(self, tmp_path):
        env = build_hook_env(
            event="startup",
            resolved_base=str(tmp_path),
            provider="lmstudio",
            model="test-model",
            git_meta={"git_present": "0"},
        )
        assert env["SWIVAL_HOOK_EVENT"] == "startup"
        assert env["SWIVAL_BASE_DIR"] == str(tmp_path.resolve())
        assert env["SWIVAL_PROVIDER"] == "lmstudio"
        assert env["SWIVAL_MODEL"] == "test-model"
        assert env["SWIVAL_GIT_PRESENT"] == "0"
        # Exit-only vars should be absent
        assert "SWIVAL_REPORT" not in env
        assert "SWIVAL_OUTCOME" not in env

    def test_exit_env(self, tmp_path):
        env = build_hook_env(
            event="exit",
            resolved_base=str(tmp_path),
            git_meta={"git_present": "1", "git_head": "abc123"},
            report_path="/tmp/report.json",
            outcome="success",
            exit_code=0,
        )
        assert env["SWIVAL_HOOK_EVENT"] == "exit"
        assert env["SWIVAL_REPORT"] == "/tmp/report.json"
        assert env["SWIVAL_OUTCOME"] == "success"
        assert env["SWIVAL_EXIT_CODE"] == "0"
        assert env["SWIVAL_GIT_HEAD"] == "abc123"


class TestRunLifecycleHook:
    def _make_hook(self, tmp_path, script_body):
        hook = tmp_path / "hook.sh"
        hook.write_text(f"#!/bin/sh\n{script_body}\n")
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
        return str(hook)

    def test_success(self, tmp_path):
        hook = self._make_hook(tmp_path, 'echo "ok"')
        result = run_lifecycle_hook(
            hook,
            "startup",
            str(tmp_path),
            git_meta={"git_present": "0"},
        )
        assert result["exit_code"] == 0
        assert result["error"] is None
        assert "ok" in result["stdout"]
        assert result["duration"] > 0

    def test_failure_open(self, tmp_path):
        hook = self._make_hook(tmp_path, "exit 1")
        result = run_lifecycle_hook(
            hook,
            "startup",
            str(tmp_path),
            fail_closed=False,
            git_meta={"git_present": "0"},
        )
        assert result["exit_code"] == 1
        assert result["error"] is not None

    def test_failure_closed(self, tmp_path):
        hook = self._make_hook(tmp_path, "exit 1")
        with pytest.raises(LifecycleError):
            run_lifecycle_hook(
                hook,
                "startup",
                str(tmp_path),
                fail_closed=True,
                git_meta={"git_present": "0"},
            )

    def test_timeout(self, tmp_path):
        hook = self._make_hook(tmp_path, "sleep 60")
        result = run_lifecycle_hook(
            hook,
            "startup",
            str(tmp_path),
            timeout=1,
            fail_closed=False,
            git_meta={"git_present": "0"},
        )
        assert result["error"] is not None
        assert "timed out" in result["error"]

    def test_timeout_fail_closed(self, tmp_path):
        hook = self._make_hook(tmp_path, "sleep 60")
        with pytest.raises(LifecycleError, match="timed out"):
            run_lifecycle_hook(
                hook,
                "startup",
                str(tmp_path),
                timeout=1,
                fail_closed=True,
                git_meta={"git_present": "0"},
            )

    def test_missing_executable(self, tmp_path):
        result = run_lifecycle_hook(
            "/nonexistent/hook",
            "startup",
            str(tmp_path),
            fail_closed=False,
            git_meta={"git_present": "0"},
        )
        assert result["error"] is not None
        assert "failed to start" in result["error"]

    def test_env_vars_passed(self, tmp_path):
        hook = self._make_hook(
            tmp_path, 'echo "$SWIVAL_HOOK_EVENT $SWIVAL_GIT_PRESENT"'
        )
        result = run_lifecycle_hook(
            hook,
            "startup",
            str(tmp_path),
            git_meta={"git_present": "1"},
        )
        assert "startup 1" in result["stdout"]

    def test_receives_event_and_basedir_args(self, tmp_path):
        hook = self._make_hook(tmp_path, 'echo "$1 $2"')
        result = run_lifecycle_hook(
            hook,
            "exit",
            str(tmp_path),
            git_meta={"git_present": "0"},
        )
        assert "exit" in result["stdout"]
        assert str(tmp_path.resolve()) in result["stdout"]

    def test_hydrates_swival_dir(self, tmp_path):
        """Startup hook can write to .swival/ before Swival reads it."""
        swival_dir = tmp_path / ".swival" / "memory"
        hook_script = (
            f'mkdir -p "{swival_dir}" && echo "test memory" > "{swival_dir}/MEMORY.md"'
        )
        hook = self._make_hook(tmp_path, hook_script)
        result = run_lifecycle_hook(
            hook,
            "startup",
            str(tmp_path),
            git_meta={"git_present": "0"},
        )
        assert result["exit_code"] == 0
        assert (swival_dir / "MEMORY.md").read_text() == "test memory\n"


class TestConfigIntegration:
    def test_config_keys_include_lifecycle(self):
        from swival.config import CONFIG_KEYS, _ARGPARSE_DEFAULTS

        assert "lifecycle_command" in CONFIG_KEYS
        assert "lifecycle_timeout" in CONFIG_KEYS
        assert "lifecycle_fail_closed" in CONFIG_KEYS
        assert "no_lifecycle" in CONFIG_KEYS
        assert CONFIG_KEYS["lifecycle_command"] is str
        assert CONFIG_KEYS["lifecycle_timeout"] is int
        assert CONFIG_KEYS["lifecycle_fail_closed"] is bool
        assert CONFIG_KEYS["no_lifecycle"] is bool

        assert "lifecycle_command" in _ARGPARSE_DEFAULTS
        assert _ARGPARSE_DEFAULTS["lifecycle_timeout"] == 300

    def test_args_to_session_kwargs(self):
        from swival.config import args_to_session_kwargs

        args = types.SimpleNamespace(
            provider="lmstudio",
            model=None,
            api_key=None,
            base_url=None,
            max_turns=100,
            max_output_tokens=32768,
            max_context_tokens=None,
            temperature=None,
            top_p=1.0,
            seed=None,
            files="some",
            yolo=False,
            commands=None,
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_skills=False,
            sandbox="builtin",
            sandbox_session=None,
            sandbox_strict_read=False,
            memory_full=False,
            config_dir=None,
            proactive_summaries=False,
            extra_body=None,
            reasoning_effort=None,
            sanitize_thinking=False,
            cache=False,
            cache_dir=None,
            retries=5,
            llm_filter=None,
            encrypt_secrets=False,
            no_encrypt_secrets=False,
            encrypt_secrets_key=None,
            encrypt_secrets_tweak=None,
            encrypt_secrets_patterns=None,
            add_dir=[],
            add_dir_ro=[],
            no_read_guard=False,
            no_history=False,
            no_memory=False,
            no_continue=False,
            no_sandbox_auto_session=False,
            no_lifecycle=False,
            quiet=False,
            skills_dir=None,
            lifecycle_command="./hook.sh",
            lifecycle_timeout=60,
            lifecycle_fail_closed=True,
        )
        kwargs = args_to_session_kwargs(args, "/tmp/test")
        assert kwargs["lifecycle_command"] == "./hook.sh"
        assert kwargs["lifecycle_timeout"] == 60
        assert kwargs["lifecycle_fail_closed"] is True
        assert kwargs["lifecycle_enabled"] is True  # no_lifecycle=False -> True

    def test_config_to_session_kwargs(self):
        from swival.config import config_to_session_kwargs

        config = {
            "lifecycle_command": "./hook.sh",
            "lifecycle_timeout": 120,
            "lifecycle_fail_closed": True,
            "no_lifecycle": True,
        }
        kwargs = config_to_session_kwargs(config)
        assert kwargs["lifecycle_command"] == "./hook.sh"
        assert kwargs["lifecycle_timeout"] == 120
        assert kwargs["lifecycle_fail_closed"] is True
        assert kwargs["lifecycle_enabled"] is False  # no_lifecycle=True -> False

    def test_command_resolution(self, tmp_path):
        from swival.config import _resolve_paths

        hook = tmp_path / "hook.sh"
        hook.write_text("#!/bin/sh\n")
        config = {"lifecycle_command": "./hook.sh"}
        _resolve_paths(config, tmp_path, "test")
        assert str(tmp_path / "hook.sh") in config["lifecycle_command"]


class TestSessionLifecycle:
    def test_session_has_lifecycle_attrs(self):
        from swival.session import Session

        s = Session(
            base_dir="/tmp",
            lifecycle_command="./hook.sh",
            lifecycle_timeout=60,
            lifecycle_fail_closed=True,
            lifecycle_enabled=False,
        )
        assert s.lifecycle_command == "./hook.sh"
        assert s.lifecycle_timeout == 60
        assert s.lifecycle_fail_closed is True
        assert s.lifecycle_enabled is False

    def test_session_defaults(self):
        from swival.session import Session

        s = Session(base_dir="/tmp")
        assert s.lifecycle_command is None
        assert s.lifecycle_timeout == 300
        assert s.lifecycle_fail_closed is False
        assert s.lifecycle_enabled is True

    @staticmethod
    def _patch_provider(monkeypatch):
        from swival import agent

        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    def test_run_calls_exit_hook(self, tmp_path, monkeypatch):
        """Session.run() must call the exit hook even without context manager."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(
            f'#!/bin/sh\n[ "$1" = "exit" ] && echo "$SWIVAL_OUTCOME $SWIVAL_EXIT_CODE" > "{marker}"\n'
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        )
        result = s.run("hello")
        assert result.answer == "answer"
        assert marker.exists()
        text = marker.read_text().strip()
        assert text == "success 0"

    def test_run_exit_hook_gets_exhausted_outcome(self, tmp_path, monkeypatch):
        """Exit hook receives outcome=exhausted when turns are exhausted."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(
            f'#!/bin/sh\n[ "$1" = "exit" ] && echo "$SWIVAL_OUTCOME $SWIVAL_EXIT_CODE" > "{marker}"\n'
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        call_count = 0

        def _exhaust_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tc = types.SimpleNamespace(
                id=f"tc{call_count}",
                function=types.SimpleNamespace(
                    name="think", arguments='{"thought": "thinking"}'
                ),
            )
            msg = types.SimpleNamespace(content=None, tool_calls=[tc], role="assistant")
            return msg, "tool_calls"

        monkeypatch.setattr(agent, "call_llm", _exhaust_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            max_turns=1,
            history=False,
        )
        result = s.run("hello")
        assert result.exhausted
        assert marker.exists()
        text = marker.read_text().strip()
        assert text == "exhausted 2"

    def test_exit_hook_fail_closed_raises_from_run(self, tmp_path, monkeypatch):
        """In fail-closed mode, a failing exit hook raises LifecycleError from run()."""
        from swival.session import Session
        from swival.lifecycle import LifecycleError
        from swival import agent

        self._patch_provider(monkeypatch)

        hook = tmp_path / "hook.sh"
        hook.write_text('#!/bin/sh\nif [ "$1" = "exit" ]; then exit 1; fi\nexit 0\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            lifecycle_fail_closed=True,
            history=False,
        )
        with pytest.raises(LifecycleError):
            s.run("hello")

    def test_exit_hook_idempotent(self, tmp_path, monkeypatch):
        """Exit hook runs only once even if __exit__ is also called."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        counter = tmp_path / "counter"
        counter.write_text("0")
        hook = tmp_path / "hook.sh"
        hook.write_text(
            f'#!/bin/sh\n[ "$1" = "exit" ] && echo $(( $(cat "{counter}") + 1 )) > "{counter}"\n'
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        with Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        ) as s:
            s.run("hello")
        # run() called exit, __exit__ skipped it
        assert counter.read_text().strip() == "1"

    def test_context_manager_runs_exit_hook_without_run(self, tmp_path, monkeypatch):
        """__exit__ runs the exit hook if run()/ask() was never called."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(f'#!/bin/sh\n[ "$1" = "exit" ] && touch "{marker}"\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        with Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        ) as s:
            s._setup()  # triggers startup hook
        assert marker.exists()

    def test_run_calls_exit_hook_on_exception(self, tmp_path, monkeypatch):
        """Session.run() calls the exit hook even when the agent loop raises."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(
            f'#!/bin/sh\n[ "$1" = "exit" ] && echo "$SWIVAL_OUTCOME $SWIVAL_EXIT_CODE" > "{marker}"\n'
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _exploding_llm(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(agent, "call_llm", _exploding_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        )
        with pytest.raises(RuntimeError, match="boom"):
            s.run("hello")
        assert marker.exists()
        text = marker.read_text().strip()
        assert text == "error 1"

    def test_ask_exit_hook_via_close(self, tmp_path, monkeypatch):
        """Session.close() after ask() runs the exit hook."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(
            f'#!/bin/sh\n[ "$1" = "exit" ] && echo "$SWIVAL_OUTCOME $SWIVAL_EXIT_CODE" > "{marker}"\n'
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        )
        result = s.ask("hello")
        assert result.answer == "answer"
        assert not marker.exists()  # ask() does not run exit hook

        s.close(outcome="success", exit_code=0)
        assert marker.exists()
        text = marker.read_text().strip()
        assert text == "success 0"

    def test_ask_exit_hook_via_context_manager(self, tmp_path, monkeypatch):
        """Context manager runs exit hook after ask() calls."""
        from swival.session import Session
        from swival import agent

        self._patch_provider(monkeypatch)

        marker = tmp_path / "exit_ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(f'#!/bin/sh\n[ "$1" = "exit" ] && touch "{marker}"\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        with Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            history=False,
        ) as s:
            s.ask("hello")
            assert not marker.exists()
        assert marker.exists()

    def test_exit_fail_closed_propagates_from_context_manager(
        self, tmp_path, monkeypatch
    ):
        """__exit__ raises LifecycleError in fail-closed mode when no other exception."""
        from swival.session import Session
        from swival.lifecycle import LifecycleError
        from swival import agent

        self._patch_provider(monkeypatch)

        hook = tmp_path / "hook.sh"
        hook.write_text('#!/bin/sh\nif [ "$1" = "exit" ]; then exit 1; fi\nexit 0\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        with pytest.raises(LifecycleError):
            with Session(
                base_dir=str(tmp_path),
                lifecycle_command=str(hook),
                lifecycle_fail_closed=True,
                history=False,
            ) as s:
                s.ask("hello")

    def test_close_cleans_up_on_fail_closed_error(self, tmp_path, monkeypatch):
        """close() releases resources even when the exit hook raises."""
        from swival.session import Session
        from swival.lifecycle import LifecycleError
        from swival import agent

        self._patch_provider(monkeypatch)

        hook = tmp_path / "hook.sh"
        hook.write_text('#!/bin/sh\nif [ "$1" = "exit" ]; then exit 1; fi\nexit 0\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        s = Session(
            base_dir=str(tmp_path),
            lifecycle_command=str(hook),
            lifecycle_fail_closed=True,
            history=False,
        )
        s.ask("hello")
        with pytest.raises(LifecycleError):
            s.close()
        # Resources should be cleaned up despite the error
        assert s._llm_cache is None
        assert s._secret_shield is None

    def test_context_manager_cleans_up_on_fail_closed(self, tmp_path, monkeypatch):
        """__exit__ cleans up resources even when re-raising LifecycleError."""
        from swival.session import Session
        from swival.lifecycle import LifecycleError
        from swival import agent

        self._patch_provider(monkeypatch)

        hook = tmp_path / "hook.sh"
        hook.write_text('#!/bin/sh\nif [ "$1" = "exit" ]; then exit 1; fi\nexit 0\n')
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)

        def _simple_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _simple_llm)

        s = None
        with pytest.raises(LifecycleError):
            with Session(
                base_dir=str(tmp_path),
                lifecycle_command=str(hook),
                lifecycle_fail_closed=True,
                history=False,
            ) as s:
                s.ask("hello")
        assert s._llm_cache is None
        assert s._secret_shield is None


class TestReportLifecycle:
    def test_record_lifecycle(self):
        from swival.report import ReportCollector

        r = ReportCollector()
        r.record_lifecycle(
            {
                "event": "startup",
                "exit_code": 0,
                "duration": 1.234,
            }
        )
        assert len(r.lifecycle_events) == 1
        assert r.lifecycle_events[0]["event"] == "startup"
        assert r.lifecycle_events[0]["duration_s"] == 1.234

        r.record_lifecycle(
            {
                "event": "exit",
                "exit_code": 1,
                "duration": 0.5,
                "error": "hook failed",
            }
        )
        assert len(r.lifecycle_events) == 2
        assert r.lifecycle_events[1]["error"] == "hook failed"

    def test_report_includes_lifecycle(self):
        from swival.report import ReportCollector

        r = ReportCollector()
        r.record_lifecycle(
            {
                "event": "startup",
                "exit_code": 0,
                "duration": 0.1,
            }
        )
        report = r.build_report(
            task="test",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
        )
        assert "lifecycle" in report["stats"]
        assert len(report["stats"]["lifecycle"]) == 1
