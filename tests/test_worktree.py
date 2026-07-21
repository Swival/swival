"""Tests for swival/worktree.py — shared isolated-worktree primitives."""

from __future__ import annotations

import subprocess
import threading
import types

import pytest

from swival.worktree import (
    BatchItemResult,
    ToolPolicy,
    Worktree,
    filter_tool_schemas,
    git,
    make_isolated_loop_kwargs,
    run_bounded_batch,
)


from tests.conftest import commit_file as _commit_file
from tests.conftest import init_git as _init_git


class TestGitWrapper:
    def test_returns_stripped_stdout(self, tmp_path):
        _init_git(tmp_path)
        _commit_file(tmp_path, "a.py", "x = 1\n")
        head = git(["rev-parse", "HEAD"], str(tmp_path))
        assert len(head) == 40

    def test_raises_on_failure(self, tmp_path):
        _init_git(tmp_path)
        with pytest.raises(RuntimeError, match="git rev-parse"):
            git(["rev-parse", "not-a-ref"], str(tmp_path))


class TestWorktree:
    def test_detached_at_explicit_commit(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        first = _commit_file(repo, "a.py", "old\n")
        _commit_file(repo, "a.py", "new\n")
        work = tmp_path / "wt"
        with Worktree(str(repo), work, commit=first) as wd:
            assert (wd / "a.py").read_text() == "old\n"
            assert git(["rev-parse", "HEAD"], str(wd)) == first
        assert not work.exists()

    def test_defaults_to_head(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "current\n")
        work = tmp_path / "wt"
        with Worktree(str(repo), work) as wd:
            assert (wd / "a.py").read_text() == "current\n"
        assert not work.exists()

    def test_cleanup_on_exception(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        work = tmp_path / "wt"
        with pytest.raises(ValueError):
            with Worktree(str(repo), work):
                raise ValueError("boom")
        assert not work.exists()

    def test_refuses_to_destroy_foreign_directory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        foreign = tmp_path / "wt"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("precious\n")
        with pytest.raises(RuntimeError, match="refusing to reuse"):
            with Worktree(str(repo), foreign):
                pass
        assert (foreign / "keep.txt").read_text() == "precious\n"

    def test_reclaims_stale_swival_owned_worktree(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        stale = tmp_path / "wt"
        # Simulate a crashed run: the worktree was created and marked by
        # swival but never cleaned up.
        crashed = Worktree(str(repo), stale)
        crashed.__enter__()
        (stale / "leftover.txt").write_text("junk\n")
        with Worktree(str(repo), stale) as wd:
            assert (wd / "a.py").exists()
            assert not (wd / "leftover.txt").exists()
        assert not stale.exists()

    def test_symlinked_reclaim_root_component_cannot_redirect_cleanup(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        external = tmp_path / "external"
        (external / "wt").mkdir(parents=True)
        (external / "wt" / "victim.txt").write_text("precious\n")
        state = repo / ".swival"
        state.mkdir()
        (state / "runid").symlink_to(external)
        target = state / "runid" / "wt"
        with pytest.raises(RuntimeError, match="refusing to reuse"):
            with Worktree(str(repo), target, reclaim_root=state):
                pass
        assert (external / "wt" / "victim.txt").read_text() == "precious\n"

    def test_symlinked_reclaim_root_itself_cannot_redirect_cleanup(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        external = tmp_path / "external"
        (external / "runid" / "wt").mkdir(parents=True)
        (external / "runid" / "wt" / "victim.txt").write_text("precious\n")
        (repo / ".swival").symlink_to(external)
        target = repo / ".swival" / "runid" / "wt"
        with pytest.raises(RuntimeError, match="refusing to reuse"):
            with Worktree(str(repo), target, reclaim_root=repo / ".swival"):
                pass
        assert (external / "runid" / "wt" / "victim.txt").read_text() == "precious\n"

    def test_reclaims_unregistered_debris_under_reclaim_root(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        state = repo / ".swival"
        debris = state / "runid" / "wt"
        debris.mkdir(parents=True)
        (debris / "leftover.txt").write_text("junk\n")
        with Worktree(str(repo), debris, reclaim_root=state) as wd:
            assert (wd / "a.py").exists()
            assert not (wd / "leftover.txt").exists()
        assert not debris.exists()

    def test_refuses_registered_worktree_not_created_by_swival(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        live = tmp_path / "wt"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(live), "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (live / "uncommitted.txt").write_text("work in progress\n")
        with pytest.raises(RuntimeError, match="did not create"):
            with Worktree(str(repo), live):
                pass
        assert (live / "uncommitted.txt").read_text() == "work in progress\n"
        assert (live / "a.py").exists()


class TestAuditCompatibility:
    def test_audit_worktree_reclaims_under_swival_state_dir(self, tmp_path):
        import swival.audit as audit_mod

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        _commit_file(repo, "a.py", "x\n")
        wt = audit_mod._worktree(str(repo), tmp_path / "wt")
        assert wt.reclaim_root == repo / ".swival"
        with wt as wd:
            assert (wd / "a.py").exists()
        assert not (tmp_path / "wt").exists()


class TestToolPolicy:
    def test_rejects_unlisted_tool(self, tmp_path):
        policy = ToolPolicy(allowed_tools=frozenset({"read_file"}))
        err = policy.check("write_file", {"file_path": "a.py"})
        assert err is not None and err.startswith("error:")
        assert policy.check("read_file", {"file_path": "a.py"}) is None


class TestFilterToolSchemas:
    def test_filters_by_name(self):
        tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
        kept = filter_tool_schemas(tools, frozenset({"read_file"}))
        assert [t["function"]["name"] for t in kept] == ["read_file"]

    def test_tolerates_malformed_entries(self):
        malformed = [{"weird": 1}, None, {"function": "not-a-dict"}]
        assert filter_tool_schemas(malformed, frozenset({"x"})) == []


class TestIsolatedLoopKwargs:
    def _ctx(self):
        return types.SimpleNamespace(
            loop_kwargs={
                "max_turns": 42,
                "verbose": True,
                "mcp_manager": object(),
                "report": object(),
                "cancel_flag": object(),
                "turn_state": {},
            }
        )

    def test_strips_external_state(self, tmp_path):
        kw = make_isolated_loop_kwargs(self._ctx(), tmp_path)
        assert kw["base_dir"] == str(tmp_path)
        assert kw["max_turns"] == 42
        assert kw["verbose"] is False
        for absent in ("mcp_manager", "report", "cancel_flag", "turn_state"):
            assert absent not in kw

    def test_tool_policy_injection(self, tmp_path):
        policy = ToolPolicy(allowed_tools=frozenset({"read_file"}))
        kw = make_isolated_loop_kwargs(self._ctx(), tmp_path, tool_policy=policy)
        assert kw["tool_policy"] is policy
        kw = make_isolated_loop_kwargs(self._ctx(), tmp_path)
        assert "tool_policy" not in kw


class TestRunBoundedBatch:
    def test_preserves_order(self):
        results = run_bounded_batch(lambda x: x * 2, [3, 1, 2], max_workers=2)
        assert [r.value for r in results] == [6, 2, 4]
        assert all(r.ok for r in results)

    def test_exception_becomes_typed_failure(self):
        def boom(x):
            if x == 1:
                raise RuntimeError("nope")
            return x

        results = run_bounded_batch(boom, [0, 1, 2], max_workers=2)
        assert results[0] == BatchItemResult(ok=True, value=0)
        assert not results[1].ok
        assert "nope" in results[1].error
        assert results[2].ok

    def test_cancellation_marks_items_failed(self):
        flag = threading.Event()
        flag.set()
        results = run_bounded_batch(
            lambda x: x, [1, 2], max_workers=1, cancel_flag=flag
        )
        assert all(not r.ok and r.error == "cancelled" for r in results)


class TestHandleToolCallPolicy:
    """The policy must be enforced by the tool executor, not the prompt."""

    def _call(self, name, args_json):
        return types.SimpleNamespace(
            id="tc1",
            function=types.SimpleNamespace(name=name, arguments=args_json),
        )

    def test_forbidden_tool_never_dispatches(self, tmp_path):
        from swival.agent import handle_tool_call
        from swival.thinking import ThinkingState

        policy = ToolPolicy(allowed_tools=frozenset({"read_file"}))
        tool_msg, meta = handle_tool_call(
            self._call("write_file", '{"file_path": "x.py", "content": "boom"}'),
            str(tmp_path),
            ThinkingState(verbose=False),
            False,
            tool_policy=policy,
        )
        assert tool_msg["content"].startswith("error:")
        assert meta["succeeded"] is False
        assert not (tmp_path / "x.py").exists()

    def test_metaskill_host_command_api_enforces_policy(self, tmp_path):
        """MetaskillHostAPI.command() reaches dispatch() directly, so the
        policy must be enforced there too."""
        from swival.metaskills import (
            MetaskillBudget,
            MetaskillHostAPI,
            MetaskillTrace,
        )

        policy = ToolPolicy(allowed_tools=frozenset({"run_metaskill"}))
        budget = MetaskillBudget()
        budget.start()
        api = MetaskillHostAPI(
            budget=budget,
            trace=MetaskillTrace(),
            loop_kwargs={"base_dir": str(tmp_path), "tool_policy": policy},
            tools=[],
            cancel_flag=None,
            report=None,
            verbose=False,
        )
        marker = tmp_path / "pwned.txt"
        result = api.command(["touch", str(marker)])
        assert result["ok"] is False
        assert result["result"].startswith("error:")
        assert not marker.exists()

    def test_metaskill_loop_kwargs_carry_tool_policy(self, tmp_path, monkeypatch):
        """A nested metaskill loop must inherit the outer host policy."""
        from swival import agent
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        policy = ToolPolicy(allowed_tools=frozenset({"run_metaskill"}))
        captured = {}

        def fake_dispatch(name, args, base_dir, **kwargs):
            captured["metaskill_loop_kwargs"] = kwargs.get("metaskill_loop_kwargs")
            return "ok"

        calls = iter(
            [
                (
                    types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                id="tc1",
                                function=types.SimpleNamespace(
                                    name="run_metaskill",
                                    arguments='{"name": "x", "input": {"a": 1}}',
                                ),
                            )
                        ],
                    ),
                    "tool_calls",
                ),
                (types.SimpleNamespace(content="done", tool_calls=None), "stop"),
            ]
        )
        monkeypatch.setattr(agent, "call_llm", lambda *a, **kw: next(calls))
        monkeypatch.setattr(agent, "dispatch", fake_dispatch)

        answer, exhausted = agent.run_agent_loop(
            [{"role": "user", "content": "go"}],
            [],
            api_base="http://localhost:1234",
            model_id="test-model",
            max_turns=2,
            max_output_tokens=1024,
            temperature=0.0,
            top_p=None,
            seed=None,
            context_length=None,
            base_dir=str(tmp_path),
            thinking_state=ThinkingState(),
            todo_state=TodoState(),
            resolved_commands={},
            skills_catalog={},
            skill_read_roots=[],
            extra_write_roots=[],
            files_mode="some",
            verbose=False,
            llm_kwargs={},
            tool_policy=policy,
        )
        assert answer == "done"
        assert captured["metaskill_loop_kwargs"]["tool_policy"] is policy
