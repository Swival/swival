"""Tests for the /btw temporary-question command."""

from __future__ import annotations

import types

import pytest

from swival import agent
from swival.goal import GoalState, GoalStatus
from swival.input_commands import INPUT_COMMANDS
from swival.input_dispatch import InputContext, StepResult, parse_input_line
from swival.thinking import ThinkingState
from swival.todo import TodoState


def _make_ctx(**overrides) -> InputContext:
    kwargs = dict(
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        base_dir="/tmp",
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=False,
        continue_here=True,
        verbose=False,
        loop_kwargs={
            "model_id": "test",
            "api_base": "http://test",
            "context_length": 128000,
            "files_mode": "some",
            "compaction_state": None,
            "command_policy": types.SimpleNamespace(mode="allowlist"),
            "top_p": None,
            "seed": None,
            "llm_kwargs": {},
            "file_tracker": None,
        },
    )
    kwargs.update(overrides)
    return InputContext(**kwargs)


def _run_btw(ctx: InputContext, question: str) -> StepResult:
    parsed = parse_input_line(question)
    assert parsed.cmd == "/btw"
    result = agent.execute_input(parsed, ctx, mode="repl")
    agent._run_after_render(result)
    return result


class _RecordingSubagent:
    """Minimal stand-in with fresh_copy/shutdown for lifecycle assertions."""

    def __init__(self):
        self.shutdown_calls = 0

    def fresh_copy(self):
        return _RecordingSubagent()

    def shutdown(self):
        self.shutdown_calls += 1


class TestRegistration:
    def test_registered_as_repl_only_agent_turn(self):
        info = INPUT_COMMANDS["/btw"]
        assert info.kind == "agent_turn"
        assert info.modes == ("repl",)
        assert info.acp is False

    def test_oneshot_mode_rejects_before_executor(self, monkeypatch):
        ctx = _make_ctx()
        called = False

        def boom(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("_execute_btw must not run in one-shot mode")

        monkeypatch.setattr(agent, "_execute_btw", boom)
        result = agent.execute_input(
            parse_input_line("/btw hello"), ctx, mode="oneshot"
        )
        assert result.text == "/btw is not available in oneshot mode."
        assert result.is_error
        assert not called

    def test_programmatic_context_rejects_even_in_repl_mode(self, monkeypatch):
        ctx = _make_ctx(interactive=False)
        monkeypatch.setattr(
            agent, "_invoke_agent_turn", lambda *a, **k: ("x", False, False)
        )
        result = agent.execute_input(parse_input_line("/btw hello"), ctx, mode="repl")
        assert result.is_error
        assert "interactive REPL" in result.text

    def test_empty_question_usage_error(self):
        ctx = _make_ctx()
        result = agent.execute_input(parse_input_line("/btw   "), ctx, mode="repl")
        assert result.is_error
        assert result.text == "usage: /btw <question>"

    def test_dispatch_routes_multiline_argument(self, monkeypatch):
        ctx = _make_ctx()
        received: list[str] = []

        def fake_execute(cmd_arg, ctx_arg):
            received.append(cmd_arg)
            return StepResult(kind="agent_turn", text="ok")

        monkeypatch.setattr(agent, "_execute_btw", fake_execute)
        result = agent.execute_input(
            parse_input_line("/btw why this\nsecond line?"), ctx, mode="repl"
        )
        assert result.kind == "agent_turn"
        assert received == ["why this\nsecond line?"]


class TestTranscriptIsolation:
    @pytest.mark.parametrize("outcome", ["success", "error", "exhausted", "interrupt"])
    def test_live_transcript_deeply_unchanged(self, outcome, monkeypatch):
        ctx = _make_ctx()
        ctx.messages.append(
            {
                "role": "assistant",
                "content": "prior answer",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        },
                    }
                ],
                "nested": {"mutable": [1, 2, 3]},
            }
        )
        import copy as copy_mod

        before = copy_mod.deepcopy(ctx.messages)

        def fake_invoke(content, ctx_arg, *, goal_launch=False):
            # _execute_btw hands us a detached copy of the live transcript;
            # the question itself gets appended inside _invoke_agent_turn.
            roles = [m["role"] for m in ctx_arg.messages]
            assert roles[0] == "system"
            assert content == "why?"
            assert ctx_arg.messages[-1]["role"] != "user" or (
                ctx_arg.messages[-1]["content"] != "why?"
            )
            # Mirror production: append and replace messages in the fork.
            ctx_arg.messages.append({"role": "user", "content": "why?"})
            ctx_arg.messages.append({"role": "user", "content": "temp"})
            ctx_arg.messages[-3] = {"role": "assistant", "content": "changed"}
            # Model the real _invoke_agent_turn contract: interactive contexts
            # get AgentError/KeyboardInterrupt folded into return values.
            if outcome == "error":
                agent.fmt.error("provider exploded")
                return None, False, False
            if outcome == "interrupt":
                return None, False, True
            return f"answer-{outcome}", outcome == "exhausted", False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        monkeypatch.setattr(agent.fmt, "error", lambda t: None)
        warnings: list[str] = []
        monkeypatch.setattr(agent.fmt, "warning", lambda t: warnings.append(t))

        result = _run_btw(ctx, "/btw why?")
        assert ctx.messages == before
        if outcome == "success":
            assert result.kind == "agent_turn"
            assert result.text == "answer-success"
            assert not result.exhausted
        elif outcome == "error":
            assert result.kind == "agent_turn"
            assert result.text is None
        elif outcome == "exhausted":
            assert result.exhausted is True
        elif outcome == "interrupt":
            assert result.interrupted is True
            assert any("aborted" in w for w in warnings)

    def test_compaction_replacing_isolated_list_leaves_live_untouched(
        self, monkeypatch
    ):
        ctx = _make_ctx()
        ctx.messages.append({"role": "user", "content": "live q"})
        before = list(map(dict, ctx.messages))

        def fake_invoke(content, ctx_arg, *, goal_launch=False):
            ctx_arg.messages[:] = [{"role": "system", "content": "compacted"}]
            return "done", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw hello")
        assert [dict(m) for m in ctx.messages] == before

    def test_interrupt_while_forking_returns_aborted_turn(self, monkeypatch):
        ctx = _make_ctx()
        warnings: list[str] = []
        monkeypatch.setattr(
            agent.copy,
            "deepcopy",
            lambda messages: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(agent.fmt, "warning", warnings.append)

        result = _run_btw(ctx, "/btw stop now")
        assert result.interrupted is True
        assert result.text is None
        assert warnings == ["/btw aborted; your conversation is unchanged."]


class TestStateIsolation:
    def test_disabled_file_tracker_stays_disabled(self, monkeypatch):
        ctx = _make_ctx(file_tracker=None)

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.file_tracker is None
            assert iso_ctx.loop_kwargs["file_tracker"] is None
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw can I write?")
        assert ctx.file_tracker is None

    @pytest.mark.parametrize(
        "status", [GoalStatus.ACTIVE, GoalStatus.PAUSED, GoalStatus.COMPLETE]
    )
    def test_goal_default_turn_limit_stays_capped(self, status, monkeypatch):
        live_goal = GoalState()
        live_goal.create("ship the parser rewrite")
        if status != GoalStatus.ACTIVE:
            live_goal.set_status(status)
        ctx = _make_ctx(
            goal_state=live_goal,
            turn_state={
                "max_turns": 500,
                "turns_used": 3,
                agent._GOAL_PREVIOUS_MAX_TURNS: 100,
            },
        )

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.goal_state is not None
            assert not iso_ctx.goal_state.has_active()
            assert iso_ctx.turn_state == {"max_turns": 100, "turns_used": 0}
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw what remains?")
        assert ctx.turn_state["max_turns"] == 500
        assert ctx.turn_state[agent._GOAL_PREVIOUS_MAX_TURNS] == 100

    def test_new_goal_saves_budget_again_after_completion(self, monkeypatch):
        ctx = _make_ctx(goal_state=GoalState())
        ctx.goal_state.create("first")
        ctx.turn_state["max_turns"] = 500
        ctx.turn_state[agent._GOAL_PREVIOUS_MAX_TURNS] = 100
        ctx.goal_state.set_status(GoalStatus.COMPLETE)
        agent._teardown_goal(ctx)
        ctx.goal_state.create("second")
        agent._raise_goal_default_max_turns(ctx.turn_state)
        assert ctx.turn_state["max_turns"] == 500
        assert ctx.turn_state[agent._GOAL_PREVIOUS_MAX_TURNS] == 100

    def test_goal_budget_restores_after_completion(self, monkeypatch):
        goal = GoalState()
        ctx = _make_ctx(goal_state=goal)
        goal.create("first")
        ctx.turn_state["max_turns"] = 500
        ctx.turn_state[agent._GOAL_PREVIOUS_MAX_TURNS] = 100
        goal.set_status(GoalStatus.COMPLETE)
        agent._teardown_goal(ctx)
        assert ctx.turn_state["max_turns"] == 100
        assert agent._GOAL_PREVIOUS_MAX_TURNS not in ctx.turn_state

    def test_loop_turns_are_capped_during_automatic_goal_budget(self):
        goal = GoalState()
        goal.create("keep working")
        ctx = _make_ctx(
            goal_state=goal,
            turn_state={"max_turns": 500, agent._GOAL_PREVIOUS_MAX_TURNS: 100},
        )
        isolated = agent._build_isolated_ctx(ctx, "loop")
        assert isolated.turn_state["max_turns"] == 100
        if isolated.subagent_manager is not None:
            isolated.subagent_manager.shutdown()

    def test_state_copy_failure_returns_error(self, monkeypatch):
        ctx = _make_ctx()

        def fail_copy(value):
            if value is ctx.thinking_state:
                raise TypeError("state copy failed")
            return value

        monkeypatch.setattr(agent.copy, "deepcopy", fail_copy)
        result = _run_btw(ctx, "/btw copy state")
        assert result.is_error
        assert "could not clone isolated state" in (result.text or "")

    def test_user_turn_limit_is_not_capped_for_btw(self, monkeypatch):
        live_goal = GoalState()
        live_goal.create("ship the parser rewrite")
        ctx = _make_ctx(
            goal_state=live_goal, turn_state={"max_turns": 500, "turns_used": 3}
        )

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.turn_state == {"max_turns": 500, "turns_used": 0}
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw what remains?")

    def test_inherited_states_are_visible_and_discarded_after(self, monkeypatch):
        tracker = agent.FileAccessTracker()
        tracker.record_read("/src/a.py")
        live_thinking = ThinkingState()
        live_thinking.think_calls = 3

        ctx = _make_ctx(file_tracker=tracker, thinking_state=live_thinking)
        seen: dict = {}

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            seen["reads"] = set(iso_ctx.file_tracker.read_files)
            seen["thinks"] = iso_ctx.thinking_state.think_calls
            iso_ctx.file_tracker.record_read("/tmp/btw-only.py")
            iso_ctx.thinking_state.think_calls += 10
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw what did I read?")

        assert seen["reads"] == {"/src/a.py"}
        assert seen["thinks"] == 3
        assert tracker.read_files == {"/src/a.py"}
        assert live_thinking.think_calls == 3

    def test_todo_and_snapshot_mutations_do_not_leak_back(self, monkeypatch):
        from swival.snapshot import SnapshotState

        live_todo = TodoState()
        live_snapshot = SnapshotState()

        ctx = _make_ctx(todo_state=live_todo, snapshot_state=live_snapshot)
        todo_summary_before = live_todo.summary_line()
        snapshot_summary_before = live_snapshot.summary_line()

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            iso_ctx.todo_state.process({"action": "add", "tasks": ["injected task"]})
            iso_ctx.snapshot_state.mark_dirty("write_file")
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw todos?")

        assert live_todo.items == []
        assert live_todo.summary_line() == todo_summary_before
        assert live_snapshot.summary_line() == snapshot_summary_before
        assert not live_snapshot.dirty

    def test_goal_not_inherited_and_complete_goal_removed_from_iso_tools(
        self, monkeypatch
    ):
        live_goal = GoalState()
        live_goal.create("ship the parser rewrite")

        goal_tool = {
            "type": "function",
            "function": {"name": "complete_goal", "parameters": {"type": "object"}},
        }
        ctx = _make_ctx(goal_state=live_goal, tools=[goal_tool])

        observed: dict = {}

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            observed["has_active"] = iso_ctx.goal_state.has_active()
            observed["iso_goal_record"] = iso_ctx.goal_state.get()
            observed["iso_tool_names"] = [
                t.get("function", {}).get("name") for t in iso_ctx.tools
            ]
            observed["live_tool_names"] = [
                t.get("function", {}).get("name") for t in ctx.tools
            ]
            # Any goal activity happens on the throwaway state only.
            iso_ctx.goal_state.create("btw private objective")
            iso_ctx.goal_state.set_status(GoalStatus.COMPLETE)
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw are we done yet?")

        assert observed["has_active"] is False
        assert observed["iso_goal_record"] is None
        assert "complete_goal" not in observed["iso_tool_names"]
        assert "complete_goal" in observed["live_tool_names"]
        assert live_goal.has_active() is True
        assert live_goal.get().status == GoalStatus.ACTIVE
        assert live_goal.get().objective == "ship the parser rewrite"

    def test_turn_counts_do_not_leak_back(self, monkeypatch):
        ctx = _make_ctx()

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            iso_ctx.turn_state["turns_used"] = 9
            iso_ctx.tools.append({"type": "function", "function": {"name": "extra"}})
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw how many turns?")
        assert ctx.turn_state == {"max_turns": 10, "turns_used": 0}
        assert all(t.get("function", {}).get("name") != "extra" for t in ctx.tools)

    def test_skill_read_root_additions_do_not_leak_back(self, monkeypatch):
        live_roots = ["/skills/root"]
        ctx = _make_ctx(skill_read_roots=live_roots)

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.skill_read_roots == ["/skills/root"]
            iso_ctx.skill_read_roots.append("/tmp/btw-only-skill")
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw load that skill")
        assert ctx.skill_read_roots == ["/skills/root"]

        assert ctx.turn_state == {"max_turns": 10, "turns_used": 0}
        assert all(t.get("function", {}).get("name") != "extra" for t in ctx.tools)

    def test_no_history_or_continue_file_writes(self, tmp_path, monkeypatch):
        ctx = _make_ctx(base_dir=str(tmp_path))
        ctx.continue_here = True
        # Real REPL loop kwargs do not include this function argument.
        ctx.loop_kwargs.pop("continue_here", None)

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.no_history is True
            assert iso_ctx.continue_here is False
            assert iso_ctx.loop_kwargs["continue_here"] is False
            return "answer", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        result = _run_btw(ctx, "/btw remember this?")
        assert result.text == "answer"
        swival_dir = tmp_path / ".swival"
        assert not (swival_dir / "HISTORY.md").exists()
        assert not (swival_dir / "continue.md").exists()

    def test_btw_disables_proactive_checkpoints(self, monkeypatch):
        compaction_state = object()
        ctx = _make_ctx()
        ctx.loop_kwargs["compaction_state"] = compaction_state

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.loop_kwargs["compaction_state"] is None
            return "answer", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw checkpoint?")
        assert ctx.loop_kwargs["compaction_state"] is compaction_state


class TestCleanupAndAccounting:
    def test_real_interrupt_does_not_reconcile_twice(self, monkeypatch):
        class Manager(_RecordingSubagent):
            created: list["Manager"] = []

            def fresh_copy(self):
                copy = Manager()
                Manager.created.append(copy)
                return copy

        manager = Manager()
        ctx = _make_ctx(subagent_manager=manager)
        reconciled: list[object] = []

        monkeypatch.setattr(
            agent,
            "run_agent_loop",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(
            agent,
            "_shutdown_and_reconcile",
            lambda mgr, cost, verbose: reconciled.append(mgr),
        )
        monkeypatch.setattr(agent.fmt, "warning", lambda t: None)

        result = _run_btw(ctx, "/btw stop now")
        assert result.interrupted is True
        assert reconciled == [Manager.created[0]]

    def test_cleanup_is_deferred_until_after_render(self, monkeypatch):
        manager = _RecordingSubagent()
        manager.fresh_copy = lambda: manager
        ctx = _make_ctx(subagent_manager=manager)
        monkeypatch.setattr(
            agent, "_invoke_agent_turn", lambda *a, **k: ("ok", False, False)
        )
        result = agent.execute_input(parse_input_line("/btw answer"), ctx, mode="repl")
        assert manager.shutdown_calls == 0
        assert result.after_render is not None
        result.after_render()
        assert manager.shutdown_calls == 1

    def test_cleanup_interrupt_keeps_completed_answer(self, monkeypatch):
        class InterruptingManager(_RecordingSubagent):
            def fresh_copy(self):
                return InterruptingManager()

            def shutdown(self):
                raise KeyboardInterrupt()

        ctx = _make_ctx(subagent_manager=InterruptingManager())
        warnings: list[str] = []
        monkeypatch.setattr(
            agent, "_invoke_agent_turn", lambda *a, **k: ("ok", False, False)
        )
        monkeypatch.setattr(agent.fmt, "warning", warnings.append)

        result = _run_btw(ctx, "/btw stop safely")
        assert result.text == "ok"
        assert not result.interrupted
        assert warnings == ["/btw cleanup was interrupted; the answer was shown."]

    def test_subagent_shutdown_on_success(self, monkeypatch):
        manager = _RecordingSubagent()
        ctx = _make_ctx(subagent_manager=manager)
        shutdown_seen: list[int] = []

        real_reconcile = agent._shutdown_and_reconcile

        def spy_reconcile(mgr, cost, verbose):
            shutdown_seen.append(getattr(mgr, "shutdown_calls", -1))
            real_reconcile(mgr, cost, verbose)

        monkeypatch.setattr(agent, "_shutdown_and_reconcile", spy_reconcile)
        monkeypatch.setattr(
            agent, "_invoke_agent_turn", lambda *a, **k: ("fine", False, False)
        )
        _run_btw(ctx, "/btw spawn anything?")
        assert shutdown_seen and shutdown_seen[-1] >= 0

    def test_subagent_shutdown_on_interrupt_and_exception_paths(self, monkeypatch):
        for outcome in ("interrupt", "exception"):
            manager = _RecordingSubagent()
            ctx = _make_ctx(subagent_manager=manager)
            reconciled: list[object] = []

            monkeypatch.setattr(
                agent,
                "_shutdown_and_reconcile",
                lambda mgr, cost, verbose: reconciled.append(mgr),
            )

            def fake_invoke(content, iso_ctx, *, goal_launch=False, _o=outcome):
                seen.append(iso_ctx.subagent_manager)
                if _o == "interrupt":
                    # Real contract: KI is folded into the return value.
                    return None, False, True
                raise RuntimeError("boom")

            seen: list[object] = []
            monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
            monkeypatch.setattr(agent.fmt, "warning", lambda t: None)

            if outcome == "exception":
                with pytest.raises(RuntimeError):
                    _run_btw(ctx, "/btw crash please")
            else:
                result = _run_btw(ctx, "/btw stop now")
                assert result.interrupted is True
            assert len(seen) == 1
            assert seen[0] is not manager
            if outcome == "interrupt":
                # The real interrupt path shuts down inside _reset_subagent.
                # This fake already returns its completed result.
                assert reconciled == []
            else:
                assert reconciled == [seen[0]]

    def test_failed_agent_command_does_not_replace_copy_answer(self, monkeypatch):
        ctx = _make_ctx()
        ctx.last_answer = "previous answer"
        monkeypatch.setattr(
            agent,
            "execute_input",
            lambda *args, **kwargs: StepResult(
                kind="agent_turn", text="error: audit failed", is_error=True
            ),
        )
        result = agent.execute_input(parse_input_line("/audit"), ctx, mode="repl")
        assert result.is_error
        assert ctx.last_answer == "previous answer"

    def test_interrupted_turn_prefers_new_transcript_answer(self, monkeypatch):
        ctx = _make_ctx()
        ctx.last_answer = "old answer"
        ctx.messages.append({"role": "assistant", "content": "new answer"})

        def interrupted_turn(content, isolated, *, goal_launch=False):
            isolated.messages.append({"role": "assistant", "content": "new answer"})
            return None, False, True

        monkeypatch.setattr(agent, "_invoke_agent_turn", interrupted_turn)
        result = agent.execute_input(parse_input_line("question"), ctx, mode="repl")
        assert result.interrupted
        assert ctx.last_answer == "new answer"

    def test_answer_returned_as_agent_turn_for_renderer_and_copy(self, monkeypatch):
        ctx = _make_ctx()
        rendered: list[str] = []
        copied: list[str | None] = []

        monkeypatch.setattr(
            agent, "_invoke_agent_turn", lambda *a, **k: ("the answer", False, False)
        )
        monkeypatch.setattr(agent, "_repl_copy", copied.append)
        result = _run_btw(ctx, "/btw tell me")
        assert result.kind == "agent_turn"
        assert result.text == "the answer"

        # Mirror the REPL render branch.
        if result.kind == "agent_turn" and result.text is not None:
            rendered.append(result.text)
        assert rendered == ["the answer"]

        agent.execute_input(parse_input_line("/copy"), ctx, mode="repl")
        assert copied == ["the answer"]

    def test_exhaustion_discards_prior_answer_and_always_warns(self, monkeypatch):
        ctx = _make_ctx(verbose=False)
        ctx.messages.append({"role": "assistant", "content": "stale live answer"})
        warnings: list[str] = []

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            iso_ctx.messages[:] = [{"role": "system", "content": "compacted"}]
            return "answer retained after compaction", True, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        monkeypatch.setattr(agent.fmt, "warning", warnings.append)
        result = _run_btw(ctx, "/btw long research")
        assert result.exhausted is True
        assert result.text == "answer retained after compaction"
        assert all("Use /continue" not in warning for warning in warnings)
        assert any("discarded" in warning for warning in warnings)

    def test_accounting_objects_still_shared_and_reported(self, monkeypatch):
        session_cost = types.SimpleNamespace(usd=0.0)
        report = types.SimpleNamespace(events=[], max_turn_seen=7)

        ctx = _make_ctx(
            loop_kwargs={
                "model_id": "test",
                "api_base": "http://test",
                "context_length": 128000,
                "files_mode": "some",
                "compaction_state": None,
                "command_policy": types.SimpleNamespace(mode="allowlist"),
                "top_p": None,
                "seed": None,
                "llm_kwargs": {},
                "session_cost": session_cost,
                "report": report,
            }
        )

        def fake_invoke(content, iso_ctx, *, goal_launch=False):
            assert iso_ctx.loop_kwargs["session_cost"] is session_cost
            assert iso_ctx.loop_kwargs["report"] is report
            report.events.append("llm_call")
            return "ok", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", fake_invoke)
        _run_btw(ctx, "/btw cost check")
        assert report.events == ["llm_call"]
        assert ctx.loop_kwargs["session_cost"] is session_cost


class TestLoopBuilderCompat:
    def test_loop_builder_still_fresh_state_and_forked_messages(self):
        live = _make_ctx()
        live.messages.append({"role": "user", "content": "live marker"})
        live.thinking_state.think_calls = 5

        iso_ctx = agent._build_isolated_ctx(live, "loop")
        try:
            assert iso_ctx.messages is not live.messages
            assert iso_ctx.messages[1] is not live.messages[1]
            assert iso_ctx.messages[1]["content"] == "live marker"
            assert iso_ctx.thinking_state is not live.thinking_state
            assert iso_ctx.thinking_state.think_calls == 0
            assert iso_ctx.turn_state["turns_used"] == 0
            assert iso_ctx.loop_registry is None
        finally:
            if iso_ctx.subagent_manager is not None:
                iso_ctx.subagent_manager.shutdown()

    def test_loop_builder_does_not_touch_live_tools(self):
        """The /loop fork must not disable goal tools on the live list."""
        live = _make_ctx(
            tools=[{"type": "function", "function": {"name": "complete_goal"}}],
            goal_state=GoalState(),
        )
        agent._build_isolated_ctx(live, "loop")
        assert "complete_goal" in [
            t.get("function", {}).get("name") for t in live.tools
        ]


class TestBuilderDirectly:
    def test_shallow_fork_reuses_nested_tool_calls(self):
        ctx = _make_ctx()
        ctx.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            }
        )
        iso_ctx = agent._build_isolated_ctx(ctx, "btw")
        iso_ctx.messages[-1]["tool_calls"][0]["id"] = "mutated"
        assert ctx.messages[-1]["tool_calls"][0]["id"] == "mutated"

    def test_fresh_goal_when_goal_support_exists_but_none_active(self):
        ctx = _make_ctx(goal_state=GoalState())
        iso_ctx = agent._build_isolated_ctx(ctx, "btw")
        assert iso_ctx.goal_state is not None
        assert not iso_ctx.goal_state.has_active()
