"""Tests for the library API: Session, Result, swival.run()."""

import types

import pytest

import swival
from swival import Session, Result, AgentError, ConfigError
from swival import agent


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    return msg


def _simple_llm(*args, **kwargs):
    """LLM stub that returns a final text answer immediately."""
    return _make_message(content="the answer"), "stop"


def _exhausting_llm(*args, **kwargs):
    """LLM stub that always returns tool calls (never a final answer)."""
    tc = types.SimpleNamespace(
        id="tc1",
        function=types.SimpleNamespace(name="read_file", arguments='{"path": "x.txt"}'),
    )
    return _make_message(content=None, tool_calls=[tc]), "stop"


class TestResult:
    def test_fields(self):
        r = Result(answer="hello", exhausted=False, messages=[], report=None)
        assert r.answer == "hello"
        assert r.exhausted is False
        assert r.messages == []
        assert r.report is None


class TestSessionRun:
    def test_simple_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        result = s.run("What is 2+2?")

        assert isinstance(result, Result)
        assert result.answer == "the answer"
        assert result.exhausted is False
        assert len(result.messages) >= 2  # system + user + assistant

    def test_run_state_isolation(self, tmp_path, monkeypatch):
        """Each run() call gets fresh per-run state."""
        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer {call_count}"), "stop"

        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)

        r1 = s.run("first question")
        r2 = s.run("second question")

        assert r1.answer == "answer 1"
        assert r2.answer == "answer 2"
        # Messages should be independent
        assert r1.messages != r2.messages

    def test_run_with_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        result = s.run("question", report=True)

        assert result.report is not None
        assert result.report["result"]["outcome"] == "success"
        assert result.report["task"] == "question"

    def test_run_exhausted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _exhausting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            agent,
            "handle_tool_call",
            lambda *a, **k: (
                {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
                {
                    "name": "read_file",
                    "arguments": {},
                    "elapsed": 0.0,
                    "succeeded": True,
                },
            ),
        )

        s = Session(base_dir=str(tmp_path), max_turns=2, history=False)
        result = s.run("do something")

        assert result.exhausted is True

    def test_skill_read_roots_isolation(self, tmp_path, monkeypatch):
        """skill_read_roots should not leak between independent run() calls."""
        captured_roots = []

        original_run_agent_loop = agent.run_agent_loop

        def spy_loop(messages, tools, **kwargs):
            captured_roots.append(kwargs.get("skill_read_roots"))
            return original_run_agent_loop(messages, tools, **kwargs)

        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(agent, "run_agent_loop", spy_loop)

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("q1")
        s.run("q2")

        assert len(captured_roots) == 2
        assert captured_roots[0] is not captured_roots[1]  # Different list objects


class TestSessionAsk:
    def test_ask_shares_context(self, tmp_path, monkeypatch):
        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer {call_count}"), "stop"

        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)

        r1 = s.ask("first question")
        r2 = s.ask("second question")

        assert r1.answer == "answer 1"
        assert r2.answer == "answer 2"
        # Second ask should have more messages (shared context)
        assert len(r2.messages) > len(r1.messages)

    def test_reset_clears_conversation(self, tmp_path, monkeypatch):
        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer {call_count}"), "stop"

        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)

        s.ask("first question")
        r_before = s.ask("second question")
        s.reset()
        r_after = s.ask("third question")

        # After reset, message count should be similar to the first ask
        # (not accumulated from the prior conversation)
        assert len(r_after.messages) < len(r_before.messages)


class TestAskFailureRollback:
    """ask() must roll back messages on failure so the session stays usable."""

    def test_agent_error_rolls_back_messages(self, tmp_path, monkeypatch):
        call_count = 0

        def failing_after_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_message(content="answer 1"), "stop"
            raise AgentError("boom")

        monkeypatch.setattr(agent, "call_llm", failing_after_first)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        r1 = s.ask("first question")
        msg_count_after_first = len(r1.messages)

        with pytest.raises(AgentError, match="boom"):
            s.ask("second question")

        # Messages should be rolled back to state after first successful ask
        assert len(s._conv_state["messages"]) == msg_count_after_first

    def test_session_usable_after_failed_ask(self, tmp_path, monkeypatch):
        call_count = 0

        def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise AgentError("transient failure")
            return _make_message(content=f"answer {call_count}"), "stop"

        monkeypatch.setattr(agent, "call_llm", fail_then_succeed)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        r1 = s.ask("first")
        assert r1.answer == "answer 1"

        with pytest.raises(AgentError):
            s.ask("second")

        r3 = s.ask("third")
        assert r3.answer == "answer 3"
        # Third ask should build on first (shared context), not include failed second
        assert len(r3.messages) > len(r1.messages)

    def test_context_overflow_rolls_back(self, tmp_path, monkeypatch):
        """ContextOverflowError is caught internally and becomes AgentError,
        but messages should still be rolled back."""
        call_count = 0

        def overflow_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_message(content="ok"), "stop"
            from swival.agent import ContextOverflowError

            raise ContextOverflowError("context window exceeded")

        monkeypatch.setattr(agent, "call_llm", overflow_on_second)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        r1 = s.ask("first")
        msg_count = len(r1.messages)

        with pytest.raises(AgentError, match="context window exceeded"):
            s.ask("second")

        assert len(s._conv_state["messages"]) == msg_count

    def test_history_not_written_on_failure(self, tmp_path, monkeypatch):
        call_count = 0

        def fail_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_message(content="ok"), "stop"
            raise AgentError("fail")

        monkeypatch.setattr(agent, "call_llm", fail_on_second)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=True)
        s.ask("first")

        history_dir = tmp_path / ".swival"
        files_before = set(history_dir.iterdir()) if history_dir.exists() else set()

        with pytest.raises(AgentError):
            s.ask("should not be in history")

        files_after = set(history_dir.iterdir()) if history_dir.exists() else set()
        # No new history files should have been created for the failed ask
        assert files_before == files_after

    def test_inplace_mutation_rolled_back(self, tmp_path, monkeypatch):
        """If the agent loop mutates messages in place (e.g. compaction),
        the rollback must restore the original content, not just trim length."""
        call_count = 0

        def mutate_and_fail(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "answer 1", False
            # Simulate compaction mutating the system prompt in place
            messages[0]["content"] = "CORRUPTED SYSTEM PROMPT"
            # Simulate compaction inserting a summary
            messages.insert(1, {"role": "user", "content": "COMPACTION SUMMARY"})
            raise AgentError("overflow after mutation")

        monkeypatch.setattr(agent, "run_agent_loop", mutate_and_fail)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        s.ask("first question")
        original_system = s._conv_state["messages"][0]["content"]
        original_len = len(s._conv_state["messages"])

        with pytest.raises(AgentError, match="overflow after mutation"):
            s.ask("second question")

        # Both content and length must be fully restored
        assert s._conv_state["messages"][0]["content"] == original_system
        assert len(s._conv_state["messages"]) == original_len


class TestSessionAllowedDirsRo:
    def test_ro_paths_in_skill_read_roots(self, tmp_path, monkeypatch):
        """allowed_dirs_ro paths appear in skill_read_roots for each run."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()

        captured_roots = []

        original_run_agent_loop = agent.run_agent_loop

        def spy_loop(messages, tools, **kwargs):
            captured_roots.append(list(kwargs.get("skill_read_roots", [])))
            return original_run_agent_loop(messages, tools, **kwargs)

        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(agent, "run_agent_loop", spy_loop)

        s = Session(
            base_dir=str(tmp_path),
            allowed_dirs_ro=[str(ro_dir)],
            history=False,
        )
        s.run("q1")

        assert len(captured_roots) == 1
        assert ro_dir.resolve() in captured_roots[0]

    def test_ro_paths_isolation_across_runs(self, tmp_path, monkeypatch):
        """Each run() gets its own copy of skill_read_roots (no cross-run leaks)."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()

        captured_roots = []

        original_run_agent_loop = agent.run_agent_loop

        def spy_loop(messages, tools, **kwargs):
            roots = kwargs.get("skill_read_roots", [])
            captured_roots.append(roots)
            return original_run_agent_loop(messages, tools, **kwargs)

        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(agent, "run_agent_loop", spy_loop)

        s = Session(
            base_dir=str(tmp_path),
            allowed_dirs_ro=[str(ro_dir)],
            history=False,
        )
        s.run("q1")
        s.run("q2")

        assert len(captured_roots) == 2
        # Both should contain the RO dir
        assert ro_dir.resolve() in captured_roots[0]
        assert ro_dir.resolve() in captured_roots[1]
        # But they should be different list objects (isolation)
        assert captured_roots[0] is not captured_roots[1]


class TestSessionApprovedBuckets:
    def test_ask_mode_with_approved_buckets(self, tmp_path, monkeypatch):
        """approved_buckets are forwarded to CommandPolicy in ask mode."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(
            base_dir=str(tmp_path),
            commands="ask",
            approved_buckets={"git", "ls"},
            history=False,
        )
        s._setup()
        assert s._command_policy.mode == "ask"
        assert s._command_policy.approved_buckets == {"git", "ls"}

    def test_ask_mode_without_approved_buckets(self, tmp_path, monkeypatch):
        """ask mode with no approved_buckets starts with empty set."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(
            base_dir=str(tmp_path),
            commands="ask",
            history=False,
        )
        s._setup()
        assert s._command_policy.mode == "ask"
        assert s._command_policy.approved_buckets == set()

    def test_approved_bucket_allows_command(self, tmp_path, monkeypatch):
        """A pre-approved bucket should pass check() without prompting."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(
            base_dir=str(tmp_path),
            commands="ask",
            approved_buckets={"git"},
            history=False,
        )
        s._setup()
        assert s._command_policy.check(["git", "status"]) is None


class TestSessionShellAllowed:
    def test_ask_mode_shell_not_allowed(self, tmp_path, monkeypatch):
        """ask mode sets _shell_allowed=False and excludes run_shell_command."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), commands="ask", history=False)
        s._setup()
        assert s._shell_allowed is False
        tool_names = [t["function"]["name"] for t in s._tools]
        assert "run_command" in tool_names
        assert "run_shell_command" not in tool_names

    def test_all_mode_shell_allowed(self, tmp_path, monkeypatch):
        """commands='all' sets _shell_allowed=True and includes run_shell_command."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), commands="all", history=False)
        s._setup()
        assert s._shell_allowed is True
        tool_names = [t["function"]["name"] for t in s._tools]
        assert "run_shell_command" in tool_names

    def test_none_mode_shell_not_allowed(self, tmp_path, monkeypatch):
        """commands='none' sets _shell_allowed=False."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), commands="none", history=False)
        s._setup()
        assert s._shell_allowed is False


class TestConvenienceRun:
    def test_run_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        answer = swival.run("What is 2+2?", base_dir=str(tmp_path), history=False)
        assert answer == "the answer"

    def test_run_raises_on_exhaustion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _exhausting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            agent,
            "handle_tool_call",
            lambda *a, **k: (
                {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
                {
                    "name": "read_file",
                    "arguments": {},
                    "elapsed": 0.0,
                    "succeeded": True,
                },
            ),
        )

        with pytest.raises(AgentError, match="exhausted"):
            swival.run(
                "do something", base_dir=str(tmp_path), max_turns=2, history=False
            )


class TestConfigError:
    def test_missing_model_huggingface(self, tmp_path):
        s = Session(base_dir=str(tmp_path), provider="huggingface", history=False)
        with pytest.raises(ConfigError, match="--model is required"):
            s.run("hello")

    def test_missing_api_key_openrouter(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        s = Session(
            base_dir=str(tmp_path),
            provider="openrouter",
            model="test/model",
            history=False,
        )
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            s.run("hello")

    def test_config_error_is_agent_error(self):
        assert issubclass(ConfigError, AgentError)

    def test_bad_huggingface_model_format(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        s = Session(
            base_dir=str(tmp_path),
            provider="huggingface",
            model="no-slash",
            history=False,
        )
        with pytest.raises(ConfigError, match="org/model format"):
            s.run("hello")


class TestVerboseOff:
    def test_silent_by_default(self, tmp_path, monkeypatch, capsys):
        """Library mode should produce no stderr output by default."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("question")

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""


class TestSessionSubagentNotifyUser:
    """Verify that Session wires notify_user into SubagentManager correctly."""

    def _build_manager(self, tmp_path, monkeypatch, event_callback=None):
        """Return the SubagentManager constructed by Session._build_loop_kwargs."""
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        s = Session(base_dir=str(tmp_path), history=False, subagents=True)
        s._setup()
        if event_callback is not None:
            s.event_callback = event_callback
        state = s._make_per_run_state(system_content=None)
        kwargs = s._build_loop_kwargs(state)
        return kwargs["subagent_manager"]

    def test_event_callback_emits_status_update(self, tmp_path, monkeypatch):
        events = []
        mgr = self._build_manager(
            tmp_path, monkeypatch, event_callback=lambda k, d: events.append((k, d))
        )
        assert mgr._notify_user is not None
        mgr._notify_user("waiting for capacity")
        assert len(events) == 1
        assert events[0][0] == "status_update"
        assert events[0][1]["text"] == "waiting for capacity"

    def test_no_event_callback_falls_back_to_fmt_info(self, tmp_path, monkeypatch):
        from swival import fmt

        fmt_calls = []
        monkeypatch.setattr(fmt, "info", lambda msg: fmt_calls.append(msg))
        mgr = self._build_manager(tmp_path, monkeypatch)
        assert mgr._notify_user is not None
        mgr._notify_user("waiting for capacity")
        assert "waiting for capacity" in fmt_calls
