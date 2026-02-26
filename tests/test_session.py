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
