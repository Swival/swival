"""Tests for stderr logging in agent.py."""

import json
import sys
import types


import pytest


def _make_message(content=None, tool_calls=None):
    """Build a fake LLM message object matching litellm's response shape."""
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    # Make it behave like a dict for the max_turns code path
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name, arguments, call_id="call_1"):
    """Build a fake tool_call object."""
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _base_args(tmp_path, **overrides):
    """Build a minimal args namespace with all required fields."""
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=1.0,
        seed=None,
        quiet=False,
        max_turns=10,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question="test",
        repl=False,
        max_context_tokens=None,
        allowed_commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        yolo=False,
        report=None,
        reviewer=None,
        version=False,
        no_read_guard=False,
        no_history=True,
        init_config=False,
        project=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestHappyPath:
    """Mock returns a tool call then a final text answer (exit=ok)."""

    def test_exit_ok_logging(self, tmp_path, capsys, monkeypatch):
        from swival import agent

        tool_call = _make_tool_call("read_file", '{"path": "hello.txt"}')
        msg1 = _make_message(content="Let me read that file.", tool_calls=[tool_call])
        msg2 = _make_message(content="The answer is 42.")

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msg1, "tool_calls"
            return msg2, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="What is the answer?")

        # Write a test file so the tool call succeeds
        (tmp_path / "hello.txt").write_text("The answer is 42.\n")

        monkeypatch.setattr(sys, "argv", ["agent", "What is the answer?"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        captured = capsys.readouterr()
        stderr = captured.err

        # Turn header (Rich Rule format)
        assert "Turn" in stderr
        # Tool call logged
        assert "read_file" in stderr
        # LLM timing
        assert "LLM responded in" in stderr
        # Completion
        assert "Agent finished" in stderr
        # Intermediate assistant text
        assert "[assistant]" in stderr
        assert "Let me read that file." in stderr

        # stdout should have the final answer only
        assert captured.out.strip() == "The answer is 42."


class TestMaxTurns:
    """Mock always returns tool calls, loop runs with max_turns=2 (exit=max_turns)."""

    def test_exit_max_turns(self, tmp_path, capsys, monkeypatch):
        from swival import agent

        def fake_call_llm(*args, **kwargs):
            tool_call = _make_tool_call("read_file", '{"path": "."}')
            msg = _make_message(content=None, tool_calls=[tool_call])
            return msg, "tool_calls"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="Loop forever", max_turns=2)

        monkeypatch.setattr(sys, "argv", ["agent", "Loop forever"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        stderr = captured.err

        assert "Agent finished" in stderr
        assert "max_turns" in stderr


class TestTruncatedResponse:
    """Mock returns finish_reason='length' then a normal answer."""

    def test_length_continuation(self, tmp_path, capsys, monkeypatch):
        from swival import agent

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Truncated response: content but no tool calls, finish_reason=length
                msg = _make_message(content="I was thinking about...")
                return msg, "length"
            # Second call: final answer
            msg = _make_message(content="Done.")
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="Think about something")

        monkeypatch.setattr(sys, "argv", ["agent", "Think about something"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        captured = capsys.readouterr()
        stderr = captured.err

        # Intermediate text should be logged
        assert "[assistant]" in stderr
        assert "I was thinking about..." in stderr
        # A second LLM call was made (continuation triggered)
        assert call_count == 2
        assert "Agent finished" in stderr


class TestArgTruncation:
    """Tool call arguments exceeding MAX_ARG_LOG are truncated in stderr."""

    def test_long_args_truncated(self, tmp_path, capsys, monkeypatch):
        from swival import agent

        # Build args longer than MAX_ARG_LOG (1000 chars)
        long_content = "x" * 1200
        tool_args = json.dumps({"content": long_content})
        tool_call = _make_tool_call("write_file", tool_args)
        msg1 = _make_message(tool_calls=[tool_call])
        msg2 = _make_message(content="Done.")

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msg1, "tool_calls"
            return msg2, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="Write a big file")

        monkeypatch.setattr(sys, "argv", ["agent", "Write a big file"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        captured = capsys.readouterr()
        stderr = captured.err

        assert "... (truncated)" in stderr
        assert "write_file" in stderr


class TestQuietFlag:
    """Tests for the --quiet flag (verbose by default, -q to suppress)."""

    def test_default_is_verbose(self):
        from swival import agent
        from swival.config import apply_config_to_args

        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        apply_config_to_args(args, {})  # resolve sentinels to defaults
        args.verbose = not args.quiet
        assert args.verbose is True

    def test_quiet_flag_suppresses(self):
        from swival import agent

        parser = agent.build_parser()
        args = parser.parse_args(["hello", "-q"])
        args.verbose = not args.quiet
        assert args.verbose is False

    def test_quiet_e2e_no_stderr(self, tmp_path, capsys, monkeypatch):
        """Running with --quiet produces no diagnostic output on stderr."""
        from swival import agent

        msg = _make_message(content="The answer.")

        def fake_call_llm(*args, **kwargs):
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="Hello", quiet=True)

        monkeypatch.setattr(sys, "argv", ["agent", "Hello", "-q"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.strip() == "The answer."

    def test_default_e2e_has_stderr(self, tmp_path, capsys, monkeypatch):
        """Running without --quiet produces diagnostic output on stderr."""
        from swival import agent

        msg = _make_message(content="The answer.")

        def fake_call_llm(*args, **kwargs):
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        args = _base_args(tmp_path, question="Hello")

        monkeypatch.setattr(sys, "argv", ["agent", "Hello"])
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        captured = capsys.readouterr()
        assert "Turn" in captured.err
        assert "Agent finished" in captured.err
        assert captured.out.strip() == "The answer."
