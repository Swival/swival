"""Tests for the event_callback and cancel_flag mechanism in run_agent_loop."""

import json
import threading
import types

from swival.agent import run_agent_loop
from swival.a2a_types import (
    EVENT_STATUS_UPDATE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_START,
    EVENT_TOOL_FINISH,
    EVENT_TOOL_ERROR,
)
from swival.thinking import ThinkingState
from swival.todo import TodoState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sys(content):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name, arguments, call_id="call_1"):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _loop_kwargs(tmp_path, **overrides):
    """Build the required keyword arguments for run_agent_loop."""
    defaults = dict(
        api_base="http://localhost",
        model_id="test-model",
        max_turns=5,
        max_output_tokens=1024,
        temperature=0.0,
        top_p=1.0,
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
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventCallbackNone:
    def test_event_callback_none_is_safe(self, tmp_path, monkeypatch):
        """run_agent_loop with event_callback=None should not crash."""

        def fake_call_llm(*args, **kwargs):
            return _make_message(content="final answer"), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("hello")]
        answer, exhausted = run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=1),
            event_callback=None,
        )
        assert answer == "final answer"
        assert exhausted is False


class TestStatusUpdate:
    def test_status_update_emitted_each_turn(self, tmp_path, monkeypatch):
        """status_update events are emitted at the start of each turn."""
        events = []
        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                tc = _make_tool_call(
                    "think",
                    json.dumps({"thought": "step"}),
                    call_id=f"call_{call_count}",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="done"), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("go")]
        run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=5),
            event_callback=lambda kind, data: events.append((kind, data)),
        )

        status_events = [(k, d) for k, d in events if k == EVENT_STATUS_UPDATE]
        assert len(status_events) == 3
        for i, (_, data) in enumerate(status_events, start=1):
            assert data["turn"] == i
            assert data["max_turns"] == 5
            assert "elapsed" in data
            assert isinstance(data["elapsed"], float)


class TestTextChunk:
    def test_text_chunk_emitted(self, tmp_path, monkeypatch):
        """text_chunk events fire when the LLM returns content."""
        events = []
        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = _make_tool_call(
                    "think",
                    json.dumps({"thought": "hmm"}),
                    call_id="call_1",
                )
                return _make_message(
                    content="reasoning text", tool_calls=[tc]
                ), "tool_calls"
            return _make_message(content="final"), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("do it")]
        run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=3),
            event_callback=lambda kind, data: events.append((kind, data)),
        )

        text_events = [(k, d) for k, d in events if k == EVENT_TEXT_CHUNK]
        # Only the final answer emits text_chunk; intermediate reasoning
        # (content with tool_calls) emits status_update instead.
        assert len(text_events) == 1
        assert text_events[0][1]["text"] == "final"
        assert text_events[0][1]["turn"] == 2

        # The intermediate reasoning emits a status_update with type=reasoning
        reasoning_events = [
            (k, d)
            for k, d in events
            if k == EVENT_STATUS_UPDATE and d.get("type") == "reasoning"
        ]
        assert len(reasoning_events) == 1
        assert reasoning_events[0][1]["text_length"] == len("reasoning text")


class TestToolStartFinish:
    def test_tool_start_finish_emitted(self, tmp_path, monkeypatch):
        """tool_start and tool_finish events fire around successful tool calls."""
        events = []

        def fake_call_llm(*args, **kwargs):
            tc = _make_tool_call(
                "think",
                json.dumps({"thought": "planning"}),
                call_id="call_1",
            )
            # First call returns tool call, second returns final answer
            msgs = args[2]
            if any(
                (m.get("role") if isinstance(m, dict) else getattr(m, "role", ""))
                == "tool"
                for m in msgs
            ):
                return _make_message(content="done"), "stop"
            return _make_message(tool_calls=[tc]), "tool_calls"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("plan")]
        run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=3),
            event_callback=lambda kind, data: events.append((kind, data)),
        )

        start_events = [d for k, d in events if k == EVENT_TOOL_START]
        finish_events = [d for k, d in events if k == EVENT_TOOL_FINISH]

        assert len(start_events) == 1
        assert start_events[0]["name"] == "think"
        assert start_events[0]["turn"] == 1

        assert len(finish_events) == 1
        assert finish_events[0]["name"] == "think"
        assert finish_events[0]["turn"] == 1
        assert "elapsed" in finish_events[0]
        assert isinstance(finish_events[0]["elapsed"], float)


class TestToolError:
    def test_tool_error_emitted(self, tmp_path, monkeypatch):
        """tool_error events fire when a tool call fails."""
        events = []
        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Call read_file with a path that doesn't exist
                tc = _make_tool_call(
                    "read_file",
                    json.dumps({"file_path": "/nonexistent/path/file.txt"}),
                    call_id="call_err",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="ok"), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("read it")]
        run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=3),
            event_callback=lambda kind, data: events.append((kind, data)),
        )

        error_events = [d for k, d in events if k == EVENT_TOOL_ERROR]
        assert len(error_events) == 1
        assert error_events[0]["name"] == "read_file"
        assert error_events[0]["turn"] == 1
        assert "error" in error_events[0]
        assert error_events[0]["error"].startswith("error:")


class TestCancelFlag:
    def test_cancel_flag_breaks_loop(self, tmp_path, monkeypatch):
        """When cancel_flag is set, the loop returns (None, True) immediately."""

        def fake_call_llm(*args, **kwargs):
            raise AssertionError("call_llm should not be reached")

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        cancel = threading.Event()
        cancel.set()

        messages = [_sys("system"), _user("go")]
        answer, exhausted = run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=5),
            cancel_flag=cancel,
        )

        assert answer is None
        assert exhausted is True

    def test_cancel_flag_emits_status_update(self, tmp_path, monkeypatch):
        """Cancellation emits a status_update with cancelled=True."""
        events = []

        def fake_call_llm(*args, **kwargs):
            raise AssertionError("call_llm should not be reached")

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        cancel = threading.Event()
        cancel.set()

        messages = [_sys("system"), _user("go")]
        run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=5),
            event_callback=lambda kind, data: events.append((kind, data)),
            cancel_flag=cancel,
        )

        status_events = [d for k, d in events if k == EVENT_STATUS_UPDATE]
        assert len(status_events) == 1
        assert status_events[0]["cancelled"] is True


class TestCallbackExceptionSwallowed:
    def test_callback_exception_swallowed(self, tmp_path, monkeypatch):
        """If the event callback raises, the loop still completes normally."""

        def exploding_callback(kind, data):
            raise RuntimeError("callback exploded")

        def fake_call_llm(*args, **kwargs):
            return _make_message(content="all good"), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        messages = [_sys("system"), _user("hello")]
        answer, exhausted = run_agent_loop(
            messages,
            [],
            **_loop_kwargs(tmp_path, max_turns=2),
            event_callback=exploding_callback,
        )

        assert answer == "all good"
        assert exhausted is False
