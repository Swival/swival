"""Tests for recovering from native tool-call markup leaked as plain text.

Some weak or poorly configured models emit tool-call template fragments
(`<tool_call>`, `</parameter>`, `[TOOL_CALLS]`, ...) inside `msg.content`
instead of returning structured `tool_calls`. The agent loop must not
treat such text as a final answer.
"""

import sys
import types

import pytest

from swival import agent
from swival.agent import (
    TRUNCATED_REASON_TEXTUAL_TOOL_CALL,
    _classify_textual_tool_call_leak,
    _strip_code_spans,
)
from swival.goal import GoalState
from swival.snapshot import SnapshotState
from swival.thinking import ThinkingState
from swival.todo import TodoState
from swival.tools import TOOLS


# ---------------------------------------------------------------------------
# Helpers (mirror test_truncated_response.py)
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None, role="assistant"):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = role
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name="think", arguments='{"thought": "ok"}', call_id="tc1"):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.type = "function"
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _base_args(tmp_path, question="hi", **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=None,
        seed=None,
        quiet=False,
        max_turns=10,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question=question,
        repl=False,
        max_context_tokens=None,
        commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        files="some",
        yolo=False,
        report=None,
        reviewer=None,
        version=False,
        no_read_guard=False,
        no_history=True,
        init_config=False,
        project=False,
        reviewer_mode=False,
        review_prompt=None,
        objective=None,
        verify=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _strip_code_spans
# ---------------------------------------------------------------------------


def test_strip_code_spans_preserves_indices():
    """Masked text must have the same length as the original."""
    original = "before ```\n<tool_call>\n``` after `inline` end"
    masked = _strip_code_spans(original)
    assert len(masked) == len(original)
    assert "<tool_call>" not in masked
    assert "inline" not in masked


def test_strip_code_spans_keeps_non_code_content():
    original = "before <tool_call> after"
    masked = _strip_code_spans(original)
    assert masked == original


# ---------------------------------------------------------------------------
# _classify_textual_tool_call_leak: positive cases
# ---------------------------------------------------------------------------


def test_classify_observed_stacked_trailing_markup():
    """The reported failure: stacked closing template fragments in the tail."""
    content = (
        "Let me fix the checkpoint_from_chapters function.\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )
    result = _classify_textual_tool_call_leak(content)
    assert result is not None
    reason, idx = result
    assert reason == TRUNCATED_REASON_TEXTUAL_TOOL_CALL
    # Index points at one of the trailing sentinels in the original content.
    assert idx > 0
    assert (
        content[idx:].lstrip().startswith(("</tool_call", "</parameter", "</function"))
    )


def test_classify_function_assignment_form():
    content = "Calling tool now.\n<function=edit_file>{...truncated"
    result = _classify_textual_tool_call_leak(content)
    assert result is not None
    assert result[0] == TRUNCATED_REASON_TEXTUAL_TOOL_CALL


def test_classify_llama3_python_tag_with_json_body():
    content = '<|python_tag|>{"name": "edit_file", "arguments": {}}'
    result = _classify_textual_tool_call_leak(content)
    assert result is not None
    assert result[0] == TRUNCATED_REASON_TEXTUAL_TOOL_CALL


def test_classify_mistral_tool_calls_header_with_json_body():
    content = '[TOOL_CALLS]\n[{"name": "edit_file", "arguments": {}}]'
    result = _classify_textual_tool_call_leak(content)
    assert result is not None
    assert result[0] == TRUNCATED_REASON_TEXTUAL_TOOL_CALL


def test_classify_deepseek_calls_begin_with_json_body():
    content = '<｜tool▁calls▁begin｜>{"name": "x"}'
    result = _classify_textual_tool_call_leak(content)
    assert result is not None


def test_classify_deepseek_call_begin_with_json_body():
    content = '<｜tool▁call▁begin｜>{"name": "x"}'
    result = _classify_textual_tool_call_leak(content)
    assert result is not None


def test_classify_qwen_function_args_sentinels():
    content = "✿FUNCTION✿: edit_file\n✿ARGS✿: {}"
    result = _classify_textual_tool_call_leak(content)
    assert result is not None


def test_classify_returns_correct_index_at_first_sentinel():
    prefix = "Some prose intro " * 5
    leak = "</tool_call>\n</parameter>"
    content = prefix + leak
    result = _classify_textual_tool_call_leak(content)
    assert result is not None
    _, idx = result
    assert content[idx:].startswith("</tool_call>")


# ---------------------------------------------------------------------------
# _classify_textual_tool_call_leak: negative cases
# ---------------------------------------------------------------------------


def test_classify_none_for_empty():
    assert _classify_textual_tool_call_leak(None) is None
    assert _classify_textual_tool_call_leak("") is None


def test_classify_ignores_sentinel_in_fenced_code():
    content = (
        "Here is the format Swival uses for tool calls:\n"
        "```xml\n"
        "<tool_call>\n"
        "  <function=name>...</function>\n"
        "</tool_call>\n"
        "```\n"
        "That is all."
    )
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_ignores_sentinel_in_inline_backticks():
    content = "The token `<tool_call>` and `</tool_call>` are template fragments."
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_allows_single_mention_in_middle():
    """A single sentinel deep in the middle, not in the tail, must not fire."""
    middle_mention = (
        "Some intro. " * 30
        + "Imagine an example like </tool_call>. "
        + "Some trailing prose " * 80
    )
    assert _classify_textual_tool_call_leak(middle_mention) is None


def test_classify_allows_header_mention_without_json_body():
    content = (
        "[TOOL_CALLS] is the Mistral sentinel that some servers emit as "
        "plain text. This response is purely a prose explanation."
    )
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_allows_two_header_marker_names_in_prose():
    """Two header sentinels in prose, neither followed by JSON, must not fire."""
    content = (
        "[TOOL_CALLS] and ✿FUNCTION✿: are template marker names, not an actual call."
    )
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_allows_two_qwen_marker_names_in_prose():
    content = "✿FUNCTION✿: and ✿ARGS✿: are Qwen marker names, not an actual call."
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_allows_header_followed_by_non_json_array():
    """Header sentinel followed by `[` that opens prose, not a JSON array of calls."""
    content = (
        "[TOOL_CALLS] [is a Mistral marker, not present here] this is just "
        "a prose explanation that happens to use brackets."
    )
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_fires_on_header_followed_by_object_array():
    """The real Mistral leak shape: `[TOOL_CALLS][{...}]`."""
    content = '[TOOL_CALLS] [ {"name": "x"} ]'
    assert _classify_textual_tool_call_leak(content) is not None


def test_classify_allows_xml_without_tool_call_sentinels():
    content = "<html><body><p>Hello, <em>world</em>.</p></body></html>"
    assert _classify_textual_tool_call_leak(content) is None


def test_classify_allows_single_weak_function_tag():
    content = "Math <function>f(x)=x^2</function> rendered."
    assert _classify_textual_tool_call_leak(content) is None


# ---------------------------------------------------------------------------
# Agent-loop integration: leak triggers a repair retry
# ---------------------------------------------------------------------------


def test_loop_recovers_from_textual_tool_call_leak(tmp_path, monkeypatch):
    from swival import fmt

    fmt.init(color=False)

    captured = []

    def fake_call_llm(*args, **kwargs):
        messages = args[2]
        captured.append([dict(m) if isinstance(m, dict) else m for m in messages])
        if len(captured) == 1:
            return (
                _make_message(
                    content=(
                        "Let me fix the checkpoint_from_chapters function.\n"
                        "</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    ),
                    tool_calls=None,
                ),
                "stop",
                [],
                0,
                (0, 0),
            )
        return _make_message(content="done"), "stop", [], 0, (0, 0)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(captured) == 2, "loop should have retried after the leak"
    second = captured[1]
    user_msgs = [m for m in second if m.get("role") == "user"]
    assert any(
        "tool-call markup as plain text" in (m.get("content") or "") for m in user_msgs
    ), "repair prompt should appear in the retry transcript"

    # The leaked tail must be replaced in history with the discard marker.
    assistants = [m for m in second if m.get("role") == "assistant"]
    found_trim = False
    for m in assistants:
        content = m.get("content") or ""
        if "discarded malformed textual tool-call markup" in content:
            found_trim = True
            assert "</tool_call>" not in content
            assert "</parameter>" not in content
    assert found_trim, "trimmed assistant message should be in history"


def test_loop_recovers_from_truncated_textual_tool_call(tmp_path, monkeypatch):
    from swival import fmt

    fmt.init(color=False)

    captured = []

    def fake_call_llm(*args, **kwargs):
        messages = args[2]
        captured.append([dict(m) if isinstance(m, dict) else m for m in messages])
        if len(captured) == 1:
            return (
                _make_message(
                    content=(
                        "I'll write the content now.\n\n"
                        "<tool_call>\n"
                        "<function=write_file>\n"
                        "<parameter=file_path>\n"
                        f"{tmp_path}/output/payload.txt\n"
                        "</parameter>\n"
                        "<parameter=content>\n" + "0123456789abcdef" * 256
                    ),
                    tool_calls=None,
                ),
                "length",
                [],
                0,
                (0, 0),
            )
        return _make_message(content="done"), "stop", [], 0, (0, 0)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(captured) == 2
    second = captured[1]
    users = [m.get("content", "") for m in second if m.get("role") == "user"]
    assert any("tool-call markup as plain text" in content for content in users)
    assert not any("Your response was cut off" in content for content in users)

    assistants = [m for m in second if m.get("role") == "assistant"]
    assert any(
        "discarded malformed textual tool-call markup" in (m.get("content") or "")
        for m in assistants
    )
    assert not any("<tool_call>" in (m.get("content") or "") for m in assistants)


def test_loop_caps_at_one_repair_then_raises(tmp_path, monkeypatch, capsys):
    """Two consecutive leaks must raise AgentError (main() reports + exits 1)."""
    from swival import fmt

    fmt.init(color=False)

    def fake_call_llm(*args, **kwargs):
        return (
            _make_message(
                content="leaking again\n</parameter>\n</function>\n</tool_call>",
                tool_calls=None,
            ),
            "stop",
            [],
            0,
            (0, 0),
        )

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    with pytest.raises(SystemExit) as excinfo:
        agent.main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "tool-call markup as plain text again" in err


def test_counter_resets_after_real_tool_call(tmp_path, monkeypatch):
    """Leak → real structured tool call → leak again must NOT raise.

    The repair-pending flag resets whenever a turn executes any real
    structured tool call.
    """
    from swival import fmt

    fmt.init(color=False)

    leaked = "</parameter>\n</function>\n</tool_call>"
    seq = [
        (_make_message(content=leaked, tool_calls=None), "stop"),
        (
            _make_message(
                content=None,
                tool_calls=[
                    _make_tool_call(name="think", arguments='{"thought":"ok"}')
                ],
            ),
            "stop",
        ),
        (_make_message(content=leaked, tool_calls=None), "stop"),
        (_make_message(content="all done"), "stop"),
    ]
    idx = {"i": 0}

    def fake_call_llm(*args, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return seq[i]

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)

    loop_kwargs = dict(
        api_base="http://x",
        model_id="m",
        max_turns=10,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
        context_length=128000,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=SnapshotState(),
        goal_state=GoalState(),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        commands_unrestricted=False,
        shell_allowed=False,
        verbose=False,
        llm_kwargs={"provider": "generic"},
        file_tracker=None,
        continue_here=False,
    )

    answer, _ = agent.run_agent_loop(
        [{"role": "user", "content": "q"}], TOOLS, **loop_kwargs
    )
    assert answer == "all done"
    assert idx["i"] == 4, "all four scripted turns should run"


def test_goal_loop_skips_no_tool_final_branch_on_leak(tmp_path, monkeypatch):
    """A textual leak with an active goal must not mark the goal as suppressed
    nor record the leaked text as next-step / blocker."""
    from swival import fmt

    fmt.init(color=False)

    leaked = "Working on it.\n</parameter>\n</function>\n</tool_call>"
    seq = [
        (_make_message(content=leaked, tool_calls=None), "stop"),
        (_make_message(content="all clear"), "stop"),
    ]
    idx = {"i": 0}

    def fake_call_llm(*args, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return seq[i] if i < len(seq) else seq[-1]

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)

    gs = GoalState()
    gs.create("Finish the task")

    loop_kwargs = dict(
        api_base="http://x",
        model_id="m",
        max_turns=4,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
        context_length=128000,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=SnapshotState(),
        goal_state=gs,
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        commands_unrestricted=False,
        shell_allowed=False,
        verbose=False,
        llm_kwargs={"provider": "generic"},
        file_tracker=None,
        continue_here=False,
    )

    agent.run_agent_loop([{"role": "user", "content": "q"}], TOOLS, **loop_kwargs)

    # The leaked text must not have leaked into goal state.
    rec = gs.get()
    last_next_step = getattr(rec, "last_next_step", None) if rec else None
    last_blocker = getattr(rec, "last_blocker", None) if rec else None
    assert (last_next_step or "").find("</tool_call>") < 0
    assert (last_blocker or "").find("</tool_call>") < 0


def test_clean_final_answer_unaffected(tmp_path, monkeypatch):
    """A normal text final answer must still return cleanly."""
    from swival import fmt

    fmt.init(color=False)

    def fake_call_llm(*args, **kwargs):
        return (
            _make_message(content="The answer is 42.", tool_calls=None),
            "stop",
            [],
            0,
            (0, 0),
        )

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()
