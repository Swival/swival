"""Tests for context window management: estimate_tokens, group_into_turns,
compact_messages, drop_middle_turns, clamp_output_tokens, and ContextOverflowError."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swival.agent import (
    estimate_tokens,
    group_into_turns,
    compact_messages,
    drop_middle_turns,
    clamp_output_tokens,
    ContextOverflowError,
    call_llm,
)
from swival.report import AgentError


# ---------------------------------------------------------------------------
# Helpers to build messages
# ---------------------------------------------------------------------------


def _sys(content):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def _assistant_tc(tool_calls):
    """Assistant message with tool_calls (list of (id, name, args_json))."""
    tcs = [
        SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))
        for tc_id, name, args in tool_calls
    ]
    return SimpleNamespace(role="assistant", content=None, tool_calls=tcs)


def _tool(tc_id, content):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_basic(self):
        msgs = [_user("hello world")]
        count = estimate_tokens(msgs)
        assert count > 0

    def test_includes_tool_calls(self):
        msgs_plain = [_assistant("hello")]
        msgs_tc = [_assistant_tc([("tc1", "read_file", '{"path": "foo.txt"}')])]
        # The tool call version should have more tokens than an empty-content message
        estimate_tokens(msgs_plain)
        count_tc = estimate_tokens(msgs_tc)
        assert count_tc > 4  # More than just per-message overhead

    def test_tools_schema_counted(self):
        msgs = [_user("hi")]
        tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}}
        ]
        count_no_tools = estimate_tokens(msgs)
        count_with_tools = estimate_tokens(msgs, tools)
        assert count_with_tools > count_no_tools

    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_none_content(self):
        # Assistant messages with tool_calls often have content=None
        msgs = [{"role": "assistant", "content": None}]
        count = estimate_tokens(msgs)
        assert count == 4  # Just per-message overhead

    def test_dict_tool_calls_counted(self):
        """Tool calls in dict-shaped messages should be counted too."""
        msgs_no_tc = [{"role": "assistant", "content": None}]
        msgs_with_tc = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "foo.txt"}',
                        },
                    }
                ],
            }
        ]
        count_no_tc = estimate_tokens(msgs_no_tc)
        count_with_tc = estimate_tokens(msgs_with_tc)
        assert count_with_tc > count_no_tc


# ---------------------------------------------------------------------------
# group_into_turns
# ---------------------------------------------------------------------------


class TestGroupIntoTurns:
    def test_basic(self):
        msgs = [_sys("sys"), _user("q"), _assistant("a")]
        turns = group_into_turns(msgs)
        assert len(turns) == 3
        assert all(len(t) == 1 for t in turns)

    def test_tool_calls(self):
        tc = _assistant_tc([("tc1", "read_file", "{}"), ("tc2", "grep", "{}")])
        tr1 = _tool("tc1", "content1")
        tr2 = _tool("tc2", "content2")
        msgs = [_sys("sys"), _user("q"), tc, tr1, tr2, _assistant("done")]
        turns = group_into_turns(msgs)
        assert len(turns) == 4  # sys, user, (tc+tr1+tr2), assistant
        assert len(turns[2]) == 3  # assistant + 2 tool results

    def test_partial_orphaned_tool_result(self):
        # A tool result without a preceding assistant with matching tool_calls
        # should be kept as a standalone turn (defensive)
        orphan = _tool("tc_orphan", "data")
        msgs = [_user("q"), orphan]
        turns = group_into_turns(msgs)
        assert len(turns) == 2
        assert turns[1] == [orphan]


# ---------------------------------------------------------------------------
# compact_messages
# ---------------------------------------------------------------------------


class TestCompactMessages:
    def test_truncates_large_results(self):
        tc = _assistant_tc([("tc1", "read_file", "{}")])
        big_content = "x" * 2000
        tr = _tool("tc1", big_content)
        # Add another turn after so the tool turn is not in the last 2
        msgs = [_sys("sys"), _user("q"), tc, tr, _assistant("mid"), _assistant("done")]
        result = compact_messages(msgs)
        # Find the tool result
        tool_msgs = [
            m
            for m in result
            if (m.get("role") if isinstance(m, dict) else None) == "tool"
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"].startswith("[compacted")
        assert "2000" in tool_msgs[0]["content"]

    def test_preserves_recent_turns(self):
        """Last 2 turns should not be compacted."""
        tc1 = _assistant_tc([("tc1", "f", "{}")])
        tr1 = _tool("tc1", "x" * 2000)
        tc2 = _assistant_tc([("tc2", "f", "{}")])
        tr2 = _tool("tc2", "y" * 2000)
        msgs = [_sys("s"), _user("q"), tc1, tr1, tc2, tr2]
        result = compact_messages(msgs)
        # tc2+tr2 is the last turn, tc1+tr1 is second-to-last
        # Both are in the last 2 turns, so neither should be compacted
        tool_msgs = [
            m
            for m in result
            if (m.get("role") if isinstance(m, dict) else None) == "tool"
        ]
        for tm in tool_msgs:
            assert not tm["content"].startswith("[compacted")

    def test_preserves_turn_atomicity(self):
        tc = _assistant_tc([("tc1", "read_file", "{}"), ("tc2", "grep", "{}")])
        tr1 = _tool("tc1", "x" * 2000)
        tr2 = _tool("tc2", "short")
        # Ensure this turn is not in the last 2
        msgs = [_sys("s"), _user("q"), tc, tr1, tr2, _assistant("a"), _assistant("b")]
        result = compact_messages(msgs)
        # Both the assistant with tool_calls and tool results should still be present
        turns = group_into_turns(result)
        # Find the turn with tool calls
        tc_turn = [t for t in turns if len(t) > 1]
        assert len(tc_turn) == 1
        assert len(tc_turn[0]) == 3  # assistant + 2 tool results


# ---------------------------------------------------------------------------
# drop_middle_turns
# ---------------------------------------------------------------------------


class TestDropMiddleTurns:
    def test_keeps_boundaries(self):
        tc1 = _assistant_tc([("tc1", "f", "{}")])
        tr1 = _tool("tc1", "result1")
        tc2 = _assistant_tc([("tc2", "f", "{}")])
        tr2 = _tool("tc2", "result2")
        tc3 = _assistant_tc([("tc3", "f", "{}")])
        tr3 = _tool("tc3", "result3")
        tc4 = _assistant_tc([("tc4", "f", "{}")])
        tr4 = _tool("tc4", "result4")
        msgs = [_sys("sys"), _user("q"), tc1, tr1, tc2, tr2, tc3, tr3, tc4, tr4]
        result = drop_middle_turns(msgs)
        # Should have: sys, user, splice marker, last 3 turns (tc2+tr2, tc3+tr3, tc4+tr4)
        roles = []
        for m in result:
            r = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            roles.append(r)
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[2] == "user"  # splice marker
        assert "[context compacted" in result[2]["content"]

    def test_no_system(self):
        """Works correctly when there's no system message."""
        tc1 = _assistant_tc([("tc1", "f", "{}")])
        tr1 = _tool("tc1", "r1")
        tc2 = _assistant_tc([("tc2", "f", "{}")])
        tr2 = _tool("tc2", "r2")
        tc3 = _assistant_tc([("tc3", "f", "{}")])
        tr3 = _tool("tc3", "r3")
        tc4 = _assistant_tc([("tc4", "f", "{}")])
        tr4 = _tool("tc4", "r4")
        msgs = [_user("q"), tc1, tr1, tc2, tr2, tc3, tr3, tc4, tr4]
        result = drop_middle_turns(msgs)
        # Leading block is just the user message
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "q"
        # Splice marker
        assert "[context compacted" in result[1]["content"]

    def test_preserves_turn_atomicity(self):
        tc1 = _assistant_tc([("tc1", "f", "{}"), ("tc1b", "g", "{}")])
        tr1a = _tool("tc1", "r1")
        tr1b = _tool("tc1b", "r1b")
        tc2 = _assistant_tc([("tc2", "f", "{}")])
        tr2 = _tool("tc2", "r2")
        tc3 = _assistant_tc([("tc3", "f", "{}")])
        tr3 = _tool("tc3", "r3")
        tc4 = _assistant_tc([("tc4", "f", "{}")])
        tr4 = _tool("tc4", "r4")
        msgs = [_sys("s"), _user("q"), tc1, tr1a, tr1b, tc2, tr2, tc3, tr3, tc4, tr4]
        result = drop_middle_turns(msgs)
        # Verify no orphaned tool results
        _validate_tool_pairing(result)

    def test_small_history(self):
        """When history is too small for a middle, returns unchanged."""
        msgs = [_sys("s"), _user("q"), _assistant("a")]
        result = drop_middle_turns(msgs)
        assert len(result) == 3
        # No splice marker
        for m in result:
            if isinstance(m, dict) and m.get("content"):
                assert "[context compacted" not in m["content"]


# ---------------------------------------------------------------------------
# clamp_output_tokens
# ---------------------------------------------------------------------------


class TestClampOutputTokens:
    def test_basic_clamping(self):
        msgs = [_user("hello " * 100)]  # Should be a decent number of tokens
        # With a tight context_length, output should be clamped
        result = clamp_output_tokens(msgs, None, 200, 16384)
        assert result < 16384
        assert result > 0

    def test_none_context_length(self):
        result = clamp_output_tokens([_user("hi")], None, None, 16384)
        assert result == 16384

    def test_available_less_than_one(self):
        # When prompt is larger than context, return 1 (minimal budget)
        msgs = [_user("x " * 10000)]
        result = clamp_output_tokens(msgs, None, 10, 16384)
        assert result == 1

    def test_no_clamping_when_room(self):
        msgs = [_user("hi")]
        result = clamp_output_tokens(msgs, None, 100000, 16384)
        assert result == 16384


# ---------------------------------------------------------------------------
# Integration: compacted messages valid for API
# ---------------------------------------------------------------------------


def _validate_tool_pairing(messages):
    """Validate that every tool result has a matching tool_call_id in a preceding assistant message."""
    # Collect all tool_call_ids from assistant messages
    available_tc_ids = set()
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "assistant":
            tcs = (
                m.get("tool_calls", None)
                if isinstance(m, dict)
                else getattr(m, "tool_calls", None)
            )
            if tcs:
                for tc in tcs:
                    tc_id = tc.id if hasattr(tc, "id") else tc["id"]
                    available_tc_ids.add(tc_id)
        elif role == "tool":
            tc_id = (
                m.get("tool_call_id")
                if isinstance(m, dict)
                else getattr(m, "tool_call_id", None)
            )
            assert tc_id in available_tc_ids, (
                f"Orphaned tool result with tool_call_id={tc_id}"
            )


class TestIntegration:
    def _build_realistic_history(self):
        """Build a realistic message sequence with multiple tool turns."""
        msgs = [
            _sys("You are a helpful assistant."),
            _user("Read foo.txt and bar.txt"),
        ]
        # Turn 1: read_file foo.txt
        tc1 = _assistant_tc([("tc1", "read_file", '{"path": "foo.txt"}')])
        tr1 = _tool("tc1", "contents of foo " * 100)
        msgs.extend([tc1, tr1])
        # Turn 2: read_file bar.txt
        tc2 = _assistant_tc([("tc2", "read_file", '{"path": "bar.txt"}')])
        tr2 = _tool("tc2", "contents of bar " * 100)
        msgs.extend([tc2, tr2])
        # Turn 3: grep
        tc3 = _assistant_tc([("tc3", "grep", '{"pattern": "TODO"}')])
        tr3 = _tool("tc3", "line1: TODO fix\nline2: TODO refactor\n" * 50)
        msgs.extend([tc3, tr3])
        # Turn 4: write_file
        tc4 = _assistant_tc(
            [("tc4", "write_file", '{"path": "out.txt", "content": "done"}')]
        )
        tr4 = _tool("tc4", "ok")
        msgs.extend([tc4, tr4])
        # Final assistant
        msgs.append(_assistant("I've completed the task."))
        return msgs

    def test_compact_then_valid_for_api(self):
        msgs = self._build_realistic_history()
        result = compact_messages(msgs)
        _validate_tool_pairing(result)

    def test_drop_then_valid_for_api(self):
        msgs = self._build_realistic_history()
        result = drop_middle_turns(msgs)
        _validate_tool_pairing(result)


# ---------------------------------------------------------------------------
# ContextOverflowError classifier
# ---------------------------------------------------------------------------


class TestContextOverflowClassifier:
    def test_typed_exception(self):
        """call_llm raises ContextOverflowError for litellm.ContextWindowExceededError."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.ContextWindowExceededError(
                message="context length exceeded",
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(ContextOverflowError):
                call_llm("http://localhost", "model", [], 100, 0.1, 1.0, None, None, False)

    def test_bad_request_with_context_keywords(self):
        """call_llm raises ContextOverflowError for BadRequestError with context keywords."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.BadRequestError(
                message="maximum context length exceeded",
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(ContextOverflowError):
                call_llm("http://localhost", "model", [], 100, 0.1, 1.0, None, None, False)

    def test_bad_request_without_context_keywords(self):
        """call_llm raises AgentError for BadRequestError without context keywords."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.BadRequestError(
                message="invalid request format",
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(AgentError):
                call_llm("http://localhost", "model", [], 100, 0.1, 1.0, None, None, False)
