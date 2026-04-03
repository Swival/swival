"""Tests for context window management: estimate_tokens, group_into_turns,
compact_messages, drop_middle_turns, clamp_output_tokens, and ContextOverflowError."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swival.agent import (
    estimate_tokens,
    group_into_turns,
    compact_messages,
    compact_tool_result,
    is_pinned,
    score_turn,
    drop_middle_turns,
    aggressive_drop_turns,
    summarize_turns,
    _RECAP_PREFIX,
    CompactionState,
    MAX_CHECKPOINTS,
    MAX_CHECKPOINT_TOKENS,
    clamp_output_tokens,
    ContextOverflowError,
    call_llm,
    _fix_orphaned_tool_calls,
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
        assert tool_msgs[0]["content"].startswith("[read_file:")
        assert "compacted" in tool_msgs[0]["content"]

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
# compact_tool_result
# ---------------------------------------------------------------------------


class TestCompactToolResult:
    def test_short_content_unchanged(self):
        content = "short result"
        assert (
            compact_tool_result("read_file", {"file_path": "f.py"}, content) == content
        )

    def test_read_file(self):
        content = "line\n" * 500  # >1000 chars, 500 newlines
        result = compact_tool_result("read_file", {"file_path": "src/app.py"}, content)
        assert result.startswith("[read_file: src/app.py,")
        assert "500 lines" in result
        assert "compacted" in result

    def test_grep(self):
        content = "match\n" * 300
        result = compact_tool_result(
            "grep", {"pattern": "TODO", "path": "src/"}, content
        )
        assert result.startswith("[grep: 'TODO' in src/,")
        assert "~300 matches" in result
        assert "compacted" in result

    def test_list_files(self):
        content = "file.py\n" * 200
        result = compact_tool_result(
            "list_files", {"pattern": "*.py", "path": "/project"}, content
        )
        assert result.startswith("[list_files: '*.py' in /project,")
        assert "~200 entries" in result

    def test_run_command(self):
        content = "output " * 200  # >1000 chars
        result = compact_tool_result(
            "run_command", {"command": ["pytest", "-v"]}, content
        )
        assert "[run_command: `pytest -v`" in result
        assert "first 200 chars" in result
        assert "last 200 chars" in result

    def test_run_command_string_cmd(self):
        content = "x" * 2000
        result = compact_tool_result("run_command", {"command": "ls -la"}, content)
        assert "`ls -la`" in result

    def test_run_shell_command(self):
        content = "output " * 200  # >1000 chars
        result = compact_tool_result(
            "run_shell_command", {"command": "ls -la | head"}, content
        )
        assert "[run_shell_command: `ls -la | head`" in result
        assert "first 200 chars" in result
        assert "last 200 chars" in result

    def test_run_shell_command_array_fallback(self):
        content = "x" * 2000
        result = compact_tool_result(
            "run_shell_command", {"command": ["echo", "hi"]}, content
        )
        assert "[run_shell_command: `echo hi`" in result

    def test_fetch_url(self):
        content = "page content " * 200
        result = compact_tool_result(
            "fetch_url", {"url": "https://example.com"}, content
        )
        assert "[fetch_url: https://example.com," in result
        assert "chars" in result
        assert "compacted" in result

    def test_unknown_tool(self):
        content = "x" * 2000
        result = compact_tool_result("some_new_tool", {}, content)
        assert "[some_new_tool:" in result
        assert "2000" in result

    def test_missing_args_keys(self):
        content = "x" * 2000
        result = compact_tool_result("read_file", {}, content)
        assert "[read_file: ?," in result

    def test_none_args(self):
        content = "x" * 2000
        result = compact_tool_result("grep", None, content)
        assert "[grep:" in result
        assert "compacted" in result

    def test_exactly_1000_chars_unchanged(self):
        content = "x" * 1000
        assert (
            compact_tool_result("read_file", {"file_path": "f.py"}, content) == content
        )

    def test_1001_chars_compacted(self):
        content = "x" * 1001
        result = compact_tool_result("read_file", {"file_path": "f.py"}, content)
        assert result != content
        assert "compacted" in result

    def test_read_multiple_files_batch_list(self):
        content = "x" * 2000
        result = compact_tool_result(
            "read_multiple_files",
            {"files": [{"file_path": "a.py"}, {"file_path": "b.py"}]},
            content,
        )
        assert result.startswith("[read_multiple_files: a.py, b.py,")
        assert "compacted" in result

    def test_read_multiple_files_bare_string(self):
        content = "x" * 2000
        result = compact_tool_result(
            "read_multiple_files",
            {"files": "x.py"},
            content,
        )
        assert result.startswith("[read_multiple_files: x.py,")
        assert "compacted" in result

    def test_read_multiple_files_mixed_entries(self):
        content = "x" * 2000
        result = compact_tool_result(
            "read_multiple_files",
            {"files": [{"file_path": "a.py"}, "b.py", 42]},
            content,
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "?" in result
        assert "compacted" in result

    def test_outline_single_file(self):
        content = "x" * 2000
        result = compact_tool_result("outline", {"file_path": "src/app.py"}, content)
        assert result == "[outline: src/app.py — compacted]"

    def test_outline_batch_list(self):
        content = "x" * 2000
        result = compact_tool_result(
            "outline",
            {"files": [{"file_path": "a.py"}, {"file_path": "b.py"}]},
            content,
        )
        assert result.startswith("[outline: a.py, b.py,")
        assert "compacted" in result

    def test_outline_batch_bare_string(self):
        content = "x" * 2000
        result = compact_tool_result("outline", {"files": "x.py"}, content)
        assert result.startswith("[outline: x.py,")
        assert "compacted" in result

    def test_outline_batch_mixed_entries(self):
        content = "x" * 2000
        result = compact_tool_result(
            "outline",
            {"files": [{"file_path": "a.py"}, "b.py", 42]},
            content,
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "?" in result
        assert "compacted" in result

    def test_mcp_tool_compacted_with_head(self):
        content = "abcdefgh" * 250  # 2000 chars
        result = compact_tool_result("mcp__server__tool", {}, content)
        assert result.startswith("[mcp__server__tool: 2000 chars")
        assert "compacted" in result
        assert "First 300 chars" in result
        assert content[:300] in result

    def test_mcp_tool_exactly_1000_unchanged(self):
        content = "x" * 1000
        assert compact_tool_result("mcp__server__tool", {}, content) == content

    def test_mcp_tool_1001_compacted(self):
        content = "y" * 1001
        result = compact_tool_result("mcp__server__tool", {}, content)
        assert result != content
        assert "mcp__server__tool" in result
        assert "compacted" in result

    def test_compact_messages_uses_structured_summaries(self):
        """compact_messages should produce per-tool summaries, not generic ones."""
        tc = _assistant_tc([("tc1", "grep", '{"pattern": "error", "path": "logs/"}')])
        big_content = "match: error found\n" * 100  # >1000 chars
        tr = _tool("tc1", big_content)
        msgs = [_sys("sys"), _user("q"), tc, tr, _assistant("mid"), _assistant("done")]
        result = compact_messages(msgs)
        tool_msgs = [
            m
            for m in result
            if (m.get("role") if isinstance(m, dict) else None) == "tool"
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"].startswith("[grep:")
        assert "'error'" in tool_msgs[0]["content"]
        assert "logs/" in tool_msgs[0]["content"]

    def test_grep_compaction_with_context(self):
        """Compaction uses the header count, not newline count, with context."""
        # Simulate grep output with context_lines — many newlines, but only 2 matches
        # Pad with enough context lines to exceed the 1000-char compaction threshold
        ctx_lines = "\n".join(f"  Line {i}: {'x' * 80}" for i in range(50))
        content = (
            f"Found 2 matches\n\nfile.py:\n{ctx_lines}\n"
            "  Line 100: match1  <<<\n"
            "  --\n"
            "  Line 200: match2  <<<\n"
        )
        assert len(content) > 1000  # ensure compaction triggers
        result = compact_tool_result("grep", {"pattern": "match", "path": "."}, content)
        assert "~2 matches" in result


# ---------------------------------------------------------------------------
# is_pinned / score_turn
# ---------------------------------------------------------------------------


class TestIsPinned:
    def test_user_turn_is_pinned(self):
        assert is_pinned([_user("hello")]) is True

    def test_assistant_turn_not_pinned(self):
        assert is_pinned([_assistant("response")]) is False

    def test_tool_turn_not_pinned(self):
        tc = _assistant_tc([("tc1", "read_file", "{}")])
        tr = _tool("tc1", "content")
        assert is_pinned([tc, tr]) is False

    def test_system_turn_not_pinned(self):
        assert is_pinned([_sys("system prompt")]) is False


class TestScoreTurn:
    def test_error_content_scores_high(self):
        tc = _assistant_tc([("tc1", "run_command", "{}")])
        tr = _tool("tc1", "error: command failed")
        score = score_turn([tc, tr])
        assert score >= 3

    def test_file_edit_scores_high(self):
        tc = _assistant_tc([("tc1", "edit_file", "{}")])
        tr = _tool("tc1", "ok")
        score = score_turn([tc, tr])
        assert score >= 5

    def test_write_file_scores_high(self):
        tc = _assistant_tc([("tc1", "write_file", "{}")])
        tr = _tool("tc1", "ok")
        score = score_turn([tc, tr])
        assert score >= 5

    def test_read_file_scores_low(self):
        tc = _assistant_tc([("tc1", "read_file", "{}")])
        tr = _tool("tc1", "file contents here")
        score = score_turn([tc, tr])
        assert score == 0

    def test_plain_assistant_scores_zero(self):
        assert score_turn([_assistant("just thinking")]) == 0

    def test_error_in_assistant_content(self):
        score = score_turn([_assistant("I encountered an error in the code")])
        assert score >= 3

    def test_combined_scores_accumulate(self):
        # A turn with both an error and a file edit should score higher
        tc = _assistant_tc([("tc1", "edit_file", "{}")])
        tr = _tool("tc1", "error: partial write failed")
        score = score_turn([tc, tr])
        assert score >= 8  # 5 (edit) + 3 (error)


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

    def test_user_turns_never_dropped(self):
        """User turns in the middle must be preserved (pinned)."""
        tc1 = _assistant_tc([("tc1", "read_file", "{}")])
        tr1 = _tool("tc1", "r1")
        user_mid = _user("can you also check bar.py?")
        tc2 = _assistant_tc([("tc2", "read_file", "{}")])
        tr2 = _tool("tc2", "r2")
        tc3 = _assistant_tc([("tc3", "read_file", "{}")])
        tr3 = _tool("tc3", "r3")
        tc4 = _assistant_tc([("tc4", "read_file", "{}")])
        tr4 = _tool("tc4", "r4")
        tc5 = _assistant_tc([("tc5", "read_file", "{}")])
        tr5 = _tool("tc5", "r5")
        msgs = [
            _sys("s"),
            _user("q"),
            tc1,
            tr1,
            user_mid,
            tc2,
            tr2,
            tc3,
            tr3,
            tc4,
            tr4,
            tc5,
            tr5,
        ]
        result = drop_middle_turns(msgs)
        # The mid-conversation user message must survive
        user_contents = [
            m["content"]
            for m in result
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert "can you also check bar.py?" in user_contents

    def test_high_scoring_turns_kept(self):
        """Turns with file edits should be kept over plain reads."""
        # Create a history with many middle turns
        tc_read1 = _assistant_tc([("r1", "read_file", "{}")])
        tr_read1 = _tool("r1", "file contents")
        tc_read2 = _assistant_tc([("r2", "read_file", "{}")])
        tr_read2 = _tool("r2", "more contents")
        tc_edit = _assistant_tc([("e1", "edit_file", "{}")])
        tr_edit = _tool("e1", "ok")
        tc_read3 = _assistant_tc([("r3", "read_file", "{}")])
        tr_read3 = _tool("r3", "contents")
        # Tail turns
        tc_tail1 = _assistant_tc([("t1", "read_file", "{}")])
        tr_tail1 = _tool("t1", "t")
        tc_tail2 = _assistant_tc([("t2", "read_file", "{}")])
        tr_tail2 = _tool("t2", "t")
        tc_tail3 = _assistant_tc([("t3", "read_file", "{}")])
        tr_tail3 = _tool("t3", "t")
        msgs = [
            _sys("s"),
            _user("q"),
            tc_read1,
            tr_read1,
            tc_read2,
            tr_read2,
            tc_edit,
            tr_edit,
            tc_read3,
            tr_read3,
            tc_tail1,
            tr_tail1,
            tc_tail2,
            tr_tail2,
            tc_tail3,
            tr_tail3,
        ]
        result = drop_middle_turns(msgs)
        # The edit turn (high score) should still be in the result
        tc_ids_in_result = set()
        for m in result:
            tcs = (
                m.get("tool_calls", None)
                if isinstance(m, dict)
                else getattr(m, "tool_calls", None)
            )
            if tcs:
                for tc in tcs:
                    fn = (
                        tc.function
                        if hasattr(tc, "function")
                        else tc.get("function", {})
                    )
                    fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    tc_ids_in_result.add(fn_name)
        assert "edit_file" in tc_ids_in_result


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

    def test_available_below_minimum_raises(self):
        # When prompt leaves fewer than MIN_OUTPUT_TOKENS, raise ContextOverflowError
        msgs = [_user("x " * 10000)]
        with pytest.raises(ContextOverflowError):
            clamp_output_tokens(msgs, None, 10, 16384)

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
                call_llm(
                    "http://localhost", "model", [], 100, 0.1, 1.0, None, None, False
                )

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
                call_llm(
                    "http://localhost", "model", [], 100, 0.1, 1.0, None, None, False
                )

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
                call_llm(
                    "http://localhost", "model", [], 100, 0.1, 1.0, None, None, False
                )

    def test_api_error_with_context_keywords(self):
        """call_llm raises ContextOverflowError for APIError with context keywords."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.APIError(
                message="ChatgptException - Your input exceeds the context window of this model.",
                status_code=500,
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(ContextOverflowError):
                call_llm(
                    "http://localhost", "model", [], 100, 0.1, 1.0, None, None, False
                )

    def test_api_error_without_context_keywords(self):
        """call_llm raises AgentError for APIError without context keywords."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.APIError(
                message="internal server error",
                status_code=500,
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(AgentError):
                call_llm(
                    "http://localhost",
                    "model",
                    [],
                    100,
                    0.1,
                    1.0,
                    None,
                    None,
                    False,
                    max_retries=1,
                )

    def test_call_llm_omits_tool_choice_when_tools_none(self):
        """When tools=None, call_llm should not include tool_choice in kwargs."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="summary", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )
            call_llm(
                "http://localhost",
                "model",
                [{"role": "user", "content": "hi"}],
                100,
                0.1,
                1.0,
                None,
                None,
                False,
            )
            # litellm.completion was called with keyword args
            call_kw = mock_comp.call_args.kwargs
            assert "tool_choice" not in call_kw
            assert "tools" not in call_kw


# ---------------------------------------------------------------------------
# ToolsNotSupportedError
# ---------------------------------------------------------------------------

_DUMMY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dummy",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class TestToolsNotSupported:
    """Tests for ToolsNotSupportedError detection in call_llm."""

    def test_hf_function_calling_not_supported_raises(self):
        """call_llm raises ToolsNotSupportedError for HuggingFace models that
        reject function calling."""
        import litellm
        from swival.report import ToolsNotSupportedError

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.BadRequestError(
                message=(
                    'HuggingfaceException - {"code":400,'
                    '"reason":"INVALID_REQUEST_BODY",'
                    '"message":"model features function calling not support",'
                    '"metadata":{}}'
                ),
                model="huggingface/google/gemma-4-31B-it",
                llm_provider="huggingface",
            )
            with pytest.raises(ToolsNotSupportedError):
                call_llm(
                    "http://localhost",
                    "model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.1,
                    1.0,
                    None,
                    _DUMMY_TOOLS,
                    False,
                    max_retries=1,
                )

    def test_tools_not_supported_not_raised_when_tools_none(self):
        """When tools=None, a matching BadRequestError should NOT raise
        ToolsNotSupportedError (it's a different problem)."""
        import litellm

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.BadRequestError(
                message="model features function calling not support",
                model="test",
                llm_provider="huggingface",
            )
            with pytest.raises(AgentError) as exc_info:
                call_llm(
                    "http://localhost",
                    "model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.1,
                    1.0,
                    None,
                    None,
                    False,
                    max_retries=1,
                )
            from swival.report import ToolsNotSupportedError

            assert not isinstance(exc_info.value, ToolsNotSupportedError)

    def test_generic_does_not_support_tools(self):
        """call_llm raises ToolsNotSupportedError for 'does not support tools'."""
        import litellm
        from swival.report import ToolsNotSupportedError

        with patch("litellm.completion") as mock_comp:
            mock_comp.side_effect = litellm.BadRequestError(
                message="This model does not support tools",
                model="test",
                llm_provider="openai",
            )
            with pytest.raises(ToolsNotSupportedError):
                call_llm(
                    "http://localhost",
                    "model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.1,
                    1.0,
                    None,
                    _DUMMY_TOOLS,
                    False,
                    max_retries=1,
                )


# ---------------------------------------------------------------------------
# ToolsNotSupportedError — run_agent_loop integration
# ---------------------------------------------------------------------------


class TestToolsNotSupportedLoop:
    """Integration tests: ToolsNotSupportedError fallback in run_agent_loop."""

    @staticmethod
    def _loop_kwargs(tmp_path, **overrides):
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        defaults = dict(
            api_base="http://127.0.0.1:1234",
            model_id="test-model",
            max_turns=1,
            max_output_tokens=1024,
            temperature=0.5,
            top_p=1.0,
            seed=None,
            context_length=None,
            base_dir=str(tmp_path),
            thinking_state=ThinkingState(verbose=False),
            resolved_commands={},
            skills_catalog={},
            skill_read_roots=[],
            extra_write_roots=[],
            files_mode="some",
            verbose=False,
            llm_kwargs={"provider": "lmstudio", "api_key": None},
            file_tracker=None,
            todo_state=TodoState(verbose=False),
        )
        defaults.update(overrides)
        return defaults

    def test_max_turns_1_retries_without_tools(self, tmp_path):
        """With max_turns=1 the plain-chat retry must still fire after
        ToolsNotSupportedError — the failed tool-enabled call should NOT
        consume the only turn."""
        from swival.agent import run_agent_loop
        from swival.report import ToolsNotSupportedError as _TNS

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call: has tools → raise
            tools_arg = args[7] if len(args) > 7 else kwargs.get("tools")
            if tools_arg is not None:
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            # Second call: no tools → succeed
            return (
                SimpleNamespace(
                    content="plain answer", tool_calls=None, role="assistant"
                ),
                "stop",
                [],
                0,
                (0, 0),
            )

        messages = [_sys("system"), _user("hello")]
        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            answer, exhausted = run_agent_loop(
                messages,
                _DUMMY_TOOLS,
                **self._loop_kwargs(tmp_path, max_turns=1),
            )

        assert answer == "plain answer"
        assert exhausted is False
        assert call_count == 2

    def test_report_records_both_calls(self, tmp_path):
        """The failed tool-enabled call and the successful plain-chat retry
        must both appear in the report."""
        from swival.agent import run_agent_loop
        from swival.report import ReportCollector, ToolsNotSupportedError as _TNS

        def fake_call_llm(*args, **kwargs):
            tools_arg = args[7] if len(args) > 7 else kwargs.get("tools")
            if tools_arg is not None:
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            return (
                SimpleNamespace(content="ok", tool_calls=None, role="assistant"),
                "stop",
                [],
                0,
                (0, 0),
            )

        report = ReportCollector()
        messages = [_sys("system"), _user("hello")]
        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            run_agent_loop(
                messages,
                _DUMMY_TOOLS,
                **self._loop_kwargs(tmp_path, max_turns=3, report=report),
            )

        llm_events = [e for e in report.events if e["type"] == "llm_call"]
        assert len(llm_events) == 2
        assert llm_events[0]["finish_reason"] == "tools_not_supported"
        assert llm_events[1]["finish_reason"] == "stop"
        # The successful retry must be tagged as a retry
        assert llm_events[1].get("is_retry") is True
        assert llm_events[1].get("retry_reason") == "drop_tools_unsupported"

    def test_overflow_then_tools_not_supported_recovers(self, tmp_path):
        """ToolsNotSupportedError discovered during a compaction retry must
        still trigger the tools-drop fallback and eventually succeed."""
        from swival.agent import run_agent_loop, ContextOverflowError
        from swival.report import (
            ReportCollector,
            ToolsNotSupportedError as _TNS,
        )

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools_arg = args[7] if len(args) > 7 else kwargs.get("tools")
            if call_count == 1:
                # First call: context overflow (triggers compaction)
                raise ContextOverflowError("context length exceeded")
            if call_count == 2 and tools_arg is not None:
                # Second call (compaction retry, still has tools): tools
                # not supported
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            # Third call: no tools → succeed
            return (
                SimpleNamespace(content="recovered", tool_calls=None, role="assistant"),
                "stop",
                [],
                0,
                (0, 0),
            )

        report = ReportCollector()
        messages = [_sys("system"), _user("hello")]
        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            answer, exhausted = run_agent_loop(
                messages,
                _DUMMY_TOOLS,
                **self._loop_kwargs(tmp_path, max_turns=2, report=report),
            )

        assert answer == "recovered"
        assert exhausted is False
        assert call_count == 3

        llm_events = [e for e in report.events if e["type"] == "llm_call"]
        # Three events: overflow, tools_not_supported, success
        assert len(llm_events) == 3
        assert llm_events[0]["finish_reason"] == "context_overflow"
        assert llm_events[1]["finish_reason"] == "tools_not_supported"
        assert llm_events[2]["finish_reason"] == "stop"
        assert llm_events[2].get("is_retry") is True

    def test_tools_not_supported_then_overflow_uses_compaction_result(self, tmp_path):
        """When tools_not_supported fires first and the no-tools retry
        overflows, the compaction-retry result must be used — the
        _is_tools_retry flag must NOT discard a successful compaction."""
        from swival.agent import run_agent_loop, ContextOverflowError
        from swival.report import (
            ReportCollector,
            ToolsNotSupportedError as _TNS,
        )

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools_arg = args[7] if len(args) > 7 else kwargs.get("tools")
            if call_count == 1:
                # First call: tools not supported
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            if call_count == 2:
                # Second call (no-tools retry): context overflow
                assert tools_arg is None
                raise ContextOverflowError("context length exceeded")
            if call_count == 3:
                # Third call (compaction retry, no tools): succeed
                assert tools_arg is None
                return (
                    SimpleNamespace(
                        content="compaction result",
                        tool_calls=None,
                        role="assistant",
                    ),
                    "stop",
                    [],
                    0,
                    (0, 0),
                )
            # Should not reach here
            return (
                SimpleNamespace(
                    content=f"extra call {call_count}",
                    tool_calls=None,
                    role="assistant",
                ),
                "stop",
                [],
                0,
                (0, 0),
            )

        report = ReportCollector()
        messages = [_sys("system"), _user("hello")]
        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            answer, exhausted = run_agent_loop(
                messages,
                _DUMMY_TOOLS,
                **self._loop_kwargs(tmp_path, max_turns=2, report=report),
            )

        assert answer == "compaction result"
        assert exhausted is False
        assert call_count == 3  # no spurious fourth call

        llm_events = [e for e in report.events if e["type"] == "llm_call"]
        assert len(llm_events) == 3
        assert llm_events[0]["finish_reason"] == "tools_not_supported"
        # The overflow event should be tagged as a retry (it was the
        # no-tools fallback attempt)
        assert llm_events[1]["finish_reason"] == "context_overflow"
        assert llm_events[1].get("is_retry") is True
        assert llm_events[2]["finish_reason"] == "stop"

    def test_tools_not_supported_then_agent_error_tagged_as_retry(self, tmp_path):
        """When the no-tools retry raises a generic AgentError, the error
        event must be tagged is_retry=True."""
        from swival.agent import run_agent_loop
        from swival.report import (
            AgentError as _AE,
            ReportCollector,
            ToolsNotSupportedError as _TNS,
        )

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools_arg = args[7] if len(args) > 7 else kwargs.get("tools")
            if call_count == 1:
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            # Second call (no-tools retry): fatal error
            assert tools_arg is None
            raise _AE("something else broke")

        report = ReportCollector()
        messages = [_sys("system"), _user("hello")]
        with pytest.raises(_AE, match="something else broke"):
            with patch("swival.agent.call_llm", side_effect=fake_call_llm):
                run_agent_loop(
                    messages,
                    _DUMMY_TOOLS,
                    **self._loop_kwargs(tmp_path, max_turns=2, report=report),
                )

        llm_events = [e for e in report.events if e["type"] == "llm_call"]
        assert len(llm_events) == 2
        assert llm_events[0]["finish_reason"] == "tools_not_supported"
        assert llm_events[1]["finish_reason"] == "error"
        assert llm_events[1].get("is_retry") is True
        assert llm_events[1].get("retry_reason") == "drop_tools_unsupported"

    def test_empty_response_message_no_tools(self, tmp_path):
        """After tools fallback, the empty-response continuation must not
        say 'available tools'."""
        from swival.agent import run_agent_loop
        from swival.report import ToolsNotSupportedError as _TNS

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tne = _TNS("function calling not support")
                tne._provider_retries = 0
                raise tne
            if call_count == 2:
                # Return empty response to trigger continuation
                return (
                    SimpleNamespace(content="", tool_calls=None, role="assistant"),
                    "stop",
                    [],
                    0,
                    (0, 0),
                )
            return (
                SimpleNamespace(
                    content="final answer", tool_calls=None, role="assistant"
                ),
                "stop",
                [],
                0,
                (0, 0),
            )

        messages = [_sys("system"), _user("hello")]
        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            answer, _ = run_agent_loop(
                messages,
                _DUMMY_TOOLS,
                **self._loop_kwargs(tmp_path, max_turns=5),
            )

        assert answer == "final answer"
        # The empty-response continuation should not reference tools
        user_msgs = [m for m in messages if m.get("role") == "user"]
        continuation = user_msgs[-1]["content"]
        assert "available tools" not in continuation
        assert "answer the question directly" in continuation


# ---------------------------------------------------------------------------
# summarize_turns
# ---------------------------------------------------------------------------


class TestSummarizeTurns:
    def _make_mock_call_llm(self, content="Summary of dropped turns."):
        """Return a mock call_llm that returns a successful response."""

        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            return (
                SimpleNamespace(content=content),
                "stop",
            )

        return mock_fn

    def _make_failing_call_llm(self, exc):
        """Return a mock call_llm that raises the given exception."""

        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            raise exc

        return mock_fn

    def _sample_turns(self):
        """Build a list of turns to summarize."""
        tc1 = _assistant_tc([("tc1", "read_file", '{"file_path": "foo.py"}')])
        tr1 = _tool("tc1", "def foo(): pass")
        tc2 = _assistant_tc([("tc2", "grep", '{"pattern": "TODO"}')])
        tr2 = _tool("tc2", "line 1: TODO fix\nline 2: TODO refactor")
        return [[tc1, tr1], [tc2, tr2]]

    def test_successful_summary(self):
        turns = self._sample_turns()
        result = summarize_turns(
            turns,
            self._make_mock_call_llm("The agent read foo.py and searched for TODOs."),
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert result == "The agent read foo.py and searched for TODOs."

    def test_returns_none_on_context_overflow(self):
        turns = self._sample_turns()
        result = summarize_turns(
            turns,
            self._make_failing_call_llm(ContextOverflowError("overflow")),
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert result is None

    def test_returns_none_on_timeout(self):
        turns = self._sample_turns()
        result = summarize_turns(
            turns,
            self._make_failing_call_llm(TimeoutError("timed out")),
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert result is None

    def test_returns_none_on_connection_error(self):
        turns = self._sample_turns()
        result = summarize_turns(
            turns,
            self._make_failing_call_llm(ConnectionError("refused")),
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert result is None

    def test_returns_none_on_empty_content(self):
        turns = self._sample_turns()
        result = summarize_turns(
            turns,
            self._make_mock_call_llm(""),
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert result is None

    def test_input_capped_at_8000_chars(self):
        """Large inputs should be truncated before being sent to the model."""
        big_turn = [_assistant("x" * 10000)]
        captured = {}

        def capturing_call_llm(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            captured["messages"] = messages
            return (SimpleNamespace(content="ok"), "stop")

        summarize_turns(
            [big_turn],
            capturing_call_llm,
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        user_content = captured["messages"][1]["content"]
        assert len(user_content) <= 8100  # 8000 + truncation marker

    def test_passes_tools_none(self):
        """summarize_turns should call LLM with tools=None."""
        captured = {}

        def capturing_call_llm(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            captured["tools"] = tools
            return (SimpleNamespace(content="ok"), "stop")

        summarize_turns(
            self._sample_turns(),
            capturing_call_llm,
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        assert captured["tools"] is None


class TestDropMiddleTurnsWithSummary:
    """Tests for AI-powered summarization in drop_middle_turns."""

    def _make_mock_call_llm(self, content="Summary recap."):
        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            return (SimpleNamespace(content=content), "stop")

        return mock_fn

    def _make_failing_call_llm(self):
        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            raise RuntimeError("LLM unavailable")

        return mock_fn

    def _build_msgs(self):
        """Build a message list with enough turns for dropping."""
        tc1 = _assistant_tc([("tc1", "read_file", "{}")])
        tr1 = _tool("tc1", "r1")
        tc2 = _assistant_tc([("tc2", "read_file", "{}")])
        tr2 = _tool("tc2", "r2")
        tc3 = _assistant_tc([("tc3", "read_file", "{}")])
        tr3 = _tool("tc3", "r3")
        tc4 = _assistant_tc([("tc4", "read_file", "{}")])
        tr4 = _tool("tc4", "r4")
        return [_sys("sys"), _user("q"), tc1, tr1, tc2, tr2, tc3, tr3, tc4, tr4]

    def test_summary_injected_as_assistant_role(self):
        msgs = self._build_msgs()
        result = drop_middle_turns(
            msgs,
            call_llm_fn=self._make_mock_call_llm("Recap of work done."),
            model_id="m",
            base_url="http://x",
            api_key="k",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        # Find the recap message
        recaps = [
            m
            for m in result
            if isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and _RECAP_PREFIX in m["content"]
        ]
        assert len(recaps) == 1
        assert recaps[0]["role"] == "assistant"
        assert recaps[0]["content"].startswith(_RECAP_PREFIX)
        assert "Recap of work done." in recaps[0]["content"]

    def test_fallback_to_static_marker_on_failure(self):
        msgs = self._build_msgs()
        result = drop_middle_turns(
            msgs,
            call_llm_fn=self._make_failing_call_llm(),
            model_id="m",
            base_url="http://x",
            api_key="k",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        # Should have the static splice marker (role=user)
        markers = [
            m
            for m in result
            if isinstance(m, dict) and "[context compacted" in m.get("content", "")
        ]
        assert len(markers) == 1
        assert markers[0]["role"] == "user"

    def test_fallback_without_llm_params(self):
        """When no call_llm_fn is provided, uses static marker."""
        msgs = self._build_msgs()
        result = drop_middle_turns(msgs)
        markers = [
            m
            for m in result
            if isinstance(m, dict) and "[context compacted" in m.get("content", "")
        ]
        assert len(markers) == 1
        assert markers[0]["role"] == "user"

    def test_no_user_or_system_role_from_summarization(self):
        """The recap must never use role=user or role=system."""
        msgs = self._build_msgs()
        result = drop_middle_turns(
            msgs,
            call_llm_fn=self._make_mock_call_llm("Recap text."),
            model_id="m",
            base_url="http://x",
            api_key="k",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        for m in result:
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if isinstance(content, str) and _RECAP_PREFIX in content:
                assert m["role"] == "assistant"


# ---------------------------------------------------------------------------
# CompactionState
# ---------------------------------------------------------------------------


class TestCompactionState:
    def _make_mock_call_llm(self, content="Checkpoint summary."):
        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            return (SimpleNamespace(content=content), "stop")

        return mock_fn

    def _make_failing_call_llm(self):
        def mock_fn(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            raise RuntimeError("LLM unavailable")

        return mock_fn

    def _llm_kwargs(self):
        return dict(
            model_id="test",
            base_url="http://localhost",
            api_key="key",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )

    def _build_messages(self, n_turns):
        """Build a message list with n tool-call turns."""
        msgs = [_sys("system prompt"), _user("initial question")]
        for i in range(n_turns):
            tc = _assistant_tc([(f"tc{i}", "read_file", "{}")])
            tr = _tool(f"tc{i}", f"result {i}")
            msgs.extend([tc, tr])
        return msgs

    def test_checkpoint_fires_at_interval(self):
        state = CompactionState(checkpoint_interval=3)
        msgs = self._build_messages(5)
        mock_llm = self._make_mock_call_llm("checkpoint 1")
        for _ in range(3):
            state.maybe_checkpoint(msgs, mock_llm, **self._llm_kwargs())
        assert len(state.summaries) == 1
        assert state.summaries[0] == "checkpoint 1"

    def test_checkpoint_does_not_fire_early(self):
        state = CompactionState(checkpoint_interval=5)
        msgs = self._build_messages(3)
        mock_llm = self._make_mock_call_llm()
        for _ in range(4):
            state.maybe_checkpoint(msgs, mock_llm, **self._llm_kwargs())
        assert len(state.summaries) == 0

    def test_counter_resets_on_failure(self):
        """After a failed checkpoint, the counter resets and doesn't retry every turn."""
        state = CompactionState(checkpoint_interval=3)
        msgs = self._build_messages(5)
        call_count = 0

        def counting_fail(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        # Trigger the checkpoint (3 turns)
        for _ in range(3):
            state.maybe_checkpoint(msgs, counting_fail, **self._llm_kwargs())
        assert call_count == 1  # called once at turn 3

        # Next turn should NOT retry
        state.maybe_checkpoint(msgs, counting_fail, **self._llm_kwargs())
        assert call_count == 1  # still 1, not retried

        # Need another full interval before next attempt
        for _ in range(2):
            state.maybe_checkpoint(msgs, counting_fail, **self._llm_kwargs())
        assert call_count == 2  # now attempted again at turn 6

    def test_summaries_never_exceed_max_checkpoints(self):
        """Even with many checkpoints, the list stays bounded."""
        state = CompactionState(checkpoint_interval=1)
        msgs = self._build_messages(5)
        mock_llm = self._make_mock_call_llm("summary")

        # Generate more checkpoints than MAX_CHECKPOINTS
        for i in range(MAX_CHECKPOINTS * 4):
            state.maybe_checkpoint(msgs, mock_llm, **self._llm_kwargs())

        assert len(state.summaries) <= MAX_CHECKPOINTS

    def test_get_full_summary_capped(self):
        state = CompactionState()
        # Manually stuff in large summaries
        state.summaries = ["x" * 5000 for _ in range(10)]
        full = state.get_full_summary()
        cap = MAX_CHECKPOINT_TOKENS * 4
        assert len(full) <= cap + 100  # small margin for truncation marker

    def test_consolidation_failure_drops_oldest(self):
        """When merge fails, oldest summaries are dropped."""
        state = CompactionState(checkpoint_interval=1)
        msgs = self._build_messages(5)
        call_counter = {"n": 0}

        def sometimes_fail(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            call_counter["n"] += 1
            # First MAX_CHECKPOINTS+1 calls succeed (building up summaries)
            # Then consolidation call fails
            if call_counter["n"] <= MAX_CHECKPOINTS + 1:
                return (SimpleNamespace(content=f"summary {call_counter['n']}"), "stop")
            raise RuntimeError("consolidation failed")

        for _ in range(MAX_CHECKPOINTS + 2):
            state.maybe_checkpoint(msgs, sometimes_fail, **self._llm_kwargs())

        # After consolidation failure, oldest are dropped
        assert len(state.summaries) <= MAX_CHECKPOINTS

    def test_checkpoint_fallback_in_drop_middle_turns(self):
        """When LLM summary fails but checkpoints exist, use checkpoint summary."""
        state = CompactionState()
        state.summaries = ["Earlier: agent read foo.py and found 3 bugs."]

        tc1 = _assistant_tc([("tc1", "read_file", "{}")])
        tr1 = _tool("tc1", "r1")
        tc2 = _assistant_tc([("tc2", "read_file", "{}")])
        tr2 = _tool("tc2", "r2")
        tc3 = _assistant_tc([("tc3", "read_file", "{}")])
        tr3 = _tool("tc3", "r3")
        tc4 = _assistant_tc([("tc4", "read_file", "{}")])
        tr4 = _tool("tc4", "r4")
        msgs = [_sys("sys"), _user("q"), tc1, tr1, tc2, tr2, tc3, tr3, tc4, tr4]

        def failing_llm(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            raise RuntimeError("LLM down")

        result = drop_middle_turns(
            msgs,
            call_llm_fn=failing_llm,
            model_id="m",
            base_url="http://x",
            api_key="k",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
            compaction_state=state,
        )
        # Should have used checkpoint summary, not static marker
        recaps = [
            m
            for m in result
            if isinstance(m, dict)
            and m.get("role") == "assistant"
            and "checkpoints" in m.get("content", "")
        ]
        assert len(recaps) == 1
        assert "3 bugs" in recaps[0]["content"]


# ---------------------------------------------------------------------------
# aggressive_drop_turns
# ---------------------------------------------------------------------------


class TestAggressiveDropTurns:
    def _build_msgs(self, n_middle_turns=5):
        """Build messages with system + user + N middle turns + nothing."""
        msgs = [_sys("system prompt"), _user("initial question")]
        for i in range(n_middle_turns):
            tc = _assistant_tc([(f"tc{i}", "read_file", "{}")])
            tr = _tool(f"tc{i}", f"result {i}")
            msgs.extend([tc, tr])
        return msgs

    def test_keeps_system_and_last_2_turns(self):
        msgs = self._build_msgs(6)
        result = aggressive_drop_turns(msgs)
        # Should have: system, recap/marker, last 2 turns (4 messages)
        roles = [
            m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            for m in result
        ]
        assert roles[0] == "system"
        # Second message should be the recap or splice marker
        assert result[1].get("role") in ("assistant", "user")

    def test_fewer_turns_than_tail_unchanged(self):
        msgs = [_sys("s"), _assistant("done")]
        result = aggressive_drop_turns(msgs)
        assert len(result) == len(msgs)

    def test_with_llm_summary(self):
        msgs = self._build_msgs(5)

        def mock_llm(
            *,
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            api_key,
            provider,
        ):
            return (SimpleNamespace(content="Aggressive recap."), "stop")

        result = aggressive_drop_turns(
            msgs,
            call_llm_fn=mock_llm,
            model_id="m",
            base_url="http://x",
            api_key="k",
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        recaps = [
            m
            for m in result
            if isinstance(m, dict)
            and m.get("role") == "assistant"
            and _RECAP_PREFIX in m.get("content", "")
        ]
        assert len(recaps) == 1
        assert "Aggressive recap." in recaps[0]["content"]

    def test_fallback_to_static_marker(self):
        msgs = self._build_msgs(5)
        result = aggressive_drop_turns(msgs)
        markers = [
            m
            for m in result
            if isinstance(m, dict) and "[context compacted" in m.get("content", "")
        ]
        assert len(markers) == 1

    def test_more_aggressive_than_drop_middle(self):
        """aggressive_drop_turns should produce fewer messages than drop_middle_turns."""
        msgs = self._build_msgs(8)
        drop_result = drop_middle_turns(list(msgs))
        aggressive_result = aggressive_drop_turns(list(msgs))
        assert len(aggressive_result) <= len(drop_result)

    def test_valid_tool_pairing_after_aggressive(self):
        msgs = self._build_msgs(6)
        result = aggressive_drop_turns(msgs)
        _validate_tool_pairing(result)


# ---------------------------------------------------------------------------
# Graduated compaction (integration)
# ---------------------------------------------------------------------------


class TestGraduatedCompaction:
    def test_compaction_levels_produce_valid_messages(self):
        """Each compaction level produces a valid message list."""
        msgs = [_sys("s"), _user("q")]
        for i in range(10):
            tc = _assistant_tc([(f"tc{i}", "read_file", '{"file_path": "f.py"}')])
            tr = _tool(f"tc{i}", "x" * 2000)
            msgs.extend([tc, tr])
        msgs.append(_assistant("done"))

        # Level 1
        r1 = compact_messages(list(msgs))
        _validate_tool_pairing(r1)

        # Level 2
        r2 = drop_middle_turns(list(msgs))
        _validate_tool_pairing(r2)

        # Level 3
        r3 = aggressive_drop_turns(list(msgs))
        _validate_tool_pairing(r3)

    def test_each_level_reduces_size(self):
        """Each graduated level should produce fewer or equal tokens."""
        msgs = [_sys("s"), _user("q")]
        for i in range(10):
            tc = _assistant_tc([(f"tc{i}", "read_file", '{"file_path": "f.py"}')])
            tr = _tool(f"tc{i}", "x" * 2000)
            msgs.extend([tc, tr])
        msgs.append(_assistant("done"))

        t_original = estimate_tokens(msgs)
        t_compact = estimate_tokens(compact_messages(list(msgs)))
        t_drop = estimate_tokens(drop_middle_turns(list(msgs)))
        t_aggressive = estimate_tokens(aggressive_drop_turns(list(msgs)))

        assert t_compact <= t_original
        assert t_drop <= t_compact
        assert t_aggressive <= t_drop

    def test_graduated_levels_all_preserve_system_prompt(self):
        """All compaction levels should preserve the system prompt."""
        msgs = [_sys("You are a helpful assistant."), _user("q")]
        for i in range(8):
            tc = _assistant_tc([(f"tc{i}", "read_file", "{}")])
            tr = _tool(f"tc{i}", "x" * 2000)
            msgs.extend([tc, tr])

        for compact_fn in [compact_messages, drop_middle_turns, aggressive_drop_turns]:
            result = compact_fn(list(msgs))
            first_role = (
                result[0].get("role")
                if isinstance(result[0], dict)
                else getattr(result[0], "role", None)
            )
            assert first_role == "system"


# ---------------------------------------------------------------------------
# _fix_orphaned_tool_calls
# ---------------------------------------------------------------------------


class TestFixOrphanedToolCalls:
    def test_removes_orphaned_tool_calls(self):
        tc = _assistant_tc([("tc1", "read_file", "{}"), ("tc2", "grep", "{}")])
        tr1 = _tool("tc1", "content1")
        # tc2 result is missing
        msgs = [_user("q"), tc, tr1, _assistant("done")]
        assert _fix_orphaned_tool_calls(msgs) is True
        # tc should now only have tc1
        remaining = tc.tool_calls
        assert len(remaining) == 1
        assert remaining[0].id == "tc1"

    def test_removes_all_tool_calls_sets_content(self):
        tc = _assistant_tc([("tc1", "read_file", "{}")])
        # No tool result at all
        msgs = [_user("q"), tc, _assistant("done")]
        assert _fix_orphaned_tool_calls(msgs) is True
        assert tc.tool_calls is None
        assert tc.content == ""

    def test_noop_when_all_results_present(self):
        tc = _assistant_tc([("tc1", "read_file", "{}")])
        tr = _tool("tc1", "ok")
        msgs = [_user("q"), tc, tr, _assistant("done")]
        assert _fix_orphaned_tool_calls(msgs) is False

    def test_dict_messages(self):
        tc = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc1", "function": {"name": "f", "arguments": "{}"}}],
        }
        msgs = [_user("q"), tc, _assistant("done")]
        assert _fix_orphaned_tool_calls(msgs) is True
        assert "tool_calls" not in tc
        assert tc["content"] == ""

    def test_preserves_content_when_tool_calls_removed(self):
        tc = SimpleNamespace(
            role="assistant",
            content="I'll read the file",
            tool_calls=[
                SimpleNamespace(
                    id="tc1", function=SimpleNamespace(name="f", arguments="{}")
                )
            ],
        )
        msgs = [_user("q"), tc, _assistant("done")]
        assert _fix_orphaned_tool_calls(msgs) is True
        assert tc.tool_calls is None
        assert tc.content == "I'll read the file"
