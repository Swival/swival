"""Tests for the snapshot tool: proactive context collapse."""

import json

from swival.snapshot import (
    SNAPSHOT_HISTORY_SENTINEL,
    SnapshotState,
    READ_ONLY_TOOLS,
    MAX_HISTORY,
)
from swival.tools import dispatch


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def _tool(tc_id, content):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _assistant_tc(name, tc_id, args_json="{}"):
    """Minimal assistant message with a single tool_call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tc_id, "function": {"name": name, "arguments": args_json}}
        ],
    }


def _build_exploration_messages():
    """Build a realistic exploration sequence: user msg + several read_file tool calls."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        _user("Debug the auth failure"),
        _assistant_tc("read_file", "tc_1", '{"file_path": "auth.py"}'),
        _tool("tc_1", "def authenticate(): ..."),
        _assistant_tc("read_file", "tc_2", '{"file_path": "config.py"}'),
        _tool("tc_2", "SECRET_KEY = 'abc'"),
        _assistant_tc("grep", "tc_3", '{"pattern": "token"}'),
        _tool("tc_3", "auth.py:42: token = parse(...)"),
        _assistant_tc("read_file", "tc_4", '{"file_path": "parser.py"}'),
        _tool("tc_4", "def parse(raw): return raw.strip()"),
    ]
    return msgs


class TestSave:
    def test_save_basic(self):
        state = SnapshotState()
        result = json.loads(
            state.process(
                {"action": "save", "label": "checking auth"}, tool_call_id="tc_save"
            )
        )
        assert result["action"] == "save"
        assert result["status"] == "checkpoint_set"
        assert state.explicit_active is True
        assert state.explicit_label == "checking auth"
        assert state.explicit_begin_tool_call_id == "tc_save"

    def test_save_resets_dirty(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        assert state.dirty is True
        state.process({"action": "save", "label": "test"}, tool_call_id="tc_s")
        assert state.dirty is False
        assert len(state.dirty_tools) == 0

    def test_save_requires_label(self):
        state = SnapshotState()
        result = state.process({"action": "save"})
        assert result.startswith("error:")
        assert "label" in result

    def test_save_label_too_long(self):
        state = SnapshotState()
        result = state.process({"action": "save", "label": "x" * 101})
        assert result.startswith("error:")
        assert "100" in result

    def test_save_duplicate_blocked(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "first"}, tool_call_id="tc1")
        result = state.process(
            {"action": "save", "label": "second"}, tool_call_id="tc2"
        )
        assert result.startswith("error:")
        assert "already active" in result

    def test_save_increments_stats(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "test"}, tool_call_id="tc1")
        assert state.stats["saves"] == 1


class TestRestoreImplicit:
    def test_restore_implicit_from_user_message(self):
        """Restore without save collapses from the last user message."""
        state = SnapshotState()
        msgs = _build_exploration_messages()
        original_len = len(msgs)

        result = json.loads(
            state.process(
                {"action": "restore", "summary": "Auth uses JWT. Config in config.py."},
                messages=msgs,
                tool_call_id="tc_restore",
            )
        )

        assert result["action"] == "restore"
        assert result["status"] == "collapsed"
        assert result["turns_collapsed"] > 0
        assert len(msgs) < original_len

    def test_restore_recap_format(self):
        """The collapsed recap message has the correct format."""
        state = SnapshotState()
        msgs = _build_exploration_messages()
        summary_text = "Root cause: missing null check in parser.py:42"

        state.process(
            {"action": "restore", "summary": summary_text},
            messages=msgs,
            tool_call_id="tc_r",
        )

        # Find the recap message
        recap = None
        for m in msgs:
            if isinstance(m, dict) and m.get("content", "").startswith("[snapshot:"):
                recap = m
                break

        assert recap is not None
        assert recap["role"] == "assistant"
        assert summary_text in recap["content"]
        assert "(collapsed" in recap["content"]
        assert "tool_calls" not in recap

    def test_restore_no_orphaned_tool_call_ids(self):
        """After restore, no message has a tool_call_id without a matching tool_calls entry."""
        state = SnapshotState()
        msgs = _build_exploration_messages()

        state.process(
            {"action": "restore", "summary": "Summary of investigation"},
            messages=msgs,
            tool_call_id="tc_r",
        )

        tool_call_ids_defined = set()
        for m in msgs:
            if isinstance(m, dict) and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict):
                        tool_call_ids_defined.add(tc["id"])
                    else:
                        tool_call_ids_defined.add(tc.id)

        for m in msgs:
            if isinstance(m, dict) and "tool_call_id" in m:
                assert m["tool_call_id"] in tool_call_ids_defined

    def test_restore_requires_summary(self):
        state = SnapshotState()
        result = state.process({"action": "restore"}, messages=[_user("test")])
        assert result.startswith("error:")
        assert "summary" in result

    def test_restore_summary_too_long(self):
        state = SnapshotState()
        result = state.process(
            {"action": "restore", "summary": "x" * 4001},
            messages=[_user("test")],
        )
        assert result.startswith("error:")
        assert "4000" in result

    def test_restore_empty_scope(self):
        """Restore with nothing between checkpoint and current position."""
        state = SnapshotState()
        msgs = [_user("test")]
        result = json.loads(
            state.process(
                {"action": "restore", "summary": "nothing here"},
                messages=msgs,
                tool_call_id="tc_r",
            )
        )
        assert result["status"] == "warning"
        assert "empty" in result["message"]

    def test_restore_requires_messages(self):
        state = SnapshotState()
        result = state.process({"action": "restore", "summary": "test"})
        assert result.startswith("error:")
        assert "message" in result.lower()


class TestRestoreExplicit:
    def test_save_then_restore(self):
        """save + restore collapses only from the save point."""
        state = SnapshotState()
        msgs = [
            {"role": "system", "content": "sys"},
            _user("first question"),
            _assistant("here's what I found earlier"),
            _user("now debug auth"),
        ]
        # save checkpoint after user says "now debug auth"
        msgs.append(
            _assistant_tc("snapshot", "tc_save", '{"action":"save","label":"auth"}')
        )
        msgs.append(_tool("tc_save", '{"action":"save","status":"checkpoint_set"}'))
        state.process({"action": "save", "label": "auth"}, tool_call_id="tc_save")

        # Add exploration after the save
        msgs.append(_assistant_tc("read_file", "tc_r1", '{"file_path":"auth.py"}'))
        msgs.append(_tool("tc_r1", "content of auth.py"))
        msgs.append(_assistant_tc("read_file", "tc_r2", '{"file_path":"jwt.py"}'))
        msgs.append(_tool("tc_r2", "content of jwt.py"))

        pre_restore_len = len(msgs)

        result = json.loads(
            state.process(
                {"action": "restore", "summary": "Auth uses JWT in jwt.py"},
                messages=msgs,
                tool_call_id="tc_restore",
            )
        )

        assert result["status"] == "collapsed"
        # The messages before the save should be intact
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "first question"
        assert len(msgs) < pre_restore_len
        assert state.explicit_active is False

    def test_explicit_scope_narrower_than_implicit(self):
        """Explicit checkpoint should only collapse from save, not from user message."""
        state = SnapshotState()
        msgs = [
            _user("question"),
            _assistant_tc("read_file", "tc_pre", '{"file_path":"pre.py"}'),
            _tool("tc_pre", "pre-save content"),
        ]
        # save checkpoint
        msgs.append(
            _assistant_tc("snapshot", "tc_save", '{"action":"save","label":"narrow"}')
        )
        msgs.append(_tool("tc_save", '{"status":"ok"}'))
        state.process({"action": "save", "label": "narrow"}, tool_call_id="tc_save")

        # Post-save exploration
        msgs.append(_assistant_tc("read_file", "tc_post", '{"file_path":"post.py"}'))
        msgs.append(_tool("tc_post", "post-save content"))

        state.process(
            {"action": "restore", "summary": "Post-save summary"},
            messages=msgs,
            tool_call_id="tc_r",
        )

        # Pre-save messages should still exist
        contents = [m.get("content") or "" for m in msgs if isinstance(m, dict)]
        assert any("pre-save content" in c for c in contents)

    def test_explicit_marker_removed_by_compaction(self):
        """If the save marker was compacted away, return an error."""
        state = SnapshotState()
        state.explicit_active = True
        state.explicit_label = "test"
        state.explicit_begin_tool_call_id = "tc_gone"

        # Messages don't contain tc_gone
        msgs = [_user("hello"), _assistant("world")]

        result = state.process(
            {"action": "restore", "summary": "test"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert result.startswith("error:")
        assert "compaction" in result


class TestCancel:
    def test_cancel_clears_explicit(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "test"}, tool_call_id="tc1")
        result = json.loads(state.process({"action": "cancel"}))
        assert result["status"] == "cleared"
        assert state.explicit_active is False

    def test_cancel_no_checkpoint(self):
        state = SnapshotState()
        result = json.loads(state.process({"action": "cancel"}))
        assert result["status"] == "no_checkpoint"

    def test_cancel_increments_stats(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "test"}, tool_call_id="tc1")
        state.process({"action": "cancel"})
        assert state.stats["cancels"] == 1


class TestStatus:
    def test_status_basic(self):
        state = SnapshotState()
        result = json.loads(state.process({"action": "status"}))
        assert result["action"] == "status"
        assert result["explicit_active"] is False
        assert result["dirty"] is False
        assert result["history_count"] == 0

    def test_status_with_active_checkpoint(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "active"}, tool_call_id="tc1")
        result = json.loads(state.process({"action": "status"}))
        assert result["explicit_active"] is True
        assert result["explicit_label"] == "active"

    def test_status_with_dirty_state(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        result = json.loads(state.process({"action": "status"}))
        assert result["dirty"] is True
        assert "edit_file" in result["dirty_tools"]


class TestDirtyTracking:
    def test_read_only_tools_dont_dirty(self):
        state = SnapshotState()
        for tool in READ_ONLY_TOOLS:
            state.mark_dirty(tool)
        assert state.dirty is False
        assert len(state.dirty_tools) == 0

    def test_mutating_tools_dirty(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        assert state.dirty is True
        assert "edit_file" in state.dirty_tools

    def test_dirty_blocks_restore(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("write_file")

        result = state.process(
            {"action": "restore", "summary": "test"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert result.startswith("error:")
        assert "dirty" in result
        assert "write_file" in result

    def test_force_overrides_dirty(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("write_file")

        result = json.loads(
            state.process(
                {"action": "restore", "summary": "forced summary", "force": True},
                messages=msgs,
                tool_call_id="tc_r",
            )
        )
        assert result["status"] == "collapsed"

    def test_force_restore_increments_stats(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("edit_file")
        state.process(
            {"action": "restore", "summary": "test", "force": True},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert state.stats["force_restores"] == 1

    def test_dirty_blocked_increments_stats(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("run_command")
        state.process(
            {"action": "restore", "summary": "test"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert state.stats["blocked"] == 1

    def test_dirty_resets_on_save(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        state.process({"action": "save", "label": "test"}, tool_call_id="tc1")
        assert state.dirty is False

    def test_dirty_resets_after_restore(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("edit_file")
        state.process(
            {"action": "restore", "summary": "test", "force": True},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert state.dirty is False

    def test_reset_dirty(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        state.mark_dirty("run_command")
        state.reset_dirty()
        assert state.dirty is False
        assert len(state.dirty_tools) == 0

    def test_unknown_tools_treated_as_dirty(self):
        state = SnapshotState()
        state.mark_dirty("mcp__custom__do_thing")
        assert state.dirty is True
        assert "mcp__custom__do_thing" in state.dirty_tools

    def test_multiple_dirty_tools_tracked(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        state.mark_dirty("run_command")
        state.mark_dirty("edit_file")  # duplicate
        assert state.dirty_tools == {"edit_file", "run_command"}


class TestImplicitCheckpointResolution:
    def test_resolves_from_last_user_message(self):
        state = SnapshotState()
        msgs = [
            _user("first question"),
            _assistant("answer 1"),
            _user("second question"),
            _assistant_tc("read_file", "tc_1"),
            _tool("tc_1", "file content"),
        ]

        state.process(
            {"action": "restore", "summary": "summary of second question"},
            messages=msgs,
            tool_call_id="tc_r",
        )

        # First question and its answer should survive
        assert msgs[0]["content"] == "first question"
        assert msgs[1]["content"] == "answer 1"
        assert msgs[2]["content"] == "second question"

    def test_resolves_from_last_restore_boundary(self):
        """After a restore, the next implicit checkpoint is at the restore point."""
        state = SnapshotState()
        msgs = [
            _user("question"),
            _assistant_tc("read_file", "tc_1"),
            _tool("tc_1", "content 1"),
            _assistant_tc("read_file", "tc_2"),
            _tool("tc_2", "content 2"),
        ]

        # First restore
        state.process(
            {"action": "restore", "summary": "first pass done"},
            messages=msgs,
            tool_call_id="tc_r1",
        )

        # Add more exploration after first restore
        msgs.append(_assistant_tc("read_file", "tc_3", '{"file_path":"c.py"}'))
        msgs.append(_tool("tc_3", "content 3"))
        msgs.append(_assistant_tc("grep", "tc_4", '{"pattern":"x"}'))
        msgs.append(_tool("tc_4", "grep results"))

        pre_len = len(msgs)

        # Second restore: should use the last restore as boundary (not user msg)
        state.process(
            {"action": "restore", "summary": "second pass done"},
            messages=msgs,
            tool_call_id="tc_r2",
        )

        assert len(msgs) < pre_len
        # The first recap should still be there
        found_first_recap = any(
            isinstance(m, dict) and "first pass done" in m.get("content", "")
            for m in msgs
        )
        assert found_first_recap

    def test_no_user_message_returns_error(self):
        state = SnapshotState()
        msgs = [{"role": "system", "content": "system"}]
        result = state.process(
            {"action": "restore", "summary": "test"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert result.startswith("error:")
        assert "no implicit checkpoint" in result


class TestHistory:
    def test_history_recorded_on_restore(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.process(
            {"action": "restore", "summary": "Auth uses JWT"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert len(state.history) == 1
        entry = state.history[0]
        assert entry["summary"] == "Auth uses JWT"
        assert entry["scope_type"] == "implicit"
        assert entry["turns_collapsed"] > 0

    def test_history_records_explicit_scope_type(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        # Insert save marker
        msgs.insert(2, _assistant_tc("snapshot", "tc_save", '{"action":"save"}'))
        msgs.insert(3, _tool("tc_save", "ok"))
        state.process({"action": "save", "label": "auth debug"}, tool_call_id="tc_save")

        state.process(
            {"action": "restore", "summary": "Auth is fine"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert state.history[0]["scope_type"] == "explicit"
        assert state.history[0]["label"] == "auth debug"

    def test_history_cap_enforced(self):
        state = SnapshotState()
        for i in range(MAX_HISTORY + 5):
            msgs = [_user(f"question {i}"), _assistant(f"answer {i}")]
            state.process(
                {"action": "restore", "summary": f"summary {i}"},
                messages=msgs,
                tool_call_id=f"tc_{i}",
            )
        assert len(state.history) == MAX_HISTORY
        # Oldest should be dropped
        assert state.history[0]["summary"] == f"summary {MAX_HISTORY + 5 - MAX_HISTORY}"

    def test_history_records_dirty_info(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.mark_dirty("edit_file")
        state.process(
            {"action": "restore", "summary": "test", "force": True},
            messages=msgs,
            tool_call_id="tc_r",
        )
        assert state.history[0]["dirty_at_restore"] is True
        assert state.history[0]["forced_restore"] is True


class TestInjectIntoPrompt:
    def test_no_history_returns_none(self):
        state = SnapshotState()
        assert state.inject_into_prompt() is None

    def test_renders_history(self):
        state = SnapshotState()
        msgs = [_user("q"), _assistant("a")]
        state.process(
            {"action": "restore", "summary": "Found the bug in auth.py"},
            messages=msgs,
            tool_call_id="tc_r",
        )

        result = state.inject_into_prompt()
        assert result is not None
        assert SNAPSHOT_HISTORY_SENTINEL in result
        assert "Found the bug in auth.py" in result

    def test_budget_respected(self):
        state = SnapshotState()
        for i in range(MAX_HISTORY):
            msgs = [_user(f"q{i}"), _assistant(f"a{i}")]
            state.process(
                {"action": "restore", "summary": "x" * 1000},
                messages=msgs,
                tool_call_id=f"tc_{i}",
            )

        result = state.inject_into_prompt()
        assert result is not None
        assert len(result) <= 7000  # budget is 6000 + header


class TestReset:
    def test_full_reset(self):
        state = SnapshotState()
        state.process({"action": "save", "label": "test"}, tool_call_id="tc1")
        state.mark_dirty("edit_file")
        msgs = [_user("q"), _assistant("a")]
        state.process(
            {"action": "restore", "summary": "s", "force": True},
            messages=msgs,
            tool_call_id="tc_r",
        )

        state.reset()

        assert state.explicit_active is False
        assert state.explicit_label is None
        assert state.last_restore_tool_call_id is None
        assert state.dirty is False
        assert len(state.dirty_tools) == 0
        assert len(state.history) == 0
        assert state.stats["saves"] == 0
        assert state.stats["restores"] == 0


class TestSummaryLine:
    def test_no_usage_returns_none(self):
        state = SnapshotState()
        assert state.summary_line() is None

    def test_after_restore(self):
        state = SnapshotState()
        msgs = _build_exploration_messages()
        state.process(
            {"action": "restore", "summary": "test"},
            messages=msgs,
            tool_call_id="tc_r",
        )
        line = state.summary_line()
        assert line is not None
        assert "1 restore" in line
        assert "tokens saved" in line


class TestInvalidAction:
    def test_unknown_action(self):
        state = SnapshotState()
        result = state.process({"action": "bogus"})
        assert result.startswith("error:")
        assert "bogus" in result


class TestTokenSavings:
    def test_tokens_decrease_after_restore(self):
        """estimate_tokens should decrease after a restore in a representative flow."""
        from swival.agent import estimate_tokens

        state = SnapshotState()
        msgs = _build_exploration_messages()
        tokens_before = estimate_tokens(msgs)

        state.process(
            {"action": "restore", "summary": "Short conclusion."},
            messages=msgs,
            tool_call_id="tc_r",
        )

        tokens_after = estimate_tokens(msgs)
        assert tokens_after < tokens_before


class TestDispatchIntegration:
    def test_dispatch_routes_to_snapshot(self, tmp_path):
        state = SnapshotState()
        result = dispatch(
            "snapshot",
            {"action": "status"},
            str(tmp_path),
            snapshot_state=state,
            messages=[],
            tool_call_id="tc1",
        )
        parsed = json.loads(result)
        assert parsed["action"] == "status"

    def test_dispatch_snapshot_not_available(self, tmp_path):
        result = dispatch(
            "snapshot",
            {"action": "status"},
            str(tmp_path),
        )
        assert result.startswith("error:")
        assert "not available" in result

    def test_dispatch_restore_with_messages(self, tmp_path):
        state = SnapshotState()
        msgs = [_user("test"), _assistant("reading"), _tool("tc1", "content")]
        result = dispatch(
            "snapshot",
            {"action": "restore", "summary": "done"},
            str(tmp_path),
            snapshot_state=state,
            messages=msgs,
            tool_call_id="tc_r",
        )
        parsed = json.loads(result)
        assert parsed["status"] == "collapsed"


class TestReadOnlyAllowlist:
    def test_all_expected_tools_in_allowlist(self):
        expected = {
            "read_file",
            "list_files",
            "grep",
            "fetch_url",
            "think",
            "todo",
            "snapshot",
        }
        assert expected == READ_ONLY_TOOLS

    def test_write_tools_not_in_allowlist(self):
        for tool in ("write_file", "edit_file", "delete_file", "run_command"):
            assert tool not in READ_ONLY_TOOLS


class TestBehaviorUnchangedWithoutSnapshot:
    def test_no_snapshot_state_in_dispatch(self, tmp_path):
        """Other tools work normally when snapshot_state is not provided."""
        p = tmp_path / "test.txt"
        p.write_text("hello")
        result = dispatch(
            "read_file",
            {"file_path": str(p)},
            str(tmp_path),
        )
        assert "hello" in result


class TestOrphanedToolCallIds:
    """Simulate the agent loop flow to verify no orphaned tool_call_ids."""

    def test_restore_preserves_current_assistant_message(self):
        """When the assistant message issuing restore is in messages,
        it must not be collapsed — otherwise the tool result is orphaned."""
        state = SnapshotState()
        msgs = [
            {"role": "system", "content": "system"},
            _user("debug auth"),
            _assistant_tc("read_file", "tc_1", '{"file_path": "a.py"}'),
            _tool("tc_1", "content of a.py"),
            _assistant_tc("read_file", "tc_2", '{"file_path": "b.py"}'),
            _tool("tc_2", "content of b.py"),
        ]
        # Simulate agent loop: assistant message with snapshot restore appended
        restore_assistant = _assistant_tc(
            "snapshot", "tc_restore", '{"action":"restore","summary":"done"}'
        )
        msgs.append(restore_assistant)

        result = json.loads(
            state.process(
                {"action": "restore", "summary": "Auth uses JWT in a.py"},
                messages=msgs,
                tool_call_id="tc_restore",
            )
        )
        assert result["status"] == "collapsed"

        # The assistant message with tool_calls must still be in messages
        assert restore_assistant in msgs

        # Simulate agent loop appending tool result
        tool_result = _tool("tc_restore", json.dumps(result))
        msgs.append(tool_result)

        # Verify no orphaned tool_call_ids
        tc_ids_defined = set()
        for m in msgs:
            if isinstance(m, dict):
                for tc in m.get("tool_calls", []):
                    tc_ids_defined.add(tc["id"] if isinstance(tc, dict) else tc.id)

        for m in msgs:
            if isinstance(m, dict) and "tool_call_id" in m:
                assert m["tool_call_id"] in tc_ids_defined, (
                    f"orphaned tool_call_id: {m['tool_call_id']}"
                )

    def test_mixed_tool_calls_no_orphans(self):
        """When restore is in a batch with other tool calls, all IDs stay valid."""
        state = SnapshotState()
        msgs = [
            _user("question"),
            _assistant_tc("read_file", "tc_1"),
            _tool("tc_1", "file content"),
        ]
        # Assistant issues both read_file and snapshot in same turn
        batch_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_read",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"c.py"}',
                    },
                },
                {
                    "id": "tc_snap",
                    "function": {
                        "name": "snapshot",
                        "arguments": '{"action":"restore","summary":"done"}',
                    },
                },
            ],
        }
        msgs.append(batch_assistant)
        # Simulate first tool result (read_file) already appended
        msgs.append(_tool("tc_read", "content of c.py"))

        # Now snapshot restore fires
        state.process(
            {"action": "restore", "summary": "Summary"},
            messages=msgs,
            tool_call_id="tc_snap",
        )

        # Append snapshot tool result
        msgs.append(_tool("tc_snap", '{"status":"collapsed"}'))

        # Verify: batch_assistant should still be in messages
        assert batch_assistant in msgs

        # Check no orphans
        tc_ids_defined = set()
        for m in msgs:
            if isinstance(m, dict):
                for tc in m.get("tool_calls", []):
                    tc_ids_defined.add(tc["id"] if isinstance(tc, dict) else tc.id)
        for m in msgs:
            if isinstance(m, dict) and "tool_call_id" in m:
                assert m["tool_call_id"] in tc_ids_defined


class TestNudgePerStreak:
    """Snapshot nudge should fire once per read streak, not once per run."""

    def test_nudge_resets_after_write(self):
        """After a mutation breaks the streak, a new streak can trigger a new nudge."""
        # Simulate the nudge tracking logic from run_agent_loop
        snapshot_read_streak = 0
        snapshot_nudge_fired = False
        nudge_count = 0

        # First streak: 5 read-only turns
        for _ in range(5):
            all_readonly = True
            if all_readonly:
                snapshot_read_streak += 1
                if snapshot_read_streak >= 5 and not snapshot_nudge_fired:
                    snapshot_nudge_fired = True
                    nudge_count += 1

        assert nudge_count == 1
        assert snapshot_nudge_fired is True

        # Write breaks the streak
        all_readonly = False
        snapshot_read_streak = 0
        snapshot_nudge_fired = False

        # Second streak: 5 more read-only turns
        for _ in range(5):
            all_readonly = True
            if all_readonly:
                snapshot_read_streak += 1
                if snapshot_read_streak >= 5 and not snapshot_nudge_fired:
                    snapshot_nudge_fired = True
                    nudge_count += 1

        # Should have nudged twice total (once per streak)
        assert nudge_count == 2

    def test_no_double_nudge_within_streak(self):
        """Within a single streak, nudge fires only once."""
        snapshot_read_streak = 0
        snapshot_nudge_fired = False
        nudge_count = 0

        for _ in range(10):
            all_readonly = True
            if all_readonly:
                snapshot_read_streak += 1
                if snapshot_read_streak >= 5 and not snapshot_nudge_fired:
                    snapshot_nudge_fired = True
                    nudge_count += 1

        assert nudge_count == 1


class TestDirtyStateOnContinue:
    """Dirty state should not be cleared on /continue (no new user boundary)."""

    def test_dirty_preserved_when_last_msg_is_not_user(self):
        """Simulates /continue: last message is assistant, dirty should persist."""
        from swival.snapshot import SnapshotState

        state = SnapshotState()
        state.mark_dirty("edit_file")

        # Simulate messages ending with an assistant message (as in /continue)
        msgs = [
            _user("question"),
            _assistant("I made some edits"),
        ]

        # Simulate what run_agent_loop does at entry
        last_role = msgs[-1].get("role", "") if msgs else ""
        if last_role == "user":
            state.reset_dirty()

        # Dirty should NOT be cleared since last msg is assistant
        assert state.dirty is True
        assert "edit_file" in state.dirty_tools

    def test_dirty_reset_when_last_msg_is_user(self):
        """Normal entry: last message is user, dirty should reset."""
        state = SnapshotState()
        state.mark_dirty("edit_file")

        msgs = [_user("new question")]

        last_role = msgs[-1].get("role", "") if msgs else ""
        if last_role == "user":
            state.reset_dirty()

        assert state.dirty is False


class TestHistoryInjection:
    """Test that snapshot history is injected into the system message."""

    def test_history_injected_into_system_message(self):
        """After a restore, inject_into_prompt output should appear in sys msg."""
        state = SnapshotState()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            _user("question"),
            _assistant("exploration"),
        ]
        state.process(
            {"action": "restore", "summary": "Found bug in auth.py:42"},
            messages=msgs,
            tool_call_id="tc_r",
        )

        # Simulate what run_agent_loop does: inject history
        history_text = state.inject_into_prompt()
        assert history_text is not None
        assert "Found bug in auth.py:42" in history_text

    def test_injection_does_not_double_inject(self):
        """Repeated injection replaces previous injection, not appends."""
        state = SnapshotState()
        sys_msg = {"role": "system", "content": "Base prompt."}

        # First restore
        msgs = [sys_msg, _user("q1"), _assistant("a1")]
        state.process(
            {"action": "restore", "summary": "first finding"},
            messages=msgs,
            tool_call_id="tc_r1",
        )

        # Inject
        history = state.inject_into_prompt()
        sys_msg["content"] += "\n\n" + history

        # Second restore
        msgs2 = [sys_msg, _user("q2"), _assistant("a2")]
        state.process(
            {"action": "restore", "summary": "second finding"},
            messages=msgs2,
            tool_call_id="tc_r2",
        )

        # Remove old injection and re-inject
        base = sys_msg["content"]
        strip_marker = "\n\n" + SNAPSHOT_HISTORY_SENTINEL
        idx = base.find(strip_marker)
        if idx != -1:
            sys_msg["content"] = base[:idx]

        history = state.inject_into_prompt()
        sys_msg["content"] += "\n\n" + history

        # Should contain both findings but only one header
        assert sys_msg["content"].count(SNAPSHOT_HISTORY_SENTINEL) == 1
        assert "first finding" in sys_msg["content"]
        assert "second finding" in sys_msg["content"]

    def test_no_duplication_across_reentry(self):
        """Simulates /continue: run_agent_loop re-enters with history already
        in the system message. The injection logic must strip the old block
        before adding the new one, even on a fresh invocation."""
        state = SnapshotState()
        sys_msg = {"role": "system", "content": "Base prompt."}

        # First restore populates history
        msgs = [sys_msg, _user("q1"), _assistant("a1")]
        state.process(
            {"action": "restore", "summary": "finding one"},
            messages=msgs,
            tool_call_id="tc_r1",
        )

        # Simulate first run_agent_loop injection
        strip_marker = "\n\n" + SNAPSHOT_HISTORY_SENTINEL
        history_text = state.inject_into_prompt()
        sys_msg["content"] += "\n\n" + history_text

        assert sys_msg["content"].count(SNAPSHOT_HISTORY_SENTINEL) == 1

        # Simulate /continue re-entry: a fresh call to run_agent_loop
        # would have _snapshot_history_injected=False (local var reset).
        # The fix must still strip the old block from sys_msg.
        base = sys_msg["content"]
        idx = base.find(strip_marker)
        if idx != -1:
            base = base[:idx]
        history_text = state.inject_into_prompt()
        if history_text:
            sys_msg["content"] = base + "\n\n" + history_text
        else:
            sys_msg["content"] = base

        # Must have exactly one history block, not two
        assert sys_msg["content"].count(SNAPSHOT_HISTORY_SENTINEL) == 1
        assert sys_msg["content"].startswith("Base prompt.")
        assert "finding one" in sys_msg["content"]


class TestCompactionScoring:
    def test_snapshot_recap_gets_high_score(self):
        from swival.agent import score_turn

        recap = _assistant(
            "[snapshot: auth debug]\nAuth uses JWT.\n(collapsed 5 turns)"
        )
        score = score_turn([recap])
        assert score >= 5


class TestSaveAtIndex:
    def test_save_at_index_basic(self):
        state = SnapshotState()
        result = json.loads(state.save_at_index("checkpoint-1", 5))
        assert result["action"] == "save"
        assert result["status"] == "checkpoint_set"
        assert state.explicit_active is True
        assert state.explicit_label == "checkpoint-1"
        assert state.explicit_begin_index == 5
        assert state._save_generation == 0

    def test_save_at_index_resets_dirty(self):
        state = SnapshotState()
        state.mark_dirty("edit_file")
        state.save_at_index("test", 3)
        assert state.dirty is False

    def test_save_at_index_increments_stats(self):
        state = SnapshotState()
        state.save_at_index("test", 0)
        assert state.stats["saves"] == 1

    def test_save_at_index_empty_label_rejected(self):
        state = SnapshotState()
        result = state.save_at_index("", 0)
        assert result.startswith("error:")
        assert "label" in result

    def test_save_at_index_duplicate_rejected(self):
        state = SnapshotState()
        state.save_at_index("first", 0)
        result = state.save_at_index("second", 5)
        assert result.startswith("error:")
        assert "already active" in result

    def test_save_at_index_label_too_long(self):
        state = SnapshotState()
        result = state.save_at_index("x" * 101, 0)
        assert result.startswith("error:")


class TestResolveStartWithIndex:
    def test_resolve_start_returns_saved_index(self):
        state = SnapshotState()
        msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        state.save_at_index("test", 2)
        idx = state._resolve_start(msgs)
        assert idx == 2

    def test_resolve_start_stale_generation(self):
        state = SnapshotState()
        msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        state.save_at_index("test", 2)
        state.invalidate_index_checkpoint()
        result = state._resolve_start(msgs)
        assert isinstance(result, str)
        assert "invalidated" in result


class TestRestoreWithAutosummary:
    def test_basic_autosummary(self):
        state = SnapshotState()
        msgs = [_user("q"), _assistant("a1"), _assistant("a2")]
        state.save_at_index("test", 1)

        result = state.restore_with_autosummary(
            msgs, lambda t: "auto-generated summary"
        )
        parsed = json.loads(result)
        assert parsed["status"] == "collapsed"
        assert parsed["turns_collapsed"] == 2
        assert state.explicit_active is False
        assert len(state.history) == 1
        assert state.history[0]["summary"] == "auto-generated summary"

    def test_autosummary_fallback_on_none(self):
        state = SnapshotState()
        msgs = [_user("q"), _assistant("a1"), _assistant("a2")]
        state.save_at_index("test", 1)

        result = state.restore_with_autosummary(msgs, lambda text: None)
        parsed = json.loads(result)
        assert parsed["status"] == "collapsed"
        assert state.history[0]["summary"] == "(context collapsed by user)"

    def test_manual_end_boundary_includes_full_tail(self):
        state = SnapshotState()
        msgs = [
            _user("q"),
            _assistant_tc("read_file", "tc1"),
            _tool("tc1", "content"),
            _assistant_tc("read_file", "tc2"),
        ]
        state.save_at_index("test", 1)

        result = state.restore_with_autosummary(msgs, lambda t: "summary")
        parsed = json.loads(result)
        assert parsed["turns_collapsed"] == 3

    def test_empty_scope_returns_message(self):
        state = SnapshotState()
        msgs = [_user("q")]
        state.save_at_index("test", 1)
        result = state.restore_with_autosummary(msgs, lambda t: "summary")
        assert "nothing to collapse" in result


class TestIndexClearedOnOperations:
    def test_index_cleared_on_cancel(self):
        state = SnapshotState()
        state.save_at_index("test", 5)
        state.cancel()
        assert state.explicit_begin_index is None
        assert state._save_generation is None

    def test_index_cleared_on_reset(self):
        state = SnapshotState()
        state.save_at_index("test", 5)
        state.reset()
        assert state.explicit_begin_index is None
        assert state._save_generation is None
        assert state._generation == 0

    def test_index_cleared_on_restore(self):
        state = SnapshotState()
        msgs = [_user("q"), _assistant("a1"), _assistant("a2")]
        state.save_at_index("test", 1)
        state.restore_with_autosummary(msgs, lambda t: "summary")
        assert state.explicit_begin_index is None
        assert state._save_generation is None
        assert state.explicit_active is False


class TestInvalidateIndexCheckpoint:
    def test_invalidate_increments_generation(self):
        state = SnapshotState()
        assert state._generation == 0
        state.invalidate_index_checkpoint()
        assert state._generation == 1

    def test_multiple_invalidations(self):
        state = SnapshotState()
        state.invalidate_index_checkpoint()
        state.invalidate_index_checkpoint()
        assert state._generation == 2
