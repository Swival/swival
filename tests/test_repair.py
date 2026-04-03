"""Tests for schema-aware tool-call argument repair."""

from swival.repair import repair_tool_args


SCHEMA_READ_FILE = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "offset": {"type": "integer", "default": 1},
        "limit": {"type": "integer", "default": 2000},
        "tail": {"type": "integer", "minimum": 1},
    },
    "required": ["file_path"],
}

SCHEMA_EDIT_FILE = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean", "default": False},
    },
    "required": ["file_path", "old_string", "new_string"],
}

SCHEMA_GREP = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string", "default": "."},
        "include": {"type": "string"},
        "case_insensitive": {"type": "boolean", "default": False},
        "context_lines": {"type": "integer", "minimum": 0, "default": 0},
    },
    "required": ["pattern"],
}

SCHEMA_READ_MULTIPLE = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "maxItems": 20,
        },
    },
    "required": ["files"],
}

SCHEMA_RUN_COMMAND = {
    "type": "object",
    "properties": {
        "command": {
            "type": "array",
            "items": {"type": "string"},
        },
        "timeout": {"type": "integer", "default": 30},
    },
    "required": ["command"],
}


class TestNoOpOnValidArgs:
    def test_valid_read_file(self):
        args = {"file_path": "foo.py", "offset": 10, "limit": 50}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result == args
        assert repairs == []

    def test_valid_edit_file(self):
        args = {
            "file_path": "f.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": True,
        }
        result, repairs = repair_tool_args(args, SCHEMA_EDIT_FILE)
        assert result == args
        assert repairs == []

    def test_none_schema_passthrough(self):
        args = {"anything": 42}
        result, repairs = repair_tool_args(args, None)
        assert result is args
        assert repairs == []

    def test_non_dict_args_passthrough_string(self):
        result, repairs = repair_tool_args("just a string", SCHEMA_READ_FILE)
        assert result == "just a string"
        assert repairs == []

    def test_non_dict_args_passthrough_int(self):
        result, repairs = repair_tool_args(42, SCHEMA_READ_FILE)
        assert result == 42
        assert repairs == []

    def test_non_dict_args_passthrough_list(self):
        result, repairs = repair_tool_args([1, 2, 3], SCHEMA_READ_FILE)
        assert result == [1, 2, 3]
        assert repairs == []

    def test_empty_properties_passthrough(self):
        args = {"x": 1}
        result, repairs = repair_tool_args(args, {"type": "object", "properties": {}})
        assert result == args
        assert repairs == []


class TestCoerceTypes:
    def test_string_to_integer(self):
        args = {"file_path": "f.py", "offset": "10", "limit": "50"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["offset"] == 10
        assert result["limit"] == 50
        assert len(repairs) == 2
        assert all(r["type"] == "coerce_type" for r in repairs)

    def test_string_to_boolean_true(self):
        args = {
            "file_path": "f.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": "true",
        }
        result, repairs = repair_tool_args(args, SCHEMA_EDIT_FILE)
        assert result["replace_all"] is True
        repair_types = [r["type"] for r in repairs]
        assert "coerce_type" in repair_types

    def test_string_to_boolean_false(self):
        args = {
            "file_path": "f.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": "false",
        }
        result, repairs = repair_tool_args(args, SCHEMA_EDIT_FILE)
        assert result["replace_all"] is False

    def test_int_0_1_to_boolean(self):
        args = {
            "file_path": "f.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": 1,
        }
        result, repairs = repair_tool_args(args, SCHEMA_EDIT_FILE)
        assert result["replace_all"] is True

    def test_int_to_string(self):
        args = {"file_path": 123}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["file_path"] == "123"
        assert any(r["type"] == "coerce_type" for r in repairs)

    def test_float_to_integer(self):
        args = {"file_path": "f.py", "offset": 10.0}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["offset"] == 10
        assert isinstance(result["offset"], int)

    def test_non_coercible_string_left_alone(self):
        args = {"file_path": "f.py", "offset": "not-a-number"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["offset"] == "not-a-number"
        coerce_repairs = [r for r in repairs if r["type"] == "coerce_type"]
        assert coerce_repairs == []

    def test_bool_string_yes_no(self):
        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
        result, _ = repair_tool_args({"flag": "yes"}, schema)
        assert result["flag"] is True
        result, _ = repair_tool_args({"flag": "no"}, schema)
        assert result["flag"] is False

    def test_int_2_not_coerced_to_bool(self):
        args = {
            "file_path": "f.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": 2,
        }
        result, repairs = repair_tool_args(args, SCHEMA_EDIT_FILE)
        assert result["replace_all"] == 2


class TestShapesLeftAlone:
    def test_dict_for_array_left_alone(self):
        """Dict passed for an array field is left as-is for the tool to handle."""
        args = {"files": {"file_path": "a.py"}}
        result, repairs = repair_tool_args(args, SCHEMA_READ_MULTIPLE)
        assert result["files"] == {"file_path": "a.py"}
        assert not repairs

    def test_list_for_object_left_alone(self):
        """List passed for an object field is left as-is for the tool to handle."""
        schema = {
            "type": "object",
            "properties": {"config": {"type": "object"}},
            "required": ["config"],
        }
        args = {"config": [{"key": "value"}]}
        result, repairs = repair_tool_args(args, schema)
        assert result["config"] == [{"key": "value"}]
        assert not repairs

    def test_string_for_array_left_alone(self):
        """String passed for an array[string] field is left as-is for the tool to handle."""
        args = {"command": "ls -la"}
        result, repairs = repair_tool_args(args, SCHEMA_RUN_COMMAND)
        assert result["command"] == "ls -la"
        assert not any(r["type"].startswith("wrap") for r in repairs)


class TestUnwrapNested:
    def test_unwrap_command_dict(self):
        """Model sends {"command": {"command": "ls"}} — unwrap to {"command": "ls"}."""
        args = {"command": {"command": "fastly service list"}}
        result, repairs = repair_tool_args(args, SCHEMA_RUN_COMMAND)
        assert result["command"] == "fastly service list"
        assert any(r["type"] == "unwrap_nested" for r in repairs)

    def test_unwrap_with_multiple_inner_keys(self):
        """Inner dict has multiple schema-valid keys — hoist all."""
        args = {"command": {"command": ["ls", "-la"], "timeout": 10}}
        result, repairs = repair_tool_args(args, SCHEMA_RUN_COMMAND)
        assert result["command"] == ["ls", "-la"]
        assert result["timeout"] == 10
        assert any(r["type"] == "unwrap_nested" for r in repairs)

    def test_no_unwrap_when_inner_keys_not_in_schema(self):
        """Inner dict keys don't match schema — leave alone."""
        args = {"command": {"cmd": "ls", "flags": "-la"}}
        result, repairs = repair_tool_args(args, SCHEMA_RUN_COMMAND)
        assert result["command"] == {"cmd": "ls", "flags": "-la"}
        assert not any(r["type"] == "unwrap_nested" for r in repairs)

    def test_no_unwrap_for_empty_inner_dict(self):
        args = {"command": {}}
        result, repairs = repair_tool_args(args, SCHEMA_RUN_COMMAND)
        assert result["command"] == {}
        assert not any(r["type"] == "unwrap_nested" for r in repairs)


class TestNearMissFields:
    def test_close_field_name(self):
        args = {"file_paht": "f.py"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert "file_path" in result
        assert result["file_path"] == "f.py"
        assert "file_paht" not in result
        assert any(r["type"] == "rename_field" for r in repairs)

    def test_no_rename_when_correct_field_exists(self):
        args = {"file_path": "correct.py", "file_paht": "wrong.py"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["file_path"] == "correct.py"
        assert not any(r["type"] == "rename_field" for r in repairs)
        assert any(
            r["type"] == "strip_unknown" and r["field"] == "file_paht" for r in repairs
        )

    def test_no_rename_for_distant_name(self):
        args = {"xyz_totally_wrong": "value"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        rename_repairs = [r for r in repairs if r["type"] == "rename_field"]
        assert rename_repairs == []
        # All fields are unknown so _strip_unknown preserves them to avoid
        # destroying the entire call.
        assert "xyz_totally_wrong" in result


class TestStripUnknown:
    def test_strip_extra_field(self):
        args = {"file_path": "f.py", "bogus_field": 42}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert "bogus_field" not in result
        assert any(r["type"] == "strip_unknown" for r in repairs)

    def test_no_strip_when_all_unknown_and_no_defaults(self):
        schema = {
            "type": "object",
            "properties": {"known": {"type": "string"}},
            "required": ["known"],
        }
        args = {"totally": "wrong", "all": "bad"}
        result, repairs = repair_tool_args(args, schema)
        assert result == args
        strip_repairs = [r for r in repairs if r["type"] == "strip_unknown"]
        assert strip_repairs == []

    def test_no_strip_when_no_unknown(self):
        args = {"file_path": "f.py", "offset": 1}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        strip_repairs = [r for r in repairs if r["type"] == "strip_unknown"]
        assert strip_repairs == []


SCHEMA_LIST_FILES = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match files."},
        "path": {
            "type": "string",
            "description": "File or directory to search in, relative to base directory.",
            "default": ".",
        },
    },
    "required": ["pattern"],
}

SCHEMA_OUTLINE = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to a single file to outline.",
        },
        "depth": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
    },
}


class TestPathGlobStripping:
    def test_dotstarstar_becomes_dot(self):
        args = {"pattern": "TODO", "path": ".**"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."
        assert any(r["type"] == "strip_glob_from_path" for r in repairs)

    def test_doublestar_becomes_dot(self):
        args = {"pattern": "TODO", "path": "**"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."

    def test_dot_slash_doublestar_becomes_dot(self):
        args = {"pattern": "TODO", "path": "./**"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."

    def test_dir_slash_doublestar_becomes_dir(self):
        args = {"pattern": "TODO", "path": "src/**"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "src"

    def test_plain_dot_unchanged(self):
        args = {"pattern": "TODO", "path": "."}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."
        assert not any(r["type"] == "strip_glob_from_path" for r in repairs)

    def test_normal_path_unchanged(self):
        args = {"pattern": "TODO", "path": "src/lib"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "src/lib"
        assert not any(r["type"] == "strip_glob_from_path" for r in repairs)

    def test_pattern_field_not_touched(self):
        """The pattern/include fields contain globs intentionally."""
        args = {"pattern": "**/*.py"}
        result, repairs = repair_tool_args(args, SCHEMA_LIST_FILES)
        assert result["pattern"] == "**/*.py"
        assert not any(r["type"] == "strip_glob_from_path" for r in repairs)

    def test_list_files_path_cleaned(self):
        args = {"pattern": "**/*.py", "path": ".**"}
        result, repairs = repair_tool_args(args, SCHEMA_LIST_FILES)
        assert result["path"] == "."
        assert result["pattern"] == "**/*.py"

    def test_star_becomes_dot(self):
        args = {"pattern": "TODO", "path": "*"}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."


class TestFieldAliases:
    def test_path_renamed_to_file_path(self):
        """outline has file_path not path — alias should catch it."""
        args = {"path": "src/main.py"}
        result, repairs = repair_tool_args(args, SCHEMA_OUTLINE)
        assert result["file_path"] == "src/main.py"
        assert "path" not in result
        assert any(r["type"] == "rename_field" and r["from"] == "path" for r in repairs)

    def test_path_not_renamed_when_schema_has_path(self):
        """grep has a real path field — no alias should fire."""
        args = {"pattern": "TODO", "path": "."}
        result, repairs = repair_tool_args(args, SCHEMA_GREP)
        assert result["path"] == "."
        assert not any(r["type"] == "rename_field" for r in repairs)

    def test_alias_plus_glob_strip(self):
        """outline(path=".**") → file_path="." via alias + glob strip."""
        args = {"path": ".**"}
        result, repairs = repair_tool_args(args, SCHEMA_OUTLINE)
        assert result["file_path"] == "."
        assert "path" not in result
        types = [r["type"] for r in repairs]
        assert "rename_field" in types
        assert "strip_glob_from_path" in types

    def test_alias_skipped_when_correct_field_exists(self):
        args = {"file_path": "real.py", "path": "extra.py"}
        result, repairs = repair_tool_args(args, SCHEMA_OUTLINE)
        assert result["file_path"] == "real.py"
        assert "path" not in result
        assert any(
            r["type"] == "strip_unknown" and r["field"] == "path" for r in repairs
        )


class TestCombinedRepairs:
    def test_rename_then_coerce(self):
        args = {"file_paht": "f.py", "offset": "10"}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["file_path"] == "f.py"
        assert result["offset"] == 10
        types = [r["type"] for r in repairs]
        assert "rename_field" in types
        assert "coerce_type" in types

    def test_coerce_and_strip(self):
        args = {"file_path": "f.py", "offset": "5", "junk": True}
        result, repairs = repair_tool_args(args, SCHEMA_READ_FILE)
        assert result["offset"] == 5
        assert "junk" not in result
        assert "limit" not in result
        types = [r["type"] for r in repairs]
        assert "coerce_type" in types
        assert "strip_unknown" in types

    def test_input_not_mutated(self):
        args = {"file_path": "f.py", "offset": "10", "extra": True}
        original = dict(args)
        repair_tool_args(args, SCHEMA_READ_FILE)
        assert args == original


def _loop_kwargs(tmp_path, **overrides):
    """Build required keyword arguments for run_agent_loop."""
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

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
        files_mode="all",
        verbose=False,
        llm_kwargs={},
    )
    defaults.update(overrides)
    return defaults


def _make_fake_llm(tool_args, answer="done"):
    """Return a fake call_llm that issues one tool call then a final answer."""
    import json
    import types as t

    call_count = 0

    def fake(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            tc = t.SimpleNamespace(
                id="call_1",
                function=t.SimpleNamespace(
                    name="read_file",
                    arguments=json.dumps(tool_args),
                ),
            )
            msg = t.SimpleNamespace(
                content=None,
                tool_calls=[tc],
                role="assistant",
                get=lambda key, default=None: None,
            )
            return msg, "tool_calls"
        msg = t.SimpleNamespace(
            content=answer,
            tool_calls=None,
            role="assistant",
            get=lambda key, default=None: getattr(msg, key, default),
        )
        return msg, "stop"

    return fake


class TestAgentLoopIntegration:
    """Integration test: a malformed tool call gets repaired and the run succeeds."""

    def test_repaired_read_file_succeeds(self, tmp_path, monkeypatch):
        from swival.agent import run_agent_loop

        target = tmp_path / "hello.txt"
        target.write_text("hello world\n")

        fake = _make_fake_llm(
            {"file_path": str(target), "offset": "1", "limit": "10"},
        )
        monkeypatch.setattr("swival.agent.call_llm", fake)

        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "read the file"},
        ]
        answer, exhausted = run_agent_loop(messages, [], **_loop_kwargs(tmp_path))
        assert answer == "done"
        assert not exhausted
        tool_msgs = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "tool"
        ]
        assert len(tool_msgs) == 1
        assert "hello world" in tool_msgs[0]["content"]


class TestAgentLoopReportIntegration:
    """End-to-end: repaired tool call flows through _post_tool_bookkeeping into ReportCollector."""

    def test_repairs_reach_report_through_agent_loop(self, tmp_path, monkeypatch):
        from swival.agent import run_agent_loop
        from swival.report import ReportCollector

        target = tmp_path / "data.txt"
        target.write_text("content here\n")

        report = ReportCollector()
        fake = _make_fake_llm(
            {"file_path": str(target), "offset": "1"},
            answer="ok",
        )
        monkeypatch.setattr("swival.agent.call_llm", fake)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        run_agent_loop(messages, [], **_loop_kwargs(tmp_path, report=report))

        tool_events = [e for e in report.events if e["type"] == "tool_call"]
        assert len(tool_events) == 1
        event = tool_events[0]
        assert "repairs" in event
        repair_types = [r["type"] for r in event["repairs"]]
        assert "coerce_type" in repair_types
        assert report.tool_stats["read_file"]["repairs"] >= 1


class TestReportIntegration:
    def test_repairs_recorded_in_report(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        repairs = [
            {
                "type": "coerce_type",
                "field": "offset",
                "from": "'1'",
                "to": "1",
                "expected_type": "integer",
            }
        ]
        rc.record_tool_call(
            turn=1,
            name="read_file",
            arguments={"file_path": "f.py", "offset": 1},
            succeeded=True,
            duration=0.1,
            result_length=100,
            repairs=repairs,
        )
        event = rc.events[-1]
        assert event["repairs"] == repairs
        assert rc.tool_stats["read_file"]["repairs"] == 1

    def test_no_repairs_field_when_empty(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_tool_call(
            turn=1,
            name="read_file",
            arguments={"file_path": "f.py"},
            succeeded=True,
            duration=0.1,
            result_length=100,
        )
        event = rc.events[-1]
        assert "repairs" not in event
        assert "repairs" not in rc.tool_stats["read_file"]
