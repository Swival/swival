"""Unit tests for swival.tool_call_repair."""

from __future__ import annotations

import json

import pytest

from swival.tool_call_repair import (
    FlattenMeta,
    StormBreaker,
    analyze_schema,
    content_is_pure_tool_call,
    flatten_schema,
    is_mutating,
    is_storm_exempt,
    nest_arguments,
    repair_truncated_json,
    scavenge_tool_calls,
)


ALLOWED = frozenset({"read_file", "list_files", "write_file", "edit_file"})


def test_repair_empty_input():
    r = repair_truncated_json("")
    assert r.repaired == "{}"
    assert r.changed is True
    assert r.fallback is False
    assert r.original_preview is None


def test_repair_whitespace_input():
    r = repair_truncated_json("   \n")
    assert r.repaired == "{}"
    assert r.changed is True
    assert r.fallback is False


def test_repair_already_valid_json():
    r = repair_truncated_json('{"a": 1}')
    assert r.changed is False
    assert r.fallback is False
    assert r.original_preview is None


def test_repair_truncated_string():
    r = repair_truncated_json('{"path": "src/foo')
    assert r.changed is True
    assert r.fallback is False
    parsed = json.loads(r.repaired)
    assert parsed == {"path": "src/foo"}
    assert "closed unterminated string" in r.notes
    assert r.original_preview is not None
    assert r.original_preview.startswith('{"path":')


def test_repair_dangling_key():
    r = repair_truncated_json('{"path":')
    assert r.changed is True
    assert r.fallback is False
    parsed = json.loads(r.repaired)
    assert parsed == {"path": None}


def test_repair_nested_truncation():
    r = repair_truncated_json('{"a": {"b": [1, 2')
    assert r.changed is True
    assert r.fallback is False
    parsed = json.loads(r.repaired)
    assert parsed == {"a": {"b": [1, 2]}}


def test_repair_trailing_comma():
    r = repair_truncated_json('{"a": 1,')
    assert r.changed is True
    parsed = json.loads(r.repaired)
    assert parsed == {"a": 1}


def test_repair_unrecoverable_garbage():
    r = repair_truncated_json("this is not json at all <<>>")
    assert r.fallback is True
    assert r.original_preview is not None
    assert "this is not json" in r.original_preview


def test_repair_unterminated_escape():
    r = repair_truncated_json('{"x": "abc\\')
    assert r.fallback or json.loads(r.repaired) is not None


def test_repair_unicode_escape_cut():
    r = repair_truncated_json('{"x": "\\u12')
    assert r.fallback is True


def test_repair_array_dangling_comma():
    r = repair_truncated_json('{"x": [1,')
    assert r.changed is True
    assert r.fallback is False
    parsed = json.loads(r.repaired)
    assert parsed == {"x": [1]}


def test_repair_preview_only_when_changed():
    r = repair_truncated_json('{"a": 1}')
    assert r.original_preview is None
    r2 = repair_truncated_json('{"a": 1,')
    assert r2.original_preview is not None


def test_repair_preview_size_bounded():
    big = '{"k": "' + "x" * 5000
    r = repair_truncated_json(big)
    if r.original_preview is not None:
        assert len(r.original_preview) <= 200


def test_scavenge_empty_inputs():
    r = scavenge_tool_calls(None, None, ALLOWED)
    assert r.calls == []
    r = scavenge_tool_calls("", "", ALLOWED)
    assert r.calls == []


def test_scavenge_simple_json_shape():
    content = '{"name": "read_file", "arguments": {"file_path": "x.txt"}}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "read_file"
    assert r.calls[0].arguments == {"file_path": "x.txt"}


def test_scavenge_openai_function_shape():
    content = (
        '{"type": "function", "function": '
        '{"name": "list_files", "arguments": {"path": "."}}}'
    )
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "list_files"
    assert r.calls[0].arguments == {"path": "."}


def test_scavenge_tool_name_shape():
    content = '{"tool_name": "read_file", "tool_args": {"file_path": "x.txt"}}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "read_file"


def test_scavenge_swival_call_block():
    content = (
        '<swival:call id="abc" name="read_file">{"file_path": "x.txt"}</swival:call>'
    )
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "read_file"


def test_scavenge_disallowed_name_skipped():
    content = '{"name": "delete_everything", "arguments": {}}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert r.calls == []


def test_scavenge_max_calls_cap():
    blobs = " ".join(
        json.dumps({"name": "read_file", "arguments": {"file_path": f"f{i}.txt"}})
        for i in range(10)
    )
    r = scavenge_tool_calls(blobs, None, ALLOWED, max_calls=3)
    assert len(r.calls) == 3


def test_scavenge_oversized_input():
    big = "{" + "x" * (100 * 1024 + 100)
    r = scavenge_tool_calls(big, None, ALLOWED)
    assert r.calls == []
    assert any("too large" in n for n in r.notes)


def test_scavenge_dedup_identical_args():
    one = json.dumps({"name": "read_file", "arguments": {"file_path": "x"}})
    r = scavenge_tool_calls(one + "\n" + one, None, ALLOWED)
    assert len(r.calls) == 1


def test_scavenge_arguments_as_json_string():
    content = json.dumps(
        {"name": "read_file", "arguments": json.dumps({"file_path": "x"})}
    )
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].arguments == {"file_path": "x"}


def test_scavenge_null_args():
    content = '{"name": "list_files", "arguments": null}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].arguments == {}


def test_scavenge_scalar_args_rejected():
    content = '{"name": "read_file", "arguments": 42}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert r.calls == []


def test_scavenge_array_args_rejected():
    content = '{"name": "read_file", "arguments": [1, 2]}'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert r.calls == []


def test_scavenge_reasoning_channel():
    reasoning = '{"name": "read_file", "arguments": {"file_path": "x"}}'
    r = scavenge_tool_calls(None, reasoning, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].source.startswith("reasoning")


def test_content_is_pure_tool_call_true_for_bare_json():
    assert (
        content_is_pure_tool_call(
            '{"name": "read_file", "arguments": {"file_path": "x"}}'
        )
        is True
    )


def test_content_is_pure_tool_call_true_for_swival():
    assert (
        content_is_pure_tool_call(
            '<swival:call name="read_file">{"file_path":"x"}</swival:call>'
        )
        is True
    )


def test_content_is_pure_tool_call_false_for_prose():
    prose = (
        "The read_file tool takes a path: "
        '{"name": "read_file", "arguments": {"file_path": "x"}}'
    )
    assert content_is_pure_tool_call(prose) is False


def test_content_is_pure_tool_call_false_for_empty():
    assert content_is_pure_tool_call("") is False
    assert content_is_pure_tool_call(None) is False
    assert content_is_pure_tool_call("   ") is False


def test_content_is_pure_tool_call_rejects_non_tool_json_when_names_given():
    mixed = (
        '{"answer": "here is an example"}\n'
        '{"name": "read_file", "arguments": {"file_path": "secret.txt"}}'
    )
    assert content_is_pure_tool_call(mixed, ALLOWED) is False


def test_content_is_pure_tool_call_rejects_disallowed_name_when_names_given():
    only_disallowed = '{"name": "delete_everything", "arguments": {}}'
    assert content_is_pure_tool_call(only_disallowed, ALLOWED) is False


def test_content_is_pure_tool_call_accepts_pure_allowed_call():
    pure = '{"name": "read_file", "arguments": {"file_path": "x"}}'
    assert content_is_pure_tool_call(pure, ALLOWED) is True


def test_swival_call_name_regex_rejects_data_name_attribute():
    content = '<swival:call data-name="read_file">{"file_path": "x.txt"}</swival:call>'
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert r.calls == []


def test_swival_call_name_regex_accepts_real_name_with_other_attrs():
    content = (
        '<swival:call data-extra="ignore" name="read_file">'
        '{"file_path": "x.txt"}</swival:call>'
    )
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "read_file"


def test_repair_preview_byte_capped_for_multibyte():
    big = '{"k": "' + "α" * 5000
    r = repair_truncated_json(big)
    if r.original_preview is not None:
        assert len(r.original_preview.encode("utf-8")) <= 200


def test_repair_does_not_inject_null_inside_unterminated_string():
    inp = '{"old_string": "x\\":'
    r = repair_truncated_json(inp)
    assert r.fallback is False
    parsed = json.loads(r.repaired)
    assert "null" not in parsed["old_string"]
    assert parsed["old_string"].endswith(":")


def test_scavenge_recovers_after_earlier_unbalanced_brace():
    content = (
        "Here is an example: { missing close brace later. "
        '{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
    )
    r = scavenge_tool_calls(content, None, ALLOWED)
    assert len(r.calls) == 1
    assert r.calls[0].name == "read_file"


def test_nest_arguments_collision_raises():
    from swival.tool_call_repair import NestCollisionError

    meta = FlattenMeta(original_schema={}, paths={"outer.inner": ["outer", "inner"]})
    with pytest.raises(NestCollisionError):
        nest_arguments({"outer": "whole", "outer.inner": 5}, meta)


def test_analyze_schema_refuses_when_outer_required_loss():
    schema = {
        "type": "object",
        "properties": {
            "auth": {
                "type": "object",
                "properties": {f"f{i}": {"type": "string"} for i in range(11)},
            }
        },
        "required": ["auth"],
    }
    d = analyze_schema(schema)
    assert d.should_flatten is False
    assert d.unsafe_reason == "required-loss"


def test_flatten_schema_deepcopies_leaves():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"field": {"type": "string", "description": "original"}},
            }
        },
    }
    flat, _ = flatten_schema(schema)
    flat["properties"]["outer.field"]["description"] = "mutated"
    assert (
        schema["properties"]["outer"]["properties"]["field"]["description"]
        == "original"
    )


def test_analyze_schema_treats_array_of_primitives_as_leaf():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            }
        },
    }
    d = analyze_schema(schema)
    assert d.max_depth == 2
    assert d.should_flatten is False


def test_storm_breaker_three_identical_reads():
    sb = StormBreaker()
    assert (
        sb.inspect("read_file", '{"file_path": "x"}', mutating=False).suppress is False
    )
    assert (
        sb.inspect("read_file", '{"file_path": "x"}', mutating=False).suppress is False
    )
    v = sb.inspect("read_file", '{"file_path": "x"}', mutating=False)
    assert v.suppress is True
    assert "repeat-loop guard tripped" in v.reason


def test_storm_breaker_suppresses_second_explicit_non_idempotent_call():
    sb = StormBreaker()
    args = '{"key": "once", "value": {"b": 2, "a": 1}}'
    first = sb.inspect(
        "mcp__soak__record_once",
        args,
        mutating=False,
        suppress_on_repeat=True,
    )
    second = sb.inspect(
        "mcp__soak__record_once",
        '{"value":{"a":1,"b":2},"key":"once"}',
        mutating=False,
        suppress_on_repeat=True,
    )
    assert first.suppress is False
    assert second.suppress is True
    assert second.count == 2


def test_storm_breaker_edit_clears_read_only():
    sb = StormBreaker()
    sb.inspect("read_file", '{"file_path": "x"}', mutating=False)
    sb.inspect("read_file", '{"file_path": "x"}', mutating=False)
    sb.inspect("edit_file", '{"file_path": "x"}', mutating=True)
    v = sb.inspect("read_file", '{"file_path": "x"}', mutating=False)
    assert v.suppress is False


def test_storm_breaker_three_identical_edits_suppressed():
    sb = StormBreaker()
    args = '{"file_path": "x", "old": "a", "new": "b"}'
    assert sb.inspect("edit_file", args, mutating=True).suppress is False
    assert sb.inspect("edit_file", args, mutating=True).suppress is False
    v = sb.inspect("edit_file", args, mutating=True)
    assert v.suppress is True


def test_storm_breaker_exempt_think():
    sb = StormBreaker()
    args = '{"step": "1", "thought": "thinking"}'
    for _ in range(5):
        v = sb.inspect("think", args, mutating=False)
        assert v.suppress is False


def test_storm_breaker_window_ages_out():
    sb = StormBreaker(window=6, threshold=3)
    sb.inspect("read_file", '{"file_path": "first"}', mutating=False)
    for i in range(6):
        sb.inspect("read_file", f'{{"file_path": "fill{i}"}}', mutating=False)
    v = sb.inspect("read_file", '{"file_path": "first"}', mutating=False)
    assert v.suppress is False


def test_storm_breaker_key_order_insensitive():
    sb = StormBreaker()
    sb.inspect("read_file", '{"a": 1, "b": 2}', mutating=False)
    sb.inspect("read_file", '{"b": 2, "a": 1}', mutating=False)
    v = sb.inspect("read_file", '{"a":1,"b":2}', mutating=False)
    assert v.suppress is True


def test_storm_breaker_whitespace_insensitive():
    sb = StormBreaker()
    sb.inspect("read_file", '{"a": 1}', mutating=False)
    sb.inspect("read_file", '{"a":1}', mutating=False)
    v = sb.inspect("read_file", '{ "a" :  1 }', mutating=False)
    assert v.suppress is True


def test_storm_breaker_different_args_not_suppressed():
    sb = StormBreaker()
    for i in range(5):
        v = sb.inspect("read_file", f'{{"file_path": "f{i}"}}', mutating=False)
        assert v.suppress is False


def test_storm_breaker_reset():
    sb = StormBreaker()
    sb.inspect("read_file", '{"a": 1}', mutating=False)
    sb.inspect("read_file", '{"a": 1}', mutating=False)
    sb.reset()
    v = sb.inspect("read_file", '{"a": 1}', mutating=False)
    assert v.suppress is False


def test_storm_breaker_broken_json_still_works():
    sb = StormBreaker()
    sb.inspect("read_file", "not json", mutating=False)
    sb.inspect("read_file", "not json", mutating=False)
    v = sb.inspect("read_file", "not json", mutating=False)
    assert v.suppress is True


def test_is_mutating_helper():
    assert is_mutating("write_file") is True
    assert is_mutating("edit_file") is True
    assert is_mutating("delete_file") is True
    assert is_mutating("run_command") is True
    assert is_mutating("run_shell_command") is True
    assert is_mutating("spawn_subagent") is True
    assert is_mutating("read_file") is False
    assert is_mutating("list_files") is False


def test_is_storm_exempt_helper():
    assert is_storm_exempt("think") is True
    assert is_storm_exempt("todo") is True
    assert is_storm_exempt("snapshot") is True
    assert is_storm_exempt("complete_goal") is True
    assert is_storm_exempt("check_subagents") is True
    assert is_storm_exempt("read_file") is False


def _make_schema(props, required=None):
    s = {"type": "object", "properties": props}
    if required is not None:
        s["required"] = required
    return s


def test_analyze_schema_simple_flat_under_threshold():
    s = _make_schema({"a": {"type": "string"}, "b": {"type": "integer"}})
    d = analyze_schema(s)
    assert d.should_flatten is False
    assert d.leaf_count == 2


def test_analyze_schema_eleven_leaves_flattens():
    props = {f"f{i}": {"type": "string"} for i in range(11)}
    d = analyze_schema(_make_schema(props))
    assert d.should_flatten is True
    assert d.leaf_count == 11


def test_analyze_schema_depth_three_flattens():
    s = _make_schema(
        {"a": _make_schema({"b": _make_schema({"c": {"type": "string"}})})}
    )
    d = analyze_schema(s)
    assert d.should_flatten is True
    assert d.max_depth >= 3


def test_analyze_schema_skips_ref():
    s = {"type": "object", "properties": {"a": {"$ref": "#/defs/x"}}}
    d = analyze_schema(s)
    assert d.should_flatten is False
    assert d.unsafe_reason == "ref"


def test_analyze_schema_skips_oneof():
    s = {"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}]}}}
    d = analyze_schema(s)
    assert d.unsafe_reason == "composition-keyword"


def test_analyze_schema_skips_open_additional_properties():
    s = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": True,
    }
    d = analyze_schema(s)
    assert d.unsafe_reason == "open-properties"


def test_analyze_schema_skips_dotted_property_names():
    s = _make_schema({"a.b": {"type": "string"}})
    d = analyze_schema(s)
    assert d.unsafe_reason == "dotted-property-name"


def test_analyze_schema_skips_array_of_objects():
    s = _make_schema(
        {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        }
    )
    d = analyze_schema(s)
    assert d.unsafe_reason == "array-of-objects"


def test_analyze_schema_skips_nullable_union():
    s = _make_schema({"a": {"type": ["string", "null"]}})
    d = analyze_schema(s)
    assert d.unsafe_reason == "nullable-union"


def test_flatten_schema_round_trip():
    schema = _make_schema(
        {
            "outer": _make_schema(
                {"inner": {"type": "string"}},
                required=["inner"],
            )
        },
        required=["outer"],
    )
    flat, meta = flatten_schema(schema)
    assert "outer.inner" in flat["properties"]
    assert flat["required"] == ["outer.inner"]
    original_args = {"outer": {"inner": "hello"}}
    flat_args = {"outer.inner": "hello"}
    assert nest_arguments(flat_args, meta) == original_args


def test_flatten_schema_required_promotion_full_chain():
    schema = _make_schema(
        {"a": _make_schema({"b": {"type": "string"}}, required=["b"])},
        required=["a"],
    )
    flat, _ = flatten_schema(schema)
    assert flat["required"] == ["a.b"]


def test_flatten_schema_optional_parent_does_not_promote():
    schema = _make_schema(
        {"a": _make_schema({"b": {"type": "string"}}, required=["b"])},
        required=[],
    )
    flat, _ = flatten_schema(schema)
    assert flat.get("required", []) == []


def test_flatten_schema_preserves_leaf_metadata():
    schema = _make_schema(
        {
            "outer": _make_schema(
                {"name": {"type": "string", "description": "the name", "default": "x"}}
            )
        }
    )
    flat, _ = flatten_schema(schema)
    leaf = flat["properties"]["outer.name"]
    assert leaf["description"] == "the name"
    assert leaf["default"] == "x"


def test_nest_arguments_multiple_keys():
    meta = FlattenMeta(
        original_schema={},
        paths={"a.b": ["a", "b"], "a.c": ["a", "c"], "d": ["d"]},
    )
    out = nest_arguments({"a.b": 1, "a.c": 2, "d": 3}, meta)
    assert out == {"a": {"b": 1, "c": 2}, "d": 3}


def test_nest_arguments_missing_path_falls_back_to_split():
    meta = FlattenMeta(original_schema={}, paths={})
    out = nest_arguments({"x.y.z": 5}, meta)
    assert out == {"x": {"y": {"z": 5}}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
