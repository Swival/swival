"""Tests for swival.traces — JSONL trace export."""

import json
import os
import types

import pytest

from swival.traces import (
    write_trace,
    write_trace_to_dir,
    _sanitize_trace_text,
    _sanitize_trace_obj,
)


def _read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Unit tests: conversion correctness ---


def test_tool_result_role(tmp_path):
    """role: "tool" messages become type: "user" with tool_result content."""
    messages = [
        {"role": "user", "content": "do something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "file contents here"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    tool_line = [
        ln for ln in lines if ln.get("type") == "user" and "toolUseResult" in ln
    ]
    assert len(tool_line) == 1
    msg = tool_line[0]["message"]
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "tool_result"
    assert msg["content"][0]["tool_use_id"] == "tc1"
    assert msg["content"][0]["content"] == "file contents here"
    assert msg["content"][0]["is_error"] is False


def test_tool_result_error(tmp_path):
    """Tool results starting with 'error:' have is_error=True."""
    messages = [
        {"role": "tool", "tool_call_id": "tc2", "content": "error: file not found"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    assert lines[0]["message"]["content"][0]["is_error"] is True


def test_tool_call_translation(tmp_path):
    """OpenAI tool_calls become Anthropic tool_use blocks."""
    messages = [
        {
            "role": "assistant",
            "content": "thinking...",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "a.py", "content": "x=1"}',
                    },
                }
            ],
        },
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    msg = lines[0]["message"]
    content = msg["content"]
    text_block = [b for b in content if b["type"] == "text"]
    tool_block = [b for b in content if b["type"] == "tool_use"]
    assert len(text_block) == 1
    assert text_block[0]["text"] == "thinking..."
    assert len(tool_block) == 1
    assert tool_block[0]["id"] == "tc1"
    assert tool_block[0]["name"] == "write_file"
    assert tool_block[0]["input"] == {"path": "a.py", "content": "x=1"}
    assert msg["stop_reason"] == "tool_use"


def test_hf_detection_fields(tmp_path):
    """message is emitted as a JSON object so HuggingFace's mixed-struct
    detection can infer it as datasets.Json() and trigger agent-traces mode."""
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    for line in lines:
        assert isinstance(line["type"], str)
        assert line["sessionId"] == "s1"
        if line.get("harness"):
            assert line["harness"] == "swival"
        if line["type"] in ("user", "assistant"):
            assert "message" in line
            assert isinstance(line["message"], dict)


def test_system_only_trace_not_written(tmp_path):
    """A trace with only a system message is not written — it would cause HuggingFace
    schema inference failures (message/toolUseResult absent → typed as null)."""
    path = str(tmp_path / "trace.jsonl")
    write_trace(
        [{"role": "system", "content": "sys"}],
        path=path,
        session_id="s1",
        base_dir="/tmp",
        model="m1",
    )
    assert not (tmp_path / "trace.jsonl").exists()


def test_namespace_messages(tmp_path):
    """Namespace objects (litellm responses) are handled correctly."""
    msg = types.SimpleNamespace(
        role="assistant",
        content="response text",
        tool_calls=None,
        id="msg-123",
        usage=types.SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            input_tokens=None,
            output_tokens=None,
        ),
    )
    messages = [msg]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    assert len(lines) == 2
    assert lines[0]["type"] == "assistant"
    msg = lines[0]["message"]
    assert msg["content"][0]["text"] == "response text"
    assert msg["stop_reason"] == "end_turn"
    assert msg["id"] == "msg-123"
    usage = msg["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20


def test_system_message(tmp_path):
    """role: "system" becomes type: "system" with content and level."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    sys_line = next(ln for ln in lines if ln["type"] == "system")
    assert sys_line["content"] == "You are a helpful assistant."
    assert sys_line["level"] == "info"
    assert sys_line["isMeta"] is False


def test_malformed_tool_arguments(tmp_path):
    """Unparseable function.arguments become {"_raw": ...} instead of crashing."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "bad_tool", "arguments": "not json at all"},
                }
            ],
        }
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    tool_block = lines[0]["message"]["content"][0]
    assert tool_block["type"] == "tool_use"
    assert tool_block["input"] == {"_raw": "not json at all"}


def test_empty_conversation(tmp_path):
    """Empty messages list produces no file."""
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages=[], path=path, session_id="s1", base_dir="/tmp", model="m1")

    assert not (tmp_path / "trace.jsonl").exists()


def test_last_prompt_line(tmp_path):
    """When task is provided, a last-prompt line is appended."""
    messages = [{"role": "user", "content": "hello"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(
        messages,
        path=path,
        session_id="s1",
        base_dir="/tmp",
        model="m1",
        task="hello",
    )

    lines = _read_lines(path)
    last = lines[-1]
    assert last["type"] == "last-prompt"
    assert last["lastPrompt"] == "hello"
    assert last["sessionId"] == "s1"
    assert last["toolUseResult"] == ""


def test_no_last_prompt_without_task(tmp_path):
    """When task is None, a last-prompt line with empty string is still written."""
    messages = [{"role": "user", "content": "hello"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    last = lines[-1]
    assert last["type"] == "last-prompt"
    assert last["lastPrompt"] == ""
    assert last["sessionId"] == "s1"
    assert last["toolUseResult"] == ""


def test_parent_uuid_chain(tmp_path):
    """UUIDs form a valid linked list: first has null parent, rest chain."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    msg_lines = [ln for ln in lines if "uuid" in ln]
    assert msg_lines[0]["parentUuid"] is None
    for i in range(1, len(msg_lines)):
        assert msg_lines[i]["parentUuid"] == msg_lines[i - 1]["uuid"]

    uuids = [ln["uuid"] for ln in msg_lines]
    assert len(set(uuids)) == len(uuids)


def test_assistant_text_only(tmp_path):
    """Assistant message without tool_calls has stop_reason end_turn."""
    messages = [{"role": "assistant", "content": "just text"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir="/tmp", model="m1")

    lines = _read_lines(path)
    msg = lines[0]["message"]
    assert msg["stop_reason"] == "end_turn"
    assert msg["content"] == [{"type": "text", "text": "just text"}]


# --- Integration tests ---


def _make_message(content=None, tool_calls=None):
    return types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant", id=None, usage=None
    )


class TestSessionTraces:
    def test_session_run_writes_trace(self, tmp_path, monkeypatch):
        """Session.run() writes a JSONL trace file when trace_dir is set."""
        from swival import Session, agent

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            return _make_message(content="the answer"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        trace_dir = str(tmp_path / "traces")
        s = Session(base_dir=str(tmp_path), history=False, trace_dir=trace_dir)
        s.run("hello")

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1
        lines = _read_lines(str(files[0]))
        assert any(ln["type"] == "user" for ln in lines)
        assert any(ln["type"] == "assistant" for ln in lines)
        assert all(ln["harness"] == "swival" for ln in lines if ln.get("harness"))

    def test_session_run_isolation(self, tmp_path, monkeypatch):
        """Two run() calls produce two separate JSONL files with different session ids."""
        from swival import Session, agent

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            return _make_message(content="answer"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        trace_dir = str(tmp_path / "traces")
        s = Session(base_dir=str(tmp_path), history=False, trace_dir=trace_dir)
        s.run("first")
        s.run("second")

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 2

        ids = set()
        for f in files:
            lines = _read_lines(str(f))
            ids.add(lines[0]["sessionId"])
        assert len(ids) == 2

    def test_session_ask_accumulation(self, tmp_path, monkeypatch):
        """Multiple ask() calls write to one file with the same session id."""
        from swival import Session, agent

        call_count = [0]

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            call_count[0] += 1
            return _make_message(content=f"answer {call_count[0]}"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        trace_dir = str(tmp_path / "traces")
        s = Session(base_dir=str(tmp_path), history=False, trace_dir=trace_dir)
        s.ask("first")
        s.ask("second")

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1

        lines = _read_lines(str(files[0]))
        session_ids = {ln["sessionId"] for ln in lines if "sessionId" in ln}
        assert len(session_ids) == 1

        user_msgs = [ln for ln in lines if ln["type"] == "user" and "message" in ln]
        assert len(user_msgs) >= 2

    def test_session_ask_rollback_excludes_failed(self, tmp_path, monkeypatch):
        """After a failed ask(), the trace reflects only successful turns."""
        from swival import Session, agent
        from swival.report import AgentError

        call_count = [0]

        def flaky_llm(base_url, model_id, messages, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_message(content="first answer"), "stop"
            raise AgentError("boom")

        monkeypatch.setattr(agent, "call_llm", flaky_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        trace_dir = str(tmp_path / "traces")
        s = Session(base_dir=str(tmp_path), history=False, trace_dir=trace_dir)
        s.ask("first")

        with pytest.raises(AgentError):
            s.ask("second")

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1

        lines = _read_lines(str(files[0]))
        user_prompts = [
            ln
            for ln in lines
            if ln["type"] == "user"
            and "message" in ln
            and isinstance(ln["message"].get("content"), str)
        ]
        assert len(user_prompts) == 1
        assert user_prompts[0]["message"]["content"] == "first"

    def test_no_trace_without_trace_dir(self, tmp_path, monkeypatch):
        """No trace file when trace_dir is not set."""
        from swival import Session, agent

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            return _make_message(content="answer"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("hello")

        assert not list(tmp_path.rglob("*.jsonl"))

    def test_config_propagation(self, tmp_path):
        """trace_dir reaches Session via config plumbing."""
        from swival.config import args_to_session_kwargs

        args = types.SimpleNamespace(
            provider="lmstudio",
            model=None,
            api_key=None,
            base_url=None,
            max_turns=10,
            max_output_tokens=1024,
            max_context_tokens=None,
            temperature=None,
            top_p=None,
            seed=None,
            files="some",
            yolo=False,
            commands="all",
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_skills=False,
            sandbox="builtin",
            sandbox_session=None,
            sandbox_strict_read=False,
            memory_full=False,
            config_dir=None,
            proactive_summaries=False,
            extra_body=None,
            reasoning_effort=None,
            sanitize_thinking=False,
            prompt_cache=True,
            cache=False,
            cache_dir=None,
            retries=5,
            llm_filter=None,
            encrypt_secrets=False,
            no_encrypt_secrets=False,
            encrypt_secrets_key=None,
            encrypt_secrets_tweak=None,
            encrypt_secrets_patterns=None,
            lifecycle_command=None,
            lifecycle_timeout=300,
            lifecycle_fail_closed=False,
            trace_dir="/tmp/my-traces",
            add_dir=[],
            add_dir_ro=[],
            no_read_guard=False,
            no_history=False,
            no_memory=False,
            no_continue=False,
            quiet=False,
            no_sandbox_auto_session=False,
            no_lifecycle=False,
            subagents=False,
            no_subagents=False,
            skills_dir=None,
        )
        kwargs = args_to_session_kwargs(args, "/tmp")
        assert kwargs["trace_dir"] == "/tmp/my-traces"


def test_cli_repl_path_has_trace_write():
    """The REPL branch in _run_main writes traces in its finally block.

    This is a source-level assertion: inspect _run_main to confirm that the
    REPL path's outer finally calls _write_trace, matching the one-shot path.
    A full integration test through _run_main is impractical because it requires
    dozens of internal args attributes and monkeypatches.
    """
    import inspect
    from swival.agent import _run_main

    source = inspect.getsource(_run_main)

    repl_idx = source.index("# REPL path")
    repl_source = source[repl_idx:]

    assert "_write_trace(messages)" in repl_source, (
        "_write_trace(messages) not found in the REPL branch of _run_main"
    )


# --- Secret encryption tests ---

_FAKE_TOKEN = "ghp_" + "A" * 36  # matches the github-pat pattern


def _all_text(lines):
    """Flatten all string values from a list of parsed JSONL dicts."""
    out = []

    def _collect(obj):
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    for ln in lines:
        _collect(ln)
    return out


def test_trace_no_plaintext_secret_ephemeral(tmp_path):
    """write_trace_to_dir with no shield creates an ephemeral one; token must not appear."""
    messages = [
        {"role": "user", "content": f"my token is {_FAKE_TOKEN} please use it"},
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir="/tmp",
        model="m",
        task=f"use token {_FAKE_TOKEN}",
    )
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = _read_lines(str(files[0]))
    texts = _all_text(lines)
    assert not any(_FAKE_TOKEN in t for t in texts)


def test_trace_persistent_shield_stable_ciphertext(tmp_path):
    """The same SecretShield produces the same ciphertext for the same token."""
    from swival.secrets import SecretShield

    messages = [{"role": "user", "content": f"token={_FAKE_TOKEN}"}]
    shield = SecretShield()
    try:
        write_trace_to_dir(
            messages,
            trace_dir=str(tmp_path),
            base_dir="/tmp",
            model="m",
            session_id="s1",
            secret_shield=shield,
        )
        write_trace_to_dir(
            messages,
            trace_dir=str(tmp_path),
            base_dir="/tmp",
            model="m",
            session_id="s2",
            secret_shield=shield,
        )
    finally:
        shield.destroy()

    lines1 = _read_lines(str(tmp_path / "s1.jsonl"))
    lines2 = _read_lines(str(tmp_path / "s2.jsonl"))
    texts1 = [t for t in _all_text(lines1) if "ghp_" in t]
    texts2 = [t for t in _all_text(lines2) if "ghp_" in t]
    assert texts1 and texts1 == texts2


def test_trace_tool_use_input_encrypted(tmp_path):
    """Secrets inside tool-call arguments are encrypted; input is still a dict."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "call_api",
                        "arguments": json.dumps(
                            {"key": _FAKE_TOKEN, "url": "https://example.com"}
                        ),
                    },
                }
            ],
        }
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir="/tmp",
        model="m",
    )
    files = list(tmp_path.glob("*.jsonl"))
    lines = _read_lines(str(files[0]))
    assistant_lines = [ln for ln in lines if ln.get("type") == "assistant"]
    assert assistant_lines
    msg = assistant_lines[0]["message"]
    tool_use_blocks = [b for b in msg["content"] if b["type"] == "tool_use"]
    assert tool_use_blocks
    inp = tool_use_blocks[0]["input"]
    assert isinstance(inp, dict)
    assert _FAKE_TOKEN not in json.dumps(inp)


def test_trace_tool_result_encrypted(tmp_path):
    """Secrets in tool results are encrypted in both toolUseResult and tool_result.content."""
    messages = [
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": f"api returned {_FAKE_TOKEN}",
        },
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir="/tmp",
        model="m",
    )
    files = list(tmp_path.glob("*.jsonl"))
    lines = _read_lines(str(files[0]))
    tool_lines = [ln for ln in lines if "toolUseResult" in ln]
    assert tool_lines
    ln = tool_lines[0]
    assert _FAKE_TOKEN not in ln["toolUseResult"]
    assert _FAKE_TOKEN not in ln["message"]["content"][0]["content"]


# --- Path redaction tests ---

_HOME = os.path.expanduser("~")
_BASE = _HOME + "/test-workspace-redaction"


def test_sanitize_text_base_dir_exact():
    assert _sanitize_trace_text(_BASE, base_dir=_BASE, home_dir=_HOME) == "BASEDIR"


def test_sanitize_text_base_dir_prefix():
    assert (
        _sanitize_trace_text(f"{_BASE}/foo.py", base_dir=_BASE, home_dir=_HOME)
        == "BASEDIR/foo.py"
    )


def test_sanitize_text_base_dir_in_prose():
    assert (
        _sanitize_trace_text(f"open {_BASE}/file.txt", base_dir=_BASE, home_dir=_HOME)
        == "open BASEDIR/file.txt"
    )


def test_sanitize_text_base_dir_after_equals():
    assert (
        _sanitize_trace_text(f"path={_BASE}/cfg.toml", base_dir=_BASE, home_dir=_HOME)
        == "path=BASEDIR/cfg.toml"
    )


def test_sanitize_text_home_dir_outside_base():
    result = _sanitize_trace_text(
        f"{_HOME}/.config/swival/config.toml", base_dir=_BASE, home_dir=_HOME
    )
    assert result == "~/.config/swival/config.toml"


def test_sanitize_text_base_dir_preferred_over_home():
    result = _sanitize_trace_text(
        f"{_BASE}/src/main.py", base_dir=_BASE, home_dir=_HOME
    )
    assert result == "BASEDIR/src/main.py"
    assert "~" not in result


def test_sanitize_text_no_false_positive_suffix():
    assert (
        _sanitize_trace_text("/tmpfile", base_dir="/tmp", home_dir="/tmp/home")
        == "/tmpfile"
    )


def test_sanitize_text_no_false_positive_equals():
    assert (
        _sanitize_trace_text("cache=/tmpdir", base_dir="/tmp", home_dir="/tmp/home")
        == "cache=/tmpdir"
    )


def test_sanitize_text_unrelated_path_unchanged():
    assert (
        _sanitize_trace_text("/var/log/syslog", base_dir=_BASE, home_dir=_HOME)
        == "/var/log/syslog"
    )


def test_sanitize_text_home_lookalike_unchanged():
    result = _sanitize_trace_text(f"{_HOME}x/other", base_dir=_BASE, home_dir=_HOME)
    assert result == f"{_HOME}x/other"


def test_sanitize_text_backtick_wrapped():
    result = _sanitize_trace_text(f"`{_BASE}`", base_dir=_BASE, home_dir=_HOME)
    assert result == "`BASEDIR`"


def test_sanitize_text_quote_wrapped():
    result = _sanitize_trace_text(f'"{_BASE}"', base_dir=_BASE, home_dir=_HOME)
    assert result == '"BASEDIR"'


def test_sanitize_text_paren_wrapped():
    result = _sanitize_trace_text(f"({_BASE})", base_dir=_BASE, home_dir=_HOME)
    assert result == "(BASEDIR)"


def test_sanitize_text_url_embedded_unchanged():
    # path embedded inside a URL token — 'm' from '.com' is alphanumeric, so the
    # left-boundary check must block the match to avoid mangling the URL.
    url = f"https://example.com{_BASE}/x"
    result = _sanitize_trace_text(url, base_dir=_BASE, home_dir=_HOME)
    assert result == url


def test_sanitize_text_exact_path_home_preferred_after_base_fix():
    # Regression: `` `/base/path` `` must become `` `BASEDIR` `` not `` `~/suffix` ``
    result = _sanitize_trace_text(f"`{_BASE}`", base_dir=_BASE, home_dir=_HOME)
    assert "BASEDIR" in result
    assert _HOME.lstrip("/").split("/")[0] not in result.replace("BASEDIR", "")


def test_sanitize_text_dot_sentence_terminator():
    # Sentence-ending dot after exact base_dir must be treated as punctuation.
    assert (
        _sanitize_trace_text(f"{_BASE}.", base_dir=_BASE, home_dir=_HOME) == "BASEDIR."
    )


def test_sanitize_text_dot_sentence_in_prose():
    result = _sanitize_trace_text(f"open {_BASE}.", base_dir=_BASE, home_dir=_HOME)
    assert result == "open BASEDIR."


def test_sanitize_text_dot_sentence_mid_prose():
    result = _sanitize_trace_text(
        f"wrote to {_BASE}. Done.", base_dir=_BASE, home_dir=_HOME
    )
    assert result == "wrote to BASEDIR. Done."


def test_sanitize_text_dot_sibling_path_not_basedir():
    # /base.git is a sibling directory — the base_dir pass must not rewrite it
    # (the .git extension shows it is a different path component), but the home_dir
    # pass still redacts the home prefix as expected.
    sibling = f"{_BASE}.git"
    result = _sanitize_trace_text(sibling, base_dir=_BASE, home_dir=_HOME)
    assert "BASEDIR" not in result
    assert result == "~/test-workspace-redaction.git"


def test_sanitize_obj_recursive():
    obj = {"path": f"{_BASE}/file.txt", "nested": [f"{_BASE}/sub", 42, None]}
    result = _sanitize_trace_obj(obj, base_dir=_BASE, home_dir=_HOME)
    assert result == {"path": "BASEDIR/file.txt", "nested": ["BASEDIR/sub", 42, None]}


def test_sanitize_obj_dict_keys_unchanged():
    key = f"{_BASE}/key"
    obj = {key: "value"}
    result = _sanitize_trace_obj(obj, base_dir=_BASE, home_dir=_HOME)
    assert key in result


def test_sanitize_obj_non_strings_unchanged():
    obj = {"count": 42, "flag": True, "nothing": None}
    result = _sanitize_trace_obj(obj, base_dir=_BASE, home_dir=_HOME)
    assert result == {"count": 42, "flag": True, "nothing": None}


def test_write_trace_cwd_is_basedir(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    for line in lines:
        if "cwd" in line:
            assert line["cwd"] == "BASEDIR"


def test_write_trace_user_content_redacted(tmp_path):
    messages = [{"role": "user", "content": f"check {_BASE}/foo.py"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    user_lines = [
        ln
        for ln in lines
        if ln.get("type") == "user" and "message" in ln and "toolUseResult" not in ln
    ]
    assert user_lines[0]["message"]["content"] == "check BASEDIR/foo.py"


def test_write_trace_assistant_text_redacted(tmp_path):
    messages = [{"role": "assistant", "content": f"wrote {_BASE}/out.txt"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    assistant_lines = [ln for ln in lines if ln.get("type") == "assistant"]
    text_block = [
        b for b in assistant_lines[0]["message"]["content"] if b["type"] == "text"
    ]
    assert text_block[0]["text"] == "wrote BASEDIR/out.txt"


def test_write_trace_tool_args_redacted(tmp_path):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": f"{_BASE}/src/main.py"}),
                    },
                }
            ],
        }
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    tool_block = [b for b in lines[0]["message"]["content"] if b["type"] == "tool_use"][
        0
    ]
    assert tool_block["input"]["file_path"] == "BASEDIR/src/main.py"


def test_write_trace_tool_result_redacted(tmp_path):
    messages = [
        {"role": "tool", "tool_call_id": "tc1", "content": f"read {_BASE}/readme.md"},
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    tool_lines = [ln for ln in lines if "toolUseResult" in ln]
    ln = tool_lines[0]
    assert ln["toolUseResult"] == "read BASEDIR/readme.md"
    assert ln["message"]["content"][0]["content"] == "read BASEDIR/readme.md"


def test_write_trace_home_path_outside_base_redacted(tmp_path):
    messages = [{"role": "user", "content": f"{_HOME}/.config/swival/config.toml"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(messages, path=path, session_id="s1", base_dir=_BASE, model="m1")
    lines = _read_lines(path)
    user_lines = [
        ln
        for ln in lines
        if ln.get("type") == "user" and "message" in ln and "toolUseResult" not in ln
    ]
    assert user_lines[0]["message"]["content"] == "~/.config/swival/config.toml"


def test_write_trace_last_prompt_redacted(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    path = str(tmp_path / "trace.jsonl")
    write_trace(
        messages,
        path=path,
        session_id="s1",
        base_dir=_BASE,
        model="m1",
        task=f"analyze {_BASE}/main.py",
    )
    lines = _read_lines(path)
    last = lines[-1]
    assert last["type"] == "last-prompt"
    assert last["lastPrompt"] == "analyze BASEDIR/main.py"


def test_write_trace_original_base_dir_absent(tmp_path):
    messages = [
        {"role": "user", "content": f"task in {_BASE}"},
        {"role": "assistant", "content": f"reading {_BASE}/file.py"},
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": f"contents from {_BASE}/file.py",
        },
    ]
    path = str(tmp_path / "trace.jsonl")
    write_trace(
        messages,
        path=path,
        session_id="s1",
        base_dir=_BASE,
        model="m1",
        task=f"work in {_BASE}",
    )
    raw = (tmp_path / "trace.jsonl").read_text()
    assert _BASE not in raw


def test_write_trace_to_dir_cwd_redacted(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir=_BASE,
        model="m",
        session_id="rd1",
    )
    lines = _read_lines(str(tmp_path / "rd1.jsonl"))
    for line in lines:
        if "cwd" in line:
            assert line["cwd"] == "BASEDIR"


def test_write_trace_to_dir_tool_input_redacted(tmp_path):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": f"{_BASE}/src.py"}),
                    },
                }
            ],
        }
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir=_BASE,
        model="m",
        session_id="rd2",
    )
    raw = (tmp_path / "rd2.jsonl").read_text()
    assert _BASE not in raw


def test_write_trace_to_dir_tool_result_redacted(tmp_path):
    messages = [
        {"role": "tool", "tool_call_id": "tc1", "content": f"file at {_BASE}/out.py"},
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir=_BASE,
        model="m",
        session_id="rd3",
    )
    raw = (tmp_path / "rd3.jsonl").read_text()
    assert _BASE not in raw


def test_write_trace_to_dir_last_prompt_redacted(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir=_BASE,
        model="m",
        session_id="rd4",
        task=f"run {_BASE}/script.py",
    )
    raw = (tmp_path / "rd4.jsonl").read_text()
    assert _BASE not in raw


def test_write_trace_to_dir_no_original_paths(tmp_path):
    messages = [
        {"role": "user", "content": f"run {_BASE}/main.py with config {_HOME}/.config"},
        {"role": "assistant", "content": f"Done, wrote to {_BASE}/out.txt"},
    ]
    write_trace_to_dir(
        messages,
        trace_dir=str(tmp_path),
        base_dir=_BASE,
        model="m",
        session_id="rd5",
        task=f"task at {_BASE}",
    )
    raw = (tmp_path / "rd5.jsonl").read_text()
    assert _BASE not in raw
    assert _HOME not in raw
