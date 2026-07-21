"""Tests for the python tool."""

import sys

import pytest

from swival.tools import PYTHON_TOOL, _run_python, dispatch, get_tool_schema


@pytest.fixture
def tmp_base(tmp_path):
    return str(tmp_path)


def test_schema_registered():
    assert PYTHON_TOOL["function"]["name"] == "run_python"
    schema = get_tool_schema("run_python")
    assert schema is not None
    assert "code" in schema["properties"]
    assert schema["required"] == ["code"]


def test_runs_print(tmp_base):
    result = _run_python("print('hello world')", tmp_base, timeout=5)
    assert "hello world" in result
    assert not result.startswith("error:")


def test_captures_stderr(tmp_base):
    result = _run_python(
        "import sys; print('out'); print('err', file=sys.stderr)",
        tmp_base,
        timeout=5,
    )
    assert "out" in result
    assert "err" in result


def test_nonzero_exit_on_exception(tmp_base):
    result = _run_python("raise ValueError('boom')", tmp_base, timeout=5)
    assert "Exit code:" in result
    assert "ValueError" in result
    assert "boom" in result


def test_syntax_error(tmp_base):
    result = _run_python("def foo(:\n  pass", tmp_base, timeout=5)
    assert "Exit code:" in result
    assert "SyntaxError" in result


def test_runs_in_base_dir(tmp_path):
    (tmp_path / "marker.txt").write_text("hi")
    result = _run_python(
        "import os; print(sorted(os.listdir('.')))",
        str(tmp_path),
        timeout=5,
    )
    assert "marker.txt" in result


def test_timeout(tmp_base):
    result = _run_python(
        "import time; time.sleep(5)",
        tmp_base,
        timeout=1,
    )
    assert "timed out" in result


def test_empty_code_rejected(tmp_base):
    result = _run_python("   ", tmp_base, timeout=5)
    assert result.startswith("error:")
    assert "non-empty" in result


def test_non_string_rejected(tmp_base):
    result = _run_python(123, tmp_base, timeout=5)  # type: ignore[arg-type]
    assert result.startswith("error:")


def test_dispatch_requires_unrestricted(tmp_base):
    result = dispatch(
        "run_python",
        {"code": "print(1)"},
        tmp_base,
        commands_unrestricted=False,
    )
    assert result.startswith("error:")
    assert "not available" in result


def test_dispatch_runs_when_unrestricted(tmp_base):
    result = dispatch(
        "run_python",
        {"code": "print('via dispatch')"},
        tmp_base,
        commands_unrestricted=True,
    )
    assert "via dispatch" in result
    assert not result.startswith("error:")


def test_dispatch_passes_timeout(tmp_base):
    result = dispatch(
        "run_python",
        {"code": "import time; time.sleep(5)", "timeout": 1},
        tmp_base,
        commands_unrestricted=True,
    )
    assert "timed out" in result


def test_dispatch_suggests_run_python_for_old_name(tmp_base):
    """OpenAI reserves the function name 'python', so the tool was renamed.

    Models trained on the old name should be pointed at run_python."""
    for old_name in ("python", "execute_python"):
        with pytest.raises(KeyError, match="run_python"):
            dispatch(old_name, {"code": "print(1)"}, tmp_base)


def test_dispatch_invalid_timeout(tmp_base):
    result = dispatch(
        "run_python",
        {"code": "print(1)", "timeout": "abc"},
        tmp_base,
        commands_unrestricted=True,
    )
    assert result.startswith("error:")
    assert "timeout" in result


def test_unicode_code(tmp_base):
    result = _run_python("print('héllo, 世界')", tmp_base, timeout=5)
    assert "héllo, 世界" in result


def test_no_shell_interpretation(tmp_base):
    """Backticks, $vars, and pipes in the code must reach the interpreter as-is."""
    code = "x = '$PATH | rm -rf /'; print(repr(x))"
    result = _run_python(code, tmp_base, timeout=5)
    assert "'$PATH | rm -rf /'" in result


def test_build_tools_exposes_python_when_enabled():
    from swival.agent import build_tools

    tools = build_tools(
        resolved_commands={},
        skills_catalog={},
        commands_unrestricted=True,
        shell_allowed=False,
        python_tool=True,
    )
    names = [t["function"]["name"] for t in tools]
    assert "run_python" in names


def test_build_tools_hides_python_when_gate_closed():
    """Unrestricted commands alone no longer expose the tool."""
    from swival.agent import build_tools

    tools = build_tools(
        resolved_commands={},
        skills_catalog={},
        commands_unrestricted=True,
        shell_allowed=False,
        python_tool=False,
    )
    names = [t["function"]["name"] for t in tools]
    assert "run_python" not in names


def test_build_tools_hides_python_when_restricted():
    """python_tool cannot smuggle the tool past the unrestricted-commands grant."""
    from swival.agent import build_tools

    tools = build_tools(
        resolved_commands={"ls": sys.executable},
        skills_catalog={},
        commands_unrestricted=False,
        shell_allowed=False,
        python_tool=True,
    )
    names = [t["function"]["name"] for t in tools]
    assert "run_python" not in names


def test_python_tool_available_detects_interpreter():
    from swival.tools import python_tool_available

    assert python_tool_available() is True


def test_python_tool_unavailable_when_no_interpreter(monkeypatch):
    import swival.tools as tools_mod

    monkeypatch.setattr(tools_mod, "_find_python_executable", lambda: None)
    assert tools_mod.python_tool_available() is False


def test_run_python_errors_without_interpreter(tmp_base, monkeypatch):
    import swival.tools as tools_mod

    monkeypatch.setattr(tools_mod, "_find_python_executable", lambda: None)
    result = _run_python("print(1)", tmp_base, timeout=5)
    assert result.startswith("error:")
    assert "interpreter" in result


def test_find_python_executable_falls_back_to_path(monkeypatch):
    import swival.tools as tools_mod

    monkeypatch.setattr(tools_mod.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        tools_mod.shutil,
        "which",
        lambda name: "/usr/bin/python3" if name == "python3" else None,
    )
    assert tools_mod._find_python_executable() == "/usr/bin/python3"
