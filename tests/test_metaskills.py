"""Tests for metaskills: discovery, tool exposure, execution, budgets, and errors."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from swival.skills import (
    SkillInfo,
    parse_frontmatter,
    discover_skills,
    format_skill_catalog,
    MAX_METASKILL_FILE_BYTES,
)
from swival.metaskills import (
    MetaskillBudget,
    MetaskillTrace,
    MetaskillError,
    run_metaskill,
    get_executable_metaskills,
    _normalize_return_value,
    _build_result_envelope,
    _check_starlark_available,
)
from swival.tools import RUN_METASKILL_TOOL


def _make_skill(
    parent: Path, name: str, description: str = "A test skill.", body: str = ""
):
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _make_metaskill(
    parent: Path,
    name: str,
    star_source: str,
    description: str = "A test metaskill.",
    *,
    metaskill_field: str | None = None,
    language: str | None = None,
):
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name}", f"description: {description}"]
    if metaskill_field:
        fm_lines.append(f"metaskill: {metaskill_field}")
    if language:
        fm_lines.append(f"metaskill_language: {language}")
    content = "---\n" + "\n".join(fm_lines) + "\n---\n\nInstructions here."
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    program_file = metaskill_field or "SKILL.star"
    (skill_dir / program_file).write_text(star_source, encoding="utf-8")
    return skill_dir


# =========================================================================
# Discovery
# =========================================================================


class TestDiscovery:
    def test_static_skill_unchanged(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "my-skill", "Does things.")
        catalog = discover_skills(str(tmp_path))
        assert "my-skill" in catalog
        skill = catalog["my-skill"]
        assert skill.metaskill_path is None
        assert skill.metaskill_language is None

    def test_metaskill_discovered_via_star_file(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "test-ms",
            'def run(input):\n    return {"status": "ok"}',
        )
        catalog = discover_skills(str(tmp_path))
        assert "test-ms" in catalog
        skill = catalog["test-ms"]
        assert skill.metaskill_path is not None
        assert skill.metaskill_path.name == "SKILL.star"
        assert skill.metaskill_language == "starlark"

    def test_metaskill_discovered_via_explicit_field(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "custom-ms",
            'def run(input):\n    return "hello"',
            metaskill_field="workflow.star",
        )
        catalog = discover_skills(str(tmp_path))
        skill = catalog["custom-ms"]
        assert skill.metaskill_path is not None
        assert skill.metaskill_path.name == "workflow.star"

    def test_metaskill_unknown_language_skipped(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "lua-ms",
            "-- lua code",
            language="lua",
        )
        catalog = discover_skills(str(tmp_path))
        skill = catalog["lua-ms"]
        assert skill.metaskill_path is None
        assert skill.metaskill_language is None

    def test_metaskill_file_too_large_skipped(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "big-ms",
            "x" * (MAX_METASKILL_FILE_BYTES + 1),
        )
        catalog = discover_skills(str(tmp_path))
        skill = catalog["big-ms"]
        assert skill.metaskill_path is None

    def test_metaskill_path_escape_rejected(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        skill_dir = skills_dir / "escape-ms"
        skill_dir.mkdir(parents=True)
        content = "---\nname: escape-ms\ndescription: Test.\nmetaskill: ../../../etc/passwd\n---\n"
        (skill_dir / "SKILL.md").write_text(content)
        catalog = discover_skills(str(tmp_path))
        skill = catalog["escape-ms"]
        assert skill.metaskill_path is None


class TestFrontmatter:
    def test_metaskill_fields_parsed(self):
        text = "---\nname: my-ms\ndescription: Does things.\nmetaskill: run.star\nmetaskill_language: starlark\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, dict)
        assert result["name"] == "my-ms"
        assert result["metaskill"] == "run.star"
        assert result["metaskill_language"] == "starlark"

    def test_unknown_fields_still_skipped(self):
        text = "---\nname: x\ndescription: y\nauthor: someone\n---\n\nBody"
        result = parse_frontmatter(text)
        assert "author" not in result


# =========================================================================
# Catalog rendering
# =========================================================================


class TestCatalog:
    def test_metaskill_tag_in_catalog(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(skills_dir, "my-ms", 'def run(input):\n    return "ok"')
        _make_skill(skills_dir, "static-skill", "A static skill.")
        catalog = discover_skills(str(tmp_path))
        text = format_skill_catalog(catalog, metaskill_names=["my-ms"])
        assert "(metaskill: starlark)" in text
        assert "static-skill" in text


# =========================================================================
# Tool exposure
# =========================================================================


class TestToolExposure:
    def test_run_metaskill_tool_schema(self):
        assert RUN_METASKILL_TOOL["function"]["name"] == "run_metaskill"
        params = RUN_METASKILL_TOOL["function"]["parameters"]
        assert "name" in params["properties"]
        assert "input" in params["properties"]
        assert "max_ask_calls" in params["properties"]
        assert params["required"] == ["name", "input"]

    def test_build_tools_includes_metaskill_tool(self, tmp_path):
        from swival.agent import build_tools

        tools = build_tools({}, {}, False, metaskill_names=["my-ms"])
        names = [t["function"]["name"] for t in tools]
        assert "run_metaskill" in names
        ms_tool = next(t for t in tools if t["function"]["name"] == "run_metaskill")
        assert ms_tool["function"]["parameters"]["properties"]["name"]["enum"] == [
            "my-ms"
        ]

    def test_build_tools_no_metaskill_tool_when_empty(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, False, metaskill_names=[])
        names = [t["function"]["name"] for t in tools]
        assert "run_metaskill" not in names

    def test_build_tools_no_metaskill_tool_when_none(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, False, metaskill_names=None)
        names = [t["function"]["name"] for t in tools]
        assert "run_metaskill" not in names


# =========================================================================
# Return value normalization
# =========================================================================


class TestNormalizeReturn:
    def test_none_return(self):
        assert _normalize_return_value(None) == {"status": "ok", "answer": ""}

    def test_string_return(self):
        assert _normalize_return_value("hello") == {"status": "ok", "answer": "hello"}

    def test_dict_return(self):
        val = {"status": "accepted", "answer": "done", "extra": 42}
        result = _normalize_return_value(val)
        assert result["status"] == "accepted"
        assert result["answer"] == "done"
        assert result["extra"] == 42

    def test_dict_defaults_filled(self):
        result = _normalize_return_value({"custom": True})
        assert result["status"] == "ok"
        assert result["answer"] == ""
        assert result["custom"] is True

    def test_invalid_type_raises(self):
        with pytest.raises(MetaskillError, match="invalid"):
            _normalize_return_value(42)

        with pytest.raises(MetaskillError, match="invalid"):
            _normalize_return_value([1, 2, 3])


# =========================================================================
# Budget
# =========================================================================


class TestBudget:
    def test_ask_budget(self):
        budget = MetaskillBudget(max_ask_calls=2)
        assert budget.ask_budget_remaining()
        budget.ask_calls_used = 1
        assert budget.ask_budget_remaining()
        budget.ask_calls_used = 2
        assert not budget.ask_budget_remaining()

    def test_command_budget(self):
        budget = MetaskillBudget(max_command_calls=1)
        assert budget.command_budget_remaining()
        budget.command_calls_used = 1
        assert not budget.command_budget_remaining()

    def test_timeout(self):
        budget = MetaskillBudget(timeout_s=0.001)
        budget.start()
        import time

        time.sleep(0.01)
        assert budget.timed_out()


# =========================================================================
# Trace
# =========================================================================


class TestTrace:
    def test_append(self):
        trace = MetaskillTrace(max_entries=3)
        trace.append("start", {"name": "test"})
        trace.append("step", {"n": 1})
        trace.append("step", {"n": 2})
        trace.append("overflow")
        assert len(trace.entries) == 3
        assert trace.entries[0]["kind"] == "start"
        assert trace.entries[2]["data"]["n"] == 2


# =========================================================================
# Executable metaskills policy
# =========================================================================


class TestGetExecutableMetaskills:
    @patch("swival.metaskills._check_starlark_available", return_value=True)
    def test_local_only(self, _starlark, tmp_path):
        local_skill = SkillInfo(
            name="local-ms",
            description="local",
            path=tmp_path / "local-ms",
            is_local=True,
            metaskill_path=tmp_path / "local-ms" / "SKILL.star",
            metaskill_language="starlark",
        )
        external_skill = SkillInfo(
            name="ext-ms",
            description="external",
            path=tmp_path / "ext-ms",
            is_local=False,
            metaskill_path=tmp_path / "ext-ms" / "SKILL.star",
            metaskill_language="starlark",
        )
        static_skill = SkillInfo(
            name="static",
            description="static",
            path=tmp_path / "static",
            is_local=True,
        )
        catalog = {
            "local-ms": local_skill,
            "ext-ms": external_skill,
            "static": static_skill,
        }

        names = get_executable_metaskills(catalog, "local")
        assert names == ["local-ms"]

        names = get_executable_metaskills(catalog, "all")
        assert names == ["ext-ms", "local-ms"]

    @patch("swival.metaskills._check_starlark_available", return_value=False)
    def test_empty_without_starlark(self, _starlark, tmp_path):
        local_skill = SkillInfo(
            name="local-ms",
            description="local",
            path=tmp_path / "local-ms",
            is_local=True,
            metaskill_path=tmp_path / "local-ms" / "SKILL.star",
            metaskill_language="starlark",
        )
        catalog = {"local-ms": local_skill}
        assert get_executable_metaskills(catalog, "all") == []


# =========================================================================
# Execution (with starlark runtime)
# =========================================================================


@pytest.fixture
def starlark_available():
    if not _check_starlark_available():
        pytest.skip("starlark-go not installed")


class TestExecution:
    def test_simple_metaskill(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "simple",
            'def run(input):\n    return {"status": "ok", "answer": "hello " + input.get("name", "world")}',
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "simple",
            {"name": "test"},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
        )
        assert "[Metaskill: simple completed]" in result
        data = json.loads(result.split("\n", 1)[1])
        assert data["status"] == "ok"
        assert data["answer"] == "hello test"

    def test_metaskill_with_trace(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "traced",
            'def run(input):\n    trace("step", {"n": 1})\n    trace("step", {"n": 2})\n    return "done"',
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "traced",
            {"task": "test"},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
        )
        data = json.loads(result.split("\n", 1)[1])
        kinds = [e["kind"] for e in data["trace"]]
        assert "step" in kinds
        assert "metaskill_start" in kinds
        assert "metaskill_finish" in kinds

    def test_unknown_metaskill_error(self, tmp_path):
        result = run_metaskill(
            "nonexistent",
            {},
            skills_catalog={},
            metaskills_policy="local",
            loop_kwargs={},
            tools=[],
        )
        assert result.startswith("error:")
        assert "unknown" in result

    def test_static_skill_not_a_metaskill(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "static", "A static skill.")
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "static",
            {},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={},
            tools=[],
        )
        assert result.startswith("error:")
        assert "not a metaskill" in result

    def test_external_metaskill_blocked_by_local_policy(self, tmp_path):
        skill = SkillInfo(
            name="ext-ms",
            description="external",
            path=tmp_path / "ext-ms",
            is_local=False,
            metaskill_path=tmp_path / "ext-ms" / "SKILL.star",
            metaskill_language="starlark",
        )
        catalog = {"ext-ms": skill}
        result = run_metaskill(
            "ext-ms",
            {},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={},
            tools=[],
        )
        assert result.startswith("error:")
        assert "not allowed" in result

    def test_syntax_error_in_metaskill(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "bad-syntax",
            "def run(input):\n    return undefined_var",
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "bad-syntax",
            {},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
        )
        assert result.startswith("error:")

    def test_runtime_error_in_metaskill(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "bad-runtime",
            "def run(input):\n    return 1 / 0",
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "bad-runtime",
            {"task": "test"},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
        )
        assert result.startswith("error:")
        assert "runtime error" in result

    def test_no_run_function_error(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "no-run",
            "x = 42",
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "no-run",
            {},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
        )
        assert result.startswith("error:")

    def test_starlark_not_installed_error(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "needs-star",
            'def run(input):\n    return "ok"',
        )
        catalog = discover_skills(str(tmp_path))
        with patch("swival.metaskills._check_starlark_available", return_value=False):
            result = run_metaskill(
                "needs-star",
                {},
                skills_catalog=catalog,
                metaskills_policy="local",
                loop_kwargs={},
                tools=[],
            )
        assert result.startswith("error:")
        assert "starlark runtime not installed" in result


# =========================================================================
# Budget enforcement in execution
# =========================================================================


class TestBudgetEnforcement:
    def test_ask_budget_exhausted(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "greedy",
            'def run(input):\n    for i in range(10):\n        ask("q" + str(i), {})\n    return "done"',
        )
        catalog = discover_skills(str(tmp_path))

        def fake_run_agent_loop(**kwargs):
            return "answer", False

        with patch("swival.metaskills.run_metaskill") as mock:
            mock.side_effect = lambda *a, **kw: run_metaskill(*a, **kw)

        with patch("swival.agent.run_agent_loop", side_effect=fake_run_agent_loop):
            result = run_metaskill(
                "greedy",
                {"task": "test"},
                skills_catalog=catalog,
                metaskills_policy="local",
                loop_kwargs={
                    "base_dir": str(tmp_path),
                    "api_base": "http://localhost",
                    "model_id": "test",
                    "max_output_tokens": 1000,
                    "temperature": 0.0,
                    "top_p": None,
                    "seed": None,
                    "context_length": 4096,
                    "resolved_commands": {},
                    "skills_catalog": catalog,
                    "skill_read_roots": [],
                    "extra_write_roots": [],
                    "files_mode": "some",
                    "commands_unrestricted": False,
                    "shell_allowed": False,
                    "verbose": False,
                    "llm_kwargs": {},
                    "file_tracker": None,
                    "report": None,
                    "command_policy": None,
                    "command_middleware": None,
                },
                tools=[],
                max_ask_calls=2,
            )
        assert result.startswith("error:")
        assert "budget" in result.lower()

    def test_max_command_calls_zero_honored(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "cmd-zero",
            'def run(input):\n    command(["echo", "hi"], {})\n    return "done"',
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "cmd-zero",
            {"task": "test"},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={"base_dir": str(tmp_path)},
            tools=[],
            max_command_calls=0,
        )
        assert result.startswith("error:")
        assert "budget" in result.lower()

    def test_command_nonzero_exit_detected(self, tmp_path, starlark_available):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_metaskill(
            skills_dir,
            "cmd-fail",
            'def run(input):\n    r = command(["false"], {})\n    return {"ok": r["ok"], "exit_code": r["exit_code"]}',
        )
        catalog = discover_skills(str(tmp_path))
        result = run_metaskill(
            "cmd-fail",
            {"task": "test"},
            skills_catalog=catalog,
            metaskills_policy="local",
            loop_kwargs={
                "base_dir": str(tmp_path),
                "resolved_commands": {"false": "/usr/bin/false"},
                "commands_unrestricted": True,
                "shell_allowed": False,
                "files_mode": "some",
                "skill_read_roots": [],
                "extra_write_roots": [],
                "verbose": False,
                "command_policy": None,
                "command_middleware": None,
            },
            tools=[],
        )
        assert "[Metaskill: cmd-fail completed]" in result
        data = json.loads(result.split("\n", 1)[1])
        assert data["ok"] is False
        assert data["exit_code"] != 0


# =========================================================================
# Result envelope
# =========================================================================


class TestResultEnvelope:
    def test_basic_envelope(self):
        budget = MetaskillBudget()
        budget.start()
        trace = MetaskillTrace()
        trace.append("step", {"n": 1})
        return_value = {"status": "accepted", "answer": "done"}
        text = _build_result_envelope(return_value, trace, budget)
        data = json.loads(text)
        assert data["status"] == "accepted"
        assert data["answer"] == "done"
        assert data["trace"][0]["kind"] == "step"

    def test_envelope_truncation(self):
        budget = MetaskillBudget(max_result_chars=100)
        budget.start()
        trace = MetaskillTrace()
        for i in range(50):
            trace.append("step", {"n": i, "data": "x" * 10})
        return_value = {"status": "ok", "answer": "short"}
        text = _build_result_envelope(return_value, trace, budget)
        assert len(text) <= 200  # some overhead is OK


# =========================================================================
# Backward compatibility: existing skill tests should still pass
# =========================================================================


class TestBackwardCompat:
    def test_static_skill_activation_unchanged(self, tmp_path):
        from swival.skills import activate_skill

        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "plain", "A plain skill.", body="Do the thing.")
        catalog = discover_skills(str(tmp_path))
        result = activate_skill("plain", catalog, [])
        assert "[Skill: plain activated]" in result
        assert "Do the thing." in result

    def test_static_skill_no_metaskill_fields(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "vanilla", "Vanilla skill.")
        catalog = discover_skills(str(tmp_path))
        skill = catalog["vanilla"]
        assert skill.metaskill_path is None
        assert skill.metaskill_language is None
