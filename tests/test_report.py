"""Tests for the JSON report feature (--report)."""

import json
import types
from unittest.mock import MagicMock, patch

import pytest

from swival.report import AgentError, ReportCollector
from swival import agent


# ---------------------------------------------------------------------------
# ReportCollector unit tests
# ---------------------------------------------------------------------------


class TestReportCollector:
    def test_empty_report(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="hello",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="done",
            exit_code=0,
            turns=0,
        )
        assert r["version"] == 1
        assert r["task"] == "hello"
        assert r["result"]["outcome"] == "success"
        assert r["result"]["answer"] == "done"
        assert r["stats"]["turns"] == 0
        assert r["stats"]["tool_calls_total"] == 0
        assert r["stats"]["llm_calls"] == 0
        assert r["timeline"] == []

    def test_llm_call_tracking(self):
        rc = ReportCollector()
        rc.record_llm_call(1, 2.5, 1000, "tool_calls")
        rc.record_llm_call(2, 1.3, 1500, "stop")
        assert rc.llm_calls == 2
        assert rc.total_llm_time == pytest.approx(3.8)
        assert rc.max_turn_seen == 2
        assert len(rc.events) == 2
        assert rc.events[0]["finish_reason"] == "tool_calls"
        assert rc.events[0]["is_retry"] is False
        assert rc.events[1]["finish_reason"] == "stop"

    def test_llm_call_retry(self):
        rc = ReportCollector()
        rc.record_llm_call(
            1,
            0.5,
            5000,
            "context_overflow",
            is_retry=False,
        )
        rc.record_llm_call(
            1,
            1.0,
            3000,
            "tool_calls",
            is_retry=True,
            retry_reason="compact_messages",
        )
        assert rc.llm_calls == 2
        assert rc.events[0]["is_retry"] is False
        assert "retry_reason" not in rc.events[0]
        assert rc.events[1]["is_retry"] is True
        assert rc.events[1]["retry_reason"] == "compact_messages"

    def test_tool_call_tracking(self):
        rc = ReportCollector()
        rc.record_tool_call(1, "read_file", {"path": "a.txt"}, True, 0.01, 500)
        rc.record_tool_call(
            1, "read_file", {"path": "b.txt"}, False, 0.02, 30, error="error: not found"
        )
        rc.record_tool_call(2, "edit_file", {"path": "a.txt"}, True, 0.05, 200)

        assert rc.tool_stats["read_file"] == {"succeeded": 1, "failed": 1}
        assert rc.tool_stats["edit_file"] == {"succeeded": 1, "failed": 0}
        assert rc.total_tool_time == pytest.approx(0.08)

        r = rc.build_report(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=2,
        )
        assert r["stats"]["tool_calls_total"] == 3
        assert r["stats"]["tool_calls_succeeded"] == 2
        assert r["stats"]["tool_calls_failed"] == 1

    def test_compaction_tracking(self):
        rc = ReportCollector()
        rc.record_compaction(3, "compact_messages", 95000, 62000)
        rc.record_compaction(5, "drop_middle_turns", 62000, 30000)
        assert rc.compactions == 1
        assert rc.turn_drops == 1
        assert len(rc.events) == 2
        assert rc.events[0]["strategy"] == "compact_messages"
        assert rc.events[1]["strategy"] == "drop_middle_turns"

    def test_guardrail_tracking(self):
        rc = ReportCollector()
        rc.record_guardrail(3, "edit_file", "nudge")
        rc.record_guardrail(4, "edit_file", "stop")
        assert rc.guardrail_interventions == 2
        assert rc.events[0]["level"] == "nudge"
        assert rc.events[1]["level"] == "stop"

    def test_truncated_response_tracking(self):
        rc = ReportCollector()
        rc.record_truncated_response(2)
        assert rc.truncated_responses == 1
        assert rc.events[0]["type"] == "truncated_response"
        assert rc.events[0]["turn"] == 2

    def test_error_outcome(self):
        rc = ReportCollector()
        rc.record_llm_call(1, 0.5, 1000, "context_overflow")
        r = rc.build_report(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="error",
            answer=None,
            exit_code=1,
            turns=1,
            error_message="context window exceeded",
        )
        assert r["result"]["outcome"] == "error"
        assert r["result"]["error_message"] == "context window exceeded"
        assert r["result"]["answer"] is None

    def test_write_creates_valid_json(self, tmp_path):
        rc = ReportCollector()
        rc.record_llm_call(1, 1.0, 500, "stop")
        rc.finalize(
            task="test",
            model="m",
            provider="p",
            settings={"a": 1},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
        )
        path = str(tmp_path / "report.json")
        rc.write(path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["result"]["answer"] == "ok"

    def test_max_turn_seen(self):
        rc = ReportCollector()
        rc.record_llm_call(1, 0.1, 100, "tool_calls")
        rc.record_llm_call(5, 0.1, 100, "stop")
        rc.record_llm_call(3, 0.1, 100, "stop")  # out of order
        assert rc.max_turn_seen == 5

    def test_todo_stats_included(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="t",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
            todo_stats={"added": 3, "completed": 2, "remaining": 1},
        )
        assert r["stats"]["todo"] == {"added": 3, "completed": 2, "remaining": 1}

    def test_todo_stats_omitted_when_none(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="t",
            model="m",
            provider="lmstudio",
            settings={},
            outcome="success",
            answer="ok",
            exit_code=0,
            turns=1,
        )
        assert "todo" not in r["stats"]


# ---------------------------------------------------------------------------
# handle_tool_call tuple return
# ---------------------------------------------------------------------------


class TestHandleToolCallTuple:
    def test_success_returns_tuple(self, tmp_path):
        from swival.thinking import ThinkingState

        (tmp_path / "test.txt").write_text("hello")
        tc = MagicMock()
        tc.id = "tc1"
        tc.function.name = "read_file"
        tc.function.arguments = json.dumps({"file_path": "test.txt"})
        ts = ThinkingState(verbose=False)

        msg, meta = agent.handle_tool_call(tc, str(tmp_path), ts, verbose=False)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc1"
        assert "hello" in msg["content"]
        assert meta["name"] == "read_file"
        assert meta["arguments"] == {"file_path": "test.txt"}
        assert meta["succeeded"] is True
        assert meta["elapsed"] >= 0

    def test_failure_returns_tuple(self, tmp_path):
        from swival.thinking import ThinkingState

        tc = MagicMock()
        tc.id = "tc2"
        tc.function.name = "read_file"
        tc.function.arguments = json.dumps({"file_path": "nonexistent.txt"})
        ts = ThinkingState(verbose=False)

        msg, meta = agent.handle_tool_call(tc, str(tmp_path), ts, verbose=False)
        assert msg["content"].startswith("error:")
        assert meta["succeeded"] is False
        assert meta["name"] == "read_file"

    def test_invalid_json_returns_stable_meta(self, tmp_path):
        from swival.thinking import ThinkingState

        tc = MagicMock()
        tc.id = "tc3"
        tc.function.name = "read_file"
        tc.function.arguments = "not valid json"
        ts = ThinkingState(verbose=False)

        msg, meta = agent.handle_tool_call(tc, str(tmp_path), ts, verbose=False)
        assert msg["content"].startswith("error:")
        assert meta["name"] == "read_file"
        assert meta["arguments"] is None
        assert meta["elapsed"] == 0.0
        assert meta["succeeded"] is False


# ---------------------------------------------------------------------------
# load_instructions tuple return
# ---------------------------------------------------------------------------


class TestLoadInstructionsTuple:
    def test_no_files(self, tmp_path):
        text, loaded = agent.load_instructions(str(tmp_path), verbose=False)
        assert text == ""
        assert loaded == []

    def test_claude_md_only(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("rules")
        text, loaded = agent.load_instructions(str(tmp_path), verbose=False)
        assert "rules" in text
        assert loaded == ["CLAUDE.md"]

    def test_both_files(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("a")
        (tmp_path / "AGENTS.md").write_text("b")
        text, loaded = agent.load_instructions(str(tmp_path), verbose=False)
        assert loaded == ["CLAUDE.md", "AGENTS.md"]


# ---------------------------------------------------------------------------
# --report + --repl validation
# ---------------------------------------------------------------------------


class TestReportCLIValidation:
    def test_report_with_repl_is_error(self):
        parser = agent.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            args = parser.parse_args(["--repl", "--report", "out.json", "hello"])
            args.verbose = True
            # Trigger the validation in main
            from swival import fmt

            fmt.init()
            if args.report and args.repl:
                parser.error("--report is incompatible with --repl")
        assert exc_info.value.code == 2

    def test_report_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--report", "/tmp/out.json", "hello"])
        assert args.report == "/tmp/out.json"

    def test_report_default_none(self):
        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        assert args.report is None


# ---------------------------------------------------------------------------
# Overflow retry seed bugfix
# ---------------------------------------------------------------------------


class TestOverflowRetrySeed:
    """Verify that the retry call_llm invocations pass seed correctly."""

    def test_seed_passed_to_retry_after_compaction(self):
        """When the first call_llm overflows and compaction is done,
        the retry must pass seed in the correct position."""
        call_count = 0

        def mock_call_llm(
            base_url,
            model_id,
            messages,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            tools,
            verbose,
            **kwargs,
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise agent.ContextOverflowError("overflow")
            # Return a simple final answer
            msg = MagicMock()
            msg.content = "done"
            msg.tool_calls = None
            return msg, "stop"

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

        with patch.object(agent, "call_llm", side_effect=mock_call_llm) as mock:
            result, exhausted = agent.run_agent_loop(
                messages,
                [],
                api_base="http://localhost",
                model_id="test-model",
                max_turns=5,
                max_output_tokens=1000,
                temperature=0.5,
                top_p=1.0,
                seed=42,
                context_length=None,
                base_dir="/tmp",
                thinking_state=MagicMock(),
                todo_state=MagicMock(),
                resolved_commands={},
                skills_catalog={},
                skill_read_roots=[],
                extra_write_roots=[],
                yolo=False,
                verbose=False,
                llm_kwargs={"provider": "lmstudio", "api_key": None},
            )

        assert result == "done"
        assert not exhausted
        assert call_count == 2
        # Verify both calls received seed=42 in the correct position
        for call in mock.call_args_list:
            args = call[0]
            # seed is the 7th positional arg (index 6)
            assert args[6] == 42, f"seed was {args[6]} instead of 42"


# ---------------------------------------------------------------------------
# Non-overflow LLM failure recording
# ---------------------------------------------------------------------------


class TestNonOverflowLLMFailure:
    """AgentError from call_llm (non-overflow) should still be recorded."""

    def test_agent_error_recorded_in_timeline(self):
        rc = ReportCollector()

        def mock_call_llm(*args, **kwargs):
            raise AgentError("LLM call failed: bad request")

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

        with patch.object(agent, "call_llm", side_effect=mock_call_llm):
            with pytest.raises(AgentError):
                agent.run_agent_loop(
                    messages,
                    [],
                    api_base="http://localhost",
                    model_id="test-model",
                    max_turns=5,
                    max_output_tokens=1000,
                    temperature=0.5,
                    top_p=1.0,
                    seed=None,
                    context_length=None,
                    base_dir="/tmp",
                    thinking_state=MagicMock(),
                    todo_state=MagicMock(),
                    resolved_commands={},
                    skills_catalog={},
                    skill_read_roots=[],
                    extra_write_roots=[],
                    yolo=False,
                    verbose=False,
                    llm_kwargs={"provider": "lmstudio", "api_key": None},
                    report=rc,
                )

        assert rc.llm_calls == 1
        assert rc.events[0]["type"] == "llm_call"
        assert rc.events[0]["finish_reason"] == "error"
        assert rc.events[0]["duration_s"] >= 0
        assert rc.max_turn_seen == 1


# ---------------------------------------------------------------------------
# Error report turns accuracy
# ---------------------------------------------------------------------------


class TestErrorReportTurns:
    """Error reports should use the actual turn count from the collector."""

    def test_error_after_turns_reports_correct_count(self):
        rc = ReportCollector()
        # Simulate 3 turns before failure
        rc.record_llm_call(1, 0.1, 100, "tool_calls")
        rc.record_llm_call(2, 0.1, 100, "tool_calls")
        rc.record_llm_call(3, 0.1, 100, "error")

        r = rc.build_report(
            task="test",
            model="m",
            provider="p",
            settings={},
            outcome="error",
            answer=None,
            exit_code=1,
            turns=rc.max_turn_seen,
            error_message="failed",
        )
        assert r["stats"]["turns"] == 3
        assert r["stats"]["llm_calls"] == 3


# ---------------------------------------------------------------------------
# Integration: report written on success
# ---------------------------------------------------------------------------


class TestReportIntegration:
    def _base_args(self, tmp_path, report_path):
        return types.SimpleNamespace(
            question="test task",
            repl=False,
            report=str(report_path),
            provider="lmstudio",
            model="test-model",
            api_key=None,
            base_url="http://localhost:1234",
            max_context_tokens=None,
            max_output_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            seed=None,
            system_prompt=None,
            no_system_prompt=True,
            quiet=False,
            max_turns=5,
            base_dir=str(tmp_path),
            allowed_commands=None,
            no_instructions=True,
            skills_dir=[],
            no_skills=True,
            add_dir=[],
            add_dir_ro=[],
            yolo=False,
            color=False,
            no_color=True,
            verbose=True,
            version=False,
            no_read_guard=False,
            reviewer=None,
            no_history=True,
            init_config=False,
            project=False,
        )

    def test_report_written_on_success(self, tmp_path):
        report_path = tmp_path / "report.json"
        fake_args = self._base_args(tmp_path, report_path)

        # Mock the LLM to return a final answer
        def mock_call_llm(*args, **kwargs):
            msg = MagicMock()
            msg.content = "final answer"
            msg.tool_calls = None
            return msg, "stop"

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(agent, "call_llm", side_effect=mock_call_llm),
            patch.object(agent, "discover_model", return_value=("test-model", None)),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["version"] == 1
        assert data["result"]["outcome"] == "success"
        assert data["result"]["answer"] == "final answer"
        assert data["result"]["exit_code"] == 0
        assert data["model"] == "test-model"
        assert data["stats"]["llm_calls"] >= 1

    def test_report_written_on_error(self, tmp_path):
        report_path = tmp_path / "report.json"
        fake_args = self._base_args(tmp_path, report_path)
        # Remove model so discover_model is called
        fake_args.model = None

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(
                agent, "discover_model", side_effect=AgentError("connection refused")
            ),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            with pytest.raises(SystemExit) as exc_info:
                agent.main()
            assert exc_info.value.code == 1

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["result"]["outcome"] == "error"
        assert "connection refused" in data["result"]["error_message"]

    def test_stdout_also_printed_when_report_active(self, tmp_path, capsys):
        report_path = tmp_path / "report.json"
        fake_args = self._base_args(tmp_path, report_path)

        def mock_call_llm(*args, **kwargs):
            msg = MagicMock()
            msg.content = "secret answer"
            msg.tool_calls = None
            return msg, "stop"

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(agent, "call_llm", side_effect=mock_call_llm),
            patch.object(agent, "discover_model", return_value=("m", None)),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        captured = capsys.readouterr()
        assert "secret answer" in captured.out

    def test_report_write_failure_does_not_crash(self, tmp_path, capsys):
        """If the report path is unwritable, emit an error instead of crashing."""
        bad_path = "/no/such/dir/report.json"
        fake_args = self._base_args(tmp_path, bad_path)

        def mock_call_llm(*args, **kwargs):
            msg = MagicMock()
            msg.content = "answer"
            msg.tool_calls = None
            return msg, "stop"

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(agent, "call_llm", side_effect=mock_call_llm),
            patch.object(agent, "discover_model", return_value=("m", None)),
            patch.object(agent.fmt, "error") as mock_fmt_error,
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            # Should not raise — write failure is caught
            agent.main()

        # fmt.error should have been called with the write failure message
        assert any(
            "Failed to write report" in str(c) for c in mock_fmt_error.call_args_list
        )

    def test_no_report_prints_to_stdout(self, tmp_path, capsys):
        fake_args = self._base_args(tmp_path, None)
        fake_args.report = None

        def mock_call_llm(*args, **kwargs):
            msg = MagicMock()
            msg.content = "visible answer"
            msg.tool_calls = None
            return msg, "stop"

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(agent, "call_llm", side_effect=mock_call_llm),
            patch.object(agent, "discover_model", return_value=("m", None)),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        captured = capsys.readouterr()
        assert "visible answer" in captured.out
