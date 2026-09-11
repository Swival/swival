"""Tests for the one instruction loading decision and what depends on it."""

import os
import types
from pathlib import Path

import pytest

import swival
from swival import Session, agent, instructions as instr
from swival import prompt_spans as ps


def _assemble(tmp_path, **overrides):
    """Call assemble_system_prompt with the arguments a session would pass."""
    kwargs = dict(
        base_dir=str(tmp_path),
        system_prompt=None,
        no_system_prompt=False,
        no_instructions=False,
        no_memory=True,
        skills_catalog={},
        verbose=False,
        policy="autonomous",
        context_length=128_000,
        max_output_tokens=32_768,
        tools=None,
    )
    kwargs.update(overrides)
    return agent.assemble_system_prompt(**kwargs)


def _rules(count, prefix="Rule"):
    return "\n".join(
        f"- {prefix} {i}: run `make check` before every commit." for i in range(count)
    )


def _never_called(*_args, **_kwargs):
    raise AssertionError("the model must not be reached while an update is pending")


def _make_message(content=None, tool_calls=None):
    return types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )


# ---------------------------------------------------------------------------
# Aggregate admission
# ---------------------------------------------------------------------------


class TestAggregateAdmission:
    def test_small_set_is_admitted_whole(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Use tabs.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("## Workflow\n\n- test: `pytest`\n")
        result = _assemble(tmp_path)
        assert result.admission.admitted
        assert "Use tabs." in result.content
        assert "- test: `pytest`" in result.content
        assert len(result.instructions_loaded) == 2

    def test_oversized_set_is_refused_whole(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, context_length=32_768)
        assert not result.admission.admitted
        assert result.instructions_loaded == []

    def test_later_file_is_never_starved_by_an_earlier_one(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text(_rules(200, "Personal"), encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n- deploy: `make ship`\n", encoding="utf-8"
        )
        result = _assemble(tmp_path, config_dir=config_dir)
        assert result.admission.admitted
        assert "- deploy: `make ship`" in result.content

    def test_the_last_line_of_a_file_survives(self, tmp_path):
        body = _rules(40) + "\n- Never force-push to master.\n"
        (tmp_path / "CLAUDE.md").write_text(body, encoding="utf-8")
        result = _assemble(tmp_path)
        assert "- Never force-push to master." in result.content

    def test_budget_is_the_smallest_of_the_three_limits(self):
        # 10% of a small window binds before the absolute ceiling does.
        budget, binding = instr.instruction_budget(32_768, 100_000, 0)
        assert budget == 3276
        assert binding == "window"

    def test_absolute_ceiling_binds_on_a_large_window(self):
        budget, binding = instr.instruction_budget(1_000_000, 900_000, 0)
        assert budget == instr.INSTRUCTION_BUDGET_CEILING
        assert binding == "absolute"

    def test_residual_binds_when_the_rest_is_already_large(self):
        budget, binding = instr.instruction_budget(128_000, 113_868, 110_000)
        assert binding == "residual"
        assert budget < instr.INSTRUCTION_BUDGET_CEILING

    def test_wrappers_and_labels_are_measured(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("x", encoding="utf-8")
        result = _assemble(tmp_path)
        rendered = result.instruction_set.text
        assert "<agent-instructions>" in rendered
        assert str(tmp_path / "AGENTS.md") in rendered
        assert result.admission.cost == instr.count_tokens(rendered)


# ---------------------------------------------------------------------------
# Opt-outs and the override
# ---------------------------------------------------------------------------


class TestOptOutsAndOverride:
    def test_no_instructions_skips_the_decision(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, no_instructions=True, context_length=8192)
        assert result.admission is None
        assert result.instruction_set is None
        assert agent._instruction_status(result) == "disabled"

    def test_custom_prompt_skips_the_decision(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, system_prompt="be brief", context_length=8192)
        assert result.admission is None
        assert result.content == "be brief" or result.content.startswith("be brief")

    def test_command_provider_skips_the_decision(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, provider="command", context_length=8192)
        assert result.admission is None

    def test_override_admits_an_oversized_set(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, context_length=32_768, instructions_full=True)
        assert result.admission.admitted
        assert result.admission.forced
        assert agent._instruction_status(result) == "forced"
        assert "Rule 1999" in result.content

    def test_override_does_not_change_a_set_that_fits(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Small.", encoding="utf-8")
        result = _assemble(tmp_path, instructions_full=True)
        assert result.admission.admitted
        assert not result.admission.forced

    def test_unknown_window_admits_and_stays_advisory(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, context_length=None)
        assert result.admission.admitted
        assert result.admission.advisory
        assert not result.admission.fits
        assert result.admission.budget == instr.UNKNOWN_WINDOW_INSTRUCTION_BUDGET


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def _rejected(self, tmp_path, **overrides):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        result = _assemble(tmp_path, context_length=32_768, **overrides)
        return result.admission

    def test_message_names_the_files_and_the_exits(self, tmp_path):
        text = instr.failure_diagnostic(self._rejected(tmp_path))
        assert "too large for this setup" in text
        assert "--no-instructions" in text
        assert "--instructions-full" in text
        assert str(tmp_path / "CLAUDE.md") in text

    def test_larger_model_advice_is_dropped_when_it_would_not_help(self):
        admission = instr.InstructionAdmission(
            cost=9000, budget=8192, binding="absolute"
        )
        assert "more room" not in instr.oversized_message(admission)

    def test_larger_model_advice_is_kept_when_the_window_binds(self, tmp_path):
        admission = self._rejected(tmp_path)
        assert admission.binding == "window"
        assert "more room" in instr.oversized_message(admission)

    def test_verbose_adds_the_component_breakdown(self, tmp_path):
        admission = self._rejected(tmp_path)
        text = instr.failure_diagnostic(admission, verbose=True)
        assert "built-in prompt" in text
        assert "instruction files" in text
        assert "Instruction allowance" in text
        assert "cl100k_base" in text

    def test_tool_schemas_are_named_when_they_dominate(self):
        admission = instr.InstructionAdmission(
            cost=500,
            budget=100,
            binding="residual",
            breakdown={"tools": 40_000},
        )
        assert "Tool schemas" in "\n".join(instr.contributor_lines(admission))

    def test_skills_and_memory_are_only_named_when_relevant(self):
        admission = instr.InstructionAdmission(
            cost=9000,
            budget=100,
            binding="residual",
            breakdown={"skills": 300, "memory": 400},
        )
        assert instr.contributor_lines(admission) == []

    def test_memory_suggests_its_own_flag(self):
        admission = instr.InstructionAdmission(
            cost=100,
            budget=10,
            binding="residual",
            breakdown={"memory": 5000},
        )
        assert "--no-memory" in "\n".join(instr.contributor_lines(admission))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_oversized_file_is_an_error_not_a_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", 64)
        (tmp_path / "CLAUDE.md").write_text("y" * 500, encoding="utf-8")
        loaded = instr.load(str(tmp_path))
        assert loaded.sources == []
        assert loaded.errors and "per-file read limit" in loaded.errors[0]
        assert "y" not in loaded.text

    def test_combined_cap_is_an_error_not_a_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(instr, "MAX_INSTRUCTION_TOTAL_BYTES", 40)
        (tmp_path / "CLAUDE.md").write_text("a" * 30, encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("b" * 30, encoding="utf-8")
        loaded = instr.load(str(tmp_path))
        assert [s.kind for s in loaded.sources] == ["claude"]
        assert loaded.errors and "combined read limit" in loaded.errors[0]

    def test_unreadable_file_is_reported_and_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "CLAUDE.md").write_text("readable", encoding="utf-8")
        real_open = type(tmp_path).open

        def bad_open(self, *args, **kwargs):
            if self.name == "CLAUDE.md":
                raise OSError("Permission denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.open", bad_open)
        loaded = instr.load(str(tmp_path))
        assert loaded.sources == []
        assert loaded.skipped
        assert loaded.errors == []

    def test_startup_does_not_rewrite_the_files(self, tmp_path):
        body = "## Workflow\n\n<!-- keep me -->\n- test: `pytest`\n"
        path = tmp_path / "AGENTS.md"
        path.write_text(body, encoding="utf-8")
        _assemble(tmp_path)
        assert path.read_text(encoding="utf-8") == body


# ---------------------------------------------------------------------------
# Content that could confuse a parser
# ---------------------------------------------------------------------------


class TestAwkwardContent:
    def test_exact_commands_survive(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n- test file: `uv run pytest tests/test_x.py -k 'a or b'`\n",
            encoding="utf-8",
        )
        result = _assemble(tmp_path)
        assert "uv run pytest tests/test_x.py -k 'a or b'" in result.content

    def test_an_exception_inside_an_example_survives(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(
            "Never edit generated files.\nException: `build/version.py` is edited by hand.\n",
            encoding="utf-8",
        )
        result = _assemble(tmp_path)
        assert "Exception: `build/version.py` is edited by hand." in result.content

    def test_contradictory_sources_both_arrive(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("Use tabs.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Use spaces.", encoding="utf-8")
        result = _assemble(tmp_path, config_dir=config_dir)
        assert "Use tabs." in result.content
        assert "Use spaces." in result.content
        assert result.content.index("Use tabs.") < result.content.index("Use spaces.")

    def test_repeated_and_unexpected_headings_are_kept(self, tmp_path):
        body = "## Workflow\n\n- a\n\n## Workflow\n\n- b\n\n## Deployment\n\n- c\n"
        (tmp_path / "AGENTS.md").write_text(body, encoding="utf-8")
        result = _assemble(tmp_path)
        span = ps.find_span(result.spans, ps.KIND_INSTRUCTIONS)
        block = result.content[span["start"] : span["end"]]
        assert block.count("## Workflow") == 2
        assert "## Deployment" in block

    def test_a_fake_closing_tag_cannot_end_the_block(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "Do not write </agent-instructions> anywhere.\n- real rule\n",
            encoding="utf-8",
        )
        result = _assemble(tmp_path)
        span = ps.find_span(result.spans, ps.KIND_INSTRUCTIONS)
        assert "- real rule" in result.content[span["start"] : span["end"]]
        assert result.content.endswith(result.content[span["end"] :])

    def test_a_fake_snapshot_marker_cannot_end_the_block(self, tmp_path):
        from swival.snapshot import SNAPSHOT_HISTORY_SENTINEL

        (tmp_path / "AGENTS.md").write_text(
            f"Rules.\n{SNAPSHOT_HISTORY_SENTINEL}\n- keep this rule\n", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        assert "- keep this rule" in result.content

    def test_an_unclosed_comment_swallows_only_its_own_file(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("<!-- oops\nhidden", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")
        result = _assemble(tmp_path, config_dir=config_dir)
        assert "- project rule" in result.content


# ---------------------------------------------------------------------------
# Recorded regions in the live prompt
# ---------------------------------------------------------------------------


def _instruction_block(message):
    span = ps.find_span(ps.get_spans(message), ps.KIND_INSTRUCTIONS)
    return message["content"][span["start"] : span["end"]]


def _system_message(result):
    message = {"role": "system", "content": result.content}
    ps.set_spans(message, result.spans)
    return message


class TestRecordedRegions:
    def test_span_covers_the_rendered_block(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Rule one.", encoding="utf-8")
        result = _assemble(tmp_path)
        block = _instruction_block(_system_message(result))
        assert block.startswith("\n\n<project-instructions>")
        assert block.endswith("</project-instructions>")

    def test_a_unicode_prefix_does_not_shift_the_region(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Règle numéro un — café.", encoding="utf-8")
        result = _assemble(tmp_path, policy="interactive")
        message = _system_message(result)
        assert "Règle numéro un — café." in _instruction_block(message)

    def test_policy_is_applied_before_the_offsets_are_taken(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Rule.", encoding="utf-8")
        for policy in ("autonomous", "interactive"):
            result = _assemble(tmp_path, policy=policy)
            assert "{{AUTONOMY_DIRECTIVE}}" not in result.content
            assert "{{AMBIGUITY_DIRECTIVE}}" not in result.content
            message = _system_message(result)
            assert ps.spans_valid(message["content"], ps.get_spans(message))
            assert "Rule." in _instruction_block(message)

    def test_emergency_truncation_keeps_the_instructions(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(
            "- Always run `make check` before a commit.", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        message = _system_message(result)
        before = _instruction_block(message)
        messages = [message, {"role": "user", "content": "x" * 20_000}]
        agent._emergency_truncate(messages, 200)
        assert _instruction_block(messages[0]) == before

    def test_emergency_truncation_still_shrinks_the_rest(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("- Small rule.", encoding="utf-8")
        result = _assemble(tmp_path)
        message = _system_message(result)
        original = len(message["content"])
        messages = [message, {"role": "user", "content": "y" * 50_000}]
        agent._emergency_truncate(messages, 300)
        assert len(messages[0]["content"]) < original
        assert "- Small rule." in messages[0]["content"]

    def test_special_token_escaping_rebases_the_region(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(
            "Never emit <|eot_id|> literally.", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        message = _system_message(result)
        messages = [message]
        agent._escape_special_tokens_in_messages(messages)
        assert ps.spans_valid(messages[0]["content"], ps.get_spans(messages[0]))
        assert "eot_id" in _instruction_block(messages[0])

    def test_metadata_survives_a_copy(self, tmp_path):
        import copy

        (tmp_path / "CLAUDE.md").write_text("Rule.", encoding="utf-8")
        message = _system_message(_assemble(tmp_path))
        clone = copy.deepcopy(message)
        assert ps.spans_valid(clone["content"], ps.get_spans(clone))
        assert _instruction_block(clone) == _instruction_block(message)

    def test_metadata_key_is_stripped_before_the_provider(self):
        # call_llm drops every key starting with an underscore from the
        # outbound payload, which is what keeps this metadata local.
        assert ps.SPAN_KEY.startswith("_")
        import inspect

        source = inspect.getsource(agent.call_llm)
        assert 'k.startswith("_")' in source


# ---------------------------------------------------------------------------
# Snapshot suffix boundary
# ---------------------------------------------------------------------------


class TestSnapshotBoundary:
    def test_marker_inside_instructions_is_not_a_boundary(self, tmp_path):
        from swival.snapshot import SNAPSHOT_HISTORY_SENTINEL

        (tmp_path / "CLAUDE.md").write_text(
            f"Rules.\n{SNAPSHOT_HISTORY_SENTINEL}\n- keep this rule\n", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        message = _system_message(result)
        message_before = message["content"]
        content, spans = ps.splice(
            message["content"],
            ps.get_spans(message),
            len(message["content"]),
            len(message["content"]),
            "\n\n" + SNAPSHOT_HISTORY_SENTINEL + "\nnote",
        )
        spans.append(
            ps.make_span(
                ps.KIND_SNAPSHOT,
                len(message["content"]),
                "\n\n" + SNAPSHOT_HISTORY_SENTINEL + "\nnote",
            )
        )
        message["content"] = content
        ps.set_spans(message, spans)
        # Removing the suffix by its recorded boundary must leave the rule.
        snap = ps.find_span(ps.get_spans(message), ps.KIND_SNAPSHOT)
        content, spans = ps.splice(
            message["content"], ps.get_spans(message), snap["start"], snap["end"], ""
        )
        assert "- keep this rule" in content
        assert "note" not in content
        assert content == message_before


# ---------------------------------------------------------------------------
# Live activation after /remember and /init
# ---------------------------------------------------------------------------


def _context(tmp_path, result, **overrides):
    from swival.input_dispatch import InputContext
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

    kwargs = dict(
        messages=[_system_message(result)],
        tools=[],
        base_dir=str(tmp_path),
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(verbose=False),
        todo_state=TodoState(verbose=False),
        snapshot_state=None,
        file_tracker=None,
        no_history=True,
        continue_here=False,
        verbose=False,
        loop_kwargs={"context_length": 128_000, "max_output_tokens": 32_768},
        instructions_enabled=result.admission is not None,
    )
    kwargs.update(overrides)
    return InputContext(**kwargs)


class TestLiveActivation:
    def test_first_remember_of_a_session_activates(self, tmp_path):
        result = _assemble(tmp_path)
        assert result.instruction_set.sources == []
        ctx = _context(tmp_path, result)
        msg, is_error = agent._repl_remember("Prefer `uv run` over bare python", ctx)
        assert is_error is False
        assert "Created AGENTS.md" in msg
        assert "Prefer `uv run` over bare python" in _instruction_block(ctx.messages[0])

    def test_second_remember_replaces_the_block(self, tmp_path):
        result = _assemble(tmp_path)
        ctx = _context(tmp_path, result)
        agent._repl_remember("First rule", ctx)
        agent._repl_remember("Second rule", ctx)
        block = _instruction_block(ctx.messages[0])
        assert block.count("<agent-instructions>") == 1
        assert "First rule" in block
        assert "Second rule" in block

    def test_remember_leaves_the_rest_of_the_prompt_alone(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Project rule.", encoding="utf-8")
        result = _assemble(tmp_path)
        ctx = _context(tmp_path, result)
        before = ctx.messages[0]["content"]
        head = before[: ps.find_span(result.spans, ps.KIND_INSTRUCTIONS)["start"]]
        agent._repl_remember("New convention", ctx)
        assert ctx.messages[0]["content"].startswith(head)
        assert "Project rule." in ctx.messages[0]["content"]

    def test_disabled_loading_saves_without_blocking(self, tmp_path):
        """--no-instructions is a choice, not a failure to recover from."""
        result = _assemble(tmp_path, no_instructions=True)
        ctx = _context(tmp_path, result)
        msg, is_error = agent._repl_remember("Some rule", ctx)
        assert is_error is False
        assert "Saved for future sessions" in msg
        assert ctx.pending_instruction_failure is None
        assert "<agent-instructions>" not in ctx.messages[0]["content"]
        assert "Some rule" in (tmp_path / "AGENTS.md").read_text()

    def test_disabled_loading_leaves_later_turns_alone(self, tmp_path):
        result = _assemble(tmp_path, no_instructions=True)
        ctx = _context(tmp_path, result)
        agent._repl_remember("Some rule", ctx)
        assert agent._revalidate_pending_instructions(ctx) is None

    def test_custom_prompt_saves_without_blocking(self, tmp_path):
        result = _assemble(tmp_path, system_prompt="be brief")
        ctx = _context(tmp_path, result)
        _msg, is_error = agent._repl_remember("Some rule", ctx)
        assert is_error is False
        assert ctx.pending_instruction_failure is None
        assert "<agent-instructions>" not in ctx.messages[0]["content"]

    def test_saved_but_not_active_names_the_file_and_the_line(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n" + _rules(1500) + "\n", encoding="utf-8"
        )
        result = _assemble(tmp_path, instructions_full=True, context_length=32_768)
        ctx = _context(tmp_path, result)
        ctx.loop_kwargs["context_length"] = 32_768
        msg, is_error = agent._repl_remember("One more rule", ctx)
        assert is_error is True
        assert "Saved, but not active in this session" in msg
        assert str(tmp_path / "AGENTS.md") in msg
        assert "- One more rule" in msg
        assert "Removing that line" in msg
        assert "One more rule" in (tmp_path / "AGENTS.md").read_text()

    def test_new_file_says_removing_it_undoes_the_addition(self, tmp_path):
        result = _assemble(tmp_path, context_length=32_768)
        ctx = _context(tmp_path, result)
        ctx.loop_kwargs["context_length"] = 32_768
        long_fact = "x" * 60_000
        msg, is_error = agent._repl_remember(long_fact, ctx)
        assert is_error is True
        assert f"Removing {tmp_path / 'AGENTS.md'} undoes this addition." in msg

    def test_session_recovers_once_the_addition_is_removed(self, tmp_path):
        from swival.input_dispatch import ParsedInput

        result = _assemble(tmp_path, context_length=32_768)
        ctx = _context(tmp_path, result)
        ctx.loop_kwargs["context_length"] = 32_768
        agent._repl_remember("y" * 60_000, ctx)
        assert ctx.pending_instruction_failure is not None

        blocked = agent._revalidate_pending_instructions(ctx)
        assert blocked is not None
        assert blocked.is_error

        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- small rule\n", encoding="utf-8"
        )
        assert agent._revalidate_pending_instructions(ctx) is None
        assert ctx.pending_instruction_failure is None
        assert "- small rule" in _instruction_block(ctx.messages[0])
        assert ParsedInput("hello").is_plain_text


# ---------------------------------------------------------------------------
# Library failure contract
# ---------------------------------------------------------------------------


def _fake_provider(monkeypatch, context_length=32_768):
    monkeypatch.setattr(
        agent, "discover_model", lambda *a: ("test-model", context_length)
    )


def _answering_llm(*args, **kwargs):
    return _make_message(content="the answer"), "stop"


class TestLibraryFailureContract:
    def test_construction_stays_lazy(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        Session(base_dir=str(tmp_path), history=False)  # must not raise

    def test_run_raises_before_the_first_model_call(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        called = []
        monkeypatch.setattr(
            agent, "call_llm", lambda *a, **k: called.append(1) or _answering_llm()
        )
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError) as excinfo:
            session.run("hello")
        assert called == []
        assert str(tmp_path / "CLAUDE.md") in excinfo.value.sources
        assert excinfo.value.breakdown["instructions"] > 0

    def test_error_is_a_config_error(self, tmp_path):
        assert issubclass(swival.InstructionLoadError, swival.ConfigError)

    def test_report_travels_on_the_exception(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError) as excinfo:
            session.run("hello", report=True)
        report = excinfo.value.report
        assert report["result"]["outcome"] == "error"
        assert report["result"]["exit_code"] == 1
        assert report["settings"]["instructions_status"] == "failed"
        assert report["settings"]["instructions_loaded"] == []

    def test_ask_leaves_the_transcript_usable(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError):
            session.ask("hello")
        assert session._conv_state is None

    def test_a_corrected_file_is_re_read_on_the_same_object(
        self, tmp_path, monkeypatch
    ):
        _fake_provider(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError):
            session.run("hello")
        (tmp_path / "CLAUDE.md").write_text("- Short rule.", encoding="utf-8")
        result = session.run("hello")
        assert result.answer == "the answer"
        assert "- Short rule." in result.messages[0]["content"]

    def test_override_admits_it_in_the_library_too(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False, instructions_full=True)
        result = session.run("hello")
        assert "Rule 1999" in result.messages[0]["content"]

    def test_no_instructions_wins_over_the_override(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")
        session = Session(
            base_dir=str(tmp_path),
            history=False,
            no_instructions=True,
            instructions_full=True,
        )
        result = session.run("hello")
        assert "Rule 1999" not in result.messages[0]["content"]


class TestSingleAdmissionEvaluation:
    def test_exactly_one_evaluation_per_request_and_it_sees_memory(
        self, tmp_path, monkeypatch
    ):
        _fake_provider(monkeypatch, context_length=128_000)
        sent = {}

        def capture(*args, **kwargs):
            sent["messages"] = [dict(m) for m in args[2]]
            return _answering_llm()

        monkeypatch.setattr(agent, "call_llm", capture)
        (tmp_path / "CLAUDE.md").write_text("- Project rule.", encoding="utf-8")
        memory_dir = tmp_path / ".swival" / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text(
            "- The build script lives in scripts/build.sh\n", encoding="utf-8"
        )

        calls = []
        real_admit = instr.admit

        def counting_admit(*args, **kwargs):
            calls.append(kwargs.get("fixed_cost"))
            return real_admit(*args, **kwargs)

        monkeypatch.setattr(instr, "admit", counting_admit)

        session = Session(base_dir=str(tmp_path), history=False)
        session.run("where is the build script?")

        assert len(calls) == 1
        system = sent["messages"][0]["content"]
        assert "- Project rule." in system
        assert "scripts/build.sh" in system
        # Memory is part of what the decision was measured against.
        assert calls[0] >= instr.count_tokens("scripts/build.sh")


class TestReports:
    def test_cli_report_settings_carry_the_loading_status(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- Project rule.", encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False)
        result = session.run("hello", report=True)
        settings = result.report["settings"]
        assert settings["instructions_status"] == "loaded"
        assert settings["instructions_loaded"] == [str(tmp_path / "CLAUDE.md")]
        assert settings["instructions_full"] is False

    def test_disabled_loading_is_distinguishable_from_failure(
        self, tmp_path, monkeypatch
    ):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- Project rule.", encoding="utf-8")
        session = Session(base_dir=str(tmp_path), history=False, no_instructions=True)
        result = session.run("hello", report=True)
        assert result.report["settings"]["instructions_status"] == "disabled"
        assert result.report["settings"]["instructions_loaded"] == []


class TestAdapterErrors:
    def test_a2a_uses_the_explanation_not_a_crash_label(self):
        from swival.a2a_server import _failure_text

        exc = swival.InstructionLoadError("The instruction files are too large.")
        assert _failure_text(exc) == "The instruction files are too large."
        assert _failure_text(RuntimeError("boom")).startswith("Internal error")


# ---------------------------------------------------------------------------
# Recovery that has to shrink around the instructions
# ---------------------------------------------------------------------------


class TestSpanPreservingRecovery:
    def test_a_request_that_only_fits_after_the_rest_shrinks(
        self, tmp_path, monkeypatch
    ):
        from swival.report import ContextOverflowError
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        (tmp_path / "CLAUDE.md").write_text(
            "- Always run `make check` before a commit.\n- Never force-push.\n",
            encoding="utf-8",
        )
        prompt = _assemble(tmp_path, context_length=8192, max_output_tokens=1024)
        assert prompt.admission.admitted
        block = _instruction_block(_system_message(prompt))

        seen = []

        def fake_call_llm(*args, **_kw):
            messages = args[2]
            size = sum(len(m.get("content") or "") for m in messages)
            seen.append(size)
            if size > 4000:
                raise ContextOverflowError("prompt too long")
            return _make_message(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        messages = [_system_message(prompt), {"role": "user", "content": "go"}]
        answer, _exhausted = agent.run_agent_loop(
            messages,
            None,
            api_base="http://127.0.0.1:1234",
            model_id="test-model",
            max_turns=2,
            max_output_tokens=1024,
            temperature=0.5,
            top_p=None,
            seed=None,
            context_length=8192,
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
        assert answer == "done"
        assert seen[-1] < seen[0]
        assert "- Always run `make check` before a commit." in messages[0]["content"]
        assert "- Never force-push." in messages[0]["content"]
        assert block.strip() in messages[0]["content"]


class TestDominantComponents:
    def test_tool_schemas_are_blamed_not_a_small_agents_file(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n- test: `pytest`\n", encoding="utf-8"
        )
        fat_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "A very long tool description. " * 60,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(40)
        ]
        result = _assemble(tmp_path, context_length=8192, tools=fat_tools)
        assert not result.admission.admitted
        text = instr.failure_diagnostic(result.admission)
        assert "Tool schemas" in text
        assert result.admission.breakdown["tools"] > result.admission.cost


# ---------------------------------------------------------------------------
# Flag and config precedence
# ---------------------------------------------------------------------------


class TestOverridePrecedence:
    def test_config_key_is_recognised(self):
        from swival.config import CONFIG_KEYS, _ARGPARSE_DEFAULTS

        assert CONFIG_KEYS["instructions_full"] is bool
        assert _ARGPARSE_DEFAULTS["instructions_full"] is False

    def test_config_key_reaches_session_kwargs(self):
        from swival.config import config_to_session_kwargs

        kwargs = config_to_session_kwargs({"instructions_full": True})
        assert kwargs["instructions_full"] is True

    def test_config_template_keeps_the_precedence_sentence(self):
        from swival.config import generate_config

        text = generate_config()
        assert "instructions_full" in text
        assert (
            "An explicit --no-instructions wins over an inherited "
            "full-loading setting" in text
        )

    def test_help_keeps_the_precedence_sentence(self):
        from swival.agent import build_parser

        parser = build_parser()
        help_texts = [
            action.help or ""
            for action in parser._actions
            if "--instructions-full" in (action.option_strings or [])
        ]
        assert help_texts
        assert (
            "An explicit --no-instructions wins over an inherited "
            "full-loading setting; reject contradictory explicit CLI flags."
            in help_texts[0]
        )

    def test_cli_flag_wins_over_project_config(self, tmp_path, monkeypatch):
        from swival.agent import build_parser
        from swival.config import apply_config_to_args

        parser = build_parser()
        args = parser.parse_args(["question"])
        apply_config_to_args(args, {"instructions_full": True})
        assert args.instructions_full is True

    def test_contradictory_cli_flags_are_rejected(self, tmp_path, monkeypatch):
        """Both flags typed together is a contradiction main() refuses."""
        from unittest.mock import MagicMock, patch

        from swival.agent import build_parser

        monkeypatch.chdir(tmp_path)
        args = build_parser().parse_args(
            ["--no-instructions", "--instructions-full", "q"]
        )
        assert args.no_instructions is True
        assert args.instructions_full is True

        parser = MagicMock()
        parser.parse_args.return_value = args
        parser.error.side_effect = SystemExit(2)
        with patch.object(agent, "build_parser", return_value=parser):
            with pytest.raises(SystemExit):
                agent.main()

        parser.error.assert_called_once()
        message = parser.error.call_args[0][0]
        assert "--no-instructions" in message
        assert "--instructions-full" in message


class TestOverrideOwnsItsContextFailures:
    def _run(self, tmp_path, monkeypatch, **overrides):
        from swival.report import ContextOverflowError
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        def always_rejects(*_args, **_kw):
            raise ContextOverflowError("prompt is too long")

        monkeypatch.setattr(agent, "call_llm", always_rejects)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 5000},
        ]
        kwargs = dict(
            api_base="http://127.0.0.1:1234",
            model_id="test-model",
            max_turns=1,
            max_output_tokens=1024,
            temperature=0.5,
            top_p=None,
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
            continue_here=False,
        )
        kwargs.update(overrides)
        return agent.run_agent_loop(messages, None, **kwargs)

    def test_the_override_is_named_when_it_is_on(self, tmp_path, monkeypatch):
        answer, _exhausted = self._run(tmp_path, monkeypatch, instructions_full=True)
        assert "--instructions-full" in answer
        assert "kept intact" in answer

    def test_a_normal_session_keeps_the_plain_notice(self, tmp_path, monkeypatch):
        answer, _exhausted = self._run(tmp_path, monkeypatch)
        assert "--instructions-full" not in answer

    def test_an_auth_failure_keeps_its_own_cause(self, tmp_path, monkeypatch):
        from swival.report import AgentError
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        def rejects_auth(*_args, **_kw):
            raise AgentError("invalid api key")

        monkeypatch.setattr(agent, "call_llm", rejects_auth)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        with pytest.raises(AgentError, match="invalid api key"):
            agent.run_agent_loop(
                messages,
                None,
                api_base="http://127.0.0.1:1234",
                model_id="test-model",
                max_turns=1,
                max_output_tokens=1024,
                temperature=0.5,
                top_p=None,
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
                continue_here=False,
                instructions_full=True,
            )


# ---------------------------------------------------------------------------
# A set that could not be read whole is never admitted
# ---------------------------------------------------------------------------


def _oversize_agents(tmp_path, monkeypatch, cap=64):
    monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", cap)
    (tmp_path / "CLAUDE.md").write_text("- Small project rule.", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("- Big rule. " * 200, encoding="utf-8")


class TestIncompleteSetIsRejected:
    def test_a_read_error_refuses_the_whole_set(self, tmp_path, monkeypatch):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path)
        assert not result.admission.admitted
        assert not result.admission.complete
        assert result.instructions_loaded == []
        assert agent._instruction_status(result) == "failed"

    def test_the_override_does_not_admit_a_partial_set(self, tmp_path, monkeypatch):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path, instructions_full=True)
        assert not result.admission.admitted
        assert not result.admission.forced

    def test_an_unknown_window_does_not_admit_a_partial_set(
        self, tmp_path, monkeypatch
    ):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path, context_length=None)
        assert not result.admission.admitted

    def test_the_message_says_it_could_not_be_read(self, tmp_path, monkeypatch):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path)
        text = instr.failure_diagnostic(result.admission)
        assert "could not be read" in text
        assert str(tmp_path / "AGENTS.md") in text
        assert "too large for this setup" not in text

    def test_the_library_raises_before_any_model_call(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        calls = []
        monkeypatch.setattr(
            agent, "call_llm", lambda *a, **k: calls.append(1) or _answering_llm()
        )
        _oversize_agents(tmp_path, monkeypatch)
        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError, match="could not be read") as e:
            session.run("hello")
        assert calls == []
        assert any("AGENTS.md" in err for err in e.value.read_errors)

    def test_refresh_keeps_the_previous_rules(self, tmp_path, monkeypatch):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- original rule\n", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        ctx = _context(tmp_path, result)
        before = _instruction_block(ctx.messages[0])
        assert "- original rule" in before

        monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", 8)
        refresh = agent.refresh_system_instructions(ctx)
        assert not refresh.ok
        assert refresh.blocking is True
        assert _instruction_block(ctx.messages[0]) == before


# ---------------------------------------------------------------------------
# A saved-but-inactive update holds back every path to the model
# ---------------------------------------------------------------------------


def _block_with_remember(tmp_path, context_length=32_768):
    result = _assemble(tmp_path, context_length=context_length)
    ctx = _context(tmp_path, result)
    ctx.loop_kwargs["context_length"] = context_length
    _msg, is_error = agent._repl_remember("z" * 60_000, ctx)
    assert is_error is True
    assert ctx.pending_instruction_failure is not None
    return ctx


def _parse(text):
    from swival.input_dispatch import parse_input_line

    return parse_input_line(text)


class TestPendingUpdateGatesEveryModelPath:
    def test_plain_text_is_blocked(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("what next?"), ctx, mode="repl")
        assert step.is_error
        assert "still not active" in step.text

    def test_an_agent_turn_command_is_blocked(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("/learn"), ctx, mode="repl")
        assert step.is_error
        assert "still not active" in step.text

    def test_init_is_blocked_too(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("/init"), ctx, mode="repl")
        assert step.is_error

    def test_a_custom_bang_command_is_blocked(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        commands_dir = Path(ctx.base_dir) / ".swival" / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)
        (commands_dir / "note.md").write_text("Summarise the repo.\n", encoding="utf-8")
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("!note"), ctx, mode="repl")
        assert step.is_error
        assert "still not active" in step.text

    def test_local_commands_stay_available(self, tmp_path):
        ctx = _block_with_remember(tmp_path)
        step = agent.execute_input(_parse("/help"), ctx, mode="repl")
        assert not step.is_error
        assert "/remember" in step.text

    def test_removing_the_addition_unblocks_every_path(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- tiny rule\n", encoding="utf-8"
        )
        seen = []
        monkeypatch.setattr(
            agent,
            "run_agent_loop",
            lambda *a, **k: seen.append(1) or ("ran", False),
        )
        step = agent.execute_input(_parse("go on"), ctx, mode="repl")
        assert step.text == "ran"
        assert seen == [1]
        assert ctx.pending_instruction_failure is None
        assert "- tiny rule" in _instruction_block(ctx.messages[0])


# ---------------------------------------------------------------------------
# The pending state travels with the conversation, not with one command call
# ---------------------------------------------------------------------------


class TestSessionCarriesPendingState:
    def _session(self, tmp_path, monkeypatch, calls):
        _fake_provider(monkeypatch, context_length=32_768)

        def counting(*args, **kwargs):
            calls.append(args[2])
            return _answering_llm()

        monkeypatch.setattr(agent, "call_llm", counting)
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- small rule\n", encoding="utf-8"
        )
        return Session(base_dir=str(tmp_path), history=False)

    def test_a_blocked_remember_stops_the_next_ask(self, tmp_path, monkeypatch):
        calls = []
        session = self._session(tmp_path, monkeypatch, calls)
        session.ask("hello")
        assert len(calls) == 1

        result = session.ask("/remember " + "q" * 60_000, parse_commands=True)
        assert "Saved, but not active in this session" in result.answer

        with pytest.raises(swival.InstructionLoadError, match="still not active"):
            session.ask("carry on")
        assert len(calls) == 1

    def test_the_session_recovers_once_the_file_is_corrected(
        self, tmp_path, monkeypatch
    ):
        calls = []
        session = self._session(tmp_path, monkeypatch, calls)
        session.ask("/remember " + "q" * 60_000, parse_commands=True)
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- small rule\n- corrected\n",
            encoding="utf-8",
        )
        result = session.ask("carry on")
        assert result.answer == "the answer"
        assert len(calls) == 1
        assert "- corrected" in calls[0][0]["content"]

    def test_local_commands_still_work_while_blocked(self, tmp_path, monkeypatch):
        calls = []
        session = self._session(tmp_path, monkeypatch, calls)
        session.ask("/remember " + "q" * 60_000, parse_commands=True)
        result = session.ask("/help", parse_commands=True)
        assert "/remember" in result.answer
        assert calls == []


# ---------------------------------------------------------------------------
# A refused startup keeps the previous session's continuation
# ---------------------------------------------------------------------------


class TestContinuationSurvivesRejection:
    def _continue_file(self, tmp_path):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Continue\n\nFinish the parser rewrite", encoding="utf-8")
        return path

    def test_a_rejected_run_leaves_the_file_in_place(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        calls = []
        monkeypatch.setattr(
            agent, "call_llm", lambda *a, **k: calls.append(1) or _answering_llm()
        )
        path = self._continue_file(tmp_path)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")

        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError):
            session.run("hello")
        assert calls == []
        assert path.exists()

    def test_the_corrected_retry_still_gets_the_continuation(
        self, tmp_path, monkeypatch
    ):
        _fake_provider(monkeypatch)
        sent = []

        def capture(*args, **kwargs):
            sent.append(args[2][0]["content"])
            return _answering_llm()

        monkeypatch.setattr(agent, "call_llm", capture)
        path = self._continue_file(tmp_path)
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")

        session = Session(base_dir=str(tmp_path), history=False)
        with pytest.raises(swival.InstructionLoadError):
            session.run("hello")

        (tmp_path / "CLAUDE.md").write_text("- Short rule.", encoding="utf-8")
        session.run("hello")
        assert "Finish the parser rewrite" in sent[0]
        assert not path.exists()

    def test_an_accepted_run_consumes_it_once(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch)
        sent = []

        def capture(*args, **kwargs):
            sent.append(args[2][0]["content"])
            return _answering_llm()

        monkeypatch.setattr(agent, "call_llm", capture)
        path = self._continue_file(tmp_path)

        session = Session(base_dir=str(tmp_path), history=False)
        session.run("first")
        assert not path.exists()
        session.run("second")
        assert "Finish the parser rewrite" in sent[0]
        assert "Finish the parser rewrite" not in sent[1]


# ---------------------------------------------------------------------------
# /goal is a state-change command that launches a turn
# ---------------------------------------------------------------------------


class TestGoalLaunchDetection:
    """The caller's launch test must agree with what _repl_goal does."""

    @pytest.mark.parametrize(
        "arg", ["", "   ", "clear", "remove", "drop", "pause", "resume", "PAUSE"]
    )
    def test_local_forms_do_not_launch(self, arg):
        from swival.goal import GoalState

        assert agent._goal_arg_launches(arg) is False
        state = GoalState(verbose=False)
        state.create("an existing objective")
        result = agent._repl_goal(arg, state, oneshot_mode=False)
        assert result.should_start_loop is False

    @pytest.mark.parametrize(
        "arg,existing",
        [("ship the parser", False), ("replace ship the parser", True)],
    )
    def test_objective_forms_launch(self, arg, existing):
        from swival.goal import GoalState

        assert agent._goal_arg_launches(arg) is True
        state = GoalState(verbose=False)
        if existing:
            state.create("an existing objective")
        result = agent._repl_goal(arg, state, oneshot_mode=False)
        assert result.should_start_loop is True


class TestGoalRespectsPendingUpdate:
    def _goal_ctx(self, tmp_path):
        from swival.goal import GoalState

        ctx = _block_with_remember(tmp_path)
        ctx.goal_state = GoalState(verbose=False)
        return ctx

    def test_setting_a_goal_is_blocked(self, tmp_path, monkeypatch):
        ctx = self._goal_ctx(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("/goal Say hello"), ctx, mode="repl")
        assert step.is_error
        assert "still not active" in step.text
        assert ctx.pending_instruction_failure is not None

    def test_a_blocked_launch_records_no_goal(self, tmp_path, monkeypatch):
        ctx = self._goal_ctx(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        agent.execute_input(_parse("/goal Say hello"), ctx, mode="repl")
        assert ctx.goal_state.get() is None

    def test_replace_is_blocked_too(self, tmp_path, monkeypatch):
        ctx = self._goal_ctx(tmp_path)
        ctx.goal_state.create("the original objective")
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent.execute_input(_parse("/goal replace something"), ctx, mode="repl")
        assert step.is_error
        assert ctx.goal_state.get().objective == "the original objective"

    def test_status_pause_and_clear_stay_available(self, tmp_path, monkeypatch):
        ctx = self._goal_ctx(tmp_path)
        ctx.goal_state.create("the original objective")
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)

        status = agent.execute_input(_parse("/goal"), ctx, mode="repl")
        assert not status.is_error

        paused = agent.execute_input(_parse("/goal pause"), ctx, mode="repl")
        assert paused.text == "goal paused"

        cleared = agent.execute_input(_parse("/goal clear"), ctx, mode="repl")
        assert cleared.text == "goal cleared"

    def test_a_goal_runs_once_the_file_is_corrected(self, tmp_path, monkeypatch):
        ctx = self._goal_ctx(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- tiny rule\n", encoding="utf-8"
        )
        seen = []
        monkeypatch.setattr(
            agent,
            "run_agent_loop",
            lambda *a, **k: seen.append(1) or ("done", False),
        )
        step = agent.execute_input(_parse("/goal Say hello"), ctx, mode="repl")
        assert step.text == "done"
        assert seen == [1]
        assert ctx.pending_instruction_failure is None


class TestInvokeAgentTurnIsTheBackstop:
    """The check sits where every turn on the shared machinery passes.

    A command added later cannot reach the model by not knowing about it, and
    no gating decision reads the command registry any more.
    """

    def test_it_refuses_whatever_reaches_it(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        with pytest.raises(agent._InstructionsPending) as raised:
            agent._invoke_agent_turn("anything", ctx)
        assert "still not active" in raised.value.step.text

    def test_run_agent_step_reports_it_as_a_step(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)
        step = agent._run_agent_step("anything", "label", ctx)
        assert step.is_error
        assert "still not active" in step.text

    def test_no_gate_reads_the_command_registry(self):
        import inspect

        assert 'kind == "agent_turn"' not in inspect.getsource(agent.execute_input)


# ---------------------------------------------------------------------------
# A file we cannot open is not a file we half-read
# ---------------------------------------------------------------------------


def _deny_open(monkeypatch, name):
    real_open = Path.open

    def bad_open(self, *args, **kwargs):
        if self.name == name:
            raise OSError("Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.open", bad_open)


class TestUnreadableFileDoesNotBlockTheRun:
    """Permissions and failing disks are the environment, not the rules.

    An oversized file means the user wrote more than we hold, and running on
    the part we read would be running on half a rule set. A file we cannot
    open says nothing about its contents, and refusing to start over it would
    refuse to start over a permission bit.
    """

    def test_it_is_reported_but_not_fatal(self, tmp_path, monkeypatch):
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n- test: `pytest`\n", encoding="utf-8"
        )
        _deny_open(monkeypatch, "CLAUDE.md")

        result = _assemble(tmp_path)
        assert result.admission.admitted
        assert result.admission.complete
        assert agent._instruction_status(result) == "loaded"
        assert "- test: `pytest`" in result.content
        assert result.instructions_loaded == [str(tmp_path / "AGENTS.md")]

    def test_the_run_reaches_the_model(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")
        _deny_open(monkeypatch, "CLAUDE.md")

        session = Session(base_dir=str(tmp_path), history=False)
        result = session.run("hello")
        assert result.answer == "the answer"
        assert "- project rule" in result.messages[0]["content"]

    def test_an_oversized_file_still_stops_the_run(self, tmp_path, monkeypatch):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path)
        assert not result.admission.admitted

    def test_a_refresh_is_not_blocked_by_an_unreadable_file(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- original rule\n", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        ctx = _context(tmp_path, result)
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        _deny_open(monkeypatch, "CLAUDE.md")

        msg, is_error = agent._repl_remember("a new rule", ctx)
        assert is_error is False
        assert "not active" not in msg
        assert ctx.pending_instruction_failure is None
        assert "a new rule" in _instruction_block(ctx.messages[0])

    def test_a_refresh_is_still_blocked_by_a_half_read_file(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- original rule\n", encoding="utf-8"
        )
        result = _assemble(tmp_path)
        ctx = _context(tmp_path, result)
        before = _instruction_block(ctx.messages[0])

        monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", 8)
        refresh = agent.refresh_system_instructions(ctx)
        assert not refresh.ok
        assert refresh.blocking is True
        assert _instruction_block(ctx.messages[0]) == before

    def test_the_override_cannot_admit_a_half_read_file(self, tmp_path, monkeypatch):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path, instructions_full=True)
        assert not result.admission.admitted


class TestTheOverrideAlwaysSaysSo:
    """Full loading warns once, whatever else was going to let the set in."""

    def _big(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(_rules(2000), encoding="utf-8")

    def test_a_known_window_reports_it_as_forced(self, tmp_path):
        self._big(tmp_path)
        result = _assemble(tmp_path, context_length=32_768, instructions_full=True)
        assert result.admission.forced
        assert agent._instruction_status(result) == "forced"

    def test_an_unknown_window_reports_it_as_forced_too(self, tmp_path):
        """Advisory would have admitted it, but the user asked for this."""
        self._big(tmp_path)
        result = _assemble(tmp_path, context_length=None, instructions_full=True)
        assert result.admission.advisory
        assert result.admission.forced
        assert agent._instruction_status(result) == "forced"

    def test_it_warns_even_when_the_session_is_advisory(self, tmp_path, capsys):
        self._big(tmp_path)
        result = _assemble(tmp_path, context_length=None, instructions_full=True)
        agent._warn_instruction_admission(
            result, verbose=False, flag="--instructions-full"
        )
        assert "--instructions-full" in capsys.readouterr().err

    def test_a_set_that_fits_says_nothing(self, tmp_path, capsys):
        (tmp_path / "CLAUDE.md").write_text("- Small rule.", encoding="utf-8")
        result = _assemble(tmp_path, instructions_full=True)
        agent._warn_instruction_admission(
            result, verbose=False, flag="--instructions-full"
        )
        assert not result.admission.forced
        assert capsys.readouterr().err == ""


class TestExplicitFlagsBeatInheritedSettings:
    def _merged(self, argv, config):
        """Run the real merge and the real precedence rule, in that order."""
        from swival.agent import build_parser
        from swival.config import apply_config_to_args

        args = build_parser().parse_args(argv)
        no_cli = args.no_instructions is True
        full_cli = args.instructions_full is True
        apply_config_to_args(args, config)
        agent._resolve_instruction_flags(
            args, no_instructions_cli=no_cli, instructions_full_cli=full_cli
        )
        return args

    def test_explicit_no_instructions_beats_inherited_full_loading(self):
        args = self._merged(["--no-instructions", "q"], {"instructions_full": True})
        assert args.no_instructions is True
        assert args.instructions_full is False

    def test_explicit_full_loading_beats_inherited_opt_out(self):
        args = self._merged(["--instructions-full", "q"], {"no_instructions": True})
        assert args.instructions_full is True
        assert args.no_instructions is False

    def test_neither_flag_leaves_the_config_alone(self):
        args = self._merged(["q"], {"no_instructions": True})
        assert args.no_instructions is True


class TestTheReportRecordsWhatWasDropped:
    """stderr is gone by the time anyone reads a report."""

    def test_a_skipped_file_appears_in_the_report(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")
        _deny_open(monkeypatch, "CLAUDE.md")

        session = Session(base_dir=str(tmp_path), history=False)
        settings = session.run("hello", report=True).report["settings"]

        assert settings["instructions_loaded"] == [str(tmp_path / "AGENTS.md")]
        assert len(settings["instructions_skipped"]) == 1
        assert str(tmp_path / "CLAUDE.md") in settings["instructions_skipped"][0]

    def test_it_says_why_the_file_was_dropped(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        _deny_open(monkeypatch, "CLAUDE.md")

        session = Session(base_dir=str(tmp_path), history=False)
        settings = session.run("hello", report=True).report["settings"]
        assert "Permission denied" in settings["instructions_skipped"][0]

    def test_the_status_still_describes_the_admission(self, tmp_path, monkeypatch):
        """A file we could not open did not change the loading decision."""
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")
        _deny_open(monkeypatch, "CLAUDE.md")

        session = Session(base_dir=str(tmp_path), history=False)
        settings = session.run("hello", report=True).report["settings"]
        assert settings["instructions_status"] == "loaded"

    def test_an_ordinary_run_records_nothing_dropped(self, tmp_path, monkeypatch):
        _fake_provider(monkeypatch, context_length=128_000)
        monkeypatch.setattr(agent, "call_llm", _answering_llm)
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")

        session = Session(base_dir=str(tmp_path), history=False)
        settings = session.run("hello", report=True).report["settings"]
        assert settings["instructions_skipped"] == []


# ---------------------------------------------------------------------------
# Every file we found is accounted for
# ---------------------------------------------------------------------------


def _deny_paths(monkeypatch, denied, *, on="open"):
    """Make specific paths fail, by full path rather than by filename.

    The stat denial goes in at ``os.stat``, below pathlib, so it reaches the
    loader whichever call ``Path.is_file`` makes on a given Python release.
    """
    denied = {str(p) for p in denied}
    real_open = Path.open
    real_stat = os.stat

    def bad_open(self, *args, **kwargs):
        if str(self) in denied:
            raise OSError("Permission denied")
        return real_open(self, *args, **kwargs)

    def bad_stat(path, *args, **kwargs):
        if str(path) in denied:
            raise OSError("Input/output error")
        return real_stat(path, *args, **kwargs)

    if on == "open":
        monkeypatch.setattr("pathlib.Path.open", bad_open)
    else:
        monkeypatch.setattr("os.stat", bad_stat)


def _mentions(messages, path):
    """True when *path* is named in one of the reported messages.

    The messages are built as ``f"{path}: {reason}"``, so containment is exact
    rather than a guess at their shape.
    """
    return [m for m in messages if str(path) in m]


class TestEveryDiscoveredFileIsAccountedFor:
    """`sources`, `errors` and `skipped` partition what discovery found.

    This is the property the recorded decision was really about, and it holds
    across both layers at once: routing a new kind of read failure into none of
    the three, or into two, fails here without any test restating the routing
    rule.
    """

    def _layout(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "CLAUDE.md").write_text("- small", encoding="utf-8")
        (config_dir / "AGENTS.md").write_text("- personal", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("P" * 500, encoding="utf-8")
        (sub / "AGENTS.md").write_text("- nested", encoding="utf-8")
        return config_dir, sub

    def _check(self, tmp_path, config_dir, sub):
        found = instr.discover(str(tmp_path), config_dir, start_dir=sub)
        loaded = instr.load(str(tmp_path), config_dir, start_dir=sub)
        placed = set()
        for path, _scope, _kind in found:
            homes = [
                name
                for name, hit in (
                    ("sources", str(path) in loaded.paths),
                    ("errors", bool(_mentions(loaded.errors, path))),
                    ("skipped", bool(_mentions(loaded.skipped, path))),
                )
                if hit
            ]
            assert homes, f"{path} was found and then went nowhere"
            assert len(homes) == 1, f"{path} landed in {homes}"
            placed.add(str(path))
        assert placed == {str(p) for p, _s, _k in found}
        assert len(found) == len(loaded.sources) + len(loaded.errors) + len(
            loaded.skipped
        )
        return loaded

    def test_when_every_file_reads(self, tmp_path):
        config_dir, sub = self._layout(tmp_path)
        loaded = self._check(tmp_path, config_dir, sub)
        assert len(loaded.sources) == 4

    def test_when_one_cannot_be_opened(self, tmp_path, monkeypatch):
        config_dir, sub = self._layout(tmp_path)
        _deny_paths(monkeypatch, [config_dir / "AGENTS.md"])
        loaded = self._check(tmp_path, config_dir, sub)
        assert len(loaded.skipped) == 1
        assert loaded.errors == []

    def test_when_one_cannot_be_stat_ed(self, tmp_path, monkeypatch):
        config_dir, sub = self._layout(tmp_path)
        _deny_paths(monkeypatch, [sub / "AGENTS.md"], on="stat")
        loaded = self._check(tmp_path, config_dir, sub)
        assert len(loaded.skipped) == 1
        assert loaded.errors == []

    def test_when_one_is_over_the_cap(self, tmp_path, monkeypatch):
        config_dir, sub = self._layout(tmp_path)
        monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", 64)
        loaded = self._check(tmp_path, config_dir, sub)
        assert len(loaded.errors) == 1
        assert loaded.skipped == []

    def test_when_both_kinds_happen_at_once(self, tmp_path, monkeypatch):
        config_dir, sub = self._layout(tmp_path)
        monkeypatch.setattr(instr, "MAX_INSTRUCTION_FILE_BYTES", 64)
        _deny_paths(monkeypatch, [config_dir / "AGENTS.md"])
        loaded = self._check(tmp_path, config_dir, sub)
        assert len(loaded.errors) == 1
        assert len(loaded.skipped) == 1

    def test_when_the_combined_cap_runs_out(self, tmp_path, monkeypatch):
        config_dir, sub = self._layout(tmp_path)
        monkeypatch.setattr(instr, "MAX_INSTRUCTION_TOTAL_BYTES", 20)
        loaded = self._check(tmp_path, config_dir, sub)
        assert loaded.errors


class TestTheReportAccountsForEveryFileToo:
    """The same partition, one layer up, where the report is written."""

    def test_loaded_and_skipped_cover_discovery(self, tmp_path, monkeypatch):
        (tmp_path / "CLAUDE.md").write_text("- unreachable", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("- project rule\n", encoding="utf-8")
        _deny_paths(monkeypatch, [tmp_path / "CLAUDE.md"])

        result = _assemble(tmp_path)
        assert agent._instruction_status(result) != "failed"
        found = {str(p) for p, _s, _k in instr.discover(str(tmp_path))}
        accounted = set(result.instructions_loaded)
        for path in found - accounted:
            assert _mentions(result.instructions_skipped, path)
        assert accounted <= found

    def test_a_failed_status_is_the_only_way_to_lose_a_file(
        self, tmp_path, monkeypatch
    ):
        _oversize_agents(tmp_path, monkeypatch)
        result = _assemble(tmp_path)
        # Nothing is claimed as loaded, and the run stops instead.
        assert agent._instruction_status(result) == "failed"
        assert result.instructions_loaded == []


class TestCommandsThatRunTheirOwnLoop:
    """Two commands do not reach the model through the shared machinery."""

    def test_init_is_blocked_before_it_clears_the_conversation(
        self, tmp_path, monkeypatch
    ):
        ctx = _block_with_remember(tmp_path)
        ctx.messages.append({"role": "user", "content": "earlier work"})
        monkeypatch.setattr(agent, "run_agent_loop", _never_called)

        step = agent.execute_input(_parse("/init"), ctx, mode="repl")
        assert step.is_error
        assert "still not active" in step.text
        # The conversation is intact: /init clears it, and it never ran.
        assert any(m.get("content") == "earlier work" for m in ctx.messages)

    def test_audit_is_blocked_before_it_starts(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(
            agent, "_execute_delegated_command", lambda *a, **k: called.append(1)
        )
        ctx = _block_with_remember(tmp_path)
        step = agent.execute_input(_parse("/audit"), ctx, mode="repl")
        assert step.is_error
        assert called == []

    def test_both_run_once_the_update_is_active(self, tmp_path, monkeypatch):
        ctx = _block_with_remember(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "## Workflow\n\n## Conventions\n\n- tiny rule\n", encoding="utf-8"
        )
        called = []
        monkeypatch.setattr(
            agent,
            "_execute_delegated_command",
            lambda *a, **k: called.append(1) or agent.StepResult(kind="agent_turn"),
        )
        agent.execute_input(_parse("/audit"), ctx, mode="repl")
        assert called == [1]
        assert ctx.pending_instruction_failure is None
