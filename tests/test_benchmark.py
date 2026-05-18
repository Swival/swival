"""Tests for the benchmark runner."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from swival.benchmark import (
    BenchmarkError,
    BenchmarkSpec,
    Task,
    Variant,
    aggregate_rows,
    build_swival_command,
    load_spec,
    load_tasks,
    no_op_control,
    render_markdown_summary,
    run_verifier,
    run_benchmark,
    summarize_outputs,
    write_summary_artifacts,
)


def _write_report(path, *, outcome="success", turns=2, prompt_tokens=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "task",
                "model": "m",
                "provider": "lmstudio",
                "result": {"outcome": outcome, "exit_code": 0},
                "stats": {
                    "turns": turns,
                    "llm_calls": 2,
                    "tool_calls_total": 3,
                    "tool_calls_failed": 1,
                    "tool_calls_by_name": {
                        "edit_file": {"succeeded": 1, "failed": 0, "repairs": 2},
                        "run_command": {"succeeded": 1, "failed": 1},
                    },
                    "compactions": 1,
                    "turn_drops": 0,
                    "guardrail_interventions": 1,
                    "truncated_responses": 0,
                    "total_llm_time_s": 1.5,
                    "total_tool_time_s": 0.25,
                    "tool_requests": {
                        "count": 1,
                        "items": [
                            {
                                "turn": 1,
                                "reason": "need edits",
                                "tools": ["edit_file"],
                            }
                        ],
                    },
                    "blocked_tool_calls": {
                        "count": 1,
                        "items": [
                            {
                                "turn": 2,
                                "name": "edit_file",
                                "reason": "not_in_toolset",
                            }
                        ],
                    },
                },
                "timeline": [
                    {
                        "type": "llm_call",
                        "turn": 1,
                        "prompt_tokens_est": prompt_tokens,
                    },
                    {
                        "type": "llm_call",
                        "turn": 2,
                        "prompt_tokens_est": prompt_tokens + 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_meta(path, *, variant="baseline", repeat=1, task_id="task-1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "repeat": repeat,
                "task_id": task_id,
                "returncode": 0,
                "duration_s": 3.25,
                "timed_out": False,
            }
        ),
        encoding="utf-8",
    )


def test_load_spec_requires_seed_and_pinned_model(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text('[{"id": "t1", "prompt": "hello"}]', encoding="utf-8")
    spec = tmp_path / "bench.toml"
    spec.write_text(
        """
tasks = "tasks.json"
out_dir = "out"
seed = 42
base_args = ["--provider", "lmstudio"]

[variants.baseline]
args = ["--model", "local-model"]
""",
        encoding="utf-8",
    )

    loaded = load_spec(spec)

    assert loaded.seed == 42
    assert loaded.repeat == 2
    assert loaded.tasks_path == tasks
    assert loaded.out_dir == tmp_path / "out"
    assert loaded.variants == (Variant("baseline", ("--model", "local-model")),)


def test_load_spec_rejects_unpinned_variant(tmp_path):
    (tmp_path / "tasks.json").write_text(
        '[{"id": "t1", "prompt": "hello"}]', encoding="utf-8"
    )
    spec = tmp_path / "bench.toml"
    spec.write_text(
        """
tasks = "tasks.json"
out_dir = "out"
seed = 7

[variants.baseline]
args = ["--provider", "lmstudio"]
""",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkError, match="must pin a model"):
        load_spec(spec)


def test_load_spec_rejects_reserved_args(tmp_path):
    (tmp_path / "tasks.json").write_text(
        '[{"id": "t1", "prompt": "hello"}]', encoding="utf-8"
    )
    spec = tmp_path / "bench.toml"
    spec.write_text(
        """
tasks = "tasks.json"
out_dir = "out"
seed = 7
base_args = ["--model", "m", "--report", "elsewhere.json"]

[variants.baseline]
args = []
""",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkError, match="reserved benchmark flag"):
        load_spec(spec)


def test_load_tasks_accepts_wrapped_corpus_and_sanitizes_ids(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "fix bug #1",
                        "prompt": "Fix it",
                        "base_dir": "repo",
                        "args": ["--max-turns", "3"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(path)

    assert tasks == (
        Task(
            id="fix-bug-1",
            prompt="Fix it",
            base_dir="repo",
            args=("--max-turns", "3"),
        ),
    )


def test_load_tasks_accepts_file_verifier(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "fix",
                    "prompt": "Fix it",
                    "verifier": {
                        "type": "file_equals",
                        "path": "status.txt",
                        "text": "state=fixed\n",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(path)

    assert tasks[0].verifier == {
        "type": "file_equals",
        "path": "status.txt",
        "text": "state=fixed\n",
    }


def test_load_tasks_rejects_bad_verifier(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "fix",
                    "prompt": "Fix it",
                    "verifier": {"type": "shell", "path": "x", "text": ""},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkError, match="verifier.type"):
        load_tasks(path)


def test_build_swival_command_includes_report_seed_variant_and_task_args(tmp_path):
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=123,
        repeat=2,
        base_args=("--provider", "lmstudio", "--model", "local"),
        variants=(Variant("compact", ("--temperature", "0.1")),),
    )
    task = Task("t1", "Do work", base_dir="repo", args=("--max-turns", "4"))
    report = tmp_path / "report.json"

    command = build_swival_command(spec, spec.variants[0], task, report)

    assert command[:3] == [sys.executable, "-m", "swival.agent"]
    assert command[3:7] == ["--report", str(report), "--seed", "123"]
    assert "--provider" in command
    assert "--temperature" in command
    assert "--base-dir" in command
    assert command[-3:] == ["--max-turns", "4", "Do work"]


def test_run_verifier_checks_file_relative_to_variant_base_dir(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    (base / "status.txt").write_text("state=fixed\n", encoding="utf-8")
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=1,
        base_args=("--model", "m"),
        variants=(Variant("baseline", ("--base-dir", str(base))),),
    )
    task = Task(
        "fix",
        "Fix it",
        verifier={
            "type": "file_equals",
            "path": "status.txt",
            "text": "state=fixed\n",
        },
    )

    result = run_verifier(spec, spec.variants[0], task)

    assert result == {
        "type": "file_equals",
        "path": "status.txt",
        "passed": True,
    }


def test_run_benchmark_creates_report_dir_and_keeps_agent_errors(tmp_path, monkeypatch):
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=1,
        base_args=("--model", "m"),
        variants=(Variant("baseline", ()),),
        stop_on_failure=True,
    )
    task = Task("a", "A")

    def fake_run(command, **kwargs):
        report = tmp_path / "out" / "baseline" / "run-01" / "a.json"
        assert command[command.index("--report") + 1] == str(report)
        assert report.parent.exists()
        _write_report(report, outcome="error")
        return subprocess.CompletedProcess(command, 1, stdout="out", stderr="err")

    monkeypatch.setattr("swival.benchmark.subprocess.run", fake_run)

    summary = run_benchmark(spec, (task,))

    row = summary["runs"][0]
    assert row["outcome"] == "error"
    assert row["success"] is False
    assert row["process_returncode"] == 1


def test_run_benchmark_escalates_and_reports_total_token_cost(tmp_path, monkeypatch):
    base = tmp_path / "repo"
    base.mkdir()
    (base / "status.txt").write_text("state=broken\n", encoding="utf-8")
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=1,
        base_args=("--provider", "lmstudio"),
        variants=(
            Variant(
                "code_read",
                ("--model", "m", "--tools-mode", "code-read"),
                escalation_args=("--model", "m", "--tools-mode", "full"),
                escalate_on=("tool_request", "blocked_tool_call", "verifier_failed"),
            ),
        ),
    )
    task = Task(
        "fix",
        "Fix it",
        base_dir=str(base),
        verifier={
            "type": "file_equals",
            "path": "status.txt",
            "text": "state=fixed\n",
        },
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        report = command[command.index("--report") + 1]
        if report.endswith(".escalated.json"):
            (base / "status.txt").write_text("state=fixed\n", encoding="utf-8")
            _write_report(Path(report), outcome="success", prompt_tokens=300)
        else:
            _write_report(Path(report), outcome="success", prompt_tokens=100)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("swival.benchmark.subprocess.run", fake_run)

    summary = run_benchmark(spec, (task,))

    assert len(commands) == 2
    row = summary["runs"][0]
    assert row["escalated"] is True
    assert row["escalation_reasons"] == [
        "tool_request",
        "blocked_tool_call",
        "verifier_failed",
    ]
    assert row["outcome"] == "success"
    assert row["success_strict"] is True
    assert row["primary_outcome"] == "success"
    assert row["primary_verifier_passed"] is False
    assert row["prompt_tokens_est"] == 205
    assert row["prompt_tokens_with_escalation"] == 810
    assert row["escalation_prompt_tokens_est"] == 605
    assert summary["by_variant"]["code_read"]["escalations"] == 1
    assert summary["by_variant"]["code_read"]["escalation_rate"] == 1.0
    assert summary["by_variant"]["code_read"]["prompt_tokens_est"] == 205
    assert summary["by_variant"]["code_read"]["prompt_tokens_with_escalation"] == 810


def test_aggregate_rows_counts_successes_and_metrics():
    rows = [
        {"success": True, "turns": 2, "prompt_tokens_est": 100, "duration_s": 1.1},
        {"success": False, "turns": 3, "prompt_tokens_est": 200, "duration_s": 1.2},
    ]

    summary = aggregate_rows(rows)

    assert summary["tasks"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["turns"] == 5
    assert summary["prompt_tokens_est"] == 300
    assert summary["prompt_tokens_with_escalation"] == 300
    assert summary["duration_s"] == 2.3


def test_no_op_control_reports_success_delta_and_flips():
    rows = [
        {"variant": "baseline", "repeat": 1, "task_id": "a", "success": True},
        {"variant": "baseline", "repeat": 1, "task_id": "b", "success": False},
        {"variant": "baseline", "repeat": 2, "task_id": "a", "success": False},
        {"variant": "baseline", "repeat": 2, "task_id": "b", "success": True},
    ]

    controls = no_op_control(rows, (Variant("baseline", ()),), 2)

    assert controls == [
        {
            "variant": "baseline",
            "against_repeat": 2,
            "success_delta": 0,
            "outcome_flips": ["a", "b"],
            "outcome_flip_count": 2,
        }
    ]


def test_summarize_outputs_builds_variance_and_variant_comparison(tmp_path):
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=2,
        base_args=("--model", "m"),
        variants=(Variant("baseline", ()), Variant("candidate", ())),
    )
    tasks = (Task("a", "A"), Task("b", "B"))

    cases = [
        ("baseline", 1, "a", "success"),
        ("baseline", 1, "b", "error"),
        ("baseline", 2, "a", "success"),
        ("baseline", 2, "b", "success"),
        ("candidate", 1, "a", "success"),
        ("candidate", 1, "b", "success"),
        ("candidate", 2, "a", "error"),
        ("candidate", 2, "b", "success"),
    ]
    for variant, repeat, task_id, outcome in cases:
        run_dir = tmp_path / "out" / variant / f"run-{repeat:02d}"
        _write_report(
            run_dir / f"{task_id}.json",
            outcome=outcome,
            turns=repeat + (5 if variant == "candidate" else 0),
            prompt_tokens=100 + repeat,
        )
        _write_meta(
            run_dir / f"{task_id}.meta.json",
            variant=variant,
            repeat=repeat,
            task_id=task_id,
        )

    summary = summarize_outputs(spec, tasks)

    assert summary["by_variant"]["baseline"]["successes"] == 3
    assert summary["by_variant_repeat"]["baseline"]["1"]["successes"] == 1
    assert summary["by_variant_task"]["baseline"]["b"]["successes"] == 1
    assert summary["by_variant_task"]["candidate"]["a"]["metrics"]["turns"] == {
        "min": 6,
        "mean": 6.5,
        "max": 7,
    }
    assert summary["no_op_control"][0]["success_delta"] == 1
    assert summary["no_op_control"][0]["outcome_flips"] == ["b"]
    assert summary["comparisons"] == [
        {
            "baseline": "baseline",
            "variant": "candidate",
            "success_delta": 0,
            "wins": ["b"],
            "losses": ["a"],
            "ties": [],
        }
    ]
    assert summary["per_repeat_comparisons"] == [
        {
            "baseline": "baseline",
            "variant": "candidate",
            "success_delta": 1,
            "wins": ["b"],
            "losses": [],
            "ties": ["a"],
            "repeat": 1,
        },
        {
            "baseline": "baseline",
            "variant": "candidate",
            "success_delta": -1,
            "wins": [],
            "losses": ["a"],
            "ties": ["b"],
            "repeat": 2,
        },
    ]
    assert summary["runs"][0]["prompt_tokens_est"] == 207
    assert summary["runs"][0]["tool_repairs"] == 2
    assert summary["runs"][0]["tool_request_count"] == 1
    assert summary["runs"][0]["blocked_tool_call_count"] == 1
    assert summary["runs"][0]["verifier_passed"] is None
    assert summary["runs"][0]["success_strict"] is None


def test_verifier_passed_and_success_strict_slots(tmp_path):
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=1,
        base_args=("--model", "m"),
        variants=(Variant("baseline", ()),),
    )
    run_dir = tmp_path / "out" / "baseline" / "run-01"
    report_path = run_dir / "a.json"
    _write_report(report_path, outcome="success")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["result"]["verifier_passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_meta(run_dir / "a.meta.json", variant="baseline", task_id="a")

    summary = summarize_outputs(spec, (Task("a", "A"),))

    assert summary["runs"][0]["success"] is True
    assert summary["runs"][0]["verifier_passed"] is False
    assert summary["runs"][0]["success_strict"] is False
    assert summary["by_variant"]["baseline"]["successes"] == 0


def test_missing_report_is_harness_error(tmp_path):
    spec = BenchmarkSpec(
        path=tmp_path / "bench.toml",
        out_dir=tmp_path / "out",
        tasks_path=tmp_path / "tasks.json",
        seed=1,
        repeat=1,
        base_args=("--model", "m"),
        variants=(Variant("baseline", ()),),
    )
    run_dir = tmp_path / "out" / "baseline" / "run-01"
    _write_meta(run_dir / "a.meta.json", variant="baseline", task_id="a")

    summary = summarize_outputs(spec, (Task("a", "A"),))

    assert summary["runs"][0]["outcome"] == "harness_error"
    assert summary["runs"][0]["verifier_passed"] is None
    assert summary["runs"][0]["success_strict"] is None
    assert summary["by_variant"]["baseline"]["successes"] == 0


def test_write_summary_artifacts(tmp_path):
    summary = {
        "spec": "bench.toml",
        "seed": 1,
        "repeat": 2,
        "tasks": ["a"],
        "variants": ["baseline"],
        "by_variant": {
            "baseline": {
                "tasks": 1,
                "successes": 1,
                "success_rate": 1.0,
                "turns": 2,
                "tool_calls_failed": 0,
                "prompt_tokens_est": 100,
                "duration_s": 3.0,
            }
        },
        "by_variant_task": {},
        "no_op_control": [],
        "comparisons": [],
        "per_repeat_comparisons": [],
        "runs": [
            {
                "variant": "baseline",
                "repeat": 1,
                "task_id": "a",
                "success": True,
                "verifier_passed": None,
                "success_strict": None,
                "outcome": "success",
            }
        ],
    }

    write_summary_artifacts(tmp_path, summary)

    assert (tmp_path / "summary.json").exists()
    assert (
        (tmp_path / "summary.csv")
        .read_text(encoding="utf-8")
        .startswith("variant,repeat,task_id")
    )
    assert "Swival Benchmark Summary" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "| baseline | 1 | 1 | 100.00%" in render_markdown_summary(summary)
