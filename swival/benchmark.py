"""Benchmark runner for repeatable Swival report comparisons.

The runner is intentionally small: it executes a fixed task corpus against one
or more CLI argument variants, writes Swival's existing ``--report`` JSON for
each task, and produces deterministic summary artifacts that can be diffed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUMMARY_VERSION = 2
DEFAULT_REPEAT = 2
METRIC_KEYS = (
    "turns",
    "llm_calls",
    "prompt_tokens_est",
    "tool_calls_total",
    "tool_calls_failed",
    "tool_repairs",
    "compactions",
    "turn_drops",
    "guardrail_interventions",
    "truncated_responses",
    "tool_request_count",
    "blocked_tool_call_count",
    "duration_s",
    "total_llm_time_s",
    "total_tool_time_s",
)


class BenchmarkError(Exception):
    """Raised for invalid benchmark specs or task corpora."""


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    base_dir: str | None = None
    args: tuple[str, ...] = ()
    verifier: dict[str, Any] | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSpec:
    path: Path
    out_dir: Path
    tasks_path: Path
    seed: int
    repeat: int
    base_args: tuple[str, ...]
    variants: tuple[Variant, ...]
    timeout_s: float | None = None
    stop_on_failure: bool = False


def _as_str_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise BenchmarkError(f"{field} must be a list of strings")
    return tuple(value)


def _resolve_path(base: Path, value: str, field: str) -> Path:
    if not value:
        raise BenchmarkError(f"{field} must not be empty")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _has_flag(args: tuple[str, ...], *names: str) -> bool:
    return any(a in names or any(a.startswith(n + "=") for n in names) for a in args)


def load_spec(path: str | Path) -> BenchmarkSpec:
    spec_path = Path(path).resolve()
    try:
        data = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BenchmarkError(f"cannot read spec {spec_path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise BenchmarkError(f"invalid TOML in {spec_path}: {e}") from e

    root = spec_path.parent
    tasks_raw = data.get("tasks")
    out_raw = data.get("out_dir")
    if not isinstance(tasks_raw, str):
        raise BenchmarkError("tasks must point to a JSON task corpus")
    if not isinstance(out_raw, str):
        raise BenchmarkError("out_dir is required")

    seed = data.get("seed")
    if not isinstance(seed, int):
        raise BenchmarkError("seed is required and must be an integer")

    repeat = data.get("repeat", DEFAULT_REPEAT)
    if not isinstance(repeat, int) or repeat < 1:
        raise BenchmarkError("repeat must be a positive integer")

    timeout_s: float | None
    timeout_raw = data.get("timeout_s")
    if timeout_raw is None:
        timeout_s = None
    elif isinstance(timeout_raw, int | float) and timeout_raw > 0:
        timeout_s = float(timeout_raw)
    else:
        raise BenchmarkError("timeout_s must be a positive number when set")

    base_args = _as_str_list(data.get("base_args", []), "base_args")
    _reject_reserved_args(base_args, "base_args")
    variants_raw = data.get("variants")
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise BenchmarkError("at least one [variants.NAME] table is required")

    variants: list[Variant] = []
    for name, body in variants_raw.items():
        if not isinstance(body, dict):
            raise BenchmarkError(f"variants.{name} must be a table")
        args = _as_str_list(body.get("args", []), f"variants.{name}.args")
        _reject_reserved_args(args, f"variants.{name}.args")
        combined = base_args + args
        if not _has_flag(combined, "--model", "-m", "--profile"):
            raise BenchmarkError(
                f"variants.{name} must pin a model with --model/-m or --profile"
            )
        variants.append(Variant(name=name, args=args))

    return BenchmarkSpec(
        path=spec_path,
        out_dir=_resolve_path(root, out_raw, "out_dir"),
        tasks_path=_resolve_path(root, tasks_raw, "tasks"),
        seed=seed,
        repeat=repeat,
        base_args=base_args,
        variants=tuple(variants),
        timeout_s=timeout_s,
        stop_on_failure=bool(data.get("stop_on_failure", False)),
    )


def _reject_reserved_args(args: tuple[str, ...], field: str) -> None:
    reserved = {"--report", "--seed", "--repl"}
    for arg in args:
        if arg in reserved or any(arg.startswith(flag + "=") for flag in reserved):
            raise BenchmarkError(
                f"{field} must not set reserved benchmark flag {arg!r}"
            )


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: str) -> str:
    cleaned = _SAFE_ID_RE.sub("-", value.strip()).strip("-._")
    return cleaned or "task"


def load_tasks(path: str | Path) -> tuple[Task, ...]:
    task_path = Path(path)
    try:
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BenchmarkError(f"cannot read task corpus {task_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise BenchmarkError(f"invalid JSON in task corpus {task_path}: {e}") from e

    if isinstance(data, dict):
        raw_tasks = data.get("tasks")
    else:
        raw_tasks = data
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise BenchmarkError("task corpus must be a non-empty list or {'tasks': [...]}")

    tasks: list[Task] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            raise BenchmarkError(f"task #{idx} must be an object")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BenchmarkError(f"task #{idx} requires a non-empty prompt")
        task_id = item.get("id", f"task-{idx:03d}")
        if not isinstance(task_id, str) or not task_id.strip():
            raise BenchmarkError(f"task #{idx} id must be a non-empty string")
        task_id = safe_id(task_id)
        if task_id in seen:
            raise BenchmarkError(f"duplicate task id {task_id!r}")
        seen.add(task_id)
        base_dir = item.get("base_dir")
        if base_dir is not None and not isinstance(base_dir, str):
            raise BenchmarkError(f"task {task_id}: base_dir must be a string")
        verifier = item.get("verifier")
        if verifier is not None:
            if not isinstance(verifier, dict):
                raise BenchmarkError(f"task {task_id}: verifier must be an object")
            vtype = verifier.get("type")
            if vtype not in ("file_contains", "file_equals"):
                raise BenchmarkError(
                    f"task {task_id}: verifier.type must be 'file_contains' or 'file_equals'"
                )
            if not isinstance(verifier.get("path"), str) or not verifier.get("path"):
                raise BenchmarkError(f"task {task_id}: verifier.path must be a string")
            if not isinstance(verifier.get("text"), str):
                raise BenchmarkError(f"task {task_id}: verifier.text must be a string")
        tasks.append(
            Task(
                id=task_id,
                prompt=prompt,
                base_dir=base_dir,
                args=_as_str_list(item.get("args", []), f"task {task_id}.args"),
                verifier=verifier,
            )
        )
        _reject_reserved_args(tasks[-1].args, f"task {task_id}.args")
    return tuple(tasks)


def build_swival_command(
    spec: BenchmarkSpec,
    variant: Variant,
    task: Task,
    report_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swival.agent",
        "--report",
        str(report_path),
        "--seed",
        str(spec.seed),
    ]
    command.extend(spec.base_args)
    command.extend(variant.args)
    if task.base_dir:
        command.extend(["--base-dir", task.base_dir])
    command.extend(task.args)
    command.append(task.prompt)
    return command


def _arg_value(args: tuple[str, ...], name: str) -> str | None:
    for idx, arg in enumerate(args):
        if arg == name and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _effective_base_dir(spec: BenchmarkSpec, variant: Variant, task: Task) -> Path:
    base_dir = _arg_value(spec.base_args + variant.args + task.args, "--base-dir")
    if task.base_dir:
        base_dir = task.base_dir
    if base_dir:
        path = Path(base_dir)
        if not path.is_absolute():
            path = spec.path.parent / path
        return path.resolve()
    return spec.path.parent.resolve()


def run_verifier(
    spec: BenchmarkSpec, variant: Variant, task: Task
) -> dict[str, Any] | None:
    verifier = task.verifier
    if verifier is None:
        return None
    base_dir = _effective_base_dir(spec, variant, task)
    rel_path = Path(str(verifier["path"]))
    target = (base_dir / rel_path).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError:
        return {
            "type": verifier["type"],
            "path": str(rel_path),
            "passed": False,
            "error": "verifier path escapes base_dir",
        }
    try:
        actual = target.read_text(encoding="utf-8")
    except OSError as e:
        return {
            "type": verifier["type"],
            "path": str(rel_path),
            "passed": False,
            "error": str(e),
        }

    expected = str(verifier["text"])
    if verifier["type"] == "file_contains":
        passed = expected in actual
    else:
        passed = actual == expected
    return {
        "type": verifier["type"],
        "path": str(rel_path),
        "passed": passed,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_benchmark(spec: BenchmarkSpec, tasks: tuple[Task, ...]) -> dict:
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    for variant in spec.variants:
        for repeat_idx in range(1, spec.repeat + 1):
            for task in tasks:
                run_dir = spec.out_dir / variant.name / f"run-{repeat_idx:02d}"
                report_path = run_dir / f"{task.id}.json"
                meta_path = run_dir / f"{task.id}.meta.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = build_swival_command(spec, variant, task, report_path)

                started = time.monotonic()
                timed_out = False
                stdout = ""
                stderr = ""
                returncode: int | None = None
                try:
                    proc = subprocess.run(
                        command,
                        cwd=spec.path.parent,
                        text=True,
                        capture_output=True,
                        timeout=spec.timeout_s,
                        check=False,
                    )
                    returncode = proc.returncode
                    stdout = proc.stdout[-4000:]
                    stderr = proc.stderr[-4000:]
                except subprocess.TimeoutExpired as e:
                    timed_out = True
                    returncode = None
                    stdout = _decode_timeout_stream(e.stdout)
                    stderr = _decode_timeout_stream(e.stderr)
                duration_s = round(time.monotonic() - started, 3)

                meta = {
                    "variant": variant.name,
                    "repeat": repeat_idx,
                    "task_id": task.id,
                    "command": command,
                    "returncode": returncode,
                    "duration_s": duration_s,
                    "timed_out": timed_out,
                    "report": str(report_path),
                    "stdout_tail": stdout,
                    "stderr_tail": stderr,
                }
                verifier_result = run_verifier(spec, variant, task)
                if verifier_result is not None:
                    meta["verifier"] = verifier_result
                _write_json(meta_path, meta)

                failed = timed_out or not report_path.exists()
                if failed and spec.stop_on_failure:
                    summary = summarize_outputs(spec, tasks)
                    write_summary_artifacts(spec.out_dir, summary)
                    raise BenchmarkError(
                        f"run failed for {variant.name} run {repeat_idx} task {task.id}"
                    )

    summary = summarize_outputs(spec, tasks)
    write_summary_artifacts(spec.out_dir, summary)
    return summary


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[-4000:]
    return value[-4000:]


def _report_prompt_tokens(report: dict) -> int:
    return sum(
        int(event.get("prompt_tokens_est", 0))
        for event in report.get("timeline", [])
        if event.get("type") == "llm_call"
    )


def _report_repairs(report: dict) -> int:
    total = 0
    for stats in report.get("stats", {}).get("tool_calls_by_name", {}).values():
        total += int(stats.get("repairs", 0))
    return total


def _count_stat_items(stats: dict, key: str) -> int:
    value = stats.get(key)
    if isinstance(value, dict):
        count = value.get("count")
        if isinstance(count, int):
            return count
        items = value.get("items")
        if isinstance(items, list):
            return len(items)
    return 0


def _verifier_passed(report: dict) -> bool | None:
    result = report.get("result", {})
    value = result.get("verifier_passed")
    if isinstance(value, bool):
        return value

    stats = report.get("stats", {})
    verifier = stats.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("passed"), bool):
        return verifier["passed"]
    return None


def _meta_verifier_passed(meta: dict) -> bool | None:
    verifier = meta.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("passed"), bool):
        return verifier["passed"]
    return None


def summarize_report(report_path: Path, meta_path: Path) -> dict:
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stats = report.get("stats", {})
        result = report.get("result", {})
        outcome = result.get("outcome", "unknown")
        report_exit = result.get("exit_code")
        verifier_passed = _verifier_passed(report)
        if verifier_passed is None:
            verifier_passed = _meta_verifier_passed(meta)
        return {
            "task_id": meta.get("task_id") or report_path.stem,
            "variant": meta.get("variant"),
            "repeat": meta.get("repeat"),
            "outcome": outcome,
            "success": outcome == "success",
            "verifier_passed": verifier_passed,
            "success_strict": None
            if verifier_passed is None
            else outcome == "success" and verifier_passed,
            "process_returncode": meta.get("returncode"),
            "report_exit_code": report_exit,
            "timed_out": bool(meta.get("timed_out", False)),
            "turns": int(stats.get("turns", 0)),
            "llm_calls": int(stats.get("llm_calls", 0)),
            "prompt_tokens_est": _report_prompt_tokens(report),
            "tool_calls_total": int(stats.get("tool_calls_total", 0)),
            "tool_calls_failed": int(stats.get("tool_calls_failed", 0)),
            "tool_repairs": _report_repairs(report),
            "compactions": int(stats.get("compactions", 0)),
            "turn_drops": int(stats.get("turn_drops", 0)),
            "guardrail_interventions": int(stats.get("guardrail_interventions", 0)),
            "truncated_responses": int(stats.get("truncated_responses", 0)),
            "tool_request_count": _count_stat_items(stats, "tool_requests"),
            "blocked_tool_call_count": _count_stat_items(stats, "blocked_tool_calls"),
            "total_llm_time_s": float(stats.get("total_llm_time_s", 0.0)),
            "total_tool_time_s": float(stats.get("total_tool_time_s", 0.0)),
            "duration_s": float(meta.get("duration_s", 0.0)),
        }

    verifier_passed = _meta_verifier_passed(meta)
    return {
        "task_id": meta.get("task_id") or report_path.stem,
        "variant": meta.get("variant"),
        "repeat": meta.get("repeat"),
        "outcome": "harness_error",
        "success": False,
        "verifier_passed": verifier_passed,
        "success_strict": None if verifier_passed is None else False,
        "process_returncode": meta.get("returncode"),
        "report_exit_code": None,
        "timed_out": bool(meta.get("timed_out", False)),
        "turns": 0,
        "llm_calls": 0,
        "prompt_tokens_est": 0,
        "tool_calls_total": 0,
        "tool_calls_failed": 0,
        "tool_repairs": 0,
        "compactions": 0,
        "turn_drops": 0,
        "guardrail_interventions": 0,
        "truncated_responses": 0,
        "tool_request_count": 0,
        "blocked_tool_call_count": 0,
        "total_llm_time_s": 0.0,
        "total_tool_time_s": 0.0,
        "duration_s": float(meta.get("duration_s", 0.0)),
    }


def _sum(rows: list[dict], key: str) -> int | float:
    return sum(row.get(key, 0) for row in rows)


def _row_succeeded(row: dict) -> bool:
    strict = row.get("success_strict")
    if isinstance(strict, bool):
        return strict
    return bool(row.get("success"))


def aggregate_rows(rows: list[dict]) -> dict:
    tasks = len(rows)
    successes = sum(1 for row in rows if _row_succeeded(row))
    return {
        "tasks": tasks,
        "successes": successes,
        "success_rate": round(successes / tasks, 4) if tasks else 0.0,
        "turns": _sum(rows, "turns"),
        "llm_calls": _sum(rows, "llm_calls"),
        "prompt_tokens_est": _sum(rows, "prompt_tokens_est"),
        "tool_calls_total": _sum(rows, "tool_calls_total"),
        "tool_calls_failed": _sum(rows, "tool_calls_failed"),
        "tool_repairs": _sum(rows, "tool_repairs"),
        "compactions": _sum(rows, "compactions"),
        "turn_drops": _sum(rows, "turn_drops"),
        "guardrail_interventions": _sum(rows, "guardrail_interventions"),
        "truncated_responses": _sum(rows, "truncated_responses"),
        "duration_s": round(float(_sum(rows, "duration_s")), 3),
        "total_llm_time_s": round(float(_sum(rows, "total_llm_time_s")), 3),
        "total_tool_time_s": round(float(_sum(rows, "total_tool_time_s")), 3),
    }


def _mean(values: list[int | float]) -> float:
    return round(float(sum(values) / len(values)), 3) if values else 0.0


def metric_distribution(rows: list[dict]) -> dict[str, dict[str, int | float]]:
    metrics: dict[str, dict[str, int | float]] = {}
    for key in METRIC_KEYS:
        values = [row.get(key, 0) for row in rows]
        if not values:
            continue
        metrics[key] = {
            "min": min(values),
            "mean": _mean(values),
            "max": max(values),
        }
    return metrics


def task_distribution(rows: list[dict]) -> dict:
    return {
        **aggregate_rows(rows),
        "metrics": metric_distribution(rows),
    }


def _rows_by(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(row.get(k) for k in keys): row for row in rows}


def no_op_control(
    rows: list[dict], variants: tuple[Variant, ...], repeat: int
) -> list[dict]:
    if repeat < 2:
        return []
    controls: list[dict] = []
    for variant in variants:
        first = [
            row
            for row in rows
            if row.get("variant") == variant.name and row.get("repeat") == 1
        ]
        first_index = _rows_by(first, "task_id")
        for other_repeat in range(2, repeat + 1):
            other = [
                row
                for row in rows
                if row.get("variant") == variant.name
                and row.get("repeat") == other_repeat
            ]
            flips = []
            for row in other:
                base = first_index.get((row.get("task_id"),))
                if base and _row_succeeded(base) != _row_succeeded(row):
                    flips.append(row.get("task_id"))
            delta = sum(1 for row in other if _row_succeeded(row)) - sum(
                1 for row in first if _row_succeeded(row)
            )
            controls.append(
                {
                    "variant": variant.name,
                    "against_repeat": other_repeat,
                    "success_delta": delta,
                    "outcome_flips": flips,
                    "outcome_flip_count": len(flips),
                }
            )
    return controls


def _comparison_for_rows(
    rows: list[dict], variants: tuple[Variant, ...], *, repeat: int | None = None
) -> list[dict]:
    if len(variants) < 2:
        return []
    baseline = variants[0].name
    comparisons = []
    base_rows = [row for row in rows if row.get("variant") == baseline]
    base_successes: dict[str, int] = {}
    for row in base_rows:
        task_id = str(row.get("task_id"))
        base_successes[task_id] = base_successes.get(task_id, 0) + int(
            _row_succeeded(row)
        )
    for variant in variants[1:]:
        candidate = [row for row in rows if row.get("variant") == variant.name]
        candidate_successes: dict[str, int] = {}
        for row in candidate:
            task_id = str(row.get("task_id"))
            candidate_successes[task_id] = candidate_successes.get(task_id, 0) + int(
                _row_succeeded(row)
            )

        wins: list[str] = []
        losses: list[str] = []
        ties: list[str] = []
        for task_id, cand_count in sorted(candidate_successes.items()):
            if task_id not in base_successes:
                continue
            base_count = base_successes[task_id]
            if cand_count > base_count:
                wins.append(task_id)
            elif cand_count < base_count:
                losses.append(task_id)
            else:
                ties.append(task_id)
        item = {
            "baseline": baseline,
            "variant": variant.name,
            "success_delta": len(wins) - len(losses),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
        if repeat is not None:
            item["repeat"] = repeat
        comparisons.append(item)
    return comparisons


def variant_comparisons(rows: list[dict], variants: tuple[Variant, ...]) -> list[dict]:
    return _comparison_for_rows(rows, variants)


def per_repeat_comparisons(
    rows: list[dict], variants: tuple[Variant, ...], repeat: int
) -> list[dict]:
    comparisons: list[dict] = []
    for repeat_idx in range(1, repeat + 1):
        repeat_rows = [row for row in rows if row.get("repeat") == repeat_idx]
        comparisons.extend(
            _comparison_for_rows(repeat_rows, variants, repeat=repeat_idx)
        )
    return comparisons


def summarize_outputs(spec: BenchmarkSpec, tasks: tuple[Task, ...]) -> dict:
    rows: list[dict] = []
    for variant in spec.variants:
        for repeat_idx in range(1, spec.repeat + 1):
            for task in tasks:
                run_dir = spec.out_dir / variant.name / f"run-{repeat_idx:02d}"
                rows.append(
                    summarize_report(
                        run_dir / f"{task.id}.json",
                        run_dir / f"{task.id}.meta.json",
                    )
                )

    by_variant = {}
    for variant in spec.variants:
        variant_rows = [r for r in rows if r.get("variant") == variant.name]
        by_variant[variant.name] = aggregate_rows(variant_rows)

    by_variant_repeat = {}
    for variant in spec.variants:
        by_variant_repeat[variant.name] = {}
        for repeat_idx in range(1, spec.repeat + 1):
            repeat_rows = [
                r
                for r in rows
                if r.get("variant") == variant.name and r.get("repeat") == repeat_idx
            ]
            by_variant_repeat[variant.name][str(repeat_idx)] = aggregate_rows(
                repeat_rows
            )

    by_variant_task = {}
    for variant in spec.variants:
        by_variant_task[variant.name] = {}
        for task in tasks:
            task_rows = [
                r
                for r in rows
                if r.get("variant") == variant.name and r.get("task_id") == task.id
            ]
            by_variant_task[variant.name][task.id] = task_distribution(task_rows)

    return {
        "version": SUMMARY_VERSION,
        "spec": str(spec.path),
        "tasks": [task.id for task in tasks],
        "seed": spec.seed,
        "repeat": spec.repeat,
        "variants": [variant.name for variant in spec.variants],
        "by_variant": by_variant,
        "by_variant_repeat": by_variant_repeat,
        "by_variant_task": by_variant_task,
        "no_op_control": no_op_control(rows, spec.variants, spec.repeat),
        "comparisons": variant_comparisons(rows, spec.variants),
        "per_repeat_comparisons": per_repeat_comparisons(
            rows, spec.variants, spec.repeat
        ),
        "runs": rows,
    }


def write_summary_artifacts(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    write_summary_csv(out_dir / "summary.csv", summary["runs"])
    (out_dir / "summary.md").write_text(render_markdown_summary(summary), "utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "variant",
        "repeat",
        "task_id",
        "success",
        "verifier_passed",
        "success_strict",
        "outcome",
        "turns",
        "llm_calls",
        "prompt_tokens_est",
        "tool_calls_total",
        "tool_calls_failed",
        "tool_repairs",
        "compactions",
        "turn_drops",
        "guardrail_interventions",
        "truncated_responses",
        "tool_request_count",
        "blocked_tool_call_count",
        "duration_s",
        "process_returncode",
        "report_exit_code",
        "timed_out",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_summary(summary: dict) -> str:
    lines = [
        "# Swival Benchmark Summary",
        "",
        f"- Spec: `{summary['spec']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Repeats: `{summary['repeat']}`",
        f"- Tasks: `{len(summary['tasks'])}`",
        "",
        "## Variants",
        "",
        "| Variant | Successes | Tasks | Rate | Turns | Tool failures | Prompt tokens | Duration s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in summary["variants"]:
        row = summary["by_variant"][name]
        lines.append(
            "| {name} | {successes} | {tasks} | {rate:.2%} | {turns} | "
            "{tool_failures} | {tokens} | {duration} |".format(
                name=name,
                successes=row["successes"],
                tasks=row["tasks"],
                rate=row["success_rate"],
                turns=row["turns"],
                tool_failures=row["tool_calls_failed"],
                tokens=row["prompt_tokens_est"],
                duration=row["duration_s"],
            )
        )

    if summary["no_op_control"]:
        lines.extend(["", "## No-Op Control", ""])
        lines.append("| Variant | Repeat | Success delta | Outcome flips |")
        lines.append("|---|---:|---:|---|")
        for row in summary["no_op_control"]:
            flips = ", ".join(row["outcome_flips"]) or "-"
            lines.append(
                f"| {row['variant']} | {row['against_repeat']} | "
                f"{row['success_delta']} | {flips} |"
            )

    if summary["comparisons"]:
        lines.extend(["", "## Comparisons", ""])
        lines.append("| Baseline | Variant | Success delta | Wins | Losses | Ties |")
        lines.append("|---|---|---:|---|---|---|")
        for row in summary["comparisons"]:
            wins = ", ".join(row["wins"]) or "-"
            losses = ", ".join(row["losses"]) or "-"
            ties = ", ".join(row["ties"]) or "-"
            lines.append(
                f"| {row['baseline']} | {row['variant']} | "
                f"{row['success_delta']} | {wins} | {losses} | {ties} |"
            )

    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swival.benchmark",
        description="Run a fixed task corpus through Swival and summarize reports.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a benchmark spec")
    run.add_argument("spec", help="TOML benchmark spec")

    summarize = sub.add_parser("summarize", help="summarize an existing run directory")
    summarize.add_argument("spec", help="TOML benchmark spec used for the run")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        tasks = load_tasks(spec.tasks_path)
        if args.cmd == "run":
            summary = run_benchmark(spec, tasks)
        else:
            summary = summarize_outputs(spec, tasks)
            write_summary_artifacts(spec.out_dir, summary)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(
        f"wrote {spec.out_dir / 'summary.json'} "
        f"({len(summary['variants'])} variant(s), {len(summary['tasks'])} task(s))"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
