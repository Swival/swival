"""Metaskill runtime: execute dynamic skill workflows in a sandboxed interpreter."""

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .report import ReportCollector
from .skills import SkillInfo


MAX_ASK_CALLS_DEFAULT = 5
MAX_COMMAND_CALLS_DEFAULT = 10
MAX_TRACE_ENTRIES_DEFAULT = 100
MAX_COMMAND_RESULT_CHARS = 20_000
MAX_ASK_RESULT_CHARS = 20_000
MAX_RESULT_CHARS = 20_000
TIMEOUT_S_DEFAULT = 300


@dataclass
class MetaskillTrace:
    entries: list[dict] = field(default_factory=list)
    max_entries: int = MAX_TRACE_ENTRIES_DEFAULT

    def append(self, kind: str, data: dict | None = None) -> None:
        if len(self.entries) < self.max_entries:
            entry = {"kind": kind}
            if data:
                entry["data"] = data
            self.entries.append(entry)


@dataclass
class MetaskillBudget:
    max_ask_calls: int = MAX_ASK_CALLS_DEFAULT
    max_command_calls: int = MAX_COMMAND_CALLS_DEFAULT
    timeout_s: float = TIMEOUT_S_DEFAULT
    max_trace_entries: int = MAX_TRACE_ENTRIES_DEFAULT
    max_result_chars: int = MAX_RESULT_CHARS

    ask_calls_used: int = 0
    command_calls_used: int = 0
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def timed_out(self) -> bool:
        return self.elapsed() > self.timeout_s

    def ask_budget_remaining(self) -> bool:
        return self.ask_calls_used < self.max_ask_calls

    def command_budget_remaining(self) -> bool:
        return self.command_calls_used < self.max_command_calls


class MetaskillError(Exception):
    pass


class BudgetExhaustedError(MetaskillError):
    pass


class MetaskillTimeoutError(MetaskillError):
    pass


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _normalize_return_value(value: Any) -> dict:
    if value is None:
        return {"status": "ok", "answer": ""}
    if isinstance(value, str):
        return {"status": "ok", "answer": value}
    if isinstance(value, dict):
        result = dict(value)
        result.setdefault("status", "ok")
        result.setdefault("answer", "")
        return result
    raise MetaskillError(f"invalid metaskill return value: {type(value).__name__}")


def _build_result_envelope(
    return_value: dict, trace: MetaskillTrace, budget: MetaskillBudget
) -> str:
    envelope = dict(return_value)
    envelope.pop("_skill_name", None)
    if trace.entries:
        envelope["trace"] = trace.entries

    def _dumps():
        return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))

    text = _dumps()
    if len(text) > budget.max_result_chars:
        if "trace" in envelope and len(envelope["trace"]) > 1:
            envelope["trace"] = [{"kind": "truncated"}]
        answer = envelope.get("answer", "")
        if len(answer) > budget.max_result_chars // 2:
            envelope["answer"] = answer[: budget.max_result_chars // 2]
            envelope["answer_truncated"] = True
        text = _dumps()

    return text


class MetaskillHostAPI:
    """Host API object exposed to metaskill scripts as `agent`."""

    def __init__(
        self,
        *,
        budget: MetaskillBudget,
        trace: MetaskillTrace,
        loop_kwargs: dict,
        tools: list,
        cancel_flag: threading.Event | None,
        report: ReportCollector | None,
        verbose: bool,
    ):
        self._budget = budget
        self._trace = trace
        self._loop_kwargs = loop_kwargs
        self._tools = tools
        self._cancel_flag = cancel_flag
        self._report = report
        self._verbose = verbose

    def _check_cancelled(self) -> None:
        if self._cancel_flag is not None and self._cancel_flag.is_set():
            raise MetaskillError("cancelled")

    def _check_timeout(self) -> None:
        if self._budget.timed_out():
            raise MetaskillTimeoutError(
                f"metaskill timeout ({self._budget.timeout_s}s) exceeded"
            )

    def ask(self, prompt: str, opts: dict | None = None) -> dict:
        self._check_cancelled()
        self._check_timeout()

        if not self._budget.ask_budget_remaining():
            raise BudgetExhaustedError(
                f"ask budget exhausted ({self._budget.max_ask_calls} calls)"
            )
        self._budget.ask_calls_used += 1

        opts = opts or {}
        purpose = opts.get("purpose", "nested")
        max_turns = min(opts.get("max_turns", 30), 30)

        from .agent import run_agent_loop
        from .thinking import ThinkingState
        from .todo import TodoState

        messages = [{"role": "user", "content": prompt}]

        nested_kwargs = dict(self._loop_kwargs)
        nested_kwargs["messages"] = messages
        nested_kwargs["tools"] = [
            t
            for t in self._tools
            if t.get("function", {}).get("name") != "run_metaskill"
        ]
        nested_kwargs["max_turns"] = max_turns
        nested_kwargs["thinking_state"] = nested_kwargs.get(
            "thinking_state", ThinkingState()
        )
        nested_kwargs["todo_state"] = nested_kwargs.get("todo_state", TodoState())
        nested_kwargs["is_subagent"] = True
        nested_kwargs.pop("goal_state", None)
        nested_kwargs.pop("subagent_manager", None)
        nested_kwargs.pop("compaction_state", None)
        nested_kwargs.pop("event_callback", None)
        nested_kwargs.pop("turn_state", None)
        nested_kwargs.pop("goal_launch_turn", None)

        if self._cancel_flag is not None:
            nested_kwargs["cancel_flag"] = self._cancel_flag

        t0 = time.monotonic()
        answer, exhausted = run_agent_loop(**nested_kwargs)
        duration = time.monotonic() - t0

        if self._report is not None:
            self._report.events.append(
                {
                    "type": "metaskill_step",
                    "operation": "ask",
                    "purpose": purpose,
                    "duration_s": round(duration, 3),
                    "success": answer is not None,
                }
            )

        answer_text = answer or ""
        answer_text, truncated = _truncate(answer_text, MAX_ASK_RESULT_CHARS)

        return {
            "answer": answer_text,
            "exhausted": exhausted,
            "turns": max_turns,
            "truncated": truncated,
        }

    def command(self, argv: list, opts: dict | None = None) -> dict:
        self._check_cancelled()
        self._check_timeout()

        if not self._budget.command_budget_remaining():
            raise BudgetExhaustedError(
                f"command budget exhausted ({self._budget.max_command_calls} calls)"
            )
        self._budget.command_calls_used += 1

        if not isinstance(argv, list) or not argv:
            return {
                "ok": False,
                "exit_code": -1,
                "result": "error: argv must be a non-empty list of strings",
                "truncated": False,
            }

        opts = opts or {}
        timeout = min(opts.get("timeout", 30), 120)

        from .tools import dispatch

        args = {"command": argv, "timeout": timeout}
        base_dir = self._loop_kwargs["base_dir"]
        dispatch_kwargs = {
            k: v for k, v in self._loop_kwargs.items() if k != "base_dir"
        }
        dispatch_kwargs["report"] = self._report

        t0 = time.monotonic()
        result = dispatch("run_command", args, base_dir, **dispatch_kwargs)
        duration = time.monotonic() - t0

        ok = not result.startswith("error:") and "Exit code:" not in result
        exit_code = 0 if ok else 1
        if not ok and "Exit code:" in result:
            import re

            m = re.search(r"Exit code: (\d+)", result)
            if m:
                exit_code = int(m.group(1))

        if self._report is not None:
            self._report.events.append(
                {
                    "type": "metaskill_step",
                    "operation": "command",
                    "argv": argv[:3],
                    "duration_s": round(duration, 3),
                    "success": ok,
                }
            )

        result_text, truncated = _truncate(result, MAX_COMMAND_RESULT_CHARS)
        return {
            "ok": ok,
            "exit_code": exit_code,
            "result": result_text,
            "truncated": truncated,
        }

    def trace(self, kind: str, data: dict | None = None) -> None:
        self._check_cancelled()
        self._check_timeout()
        self._trace.append(kind, data)


def _check_starlark_available() -> bool:
    try:
        import starlark_go  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def _raise_from_eval_error(e: Exception) -> None:
    err_str = str(e)
    if "BudgetExhaustedError" in err_str or "budget" in err_str.lower():
        raise BudgetExhaustedError(err_str)
    if "TimeoutError" in err_str or "timeout" in err_str.lower():
        raise MetaskillTimeoutError(err_str)
    if "cancelled" in err_str.lower():
        raise MetaskillError("cancelled")
    raise MetaskillError(f"runtime error: {e}")


def _run_starlark(
    source: str, host_api: MetaskillHostAPI, input_data: dict, timeout_s: float
) -> Any:
    import starlark_go

    def _ask(prompt, opts=None):
        if opts is None:
            opts = {}
        return host_api.ask(prompt, opts)

    def _command(argv, opts=None):
        if opts is None:
            opts = {}
        return host_api.command(argv, opts)

    def _trace(kind, data=None):
        if data is None:
            data = {}
        host_api.trace(kind, data)

    s = starlark_go.Starlark()
    s.set(ask=_ask, command=_command, trace=_trace, input=input_data)

    try:
        s.exec(source)
    except starlark_go.ResolveError as e:
        raise MetaskillError(f"syntax/resolve error: {e}")
    except starlark_go.EvalError as e:
        _raise_from_eval_error(e)

    try:
        result = s.eval("run(input)")
    except starlark_go.ResolveError as e:
        raise MetaskillError(f"metaskill must define a run(input) function: {e}")
    except starlark_go.EvalError as e:
        _raise_from_eval_error(e)

    return result


def _run_starlark_with_timeout(
    source: str, host_api: MetaskillHostAPI, input_data: dict, timeout_s: float
) -> Any:
    """Run Starlark in a daemon thread with a wall-clock timeout.

    Limitation: starlark-go does not expose interpreter cancellation, so
    on timeout the daemon thread may continue running until the next host
    API call (which checks budget/cancellation) or until process exit.
    Pure-computation loops without host calls cannot be interrupted.
    """
    result_box: list = []
    error_box: list = []

    def _target():
        try:
            result_box.append(_run_starlark(source, host_api, input_data, timeout_s))
        except BaseException as e:
            error_box.append(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise MetaskillTimeoutError(f"metaskill timeout ({timeout_s}s) exceeded")

    if error_box:
        raise error_box[0]

    return result_box[0] if result_box else None


def run_metaskill(
    name: str,
    input_data: dict | None,
    *,
    skills_catalog: dict[str, SkillInfo],
    metaskills_policy: str,
    loop_kwargs: dict,
    tools: list,
    cancel_flag: threading.Event | None = None,
    report: ReportCollector | None = None,
    verbose: bool = False,
    max_ask_calls: int | None = None,
    max_command_calls: int | None = None,
) -> str:
    """Execute a metaskill by name. Returns a string result (tool contract)."""
    skill = skills_catalog.get(name)
    if skill is None:
        return f"error: unknown metaskill: {name!r}"

    if skill.metaskill_path is None:
        return f"error: skill {name!r} is not a metaskill"

    if not skill.is_local and metaskills_policy != "all":
        return f"error: external metaskill {name!r} is not allowed (policy: {metaskills_policy})"

    if not _check_starlark_available():
        return (
            "error: starlark runtime not installed. "
            "Install with: pip install 'swival[metaskills]'"
        )

    try:
        source = skill.metaskill_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"error: failed to read metaskill file: {e}"

    if not input_data:
        from .skills import activate_skill

        hint = activate_skill(name, skills_catalog, [], enabled_metaskills={name})
        return (
            f"error: input is required. The metaskill needs specific keys in the "
            f"input object. Here are the skill instructions:\n\n{hint}\n\n"
            f"Retry run_metaskill with the correct input keys."
        )

    budget = MetaskillBudget(
        max_ask_calls=max_ask_calls
        if max_ask_calls is not None
        else MAX_ASK_CALLS_DEFAULT,
        max_command_calls=(
            max_command_calls
            if max_command_calls is not None
            else MAX_COMMAND_CALLS_DEFAULT
        ),
    )
    budget.start()

    trace = MetaskillTrace(max_entries=budget.max_trace_entries)
    trace.append(
        "metaskill_start", {"name": name, "language": skill.metaskill_language}
    )

    if report is not None:
        report.events.append(
            {
                "type": "metaskill_start",
                "name": name,
                "language": skill.metaskill_language,
            }
        )

    host_api = MetaskillHostAPI(
        budget=budget,
        trace=trace,
        loop_kwargs=loop_kwargs,
        tools=tools,
        cancel_flag=cancel_flag,
        report=report,
        verbose=verbose,
    )

    try:
        raw_result = _run_starlark_with_timeout(
            source, host_api, input_data, budget.timeout_s
        )
        return_value = _normalize_return_value(raw_result)
    except Exception as e:
        trace.append("metaskill_error", {"error": str(e)})
        if report is not None:
            report.events.append(
                {"type": "metaskill_error", "name": name, "error": str(e)}
            )
        sep = ":" if isinstance(e, MetaskillError) else " failed:"
        return f"error: metaskill {name!r}{sep} {e}"

    trace.append(
        "metaskill_finish",
        {
            "name": name,
            "status": return_value.get("status", "ok"),
            "ask_calls": budget.ask_calls_used,
            "command_calls": budget.command_calls_used,
            "duration_s": round(budget.elapsed(), 3),
        },
    )

    if report is not None:
        report.events.append(
            {
                "type": "metaskill_finish",
                "name": name,
                "status": return_value.get("status", "ok"),
                "duration_s": round(budget.elapsed(), 3),
            }
        )

    result_text = _build_result_envelope(return_value, trace, budget)
    return f"[Metaskill: {name} completed]\n{result_text}"


def get_executable_metaskills(catalog: dict[str, SkillInfo], policy: str) -> list[str]:
    """Return names of metaskills that can be executed under the given policy.

    Returns an empty list when the Starlark runtime is not installed, so a
    Swival installed without the ``metaskills`` extra transparently treats
    metaskill-bearing skills as ordinary static skills.
    """
    if not _check_starlark_available():
        return []
    names = []
    for name, skill in catalog.items():
        if skill.metaskill_path is None:
            continue
        if not skill.is_local and policy != "all":
            continue
        names.append(name)
    return sorted(names)
