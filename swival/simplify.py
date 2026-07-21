"""Text-first multi-agent orchestration for ``/simplify``.

The reviewers produce ordinary prose.  The main agent reads their reports,
checks the code itself, and owns both the edits and validation.  There is no
machine-readable candidate protocol for a model to satisfy.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from . import fmt
from .a2a_types import EVENT_STATUS_UPDATE, EVENT_TOOL_START
from .worktree import (
    READ_ONLY_AGENT_TOOLS,
    ToolPolicy,
    filter_tool_schemas,
    make_isolated_loop_kwargs,
    run_bounded_batch,
)

if TYPE_CHECKING:
    from .input_dispatch import InputContext


_REVIEW_MAX_TURNS = 30
_REPORT_CHAR_LIMIT = 12_000

_REVIEW_ROLES = {
    "local": (
        "Find small, local ways to delete or straighten code: redundant branches, "
        "temporary state, wrappers, indirection, or verbose language constructs. "
        "Prefer changes contained in one function or file."
    ),
    "reuse": (
        "Find code that can be removed by calling an existing project helper or "
        "by reusing a result already computed on the same path. Verify that call "
        "order, errors, mutation, ownership, and logging stay the same. Do not "
        "propose a new shared abstraction."
    ),
    "contracts": (
        "Map the behavior that candidate simplifications must preserve. Inspect "
        "callers and tests for public contracts, edge cases, side effects, error "
        "boundaries, ordering, concurrency, and platform behavior. Flag risky "
        "ideas and suggest the smallest checks that would catch breakage."
    ),
}

_REVIEWER_SYSTEM = """\
You are a read-only reviewer helping simplify an existing project without
changing its behavior. First inspect the workspace instructions, then the target
code, its callers, and relevant tests. Discover the project's language and
conventions instead of assuming them.

Be conservative. Favor deletion and direct local code over new helpers, layers,
or architecture. Do not propose dependency changes, public API changes, test
weakening, broad redesigns, or changes spanning unrelated areas. If safety is
unclear, report the uncertainty instead of guessing. Do not edit files.

Return concise plain text with at most three findings, best first. For each use
these simple headings: Location, Change, Why simpler, Preserve, Risk, Validate.
Use exact file paths and symbols. A finding must describe one bounded change, not
a refactoring campaign. It is valid and useful to return "No safe simplification"
after stating what you inspected.
"""


def _scope_text(focus: str) -> str:
    if focus:
        return focus
    return (
        "Review the current workspace broadly. Inspect existing implementation "
        "code, not only the most obvious or recently touched file, while keeping "
        "recommendations concrete enough for one simplification pass."
    )


def _reviewer_prompt(role: str, focus: str) -> str:
    return (
        f"Review lens: {role}\n\n"
        f"{_REVIEW_ROLES[role]}\n\n"
        f"Scope requested by the user:\n{_scope_text(focus)}\n\n"
        "Follow the review steps in the system message. Stay within this lens; "
        "do not duplicate the other reviewers' jobs. If the area is already "
        "appropriately simple, say what you inspected and why you would leave it "
        "alone."
    )


class _ReviewerBoard:
    """Thread-safe live status line for the parallel reviewers.

    Each reviewer's isolated loop reports turn starts and tool calls through
    its event callback; the board folds them into one compact spinner label
    such as ``simplify: local 3/30 grep · contracts done · ...`` so the user
    can watch all reviewers make progress instead of staring at a single
    frozen message.
    """

    def __init__(self, roles: list[str], update_label) -> None:
        self._update_label = update_label
        self._lock = threading.Lock()
        self._rows = dict.fromkeys(roles, "starting")
        self.turns = dict.fromkeys(roles, 0)

    def _set(self, role: str, text: str) -> None:
        with self._lock:
            self._rows[role] = text
            line = " · ".join(f"{r} {s}" for r, s in self._rows.items())
        self._update_label(f"simplify: {line}")

    def watcher(self, role: str):
        """An event callback that mirrors one reviewer's loop onto the board."""

        def _on_event(kind: str, data: dict) -> None:
            if kind == EVENT_STATUS_UPDATE and not data.get("cancelled"):
                self.turns[role] = data.get("turn") or self.turns[role]
                self._set(role, f"{self.turns[role]}/{_REVIEW_MAX_TURNS}")
            elif kind == EVENT_TOOL_START:
                name = data.get("name") or ""
                self._set(role, f"{self.turns[role]}/{_REVIEW_MAX_TURNS} {name}")

        return _on_event

    def finish(self, role: str) -> None:
        self._set(role, "done")

    def fail(self, role: str) -> None:
        self._set(role, "failed")


def _trim_report(report: str) -> str:
    report = report.strip()
    if not report:
        return "The reviewer returned no usable report."
    if len(report) <= _REPORT_CHAR_LIMIT:
        return report
    return report[:_REPORT_CHAR_LIMIT].rstrip() + "\n[report truncated]"


def build_simplify_prompt(focus: str, reports: dict[str, str]) -> str:
    """Build the task the main agent will execute from plain-text reports."""
    rendered_reports = "\n\n".join(
        f"--- {role} reviewer ---\n{_trim_report(reports.get(role, ''))}"
        for role in _REVIEW_ROLES
    )
    return f"""\
Simplify the project now, using the three independent reviewer reports below as
advice. The local and reuse reviewers propose bounded changes; the contracts
reviewer identifies behavior and risks. Reports may overlap, disagree, be
incomplete, or be wrong. Interpret them; do not parse them as a protocol.

Scope requested by the user:
{_scope_text(focus)}

{rendered_reports}

Inspect the referenced code yourself before editing. Reconcile duplicates and
disagreements using the actual implementation, its callers, tests, project
instructions, and the contracts review. Rank ideas by safety and code removed.
Apply only small, high-confidence, behavior-preserving simplifications directly
to the current workspace; do not merely list proposals. Prefer changes within
one function or file, and normally reject an idea that needs more than two files.

Preserve user changes and public behavior. Do not introduce a new abstraction,
dependency, or architectural layer; change public APIs; weaken tests; or bundle
unrelated cleanup. Skip suggestions whose benefit or safety does not survive
inspection. A successful run may make no changes.

Run the relevant formatter, tests, lint, build, or other checks defined by the
project. Fix regressions you introduced or revert the unsafe change. In the final
answer, summarize what changed, what validation passed, and any notable reviewer
ideas you deliberately rejected. Do not output JSON.
"""


def prepare_simplify_prompt(cmd_arg: str, ctx: "InputContext") -> str:
    """Run three complementary reviewers and return the main-agent task."""
    from .agent import run_agent_loop

    focus = cmd_arg.strip()
    tools = filter_tool_schemas(ctx.tools, READ_ONLY_AGENT_TOOLS)
    policy = ToolPolicy(allowed_tools=READ_ONLY_AGENT_TOOLS)
    roles = list(_REVIEW_ROLES)

    fmt.info("simplify: running three read-only reviewers in parallel")
    with fmt.step_spinner("simplify: reviewers inspecting the workspace") as update:
        board = _ReviewerBoard(roles, update)

        def _run_reviewer(role: str) -> str:
            messages = [
                {"role": "system", "content": _REVIEWER_SYSTEM},
                {"role": "user", "content": _reviewer_prompt(role, focus)},
            ]
            kwargs = make_isolated_loop_kwargs(
                ctx,
                ctx.base_dir,
                max_turns=_REVIEW_MAX_TURNS,
                tool_policy=policy,
                cancel_flag=ctx.loop_kwargs.get("cancel_flag"),
                event_callback=board.watcher(role),
            )
            started = time.monotonic()
            try:
                answer, exhausted = run_agent_loop(messages, tools, **kwargs)
            except BaseException:
                board.fail(role)
                raise
            report = _trim_report(answer or "")
            if exhausted:
                report += "\n[reviewer reached its turn limit]"
            board.finish(role)
            fmt.info(
                f"simplify: {role} reviewer finished after "
                f"{board.turns[role]} turns in "
                f"{time.monotonic() - started:.0f}s "
                f"({len(report):,} chars of advice)"
            )
            return report

        results = run_bounded_batch(_run_reviewer, roles, max_workers=len(roles))

    reports = {}
    for role, result in zip(roles, results):
        if result.ok:
            reports[role] = result.value
        else:
            reports[role] = f"The {role} reviewer was unavailable: {result.error}"
            fmt.warning(f"simplify: {role} reviewer failed: {result.error}")
    fmt.info("simplify: reviewer reports collected; main agent is applying findings")
    return build_simplify_prompt(focus, reports)


__all__ = ["build_simplify_prompt", "prepare_simplify_prompt"]
