"""Instruction file discovery, rendering and admission.

``CLAUDE.md`` and ``AGENTS.md`` hold mandatory rules: exact commands, hard
requirements, the exceptions to them. A rule cut in half still reads like a
complete rule, so this module never cuts one. It reads every applicable file
whole, renders the complete set, measures that set once, and then either admits
all of it or explains what did not fit.

The measurement is a token count of the rendered text, wrappers and source
labels included, against a budget derived from the model's window. Files are
read with their own byte caps so a runaway file fails loudly instead of
arriving as a plausible-looking prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from . import fmt
from .report import ConfigError
from .tokens import count_tokens, encoder_is_fallback

# Read caps. These bound memory, not content: a file under the cap is loaded
# whole, and a file over it is an error rather than a prefix.
MAX_INSTRUCTION_FILE_BYTES = 1024 * 1024
MAX_INSTRUCTION_TOTAL_BYTES = 4 * 1024 * 1024

# Admission calibration. The share and the ceiling are candidates, not proven
# optimums; release 2 revisits them with measurements from real projects.
INSTRUCTION_BUDGET_SHARE = 0.10
INSTRUCTION_BUDGET_CEILING = 8192
WORKING_RESERVE_SHARE = 0.20
WORKING_RESERVE_CEILING = 4096
UNKNOWN_WINDOW_INSTRUCTION_BUDGET = 2048

_SCOPE_ORDER = ("user", "global", "project")


class InstructionLoadError(ConfigError):
    """The applicable instruction files do not fit this setup.

    Raised before the first model call. ``sources`` lists the files that were
    read and ``breakdown`` maps each prompt component to its token estimate, so
    a caller can tell an oversized ``AGENTS.md`` from a window filled by tool
    schemas. ``read_errors`` is non-empty for the other refusal: a file that
    could not be read whole, which no allowance and no override makes safe.
    """

    def __init__(
        self,
        message: str,
        *,
        sources: "list[str] | tuple[str, ...]" = (),
        breakdown: dict | None = None,
        report: dict | None = None,
        read_errors: "list[str] | None" = None,
    ) -> None:
        super().__init__(message)
        self.sources = list(sources)
        self.breakdown = dict(breakdown or {})
        self.report = report
        self.read_errors = list(read_errors or [])


@dataclass(frozen=True)
class InstructionSource:
    """One instruction file that was read completely."""

    path: str
    scope: str  # "project", "user" or "global"
    kind: str  # "claude" or "agents"
    text: str


@dataclass
class InstructionSet:
    """Every applicable instruction file, rendered as one block."""

    # These three partition what discovery found: every file lands in exactly
    # one of them. TestEveryDiscoveredFileIsAccountedFor holds them to it, so
    # a new kind of read failure cannot quietly fall between them.
    sources: list[InstructionSource] = field(default_factory=list)
    # Files we hold only part of. They stop the run: half a rule set reads
    # like a whole one.
    errors: list[str] = field(default_factory=list)
    # Files we could not open at all. Reported, then treated as absent.
    skipped: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def paths(self) -> list[str]:
        return [s.path for s in self.sources]


def _collect_project_dirs(base_dir: Path, start_dir: Path) -> list[Path]:
    """Return directories from base_dir down to start_dir, inclusive.

    base_dir must be an ancestor of start_dir (or equal to it). If start_dir
    is not under base_dir, returns [base_dir] — safe fallback to today's behavior.
    """
    base = base_dir.resolve()
    start = start_dir.resolve()
    try:
        rel = start.relative_to(base)
    except ValueError:
        return [base]
    dirs = [base]
    current = base
    for part in rel.parts:
        current = current / part
        dirs.append(current)
    return dirs


def global_agents_md_path() -> Path:
    """Return the cross-agent global AGENTS.md path (testable seam)."""
    return Path.home() / ".agents" / "AGENTS.md"


def discover(
    base_dir: str,
    config_dir: "Path | None" = None,
    *,
    start_dir: "Path | None" = None,
) -> list[tuple[Path, str, str]]:
    """List the applicable instruction files as ``(path, scope, kind)``.

    The order is the loading order and it does not change: the project
    ``CLAUDE.md`` first, then ``AGENTS.md`` from the user config directory, the
    cross-agent directory, and finally each project directory from *base_dir*
    down to *start_dir*.
    """
    root = Path(base_dir).resolve()
    found: list[tuple[Path, str, str]] = []

    claude_path = root / "CLAUDE.md"
    if claude_path.is_file():
        found.append((claude_path, "project", "claude"))

    if config_dir is not None:
        user_path = Path(config_dir) / "AGENTS.md"
        if user_path.is_file():
            found.append((user_path, "user", "agents"))

    global_path = global_agents_md_path()
    if global_path.is_file():
        found.append((global_path, "global", "agents"))

    proj_dirs = (
        _collect_project_dirs(root, start_dir) if start_dir is not None else [root]
    )
    for proj_dir in proj_dirs:
        proj_path = proj_dir / "AGENTS.md"
        if proj_path.is_file():
            found.append((proj_path, "project", "agents"))

    return found


class _ReadOutcome(NamedTuple):
    """What one instruction file gave back."""

    text: str | None
    size: int
    message: str | None
    oversized: bool


def _read_whole(path: Path, remaining: int) -> _ReadOutcome:
    """Read *path* completely, or say why not.

    The *oversized* flag separates two failures
    that look alike and are not.

    A file over one of the read caps is *oversized*: it exists, it is readable,
    and the only reason we do not have all of it is that the user has written
    more than we will hold. Loading the part we read would be loading half a
    rule set, so that stops the run and the message says how to fix it.

    A file we cannot stat or open is a different thing. The rules may be
    perfectly ordinary and a permission bit or a failing disk is in the way.
    Refusing to run over it would be refusing to run over the environment, so
    that is reported and skipped, like a file that is not there.
    """

    def failure(message: str, *, oversized: bool) -> _ReadOutcome:
        return _ReadOutcome(None, 0, message, oversized)

    try:
        size = path.stat().st_size
    except OSError as exc:
        return failure(f"{path}: {exc.strerror or exc}", oversized=False)
    if size > MAX_INSTRUCTION_FILE_BYTES:
        return failure(
            f"{path}: {size} bytes exceeds the "
            f"{MAX_INSTRUCTION_FILE_BYTES}-byte per-file read limit",
            oversized=True,
        )
    if size > remaining:
        return failure(
            f"{path}: reading it would pass the "
            f"{MAX_INSTRUCTION_TOTAL_BYTES}-byte combined read limit",
            oversized=True,
        )
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_INSTRUCTION_FILE_BYTES + 1)
    except OSError as exc:
        return failure(f"{path}: {exc.strerror or exc}", oversized=False)
    if len(raw) > MAX_INSTRUCTION_FILE_BYTES:
        return failure(
            f"{path}: grew past the {MAX_INSTRUCTION_FILE_BYTES}-byte "
            "per-file read limit while it was being read",
            oversized=True,
        )
    return _ReadOutcome(raw.decode("utf-8", errors="replace"), len(raw), None, False)


def load(
    base_dir: str,
    config_dir: "Path | None" = None,
    *,
    start_dir: "Path | None" = None,
    verbose: bool = False,
) -> InstructionSet:
    """Read every applicable instruction file and render them as one block."""
    from .skills import strip_markdown_comments

    result = InstructionSet()
    remaining = MAX_INSTRUCTION_TOTAL_BYTES

    for path, scope, kind in discover(base_dir, config_dir, start_dir=start_dir):
        read = _read_whole(path, remaining)
        if read.message is not None:
            (result.errors if read.oversized else result.skipped).append(read.message)
            fmt.warning(f"instruction file not loaded — {read.message}")
            continue
        remaining -= read.size
        result.sources.append(
            InstructionSource(
                path=str(path),
                scope=scope,
                kind=kind,
                text=strip_markdown_comments(read.text),
            )
        )
        if verbose:
            name = "CLAUDE.md" if kind == "claude" else "AGENTS.md"
            fmt.info(f"Loaded {name} ({read.size} bytes) from {path.parent}")

    result.text = render(result.sources)
    return result


def render(sources: "list[InstructionSource]") -> str:
    """Render *sources* as the XML block that goes into the system prompt.

    The project ``CLAUDE.md`` keeps its own wrapper. The ``AGENTS.md`` files
    share one wrapper and each carries a comment naming its scope and path, so
    the model can tell a personal preference from a project rule.
    """
    sections: list[str] = []

    claude = [s for s in sources if s.kind == "claude"]
    for source in claude:
        sections.append(
            f"<project-instructions>\n{source.text}\n</project-instructions>"
        )

    agents = [s for s in sources if s.kind == "agents"]
    by_scope = {scope: [] for scope in _SCOPE_ORDER}
    for source in agents:
        by_scope[source.scope].append(source)
    parts = [
        f"<!-- {source.scope}: {source.path} -->\n{source.text}"
        for scope in _SCOPE_ORDER
        for source in by_scope[scope]
    ]
    if parts:
        inner = "\n\n".join(parts)
        sections.append(f"<agent-instructions>\n{inner}\n</agent-instructions>")

    return "\n\n".join(sections)


def instruction_budget(
    context_length: int | None,
    prompt_budget: int | None,
    fixed_cost: int,
) -> tuple[int, str]:
    """Token allowance for the complete instruction set.

    Returns ``(budget, binding)`` where *binding* names what limits the
    allowance: ``"absolute"`` when the fixed ceiling decides, ``"window"`` when
    the share of the window decides, and ``"residual"`` when the rest of the
    prompt has already taken the room. Only the last two get smaller or larger
    with the model, which is what the failure message needs to know.
    """
    if context_length is None:
        return UNKNOWN_WINDOW_INSTRUCTION_BUDGET, "unknown"

    working = min(WORKING_RESERVE_CEILING, int(context_length * WORKING_RESERVE_SHARE))
    window_share = int(context_length * INSTRUCTION_BUDGET_SHARE)
    residual = (prompt_budget or 0) - fixed_cost - working

    candidates = {
        "absolute": INSTRUCTION_BUDGET_CEILING,
        "window": window_share,
        "residual": residual,
    }
    binding = min(candidates, key=lambda k: candidates[k])
    return max(0, candidates[binding]), binding


def working_reserve(context_length: int | None) -> int:
    """Room kept for the turn's own work, not for the prompt we start with."""
    if context_length is None:
        return 0
    return min(WORKING_RESERVE_CEILING, int(context_length * WORKING_RESERVE_SHARE))


@dataclass
class InstructionAdmission:
    """The one loading decision for a freshly assembled request."""

    cost: int
    budget: int
    binding: str
    window: int | None = None
    sources: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    estimate_is_exact: bool = True
    override: bool = False
    read_errors: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        """Whether this set goes into the prompt."""
        return self.complete and (self.override or self.advisory or self.fits)

    @property
    def forced(self) -> bool:
        """Whether the override is what carried it over its allowance."""
        return self.complete and self.override and not self.fits

    @property
    def complete(self) -> bool:
        """True when every applicable file was read whole."""
        return not self.read_errors

    @property
    def advisory(self) -> bool:
        """Whether the allowance is a number worth blocking a startup on.

        A provider that never states its window gives us nothing to measure
        against, and a session with no tokenizer data counts UTF-8 bytes.
        """
        return self.window is None or not self.estimate_is_exact

    @property
    def fits(self) -> bool:
        """True when the set is inside the allowance we computed."""
        return self.cost <= self.budget

    @property
    def status(self) -> str:
        if self.forced:
            return "forced"
        return "loaded" if self.admitted else "failed"


def admit(
    rendered: str,
    *,
    context_length: int | None,
    prompt_budget: int | None,
    fixed_cost: int,
    sources: list[str],
    breakdown: dict | None = None,
    instructions_full: bool = False,
    read_errors: "list[str] | None" = None,
    cost: int | None = None,
) -> InstructionAdmission:
    """Decide whether the complete instruction set fits, once.

    A file that could not be read whole ends the decision immediately. The set
    on offer is not the set the user wrote, and no allowance makes a missing
    rule safe to run without. Nothing overrides that: the size override exists
    to spend more of the window, not to proceed on a partial set.

    Past that, the decision is only binding when both numbers are worth
    trusting. Two cases make the allowance advisory instead: a provider that
    never states its window, and a session with no tokenizer data, where sizes
    are UTF-8 bytes. Blocking a startup on either would refuse a set that
    probably fits. Those sessions load the complete set and lean on
    provider-overflow recovery, which is what they already do for everything
    else.
    """
    budget, binding = instruction_budget(context_length, prompt_budget, fixed_cost)
    if cost is None:
        cost = count_tokens(rendered) if rendered else 0
    return InstructionAdmission(
        cost=cost,
        window=context_length,
        budget=budget,
        binding=binding,
        sources=list(sources),
        breakdown=dict(breakdown or {}),
        estimate_is_exact=not encoder_is_fallback(),
        override=bool(instructions_full),
        read_errors=list(read_errors or []),
    )


def unreadable_message(admission: InstructionAdmission) -> str:
    """The terminal diagnostic for a set we could not read whole."""
    lines = [
        "Some instruction files could not be read.",
        "Fix or shorten the listed files, or restart with:",
        "  --no-instructions     Skip project and personal instruction files.",
        "",
        "Could not read:",
    ]
    lines.extend(f"  {error}" for error in admission.read_errors)
    return "\n".join(lines)


def oversized_message(admission: InstructionAdmission) -> str:
    """The terminal diagnostic for an instruction set that does not fit."""
    lines = [
        "The instruction files are too large for this setup.",
    ]
    if admission.binding == "absolute":
        lines.append("Shorten the listed files, or restart with:")
    else:
        lines.append(
            "Shorten the listed files, choose a model with more room, or restart with:"
        )
    lines.append("  --no-instructions     Skip project and personal instruction files.")
    lines.append(
        "  --instructions-full   Load all instructions; this may crowd out work or fail."
    )
    lines.append("")
    lines.append("Files:")
    for path in admission.sources:
        lines.append(f"  {path}")
    return "\n".join(lines)


# Components the user can actually turn off, with the flag that does it.
_CONTRIBUTORS = (
    ("tools", "tool schemas", None),
    ("skills", "the skill catalog", "--no-skills"),
    ("memory", "memory", "--no-memory"),
    ("continuation", "the continuation file", "--no-continue"),
    ("external_tool_notes", "external tool descriptions", None),
)

_BREAKDOWN_LABELS = (
    ("builtin", "built-in prompt"),
    ("instructions", "instruction files"),
    ("memory", "memory"),
    ("skills", "skill catalog"),
    ("tools", "tool schemas"),
    ("external_tool_notes", "external tool notes"),
    ("continuation", "continuation"),
    ("task", "task input"),
    ("output_reserve", "output reservation"),
    ("working_reserve", "working reservation"),
)


def contributor_lines(admission: InstructionAdmission) -> list[str]:
    """Name what else is filling the window, when something else is.

    A small ``AGENTS.md`` is not the problem when the tool schemas are three
    times its size, and saying so saves the user from shortening the wrong file.
    """
    lines = []
    for key, label, flag in _CONTRIBUTORS:
        cost = admission.breakdown.get(key, 0)
        if cost <= 0 or cost < admission.cost:
            continue
        if flag:
            lines.append(
                f"{label.capitalize()} takes at least as much of this window "
                f"as the instructions do; {flag} frees it."
            )
        else:
            lines.append(
                f"{label.capitalize()} take at least as much of this window "
                "as the instructions do."
            )
    return lines


def breakdown_lines(admission: InstructionAdmission) -> list[str]:
    """Per-component token estimates, for verbose diagnostics and reports."""
    lines = ["Estimated cost, in tokens:"]
    for key, label in _BREAKDOWN_LABELS:
        cost = admission.breakdown.get(key)
        if cost:
            lines.append(f"  {label:<22}{cost}")
        if key == "instructions":
            for path, source_cost in admission.breakdown.get(
                "instructions_by_source", {}
            ).items():
                lines.append(f"    {path}: {source_cost}")
    lines.append(f"Instruction allowance: {admission.budget} tokens.")
    lines.append(
        "Counts come from the cl100k_base tokenizer."
        if admission.estimate_is_exact
        else "Counts are UTF-8 byte estimates; no tokenizer data was available."
    )
    return lines


def failure_diagnostic(
    admission: InstructionAdmission, *, verbose: bool = False
) -> str:
    """The message a user sees when the instruction set cannot be loaded."""
    if not admission.complete:
        parts = [unreadable_message(admission)]
        if verbose:
            parts.append("")
            parts.extend(breakdown_lines(admission))
        return "\n".join(parts)
    parts = [oversized_message(admission)]
    extra = contributor_lines(admission)
    if extra:
        parts.append("")
        parts.extend(extra)
    if verbose:
        parts.append("")
        parts.extend(breakdown_lines(admission))
    return "\n".join(parts)


FULL_LOADING_OVERFLOW_MESSAGE = (
    "Full instruction loading (--instructions-full) is enabled. This request still\n"
    "exceeds the model's context after compaction; the instructions were kept intact.\n"
    "Shorten the files, disable full loading, or use --no-instructions to skip them."
)
