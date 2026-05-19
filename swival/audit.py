"""Staged security audit over committed Git-tracked code."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_dispatch import InputContext

from . import fmt

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

AUDIT_PROVENANCE_URL = "https://swival.dev"

_AUDIT_TRUNCATION_MARKER = "\n\n[truncated — file too large for context window]"
_AUDIT_TRUNCATION_FLOOR = 200

# Per-file cap on hunter seed-file evidence. Keeps a single huge generated file
# from blowing the global budget across a batch of parallel tasks.
_HUNT_SEED_FILE_CAP = 16_384

_LARGE_SCOPE_THRESHOLD = 500
_DEFAULT_PATCH_MAX_TURNS = 50

_DEFAULT_METRICS: dict[str, int] = {
    "parse_failures": 0,
    "parse_failures_profile": 0,
    "parse_failures_triage": 0,
    "parse_failures_finding": 0,
    "parse_failures_expansion": 0,
    "repair_successes": 0,
    "repair_failures": 0,
    "analytical_retries": 0,
    "multiline_continuations": 0,
}

# ---------------------------------------------------------------------------
# Debug log
# ---------------------------------------------------------------------------

_debug_log_path: Path | None = None
_debug_log_lock = threading.Lock()


def _debug_log(event: str, **fields) -> None:
    if _debug_log_path is None:
        return
    entry = {"ts": time.time(), "event": event, **fields}
    line = json.dumps(entry, default=str) + "\n"
    with _debug_log_lock:
        with open(_debug_log_path, "a") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------

_SOURCE_EXTS = frozenset(
    {
        ".d",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".swift",
        ".scala",
        ".sh",
        ".zig",
    }
)

_CONFIG_EXTS = frozenset(
    {
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".xml",
        ".ini",
        ".conf",
        ".sql",
        ".graphql",
        ".proto",
        ".rego",
        ".tf",
        ".cue",
    }
)

_AUDITABLE_EXTS = _SOURCE_EXTS | _CONFIG_EXTS

# ---------------------------------------------------------------------------
# Attack-surface keywords (cheap heuristic for file ordering)
# ---------------------------------------------------------------------------

_ATTACK_SURFACE_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(exec|system|popen|subprocess|spawn|eval)\b", re.I), 5),
    (re.compile(r"\b(os\.path|open|fopen|readFile|writeFile|unlink|rmdir)\b", re.I), 4),
    (
        re.compile(
            r"\b(request|response|handler|route|endpoint|app\.(get|post|put|delete))\b",
            re.I,
        ),
        4,
    ),
    (re.compile(r"\b(auth|login|password|token|secret|credential|session)\b", re.I), 4),
    (
        re.compile(
            r"\b(parse|decode|deserialize|unmarshal|fromJSON|load|loads)\b", re.I
        ),
        3,
    ),
    (re.compile(r"\b(sql|query|execute|cursor|prepare|raw_sql)\b", re.I), 3),
    (re.compile(r"\b(render|template|jinja|mustache|handlebars)\b", re.I), 2),
    (re.compile(r"\b(lock|mutex|semaphore|thread|async|await|concurrent)\b", re.I), 2),
    (re.compile(r"\b(connect|socket|listen|bind|http|https|fetch|urllib)\b", re.I), 3),
    (re.compile(r"\b(transaction|commit|rollback|migrate)\b", re.I), 2),
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _coerce_focus(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return list(raw)  # type: ignore[arg-type]


def _normalize_focus(entries: list[str]) -> list[str]:
    normalized = (
        e if any(c in e for c in "*?[") else (e.rstrip("/") or e) for e in entries
    )
    return list(dict.fromkeys(normalized))


@dataclass(frozen=True)
class AuditScope:
    branch: str
    commit: str
    tracked_files: list[str]
    mandatory_files: list[str]
    focus: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AuditScope:
        focus = _normalize_focus(_coerce_focus(d.get("focus")))
        return cls(**{**d, "focus": focus})


@dataclass
class TriageRecord:
    path: str
    priority: str  # ESCALATE_HIGH | ESCALATE_MEDIUM | SKIP
    confidence: str
    bug_classes: list[str]
    summary: str
    relevant_symbols: list[str]
    suspicious_flows: list[str]
    needs_followup: bool
    promotion_reasons: list[str] = field(default_factory=list)
    triage_failure_mode: str | None = None
    confirmation_outcome: str | None = None


@dataclass
class FindingRecord:
    title: str
    finding_type: str
    severity: str
    locations: list[str]
    preconditions: list[str]
    proof: list[str]
    fix_outline: str
    source_file: str
    triage_decision: str | None = None  # "escalated" | "skipped" | None
    local_bug: str = ""
    source_boundary: str = ""
    reachability_path: list[str] = field(default_factory=list)
    sink_operation: str = ""
    exploit_chain: list[str] = field(default_factory=list)
    reachability_status: str = "unknown"
    hunt_task_id: str = ""
    attack_class: str = ""
    attacker_position: str = ""
    controlled_inputs: list[str] = field(default_factory=list)


@dataclass
class VerifiedFinding:
    finding: FindingRecord
    correctness_reason: str
    rebuttal_reason: str
    reproducer: dict | None = None
    adversarial: dict | None = None
    trace: dict | None = None
    root_cause_group_id: str = ""


@dataclass
class DeepReviewResult:
    path: str
    findings: list[FindingRecord] | None = None
    error: str | None = None


@dataclass
class VerificationResult:
    finding_key: str
    verified_finding: VerifiedFinding | None = None
    discarded: bool = False
    error: str | None = None
    attempts: int = 1


@dataclass
class PatchGenerationResult:
    patch_text: str | None = None
    error_code: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Hunt task model
# ---------------------------------------------------------------------------

# Recognised attack classes. Adding new ones is the cheapest path to broaden
# coverage; deletions affect persisted state, so retire by mapping at load time.
_HUNT_ATTACK_CLASSES = (
    "command_execution",
    "authorization",
    "path_traversal",
    "ssrf",
    "sqli",
    "xxe",
    "xss",
    "deserialization",
    "parser_differential",
    "cryptography",
    "memory_safety",
    "policy_bypass",
    "toctou",
    "open_redirect",
    "secret_exposure",
    "logic_flaw",
)

# Attacker positions that make a task concrete enough to act on. Free-form
# strings outside this set are allowed but emit a warning at validation time so
# the operator sees that the task may be too generic.
_HUNT_ATTACKER_POSITIONS = (
    "unauthenticated remote client",
    "authenticated low-privileged user",
    "authenticated peer tenant",
    "malicious peer",
    "attacker-controlled backend",
    "crafted file author",
    "lower-privileged local user",
    "compromised dependency",
    "sandboxed workload",
)

_HUNT_TASK_KIND_HUNT = "hunt"
_HUNT_TASK_KIND_REACHABILITY = "reachability"
_HUNT_TASK_KINDS = (_HUNT_TASK_KIND_HUNT, _HUNT_TASK_KIND_REACHABILITY)
_HUNT_TASK_STATUSES = ("pending", "running", "done", "failed")
_REACH_REACHABLE = "reachable"
_REACH_LOCAL_ONLY = "local_only"
_REACH_NOT_REACHABLE = "not_reachable"
_REACH_UNKNOWN = "unknown"
_HUNT_REACHABILITY_STATUSES = (
    _REACH_REACHABLE,
    _REACH_LOCAL_ONLY,
    _REACH_NOT_REACHABLE,
    _REACH_UNKNOWN,
)
_REACHABILITY_TRACE_VERDICTS = (
    _REACH_REACHABLE,
    _REACH_NOT_REACHABLE,
    _REACH_UNKNOWN,
)
_REACH_TASK_SOURCE_LOCAL_ONLY = "local_only_promotion"
_HUNT_CONFIDENCE_LEVELS = ("high", "medium", "low")


@dataclass
class HuntTask:
    """A narrow attacker-anchored research task.

    Generated from the repo profile + sink/source scans (and optionally from
    triage records when hunt-mode fallback triage runs). The hunter prompt is
    expected to address exactly one attack class for exactly one attacker model.
    """

    id: str
    task_kind: str
    attack_class: str
    attacker_position: str
    controlled_inputs: list[str]
    trust_boundary_crossed: str
    scope_hint: str
    seed_files: list[str]
    seed_symbols: list[str]
    sink_files: list[str]
    source: str
    priority: str
    status: str = "pending"
    coverage_notes: list[str] = field(default_factory=list)
    parent_task_id: str = ""
    parent_finding_key: str = ""
    attempts: int = 0
    last_error: str = ""
    language_hint: str = ""


@dataclass
class CoverageRecord:
    task_id: str
    observed_files: list[str] = field(default_factory=list)
    observed_symbols: list[str] = field(default_factory=list)
    observed_edges: list[str] = field(default_factory=list)
    covered_fraction_of_seed_files: float = 0.0
    covered_fraction_of_sink_hits: float = 0.0
    unobserved_high_priority_seeds: list[str] = field(default_factory=list)
    explicit_not_covered: list[str] = field(default_factory=list)
    followup_task_ids: list[str] = field(default_factory=list)
    confidence: str = "low"


@dataclass
class RootCauseGroup:
    id: str
    primary_finding_key: str
    variant_finding_keys: list[str]
    root_cause_summary: str
    source_boundaries: list[str] = field(default_factory=list)
    affected_locations: list[str] = field(default_factory=list)
    finding_type: str = ""
    attack_class: str = ""


class HuntTaskValidationError(ValueError):
    """Raised when a HuntTask is missing one of the attacker-model fields."""


def _validate_hunt_task(task: HuntTask) -> None:
    """Reject a HuntTask before execution if attacker-model fields are missing.

    `attacker_position`, `controlled_inputs`, and `trust_boundary_crossed` are
    load-bearing: without them the generator falls back to generic per-class
    prompts that motivated the plan.
    """
    missing: list[str] = []
    if task.task_kind not in _HUNT_TASK_KINDS:
        raise HuntTaskValidationError(
            f"task {task.id}: task_kind must be one of "
            f"{_HUNT_TASK_KINDS}, got {task.task_kind!r}"
        )
    if task.attack_class not in _HUNT_ATTACK_CLASSES:
        task.coverage_notes.append(f"unknown attack_class={task.attack_class!r}")
    if not task.attacker_position.strip():
        missing.append("attacker_position")
    if not task.controlled_inputs or not any(c.strip() for c in task.controlled_inputs):
        missing.append("controlled_inputs")
    if not task.trust_boundary_crossed.strip():
        missing.append("trust_boundary_crossed")
    if missing:
        raise HuntTaskValidationError(
            f"task {task.id} ({task.attack_class}): missing attacker-model fields: "
            + ", ".join(missing)
        )


_BUDGET_PHASES: tuple[str, ...] = (
    "profile",
    "triage",
    "deep_review",
    "hunt",
    "reachability",
    "proof",
    "disproof",
    "gapfill",
    "dedupe",
    "trace",
    "report",
)

# Default proportional allocation across phases. Hunt + proof + disproof get
# the bulk because they are the new dominant cost centers; triage and profile
# stay small because they run once and feed the rest. The fractions sum to 1.
_BUDGET_DEFAULT_SHARES: dict[str, float] = {
    "profile": 0.03,
    "triage": 0.07,
    "deep_review": 0.10,
    "hunt": 0.30,
    "reachability": 0.10,
    "proof": 0.15,
    "disproof": 0.10,
    "gapfill": 0.07,
    "dedupe": 0.02,
    "trace": 0.03,
    "report": 0.03,
}


class BudgetPlanner:
    """Global token budget for an audit run.

    The planner is wholly stateless except for the references it holds: all
    persistent state lives in ``state.budget_tokens`` (the cap) and
    ``state.budget_used`` (per-phase consumption). That keeps resume cheap and
    lets ``--quiet`` final output read straight from state.

    When ``state.budget_tokens == 0`` the planner is effectively disabled —
    ``remaining`` returns ``float('inf')`` so callers never have to special-case
    "no budget set". This is the default for bare ``/audit`` runs.
    """

    def __init__(self, state: AuditRunState):
        self._state = state

    @property
    def total(self) -> int:
        return self._state.budget_tokens

    @property
    def enabled(self) -> bool:
        return self._state.budget_tokens > 0

    def allocation(self, phase: str) -> float:
        if self.total <= 0:
            return float("inf")
        share = _BUDGET_DEFAULT_SHARES.get(phase, 0.05)
        return self.total * share

    def used(self, phase: str | None = None) -> int:
        if phase is None:
            return sum(self._state.budget_used.values())
        return int(self._state.budget_used.get(phase, 0))

    def remaining(self, phase: str | None = None) -> float:
        if not self.enabled:
            return float("inf")
        if phase is None:
            return max(0, self.total - self.used())
        return max(0.0, self.allocation(phase) - self.used(phase))

    def charge(self, phase: str, tokens: int) -> None:
        if tokens <= 0:
            return
        cur = int(self._state.budget_used.get(phase, 0))
        self._state.budget_used[phase] = cur + int(tokens)

    def exhausted(self) -> bool:
        """True only when the global pool is gone.

        Per-phase allocations are soft hints used by ``should_throttle_low_priority``
        and verbose accounting. A phase that has blown past its allocation can
        still spend so long as the global pool has room, because every audit has
        cheap phases ``hunt`` can borrow from.
        """
        if not self.enabled:
            return False
        return self.remaining(None) <= 0

    def should_throttle_low_priority(self) -> bool:
        if not self.enabled:
            return False
        return self.used() >= 0.5 * self.total

    def charge_call(self, phase: str, request_text: str, response_text: str) -> None:
        """Estimate input+output tokens for one LLM call and charge ``phase``.

        Uses ``_msg._estimate_tokens`` so all in-process accounting (planner,
        snapshot, compaction) agrees on the same rough ``len // 4`` heuristic.
        """
        from ._msg import _estimate_tokens

        self.charge(
            phase, _estimate_tokens(request_text) + _estimate_tokens(response_text)
        )

    def summary_line(self) -> str | None:
        """Concise one-line budget summary for ``--quiet`` final output."""
        if not self.enabled:
            return None
        total = self.total
        used = self.used()
        pct = 100.0 * used / total if total else 0.0
        return f"budget: {used:,}/{total:,} tokens used ({pct:.1f}%)"

    def detail_lines(self) -> list[str]:
        """Per-phase breakdown for verbose/debug output."""
        if not self.enabled:
            return []
        lines: list[str] = []
        for phase in _BUDGET_PHASES:
            used = self.used(phase)
            if used <= 0:
                continue
            alloc = self.allocation(phase)
            pct = 100.0 * used / alloc if alloc else 0.0
            lines.append(f"  {phase:>12}: {used:,} / {int(alloc):,} ({pct:.1f}%)")
        return lines


# ---------------------------------------------------------------------------
# Hunt sink/source catalog
# ---------------------------------------------------------------------------

# Sink-pattern suggestions are starting points; the generator overrides them
# when Phase 1 evidence is stronger (e.g. when an HTTP entry point clearly
# applies).
_HUNT_SINK_CATALOG: dict[str, list[tuple[re.Pattern, str, tuple[str, ...]]]] = {
    "command_execution": [
        (
            re.compile(
                r"\b(?:os\.system|subprocess\.(?:run|call|check_output|Popen)|shell_exec|exec[lv]?p?|popen)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("request body", "query parameter", "header value"),
        ),
        (
            re.compile(r"\b(?:Runtime\.exec|ProcessBuilder|child_process\.exec)\b"),
            "unauthenticated remote client",
            ("request body", "query parameter"),
        ),
    ],
    "path_traversal": [
        (
            re.compile(
                r"\b(?:os\.path\.join|open|Path|fopen|readFile|writeFile|sendfile|send_file|serve_file)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("file path parameter", "uploaded archive entry", "template name"),
        ),
        (
            re.compile(r"\b(?:ZipFile|TarFile|tarfile\.open|zipfile\.ZipFile)\b"),
            "crafted file author",
            ("uploaded archive entry path",),
        ),
    ],
    "ssrf": [
        (
            re.compile(
                r"\b(?:requests\.(?:get|post|put|head|delete)|urllib\.request|httpx\.|fetch|aiohttp\.)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("URL parameter", "callback URL", "webhook target"),
        ),
        (
            re.compile(r"\b(?:urlopen|urlretrieve|http\.Client|net/http)\b"),
            "unauthenticated remote client",
            ("URL parameter",),
        ),
    ],
    "sqli": [
        (
            re.compile(
                r"\b(?:cursor\.execute|conn\.execute|raw_query|db\.query|prepare)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("query parameter", "JSON body field"),
        ),
    ],
    "xxe": [
        (
            re.compile(
                r"\b(?:etree\.parse|xml\.etree|XMLReader|DocumentBuilder|libxml2)\b"
            ),
            "unauthenticated remote client",
            ("uploaded XML payload",),
        ),
    ],
    "deserialization": [
        (
            re.compile(r"\b(?:pickle\.loads?|yaml\.load|cPickle|marshal\.loads)\b"),
            "unauthenticated remote client",
            ("serialized payload",),
        ),
        (
            re.compile(r"\bObjectInputStream|readObject|unserialize\b"),
            "unauthenticated remote client",
            ("serialized payload",),
        ),
    ],
    "xss": [
        (
            re.compile(
                r"\b(?:render_template_string|innerHTML|dangerouslySetInnerHTML|Markup|safe_html|format_html)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("HTML body", "search parameter"),
        ),
    ],
    "cryptography": [
        (
            re.compile(
                r"\b(?:verify(?:_signature|_mac)?|hmac\.compare|hashlib\.compare|jwt\.decode|verify_jwt|verify_token)\b",
                re.I,
            ),
            "malicious peer",
            ("forged signature", "forged JWT", "tampered MAC"),
        ),
    ],
    "memory_safety": [
        (
            re.compile(
                r"\bunsafe\s*\{|memcpy\b|memmove\b|strcpy\b|strncpy\b|sprintf\b|alloca\b"
            ),
            "malicious peer",
            ("length field", "size header", "offset field"),
        ),
    ],
    "parser_differential": [
        (
            re.compile(
                r"\b(?:parse_request|parse_header|HTTP\d?Parser|chunked|content_length|protocol_)\b",
                re.I,
            ),
            "malicious peer",
            ("framing field", "header value", "chunk size"),
        ),
    ],
    "authorization": [
        (
            re.compile(
                r"\b(?:require_role|has_permission|check_permission|is_admin|@authorize|@requires_role|tenant_id)\b",
                re.I,
            ),
            "authenticated low-privileged user",
            ("user-controlled identifier", "tenant id in URL"),
        ),
    ],
    "policy_bypass": [
        (
            re.compile(
                r"\b(?:seccomp|capabilit|namespace|chroot|setuid|setgid|prctl)\b",
                re.I,
            ),
            "lower-privileged local user",
            ("syscall arguments",),
        ),
    ],
    "toctou": [
        (
            re.compile(r"\b(?:lstat|stat|access|open|rename|symlink)\b"),
            "lower-privileged local user",
            ("symlink target", "racing rename"),
        ),
    ],
    "open_redirect": [
        (
            re.compile(
                r"\b(?:redirect|Location|HttpResponseRedirect|res\.redirect)\b",
                re.I,
            ),
            "unauthenticated remote client",
            ("return-to parameter",),
        ),
    ],
    "secret_exposure": [
        (
            re.compile(r"\b(?:log(?:ger)?\.(?:info|debug|warning|error)|print)\b"),
            "compromised dependency",
            ("logged response",),
        ),
    ],
}


_LANGUAGE_HINTS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".zig": "zig",
}


def _detect_language(files: list[str]) -> str:
    """Pick the dominant language extension across ``files``."""
    counts: dict[str, int] = {}
    for f in files:
        for ext, lang in _LANGUAGE_HINTS.items():
            if f.endswith(ext):
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _build_sink_class_unions() -> dict[
    str, tuple[re.Pattern, tuple[tuple[re.Pattern, str, tuple[str, ...]], ...]]
]:
    """For each attack class, build one union regex plus the per-pattern lookup
    table needed to recover the matching pattern's suggested attacker position
    and controlled inputs.

    The union is what gets searched per file: a single ``re.search`` per
    class per file decides whether to recurse into per-pattern matching
    (which only happens on files that actually hit). For a repo where most
    files are inert, the union catches that with one regex op.
    """
    out: dict[
        str, tuple[re.Pattern, tuple[tuple[re.Pattern, str, tuple[str, ...]], ...]]
    ] = {}
    for attack_class, entries in _HUNT_SINK_CATALOG.items():
        union_src = "|".join(f"(?:{e[0].pattern})" for e in entries)
        flags = entries[0][0].flags if entries else 0
        union = re.compile(union_src, flags)
        out[attack_class] = (union, tuple(entries))
    return out


_HUNT_SINK_CLASS_UNIONS = _build_sink_class_unions()


def _scan_sink_hits(
    content_cache: dict[str, str],
) -> dict[str, list[tuple[str, str, tuple[str, ...]]]]:
    """Run the sink catalog against the file content cache.

    Returns ``{attack_class: [(file, suggested_position, suggested_inputs), ...]}``.
    Each file is visited once per class via a unioned regex; on a hit the
    per-pattern entries are consulted to recover the matching pattern's
    suggested attacker position and controlled inputs.
    """
    by_class: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
    for attack_class, (union, entries) in _HUNT_SINK_CLASS_UNIONS.items():
        bucket: list[tuple[str, str, tuple[str, ...]]] = []
        for path, body in content_cache.items():
            if not body or not union.search(body):
                continue
            for pat, pos, inputs in entries:
                if pat.search(body):
                    bucket.append((path, pos, inputs))
                    break
        if bucket:
            by_class[attack_class] = bucket
    return by_class


_BOUNDARY_HINT_REGEXES: dict[str, re.Pattern] = {
    "remote client": re.compile(r"\b(?:api|route|handler|webhook|http)\b", re.I),
    "peer": re.compile(r"\b(?:peer|protocol|wire|tls|ssl|p2p)\b", re.I),
    "crafted file": re.compile(r"\b(?:upload|file|archive|zip)\b", re.I),
}


def _infer_trust_boundary(
    repo_profile: dict | None,
    suggested_position: str,
    sink_file: str,
) -> str:
    """Pick a concrete trust boundary string from Phase 1 evidence.

    Falls back to the suggested position when Phase 1 does not list anything
    obviously matching. Reachability is settled later — this is only the
    *starting* boundary the hunter prompt should orient around.
    """
    if not repo_profile:
        return f"caller of {sink_file}"
    boundaries = [
        b for b in repo_profile.get("trust_boundaries", []) if isinstance(b, str)
    ]
    auth = [a for a in repo_profile.get("auth_surfaces", []) if isinstance(a, str)]
    entries = [e for e in repo_profile.get("entry_points", []) if isinstance(e, str)]

    pos = suggested_position.lower()
    # Each row: (substring trigger, candidate list, hint key). First row whose
    # substring is in `pos` and whose candidate list is non-empty wins.
    dispatch = (
        ("remote client", entries, "remote client"),
        ("peer", boundaries, "peer"),
        ("low-privileged", auth, None),
        ("crafted file", boundaries, "crafted file"),
    )
    for trigger, candidates, hint_key in dispatch:
        if trigger in pos and candidates:
            if hint_key is None:
                return candidates[0]
            hint_re = _BOUNDARY_HINT_REGEXES[hint_key]
            return min(candidates, key=lambda c, r=hint_re: 0 if r.search(c) else 1)
    if boundaries:
        return boundaries[0]
    if entries:
        return entries[0]
    return f"caller of {sink_file}"


def _generate_hunt_tasks(
    state: AuditRunState,
    content_cache: dict[str, str],
    *,
    configured: list[dict] | None = None,
    max_tasks: int = 0,
) -> list[HuntTask]:
    """Build the hunt task queue from Phase 1 evidence plus configured tasks.

    Generation is fully deterministic: same content cache + same configured
    tasks always produce the same task IDs, which makes resume a no-op and
    makes test goldens stable.

    Configured tasks (from ``[[audit.hunt_task]]``) come first and never get
    dropped by ``max_tasks``; they represent operator intent. Scan-derived
    tasks fill the remainder.
    """
    tasks: list[HuntTask] = []
    task_ids: set[str] = set()

    def _push(task: HuntTask) -> None:
        try:
            _validate_hunt_task(task)
        except HuntTaskValidationError as e:
            _debug_log("hunt_task_rejected", error=str(e), task_id=task.id)
            return
        if task.id in task_ids:
            return
        task_ids.add(task.id)
        tasks.append(task)

    # 1) Configured tasks (operator intent first).
    for entry in configured or []:
        attack_class = str(entry.get("attack_class", "")).strip()
        attacker_position = str(entry.get("attacker_position", "")).strip()
        if not attack_class or not attacker_position:
            continue
        controlled_inputs = entry.get("controlled_inputs") or []
        if isinstance(controlled_inputs, str):
            controlled_inputs = [controlled_inputs]
        trust_boundary = str(entry.get("trust_boundary_crossed", "")).strip()
        if not trust_boundary:
            trust_boundary = str(entry.get("trust_boundary", "")).strip()
        scope_hint = str(entry.get("scope", entry.get("scope_hint", ""))).strip()
        seed_files = [
            f
            for f in state.scope.mandatory_files
            if not scope_hint or _match_path_glob(f, scope_hint)
        ]
        seed_files = seed_files[:20]
        priority = str(entry.get("priority", "high")).strip() or "high"
        tid = _hunt_task_id(
            attack_class, attacker_position, scope_hint, seed_files, "cfg"
        )
        _push(
            HuntTask(
                id=tid,
                task_kind="hunt",
                attack_class=attack_class,
                attacker_position=attacker_position,
                controlled_inputs=list(controlled_inputs),
                trust_boundary_crossed=trust_boundary,
                scope_hint=scope_hint,
                seed_files=seed_files,
                seed_symbols=[],
                sink_files=list(seed_files[:5]),
                source="user",
                priority=priority,
                language_hint=_detect_language(seed_files),
            )
        )

    # 2) Scan-derived tasks from Phase 1 + sink catalog.
    sink_hits = _scan_sink_hits(content_cache)
    for attack_class, hits in sink_hits.items():
        # Group sink hits by inferred entry boundary so we get one task per
        # boundary rather than one per file (the latter would explode in
        # large repos and produce hundreds of effectively-identical prompts).
        groups: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
        for path, pos, inputs in hits:
            boundary = _infer_trust_boundary(state.repo_profile, pos, path)
            key = f"{pos}|{boundary}"
            groups.setdefault(key, []).append((path, pos, inputs))

        for key, items in groups.items():
            pos, boundary = key.split("|", 1)
            seed_files = [p for p, _, _ in items[:10]]
            controlled: list[str] = []
            for _p, _pos, inputs in items[:3]:
                for c in inputs:
                    if c and c not in controlled:
                        controlled.append(c)
            seed_symbols: list[str] = []
            for f in seed_files[:3]:
                body = content_cache.get(f, "")
                if not body:
                    continue
                for s in _extract_exports(body):
                    if s not in seed_symbols:
                        seed_symbols.append(s)
                        if len(seed_symbols) >= 6:
                            break
                if len(seed_symbols) >= 6:
                    break
            scope_hint = ""
            if len(seed_files) == 1:
                scope_hint = seed_files[0]
            tid = _hunt_task_id(attack_class, pos, boundary, seed_files)
            priority = (
                "high" if state.attack_scores.get(seed_files[0], 0) >= 5 else "medium"
            )
            _push(
                HuntTask(
                    id=tid,
                    task_kind="hunt",
                    attack_class=attack_class,
                    attacker_position=pos,
                    controlled_inputs=controlled or ["attacker-controlled input"],
                    trust_boundary_crossed=boundary,
                    scope_hint=scope_hint,
                    seed_files=seed_files,
                    seed_symbols=seed_symbols,
                    sink_files=seed_files,
                    source="phase1",
                    priority=priority,
                    language_hint=_detect_language(seed_files),
                )
            )

    if max_tasks and len(tasks) > max_tasks:
        # Keep configured tasks (which were pushed first) and trim the tail.
        kept_configured = [t for t in tasks if t.source == "user"]
        rest = [t for t in tasks if t.source != "user"]
        # Prefer ``high`` over ``medium`` over ``low`` when trimming.
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        rest.sort(key=lambda t: (priority_rank.get(t.priority, 3), t.id))
        room = max(0, max_tasks - len(kept_configured))
        tasks = kept_configured + rest[:room]

    return tasks


def _hunt_task_id(
    attack_class: str,
    attacker_position: str,
    scope_hint: str,
    seed_files: list[str],
    suffix: str = "",
) -> str:
    """Deterministic, short, human-readable task id.

    Format: ``{class}-{first8(hash)}``. Inputs are normalised so that the same
    logical task produced on different runs gets the same id, which keeps the
    resume path stable.
    """
    payload = "|".join(
        [
            attack_class.strip().lower(),
            attacker_position.strip().lower(),
            scope_hint.strip(),
            ",".join(sorted(s.strip() for s in seed_files if s.strip())),
            suffix.strip(),
        ]
    )
    h = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{attack_class}-{h}"


@dataclass
class AuditRunState:
    run_id: str
    scope: AuditScope
    queued_files: list[str]
    reviewed_files: set[str] = field(default_factory=set)
    triage_records: dict[str, TriageRecord] = field(default_factory=dict)
    candidate_files: list[str] = field(default_factory=list)
    deep_reviewed_files: set[str] = field(default_factory=set)
    proposed_findings: list[FindingRecord] = field(default_factory=list)
    verified_findings: list[VerifiedFinding] = field(default_factory=list)
    repo_profile: dict | None = None
    import_index: dict[str, list[str]] = field(default_factory=dict)
    # Despite the name, caller_index[f] is the set of files exporting symbols
    # that f references — i.e. f's dependencies, not f's callers. Use
    # `dependency_index` for new code; preserved as caller_index in saved state
    # for backward compatibility.
    caller_index: dict[str, list[str]] = field(default_factory=dict)
    attack_scores: dict[str, int] = field(default_factory=dict)
    artifact_dir: Path = field(default_factory=lambda: Path("audit-findings"))
    state_dir: Path = field(default_factory=lambda: Path(".swival/audit"))
    verification_state: dict[str, dict] = field(default_factory=dict)
    artifact_state: dict[str, dict] = field(default_factory=dict)
    phase: str = "init"
    metrics: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_METRICS))
    select_all: bool = False
    measure_triage: bool = False
    # When ``measure_triage`` is set, this records the candidate set that
    # *real* triage produced after Phase 2. Phase 3 then expands to the full
    # mandatory set; findings whose source path is not in this set are
    # tagged as having been a triage SKIP.
    measurement_escalated_paths: set[str] = field(default_factory=set)
    # Hunt-harness state. All optional; bare ``/audit`` does not populate these
    # unless ``--hunt`` is set.
    hunt_mode: bool = False
    hunt_tasks: dict[str, HuntTask] = field(default_factory=dict)
    coverage_records: dict[str, CoverageRecord] = field(default_factory=dict)
    adversarial_state: dict[str, dict] = field(default_factory=dict)
    root_cause_groups: list[RootCauseGroup] = field(default_factory=list)
    trace_state: dict[str, dict] = field(default_factory=dict)
    proof_strict: bool = False
    trace_reachability: bool = False
    budget_tokens: int = 0
    # Token consumption is tallied per top-level phase; the keys are the same
    # phase labels we already use in `state.phase` plus a few sub-buckets.
    budget_used: dict[str, int] = field(default_factory=dict)
    # Per-attack-class metrics: ``{class: {"proposed": int, "verified": int}}``.
    attack_class_metrics: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def dependency_index(self) -> dict[str, list[str]]:
        """Accurate-name alias for caller_index.

        caller_index[f] holds files that export symbols f references — i.e.
        f's dependencies — not callers of f. This alias avoids confusion in
        new code without invalidating saved state.
        """
        return self.caller_index

    def save(self) -> None:
        d = self.state_dir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        blob = {
            "run_id": self.run_id,
            "scope": self.scope.to_dict(),
            "queued_files": self.queued_files,
            "reviewed_files": sorted(self.reviewed_files),
            "triage_records": {k: asdict(v) for k, v in self.triage_records.items()},
            "candidate_files": self.candidate_files,
            "deep_reviewed_files": sorted(self.deep_reviewed_files),
            "proposed_findings": [asdict(f) for f in self.proposed_findings],
            "verified_findings": [
                {
                    "finding": asdict(vf.finding),
                    "correctness_reason": vf.correctness_reason,
                    "rebuttal_reason": vf.rebuttal_reason,
                    "reproducer": vf.reproducer,
                    "adversarial": vf.adversarial,
                    "trace": vf.trace,
                    "root_cause_group_id": vf.root_cause_group_id,
                }
                for vf in self.verified_findings
            ],
            "repo_profile": self.repo_profile,
            "import_index": self.import_index,
            "caller_index": self.caller_index,
            "attack_scores": self.attack_scores,
            "verification_state": self.verification_state,
            "artifact_state": self.artifact_state,
            "phase": self.phase,
            "metrics": self.metrics,
            "select_all": self.select_all,
            "measure_triage": self.measure_triage,
            "measurement_escalated_paths": sorted(self.measurement_escalated_paths),
            "hunt_mode": self.hunt_mode,
            "hunt_tasks": {k: asdict(v) for k, v in self.hunt_tasks.items()},
            "coverage_records": {
                k: asdict(v) for k, v in self.coverage_records.items()
            },
            "adversarial_state": self.adversarial_state,
            "root_cause_groups": [asdict(g) for g in self.root_cause_groups],
            "trace_state": self.trace_state,
            "proof_strict": self.proof_strict,
            "trace_reachability": self.trace_reachability,
            "budget_tokens": self.budget_tokens,
            "budget_used": self.budget_used,
            "attack_class_metrics": self.attack_class_metrics,
        }
        state_path = d / "state.json"
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=2))
        tmp.replace(state_path)

    @classmethod
    def load(cls, state_dir: Path, run_id: str) -> AuditRunState:
        d = state_dir / run_id / "state.json"
        blob = json.loads(d.read_text())
        scope = AuditScope.from_dict(blob["scope"])
        triage_records = {
            k: TriageRecord(**v) for k, v in blob.get("triage_records", {}).items()
        }
        proposed = [FindingRecord(**f) for f in blob.get("proposed_findings", [])]
        verified = []
        for vf in blob.get("verified_findings", []):
            verified.append(
                VerifiedFinding(
                    finding=FindingRecord(**vf["finding"]),
                    correctness_reason=vf["correctness_reason"],
                    rebuttal_reason=vf["rebuttal_reason"],
                    reproducer=vf.get("reproducer"),
                    adversarial=vf.get("adversarial"),
                    trace=vf.get("trace"),
                    root_cause_group_id=vf.get("root_cause_group_id", ""),
                )
            )
        if "artifact_state" not in blob:
            raise ValueError(
                f"audit state file at {state_dir / run_id / 'state.json'} predates "
                "the artifact_state field. Re-run /audit from scratch."
            )

        hunt_tasks = {k: HuntTask(**v) for k, v in blob.get("hunt_tasks", {}).items()}
        coverage_records = {
            k: CoverageRecord(**v) for k, v in blob.get("coverage_records", {}).items()
        }
        root_cause_groups = [
            RootCauseGroup(**g) for g in blob.get("root_cause_groups", [])
        ]

        state = cls(
            run_id=blob["run_id"],
            scope=scope,
            queued_files=blob["queued_files"],
            reviewed_files=set(blob.get("reviewed_files", [])),
            triage_records=triage_records,
            candidate_files=blob.get("candidate_files", []),
            deep_reviewed_files=set(blob.get("deep_reviewed_files", [])),
            proposed_findings=proposed,
            verified_findings=verified,
            repo_profile=blob.get("repo_profile"),
            import_index=blob.get("import_index", {}),
            caller_index=blob.get("caller_index", {}),
            attack_scores=blob.get("attack_scores", {}),
            verification_state=blob.get("verification_state", {}),
            artifact_state=blob["artifact_state"],
            state_dir=state_dir,
            phase=blob.get("phase", "init"),
            metrics=blob.get("metrics", dict(_DEFAULT_METRICS)),
            select_all=bool(blob.get("select_all", False)),
            measure_triage=bool(blob.get("measure_triage", False)),
            measurement_escalated_paths=set(
                blob.get("measurement_escalated_paths", [])
            ),
            hunt_mode=bool(blob.get("hunt_mode", False)),
            hunt_tasks=hunt_tasks,
            coverage_records=coverage_records,
            adversarial_state=blob.get("adversarial_state", {}),
            root_cause_groups=root_cause_groups,
            trace_state=blob.get("trace_state", {}),
            proof_strict=bool(blob.get("proof_strict", False)),
            trace_reachability=bool(blob.get("trace_reachability", False)),
            budget_tokens=int(blob.get("budget_tokens", 0)),
            budget_used=blob.get("budget_used", {}),
            attack_class_metrics=blob.get("attack_class_metrics", {}),
        )
        return state

    @classmethod
    def find_resumable(
        cls,
        state_dir: Path,
        commit: str,
        focus: list[str] | None,
        include_done: bool = False,
    ) -> AuditRunState | None:
        if not state_dir.exists():
            return None
        want = None if focus is None else set(_normalize_focus(focus))
        best = None
        best_mtime = 0.0
        for entry in state_dir.iterdir():
            sf = entry / "state.json"
            if not sf.exists():
                continue
            try:
                blob = json.loads(sf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if blob.get("scope", {}).get("commit") != commit:
                continue
            if want is not None:
                persisted = _normalize_focus(
                    _coerce_focus(blob.get("scope", {}).get("focus"))
                )
                if set(persisted) != want:
                    continue
            if blob.get("phase") == "done" and not include_done:
                continue
            mtime = sf.stat().st_mtime
            if mtime > best_mtime:
                try:
                    candidate = cls.load(state_dir, blob["run_id"])
                except ValueError as e:
                    fmt.warning(str(e))
                    continue
                best_mtime = mtime
                best = candidate
        return best


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: str) -> str:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _resolve_scope(base_dir: str, focus: list[str]) -> AuditScope:
    branch = _git(["branch", "--show-current"], base_dir) or "HEAD"
    commit = _git(["rev-parse", "HEAD"], base_dir)
    raw = _git(["ls-tree", "-r", "--name-only", "HEAD"], base_dir)
    tracked = raw.splitlines() if raw else []

    focus = _normalize_focus(focus)
    if focus:
        tracked = [f for f in tracked if any(_match_path_glob(f, p) for p in focus)]

    mandatory = [f for f in tracked if _is_auditable(f)]
    return AuditScope(
        branch=branch,
        commit=commit,
        tracked_files=tracked,
        mandatory_files=mandatory,
        focus=focus,
    )


def _match_path_glob(file: str, glob: str) -> bool:
    """True when ``file`` (a repo-relative path) is selected by ``glob``.

    Selection rules, in order:

    - exact match: ``src/foo.rs`` selects only ``src/foo.rs``.
    - prefix match (no wildcard): ``src`` and ``src/`` both select
      anything under ``src/``.
    - pathlib wildcard match via ``PurePosixPath.full_match``: ``*``
      does *not* cross ``/``, ``?`` matches a single non-separator
      character, ``**`` matches any number of intermediate directories,
      and ``[abc]`` is a character class. A wildcard pattern with no
      ``/`` is implicitly recursive: ``*.rs`` is rewritten to
      ``**/*.rs`` so it keeps selecting every ``.rs`` file at any depth.
      This means ``src/*.rs`` matches only direct ``.rs`` children of a
      top-level ``src/``, and ``src/**/*.rs`` is the recursive form.
    """
    from pathlib import PurePosixPath

    if file == glob:
        return True

    has_wildcard = any(c in glob for c in "*?[")
    if not has_wildcard:
        prefix = glob if glob.endswith("/") else glob + "/"
        return file.startswith(prefix)

    pattern = glob.rstrip("/") or glob
    if "/" not in pattern:
        pattern = f"**/{pattern}"
    return PurePosixPath(file).full_match(pattern)


def _is_auditable(path: str) -> bool:
    return Path(path).suffix.lower() in _AUDITABLE_EXTS


def _git_show(path: str, base_dir: str) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        cwd=base_dir,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git show HEAD:{path} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode(errors="replace")


# ---------------------------------------------------------------------------
# Attack-surface scoring
# ---------------------------------------------------------------------------


def _score_attack_surface(content: str) -> int:
    score = 0
    for pattern, weight in _ATTACK_SURFACE_PATTERNS:
        if pattern.search(content):
            score += weight
    return score


def _order_by_attack_surface(
    files: list[str], content_cache: dict[str, str]
) -> tuple[list[str], dict[str, int]]:
    """Return (files ordered by descending score, score map)."""
    score_map: dict[str, int] = {}
    scored: list[tuple[int, str]] = []
    for f in files:
        score = _score_attack_surface(content_cache.get(f, ""))
        score_map[f] = score
        scored.append((-score, f))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [f for _, f in scored], score_map


# ---------------------------------------------------------------------------
# Import / caller context extraction
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(
    r"(?:"
    r"^\s*import\s+([\w.]+)"  # Python: import foo.bar
    r"|^\s*from\s+([\w.]+)\s+import"  # Python: from foo import bar
    r"|^\s*(?:const|let|var|import)\s+.*?from\s+['\"]([^'\"]+)['\"]"  # JS/TS
    r"|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"  # Node require
    r"|^\s*#include\s*[\"<]([^\"'>]+)[\"'>]"  # C/C++
    r"|^\s*use\s+([\w:\\]+)"  # Rust / PHP
    r"|@import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"  # Zig
    r")",
    re.MULTILINE,
)

_EXPORT_RE = re.compile(
    r"(?:"
    r"^def\s+(\w+)"  # Python def
    r"|^class\s+(\w+)"  # Python class
    r"|^export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)"  # JS/TS
    r"|^func\s+(\w+)"  # Go
    r"|^pub\s+fn\s+(\w+)"  # Rust/Zig
    r"|^\s*(?:public\s+)?function\s+(\w+)"  # PHP
    r")",
    re.MULTILINE,
)


def _extract_imports(content: str) -> list[str]:
    imports = []
    for m in _IMPORT_RE.finditer(content):
        imp = next((g for g in m.groups() if g), None)
        if imp:
            imports.append(imp)
    return imports


def _extract_exports(content: str) -> list[str]:
    exports = []
    for m in _EXPORT_RE.finditer(content):
        sym = next((g for g in m.groups() if g), None)
        if sym and not sym.startswith("_"):
            exports.append(sym)
    return exports


def _git_show_many(paths: list[str], base_dir: str) -> dict[str, str]:
    """Read many files from git in one ``git cat-file --batch`` process.

    Returns a ``path->content`` dict. Missing or non-blob entries are silently
    skipped, as is anything beyond a record that breaks the framing. Newlines
    in paths would corrupt the protocol, so they raise ``RuntimeError``.
    Callers that want a per-path fallback for any path the batch didn't return
    should walk the input list against the returned dict — see
    ``_load_file_contents``.
    """
    rejected = [p for p in paths if "\n" in p]
    if rejected:
        raise RuntimeError(f"refusing to batch paths with newlines: {rejected[:3]}")
    safe_paths = paths
    if not safe_paths:
        return {}

    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=base_dir,
    )

    def _writer():
        try:
            for p in safe_paths:
                proc.stdin.write(f"HEAD:{p}\n".encode())
            proc.stdin.close()
        except BrokenPipeError:
            pass

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    cache: dict[str, str] = {}
    out = proc.stdout
    try:
        for p in safe_paths:
            header = out.readline()
            if not header:
                break
            parts = header.rstrip(b"\n").split(b" ")
            if len(parts) >= 2 and parts[1] == b"missing":
                continue
            if len(parts) < 3:
                break
            objtype = parts[1]
            try:
                size = int(parts[2])
            except ValueError:
                break
            chunks: list[bytes] = []
            remaining = size
            while remaining > 0:
                chunk = out.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining > 0:
                break
            sep = out.read(1)
            if sep != b"\n":
                break
            if objtype != b"blob":
                continue
            cache[p] = b"".join(chunks).decode(errors="replace")
    finally:
        writer.join(timeout=5)
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return cache


def _load_file_contents(files: list[str], base_dir: str) -> dict[str, str]:
    """Read all files from git once and return a path->content cache.

    Tries ``git cat-file --batch`` for the whole list first. Any path the
    batch process didn't return (because the process died, a record was
    malformed and broke framing, or the path contained a newline) falls back
    to per-file ``git show``.
    """
    cache: dict[str, str] = {}
    try:
        cache = _git_show_many(files, base_dir)
    except RuntimeError:
        cache = {}

    missing = [f for f in files if f not in cache]
    for f in missing:
        try:
            cache[f] = _git_show(f, base_dir)
        except RuntimeError:
            pass
    return cache


def _repo_profile_json(state: AuditRunState) -> str:
    if state.repo_profile is None:
        return "{}"
    cached = getattr(state, "_repo_profile_json_cached", None)
    if cached is None:
        cached = json.dumps(state.repo_profile, indent=2)
        state._repo_profile_json_cached = cached
    return cached


_IDENT_RE = re.compile(r"\b\w+\b")


def _build_context_indices(
    files: list[str],
    content_cache: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    import_index: dict[str, list[str]] = {}
    export_map: dict[str, list[str]] = {}

    for f in files:
        content = content_cache.get(f)
        if content is None:
            continue
        import_index[f] = _extract_imports(content)
        for sym in _extract_exports(content):
            export_map.setdefault(sym, []).append(f)

    export_keys = set(export_map)

    caller_index: dict[str, list[str]] = {}
    for f in files:
        content = content_cache.get(f)
        if content is None:
            continue
        tokens = {m.group() for m in _IDENT_RE.finditer(content)}
        callers: set[str] = set()
        for sym in tokens & export_keys:
            sources = export_map[sym]
            if f not in sources:
                callers.update(sources)
        if callers:
            caller_index[f] = sorted(callers)

    return import_index, caller_index


# ---------------------------------------------------------------------------
# Structured-text record helpers (`@@ name @@` blocks with `key: value` lines)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordSchema:
    """Describes one record type (the body of a @@ name @@ block)."""

    name: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)
    booleans: tuple[str, ...] = ()
    repeated: dict[str, str] = field(default_factory=dict)
    multiline: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseSchema:
    """Describes the expected response shape for one audit phase."""

    record: RecordSchema
    cardinality: str  # "one" | "zero_or_more"
    allow_none: bool = False


_RECORD_HEADER_RE = re.compile(r"^\s*@@\s*([a-zA-Z_]+)\s*@@\s*$")
_RECORD_KV_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.*)$")
_RECORD_FENCE_RE = re.compile(
    r"^```(?:[\w-]+)?\s*\n(.*)\n```\s*$",
    re.DOTALL,
)
_ENTRY_POINT_HINT_RE = re.compile(
    r"(main|app|server|handler|index|cli|entry)", re.IGNORECASE
)

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".hpp": "c++",
    ".hh": "c++",
    ".cs": "c#",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".zig": "zig",
    ".d": "d",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".lua": "lua",
    ".pl": "perl",
    ".r": "r",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".nim": "nim",
    ".S": "assembly",
    ".s": "assembly",
    ".asm": "assembly",
}


def _phase1_source_inventory(tracked: list[str]) -> str:
    """Deterministic per-language file counts plus a few sample paths.

    Gives the model concrete facts for the required `language` field even when
    no recognized manifest is present in scope.
    """
    by_lang: dict[str, list[str]] = {}
    other_exts: dict[str, int] = {}
    for f in tracked:
        suffix = Path(f).suffix
        lang = _EXT_TO_LANG.get(suffix)
        if lang is None and suffix:
            other_exts[suffix] = other_exts.get(suffix, 0) + 1
            continue
        if lang is None:
            continue
        by_lang.setdefault(lang, []).append(f)

    if not by_lang and not other_exts:
        return ""

    lines = ["--- source inventory ---"]
    ranked = sorted(by_lang.items(), key=lambda kv: -len(kv[1]))
    for lang, files in ranked:
        samples = ", ".join(files[:3])
        more = f" (+{len(files) - 3} more)" if len(files) > 3 else ""
        lines.append(f"{lang}: {len(files)} file(s); examples: {samples}{more}")
    if other_exts:
        top_other = sorted(other_exts.items(), key=lambda kv: -kv[1])[:5]
        rendered = ", ".join(f"{ext}={n}" for ext, n in top_other)
        lines.append(f"other extensions: {rendered}")
    return "\n".join(lines)


def _parse_records(
    text: str,
    schema: PhaseSchema,
    *,
    metrics: dict[str, int] | None = None,
) -> list[dict]:
    """Parse @@-block records into a list of dicts, validated against schema.

    Permissive on input: case-insensitive headers and keys, optional blank
    lines between records, `:` or `=` separator, indented continuations on
    declared multiline fields, fenced code-block wrapping unwrapped, preamble
    and trailer prose tolerated.

    Strict on output: required keys must be present and non-empty, declared
    enums validated, booleans coerced, cardinality enforced, scalar duplicates
    rejected.

    When ``metrics`` is provided, ``multiline_continuations`` is incremented
    by the number of records that used at least one continuation line — but
    only after parsing and validation succeed.

    Raises ``ValueError`` on any failure with a message naming the offending
    key/record.
    """
    if not text or not text.strip():
        raise ValueError("empty LLM response")

    cleaned = text.strip()
    fence = _RECORD_FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    rec_schema = schema.record
    target_name = rec_schema.name.lower()
    known_keys: set[str] = {
        k.lower()
        for group in (
            rec_schema.required,
            rec_schema.optional,
            rec_schema.repeated,
            rec_schema.multiline,
            rec_schema.booleans,
        )
        for k in group
    }
    repeated_map = {k.lower(): v for k, v in rec_schema.repeated.items()}
    multiline_keys = {k.lower() for k in rec_schema.multiline}

    records: list[dict] = []
    current: dict | None = None
    last_key: str | None = None
    saw_none = False
    multiline_records = 0
    current_used_multiline = False

    def commit() -> None:
        nonlocal current, last_key, current_used_multiline, multiline_records
        if current is not None:
            if current_used_multiline:
                multiline_records += 1
            records.append(current)
        current = None
        last_key = None
        current_used_multiline = False

    for raw_line in cleaned.splitlines():
        if not raw_line.strip():
            last_key = None
            continue

        m = _RECORD_HEADER_RE.match(raw_line)
        if m:
            name = m.group(1).lower()
            commit()
            if name == "none":
                if not schema.allow_none:
                    raise ValueError("'@@ none @@' sentinel not permitted by schema")
                saw_none = True
                continue
            if name != target_name:
                raise ValueError(
                    f"unexpected record type '@@ {name} @@', "
                    f"expected '@@ {target_name} @@'"
                )
            current = {}
            continue

        if current is None:
            continue

        if raw_line.startswith("  "):
            if last_key is None:
                raise ValueError(
                    f"continuation line before any key in record {len(records)}"
                )
            if last_key not in multiline_keys:
                raise ValueError(
                    f"continuation on key '{last_key}' which is not multiline "
                    f"in record {len(records)}"
                )
            cont_text = raw_line.strip()
            existing = current.get(last_key, "")
            current[last_key] = f"{existing} {cont_text}" if existing else cont_text
            current_used_multiline = True
            continue

        kv = _RECORD_KV_RE.match(raw_line)
        if not kv:
            commit()
            continue

        key = kv.group(1).lower()
        value = kv.group(2).rstrip()

        if key not in known_keys:
            commit()
            continue

        if key in repeated_map:
            if value.strip():
                current.setdefault(key, []).append(value)
        else:
            if key in current:
                raise ValueError(f"duplicate key '{key}' in record {len(records)}")
            current[key] = value
        last_key = key

    commit()

    if saw_none:
        if records:
            raise ValueError("'@@ none @@' sentinel mixed with actual records")
        return []

    validated = _validate_records(records, schema, repeated_map)
    if metrics is not None and multiline_records > 0:
        metrics["multiline_continuations"] = (
            metrics.get("multiline_continuations", 0) + multiline_records
        )
    return validated


def _validate_records(
    records: list[dict],
    schema: PhaseSchema,
    repeated_map: dict[str, str],
) -> list[dict]:
    rec_schema = schema.record
    enum_lc = {k.lower(): {a.lower() for a in v} for k, v in rec_schema.enums.items()}
    enum_orig = {k.lower(): v for k, v in rec_schema.enums.items()}
    booleans = {k.lower() for k in rec_schema.booleans}

    for idx, record in enumerate(records):
        for required in rec_schema.required:
            rk = required.lower()
            if rk in repeated_map:
                vals = record.get(rk, [])
                if not vals or not any(v and v.strip() for v in vals):
                    raise ValueError(
                        f"missing required key '{required}' in record {idx}"
                    )
            else:
                val = record.get(rk, "")
                if not val or not val.strip():
                    raise ValueError(
                        f"missing required key '{required}' in record {idx}"
                    )

        for key, allowed_lc in enum_lc.items():
            if key in record:
                v = record[key]
                values = v if isinstance(v, list) else [v]
                for val in values:
                    if val.strip().lower() not in allowed_lc:
                        raise ValueError(
                            f"invalid enum value '{val}' for key '{key}' "
                            f"in record {idx}; allowed: "
                            f"{sorted(enum_orig[key])}"
                        )

        for key in booleans:
            if key in record:
                raw_val = record[key].strip().lower()
                if raw_val in ("true", "yes", "1"):
                    record[key] = True
                elif raw_val in ("false", "no", "0"):
                    record[key] = False
                else:
                    raise ValueError(
                        f"invalid boolean '{record[key]}' for key '{key}' "
                        f"in record {idx}"
                    )

    if schema.cardinality == "one":
        if len(records) != 1:
            raise ValueError(
                f"expected exactly one '@@ {rec_schema.name} @@' record, "
                f"got {len(records)}"
            )
    elif schema.cardinality == "zero_or_more":
        if not records:
            msg = f"expected at least one '@@ {rec_schema.name} @@' record"
            if schema.allow_none:
                msg += " or the '@@ none @@' sentinel"
            raise ValueError(msg)

    for record in records:
        for singular, plural in repeated_map.items():
            if singular in record:
                record[plural] = record.pop(singular)
            elif plural not in record:
                record[plural] = []

    return records


def _bump_parse_failure(metrics: dict[str, int], record_name: str) -> None:
    """Increment both the aggregate and the per-record-type parse-failure counter.

    Per-type counters (``parse_failures_<name>``) are added to the aggregate so
    a high-level reader can still see total failures, while a debugger can see
    which phase contributed.
    """
    metrics["parse_failures"] = metrics.get("parse_failures", 0) + 1
    typed_key = f"parse_failures_{record_name}"
    metrics[typed_key] = metrics.get(typed_key, 0) + 1


_PARSE_FAILURE_BREAKDOWN: tuple[tuple[str, str], ...] = (
    ("parse_failures_profile", "profile"),
    ("parse_failures_triage", "triage"),
    ("parse_failures_finding", "finding"),
    ("parse_failures_expansion", "expansion"),
)

_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("repair_successes", "repairs succeeded"),
    ("repair_failures", "repairs failed"),
    ("analytical_retries", "analytical retries"),
    ("multiline_continuations", "multiline continuations"),
)


def _format_audit_metrics(metrics: dict[str, int]) -> str:
    """Render the audit metrics dict for the phase 3 summary line.

    `parse_failures` shows the aggregate count and, when typed counters are
    populated, a parenthesized breakdown by record type. Other counters render
    as plain ``"N label"`` clauses joined by commas.
    """
    parts: list[str] = []
    pf = metrics.get("parse_failures", 0)
    if pf:
        breakdown = ", ".join(
            f"{metrics.get(k, 0)} {label}"
            for k, label in _PARSE_FAILURE_BREAKDOWN
            if metrics.get(k)
        )
        if breakdown:
            parts.append(f"{pf} parse failures ({breakdown})")
        else:
            parts.append(f"{pf} parse failures")
    for k, label in _METRIC_LABELS:
        if metrics.get(k):
            parts.append(f"{metrics[k]} {label}")
    return ", ".join(parts)


_RECORD_REPAIR_SYSTEM = """\
You are a format repair tool for structured text records.

You will receive a malformed model output that was supposed to follow a
specific @@ block @@ format, the parse error that was raised, and a worked
example of the correct format.

Rules:
- Fix only format errors: missing keys, malformed headers, wrong enum values,
  unbalanced indentation.
- Do not add, remove, or modify any factual content.
- Do not invent new records, claims, or values not present in the original.
- Output only the repaired records. No prose, no fences, no commentary."""


def _repair_records_response(
    ctx: InputContext,
    malformed: str,
    error_msg: str,
    worked_example: str,
    schema: PhaseSchema,
    metrics: dict[str, int] | None = None,
) -> list[dict]:
    record_name = schema.record.name
    user_msg = (
        f"Parse error: {error_msg}\n\n"
        f"Required format example:\n{worked_example}\n\n"
        f"Malformed output:\n{malformed}\n\n"
        f"Re-emit the same content using the exact format shown above. "
        f"Do not introduce new {record_name} records, do not add new claims, "
        f"and do not change any factual content; only fix the format."
    )
    messages = [
        {"role": "system", "content": _RECORD_REPAIR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = _call_audit_llm(
        ctx, messages, trace_task=f"audit: records repair {record_name}"
    )
    return _parse_records(raw, schema, metrics=metrics)


def _parse_records_with_repair(
    ctx: InputContext,
    raw: str,
    schema: PhaseSchema,
    worked_example: str,
    metrics: dict[str, int],
) -> list[dict]:
    """Parse records, falling back to one LLM repair pass on failure."""
    try:
        result = _parse_records(raw, schema, metrics=metrics)
        _debug_log("records_parse_ok", record_type=schema.record.name)
        return result
    except ValueError as e:
        _bump_parse_failure(metrics, schema.record.name)
        error_msg = str(e)
        _debug_log(
            "records_parse_failed",
            record_type=schema.record.name,
            error=error_msg,
            raw_len=len(raw),
            raw_preview=raw[:3000],
        )
        fmt.info(f"  parse failed ({error_msg}), attempting repair...")
        try:
            result = _repair_records_response(
                ctx, raw, error_msg, worked_example, schema, metrics=metrics
            )
            metrics["repair_successes"] = metrics.get("repair_successes", 0) + 1
            _debug_log("records_repair_ok", record_type=schema.record.name)
            fmt.info("  repair succeeded")
            return result
        except ValueError as repair_err:
            metrics["repair_failures"] = metrics.get("repair_failures", 0) + 1
            _debug_log(
                "records_repair_failed",
                record_type=schema.record.name,
                error=str(repair_err),
            )
            fmt.info("  repair failed")
            raise


# ---------------------------------------------------------------------------
# Trace helper
# ---------------------------------------------------------------------------


def _write_audit_trace(
    ctx: InputContext, messages: list, task: str | None = None
) -> None:
    trace_dir = getattr(ctx, "trace_dir", None)
    if not trace_dir or not messages:
        return
    from .traces import write_trace_to_dir

    write_trace_to_dir(
        messages,
        trace_dir=trace_dir,
        base_dir=ctx.base_dir,
        model=ctx.loop_kwargs.get("model_id", "unknown"),
        task=task,
    )


# ---------------------------------------------------------------------------
# Direct LLM call wrapper
# ---------------------------------------------------------------------------


def _call_audit_llm(
    ctx: InputContext,
    messages: list[dict],
    temperature: float | None = None,
    trace_task: str | None = None,
    model_override: str | None = None,
) -> str:
    from .agent import call_llm, ContextOverflowError
    from ._msg import _msg_content

    kw = ctx.loop_kwargs
    llm_kwargs = kw.get("llm_kwargs", {})
    effective_model = model_override or kw["model_id"]

    cache_info = None

    def _do_call(msgs):
        nonlocal cache_info
        msg, _finish, _activity, _retries, cache_info = call_llm(
            kw["api_base"],
            effective_model,
            msgs,
            kw.get("max_output_tokens"),
            temperature,
            kw.get("top_p"),
            kw.get("seed"),
            None,  # tools
            False,  # verbose
            provider=llm_kwargs.get("provider", "lmstudio"),
            api_key=llm_kwargs.get("api_key"),
            user_agent=llm_kwargs.get("user_agent"),
            prompt_cache=True,
            aws_profile=llm_kwargs.get("aws_profile"),
        )
        return _msg_content(msg) or ""

    user_msg = messages[-1]
    original_text = user_msg.get("content", "")
    limit = len(original_text)
    messages = list(messages)
    overflowed_once = False

    _debug_log(
        "llm_request",
        task=trace_task,
        system=messages[0].get("content", "")[:500] if messages else "",
        user_len=len(original_text),
    )

    empty_retries = 0
    while True:
        try:
            content = _do_call(messages)
            if not content and limit > _AUDIT_TRUNCATION_FLOOR:
                empty_retries += 1
                if empty_retries > 3:
                    break
                _debug_log("llm_empty", task=trace_task, limit=limit)
                limit = limit // 2
                messages[-1] = {
                    **user_msg,
                    "content": original_text[:limit] + _AUDIT_TRUNCATION_MARKER,
                }
                continue
            break
        except ContextOverflowError:
            _debug_log("llm_overflow", task=trace_task, limit=limit)
            if not overflowed_once:
                overflowed_once = True
                _write_audit_trace(
                    ctx,
                    messages
                    + [
                        {
                            "role": "assistant",
                            "content": "[context overflow — retrying with truncation]",
                        }
                    ],
                    task=(trace_task + " (overflow)" if trace_task else "overflow"),
                )
            limit = limit // 2
            if limit < _AUDIT_TRUNCATION_FLOOR:
                raise
            messages[-1] = {
                **user_msg,
                "content": original_text[:limit] + _AUDIT_TRUNCATION_MARKER,
            }

    _debug_log(
        "llm_response",
        task=trace_task,
        response_len=len(content),
        response_preview=content[:2000],
        cache=cache_info,
    )

    trace_messages = list(messages) + [{"role": "assistant", "content": content}]
    _write_audit_trace(ctx, trace_messages, task=trace_task)
    return content


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PHASE1_PROFILE_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="profile",
        required=("language", "summary"),
        repeated={
            "language": "languages",
            "framework": "frameworks",
            "entry_point": "entry_points",
            "trust_boundary": "trust_boundaries",
            "persistence_layer": "persistence_layers",
            "auth_surface": "auth_surfaces",
            "dangerous_operation": "dangerous_operations",
        },
    ),
    cardinality="one",
    allow_none=False,
)

_PHASE1_WORKED_EXAMPLE = """\
@@ profile @@
language: python
language: rust
framework: pytest
framework: uv
entry_point: swival/agent.py
entry_point: swival/audit.py
trust_boundary: cli args
trust_boundary: mcp servers
trust_boundary: llm provider responses
persistence_layer: .swival/HISTORY.md
persistence_layer: sqlite cache
auth_surface: chatgpt oauth device flow
dangerous_operation: subprocess
dangerous_operation: file write
dangerous_operation: eval-like template
summary: a python cli coding agent with mcp client and audit pipeline."""

_PHASE1_SYSTEM = f"""\
You are preparing a compact repository profile for a staged security audit.

This phase does not find bugs.
Its only job is to extract reusable repository facts that improve later review.

You have no tools, no shell access, and no ability to run commands.
All the source code you need is included below. Do not request additional information.

Output format: a single `@@ profile @@` block. The keys, one per line:
- language: one language per line; repeat the line for each
- framework: one framework or build/test tool per line; repeat the line for each
- entry_point: one repo-relative path per line; repeat the line for each
- trust_boundary: one short string per line; repeat the line for each
- persistence_layer: one short string per line; repeat the line for each
- auth_surface: one short string per line; repeat the line for each
- dangerous_operation: one short string per line; repeat the line for each
- summary: one line, under 120 words

Use exactly the keys shown. Do not quote, escape, or wrap values. Each value
runs to the end of its line. Omit a repeated key entirely when there is
nothing to add for it; do not emit empty values. At least one `language` line
and a non-empty `summary` are required.

Worked example:

{_PHASE1_WORKED_EXAMPLE}

End of example. Now produce the real profile for the repository below.

Rules:
- Use only the provided committed repository evidence.
- Do not speculate.
- Do not mention findings, vulnerabilities, or risks unless needed to describe a trust boundary.
- Keep every field short and reusable in later prompts."""

_PHASE2_TRIAGE_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="triage",
        required=("priority", "confidence", "summary"),
        enums={
            "priority": ("ESCALATE_HIGH", "ESCALATE_MEDIUM", "SKIP"),
            "confidence": ("high", "medium", "low"),
        },
        booleans=("needs_followup",),
        repeated={
            "bug_class": "bug_classes",
            "relevant_symbol": "relevant_symbols",
            "suspicious_flow": "suspicious_flows",
        },
    ),
    cardinality="one",
    allow_none=False,
)

_PHASE2_WORKED_EXAMPLE = """\
@@ triage @@
priority: ESCALATE_MEDIUM
confidence: medium
summary: parses untrusted url before authentication checks
bug_class: input_validation
bug_class: trust_boundary_breaks
relevant_symbol: parse_url
relevant_symbol: authenticate
suspicious_flow: external request body reaches parse_url before any auth gate
needs_followup: false"""

_PHASE2_SYSTEM = f"""\
You are performing phase 2 security triage for one committed file with its direct local context.

Goal:
- decide whether this file deserves deep review
- optimize for precision over recall
- avoid false positives and confirmation bias

Allowed priority labels:
- ESCALATE_HIGH
- ESCALATE_MEDIUM
- SKIP

Scope is **security only**. This is not a general code-quality reviewer. A bug is in scope only when an untrusted actor (remote client, malicious peer, attacker-controlled file or backend, lower-privileged local user) has a concrete trigger that yields a concrete security-relevant outcome under today's code. Acceptable outcomes are limited to:
- denial of service triggered by attacker-controlled input (not by admin misconfiguration or normal shutdown sequencing)
- information disclosure (secrets, tokens, internal addresses, cross-tenant data, memory contents)
- integrity bypass an attacker can observe or weaponize (HTTP smuggling, cache poisoning, log/header injection, signature/auth desync)
- authentication bypass, authorization bypass, or privilege escalation
- arbitrary code or command execution (remote, local, or in-process)
- repudiation (attacker hides their actions from logs in a way that defeats audit)

Narrow exception — security_control_failure: a high or critical, deterministic logic bug **inside the code that itself implements a named security control** is in scope even when the attacker is implicit. The function's job must be the control: a signature or MAC verifier, an authentication check, an authorization decision, a sandbox or seccomp boundary, an access-control filter, a crypto primitive, or a parser whose declared purpose is accepting hostile input. The "gain" is the control failing open or fail-permissive. This exception does NOT apply to code that is merely called from a security path, that handles untrusted data without being the decision point, or where the control name is generic ("validation", "checks").

Out of scope and must SKIP:
- function-contract bugs with no observable adversarial outcome (a function returns success when it should not, but no untrusted caller benefits and no named security control fails open)
- shutdown-time hangs, missed wakeups, leaked locks during teardown — these are correctness issues unless a remote attacker can both trigger them and gain something
- resource-lifecycle, error-handling, data-integrity, concurrency, or invariant-violation bugs that lack a concrete untrusted trigger and a concrete attacker gain, and do not make a named security control fail open
- protocol-framing or short-write correctness bugs where the only victim is the same trust domain as the input (server-to-server within the same operator)
- DoS that requires an admin or operator to author the malicious config or input — the operator is trusted
- generic robustness, missing tests, defensive hardening, "this should also validate X" suggestions

Do NOT escalate defense-in-depth concerns. A finding is defense-in-depth (and must SKIP) when an existing control in the code already prevents the attack and the proposal is to add a redundant secondary control, tighten an already-sufficient check, or harden against an attacker who has no real path to reach the code.

Before escalating, you must be able to name three things: who the attacker is (and why they are untrusted), what they control as input, and what they gain. If the finding is a security_control_failure, "attacker" may be any caller of the control and "trigger" may be any input the control is meant to reject — but you must still name the specific control by purpose (e.g., "Ed25519 signature verifier", "JWT audience check", "seccomp policy enforcer"). If you cannot, SKIP.

Security review lenses to consider. These are hints, not sufficient reasons to escalate:
- project_specific_invariant_break
- authorization
- input_validation
- path_traversal
- command_execution
- serialization
- arithmetic
- cryptography
- secret_lifecycle
- memory_safety
- overflows
- injection
- protocol_desynchronization
- parser_differential
- canonicalization_mismatch
- stale_authorization_cache
- fail_open_fallback
- capability_confusion
- namespace_confusion
- downgrade_or_policy_bypass
- trust_boundary_breaks
- unsafe_data_flow
- dangerous_api_misuse
- business_logic
- cross_component_contracts
- sandbox_escapes
- taxonomy_free_unknowns_with_security_impact

You have no tools, no shell access, and no ability to run commands.
All the source code you need is included below. Do not request additional information.

Output format: a single `@@ triage @@` block with these keys, one per line:
- priority: ESCALATE_HIGH | ESCALATE_MEDIUM | SKIP
- confidence: high | medium | low
- summary: short string, max 25 words
- bug_class: one taxonomy string per line; repeat the line for each class
- relevant_symbol: one symbol name per line; repeat the line for each symbol
- suspicious_flow: one short string per line; repeat the line for each flow
- needs_followup: true or false

Use exactly the keys shown. Do not quote, escape, or wrap values. Each value
runs to the end of its line. Omit `bug_class`, `relevant_symbol`, and
`suspicious_flow` lines entirely when there is nothing to add for them — do
not emit empty values.

Worked example:

{_PHASE2_WORKED_EXAMPLE}

End of example. Now produce the real triage for the file below.

Rules:
- Use only repository-grounded reasoning.
- Do not declare that a bug is proven in this phase.
- Prefer SKIP if escalation cannot be justified.
- Use ESCALATE_HIGH only when the evidence bundle contains a concrete suspicious path or invariant break worth deep review."""


_PHASE2_CONFIRM_SYSTEM = (
    _PHASE2_SYSTEM
    + """

This is a confirmation pass. The first triage marked the file SKIP with low
confidence. You now have additional evidence (file-level dependencies plus,
when available, the contents of the highest-attack-surface dependency, and
any direct entry-point reachability hint). Re-evaluate. Do not flip to
ESCALATE unless the new evidence reveals a concrete attacker-controlled path
that was not visible before."""
)

_SEVERITIES = ("low", "medium", "high", "critical")

_PHASE3A_FINDING_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="finding",
        required=(
            "title",
            "severity",
            "location",
            "attacker",
            "trigger",
            "impact",
            "claim",
        ),
        enums={"severity": _SEVERITIES},
    ),
    cardinality="zero_or_more",
    allow_none=True,
)

_PHASE3A_WORKED_EXAMPLE = """\
@@ finding @@
title: attacker-controlled length wraps allocation size
severity: high
location: src/decode.c:142
attacker: unauthenticated client sending a crafted request
trigger: header_len and payload_len are added before allocation
impact: heap write past the allocation, crashing or corrupting the request worker
claim: crafted lengths wrap the allocation size"""

_PHASE3A_SYSTEM = f"""\
You are performing phase 3 deep security review for one candidate file.

Review only the committed repository evidence provided.
Reject any claim that is not fully proven.

You have no tools, no shell access, and no ability to run commands.
All the source code you need is included below. Do not request additional information.

Output format: one or more `@@ finding @@` blocks. Each block has these keys, one per line:
- title: short title
- severity: low | medium | high | critical
- location: path:line
- attacker: specific untrusted actor, under 15 words
- trigger: attacker-controlled input or action reaching the bug, under 20 words
- impact: concrete security outcome, under 20 words
- claim: one-line bug statement under 20 words

Use exactly the keys shown. Do not quote, escape, or wrap values. Each value
runs to the end of its line.

If there are no findings, output the single line:

@@ none @@

Worked example:

{_PHASE3A_WORKED_EXAMPLE}

End of example. Now produce the real findings for the file below.

Rules:
- Report zero findings rather than a speculative finding.
- Every finding must be provable from the provided repository evidence.
- At most 3 findings per file.
- Each claim under 20 words.
- Prefer the narrowest bug that the evidence directly proves.
- For undefined behavior or uninitialized-state bugs, describe the direct invariant violation or invalid read/write instead of assuming a specific runtime value or deterministic branch outcome unless the evidence proves it.
- Use exact path:line citations.
- Do not include best practices, missing tests, or generic hardening advice.
- The attacker, trigger, and impact fields are required evidence gates. Do not
  use vague placeholders like "user", "input", "crash", or "could be bad".
- Spend review effort on project-specific security invariants and uncommon
  flaws: parser differential behavior, canonicalization mismatches,
  cross-component contract drift, fail-open fallback paths, stale authorization
  caches, namespace or capability confusion, protocol desynchronization,
  downgrade paths, and privilege-boundary mistakes. Do not stop at common
  checklist categories.

Scope gate — apply to every candidate finding before emitting it:

A finding is in scope only if you can answer all three of these from the evidence, in one breath:
  1. Attacker: who is the untrusted actor? (remote client, malicious peer, attacker-controlled file/backend, lower-privileged local user)
  2. Trigger: what input or action under their control reaches the bug?
  3. Gain: which security-relevant outcome do they get? — limited to denial of service triggered by attacker input, information disclosure, integrity bypass an attacker can weaponize (smuggling, cache poisoning, log/header injection, auth desync), authentication or authorization bypass, privilege escalation, code/command execution, or repudiation.

If the answer to any of the three is missing, vague, or only achievable by an admin/operator/trusted process, omit the finding and emit `@@ none @@` for the file (or whatever findings remain) — UNLESS the carve-out below applies.

Narrow exception — security_control_failure: a high or critical, deterministic logic bug **inside code that itself implements a named security control** is in scope even when the attacker is implicit. The cited function or module must be the security decision point: a signature/MAC verifier, an authentication check, an authorization decision, a sandbox or seccomp boundary, an access-control filter, a crypto primitive, or a parser whose declared purpose is accepting hostile input. For these findings:
  - attacker may read "any caller of the control"
  - trigger may read "input the control is required to reject"
  - impact must read "<named control> fails open: <observable consequence>"
  - severity must be high or critical
  - the claim must name the specific control by purpose ("Ed25519 signature verifier", "JWT audience check", "seccomp filter compiler", not "validation" or "input handling")
This exception does NOT apply to code that is merely called from a security path, that handles untrusted data without being the decision point, or where the control name is generic.

Explicitly out of scope — emit nothing for these even when the bug is real:
- function-contract violations in helpers where no named security control fails open
- shutdown-time hangs, missed wakeups during teardown, leaked locks on cleanup paths
- resource-lifecycle, error-handling, data-integrity, concurrency, or invariant-violation bugs that lack an untrusted trigger AND do not make a named security control fail open
- protocol-framing, short-write, or partial-IO bugs whose only effect is local correctness within the same trust domain
- DoS that requires an admin or operator to author the malicious config, regex, or input
- defense-in-depth: "an additional check would be safer", "this should also validate X", "this could leak in some other deployment", "redundant guard missing"

Only report bugs where the structured fields specifically answer attacker / trigger / gain — for example "unauthenticated open redirect on failed form auth" or "out-of-bounds read on attacker-supplied trailing CR" — or where the security_control_failure carve-out lets the cited function's own job (an Ed25519 verifier, a seccomp policy enforcer) supply the security context."""

# ---------------------------------------------------------------------------
# Phase Hunt prompts (replaces Phase 3a for `/audit --hunt`)
# ---------------------------------------------------------------------------

_PHASE_HUNT_FINDING_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="finding",
        required=(
            "title",
            "severity",
            "location",
            "local_bug",
            "source_boundary",
            "sink_operation",
            "reachability_status",
            "claim",
        ),
        enums={
            "severity": _SEVERITIES,
            "reachability_status": (
                "reachable",
                "local_only",
                "unknown",
                "not_reachable",
            ),
        },
        repeated={
            "controlled_input": "controlled_inputs",
            "reachability_step": "reachability_path",
            "exploit_chain_step": "exploit_chain",
        },
    ),
    cardinality="zero_or_more",
    allow_none=True,
)

_PHASE_HUNT_COVERAGE_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="coverage",
        required=("confidence",),
        enums={"confidence": ("high", "medium", "low")},
        repeated={
            "observed_file": "observed_files",
            "observed_symbol": "observed_symbols",
            "observed_edge": "observed_edges",
            "not_covered": "not_covered",
            "followup_task": "followup_tasks",
        },
    ),
    cardinality="zero_or_one",
    allow_none=True,
)

_PHASE_HUNT_WORKED_EXAMPLE = """\
@@ finding @@
title: callback_url reaches outbound fetch without host allowlist
severity: high
location: src/clients/fetch.py:42
local_bug: outbound HTTP GET trusts caller-supplied URL with no host or scheme check
source_boundary: POST /webhook handler
controlled_input: callback_url JSON field
reachability_step: POST /webhook
reachability_step: handle_webhook
reachability_step: client.fetch_remote
sink_operation: requests.get(url)
reachability_status: reachable
exploit_chain_step: attacker sends POST /webhook with callback_url=http://169.254.169.254/
exploit_chain_step: server retrieves cloud metadata
claim: SSRF: attacker controls callback_url end-to-end into requests.get

@@ coverage @@
observed_file: src/api/webhook.py
observed_file: src/clients/fetch.py
observed_symbol: handle_webhook
observed_symbol: fetch_remote
observed_edge: src/api/webhook.py->src/clients/fetch.py
not_covered: did not inspect retry path in src/clients/retry.py
followup_task: ssrf | unauthenticated remote client | retry URL parameter | retry path uncovered
confidence: medium"""

_PHASE_HUNT_SYSTEM = f"""\
You are a vulnerability hunter running one narrow task in a staged audit.

Each task gives you exactly one attack class, one attacker model, and a small
evidence bundle. Do not broaden the search. Do not look for other bug classes
on this task; that work belongs to other tasks.

You have no tools, no shell access, and no ability to run commands.
All the source code you need is included below. Do not request additional information.

Output format:
- Zero, one, two, or three `@@ finding @@` blocks. If you find no bug, emit the
  single line `@@ none @@` in place of any finding blocks.
- Then exactly one `@@ coverage @@` block, even when you emitted `@@ none @@`.

Each `@@ finding @@` block has these keys, one per line. Use exactly the keys
shown. Do not quote, escape, or wrap values. Each value runs to the end of its
line. Omit repeated keys entirely when there is nothing to add for them.

- title: short attacker-anchored title
- severity: low | medium | high | critical
- location: path:line of the sink
- local_bug: one short sentence describing the invariant break or unsafe sink
- source_boundary: where attacker-controlled data first enters the code, exact
  named entry point or trust boundary
- controlled_input: each attacker-controlled input (parameter, header, file,
  field) on its own line
- reachability_step: each step in the data path from source to sink on its own
  line, in order; aim for one line per file or function
- sink_operation: the exact dangerous operation reached (e.g. `requests.get`,
  `subprocess.run`, `pickle.loads`)
- reachability_status: reachable | local_only | unknown | not_reachable
- exploit_chain_step: each concrete attacker action on its own line, in order
- claim: one-line bug statement under 20 words

Rules:
- Investigate exactly one attack class for exactly one attacker model.
- Use the supplied seed files and symbols first; only follow bounded edges
  from the dependency graph.
- Every finding must answer: who is the attacker, what input do they control,
  what is the reachable security-relevant outcome.
- Report zero findings rather than a speculative finding.
- At most three findings per task.
- If you cannot prove attacker reachability from the named source boundary,
  set `reachability_status: local_only` and let the harness queue a reachability
  task. Do not invent reachability.

The `@@ coverage @@` block has these keys, one per line:
- observed_file: each file you actually inspected, one per line
- observed_symbol: each symbol or function you examined, one per line
- observed_edge: dataflow or import edges you followed, as `src->dst`, one per
  line
- not_covered: each scope item you did not inspect, one per line; use this
  field aggressively — under-claiming coverage is safer than overclaiming
- followup_task: each suggested next task as
  `attack_class | attacker_position | controlled_input | scope_hint | reason`,
  one per line; omit when nothing comes to mind
- confidence: high | medium | low — your own honest read of how much of the
  task scope you actually examined

Do not certify broad coverage in prose. The `not_covered` and `confidence`
fields are the only coverage signal the harness trusts.

Worked example:

{_PHASE_HUNT_WORKED_EXAMPLE}

End of example. Now run the real hunt for the task below."""


_PHASE3B_OUT_OF_SCOPE_TYPE = "out-of-scope"
_PHASE3B_SECURITY_CONTROL_FAILURE_TYPE = "security_control_failure"

_PHASE3B_SECURITY_TYPE_VALUES = (
    "denial of service",
    "information disclosure",
    "memory corruption",
    "out-of-bounds read",
    "out-of-bounds write",
    "authentication bypass",
    "authorization bypass",
    "privilege escalation",
    "code execution",
    "command execution",
    "injection",
    "header injection",
    "open redirect",
    "ssrf",
    "request smuggling",
    "cache poisoning",
    "log injection",
    "cryptographic flaw",
    "path traversal",
    "policy bypass",
    "repudiation",
    "cross-tenant isolation break",
    "sandbox escape",
    _PHASE3B_SECURITY_CONTROL_FAILURE_TYPE,
)

_PHASE3B_TYPE_VALUES = _PHASE3B_SECURITY_TYPE_VALUES + (_PHASE3B_OUT_OF_SCOPE_TYPE,)

_SECURITY_CONTROL_FAILURE_MIN_SEVERITIES = _SEVERITIES[_SEVERITIES.index("high") :]

_PHASE3B_EXPANSION_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="expansion",
        required=(
            "type",
            "attacker",
            "trigger",
            "impact",
            "preconditions",
            "proof",
            "fix_outline",
        ),
        enums={"type": _PHASE3B_TYPE_VALUES},
        multiline=("proof",),
    ),
    cardinality="one",
    allow_none=False,
)

_PHASE3B_WORKED_EXAMPLE = """\
@@ expansion @@
type: code execution
attacker: unauthenticated http client
trigger: query-string parameter reaches shell command construction
impact: arbitrary command execution as the server user
preconditions: caller passes a user-supplied string as the cmd argument
proof:
  The public request handler passes the parameter to run_shell at line 400
  without validation. At line 412 it reaches subprocess(shell=true), so shell
  metacharacters execute under the server account.
fix_outline: pass argv list and drop shell=true, or shlex.quote each segment"""

_PHASE3B_SECURITY_CONTROL_FAILURE_TEMPLATE = """\
type: security_control_failure
attacker: any caller of the signature verifier
trigger: signature buffer whose final byte is zero
impact: Ed25519 signature verifier fails open and accepts forged signatures
preconditions: caller invokes verify_sig with a 64-byte signature buffer
proof:
  verify_sig at crypto/ed25519.c:88 implements Ed25519 signature verification.
  At line 102 it returns 0 (accept) when an early return on sig[63] == 0
  short-circuits the compare loop, so any forged signature whose last byte is
  zero is accepted as valid. The function is the verification decision point;
  no other check guards its callers.
fix_outline: remove the early return and complete the constant-time compare"""

_PHASE3B_SYSTEM = f"""\
You are expanding one security finding with proof details.

You have no tools, no shell access, and no ability to run commands.
All the source code you need is included below. Do not request additional information.

Output format: a single `@@ expansion @@` block with these keys, one per line:
- type: one of {", ".join(_PHASE3B_TYPE_VALUES)} - pick the security impact label, never a generic-correctness label
- attacker: specific untrusted actor, under 15 words
- trigger: attacker-controlled input or action reaching the bug, under 20 words
- impact: concrete security outcome, under 20 words
- preconditions: minimum justified preconditions, under 20 words
- proof: propagation path, failing operation, and reachability - under 100
  words total. The proof value may span multiple lines: any line that begins
  with two or more spaces continues the proof value.
- fix_outline: smallest correct fix, under 20 words

Use exactly the keys shown. Do not quote, escape, or wrap values. Every value
runs to the end of its line, and only `proof:` may continue on indented lines.

Worked example:

{_PHASE3B_WORKED_EXAMPLE}

End of example. Now produce the real expansion for the finding below.

Rules:
- Use only the provided repository evidence.
- Prefer the narrowest bug that the evidence directly proves.
- For undefined behavior or uninitialized-state bugs, describe the direct invariant violation.
- Do not speculate beyond what the code proves.
- The attacker, trigger, and impact fields must restate the security scope in
  concrete terms. For every type other than `security_control_failure`, all
  three must point to a specific untrusted actor, an input or action under
  their control, and a security-relevant outcome they observe.
- The `type` value must be a security impact label. Do not pick a label
  whose closest plain-English fit would be "logic error", "data integrity",
  "error handling", "resource lifecycle", or "invariant violation"; if the bug
  matches one of those without further security framing, it is out of scope.

`security_control_failure` rules — apply only when the cited code itself
implements a named security control:
- Severity must be `high` or `critical`. Lower-severity control failures are
  out of scope for this exception.
- The proof must name the specific control by its security purpose
  (for example: "Ed25519 signature verifier", "JWT audience authorization
  check", "seccomp filter compiler", "TLS hostname matcher"). Generic phrases
  like "validation", "input handling", "the parser" are not enough.
- The proof must show a deterministic path on which the control returns
  accept/allow/true for an input it must reject, or returns the wrong
  decision in a way an attacker can rely on. Speculative fail-open paths,
  partial-coverage gaps, and "could be hardened" notes do not qualify.
- attacker may read "any caller of the control"; trigger may read
  "input the control is required to reject"; impact must read
  "<named control> fails open: <observable consequence>".
- Do NOT use this type when the function only happens to be called from a
  security-sensitive path, when the bug is a robustness gap, when the proof
  is a contract violation in a helper that no security decision depends on,
  or when untrusted data merely passes through the function on its way
  somewhere else.

A correctly-shaped `security_control_failure` block looks like:

{_PHASE3B_SECURITY_CONTROL_FAILURE_TEMPLATE}

If the candidate is a real bug but out of security scope, still emit a valid
block so the pipeline can discard it:
  type: out-of-scope
  attacker: missing
  trigger: missing
  impact: missing
  preconditions: out-of-scope
  proof: out-of-scope because attacker, trigger, or security impact is not proven
  fix_outline: no security fix"""

_PHASE4_VERIFY_SYSTEM = """\
You are verifying one proposed security finding using the committed source code in an isolated worktree.

Your job is to determine whether the finding describes a real bug that can be triggered in practice. Treat the finding as a hypothesis, not as ground truth.

Rules:
- You may inspect the code only, or you may compile/run small proof-of-concept code if that helps.
- Use the committed source evidence and the isolated worktree only.
- If the exact claim is wrong but the evidence proves a narrower directly source-grounded local bug, prove that narrower bug instead.
- A proof counts if you can identify the trigger, reachability conditions, propagation path, failing operation or violated invariant, and practical impact from the code, or demonstrate equivalent runtime evidence.
- For undefined behavior, uninitialized-state, and memory-safety bugs, either source-based proof or convincing runtime evidence is acceptable.
- Reject only when the code does not support a practical trigger path.
- Reject defense-in-depth findings as NOTREPRODUCED. If an existing control in the committed code already blocks the attack, the finding is not real — even if adding another layer would be safer. The bar is a concrete attacker with a concrete input producing a concrete security-relevant outcome under today's code, not "this would be more robust if it also did X".
- Treat the proposed finding's attacker, trigger, and impact text as claims to
  verify, not as facts. A finding that reproduces a correctness failure but not
  those three security claims is NOTREPRODUCED.
- Reject as NOTREPRODUCED any finding that fails the security scope test, even if the underlying bug is real:
  - You must be able to name an untrusted actor (remote client, malicious peer, attacker-controlled file or backend, lower-privileged local user) who controls the trigger. If the only way to reach the bug is through an admin, operator, or otherwise trusted process authoring the malicious config, regex, or input, the finding is out of scope.
  - You must be able to name a security-relevant outcome from this list: denial of service triggered by attacker-controlled input, information disclosure, integrity bypass an attacker can weaponize (smuggling, cache poisoning, log/header injection, auth desync), authentication or authorization bypass, privilege escalation, arbitrary code/command execution, or repudiation. "The function returns success when it should not", "the protocol can desync between two trusted endpoints", "a worker thread hangs at shutdown", and "a missed wakeup leaves a producer asleep" are not security-relevant outcomes on their own.
  - Generic correctness, contract, robustness, error-handling, resource-lifecycle, missed-wakeup, partial-IO, or invariant-violation bugs without an untrusted trigger AND a concrete attacker gain must be rejected as NOTREPRODUCED. Reproducibility of the bug behavior is not enough; the security impact must also reproduce.
- Narrow exception — `security_control_failure` findings: a high or critical, deterministic logic bug **inside code that itself implements a named security control** is in scope even when the attacker is implicit. Verify these claims, in order, before accepting:
  1. The cited function or module **is itself the security control** named in the proof — a signature/MAC verifier, an authentication check, an authorization decision, a sandbox or seccomp boundary, an access-control filter, a crypto primitive, or a parser whose declared purpose is accepting hostile input. Not "called from a security path", not "handles untrusted data on the way somewhere else".
  2. The proof names that specific control by purpose. Generic phrases like "validation", "input handling", or "the parser" do not satisfy this — reject as NOTREPRODUCED.
  3. The bug deterministically makes that control return accept/allow/true for an input it must reject (or otherwise return the wrong security decision). Speculative fail-open paths, partial-coverage gaps, and "this would be more robust if" arguments do not qualify — reject as NOTREPRODUCED.
  4. Severity is high or critical. If a proposed `security_control_failure` is low or medium severity, reject as NOTREPRODUCED. Do not reclassify it under the standard scope test during verification — a lower-severity bug with a concrete attacker, trigger, and gain belongs under its concrete impact type (`authorization bypass`, `information disclosure`, etc.), and that reclassification is phase 3B's job, not yours.
  If all four hold, treat the finding as REPRODUCED even though the attacker is "any caller of the control" and the trigger is "input the control is required to reject". If any one fails, reject as NOTREPRODUCED — including the case where the underlying logic bug is real but the function is merely security-adjacent rather than the control itself.
- End your final response with exactly one fenced proof block using the
  ``swival-audit-proof-v1`` sentinel. The harness rejects any other format.
  Emit nothing after the closing fence.

The block layout is:

```swival-audit-proof-v1
verdict: REPRODUCED | NOTREPRODUCED
proof_kind: runtime | source | mixed
commands:
  - shell command actually executed, repeatable, or none
artifacts:
  - generated PoC or proof file path under the worktree, repeatable, or none
observed_output: short observed output or specific source citation
trigger: attacker-controlled input or action that drives the bug
impact: demonstrated security outcome (DoS, RCE, info disclosure, etc.)
limitations: narrow caveats, or none
```

Rules for the block:
- ``verdict`` is REPRODUCED only when the proof actually demonstrates the
  attacker, trigger, and security-relevant impact under today's code.
- ``proof_kind`` is ``runtime`` for executed PoCs, ``source`` for source-only
  proofs (acceptable for security_control_failure and simple logic bugs only),
  and ``mixed`` when both apply.
- Use ``none`` (lowercase) for empty lists.
- For ``NOTREPRODUCED``, ``proof_kind`` is optional: leave it empty or write
  ``none`` if no proof attempt was made; otherwise set it to ``runtime``,
  ``source``, or ``mixed`` describing what you tried. The other fields should
  explain what the proof attempt found instead.
- Do not embed nested ``swival-audit-proof-v1`` markers anywhere in the
  block content."""

_PHASE5_REPORT_TEMPLATE = """\
You are writing the final markdown report for one reproduced and patched finding.

Use exactly this structure:
- # <short finding title>
- ## Classification
- ## Affected Locations
- ## Summary
- ## Provenance
- ## Preconditions
- ## Proof
- ## Why This Is A Real Bug
- ## Fix Requirement
- ## Patch Rationale
- ## Residual Risk
- ## Patch

You have no tools, no shell access, and no ability to run commands.
All the information you need is included below. Do not request additional information.
Your entire response must be a single markdown document and nothing else.

Rules:
- Confidence must be certain.
- Provenance must include a link to the Swival.dev Security Scanner URL: {provenance_url}
- Residual Risk must be `None` unless a narrow evidence-based concern remains.
- Be terse, factual, and evidence-driven."""

# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _phase1_repo_profile(state: AuditRunState, ctx: InputContext) -> dict:
    """Build a compact repository profile from committed evidence."""
    evidence_parts = []
    inventory = _phase1_source_inventory(state.scope.tracked_files)
    if inventory:
        evidence_parts.append(inventory)

    manifest_names = {
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Makefile",
        "Makefile.am",
        "Makefile.in",
        "configure.ac",
        "CMakeLists.txt",
        "meson.build",
        "build.zig",
        "build.zig.zon",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "pom.xml",
        "build.gradle",
        "Gemfile",
    }
    for f in state.scope.tracked_files:
        if Path(f).name in manifest_names:
            try:
                content = _git_show(f, ctx.base_dir)
                evidence_parts.append(f"--- {f} ---\n{content[:2000]}")
            except RuntimeError:
                pass

    entry_hints = [
        f
        for f in state.scope.mandatory_files
        if _ENTRY_POINT_HINT_RE.search(Path(f).stem)
    ]
    for f in entry_hints[:5]:
        try:
            content = _git_show(f, ctx.base_dir)
            evidence_parts.append(
                f"--- {f} (entry point candidate) ---\n{content[:3000]}"
            )
        except RuntimeError:
            pass

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
    suffix = f"Committed repository evidence:\n{evidence}"

    messages = [
        {"role": "system", "content": _PHASE1_SYSTEM},
        {"role": "user", "content": suffix},
    ]
    raw = _call_audit_llm(ctx, messages, trace_task="audit: phase 1 repo profile")
    records = _parse_records_with_repair(
        ctx,
        raw,
        schema=_PHASE1_PROFILE_SCHEMA,
        worked_example=_PHASE1_WORKED_EXAMPLE,
        metrics=state.metrics,
    )
    return records[0]


def _triage_record_from_parsed(path: str, parsed: dict) -> TriageRecord:
    """Build a TriageRecord from a parsed Phase-2 LLM record dict."""
    return TriageRecord(
        path=path,
        priority=parsed["priority"].upper(),
        confidence=parsed["confidence"].lower(),
        bug_classes=parsed.get("bug_classes", []),
        summary=parsed["summary"],
        relevant_symbols=parsed.get("relevant_symbols", []),
        suspicious_flows=parsed.get("suspicious_flows", []),
        needs_followup=parsed.get("needs_followup", False),
    )


def _phase2_triage_one(
    path: str,
    state: AuditRunState,
    ctx: InputContext,
) -> TriageRecord:
    """Triage a single file."""
    try:
        content = _git_show(path, ctx.base_dir)
    except RuntimeError:
        return TriageRecord(
            path=path,
            priority="SKIP",
            confidence="low",
            bug_classes=[],
            summary="file not readable",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=False,
        )

    imports_summary = ", ".join(state.import_index.get(path, [])[:20]) or "(none)"
    callers_summary = ", ".join(state.caller_index.get(path, [])[:10]) or "(none)"
    # Read from the cached score map populated in Phase 1 so the prompt and
    # the promotion gate at _apply_promotions agree. Fall back to a fresh
    # score if the cache is empty (legacy state files predate the cache).
    if path in state.attack_scores:
        score = state.attack_scores[path]
    else:
        score = _score_attack_surface(content)
        state.attack_scores[path] = score

    profile_json = _repo_profile_json(state)

    suffix = (
        f"Repository profile:\n{profile_json}\n\n"
        f"Attack-surface metadata:\nscore={score}\n\n"
        f"Direct imports/includes:\n{imports_summary}\n\n"
        f"Direct callers:\n{callers_summary}\n\n"
        f"Committed primary file contents:\n{content}\n\n"
        f"The file is: {path}"
    )
    messages = [
        {"role": "system", "content": _PHASE2_SYSTEM},
        {"role": "user", "content": suffix},
    ]
    try:
        raw = _call_audit_llm(ctx, messages, trace_task=f"audit: phase 2 triage {path}")
    except Exception as e:
        kind = type(e).__name__
        _debug_log(
            "triage_llm_failed",
            path=path,
            error=str(e),
            kind=kind,
        )
        return TriageRecord(
            path=path,
            priority="SKIP",
            confidence="low",
            bug_classes=[],
            summary=f"triage failed (llm call failed: {kind})",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=False,
            triage_failure_mode=f"llm_call_failed:{kind}",
        )

    try:
        records = _parse_records(raw, _PHASE2_TRIAGE_SCHEMA, metrics=state.metrics)
    except ValueError as e:
        _bump_parse_failure(state.metrics, _PHASE2_TRIAGE_SCHEMA.record.name)
        _debug_log(
            "records_parse_failed",
            record_type="triage",
            path=path,
            error=str(e),
            raw_len=len(raw),
            raw_preview=raw[:1000],
        )
        return TriageRecord(
            path=path,
            priority="SKIP",
            confidence="low",
            bug_classes=[],
            summary="triage failed (unparseable LLM response)",
            relevant_symbols=[],
            suspicious_flows=[],
            needs_followup=False,
            triage_failure_mode="parse_error",
        )

    return _triage_record_from_parsed(path, records[0])


# ---------------------------------------------------------------------------
# Phase 2 confirmation pass (low-confidence SKIPs)
# ---------------------------------------------------------------------------


def _phase2_confirm_one(
    path: str,
    state: AuditRunState,
    ctx: InputContext,
) -> TriageRecord | None:
    """Re-triage one path with richer evidence.

    Returns a new record when the second pass produces a verdict; returns
    None when the call itself fails so the caller can preserve the
    original SKIP record (the file is no worse off for the failed retry).
    """
    deps = state.dependency_index.get(path, [])[:30]
    deps_summary = ", ".join(deps) or "(none)"

    top_dep = ""
    top_dep_score = -1
    for d in deps:
        s = state.attack_scores.get(d, 0)
        if s > top_dep_score:
            top_dep_score = s
            top_dep = d

    paths_to_load = [path] + ([top_dep] if top_dep else [])
    cache = _load_file_contents(paths_to_load, ctx.base_dir)
    if path not in cache:
        return None
    content = cache[path]
    top_dep_excerpt = ""
    if top_dep:
        dep_content = cache.get(top_dep)
        if dep_content is None:
            top_dep = ""
        else:
            top_dep_excerpt = dep_content[:3000]

    entries = _entry_point_paths(state)
    reachable_from: list[str] = []
    for entry in entries:
        if path in state.dependency_index.get(entry, []):
            reachable_from.append(entry)
    reach_summary = (
        ", ".join(reachable_from) if reachable_from else "(no direct entry-point reach)"
    )

    score = state.attack_scores.get(path, 0)
    profile_json = _repo_profile_json(state)

    suffix = (
        f"Repository profile:\n{profile_json}\n\n"
        f"Attack-surface metadata:\nscore={score}\n\n"
        f"Files this calls into / depends on:\n{deps_summary}\n\n"
        f"Direct entry-point reachability:\n{reach_summary}\n\n"
    )
    if top_dep_excerpt:
        suffix += (
            f"Top dependency by score: {top_dep} (score={top_dep_score}). "
            f"Excerpt (first 3000 chars):\n{top_dep_excerpt}\n\n"
        )
    suffix += (
        f"Committed primary file contents:\n{content}\n\n"
        f"The file is: {path}\n\n"
        "First-pass triage marked this file SKIP with low confidence."
    )

    messages = [
        {"role": "system", "content": _PHASE2_CONFIRM_SYSTEM},
        {"role": "user", "content": suffix},
    ]
    try:
        raw = _call_audit_llm(
            ctx, messages, trace_task=f"audit: phase 2 confirm {path}"
        )
    except Exception as e:
        _debug_log("triage_confirm_failed", path=path, error=str(e))
        return None

    try:
        records = _parse_records(raw, _PHASE2_TRIAGE_SCHEMA, metrics=state.metrics)
    except ValueError as e:
        _debug_log("triage_confirm_parse_failed", path=path, error=str(e))
        return None

    return _triage_record_from_parsed(path, records[0])


# ---------------------------------------------------------------------------
# Phase 2 promotion (recover false negatives from triage)
# ---------------------------------------------------------------------------

_PROMOTION_SCORE_THRESHOLD = 8


def _trust_boundary_paths(state: AuditRunState) -> set[str]:
    """Return paths flagged as trust boundaries by Phase 1.

    The ``trust_boundary`` field is free-form text. Treat any value that
    matches a mandatory file path (or ends with one) as a path reference.
    """
    profile = state.repo_profile or {}
    boundaries = profile.get("trust_boundaries", []) or []
    mandatory = set(state.scope.mandatory_files)
    paths: set[str] = set()
    for entry in boundaries:
        if not isinstance(entry, str):
            continue
        if entry in mandatory:
            paths.add(entry)
            continue
        for f in mandatory:
            if entry.endswith(f) or f in entry.split():
                paths.add(f)
    return paths


def _entry_point_paths(state: AuditRunState) -> list[str]:
    profile = state.repo_profile or {}
    entries = profile.get("entry_points", []) or []
    mandatory = set(state.scope.mandatory_files)
    return [e for e in entries if isinstance(e, str) and e in mandatory]


def _compute_promotion_reasons(
    state: AuditRunState,
    force_review_matches: dict[str, str],
) -> dict[str, list[str]]:
    """Return ``{path: [reason, ...]}`` for files that should be promoted
    out of any SKIP verdict by deterministic signals."""
    reasons: dict[str, list[str]] = {}

    def _add(p: str, msg: str) -> None:
        reasons.setdefault(p, []).append(msg)

    # Score-based promotion.
    for p in state.queued_files:
        score = state.attack_scores.get(p, 0)
        if score >= _PROMOTION_SCORE_THRESHOLD:
            _add(p, f"attack-surface score {score}")

    # Entry points.
    entries = _entry_point_paths(state)
    for p in entries:
        _add(p, "phase 1 entry point")

    # Trust boundaries.
    for p in _trust_boundary_paths(state):
        _add(p, "phase 1 trust boundary")

    # One-hop reverse from each entry point: caller_index[entry] holds the
    # files that entry references. Anything an entry point reaches with a
    # non-zero attack score deserves a look.
    for entry in entries:
        for dep in state.caller_index.get(entry, []):
            if state.attack_scores.get(dep, 0) > 0:
                _add(dep, f"reached from entry point {entry}")

    # Triage said "needs followup" — promote outright.
    for p, rec in state.triage_records.items():
        if rec.needs_followup:
            _add(p, "triage needs_followup")

    # Triage infrastructure failure (LLM call failed, parse error, etc.).
    for p, rec in state.triage_records.items():
        if rec.triage_failure_mode is not None:
            _add(p, f"triage infrastructure failure: {rec.triage_failure_mode}")

    # Force-review (user override from swival.toml).
    for p, source in force_review_matches.items():
        _add(p, f"forced via swival.toml ({source})")

    return reasons


def _apply_promotions(
    state: AuditRunState,
    force_review_matches: dict[str, str],
) -> dict[str, list[str]]:
    """Compute promotions and apply them to ``state.triage_records``.

    Synthesizes a record for any queued path missing from triage_records
    (belt-and-suspenders against silent drops). Returns the per-path
    promotion reasons that fired.
    """
    # Belt-and-suspenders: every queued file must have a record. If one
    # is missing here it means a worker silently dropped it; synthesize
    # a record marked as missing so it can be promoted on the same rule
    # as other infrastructure failures.
    for p in state.queued_files:
        if p not in state.triage_records:
            state.triage_records[p] = TriageRecord(
                path=p,
                priority="SKIP",
                confidence="low",
                bug_classes=[],
                summary="triage record missing after retries",
                relevant_symbols=[],
                suspicious_flows=[],
                needs_followup=False,
                triage_failure_mode="missing",
            )
            state.reviewed_files.add(p)

    promotions = _compute_promotion_reasons(state, force_review_matches)

    for path, why in promotions.items():
        rec = state.triage_records.get(path)
        if rec is None:
            continue
        for r in why:
            if r not in rec.promotion_reasons:
                rec.promotion_reasons.append(r)
        if rec.priority == "SKIP":
            rec.priority = "ESCALATE_MEDIUM"

    return promotions


def _emit_measure_triage_recall(state: AuditRunState) -> None:
    """Print the calibration mode recall section.

    Counts verified findings by ``triage_decision`` and breaks them out by
    severity. Findings whose source file was a Phase-2 SKIP are the false
    negatives this mode exists to surface.
    """
    by_decision_severity: dict[tuple[str, str], int] = {}
    for vf in state.verified_findings:
        decision = vf.finding.triage_decision or "unknown"
        sev = vf.finding.severity or "unknown"
        by_decision_severity[(decision, sev)] = (
            by_decision_severity.get((decision, sev), 0) + 1
        )

    fmt.info("--- triage recall (--measure-triage) ---")
    fmt.info(
        f"  candidate set after triage: {len(state.measurement_escalated_paths)} files"
    )
    fmt.info(f"  expanded deep-review set:   {len(state.candidate_files)} files")
    n_escalated = sum(
        n for (d, _s), n in by_decision_severity.items() if d == "escalated"
    )
    n_skipped = sum(n for (d, _s), n in by_decision_severity.items() if d == "skipped")
    fmt.info(f"  verified findings on escalated files: {n_escalated}")
    fmt.info(f"  verified findings on SKIPped files:   {n_skipped} (false negatives)")
    if n_skipped:
        for sev in _SEVERITIES + ("unknown",):
            n = by_decision_severity.get(("skipped", sev), 0)
            if n:
                fmt.info(f"    skipped × {sev}: {n}")
        for vf in state.verified_findings:
            if vf.finding.triage_decision == "skipped":
                fmt.info(
                    f"    - {vf.finding.source_file}: "
                    f"[{vf.finding.severity}] {vf.finding.title}"
                )


# (reason_prefix, label, emitter). The emitter is fmt.warning for
# infrastructure-failure promotions because they signal a flaky model or
# upstream rather than a routine selection.
_PROMOTION_REASON_LABELS: tuple[tuple[str, str, str], ...] = (
    ("attack-surface score", "promoted by attack-surface score", "info"),
    ("phase 1 entry point", "promoted as phase 1 entry point", "info"),
    ("reached from entry point", "promoted by entry-point reachability", "info"),
    ("phase 1 trust boundary", "promoted as phase 1 trust boundary", "info"),
    ("triage needs_followup", "promoted by triage needs_followup", "info"),
    (
        "triage infrastructure failure",
        "promoted due to triage infrastructure failure",
        "warning",
    ),
    ("forced via swival.toml", "force-listed in swival.toml", "info"),
)


def _emit_phase2_summary(
    state: AuditRunState,
    promotions: dict[str, list[str]],
) -> None:
    """Write the end-of-phase-2 status lines, broken down by promotion reason."""
    records = state.triage_records
    n_high = sum(1 for r in records.values() if r.priority == "ESCALATE_HIGH")
    n_medium = sum(1 for r in records.values() if r.priority == "ESCALATE_MEDIUM")
    candidate_set = set(state.candidate_files)
    by_llm = sum(
        1 for p, r in records.items() if p in candidate_set and not r.promotion_reasons
    )
    promotion_only = [p for p in candidate_set if records[p].promotion_reasons]

    counts: dict[str, int] = {}
    for prefix, _label, _emit in _PROMOTION_REASON_LABELS:
        counts[prefix] = sum(
            1
            for p in promotion_only
            if any(r.startswith(prefix) for r in records[p].promotion_reasons)
        )
    by_confirm = sum(
        1 for p in candidate_set if records[p].confirmation_outcome == "promoted"
    )

    fmt.info(
        f"phase 2 complete. {len(state.candidate_files)} files escalated "
        f"({n_high} high, {n_medium} medium)"
    )
    if promotions or by_confirm:
        fmt.info(f"  - {by_llm} by LLM triage")
        for prefix, label, emit in _PROMOTION_REASON_LABELS:
            n = counts[prefix]
            if not n:
                continue
            (fmt.warning if emit == "warning" else fmt.info)(f"  - {n} {label}")
        if by_confirm:
            fmt.info(f"  - {by_confirm} recovered by confirmation pass")

    skipped = [p for p, r in records.items() if r.priority == "SKIP"]
    if skipped:
        ranked = sorted(skipped, key=lambda p: (-state.attack_scores.get(p, 0), p))[:5]
        top = ", ".join(f"{p}:{state.attack_scores.get(p, 0)}" for p in ranked)
        fmt.info(f"  {len(skipped)} files SKIPped (top by score: {top})")


def _phase3a_inventory(
    path: str,
    state: AuditRunState,
    ctx: InputContext,
    content: str,
) -> list[dict]:
    """Phase 3a: compact finding inventory for one file."""
    triage = state.triage_records.get(path)
    bug_classes = ", ".join(triage.bug_classes) if triage else "all"

    related_parts = []
    for imp_file in (state.import_index.get(path, []))[:5]:
        for tf in state.scope.tracked_files:
            if imp_file in tf:
                try:
                    related_parts.append(
                        f"--- {tf} ---\n{_git_show(tf, ctx.base_dir)[:3000]}"
                    )
                except RuntimeError:
                    pass
                break

    profile_json = _repo_profile_json(state)
    triage_json = json.dumps(asdict(triage), indent=2) if triage else "{}"
    related = "\n\n".join(related_parts) if related_parts else "(none)"

    suffix = (
        f"Focus bug classes: {bug_classes}\n\n"
        f"Repository profile:\n{profile_json}\n\n"
        f"Phase 2 triage result:\n{triage_json}\n\n"
        f"Committed evidence bundle:\n{content}\n\n"
        f"Related context:\n{related}\n\n"
        f"Primary file: {path}"
    )
    messages = [
        {"role": "system", "content": _PHASE3A_SYSTEM},
        {"role": "user", "content": suffix},
    ]
    raw = _call_audit_llm(ctx, messages, trace_task=f"audit: phase 3a inventory {path}")
    return _parse_records_with_repair(
        ctx,
        raw,
        schema=_PHASE3A_FINDING_SCHEMA,
        worked_example=_PHASE3A_WORKED_EXAMPLE,
        metrics=state.metrics,
    )


def _build_hunt_user_prompt(
    task: HuntTask,
    state: AuditRunState,
    file_evidence: dict[str, str],
) -> str:
    """Assemble the per-task user prompt for a hunter call."""
    profile_json = _repo_profile_json(state)
    related = _format_seed_evidence(task.seed_files, file_evidence, cap_files=5)

    sinks = ", ".join(task.sink_files[:8]) or "(none)"
    syms = ", ".join(task.seed_symbols[:8]) or "(none)"
    controlled = "\n".join(f"  - {c}" for c in task.controlled_inputs)
    language = task.language_hint or "(unknown)"

    language_note = ""
    if task.language_hint == "rust":
        language_note = (
            "Rust-specific hints: pay attention to `unsafe` blocks, FFI boundaries, "
            "lifetime extensions, `Pin`/`Send`/`Sync` violations, and unchecked "
            "transmutes. Index/slice operations are bounds-checked unless `unchecked` "
            "is used; report only when the unsafe path is actually reached.\n"
        )
    elif task.language_hint in ("c", "cpp"):
        language_note = (
            "C/C++-specific hints: focus on length math, allocator return checks, "
            "`memcpy`/`memmove` size arguments, return-value misuse, and signed/"
            "unsigned conversions. Treat undefined behavior as a finding only when "
            "an attacker can supply the triggering input.\n"
        )
    elif task.language_hint in ("python", "javascript", "typescript", "ruby", "php"):
        language_note = (
            f"{task.language_hint.title()}-specific hints: focus on framework "
            "decorators (route handlers, permission gates), serializer/loader APIs, "
            "and shell-out helpers. Report only when an external request boundary "
            "actually controls the input.\n"
        )

    return (
        f"Hunt task: {task.id}\n"
        f"Attack class: {task.attack_class}\n"
        f"Attacker model: {task.attacker_position}\n"
        f"Controlled inputs:\n{controlled}\n"
        f"Trust boundary crossed: {task.trust_boundary_crossed}\n"
        f"Scope hint: {task.scope_hint or '(unbounded — use seed files)'}\n"
        f"Seed files: {', '.join(task.seed_files) or '(none)'}\n"
        f"Seed symbols: {syms}\n"
        f"Candidate sink files: {sinks}\n"
        f"Dominant language: {language}\n\n"
        f"{language_note}"
        f"Repository profile:\n{profile_json}\n\n"
        f"Committed seed-file evidence:\n{related}\n\n"
        f"Now perform the hunt. Emit at most three `@@ finding @@` blocks (or `@@ none @@`) "
        f"followed by exactly one `@@ coverage @@` block."
    )


def _parse_hunt_response(
    raw: str, metrics: dict[str, int]
) -> tuple[list[dict], dict | None]:
    """Split a hunter response into (finding_dicts, coverage_dict or None).

    The hunter emits ``@@ finding @@`` blocks (≤3) followed by exactly one
    ``@@ coverage @@`` block, or ``@@ none @@`` plus the coverage block when
    there is nothing to report. The two record families share the ``@@ ... @@``
    delimiter so we parse them via the standard ``_parse_records`` helper after
    splitting on the coverage marker.
    """
    # Split on the coverage header. The coverage section is always last.
    parts = re.split(r"^\s*@@\s*coverage\s*@@\s*$", raw, maxsplit=1, flags=re.M)
    finding_part = parts[0]
    coverage_part = parts[1] if len(parts) == 2 else ""

    try:
        findings = _parse_records(
            finding_part, _PHASE_HUNT_FINDING_SCHEMA, metrics=metrics
        )
    except ValueError:
        findings = []

    coverage: dict | None = None
    if coverage_part.strip():
        # Re-prefix so _parse_records sees the standard ``@@ coverage @@`` line
        try:
            records = _parse_records(
                "@@ coverage @@\n" + coverage_part,
                _PHASE_HUNT_COVERAGE_SCHEMA,
                metrics=metrics,
            )
            coverage = records[0] if records else None
        except ValueError:
            coverage = None

    return findings, coverage


def _hunt_finding_to_record(
    raw: dict,
    task: HuntTask,
) -> FindingRecord:
    """Normalise one hunter ``@@ finding @@`` block into a FindingRecord."""
    severity = (raw.get("severity") or "low").strip().lower()
    if severity not in _SEVERITIES:
        severity = "low"

    location = raw.get("location") or (task.seed_files[0] if task.seed_files else "")
    locations = [location] if isinstance(location, str) else list(location or [])
    controlled = raw.get("controlled_inputs", []) or list(task.controlled_inputs)
    path = raw.get("reachability_path", []) or []
    chain = raw.get("exploit_chain", []) or []
    status = (raw.get("reachability_status") or "unknown").strip().lower()
    if status not in ("reachable", "local_only", "not_reachable", "unknown"):
        status = "unknown"

    proof_lines = [
        f"local_bug: {raw.get('local_bug', '')}",
        f"source_boundary: {raw.get('source_boundary', '')}",
        f"sink_operation: {raw.get('sink_operation', '')}",
        f"reachability_status: {status}",
        f"claim: {raw.get('claim', '')}",
    ]
    if path:
        proof_lines.append("reachability_path: " + " -> ".join(path))
    if chain:
        proof_lines.append("exploit_chain: " + " -> ".join(chain))

    if task.seed_files:
        source_file = task.seed_files[0]
    elif locations:
        source_file = locations[0]
    else:
        source_file = ""

    return FindingRecord(
        title=raw.get("title", "untitled"),
        finding_type=task.attack_class or "vulnerability",
        severity=severity,
        locations=locations,
        preconditions=[],
        proof=[p for p in proof_lines if not p.endswith(": ")],
        fix_outline="",
        source_file=source_file,
        local_bug=raw.get("local_bug", ""),
        source_boundary=raw.get("source_boundary", ""),
        reachability_path=list(path),
        sink_operation=raw.get("sink_operation", ""),
        exploit_chain=list(chain),
        reachability_status=status,
        hunt_task_id=task.id,
        attack_class=task.attack_class,
        attacker_position=task.attacker_position,
        controlled_inputs=list(controlled),
    )


def _coverage_record_from_parsed(
    task: HuntTask,
    state: AuditRunState,
    parsed: dict | None,
) -> CoverageRecord:
    """Build a CoverageRecord from the parsed coverage block plus task scope.

    The observed-coverage signal comes only from what the hunter actually
    reported; we never derive coverage from prompt content or seed files alone,
    per the plan ("Self-reported coverage is unreliable. ... Coverage must be
    derived from observable evidence where possible.")
    """
    parsed = parsed or {}
    observed_files = list(parsed.get("observed_files", []) or [])
    observed_symbols = list(parsed.get("observed_symbols", []) or [])
    observed_edges = list(parsed.get("observed_edges", []) or [])
    not_covered = list(parsed.get("not_covered", []) or [])
    confidence = (parsed.get("confidence") or "low").strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    # Map observed signals back to the task seed/sink set for user-facing ratios.
    seed_set = {f for f in task.seed_files if f}
    sink_set = {f for f in task.sink_files if f}
    observed_set = {f for f in observed_files if f}
    seed_hits = observed_set & seed_set
    sink_hits = observed_set & sink_set
    covered_seed = (
        len(seed_hits) / len(seed_set) if seed_set else (1.0 if observed_set else 0.0)
    )
    covered_sink = (
        len(sink_hits) / len(sink_set) if sink_set else (1.0 if observed_set else 0.0)
    )
    unobserved_seeds = sorted(seed_set - observed_set)

    return CoverageRecord(
        task_id=task.id,
        observed_files=observed_files,
        observed_symbols=observed_symbols,
        observed_edges=observed_edges,
        covered_fraction_of_seed_files=round(covered_seed, 3),
        covered_fraction_of_sink_hits=round(covered_sink, 3),
        unobserved_high_priority_seeds=unobserved_seeds,
        explicit_not_covered=not_covered,
        followup_task_ids=[],
        confidence=confidence,
    )


def _phase_hunt_one(
    task: HuntTask,
    state: AuditRunState,
    ctx: InputContext,
) -> tuple[list[FindingRecord], CoverageRecord, str | None]:
    """Run a single hunter task. Returns ``(findings, coverage, error)``.

    The function never raises: failures are reported in the third tuple slot so
    the parallel batch executor can keep going.
    """
    try:
        _validate_hunt_task(task)
    except HuntTaskValidationError as e:
        return ([], CoverageRecord(task_id=task.id), f"invalid task: {e}")

    budget = BudgetPlanner(state)
    if budget.exhausted():
        return (
            [],
            CoverageRecord(task_id=task.id),
            "budget exhausted before task started",
        )

    # Read the committed snapshot via the batched git-cat-file helper instead
    # of one shell-out per file.
    file_evidence = _load_file_contents(task.seed_files[:5], ctx.base_dir)

    user_prompt = _build_hunt_user_prompt(task, state, file_evidence)
    messages = [
        {"role": "system", "content": _PHASE_HUNT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = _call_audit_llm(ctx, messages, trace_task=f"audit: hunt {task.id}")
    except Exception as e:
        return ([], CoverageRecord(task_id=task.id), f"llm error: {e}")

    budget.charge_call("hunt", user_prompt, raw)

    findings_raw, coverage_raw = _parse_hunt_response(raw, state.metrics)
    findings = [_hunt_finding_to_record(f, task) for f in findings_raw[:3]]
    coverage = _coverage_record_from_parsed(task, state, coverage_raw)

    bucket = state.attack_class_metrics.setdefault(
        task.attack_class, {"proposed": 0, "verified": 0}
    )
    bucket["proposed"] = int(bucket.get("proposed", 0)) + len(findings)

    return (findings, coverage, None)


_PHASE_REACHABILITY_RESULT_SCHEMA = PhaseSchema(
    record=RecordSchema(
        name="reachability",
        required=("verdict",),
        enums={"verdict": _REACHABILITY_TRACE_VERDICTS},
        repeated={"reachability_step": "reachability_path"},
    ),
    cardinality="one",
)

_PHASE_REACHABILITY_SYSTEM = """\
You are a reachability tracer running one narrow task in a staged audit.

You are given a confirmed local bug at a specific sink and a named trust
boundary. Your only job is to decide whether attacker-controlled data from that
boundary can actually reach the sink in committed code.

You have no tools, no shell access, and no ability to run commands. All the
source code you need is included below.

Output format: exactly one `@@ reachability @@` block with these keys, one per
line. Use exactly the keys shown. Do not quote or wrap values.

- verdict: reachable | not_reachable | unknown
- reachability_step: one step in the data path from source to sink, in order;
  one line per file or function; omit when not reachable
- blocker: the named guard, control, or absent edge that breaks the path; only
  set when verdict is not_reachable
- confidence: high | medium | low

Rules:
- Use only the supplied evidence. Do not invent files or symbols.
- Report `unknown` when the path is plausible but the evidence is incomplete;
  do not guess.
- Report `not_reachable` only when a concrete guard or missing edge blocks the
  path, and name it in `blocker`.
- Do not propose new findings. Do not change the bug claim. Only decide whether
  the existing bug is externally reachable.
"""


def _format_seed_evidence(
    seed_files: list[str],
    file_evidence: dict[str, str],
    cap_files: int,
) -> str:
    seed_lines = []
    for f in seed_files[:cap_files]:
        body = file_evidence.get(f, "")
        if not body:
            continue
        if len(body) > _HUNT_SEED_FILE_CAP:
            body = body[:_HUNT_SEED_FILE_CAP] + _AUDIT_TRUNCATION_MARKER
        seed_lines.append(f"--- {f} ---\n{body}")
    return "\n\n".join(seed_lines) if seed_lines else "(seed file evidence missing)"


def _is_scf_high_or_critical(finding: FindingRecord) -> bool:
    """True when the SCF carve-out applies: high/critical security_control_failure.

    Mirrors the existing `_scf_below_min_severity` predicate but in the positive
    direction so callers reading "keep if SCF-high" don't have to negate.
    """
    return (
        finding.finding_type == _PHASE3B_SECURITY_CONTROL_FAILURE_TYPE
        and finding.severity in _SECURITY_CONTROL_FAILURE_MIN_SEVERITIES
    )


def _build_reachability_user_prompt(
    task: HuntTask,
    parent_finding: FindingRecord,
    state: AuditRunState,
    file_evidence: dict[str, str],
) -> str:
    profile_json = _repo_profile_json(state)
    related = _format_seed_evidence(task.seed_files, file_evidence, cap_files=6)
    controlled = "\n".join(f"  - {c}" for c in task.controlled_inputs) or "  - (none)"
    locations = ", ".join(parent_finding.locations) or parent_finding.source_file
    prior_path = (
        " -> ".join(parent_finding.reachability_path)
        if parent_finding.reachability_path
        else "(none reported)"
    )

    return (
        f"Reachability task: {task.id}\n"
        f"Parent finding: {parent_finding.title}\n"
        f"Attack class: {task.attack_class}\n"
        f"Attacker model: {task.attacker_position}\n"
        f"Controlled inputs:\n{controlled}\n"
        f"Trust boundary: {task.trust_boundary_crossed}\n"
        f"Local bug: {parent_finding.local_bug or '(unspecified)'}\n"
        f"Sink operation: {parent_finding.sink_operation or '(unspecified)'}\n"
        f"Sink location(s): {locations}\n"
        f"Hunter-reported partial path: {prior_path}\n\n"
        f"Repository profile:\n{profile_json}\n\n"
        f"Committed evidence:\n{related}\n\n"
        f"Decide whether attacker input from the named trust boundary actually "
        f"reaches the sink. Emit exactly one `@@ reachability @@` block."
    )


def _parse_reachability_response(raw: str, metrics: dict[str, int]) -> dict | None:
    """Return the parsed reachability dict, or None when malformed."""
    try:
        records = _parse_records(
            raw, _PHASE_REACHABILITY_RESULT_SCHEMA, metrics=metrics
        )
    except ValueError:
        return None
    return records[0] if records else None


def _generate_reachability_tasks(state: AuditRunState) -> list[HuntTask]:
    """Promote each LOCAL_ONLY hunt finding to one reachability follow-up.

    The post-reachability filter discards LOCAL_ONLY findings that do not get
    promoted, so anything we drop here permanently leaves the run unless it
    matches the high/critical security_control_failure carve-out.
    """
    new_tasks: list[HuntTask] = []
    existing_parents = {
        t.parent_finding_key
        for t in state.hunt_tasks.values()
        if t.task_kind == _HUNT_TASK_KIND_REACHABILITY
    }
    for finding in state.proposed_findings:
        if finding.reachability_status != _REACH_LOCAL_ONLY:
            continue
        if _is_scf_high_or_critical(finding):
            continue
        fkey = _finding_key(finding)
        if fkey in existing_parents:
            continue
        parent = state.hunt_tasks.get(finding.hunt_task_id)
        if parent is None:
            continue
        seed_files = list(
            dict.fromkeys(
                [*parent.seed_files, *(loc.split(":")[0] for loc in finding.locations)]
            )
        )
        sink_files = list(
            dict.fromkeys(
                [
                    finding.source_file or (seed_files[0] if seed_files else ""),
                    *parent.sink_files,
                ]
            )
        )
        task_id = _hunt_task_id(
            parent.attack_class,
            parent.attacker_position,
            parent.scope_hint,
            seed_files,
            suffix=f"reachability:{fkey}",
        )
        new_tasks.append(
            HuntTask(
                id=task_id,
                task_kind=_HUNT_TASK_KIND_REACHABILITY,
                attack_class=parent.attack_class,
                attacker_position=parent.attacker_position,
                controlled_inputs=list(parent.controlled_inputs),
                trust_boundary_crossed=(
                    finding.source_boundary or parent.trust_boundary_crossed
                ),
                scope_hint=parent.scope_hint,
                seed_files=seed_files[:8],
                seed_symbols=list(parent.seed_symbols),
                sink_files=[f for f in sink_files if f][:8],
                source=_REACH_TASK_SOURCE_LOCAL_ONLY,
                priority=parent.priority,
                parent_task_id=parent.id,
                parent_finding_key=fkey,
                language_hint=parent.language_hint,
            )
        )
    return new_tasks


def _phase_reachability_one(
    task: HuntTask,
    parent_finding: FindingRecord,
    state: AuditRunState,
    ctx: InputContext,
    file_evidence: dict[str, str] | None = None,
) -> tuple[dict | None, str | None]:
    try:
        _validate_hunt_task(task)
    except HuntTaskValidationError as e:
        return (None, f"invalid task: {e}")

    budget = BudgetPlanner(state)
    if budget.exhausted():
        return (None, "budget exhausted before task started")

    if file_evidence is None:
        file_evidence = _load_file_contents(task.seed_files[:6], ctx.base_dir)
    user_prompt = _build_reachability_user_prompt(
        task, parent_finding, state, file_evidence
    )
    messages = [
        {"role": "system", "content": _PHASE_REACHABILITY_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = _call_audit_llm(
            ctx, messages, trace_task=f"audit: reachability {task.id}"
        )
    except Exception as e:
        return (None, f"llm error: {e}")

    budget.charge_call(_HUNT_TASK_KIND_REACHABILITY, user_prompt, raw)

    parsed = _parse_reachability_response(raw, state.metrics)
    if parsed is None:
        return (None, "malformed reachability response")
    return (parsed, None)


def _apply_reachability_result(
    parent_finding: FindingRecord,
    parsed: dict,
) -> None:
    verdict = (parsed.get("verdict") or _REACH_UNKNOWN).strip().lower()
    if verdict not in _REACHABILITY_TRACE_VERDICTS:
        verdict = _REACH_UNKNOWN
    parent_finding.reachability_status = verdict
    path = list(parsed.get("reachability_path", []) or [])
    if path:
        parent_finding.reachability_path = path


def _filter_to_reachable_findings(state: AuditRunState) -> int:
    """Drop hunt-derived findings that did not reach `reachable`.

    File-centric findings (no `hunt_task_id`) and the high/critical
    security_control_failure carve-out are preserved. Returns the discard
    count so the caller can log it.
    """
    if not state.hunt_mode:
        return 0
    kept: list[FindingRecord] = []
    discarded = 0
    for f in state.proposed_findings:
        if not f.hunt_task_id:
            kept.append(f)
        elif _is_scf_high_or_critical(f):
            kept.append(f)
        elif f.reachability_status == _REACH_REACHABLE:
            kept.append(f)
        else:
            discarded += 1
    state.proposed_findings = kept
    return discarded


def _phase3b_expand_one(
    item: tuple[dict, str, str, AuditRunState, InputContext],
) -> dict | None:
    """Phase 3b: expand one inventory finding with proof details."""
    finding_stub, path, content, state, ctx = item

    suffix = (
        f"Committed evidence for {path}:\n{content}\n\n"
        f"Finding to expand:\n"
        f"  Title: {finding_stub.get('title', '')}\n"
        f"  Severity: {finding_stub.get('severity', '')}\n"
        f"  Location: {finding_stub.get('location', '')}\n"
        f"  Attacker: {finding_stub.get('attacker', '')}\n"
        f"  Trigger: {finding_stub.get('trigger', '')}\n"
        f"  Impact: {finding_stub.get('impact', '')}\n"
        f"  Claim: {finding_stub.get('claim', '')}"
    )
    messages = [
        {"role": "system", "content": _PHASE3B_SYSTEM},
        {"role": "user", "content": suffix},
    ]
    raw = _call_audit_llm(ctx, messages, trace_task=f"audit: phase 3b expand {path}")
    try:
        records = _parse_records_with_repair(
            ctx,
            raw,
            schema=_PHASE3B_EXPANSION_SCHEMA,
            worked_example=_PHASE3B_WORKED_EXAMPLE,
            metrics=state.metrics,
        )
    except ValueError:
        return None
    return records[0] if records else None


def _canonicalize_finding(
    inventory_item: dict,
    expansion: dict,
    source_file: str,
) -> FindingRecord:
    """Build a FindingRecord from compact inventory + expansion dicts."""
    severity = (inventory_item.get("severity") or "low").lower()
    if severity not in _SEVERITIES:
        severity = "low"

    location = inventory_item.get("location", source_file)
    preconditions_raw = expansion.get("preconditions", "")
    proof_parts = [
        f"attacker: {expansion.get('attacker', '')}",
        f"trigger: {expansion.get('trigger', '')}",
        f"impact: {expansion.get('impact', '')}",
        expansion.get("proof", ""),
    ]
    proof_raw = " ".join(p for p in proof_parts if p and not p.endswith(": "))

    return FindingRecord(
        title=inventory_item.get("title", "untitled"),
        finding_type=expansion.get("type", "unknown"),
        severity=severity,
        locations=[location] if isinstance(location, str) else location,
        preconditions=[preconditions_raw] if preconditions_raw else [],
        proof=[proof_raw] if proof_raw else [],
        fix_outline=expansion.get("fix_outline", ""),
        source_file=source_file,
    )


def _is_out_of_scope_expansion(expansion: dict) -> bool:
    return expansion.get("type", "").strip().lower() == _PHASE3B_OUT_OF_SCOPE_TYPE


def _scf_below_min_severity(finding_type: str, severity: str) -> bool:
    if finding_type.strip().lower() != _PHASE3B_SECURITY_CONTROL_FAILURE_TYPE:
        return False
    return severity.strip().lower() not in _SECURITY_CONTROL_FAILURE_MIN_SEVERITIES


def _phase3_deep_review(
    path: str,
    state: AuditRunState,
    ctx: InputContext,
) -> list[FindingRecord]:
    """Deep review a single escalated file using inventory + expansion."""
    try:
        content = _git_show(path, ctx.base_dir)
    except RuntimeError:
        return []

    inventory = _phase3a_inventory(path, state, ctx, content)
    if not inventory:
        return []

    expansion_items = [(stub, path, content, state, ctx) for stub in inventory]
    expansions = []
    for item in expansion_items:
        try:
            expansions.append(_phase3b_expand_one(item))
        except Exception as e:
            fmt.warning(f"expansion failed for {path}: {e}")
            expansions.append(None)

    findings = []
    failed_expansions = 0
    for stub, expansion in zip(inventory, expansions):
        if expansion is None:
            failed_expansions += 1
            continue
        if _is_out_of_scope_expansion(expansion):
            continue
        if _scf_below_min_severity(expansion.get("type", ""), stub.get("severity", "")):
            continue
        findings.append(_canonicalize_finding(stub, expansion, path))

    if failed_expansions > 0:
        if not findings:
            raise ValueError(
                f"all {failed_expansions} expansion(s) failed for {path} "
                f"({len(inventory)} inventory finding(s))"
            )
        fmt.warning(
            f"  {path}: {failed_expansions}/{len(inventory)} expansion(s) failed, "
            f"{len(findings)} finding(s) retained"
        )
    return findings


def _deep_review_one(
    path: str,
    state: AuditRunState,
    ctx: InputContext,
) -> DeepReviewResult:
    """Run deep review with repair-first retry policy.

    Parse failures trigger a cheap LLM repair pass (inside _parse_records_with_repair).
    If the entire pipeline still fails, one full analytical retry runs before
    giving up.
    """
    try:
        findings = _phase3_deep_review(path, state, ctx)
        return DeepReviewResult(path=path, findings=findings)
    except (ValueError, RuntimeError) as e:
        if isinstance(e, ValueError):
            state.metrics["analytical_retries"] += 1
        fmt.info(f"  retrying deep review for {path} after error: {e}")
        try:
            findings = _phase3_deep_review(path, state, ctx)
            return DeepReviewResult(path=path, findings=findings)
        except (ValueError, RuntimeError) as e2:
            return DeepReviewResult(path=path, error=str(e2))


def _gather_evidence(finding: FindingRecord, ctx: InputContext) -> tuple[str, int]:
    """Collect committed file contents for all locations referenced by a finding."""
    seen = set()
    parts = []
    for loc in finding.locations:
        fpath = loc.split(":")[0]
        if fpath in seen:
            continue
        seen.add(fpath)
        try:
            content = _git_show(fpath, ctx.base_dir)
            parts.append(f"--- {fpath} ---\n{content}")
        except RuntimeError:
            pass
    if finding.source_file not in seen:
        try:
            content = _git_show(finding.source_file, ctx.base_dir)
            parts.append(f"--- {finding.source_file} ---\n{content}")
        except RuntimeError:
            pass
    text = "\n\n".join(parts) if parts else "(no evidence available)"
    return text, len(parts)


_REPRODUCE_KEYWORD = "REPRODUCED"
_NO_REPRODUCE_KEYWORD = "NOTREPRODUCED"

_PROOF_SENTINEL = "swival-audit-proof-v1"
_PROOF_BLOCK_RE = re.compile(
    r"```" + _PROOF_SENTINEL + r"\s*\n(.*?)\n```",
    re.DOTALL,
)
_PROOF_LIST_KEYS = ("commands", "artifacts")
_PROOF_SCALAR_KEYS = (
    "verdict",
    "proof_kind",
    "observed_output",
    "trigger",
    "impact",
    "limitations",
)
_PROOF_VERDICTS = (_REPRODUCE_KEYWORD, _NO_REPRODUCE_KEYWORD)
_PROOF_KINDS = ("runtime", "source", "mixed")


class ProofBlockError(ValueError):
    """Raised when Phase 4's structured proof block is missing or malformed."""


def _parse_kv_block(
    body: str,
    scalar_keys: tuple[str, ...],
    list_keys: tuple[str, ...],
) -> dict:
    """Line-oriented walker shared by all ``swival-audit-*-v1`` blocks.

    ``key: value`` lines populate scalars; ``key:`` followed by ``  - item``
    lines populates the named list. ``none`` (case-insensitive) clears a list
    item. Callers apply their own verdict-shape validation after.
    """
    out: dict = {k: "" for k in scalar_keys}
    for k in list_keys:
        out[k] = []
    current_list: str | None = None
    for raw in body.splitlines():
        if not raw.strip():
            current_list = None
            continue
        if current_list is not None and raw.lstrip().startswith("- "):
            item = raw.lstrip()[2:].strip()
            if item and item.lower() != "none":
                out[current_list].append(item)
            continue
        if ":" not in raw:
            current_list = None
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        val = val.strip()
        if key in list_keys:
            current_list = key
            if val and val.lower() != "none":
                out[key].append(val)
            continue
        current_list = None
        if key in scalar_keys:
            out[key] = val
    return out


def _parse_sentinel_block(
    answer: str,
    block_re: re.Pattern,
    body_parser,
    exc_cls: type,
    missing_msg: str,
):
    """Find the structured block in a response. Prefer the last valid match so
    that captured command output containing the sentinel cannot poison the
    verdict."""
    matches = list(block_re.finditer(answer))
    if not matches:
        raise exc_cls(missing_msg)
    first_err = None
    for m in reversed(matches):
        try:
            return body_parser(m.group(1))
        except exc_cls as e:
            if first_err is None:
                first_err = e
    raise first_err or exc_cls("no valid block")


def _parse_proof_body(body: str) -> dict:
    out = _parse_kv_block(body, _PROOF_SCALAR_KEYS, _PROOF_LIST_KEYS)
    if out["verdict"] not in _PROOF_VERDICTS:
        raise ProofBlockError(f"invalid or missing verdict: {out['verdict']!r}")
    if out["proof_kind"].lower() in ("", "none"):
        out["proof_kind"] = ""
    elif out["proof_kind"] not in _PROOF_KINDS:
        raise ProofBlockError(f"invalid proof_kind: {out['proof_kind']!r}")
    if out["verdict"] == _REPRODUCE_KEYWORD and not out["proof_kind"]:
        raise ProofBlockError("proof_kind is required for REPRODUCED verdicts")
    return out


def _parse_proof_block(answer: str) -> dict:
    return _parse_sentinel_block(
        answer,
        _PROOF_BLOCK_RE,
        _parse_proof_body,
        ProofBlockError,
        "no swival-audit-proof-v1 block in verifier output",
    )


def _finding_key(finding: FindingRecord) -> str:
    """Stable content-based key for a finding, used for verification state and worktrees."""
    blob = json.dumps(asdict(finding), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


_DISPROOF_SENTINEL = "swival-audit-disproof-v1"
_DISPROOF_BLOCK_RE = re.compile(
    r"```" + _DISPROOF_SENTINEL + r"\s*\n(.*?)\n```",
    re.DOTALL,
)
_DISPROOF_VERDICT_PLAUSIBLE = "PLAUSIBLE"
_DISPROOF_VERDICT_NEEDS_PROOF = "NEEDS_PROOF"
_DISPROOF_VERDICT_INVALID = "INVALID"
_DISPROOF_VERDICTS = (
    _DISPROOF_VERDICT_PLAUSIBLE,
    _DISPROOF_VERDICT_NEEDS_PROOF,
    _DISPROOF_VERDICT_INVALID,
)
_DISPROOF_SCALAR_KEYS = (
    "verdict",
    "reason",
    "blocking_control",
    "missing_link",
    "required_next_proof",
    "review_model",
)
_DISPROOF_LIST_KEYS = ("counterclaim_checked",)

_DISPROOF_STATUS_INVALID = "invalid"
_DISPROOF_STATUS_PLAUSIBLE = "plausible"
_DISPROOF_STATUS_NEEDS_PROOF = "needs_proof"
_DISPROOF_STATUS_FAILED_OPEN = "failed_open"
_DISPROOF_STATUS_GAPFILL = "gapfill"
_DISPROOF_TERMINAL_STATUSES = (
    _DISPROOF_STATUS_INVALID,
    _DISPROOF_STATUS_PLAUSIBLE,
    _DISPROOF_STATUS_NEEDS_PROOF,
    _DISPROOF_STATUS_FAILED_OPEN,
)


class DisproofBlockError(ValueError):
    """Raised when the adversarial-disproof block is missing or malformed."""


def _parse_disproof_body(body: str) -> dict:
    out = _parse_kv_block(body, _DISPROOF_SCALAR_KEYS, _DISPROOF_LIST_KEYS)
    if out["verdict"] not in _DISPROOF_VERDICTS:
        raise DisproofBlockError(f"invalid or missing verdict: {out['verdict']!r}")
    for k in ("blocking_control", "missing_link", "required_next_proof"):
        if out[k].lower() == "none":
            out[k] = ""
    return out


def _parse_disproof_block(answer: str) -> dict:
    return _parse_sentinel_block(
        answer,
        _DISPROOF_BLOCK_RE,
        _parse_disproof_body,
        DisproofBlockError,
        "no swival-audit-disproof-v1 block in reviewer output",
    )


_PHASE_DISPROOF_SYSTEM = """\
You are an adversarial reviewer for one proposed security finding from a staged audit.

Your job is to falsify the finding. Do not investigate new bugs. Do not propose new findings. Only attempt to disprove the one finding you are given, using the cited evidence.

You have no tools, no shell access, and no ability to run commands. All the source code you need is included below.

Instantiate explicit counterclaims to test, for example:
- name a concrete guard in committed code that already blocks the trigger
- show the supposedly attacker-controlled input is trusted-only (only admin/operator can supply it)
- show the named trust boundary is not externally reachable
- show the impact is correctness-only and has no security-relevant outcome
- show the sink cannot receive attacker-controlled data on any path

Verdicts:
- INVALID: you have concrete counterevidence that falsifies the finding (cite the guard, missing path, trusted-only trigger, non-security impact, or contradicting code).
- NEEDS_PROOF: the finding is not obviously false but the evidence as written is insufficient. Specify what proof the verifier should attempt next.
- PLAUSIBLE: you could not falsify the finding from the cited evidence.

Output exactly one fenced block using the ``swival-audit-disproof-v1`` sentinel. Emit nothing after the closing fence.

```swival-audit-disproof-v1
verdict: PLAUSIBLE | NEEDS_PROOF | INVALID
reason: one short sentence describing the disproof outcome
counterclaim_checked:
  - explicit falsification target tested, repeatable, one per line
blocking_control: concrete guard you found in code, or none
missing_link: missing attacker/reachability/proof step that blocks falsification, or none
required_next_proof: specific proof the verifier should attempt for NEEDS_PROOF, or none
review_model: leave blank; the harness fills this in
```

Rules for the block:
- ``INVALID`` requires a concrete blocking_control or missing_link citation. Do not return INVALID on speculation.
- ``NEEDS_PROOF`` requires a non-empty required_next_proof string.
- For ``PLAUSIBLE``, all counterevidence fields may be ``none``.
- Use ``none`` (lowercase) for empty lists and empty scalar fields.
- Do not embed nested ``swival-audit-disproof-v1`` markers in the block content."""


class _TransientVerifierError(Exception):
    """Raised when a verifier worker hits a transient provider or transport error."""


def _phase_disproof_one(
    finding: FindingRecord,
    state: AuditRunState,
    ctx: InputContext,
    *,
    reviewer_model: str = "",
) -> tuple[dict | None, str | None]:
    """Run the adversarial reviewer on one proposed finding.

    On success the parsed disproof block is returned with ``review_model`` set
    to the model actually used (the override when configured, otherwise the
    main model). On failure ``parsed`` is None and the caller routes via the
    strict/balanced policy.
    """
    locs = ", ".join(finding.locations) if finding.locations else finding.source_file
    evidence, _n = _gather_evidence(finding, ctx)
    finding_json = json.dumps(asdict(finding), indent=2)
    messages = [
        {"role": "system", "content": _PHASE_DISPROOF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Proposed finding:\n{finding_json}\n\n"
                f"Committed evidence bundle:\n{evidence}"
            ),
        },
    ]
    used_model = reviewer_model or ctx.loop_kwargs.get("model_id", "") or ""
    try:
        answer = _call_audit_llm(
            ctx,
            messages,
            trace_task=f"audit: disproof {finding.title}",
            model_override=reviewer_model or None,
        )
    except (ConnectionError, TimeoutError, OSError) as e:
        return None, f"transport error: {e}"
    except Exception as e:
        return None, f"disproof call failed: {e}"
    budget = BudgetPlanner(state)
    budget.charge_call("disproof", messages[-1]["content"], answer or "")
    try:
        parsed = _parse_disproof_block(answer or "")
    except DisproofBlockError as e:
        fmt.info(f"    reviewer [{locs}]: malformed disproof block: {e}")
        return None, f"malformed disproof block: {e}"
    parsed["review_model"] = used_model
    return parsed, None


def _set_agreement(state: AuditRunState, finding_key: str, outcome: str) -> None:
    """Stamp ``agreement_with_proof`` on the adversarial record for a finding.

    Outcomes are ``agreed``, ``disagreed``, ``agreed_pre_proof``, ``no_review``.
    """
    record = state.adversarial_state.get(finding_key)
    if record is not None:
        record["agreement_with_proof"] = outcome


_DISPROOF_REPRODUCED_AGREEMENT = {
    _DISPROOF_STATUS_PLAUSIBLE: "agreed",
    _DISPROOF_STATUS_NEEDS_PROOF: "agreed",
    _DISPROOF_STATUS_FAILED_OPEN: "no_review",
}
_DISPROOF_DISCARDED_AGREEMENT = {
    _DISPROOF_STATUS_PLAUSIBLE: "disagreed",
    _DISPROOF_STATUS_NEEDS_PROOF: "agreed",
    _DISPROOF_STATUS_FAILED_OPEN: "no_review",
}


def _disproof_routing(
    parsed: dict | None,
    error: str | None,
    *,
    proof_strict: bool,
) -> dict:
    """Convert a per-finding disproof result into a persistent state record.

    Rules:
    - INVALID with a concrete blocking_control or missing_link discards before
      verification.
    - NEEDS_PROOF carries ``required_next_proof`` into the verifier prompt.
    - PLAUSIBLE proceeds normally.
    - On parser or transport failure: balanced mode fails open to verification;
      strict mode marks ``gapfill`` so the operator can route it manually.
    """
    if parsed is None:
        status = (
            _DISPROOF_STATUS_GAPFILL if proof_strict else _DISPROOF_STATUS_FAILED_OPEN
        )
        return {
            "status": status,
            "verdict": "",
            "reason": error or "",
            "counterclaim_checked": [],
            "blocking_control": "",
            "missing_link": "",
            "required_next_proof": "",
            "review_model": "",
        }
    verdict = parsed["verdict"]
    if verdict == _DISPROOF_VERDICT_INVALID and not (
        parsed.get("blocking_control") or parsed.get("missing_link")
    ):
        verdict = _DISPROOF_VERDICT_PLAUSIBLE
    status = {
        _DISPROOF_VERDICT_INVALID: _DISPROOF_STATUS_INVALID,
        _DISPROOF_VERDICT_NEEDS_PROOF: _DISPROOF_STATUS_NEEDS_PROOF,
        _DISPROOF_VERDICT_PLAUSIBLE: _DISPROOF_STATUS_PLAUSIBLE,
    }[verdict]
    return {
        "status": status,
        "verdict": verdict,
        "reason": parsed.get("reason", ""),
        "counterclaim_checked": list(parsed.get("counterclaim_checked", [])),
        "blocking_control": parsed.get("blocking_control", ""),
        "missing_link": parsed.get("missing_link", ""),
        "required_next_proof": parsed.get("required_next_proof", ""),
        "review_model": parsed.get("review_model", ""),
    }


def _phase4c_reproduce(
    finding: FindingRecord,
    state: AuditRunState,
    ctx: InputContext,
    work_dir: Path,
) -> dict | None:
    """Run a verifier agent. Returns proof dict or None (NOTREPRODUCED).

    Infrastructure failures (worktree setup, agent loop crashes) propagate
    as exceptions so callers can distinguish them from legitimate negative
    verdicts.
    """
    from .agent import run_agent_loop

    finding_json = json.dumps(asdict(finding), indent=2)
    locs = ", ".join(finding.locations) if finding.locations else finding.source_file
    fmt.info(f"    verifier [{locs}]: collecting evidence for {finding.title}")
    evidence, n_files = _gather_evidence(finding, ctx)
    fmt.info(f"    verifier [{locs}]: gathered {n_files} evidence file(s)")

    fmt.info(f"    verifier [{locs}]: preparing isolated worktree")
    wt = _worktree(ctx.base_dir, work_dir)
    wt.__enter__()

    try:
        fmt.info(
            f"    verifier [{locs}]: running verification agent "
            f"(severity={finding.severity})"
        )
        next_proof = state.adversarial_state.get(_finding_key(finding), {}).get(
            "required_next_proof", ""
        )
        guidance = (
            f"\n\nAdversarial reviewer requested next proof:\n{next_proof}"
            if next_proof
            else ""
        )
        messages = [
            {"role": "system", "content": _PHASE4_VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Proposed finding:\n{finding_json}\n\n"
                    f"Committed evidence bundle:\n{evidence}{guidance}"
                ),
            },
        ]

        kw = _make_isolated_loop_kwargs(ctx, work_dir)
        try:
            answer, _exhausted = run_agent_loop(messages, ctx.tools, **kw)
        except (ConnectionError, TimeoutError, OSError) as e:
            raise _TransientVerifierError(str(e)) from e
        finally:
            _write_audit_trace(
                ctx, messages, task=f"audit: phase 4 verify {finding.title}"
            )

        answer = answer or ""
        try:
            proof = _parse_proof_block(answer)
        except ProofBlockError as e:
            fmt.info(
                f"    verifier [{locs}]: malformed proof block for {finding.title}: {e}"
            )
            return None

        if proof["verdict"] != _REPRODUCE_KEYWORD:
            fmt.info(f"    verifier [{locs}]: NOTREPRODUCED: {finding.title}")
            return None

        reproducer = {
            "reproduced": True,
            "proof_kind": proof["proof_kind"],
            "commands": list(proof["commands"]),
            "artifacts": list(proof["artifacts"]),
            "observed_output": proof["observed_output"],
            "trigger": proof["trigger"],
            "impact": proof["impact"],
            "limitations": proof["limitations"],
            "summary": answer[-1000:],
        }
        _persist_proof_artifact(state, finding, answer, reproducer)
        fmt.info(
            f"    verifier [{locs}]: REPRODUCED ({proof['proof_kind']}): "
            f"{finding.title}"
        )
        return reproducer
    finally:
        wt.__exit__(None, None, None)


_PROOF_RAW_ANSWER_CAP = 256 * 1024


def _verify_dir(state: AuditRunState, finding_key: str) -> Path:
    """Per-finding artifact root under .swival/audit/<run-id>/verify/.

    The verifier worktree lives at ``<verify_dir>/work`` and is removed on
    teardown; proof artifacts go directly under ``<verify_dir>`` so they
    survive the cleanup.
    """
    return state.state_dir / state.run_id / "verify" / finding_key


def _persist_proof_artifact(
    state: AuditRunState,
    finding: FindingRecord,
    raw_answer: str,
    reproducer: dict,
) -> None:
    finding_key = _finding_key(finding)
    proof_dir = _verify_dir(state, finding_key)
    if len(raw_answer) > _PROOF_RAW_ANSWER_CAP:
        raw_answer = raw_answer[:_PROOF_RAW_ANSWER_CAP] + _AUDIT_TRUNCATION_MARKER
    try:
        proof_dir.mkdir(parents=True, exist_ok=True)
        (proof_dir / "proof.txt").write_text(raw_answer)
        (proof_dir / "proof.json").write_text(json.dumps(reproducer, indent=2))
        if reproducer.get("commands"):
            (proof_dir / "commands.log").write_text(
                "\n".join(reproducer["commands"]) + "\n"
            )
    except OSError as e:
        _debug_log(
            "proof_persist_failed",
            finding_key=finding_key,
            error=str(e),
        )


def _evidence_file_paths(finding: FindingRecord) -> list[str]:
    """Return deduplicated repo-relative paths referenced by a finding."""
    seen: set[str] = set()
    paths: list[str] = []
    for loc in finding.locations:
        fpath = loc.split(":")[0]
        if fpath not in seen:
            seen.add(fpath)
            paths.append(fpath)
    if finding.source_file not in seen:
        paths.append(finding.source_file)
    return paths


def _phase5_patch(
    vf: VerifiedFinding,
    ctx: InputContext,
    state: AuditRunState,
    patch_max_turns: int = _DEFAULT_PATCH_MAX_TURNS,
) -> PatchGenerationResult:
    """Generate a patch by running an agent loop in a worktree, then capturing git diff."""
    from .agent import run_agent_loop

    finding_json = json.dumps(asdict(vf.finding), indent=2)
    evidence, n_files = _gather_evidence(vf.finding, ctx)

    work_dir = Path(ctx.base_dir) / state.state_dir / state.run_id / "patch-gen"
    try:
        wt = _worktree(ctx.base_dir, work_dir)
        wt.__enter__()
    except RuntimeError as e:
        fmt.info(f"    patch: worktree failed: {e}")
        return PatchGenerationResult(
            error_code="patch_worktree_error", error=f"worktree failed: {e}"
        )

    try:
        prompt = (
            f"Fix the following security finding with the smallest correct change. "
            f"Use edit_file to make the fix. Do not make unrelated changes.\n\n"
            f"{finding_json}\n\n"
            f"Committed source for affected files:\n{evidence}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are fixing a security bug. Make the minimal correct fix using edit_file.",
            },
            {"role": "user", "content": prompt},
        ]

        kw = _make_isolated_loop_kwargs(ctx, work_dir, max_turns=patch_max_turns)

        # Pre-seed the file tracker so the agent can edit without a read_file
        # round-trip — the committed source is already in the prompt.
        tracker = kw.get("file_tracker")
        if tracker is not None:
            evidence_paths = _evidence_file_paths(vf.finding)
            for rel in evidence_paths:
                tracker.record_read(str(work_dir / rel))

        try:
            _answer, exhausted = run_agent_loop(messages, ctx.tools, **kw)
        except Exception as e:
            fmt.info(f"    patch: agent loop failed: {e}")
            return PatchGenerationResult(
                error_code="patch_agent_error", error=f"agent loop failed: {e}"
            )
        finally:
            _write_audit_trace(
                ctx, messages, task=f"audit: phase 5 patch {vf.finding.title}"
            )

        if exhausted:
            fmt.info("    patch: turn budget exhausted, discarding incomplete work")
            return PatchGenerationResult(
                error_code="patch_turn_budget_exhausted",
                error="turn budget exhausted",
            )

        diff = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            cwd=str(work_dir),
            timeout=10,
        )
        patch_text = diff.stdout.decode(errors="replace").strip()
        if not patch_text:
            fmt.info("    patch: no changes produced")
            return PatchGenerationResult(
                error_code="patch_no_diff", error="no changes produced"
            )
        return PatchGenerationResult(patch_text=patch_text + "\n")
    finally:
        wt.__exit__(None, None, None)


def _phase5_report(
    vf: VerifiedFinding,
    patch_filename: str,
    patch_text: str,
    ctx: InputContext,
) -> str:
    """Generate the markdown report."""
    finding_json = json.dumps(asdict(vf.finding), indent=2)
    reproducer_json = json.dumps(vf.reproducer, indent=2) if vf.reproducer else "{}"
    proof_kind = (vf.reproducer or {}).get("proof_kind") or "unspecified"
    evidence, _n = _gather_evidence(vf.finding, ctx)

    system = _PHASE5_REPORT_TEMPLATE.format(provenance_url=AUDIT_PROVENANCE_URL)
    suffix = (
        f"Verified finding:\n{finding_json}\n\n"
        f"Proof kind: {proof_kind}\n\n"
        f"Reproducer summary:\n{reproducer_json}\n\n"
        f"Affected source:\n{evidence}\n\n"
        f"Patch ({patch_filename}):\n```diff\n{patch_text}```"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": suffix},
    ]
    return _call_audit_llm(
        ctx, messages, trace_task=f"audit: phase 5 report {vf.finding.title}"
    )


def _make_isolated_loop_kwargs(
    ctx: "InputContext",
    work_dir: Path,
    max_turns: int | None = None,
) -> dict:
    """Build loop kwargs for an isolated agent loop in a worktree."""
    from .thinking import ThinkingState
    from .todo import TodoState
    from .tracker import FileAccessTracker

    kw = dict(ctx.loop_kwargs)
    kw["base_dir"] = str(work_dir)
    kw["max_turns"] = max_turns if max_turns is not None else kw.get("max_turns", 100)
    kw["thinking_state"] = ThinkingState(verbose=False)
    kw["todo_state"] = TodoState(verbose=False)
    kw["snapshot_state"] = None
    kw["file_tracker"] = FileAccessTracker()
    kw["extra_write_roots"] = []
    kw["skill_read_roots"] = []
    kw["skills_catalog"] = {}
    kw["verbose"] = False
    for k in (
        "compaction_state",
        "mcp_manager",
        "a2a_manager",
        "subagent_manager",
        "report",
        "event_callback",
        "cancel_flag",
        "turn_state",
    ):
        kw.pop(k, None)
    return kw


class _worktree:
    """Context manager for a temporary git worktree from HEAD."""

    def __init__(self, base_dir: str, work_dir: Path):
        self.base_dir = base_dir
        self.work_dir = work_dir

    def __enter__(self) -> Path:
        self.work_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.work_dir.exists():
            _git(["worktree", "remove", "--force", str(self.work_dir)], self.base_dir)
        _git(
            ["worktree", "add", "--detach", str(self.work_dir), "HEAD"],
            self.base_dir,
        )
        return self.work_dir

    def __exit__(self, *exc):
        try:
            _git(
                ["worktree", "remove", "--force", str(self.work_dir)],
                self.base_dir,
            )
        except RuntimeError:
            pass
        return False


# ---------------------------------------------------------------------------
# Single-finding verification (4a + 4b + 4c)
# ---------------------------------------------------------------------------


def _verify_single_finding(
    finding: FindingRecord,
    state: AuditRunState,
    ctx: InputContext,
    work_dir: Path,
) -> VerifiedFinding | None:
    """Run the PoC-based verifier on one finding. Returns None if discarded."""
    reproducer = _phase4c_reproduce(finding, state, ctx, work_dir)
    key = _finding_key(finding)
    if reproducer is None:
        status = state.adversarial_state.get(key, {}).get("status", "")
        outcome = _DISPROOF_DISCARDED_AGREEMENT.get(status)
        if outcome is not None:
            _set_agreement(state, key, outcome)
        fmt.info(f"  discarded (no reproduction): {finding.title}")
        return None

    status = state.adversarial_state.get(key, {}).get("status", "")
    outcome = _DISPROOF_REPRODUCED_AGREEMENT.get(status)
    if outcome is not None:
        _set_agreement(state, key, outcome)
    adv_record = state.adversarial_state.get(key)

    return VerifiedFinding(
        finding=finding,
        correctness_reason="verified by proof-of-concept reproduction",
        rebuttal_reason="not used; PoC verifier is authoritative",
        reproducer=reproducer,
        adversarial=dict(adv_record) if adv_record else None,
    )


def _verify_one_finding(
    item: tuple[str, FindingRecord],
    state: AuditRunState,
    ctx: "InputContext",
) -> VerificationResult:
    """Verify a single finding with retry on transient errors. Never raises."""
    finding_key, finding = item
    work_dir = (
        Path(ctx.base_dir)
        / state.state_dir
        / state.run_id
        / "verify"
        / finding_key
        / "work"
    )
    attempts = 0
    try:
        attempts += 1
        verified = _verify_single_finding(finding, state, ctx, work_dir)
    except _TransientVerifierError as e:
        fmt.info(
            f"  [{finding_key}] retrying {finding.title} after transient error: {e}"
        )
        try:
            attempts += 1
            verified = _verify_single_finding(finding, state, ctx, work_dir)
        except Exception as e2:
            return VerificationResult(
                finding_key=finding_key, error=str(e2), attempts=attempts
            )
    except Exception as e:
        return VerificationResult(
            finding_key=finding_key, error=str(e), attempts=attempts
        )

    if verified is None:
        return VerificationResult(
            finding_key=finding_key, discarded=True, attempts=attempts
        )
    return VerificationResult(
        finding_key=finding_key, verified_finding=verified, attempts=attempts
    )


# ---------------------------------------------------------------------------
# Parallel batch helper
# ---------------------------------------------------------------------------


def _run_batch(fn, items, max_workers: int = 4):
    """Run fn(item) in parallel, return results preserving order."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                _debug_log("batch_error", idx=idx, error=str(e))
                fmt.warning(f"batch item {idx} failed: {e}")
                results[idx] = None
    return results


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------


def _make_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] if slug else "finding"


def _artifact_key(vf: VerifiedFinding) -> str:
    return _finding_key(vf.finding)


def _artifact_filenames(index: int, finding: FindingRecord) -> tuple[str, str]:
    slug = _make_slug(finding.title)
    return f"{index:03d}-{slug}.patch", f"{index:03d}-{slug}.md"


def _ensure_artifact_state(state: AuditRunState) -> None:
    """Ensure each verified finding has a stable artifact entry."""
    current: dict[str, VerifiedFinding] = {
        _artifact_key(vf): vf for vf in state.verified_findings
    }
    for key in list(state.artifact_state):
        if key not in current:
            del state.artifact_state[key]

    for key, vf in current.items():
        if key in state.artifact_state:
            continue
        next_idx = (
            max(
                (int(entry["index"]) for entry in state.artifact_state.values()),
                default=0,
            )
            + 1
        )
        patch_filename, report_filename = _artifact_filenames(next_idx, vf.finding)
        state.artifact_state[key] = {
            "status": "pending",
            "index": next_idx,
            "patch_filename": patch_filename,
            "report_filename": report_filename,
            "attempts": 0,
            "last_error_code": None,
            "last_error": None,
            "last_patch_max_turns": None,
        }


def _parse_finding_selector(raw: str, total: int) -> set[int]:
    """Parse a 1-based finding selector into zero-based indexes."""
    selected: set[int] = set()
    if raw.strip() == "":
        raise ValueError("--finding requires a non-empty selector")
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--finding contains an empty selector")
        if "-" in part:
            pieces = part.split("-", 1)
            if not pieces[0] or not pieces[1]:
                raise ValueError(f"invalid --finding range {part!r}")
            try:
                start = int(pieces[0])
                end = int(pieces[1])
            except ValueError as e:
                raise ValueError(f"invalid --finding range {part!r}") from e
            if start > end:
                raise ValueError(f"invalid --finding range {part!r}")
            nums = range(start, end + 1)
        else:
            try:
                nums = (int(part),)
            except ValueError as e:
                raise ValueError(f"invalid --finding selector {part!r}") from e
        for n in nums:
            if n < 1 or n > total:
                raise ValueError(
                    f"--finding index {n} out of range; valid range is 1..{total}"
                )
            selected.add(n - 1)
    if not selected:
        raise ValueError("--finding did not select any findings")
    return selected


def _reset_artifact_targets_for_regen(
    state: AuditRunState, selected_indexes: set[int] | None
) -> None:
    _ensure_artifact_state(state)
    selected_keys = {
        _artifact_key(vf)
        for i, vf in enumerate(state.verified_findings)
        if selected_indexes is None or i in selected_indexes
    }
    for key in selected_keys:
        entry = state.artifact_state[key]
        entry["status"] = "pending"
        entry["last_error_code"] = None
        entry["last_error"] = None


def _artifact_summary(state: AuditRunState) -> tuple[int, int, int]:
    entries = [
        state.artifact_state[_artifact_key(vf)] for vf in state.verified_findings
    ]
    failed = sum(1 for e in entries if e.get("status") == "failed")
    pending = sum(1 for e in entries if e.get("status") == "pending")
    written = sum(1 for e in entries if e.get("status") == "written")
    return failed, pending, written


def _mark_artifact_failed(
    entry: dict,
    state: AuditRunState,
    fi: int,
    total: int,
    error_code: str,
    error_msg: str,
) -> None:
    entry["status"] = "failed"
    entry["last_error_code"] = error_code
    entry["last_error"] = error_msg
    state.save()
    fmt.info(f"  [{fi}/{total}] failed ({error_msg})")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _load_audit_config(base_dir: str) -> dict:
    """Load merged ``[audit]`` settings from global+project config.

    Returns a dict with keys ``force_review``, ``force_review_sources``,
    ``patch_max_turns``, ``budget_tokens``, ``max_gapfill_tasks``,
    ``max_hunt_tasks``, ``proof_strict``, ``reviewer_profile``,
    ``reviewer_model``, ``hunt_tasks``. Errors loading config are non-fatal: the
    rest of swival has already validated the file at startup, so failures are
    reported as warnings and treated as defaults.
    """
    empty = {
        "force_review": [],
        "force_review_sources": {},
        "patch_max_turns": None,
        "budget_tokens": None,
        "max_gapfill_tasks": None,
        "max_hunt_tasks": None,
        "proof_strict": None,
        "reviewer_profile": "",
        "reviewer_model": "",
        "hunt_tasks": [],
    }
    try:
        from .config import (
            _load_single,
            global_config_dir,
            merge_audit_bool,
            merge_audit_force_review,
            merge_audit_hunt_tasks,
            merge_audit_int,
            merge_audit_patch_max_turns,
            merge_audit_reviewer,
        )

        global_path = global_config_dir() / "config.toml"
        project_path = Path(base_dir).resolve() / "swival.toml"
        global_audit = _load_single(global_path, str(global_path)).get("audit")
        project_audit = _load_single(project_path, str(project_path)).get("audit")
    except Exception as e:
        fmt.warning(f"audit: ignoring swival.toml audit config (load failed: {e})")
        return empty
    globs, sources = merge_audit_force_review(global_audit, project_audit)
    profile, model = merge_audit_reviewer(global_audit, project_audit)
    return {
        "force_review": globs,
        "force_review_sources": sources,
        "patch_max_turns": merge_audit_patch_max_turns(global_audit, project_audit),
        "budget_tokens": merge_audit_int(global_audit, project_audit, "budget_tokens"),
        "max_gapfill_tasks": merge_audit_int(
            global_audit, project_audit, "max_gapfill_tasks"
        ),
        "max_hunt_tasks": merge_audit_int(
            global_audit, project_audit, "max_hunt_tasks"
        ),
        "proof_strict": merge_audit_bool(global_audit, project_audit, "proof_strict"),
        "reviewer_profile": profile,
        "reviewer_model": model,
        "hunt_tasks": merge_audit_hunt_tasks(global_audit, project_audit),
    }


def _resolve_force_review(
    globs: list[str],
    sources: dict[str, str],
    mandatory_files: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Match force_review globs against the mandatory file list.

    Returns ``(matches, warnings)`` where ``matches[path]`` is the
    *strongest* origin tag of any glob that matched (project beats
    global on ties). ``warnings`` contains text for project-origin
    globs that matched zero files.
    """
    matches: dict[str, str] = {}
    warnings: list[str] = []
    for g in globs:
        source = sources.get(g, "project")
        matched = [f for f in mandatory_files if _match_path_glob(f, g)]
        for f in matched:
            if matches.get(f) != "project":
                matches[f] = source
        if not matched and source == "project":
            warnings.append(
                f"audit: swival.toml force_review glob {g!r} matched zero files"
            )
    return matches, warnings


def run_audit_command(cmd_arg: str, ctx: InputContext) -> str:
    """Entry point for the /audit command. Returns summary text."""
    global _debug_log_path

    base_dir = ctx.base_dir
    workers = 4

    arg = cmd_arg.strip()
    resume = False
    regen = False
    debug = False
    select_all = False
    measure_triage = False
    hunt_mode = False
    proof_strict = False
    trace_reachability = False
    patch_max_turns_cli: int | None = None
    budget_tokens_cli: int | None = None
    gapfill_cli: int | None = None
    finding_selector: str | None = None
    focus: list[str] | None = None

    parts = arg.split()
    filtered = []
    i = 0
    while i < len(parts):
        if parts[i] == "--resume":
            resume = True
        elif parts[i] == "--regen":
            regen = True
        elif parts[i] == "--debug":
            debug = True
        elif parts[i] == "--all":
            select_all = True
        elif parts[i] == "--measure-triage":
            measure_triage = True
        elif parts[i] == "--hunt":
            hunt_mode = True
        elif parts[i] == "--proof-strict":
            proof_strict = True
        elif parts[i] == "--trace-reachability":
            trace_reachability = True
        elif parts[i] == "--workers":
            if i + 1 >= len(parts):
                return "error: --workers requires an integer"
            i += 1
            try:
                workers = int(parts[i])
            except ValueError:
                return f"error: --workers requires an integer, got {parts[i]!r}"
        elif parts[i] == "--patch-max-turns":
            if i + 1 >= len(parts):
                return "error: --patch-max-turns requires an integer"
            i += 1
            try:
                patch_max_turns_cli = int(parts[i])
            except ValueError:
                return f"error: --patch-max-turns requires an integer, got {parts[i]!r}"
            if patch_max_turns_cli < 1:
                return "error: --patch-max-turns must be at least 1"
        elif parts[i] == "--budget-tokens":
            if i + 1 >= len(parts):
                return "error: --budget-tokens requires an integer"
            i += 1
            try:
                budget_tokens_cli = int(parts[i].replace("_", "").replace(",", ""))
            except ValueError:
                return f"error: --budget-tokens requires an integer, got {parts[i]!r}"
            if budget_tokens_cli < 0:
                return "error: --budget-tokens must be non-negative"
        elif parts[i] == "--gapfill":
            if i + 1 >= len(parts):
                return "error: --gapfill requires an integer"
            i += 1
            try:
                gapfill_cli = int(parts[i])
            except ValueError:
                return f"error: --gapfill requires an integer, got {parts[i]!r}"
            if gapfill_cli < 0:
                return "error: --gapfill must be non-negative"
        elif parts[i] == "--finding":
            if finding_selector is not None:
                return "error: --finding may only be provided once; use --finding 2,5"
            if i + 1 >= len(parts) or parts[i + 1].startswith("-"):
                return "error: --finding requires a selector"
            i += 1
            finding_selector = parts[i]
        elif parts[i].startswith("-"):
            return (
                f"error: unknown option {parts[i]!r}. "
                f"Known flags: --resume, --regen, --debug, --all, "
                f"--measure-triage, --hunt, --workers N, --patch-max-turns N, "
                f"--budget-tokens N, --finding N."
            )
        else:
            filtered.append(parts[i])
        i += 1
    if filtered:
        focus = _normalize_focus(filtered)

    if finding_selector is not None and not regen:
        return "error: --finding requires --regen"

    if trace_reachability:
        return (
            "error: --trace-reachability is reserved for the upcoming in-repo "
            "reachability trace phase; not yet implemented."
        )
    if gapfill_cli is not None:
        return (
            "error: --gapfill is reserved for the upcoming observable-coverage "
            "gapfill phase; not yet implemented."
        )

    audit_cfg = _load_audit_config(base_dir)
    force_review = audit_cfg["force_review"]
    force_review_sources = audit_cfg["force_review_sources"]
    config_patch_max_turns = audit_cfg["patch_max_turns"]
    patch_max_turns = (
        patch_max_turns_cli
        if patch_max_turns_cli is not None
        else config_patch_max_turns or _DEFAULT_PATCH_MAX_TURNS
    )
    budget_tokens = (
        budget_tokens_cli
        if budget_tokens_cli is not None
        else int(audit_cfg.get("budget_tokens") or 0)
    )
    if audit_cfg.get("proof_strict") is True:
        proof_strict = True
    reviewer_model = audit_cfg.get("reviewer_model") or ""
    if audit_cfg.get("reviewer_profile"):
        return (
            "error: [audit.reviewer] profile is reserved for a future release "
            "(needs provider/api_base overlay). Use [audit.reviewer] model "
            "for a same-provider reviewer model override."
        )
    if int(audit_cfg.get("max_gapfill_tasks") or 0) > 0:
        return (
            "error: [audit] max_gapfill_tasks is reserved for the upcoming "
            "observable-coverage gapfill phase; not yet implemented."
        )
    max_hunt_tasks = int(audit_cfg.get("max_hunt_tasks") or 0)
    configured_hunt_tasks = list(audit_cfg.get("hunt_tasks") or [])

    if debug:
        log_dir = Path(base_dir) / ".swival" / "audit"
        log_dir.mkdir(parents=True, exist_ok=True)
        _debug_log_path = log_dir / "debug.jsonl"
        _debug_log("audit_start", args=cmd_arg.strip())
        fmt.info(f"debug log: {_debug_log_path}")
    else:
        _debug_log_path = None

    state_dir = Path(base_dir) / ".swival" / "audit"

    try:
        return _run_audit_phases(
            cmd_arg,
            ctx,
            base_dir,
            state_dir,
            workers,
            resume,
            regen,
            focus,
            select_all,
            force_review=force_review,
            force_review_sources=force_review_sources,
            measure_triage=measure_triage,
            patch_max_turns=patch_max_turns,
            finding_selector=finding_selector,
            hunt_mode=hunt_mode,
            budget_tokens=budget_tokens,
            max_hunt_tasks=max_hunt_tasks,
            configured_hunt_tasks=configured_hunt_tasks,
            proof_strict=proof_strict,
            reviewer_model=reviewer_model,
        )
    finally:
        _debug_log_path = None


def _run_audit_phases(
    cmd_arg: str,
    ctx: InputContext,
    base_dir: str,
    state_dir: Path,
    workers: int,
    resume: bool,
    regen: bool,
    focus: list[str] | None,
    select_all: bool = False,
    *,
    force_review: list[str] | None = None,
    force_review_sources: dict[str, str] | None = None,
    measure_triage: bool = False,
    patch_max_turns: int = _DEFAULT_PATCH_MAX_TURNS,
    finding_selector: str | None = None,
    hunt_mode: bool = False,
    budget_tokens: int = 0,
    max_hunt_tasks: int = 0,
    configured_hunt_tasks: list[dict] | None = None,
    proof_strict: bool = False,
    reviewer_model: str = "",
) -> str:
    configured_hunt_tasks = list(configured_hunt_tasks or [])
    force_review = list(force_review or [])
    force_review_sources = dict(force_review_sources or {})
    selected_indexes: set[int] | None = None
    if resume or regen:
        try:
            commit = _git(["rev-parse", "HEAD"], base_dir)
        except RuntimeError as e:
            return f"error: cannot resolve git state: {e}"

        state = AuditRunState.find_resumable(
            state_dir, commit, focus, include_done=regen
        )
        if state is None:
            label = "regenerable" if regen else "resumable"
            return f"error: no {label} audit found for current commit and scope."
        if measure_triage != state.measure_triage:
            return (
                f"error: --measure-triage mismatch. saved run was "
                f"{'measure-triage' if state.measure_triage else 'normal'}; "
                f"this invocation is "
                f"{'measure-triage' if measure_triage else 'normal'}. "
                f"Start a fresh run instead of resuming."
            )
        if hunt_mode != state.hunt_mode:
            return (
                f"error: --hunt mismatch. saved run was "
                f"{'hunt' if state.hunt_mode else 'file-centric'}; "
                f"this invocation is {'hunt' if hunt_mode else 'file-centric'}. "
                f"Start a fresh run instead of resuming."
            )
        if proof_strict != state.proof_strict:
            saved = "strict" if state.proof_strict else "balanced"
            now = "strict" if proof_strict else "balanced"
            return (
                f"error: --proof-strict mismatch. saved run was {saved}; "
                f"this invocation is {now}. "
                f"Start a fresh run instead of resuming."
            )
        # Budget override on resume is opt-in: 0 (the default when no flag is
        # passed) keeps the saved cap. To disable, start a fresh run.
        if budget_tokens is not None and int(budget_tokens) > 0:
            state.budget_tokens = int(budget_tokens)
        if regen:
            if not state.verified_findings:
                return "error: no verified findings to regenerate artifacts for."
            if finding_selector is not None:
                try:
                    selected_indexes = _parse_finding_selector(
                        finding_selector, len(state.verified_findings)
                    )
                except ValueError as e:
                    return f"error: {e}"
            state.phase = "artifacts"
            _reset_artifact_targets_for_regen(state, selected_indexes)
            state.save()
            label = (
                f"{len(selected_indexes)} selected"
                if selected_indexes is not None
                else f"{len(state.verified_findings)} verified"
            )
            fmt.info(
                f"regenerating artifacts for audit run {state.run_id} "
                f"({label} findings)"
            )
        else:
            fmt.info(f"resuming audit run {state.run_id} from phase {state.phase}")
    else:
        try:
            scope = _resolve_scope(base_dir, focus or [])
        except RuntimeError as e:
            return f"error: cannot resolve git scope: {e}"

        if not scope.mandatory_files:
            return "No auditable files found in scope."

        state = AuditRunState(
            run_id=str(uuid.uuid4())[:8],
            scope=scope,
            queued_files=list(scope.mandatory_files),
            state_dir=state_dir,
            select_all=select_all,
            measure_triage=measure_triage,
            hunt_mode=hunt_mode,
            budget_tokens=int(budget_tokens or 0),
            proof_strict=bool(proof_strict),
        )
        mode_flags = []
        if state.select_all:
            mode_flags.append("--all")
        if state.measure_triage:
            mode_flags.append("--measure-triage")
        if state.hunt_mode:
            mode_flags.append("--hunt")
        if state.proof_strict:
            mode_flags.append("--proof-strict")
        if state.budget_tokens:
            mode_flags.append(f"--budget-tokens {state.budget_tokens}")
        mode_tail = (" " + " ".join(mode_flags)) if mode_flags else ""
        fmt.info(
            f"audit {state.run_id}: {len(scope.mandatory_files)} files, "
            f"branch={scope.branch}, commit={scope.commit[:8]}{mode_tail}"
        )
        if len(scope.mandatory_files) > _LARGE_SCOPE_THRESHOLD:
            n = len(scope.mandatory_files)
            if state.select_all:
                preamble = f"{n} files in scope with --all (triage selection skipped)"
                detail = (
                    "phase 3 will deep-review every file in scope, "
                    "issuing at least one LLM call per file"
                )
                hint = "/audit --all <subdir>"
            else:
                preamble = f"{n} files in scope"
                detail = f"phase 2 may issue up to {n} LLM calls"
                hint = "/audit <subdir>"
            fmt.warning(
                f"{preamble}. {detail}. "
                f"consider narrowing with `{hint}` (one or more paths/globs)."
            )

    # Phase 1: scope + profile
    if state.phase == "init":
        fmt.info(
            f"phase 1: loading {len(state.scope.mandatory_files)} file contents..."
        )
        content_cache = _load_file_contents(state.scope.mandatory_files, base_dir)
        fmt.info("phase 1: building import/caller indices...")
        state.import_index, state.caller_index = _build_context_indices(
            state.scope.mandatory_files, content_cache
        )
        fmt.info("phase 1: ordering by attack surface...")
        state.queued_files, state.attack_scores = _order_by_attack_surface(
            state.scope.mandatory_files, content_cache
        )
        fmt.info("phase 1: calling LLM for repo profile...")
        state.repo_profile = _phase1_repo_profile(state, ctx)
        if state.hunt_mode:
            fmt.info("phase 1: generating hunt tasks from sink/source scan...")
            effective_max = max_hunt_tasks
            # Soft budget cap: assume each hunt task spends roughly its average
            # allocation share. If a configured max would blow past the global
            # hunt budget, lower it. Falls back to ``max_hunt_tasks`` when no
            # budget is set (planner returns inf there).
            if state.budget_tokens > 0:
                hunt_alloc = BudgetPlanner(state).allocation("hunt")
                est_per_task = 20_000  # conservative; revised by per-task accounting
                budget_cap = max(1, int(hunt_alloc // est_per_task))
                if effective_max <= 0 or budget_cap < effective_max:
                    effective_max = budget_cap
            tasks = _generate_hunt_tasks(
                state,
                content_cache,
                configured=configured_hunt_tasks,
                max_tasks=effective_max,
            )
            for t in tasks:
                state.hunt_tasks[t.id] = t
            fmt.info(
                f"phase 1: {len(tasks)} hunt tasks queued across "
                f"{len({t.attack_class for t in tasks})} attack classes"
            )
            state.phase = "hunt"
        else:
            state.phase = "triage"
        state.save()
        fmt.info(
            f"phase 1 complete. profile: {state.repo_profile.get('summary', '')[:80]}"
        )

    if state.phase == "hunt":
        # Reset stale `running` rows so a previous crash mid-batch retries.
        for t in state.hunt_tasks.values():
            if t.status == "running":
                t.status = "pending"

        budget = BudgetPlanner(state)
        max_hunt_attempts = 3

        for hunt_attempt in range(max_hunt_attempts):
            pending_tasks = [
                t
                for t in state.hunt_tasks.values()
                if t.status in ("pending", "failed")
            ]
            if not pending_tasks:
                break

            if budget.should_throttle_low_priority():
                dropped = [t for t in pending_tasks if t.priority == "low"]
                for t in dropped:
                    t.status = "failed"
                    t.last_error = "budget throttle: low-priority tasks dropped"
                pending_tasks = [t for t in pending_tasks if t.priority != "low"]
                if dropped:
                    fmt.warning(
                        f"phase hunt: budget over 50% used, dropped "
                        f"{len(dropped)} low-priority task(s)"
                    )

            if budget.exhausted():
                for t in pending_tasks:
                    t.status = "failed"
                    t.last_error = "budget exhausted before task started"
                fmt.warning("phase hunt: global token budget exhausted; halting")
                pending_tasks = []
                state.save()
                break

            if hunt_attempt == 0:
                fmt.info(
                    f"phase hunt: running {len(pending_tasks)} hunter task(s) "
                    f"with {workers} worker(s)..."
                )
            else:
                fmt.info(
                    f"phase hunt: retrying {len(pending_tasks)} task(s) "
                    f"(attempt {hunt_attempt + 1}/{max_hunt_attempts})..."
                )

            def _hunt(task):
                return (task, _phase_hunt_one(task, state, ctx))

            for batch_start in range(0, len(pending_tasks), workers * 2):
                batch = pending_tasks[batch_start : batch_start + workers * 2]
                for t in batch:
                    t.status = "running"
                    t.attempts += 1
                state.save()

                results = _run_batch(_hunt, batch, max_workers=workers)
                for entry in results:
                    if entry is None:
                        continue
                    task, (findings, coverage, error) = entry
                    if error is not None:
                        task.status = "failed"
                        task.last_error = error
                        fmt.warning(
                            f"  task {task.id} ({task.attack_class}) failed: {error[:120]}"
                        )
                        continue
                    task.status = "done"
                    task.last_error = ""
                    state.coverage_records[task.id] = coverage
                    if findings:
                        state.proposed_findings.extend(findings)
                    fmt.info(
                        f"  task {task.id} {task.attack_class}: "
                        f"{len(findings)} finding(s) "
                        f"(coverage confidence={coverage.confidence})"
                    )
                state.save()

        # Do not advance to verification while any task is failed or running:
        # a false "no findings" answer is worse than asking the operator to
        # resume.
        stuck = [t for t in state.hunt_tasks.values() if t.status == "failed"]
        if stuck:
            n_attempted = sum(t.attempts for t in stuck)
            state.save()
            return (
                f"Audit incomplete: {len(stuck)} hunt task(s) failed after "
                f"{n_attempted} attempt(s). Use /audit --resume to retry. "
                f"First failure: {stuck[0].id} ({stuck[0].attack_class}): "
                f"{stuck[0].last_error[:120] or 'unknown error'}"
            )

        running = [t for t in state.hunt_tasks.values() if t.status == "running"]
        if running:
            state.save()
            return (
                f"Audit incomplete: {len(running)} hunt task(s) still marked "
                f"running. Use /audit --resume to retry."
            )

        # Satisfy the file-centric completion guards so the safety nets at the
        # end of the audit accept this run.
        state.reviewed_files.update(state.scope.mandatory_files)
        covered = {f for t in state.hunt_tasks.values() for f in t.seed_files}
        state.candidate_files = sorted(covered)
        state.deep_reviewed_files.update(state.candidate_files)
        state.deep_reviewed_files.update(state.scope.mandatory_files)
        reach_tasks = _generate_reachability_tasks(state)
        if reach_tasks:
            for t in reach_tasks:
                state.hunt_tasks[t.id] = t
            fmt.info(
                f"phase hunt: queued {len(reach_tasks)} reachability task(s) "
                f"for LOCAL_ONLY findings"
            )
        state.phase = _HUNT_TASK_KIND_REACHABILITY
        state.save()
        n_hunt_done = sum(
            1
            for t in state.hunt_tasks.values()
            if t.task_kind == _HUNT_TASK_KIND_HUNT and t.status == "done"
        )
        n_hunt_total = sum(
            1 for t in state.hunt_tasks.values() if t.task_kind == _HUNT_TASK_KIND_HUNT
        )
        fmt.info(
            f"phase hunt complete. {len(state.proposed_findings)} proposed "
            f"finding(s) from {n_hunt_done}/{n_hunt_total} hunt task(s)."
        )

    if state.phase == _HUNT_TASK_KIND_REACHABILITY:
        finding_by_key = {_finding_key(f): f for f in state.proposed_findings}

        for t in state.hunt_tasks.values():
            if t.task_kind == _HUNT_TASK_KIND_REACHABILITY and t.status == "running":
                t.status = "pending"

        # Reachability tasks for sibling findings off the same parent hunt task
        # routinely share seed files, so prefetch the union once per phase
        # entry and reuse across batches.
        reach_seed_union = sorted(
            {
                f
                for t in state.hunt_tasks.values()
                if t.task_kind == _HUNT_TASK_KIND_REACHABILITY
                for f in t.seed_files[:6]
            }
        )
        reach_file_cache = (
            _load_file_contents(reach_seed_union, base_dir) if reach_seed_union else {}
        )

        budget = BudgetPlanner(state)
        max_reach_attempts = 3
        for reach_attempt in range(max_reach_attempts):
            pending = [
                t
                for t in state.hunt_tasks.values()
                if t.task_kind == _HUNT_TASK_KIND_REACHABILITY
                and t.status in ("pending", "failed")
            ]
            if not pending:
                break

            if budget.exhausted():
                for t in pending:
                    t.status = "failed"
                    t.last_error = "budget exhausted before task started"
                fmt.warning(
                    "phase reachability: global token budget exhausted; halting"
                )
                state.save()
                break

            if reach_attempt == 0:
                fmt.info(
                    f"phase reachability: running {len(pending)} trace task(s) "
                    f"with {workers} worker(s)..."
                )
            else:
                fmt.info(
                    f"phase reachability: retrying {len(pending)} task(s) "
                    f"(attempt {reach_attempt + 1}/{max_reach_attempts})..."
                )

            def _trace(task):
                parent = finding_by_key.get(task.parent_finding_key)
                if parent is None:
                    return (task, (None, "parent finding missing"))
                return (
                    task,
                    _phase_reachability_one(
                        task, parent, state, ctx, file_evidence=reach_file_cache
                    ),
                )

            for batch_start in range(0, len(pending), workers * 2):
                batch = pending[batch_start : batch_start + workers * 2]
                for t in batch:
                    t.status = "running"
                    t.attempts += 1
                state.save()

                results = _run_batch(_trace, batch, max_workers=workers)
                for entry in results:
                    if entry is None:
                        continue
                    task, (parsed, error) = entry
                    if error is not None:
                        task.status = "failed"
                        task.last_error = error
                        fmt.warning(
                            f"  trace {task.id} ({task.attack_class}) failed: "
                            f"{error[:120]}"
                        )
                        continue
                    task.status = "done"
                    task.last_error = ""
                    parent = finding_by_key.get(task.parent_finding_key)
                    if parent is not None and parsed is not None:
                        _apply_reachability_result(parent, parsed)
                        fmt.info(
                            f"  trace {task.id}: "
                            f"{parent.reachability_status}: {parent.title}"
                        )
                state.save()

        stuck = [
            t
            for t in state.hunt_tasks.values()
            if t.task_kind == _HUNT_TASK_KIND_REACHABILITY and t.status == "failed"
        ]
        if stuck:
            n_attempted = sum(t.attempts for t in stuck)
            state.save()
            return (
                f"Audit incomplete: {len(stuck)} reachability task(s) failed "
                f"after {n_attempted} attempt(s). Use /audit --resume to retry. "
                f"First failure: {stuck[0].id}: "
                f"{stuck[0].last_error[:120] or 'unknown error'}"
            )

        before = len(state.proposed_findings)
        discarded = _filter_to_reachable_findings(state)
        if discarded:
            fmt.info(
                f"phase reachability: dropped {discarded} unreachable "
                f"finding(s) ({before} -> {len(state.proposed_findings)})"
            )
        state.phase = "disproof"
        state.save()

    # Resume rule: if force_review changed since the run was saved, apply
    # any new matches before the Phase-2 gate. New matches re-promote saved
    # SKIPs; removed entries are not honored (rescinding mid-audit causes
    # more confusion than it fixes — re-run from scratch instead).
    if resume and state.phase in ("triage", "deep_review") and force_review:
        force_matches, force_warnings = _resolve_force_review(
            force_review, force_review_sources, state.scope.mandatory_files
        )
        for w in force_warnings:
            fmt.warning(w)
        re_promoted = 0
        for path, source in force_matches.items():
            rec = state.triage_records.get(path)
            if rec is None:
                continue
            reason = f"forced via swival.toml ({source})"
            if reason in rec.promotion_reasons:
                continue
            rec.promotion_reasons.append(reason)
            if rec.priority == "SKIP":
                rec.priority = "ESCALATE_MEDIUM"
                if path not in state.candidate_files:
                    state.candidate_files.append(path)
                re_promoted += 1
        if re_promoted:
            fmt.info(
                f"resume: {re_promoted} file(s) promoted via updated "
                f"swival.toml force_review"
            )
            state.save()

    if state.phase == "triage" and state.select_all:
        state.candidate_files = list(state.queued_files)
        state.reviewed_files.update(state.queued_files)
        state.phase = "deep_review"
        state.save()
        fmt.info(
            f"phase 2: skipped (--all); {len(state.candidate_files)} files "
            f"queued for deep review"
        )

    if state.phase == "triage":

        def _triage(path):
            return _phase2_triage_one(path, state, ctx)

        def _collect_triage(results):
            for rec in results:
                if rec is not None:
                    state.triage_records[rec.path] = rec
                    state.reviewed_files.add(rec.path)

        pending = [f for f in state.queued_files if f not in state.reviewed_files]
        if pending:
            fmt.info(f"phase 2: triaging {len(pending)} files...")
            for batch_start in range(0, len(pending), workers * 2):
                batch = pending[batch_start : batch_start + workers * 2]
                _collect_triage(_run_batch(_triage, batch, max_workers=workers))
                state.save()
                done = len(state.reviewed_files)
                total = len(state.queued_files)
                fmt.info(f"  triage progress: {done}/{total}")

        for _triage_retry in range(2):
            still_pending = [
                f for f in state.queued_files if f not in state.reviewed_files
            ]
            if not still_pending:
                break
            fmt.info(
                f"phase 2: retrying {len(still_pending)} files "
                f"(attempt {_triage_retry + 2})..."
            )
            _collect_triage(_run_batch(_triage, still_pending, max_workers=workers))
            state.save()

        force_matches, force_warnings = _resolve_force_review(
            force_review, force_review_sources, state.scope.mandatory_files
        )
        for w in force_warnings:
            fmt.warning(w)
        promotions = _apply_promotions(state, force_matches)

        # Confirmation pass for low-confidence SKIPs that promotion did not
        # already rescue. Runs in parallel; bounded ~10-20% of triage cost.
        confirm_targets = [
            p
            for p, r in state.triage_records.items()
            if r.priority == "SKIP" and r.confidence == "low"
        ]
        if confirm_targets:
            fmt.info(
                f"phase 2: confirmation pass on {len(confirm_targets)} "
                f"low-confidence SKIP(s)..."
            )

            def _confirm(p):
                return _phase2_confirm_one(p, state, ctx)

            results = _run_batch(_confirm, confirm_targets, max_workers=workers)
            for original_path, new_rec in zip(confirm_targets, results):
                old = state.triage_records[original_path]
                if new_rec is None:
                    old.confirmation_outcome = "kept"
                    continue
                if new_rec.priority in ("ESCALATE_HIGH", "ESCALATE_MEDIUM"):
                    new_rec.confirmation_outcome = "promoted"
                    new_rec.promotion_reasons = list(old.promotion_reasons)
                    state.triage_records[original_path] = new_rec
                else:
                    old.confirmation_outcome = "kept"
            state.save()

        state.candidate_files = [
            path
            for path, rec in state.triage_records.items()
            if rec.priority in ("ESCALATE_HIGH", "ESCALATE_MEDIUM")
        ]
        if state.measure_triage:
            state.measurement_escalated_paths = set(state.candidate_files)
            state.candidate_files = list(state.queued_files)
            fmt.info(
                f"phase 2 (measure-triage): expanding deep-review set from "
                f"{len(state.measurement_escalated_paths)} to "
                f"{len(state.candidate_files)} files"
            )
        state.phase = "deep_review"
        state.save()

        _emit_phase2_summary(state, promotions)

    # Phase 3: deep review
    if state.phase == "deep_review":

        def _review(path):
            return _deep_review_one(path, state, ctx)

        def _collect_deep_review(results, pending_batch, total_files):
            for result in results:
                if result is None:
                    continue
                done = len(state.deep_reviewed_files)
                if result.error is not None:
                    fmt.warning(
                        f"  [{done}/{total_files}] failed: {result.path} "
                        f"({result.error[:80]})"
                    )
                    continue
                n = len(result.findings) if result.findings else 0
                if result.findings:
                    if state.measure_triage:
                        decision = (
                            "escalated"
                            if result.path in state.measurement_escalated_paths
                            else "skipped"
                        )
                        for f in result.findings:
                            f.triage_decision = decision
                    state.proposed_findings.extend(result.findings)
                state.deep_reviewed_files.add(result.path)
                done = len(state.deep_reviewed_files)
                label = f"{n} finding(s)" if n else "no findings"
                fmt.info(f"  [{done}/{total_files}] {result.path}: {label}")

        for _dr_attempt in range(3):
            pending = [
                f for f in state.candidate_files if f not in state.deep_reviewed_files
            ]
            if not pending:
                break
            total_files = len(state.candidate_files)
            if _dr_attempt == 0:
                fmt.info(f"phase 3: deep review of {len(pending)} files...")
            else:
                fmt.info(
                    f"phase 3: retrying {len(pending)} files "
                    f"(attempt {_dr_attempt + 1})..."
                )

            for batch_start in range(0, len(pending), workers * 2):
                batch = pending[batch_start : batch_start + workers * 2]
                results = _run_batch(_review, batch, max_workers=workers)
                _collect_deep_review(results, batch, total_files)
                state.save()

        if any(f not in state.deep_reviewed_files for f in state.candidate_files):
            remaining = [
                f for f in state.candidate_files if f not in state.deep_reviewed_files
            ]
            return (
                f"Audit incomplete: {len(remaining)} escalated files failed deep "
                f"review after retries. Use /audit --resume to retry."
            )

        state.phase = "disproof"
        state.save()
        metrics_summary = _format_audit_metrics(state.metrics)
        fmt.info(
            f"phase 3 complete. {len(state.proposed_findings)} proposed findings."
            + (f" ({metrics_summary})" if metrics_summary else "")
        )

    # ``gapfill`` is deliberately omitted from the skip set: under
    # --proof-strict it is the deferred bucket for reviewer transport
    # failures, and /audit --resume must actually retry those.
    if state.phase == "disproof":
        finding_keys: dict[int, str] = {
            id(f): _finding_key(f) for f in state.proposed_findings
        }
        targets = [
            (f, finding_keys[id(f)])
            for f in state.proposed_findings
            if state.adversarial_state.get(finding_keys[id(f)], {}).get("status")
            not in _DISPROOF_TERMINAL_STATUSES
        ]
        if targets:
            same_model = not reviewer_model
            mode_label = "strict" if state.proof_strict else "balanced"
            fmt.info(
                f"phase disproof: reviewing {len(targets)} finding(s) "
                f"({mode_label}; reviewer={reviewer_model or 'same-model fallback'})"
            )
            if same_model:
                state.metrics["disproof_same_model"] = (
                    int(state.metrics.get("disproof_same_model", 0)) + 1
                )

            def _disproof_worker(item):
                f, key = item
                parsed, err = _phase_disproof_one(
                    f, state, ctx, reviewer_model=reviewer_model
                )
                if err is not None and parsed is None:
                    parsed2, err2 = _phase_disproof_one(
                        f, state, ctx, reviewer_model=reviewer_model
                    )
                    if parsed2 is not None:
                        parsed, err = parsed2, None
                    else:
                        err = err2 or err
                return key, parsed, err

            disproof_workers = min(workers, 4)
            results = _run_batch(
                _disproof_worker, targets, max_workers=disproof_workers
            )
            for entry in results:
                if entry is None:
                    continue
                key, parsed, err = entry
                record = _disproof_routing(parsed, err, proof_strict=state.proof_strict)
                record["attempts"] = (
                    int(state.adversarial_state.get(key, {}).get("attempts", 0)) + 1
                )
                state.adversarial_state[key] = record
                metric_key = f"disproof_{record['status']}"
                state.metrics[metric_key] = int(state.metrics.get(metric_key, 0)) + 1
                state.save()

        invalid_keys = {
            k
            for k, v in state.adversarial_state.items()
            if v.get("status") == _DISPROOF_STATUS_INVALID
        }
        if invalid_keys:
            kept: list[FindingRecord] = []
            dropped = 0
            for f in state.proposed_findings:
                key = finding_keys.get(id(f)) or _finding_key(f)
                if key in invalid_keys:
                    _set_agreement(state, key, "agreed_pre_proof")
                    dropped += 1
                    continue
                kept.append(f)
            state.proposed_findings = kept
            fmt.info(
                f"phase disproof: discarded {dropped} INVALID finding(s) "
                f"before verification"
            )

        if state.proof_strict:
            stuck = [
                k
                for k, v in state.adversarial_state.items()
                if v.get("status") == _DISPROOF_STATUS_GAPFILL
            ]
            if stuck:
                state.save()
                return (
                    f"Audit incomplete: {len(stuck)} disproof review(s) failed "
                    f"under --proof-strict. Re-run with /audit --resume after the "
                    f"reviewer is healthy, or drop --proof-strict to fail open."
                )

        state.phase = "verification"
        state.save()

    # Phase 4: verification (parallel)
    if state.phase == "verification":
        seen_keys: set[str] = set()
        deduped: list[FindingRecord] = []
        deduped_keys: list[str] = []
        for f in state.proposed_findings:
            key = _finding_key(f)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(f)
                deduped_keys.append(key)
        if len(deduped) < len(state.proposed_findings):
            fmt.info(
                f"  deduplicated {len(state.proposed_findings)} proposed findings "
                f"to {len(deduped)}"
            )
            state.proposed_findings = deduped

        stale = [k for k in state.verification_state if k not in seen_keys]
        for k in stale:
            del state.verification_state[k]

        already_verified_keys = {
            _finding_key(vf.finding) for vf in state.verified_findings
        }
        for key in deduped_keys:
            if key not in state.verification_state:
                state.verification_state[key] = {
                    "status": "verified" if key in already_verified_keys else "pending",
                    "attempts": 0,
                    "last_error": None,
                    "summary": None,
                }

        max_verify_attempts = 3

        def _verify(item):
            return _verify_one_finding(item, state, ctx)

        for _v_attempt in range(max_verify_attempts):
            for vs in state.verification_state.values():
                if vs["status"] == "running":
                    vs["status"] = "pending"

            pending = [
                (key, f)
                for key, f in zip(deduped_keys, state.proposed_findings)
                if state.verification_state[key]["status"] in ("pending", "failed")
            ]

            if not pending:
                break

            verify_workers = min(workers, 2)
            if _v_attempt == 0:
                fmt.info(
                    f"phase 4: verifying {len(pending)} findings "
                    f"with {verify_workers} workers..."
                )
            else:
                fmt.info(
                    f"phase 4: retry {_v_attempt}/{max_verify_attempts - 1}, "
                    f"{len(pending)} findings remaining..."
                )

            for key, _ in pending:
                state.verification_state[key]["status"] = "running"
            state.save()

            results = _run_batch(_verify, pending, max_workers=verify_workers)

            verified_count = 0
            discarded_count = 0
            failed_count = 0
            total = len(pending)

            for i, result in enumerate(results):
                key = pending[i][0]
                finding_title = pending[i][1].title
                vs = state.verification_state[key]
                vs["attempts"] = vs.get("attempts", 0) + (
                    result.attempts if isinstance(result, VerificationResult) else 1
                )

                if result is None or (
                    isinstance(result, VerificationResult) and result.error is not None
                ):
                    vs["status"] = "failed"
                    vs["last_error"] = (
                        result.error
                        if isinstance(result, VerificationResult)
                        else "unexpected worker failure"
                    )
                    failed_count += 1
                    fmt.info(f"  [{i + 1}/{total}] failed: {finding_title}")
                elif result.discarded:
                    vs["status"] = "discarded"
                    vs["summary"] = "not reproduced"
                    discarded_count += 1
                    fmt.info(f"  [{i + 1}/{total}] discarded: {finding_title}")
                elif result.verified_finding is not None:
                    vs["status"] = "verified"
                    vs["summary"] = "verified by proof-of-concept reproduction"
                    state.verified_findings.append(result.verified_finding)
                    verified_count += 1
                    fmt.info(f"  [{i + 1}/{total}] verified: {finding_title}")

                state.save()

            fmt.info(
                f"  batch complete: {verified_count} verified, "
                f"{discarded_count} discarded, {failed_count} failed"
            )

        non_terminal = [
            key
            for key, vs in state.verification_state.items()
            if vs["status"] not in ("verified", "discarded")
        ]
        if non_terminal:
            n_failed = sum(
                1
                for key in non_terminal
                if state.verification_state[key]["status"] == "failed"
            )
            return (
                f"Audit incomplete: {len(non_terminal)} findings not verified "
                f"after {max_verify_attempts} attempts ({n_failed} failed). "
                f"Use /audit --resume to retry."
            )

        # Deduplicate verified_findings by content key
        seen_vf_keys: set[str] = set()
        deduped_vf: list[VerifiedFinding] = []
        for vf in state.verified_findings:
            vf_key = _finding_key(vf.finding)
            if vf_key not in seen_vf_keys:
                seen_vf_keys.add(vf_key)
                deduped_vf.append(vf)
        state.verified_findings = deduped_vf

        state.phase = "artifacts"
        state.save()
        fmt.info(f"phase 4 complete. {len(state.verified_findings)} verified findings.")

    # Phase 5: artifacts
    if state.phase == "artifacts":
        if state.verified_findings:
            artifact_dir = Path(base_dir) / state.artifact_dir
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _ensure_artifact_state(state)

            targets = []
            total = len(state.verified_findings)
            for fi, vf in enumerate(state.verified_findings, 1):
                key = _artifact_key(vf)
                entry = state.artifact_state[key]
                if selected_indexes is not None:
                    if fi - 1 in selected_indexes:
                        targets.append((fi, vf, key, entry))
                elif entry.get("status") in ("pending", "failed"):
                    targets.append((fi, vf, key, entry))

            if targets:
                fmt.info(
                    f"phase 5: generating artifacts for {len(targets)} "
                    f"of {len(state.verified_findings)} findings..."
                )

            for target_i, (fi, vf, key, entry) in enumerate(targets, 1):
                patch_filename = entry["patch_filename"]
                report_filename = entry["report_filename"]
                if selected_indexes is None:
                    progress = f"[{fi}/{total}] generating patch"
                else:
                    progress = (
                        f"[{target_i}/{len(targets)}] regenerating finding {fi}/{total}"
                    )

                fmt.info(f"  {progress}: {vf.finding.title}")
                patch_result = _phase5_patch(vf, ctx, state, patch_max_turns)
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["last_patch_max_turns"] = patch_max_turns
                if patch_result.patch_text is None:
                    _mark_artifact_failed(
                        entry,
                        state,
                        fi,
                        total,
                        patch_result.error_code or "patch_failed",
                        patch_result.error or "patch generation failed",
                    )
                    continue

                fmt.info(f"  [{fi}/{total}] generating report...")
                try:
                    report_text = _phase5_report(
                        vf, patch_filename, patch_result.patch_text, ctx
                    )
                except Exception as e:
                    _mark_artifact_failed(
                        entry, state, fi, total, "report_generation_error", str(e)
                    )
                    continue

                try:
                    (artifact_dir / patch_filename).write_text(patch_result.patch_text)
                    (artifact_dir / report_filename).write_text(report_text)
                except OSError as e:
                    _mark_artifact_failed(
                        entry, state, fi, total, "write_artifact_error", str(e)
                    )
                    continue

                entry["status"] = "written"
                entry["last_error_code"] = None
                entry["last_error"] = None
                state.save()
                fmt.info(f"  [{fi}/{total}] wrote {report_filename} + {patch_filename}")

        # Safety-net checks before marking done — rewind to the phase that
        # can fill the gap so /audit --resume actually recovers.
        unreviewed = [
            f for f in state.scope.mandatory_files if f not in state.reviewed_files
        ]
        if unreviewed:
            state.phase = "triage"
            state.save()
            return (
                f"Audit incomplete: {len(unreviewed)} files were not reviewed. "
                f"Use /audit --resume to continue."
            )
        undeep_reviewed = [
            f for f in state.candidate_files if f not in state.deep_reviewed_files
        ]
        if undeep_reviewed:
            state.phase = "deep_review"
            state.save()
            return (
                f"Audit incomplete: {len(undeep_reviewed)} escalated files failed "
                f"deep review. Use /audit --resume to continue."
            )

        failed, pending, written = _artifact_summary(state)
        if failed or pending:
            state.phase = "artifacts"
            state.save()
            return (
                "Audit incomplete: artifact generation has "
                f"{failed} failed and {pending} pending out of "
                f"{len(state.verified_findings)} verified finding(s). "
                "Use /audit --resume --patch-max-turns 75 to retry incomplete "
                "artifacts, or /audit --regen --finding 1 --patch-max-turns 75 "
                "to retry a specific finding."
            )

        if state.measure_triage:
            _emit_measure_triage_recall(state)

        state.phase = "done"
        state.save()
    else:
        _failed, _pending, written = _artifact_summary(state)

    budget = BudgetPlanner(state)
    budget_tail = ""
    if budget.enabled:
        line = budget.summary_line()
        if line:
            budget_tail = f"\n{line}"

    if written == 0:
        return (
            "No provable security bugs or security-control failures found "
            "in Git-tracked files." + budget_tail
        )

    return (
        f"Audit complete. {written} finding(s) written to {state.artifact_dir}/. "
        f"Run `ls {state.artifact_dir}/` to review." + budget_tail
    )
