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

_LARGE_SCOPE_THRESHOLD = 500

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


@dataclass
class VerifiedFinding:
    finding: FindingRecord
    correctness_reason: str
    rebuttal_reason: str
    reproducer: dict | None = None


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
    caller_index: dict[str, list[str]] = field(default_factory=dict)
    artifact_dir: Path = field(default_factory=lambda: Path("audit-findings"))
    state_dir: Path = field(default_factory=lambda: Path(".swival/audit"))
    verification_state: dict[str, dict] = field(default_factory=dict)
    next_index: int = 1
    phase: str = "init"
    metrics: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_METRICS))

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
                }
                for vf in self.verified_findings
            ],
            "repo_profile": self.repo_profile,
            "import_index": self.import_index,
            "caller_index": self.caller_index,
            "verification_state": self.verification_state,
            "next_index": self.next_index,
            "phase": self.phase,
            "metrics": self.metrics,
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
                )
            )
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
            verification_state=blob.get("verification_state", {}),
            state_dir=state_dir,
            next_index=blob.get("next_index", 1),
            phase=blob.get("phase", "init"),
            metrics=blob.get("metrics", dict(_DEFAULT_METRICS)),
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
                best_mtime = mtime
                best = cls.load(state_dir, blob["run_id"])
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
        import fnmatch

        pats = [(p, p if p.endswith("/") else p + "/") for p in focus]

        def matches(f: str) -> bool:
            return any(
                fnmatch.fnmatch(f, p) or f == p or f.startswith(pre) for p, pre in pats
            )

        tracked = [f for f in tracked if matches(f)]

    mandatory = [f for f in tracked if _is_auditable(f)]
    return AuditScope(
        branch=branch,
        commit=commit,
        tracked_files=tracked,
        mandatory_files=mandatory,
        focus=focus,
    )


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
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for f in files:
        score = _score_attack_surface(content_cache.get(f, ""))
        scored.append((-score, f))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [f for _, f in scored]


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
) -> str:
    from .agent import call_llm, ContextOverflowError
    from ._msg import _msg_content

    kw = ctx.loop_kwargs
    llm_kwargs = kw.get("llm_kwargs", {})

    cache_info = None

    def _do_call(msgs):
        nonlocal cache_info
        msg, _finish, _activity, _retries, cache_info = call_llm(
            kw["api_base"],
            kw["model_id"],
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
- End your final response with exactly one of these tokens on its own line:
  REPRODUCED
  NOTREPRODUCED"""

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
- Provenance must include a link to the Swival Security Scanner URL: {provenance_url}
- Residual Risk must be `None` unless a narrow evidence-based concern remains.
- Be terse, factual, and evidence-driven."""

# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _phase1_repo_profile(state: AuditRunState, ctx: InputContext) -> dict:
    """Build a compact repository profile from committed evidence."""
    evidence_parts = []
    # Grab manifests
    manifest_names = {
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "pom.xml",
        "build.gradle",
        "Gemfile",
        "build.zig.zon",
    }
    for f in state.scope.tracked_files:
        if Path(f).name in manifest_names:
            try:
                content = _git_show(f, ctx.base_dir)
                evidence_parts.append(f"--- {f} ---\n{content[:2000]}")
            except RuntimeError:
                pass

    # Entry point hints: files with main, app, server, handler in name
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

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no manifests found)"
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
    score = _score_attack_surface(content)

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
    raw = _call_audit_llm(ctx, messages, trace_task=f"audit: phase 2 triage {path}")
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
        )

    parsed = records[0]
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


def _finding_key(finding: FindingRecord) -> str:
    """Stable content-based key for a finding, used for verification state and worktrees."""
    blob = json.dumps(asdict(finding), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class _TransientVerifierError(Exception):
    """Raised when a verifier worker hits a transient provider or transport error."""


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
        messages = [
            {"role": "system", "content": _PHASE4_VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Proposed finding:\n{finding_json}\n\n"
                    f"Committed evidence bundle:\n{evidence}"
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
        if _REPRODUCE_KEYWORD in answer and _NO_REPRODUCE_KEYWORD not in answer:
            fmt.info(f"    verifier [{locs}]: REPRODUCED — {finding.title}")
            return {"reproduced": True, "summary": answer[-1000:]}

        fmt.info(f"    verifier [{locs}]: NOTREPRODUCED — {finding.title}")
        return None
    finally:
        wt.__exit__(None, None, None)


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
) -> str | None:
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
        return None

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

        kw = _make_isolated_loop_kwargs(ctx, work_dir, max_turns=25)

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
            return None
        finally:
            _write_audit_trace(
                ctx, messages, task=f"audit: phase 5 patch {vf.finding.title}"
            )

        if exhausted:
            fmt.info("    patch: turn budget exhausted, discarding incomplete work")
            return None

        diff = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            cwd=str(work_dir),
            timeout=10,
        )
        patch_text = diff.stdout.decode(errors="replace").strip()
        if not patch_text:
            fmt.info("    patch: no changes produced")
            return None
        return patch_text + "\n"
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
    evidence, _n = _gather_evidence(vf.finding, ctx)

    system = _PHASE5_REPORT_TEMPLATE.format(provenance_url=AUDIT_PROVENANCE_URL)
    suffix = (
        f"Verified finding:\n{finding_json}\n\n"
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
    if reproducer is None:
        fmt.info(f"  discarded (no reproduction): {finding.title}")
        return None

    return VerifiedFinding(
        finding=finding,
        correctness_reason="verified by proof-of-concept reproduction",
        rebuttal_reason="not used; PoC verifier is authoritative",
        reproducer=reproducer,
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


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_audit_command(cmd_arg: str, ctx: InputContext) -> str:
    """Entry point for the /audit command. Returns summary text."""
    global _debug_log_path

    base_dir = ctx.base_dir
    workers = 4

    arg = cmd_arg.strip()
    resume = False
    regen = False
    debug = False
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
        elif parts[i] == "--workers" and i + 1 < len(parts):
            i += 1
            try:
                workers = int(parts[i])
            except ValueError:
                return f"error: --workers requires an integer, got {parts[i]!r}"
        else:
            filtered.append(parts[i])
        i += 1
    if filtered:
        focus = _normalize_focus(filtered)

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
            cmd_arg, ctx, base_dir, state_dir, workers, resume, regen, focus
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
) -> str:
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
        if regen:
            if not state.verified_findings:
                return "error: no verified findings to regenerate artifacts for."
            state.phase = "artifacts"
            state.next_index = 1
            state.save()
            fmt.info(
                f"regenerating artifacts for audit run {state.run_id} "
                f"({len(state.verified_findings)} verified findings)"
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
        )
        fmt.info(
            f"audit {state.run_id}: {len(scope.mandatory_files)} files, "
            f"branch={scope.branch}, commit={scope.commit[:8]}"
        )
        if len(scope.mandatory_files) > _LARGE_SCOPE_THRESHOLD:
            fmt.warning(
                f"{len(scope.mandatory_files)} files in scope. "
                f"phase 2 may issue up to {len(scope.mandatory_files)} LLM calls. "
                f"consider narrowing with `/audit <subdir>` "
                f"(one or more paths/globs)."
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
        state.queued_files = _order_by_attack_surface(
            state.scope.mandatory_files, content_cache
        )
        fmt.info("phase 1: calling LLM for repo profile...")
        state.repo_profile = _phase1_repo_profile(state, ctx)
        state.phase = "triage"
        state.save()
        fmt.info(
            f"phase 1 complete. profile: {state.repo_profile.get('summary', '')[:80]}"
        )

    # Phase 2: triage
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

        state.candidate_files = [
            path
            for path, rec in state.triage_records.items()
            if rec.priority in ("ESCALATE_HIGH", "ESCALATE_MEDIUM")
        ]
        state.phase = "deep_review"
        state.save()
        fmt.info(
            f"phase 2 complete. {len(state.candidate_files)} files escalated "
            f"({sum(1 for r in state.triage_records.values() if r.priority == 'ESCALATE_HIGH')} high, "
            f"{sum(1 for r in state.triage_records.values() if r.priority == 'ESCALATE_MEDIUM')} medium)"
        )

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

        state.phase = "verification"
        state.save()
        metrics_summary = _format_audit_metrics(state.metrics)
        fmt.info(
            f"phase 3 complete. {len(state.proposed_findings)} proposed findings."
            + (f" ({metrics_summary})" if metrics_summary else "")
        )

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
    artifacts_written = 0
    if state.phase == "artifacts":
        if state.verified_findings:
            artifact_dir = Path(base_dir) / state.artifact_dir
            artifact_dir.mkdir(parents=True, exist_ok=True)

            fmt.info(
                f"phase 5: generating artifacts for {len(state.verified_findings)} findings..."
            )
            total = len(state.verified_findings)
            for fi, vf in enumerate(state.verified_findings, 1):
                idx = state.next_index
                slug = _make_slug(vf.finding.title)
                patch_filename = f"{idx:03d}-{slug}.patch"
                report_filename = f"{idx:03d}-{slug}.md"

                fmt.info(f"  [{fi}/{total}] generating patch: {vf.finding.title}")
                patch_text = _phase5_patch(vf, ctx, state)
                if patch_text is None:
                    fmt.info(f"  [{fi}/{total}] skipped (no patch produced)")
                    state.next_index += 1
                    continue

                fmt.info(f"  [{fi}/{total}] generating report...")
                report_text = _phase5_report(vf, patch_filename, patch_text, ctx)

                (artifact_dir / patch_filename).write_text(patch_text)
                (artifact_dir / report_filename).write_text(report_text)
                artifacts_written += 1

                state.next_index += 1
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

        state.phase = "done"
        state.save()

    if artifacts_written == 0:
        return (
            "No provable security bugs or security-control failures found "
            "in Git-tracked files."
        )

    return (
        f"Audit complete. {artifacts_written} finding(s) written to {state.artifact_dir}/. "
        f"Run `ls {state.artifact_dir}/` to review."
    )
