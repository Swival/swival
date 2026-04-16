# Security Audit

The `/audit` command runs a multi-phase security audit over committed Git-tracked code.

It triages files by attack surface, performs deep review on escalated files, verifies each finding with an isolated proof-of-concept agent, generates patches, and writes structured reports. Only provable bugs survive to the final output.

```text
/audit [path|glob] [--resume] [--workers N]
```

Available only in interactive mode (REPL). Runs against `HEAD`, so dirty working-directory changes are ignored.

## Quick Start

Start an audit from the REPL:

```text
swival> /audit
```

Scope it to a directory or glob:

```text
swival> /audit src/auth/
swival> /audit **/*.py
```

When the audit finishes, findings are written to `audit-findings/` in the project root:

```text
swival> /audit
Audit complete. 2 finding(s) written to audit-findings/. Run `ls audit-findings/` to review.
```

If no bugs are found:

```text
No provable logic or security bugs found in Git-tracked files.
```

## How It Works

The audit runs in five sequential phases. State is checkpointed after each phase and after every batch within a phase, so interrupted audits can be resumed.

### Phase 1: Repository Profiling

Reads manifests (`package.json`, `pyproject.toml`, `Cargo.toml`, `Makefile`, etc.) and entry-point candidates from committed code, then calls the LLM to produce a compact repository profile: detected languages, frameworks, entry points, trust boundaries, persistence layers, auth surfaces, and dangerous operations. This profile is reused as context in every subsequent phase.

Files are ordered by an attack-surface heuristic that scores keywords like `exec`, `eval`, `auth`, `token`, `sql`, `template`, and `socket`. Higher-scoring files are processed first.

### Phase 2: Triage

Each auditable file is triaged independently. The LLM sees the file contents, its attack-surface score, import/caller context, and the repository profile. It returns one of three labels:

- **ESCALATE_HIGH** — concrete suspicious path or invariant break worth deep review
- **ESCALATE_MEDIUM** — plausible concern, lower confidence
- **SKIP** — no evidence for escalation

Files labeled SKIP are not reviewed further. Triage runs in parallel with configurable worker count.

### Phase 3: Deep Review

Each escalated file goes through a two-step deep review.

**Inventory (3a):** The LLM produces a compact list of finding stubs — title, severity, exact `path:line` location, and a one-line claim under 20 words. At most 3 findings per file. Speculative findings are explicitly rejected.

**Expansion (3b):** Each finding stub is expanded with proof details — finding type, preconditions, a propagation-path proof, and a minimal fix outline. Expansion runs in parallel (up to 2 workers per file).

The two are merged into canonical `FindingRecord` objects. JSON parse failures trigger an automatic LLM repair pass; if repair also fails, the entire file gets one analytical retry.

### Phase 4: Verification

Each proposed finding is treated as a hypothesis. A verifier agent runs in an isolated Git worktree at HEAD with full access to the committed source code.

The verifier can inspect code and optionally compile or run small proof-of-concept programs. It must end with one of two verdicts:

- **REPRODUCED** — the finding is real and the verifier demonstrated it
- **NOTREPRODUCED** — the code does not support a practical trigger path

Verified findings advance to artifact generation. Discarded findings are dropped. Failed verifications (infrastructure errors, timeouts) are retried once for transient errors and can be resumed with `--resume`.

Verification runs in parallel, capped at 2 concurrent workers regardless of the `--workers` setting.

### Phase 5: Artifact Generation

For each verified finding:

1. A patch agent runs in an isolated worktree and applies the minimal correct fix using `edit_file`. The resulting `git diff` is captured.
2. The LLM writes a structured markdown report.

Both are saved to the `audit-findings/` directory:

```text
audit-findings/
  001-command-injection-in-handler.md
  001-command-injection-in-handler.patch
  002-missing-null-check-in-parser.md
  002-missing-null-check-in-parser.patch
```

## Report Format

Each `.md` report follows a fixed structure:

```text
# <finding title>
## Classification
## Affected Locations
## Summary
## Provenance
## Preconditions
## Proof
## Why This Is A Real Bug
## Fix Requirement
## Patch Rationale
## Residual Risk
## Patch
```

The `## Patch` section includes the full unified diff inline. Patches can also be applied directly:

```sh
git apply audit-findings/001-command-injection-in-handler.patch
```

## Options

`--resume` resumes a previous audit run from its last checkpoint. The resume matches against the current commit and scope (focus argument). If the commit or scope changed since the original run, no match is found and the command returns an error. On resume, completed phases are skipped, and failed verifications are requeued.

```text
swival> /audit --resume
```

`--workers N` sets the number of parallel workers for triage and verification (default: 4). Verification is always capped at 2 regardless of this value.

```text
swival> /audit --workers 8
```

Both can be combined with a focus path:

```text
swival> /audit src/api/ --resume --workers 6
```

## Scope

The audit examines only committed Git-tracked files at HEAD. Unstaged or uncommitted changes are invisible to the audit.

Only files with recognized source or configuration extensions are auditable:

**Source:** `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.kt`, `.rb`, `.php`, `.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`, `.scala`, `.sh`

**Configuration:** `.json`, `.toml`, `.yaml`, `.yml`, `.xml`, `.ini`, `.conf`, `.sql`, `.graphql`, `.proto`, `.rego`, `.tf`, `.cue`

Other file types (`.md`, `.png`, `.csv`, etc.) are excluded.

When a focus argument is provided, it works as both an fnmatch pattern and a prefix filter. For example, `/audit src/` includes all files under `src/`, and `/audit *.py` includes all Python files.

## State and Storage

Audit state is persisted in `.swival/audit/<run_id>/state.json`. This includes:

- Scope (branch, commit, file list, focus)
- All triage records
- Proposed and verified findings
- Verification status for each finding (pending, running, verified, discarded, failed)
- Metrics (parse failures, repair successes, analytical retries)
- Current phase and next artifact index

LLM interactions are traced to `.swival/audit/<run_id>/traces/` when `--trace-dir` is set on the outer session.

Temporary worktrees for verification and patch generation are created under `.swival/audit/<run_id>/verify/` and `.swival/audit/<run_id>/patch-gen/`, and cleaned up automatically.

Final artifacts go to `audit-findings/` in the project root.

## Interruption and Recovery

The audit is designed to be interrupted and resumed. `Ctrl+C` during any phase stops the audit gracefully. State is always saved before the interrupt is handled, so `/audit --resume` picks up where it left off.

If verification produces partial results (some findings verified, some failed), the audit reports the incomplete state and asks you to resume:

```text
Audit incomplete: 2 findings not verified (1 failed). Use /audit --resume to continue.
```

A completed audit (phase `"done"`) is not resumable.

## Limitations

The audit depends heavily on the quality of the underlying LLM. Models with weak code understanding will produce lower-quality triage and more false negatives. The verification phase catches many false positives, but a weak verifier model may also miss real bugs or incorrectly confirm speculative findings.

The audit sees only committed code. Runtime configuration, environment variables, deployment topology, and dynamic code paths that depend on external state are outside its view.

Large repositories with many auditable files can take significant time and LLM tokens to process.
