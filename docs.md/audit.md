# Swival Security Scanner

The `/audit` command runs a multi-phase security audit over committed Git-tracked code.

It triages files by attack surface, performs deep review on escalated files, verifies each finding with an isolated proof-of-concept agent, generates patches, and writes structured reports. Only provable bugs survive to the final output.

```text
/audit [path|glob ...] [--resume] [--regen] [--finding N[,M-R]] [--all] [--measure-triage] [--workers N] [--patch-max-turns N] [--debug]
```

Works in both interactive (REPL) and one-shot mode (requires `--oneshot-commands`). Runs against `HEAD`, so dirty working-directory changes are ignored.

## Quick Start

Start an audit from the REPL:

```text
swival> /audit
```

Scope it to a directory or glob:

```text
swival> /audit src/auth/
swival> /audit *.py
swival> /audit src/*.py
swival> /audit src/**/*.py
```

The matcher uses `pathlib.PurePosixPath.full_match` for any pattern that contains a wildcard. A bare `*` does not cross directory separators, so `src/*.py` matches only direct children of a top-level `src/` directory.

Use `**` when you want recursion: `src/**/*.py` matches every `.py` file at any depth under `src/`. As a convenience, a wildcard pattern with no `/` is treated as recursive on its own, so `*.py` still selects every Python file in the repository.

Multiple paths can be passed; they are unioned into a single audit run with one
state file and one set of reports:

```text
swival> /audit src/auth/ src/api/
```

When the audit finishes, findings are written to `audit-findings/` in the project root:

```text
swival> /audit
Audit complete. 2 finding(s) written to audit-findings/. Open audit-findings/README.md to review.
```

That `README.md` is the entry point for a reviewer: it carries the run metadata (commit, branch, scope), a one-row-per-finding table grouped by finding type (severity-ordered within each group), the run totals, and, when an audit only partially completed, a section listing each failed or pending artifact with its error code and the exact retry command. The per-finding `.md` and `.patch` files are still authoritative for the narrative and the fix; the README is what you open first.

If no bugs are found:

```text
No provable security bugs or security-control failures found in Git-tracked files.
```

## Example Audits

A growing collection of security audits run against open-source projects with Swival is published at [github.com/swival/security-audits](https://github.com/swival/security-audits). Each audit there was generated automatically by `/audit` and contains the full set of findings, reports, and patches.

## How It Works

The audit runs in a sequence of phases. State is checkpointed after each phase and after every batch within a phase, so interrupted audits can be resumed. Phases 1 through 3 narrow the codebase down to provable findings, Phase 4 reproduces them, the Phase 4.5 gate throws out the ones that are overstated or irrelevant, and Phase 5 writes the reports and patches.

### Phase 1: Repository Profiling

Reads manifests (`package.json`, `pyproject.toml`, `Cargo.toml`, `Makefile`, etc.) and entry-point candidates from committed code, then calls the LLM to produce a compact repository profile: detected languages, frameworks, entry points, trust boundaries, persistence layers, auth surfaces, and dangerous operations. This profile is reused as context in every subsequent phase.

Files are ordered by an attack-surface heuristic that scores keywords like `exec`, `eval`, `auth`, `token`, `sql`, `template`, and `socket`. Higher-scoring files are processed first.

### Phase 2: Triage

Each auditable file is triaged independently. The LLM sees the file contents, its attack-surface score, import/dependency context, and the repository profile. It returns one of three labels:

- **ESCALATE_HIGH**: concrete suspicious path or invariant break worth deep review
- **ESCALATE_MEDIUM**: plausible concern, lower confidence
- **SKIP**: no evidence for escalation

The triage prompt is intentionally precision-biased: it prefers SKIP under uncertainty. To recover false negatives, several deterministic signals override SKIP after the LLM verdict:

- A file with an attack-surface score of 8 or more is escalated regardless of the LLM verdict.
- A file listed by Phase 1 as an entry point or trust boundary is escalated.
- A file that an entry point references directly (one-hop dependency) and that has a non-zero attack-surface score is escalated.
- A triage record with `needs_followup: true` is escalated outright. Triage already produces this signal; we now act on it.
- A file whose triage call timed out, raised a network error, or produced an unparseable response is escalated. This is fail-open behavior: the model never gave a real verdict, so we err on the side of looking.
- Any file matched by a `[audit] force_review` glob in `swival.toml` (see Configuration, below).
- A second confirmation pass for any file the LLM marked SKIP with low confidence: the same file is re-triaged with richer evidence (its dependency list and the contents of its highest-scoring dependency). The confirmation pass typically affects 10 to 20 percent of triage targets.

Triage runs in parallel with configurable worker count. The end-of-phase output breaks down the escalated count by reason and prints the top five SKIPped files by attack-surface score, so a wrong call is catchable before Phase 3 begins.

### Phase 3: Deep Review

Each escalated file goes through a two-step deep review.

**Inventory (3a):** The LLM produces a compact list of finding stubs (title, severity, exact `path:line` location, and a one-line claim under 20 words). At most 3 findings per file. Speculative findings are explicitly rejected.

**Expansion (3b):** Each finding stub is expanded with proof details: finding type, preconditions, a propagation-path proof, and a minimal fix outline. A file's stubs are expanded one after another; files themselves are deep-reviewed in parallel across the worker pool.

Both steps see more than the file itself. Swival resolves the cross-file functions the file actually calls, preferring explicit imports and falling back to the dependency index built in Phase 1, and appends their exact committed bodies as a "Called function definitions" section, each labeled with its own path and line span. A validation helper that lives two files away is reviewed next to its call site, which is what lets a finding land on the helper that is actually broken rather than on the caller. The section is budgeted (8000 bytes per definition, 24000 bytes per prompt); callees that do not fit degrade to one-line pointers instead of vanishing. The same enrichment rides along on the evidence bundles used in verification, adjudication, and report generation.

The two are merged into canonical `FindingRecord` objects. JSON parse failures trigger an automatic LLM repair pass; if repair also fails, the entire file gets one analytical retry.

### Phase 4: Verification

Each proposed finding is treated as a hypothesis. A verifier agent runs in an isolated Git worktree at HEAD with full access to the committed source code.

The verifier can inspect code and optionally compile or run small proof-of-concept programs. It must end with one of two verdicts:

- **REPRODUCED**: the finding is real and the verifier demonstrated it
- **NOTREPRODUCED**: the code does not support a practical trigger path

Verified findings advance to artifact generation. Discarded findings are dropped. Failed verifications (infrastructure errors, timeouts) are retried once for transient errors and can be resumed with `--resume`.

Verification runs in parallel, capped at 2 concurrent workers regardless of the `--workers` setting.

### Phase 4.5: Adjudication

Reproduction proves a bug exists. It does not prove the bug matters. The verifier spends its turns trying to *confirm* a finding, which biases it toward saying yes, and severity is assigned early in Phase 3 from a single-file view that never gets revisited. The result is a pile of findings that are technically real but overstated, self-inflicted, or irrelevant to how the project is actually deployed.

The adjudication gate is the answer to that. It runs after verification and before any report or patch is written, so the expensive Phase 5 work only happens for findings that survive.

Each verified finding faces a panel of three independent reviewers, each told to *refute* it and to default to false positive when unsure. The three look through different lenses: one asks whether an untrusted actor can really reach the code, one asks whether it matters in the project's expected threat model, and one asks whether the severity is justified or the issue is already mitigated. A finding is kept only when a majority of the panel confirms it is both real and relevant. Ties drop, because the whole point is to stop shipping false positives.

Reviewers judge against the deployment surface drawn from the Phase 1 profile, so a denial of service that only the local user can inflict on their own CLI is recognized for what it is and dropped rather than filed as high severity.

Survivors then pass through a consolidation step that recalibrates severity to the realistic worst case (severity can only be lowered here, never raised), tightens the title and impact type, and records a short threat-model statement that the report carries forward. A `security_control_failure` stays at high or critical, since that is the floor for the carve-out.

Dropped findings are kept in the run state and listed in the README under "Dropped in adjudication" with the reason, so the gap between verified and written counts is always explained.

Adjudication is skipped under `--measure-triage`, where the goal is to measure triage recall rather than to ship a clean set.

### Phase 5: Artifact Generation

For each verified finding:

1. A patch agent runs in an isolated worktree and applies the minimal correct fix using `edit_file`. The resulting `git diff` is captured.
2. The LLM writes a structured markdown report.

Both are saved to the `audit-findings/` directory:

```text
audit-findings/
  README.md
  001-command-injection-in-handler.md
  001-command-injection-in-handler.patch
  002-missing-null-check-in-parser.md
  002-missing-null-check-in-parser.patch
```

Each verified finding is assigned a stable index when it is first reached, and that index sticks across retries: if patch generation runs out of turns, the next attempt writes `002-...` for the same finding rather than consuming a new number.

Once at least one finding's artifacts have landed, the loop rewrites `README.md` as the last step of the artifact phase. The README is regenerated on every `--resume` and `--regen` invocation, so it always reflects the current state of the run, including any newly failed retries. The per-finding files stay authoritative; the README is the reviewer's index into them.

Patch failures, report exceptions, and write errors are all persisted as retryable Phase 5 state, so an audit that finishes Phase 4 but stumbles in Phase 5 stays resumable. See [Options](#options) for `--patch-max-turns` and the targeted `--regen --finding` form.

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

Saved audit state from versions before Phase 5 artifact retry is not supported after this state-model change. Finish in-flight audits before upgrading, or re-run `/audit` from scratch.


`--resume` resumes a previous audit run from its last checkpoint. The resume matches against the current commit and scope (focus argument). If the commit or scope changed since the original run, no match is found and the command returns an error. On resume, completed phases are skipped, failed verifications are requeued, and failed Phase 5 artifact generation is retried.

```text
swival> /audit --resume
```

`--regen` regenerates reports and patches for a completed audit run. It reuses the verified findings from the original run and re-runs only phase 5 (artifact generation). This is useful when you want to improve patch quality without repeating the expensive triage, deep review, and verification phases.

Use `--finding` with 1-based Phase 5 finding numbers to regenerate only selected artifacts. `--finding` requires `--regen` and is rejected if you pass it on a fresh run.

```text
swival> /audit --regen
swival> /audit --regen --finding 2 --patch-max-turns 75
swival> /audit --regen --finding 2,4-6
```

`--all` skips the Phase 2 triage selection and sends every file in scope straight to deep review. Useful when you have already narrowed scope to a subtree you want exhaustively reviewed and do not want the triage step deciding which files are worth a closer look.

```text
swival> /audit --all swival/
```

The flag composes with focus paths and is best paired with one: bare `/audit --all` deep-reviews every auditable file in the repo, which on a non-tiny project is expensive.

It is recorded on the run when it starts and is *not* part of the resume-matching key. A bare `/audit --resume` will pick up an `--all` run, and passing `--all` on a resume invocation has no effect (the persisted value wins). When more than one matching run exists, `--resume` picks the most recently modified one, so a fresh `--all` run shadows an older non-`--all` run with the same scope.

Triage occasionally catches that a file is vendored or generated and skips it. With `--all`, those files reach Phase 3 anyway and burn LLM calls there; scope `--all` to directories you actually wrote.

`--measure-triage` is a calibration mode for the Phase 2 selector. It runs Phase 2 normally, snapshots which files were escalated, then deep-reviews every file in scope (the `--all` set).

Each verified finding is tagged with whether its source file was escalated or skipped by triage. The Phase 5 output ends with a recall section that counts findings on skipped files: those are the false negatives. Use this to quantify recall before or after tuning promotion thresholds.

The mode is expensive (it pays the full `--all` cost plus an extra Phase 2), so it is a calibration tool, not a default. A run started with `--measure-triage` cannot be resumed without it (and vice versa); start a fresh run instead.

```text
swival> /audit --measure-triage swival/
```

`--workers N` sets the number of parallel workers for triage and verification (default: 4). Verification is always capped at 2 regardless of this value.

```text
swival> /audit --workers 8
```

`--patch-max-turns N` sets the isolated Phase 5 patch-generation turn budget (default: 50). The CLI flag overrides `[audit].patch_max_turns` in `swival.toml`; project config overrides global config. Raising this value can rescue complex patches, but it also increases LLM spend for stubborn findings.

```text
swival> /audit --resume --patch-max-turns 75
```

```toml
[audit]
patch_max_turns = 50
```

`--debug` writes a real-time JSONL trace of every audit step to `.swival/audit/debug.jsonl`. Useful when investigating a stuck phase, a missing finding, or unexpected resume behavior.

```text
swival> /audit --debug
```

All options can be combined with a focus path:

```text
swival> /audit src/api/ --resume --workers 6
swival> /audit src/api/ --regen
```

## Configuration

`swival.toml` (or the global `~/.config/swival/config.toml`) accepts an `[audit]` section:

```toml
[audit]
force_review = ["swival/audit.py", "swival/edit.py", "swival/sandbox_*.py"]
```

`force_review` is a list of path globs evaluated against repo-relative paths from `git ls-files`, using the same matcher as `/audit` focus arguments (see "Filtering" below for the full rules). A trailing `/` on a non-wildcard entry expands to the directory and everything below it (`src/` matches `src/a.py`, `src/sub/b.py`, and so on); a single `*` does not cross `/`, so `src/*.py` matches only direct children, while `src/**/*.py` recurses.

Matching files are unconditionally promoted into Phase 3, regardless of what triage decides. It is the surgical alternative to `--all` for paths you always want deep-reviewed.

A glob in the project file that matches zero paths in scope produces a warning, since it usually means a stale entry after a rename. Globs in the global file are silent on zero matches, on the assumption that a global glob like `swival/audit.py` will trivially miss in unrelated repositories. Globs from both files are merged: project entries layer on top of global entries.

Adding a glob between runs takes effect on resume: if a saved run has a SKIP record for a path that now matches `force_review`, the resume promotes that record before Phase 3 sees it. Removing a glob is *not* honored on resume; rescinding mid-audit is more confusing than it is worth, so re-run from scratch instead.

## Scope

The audit examines only committed Git-tracked files at HEAD. Unstaged or uncommitted changes are invisible to the audit.

Only files with recognized source or configuration extensions are auditable:

**Source:** `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.kt`, `.rb`, `.php`, `.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`, `.scala`, `.sh`, `.zig`, `.d`, `.pl`, `.pm`, `.psgi`

**Configuration:** `.json`, `.toml`, `.yaml`, `.yml`, `.xml`, `.ini`, `.conf`, `.sql`, `.graphql`, `.proto`, `.rego`, `.tf`, `.cue`

Other file types (`.md`, `.png`, `.csv`, etc.) are excluded.

A focus argument is matched against each repo-relative path with three rules, evaluated in order:

1. Exact match. `src/foo.py` selects only `src/foo.py`.
2. Prefix match for entries with no wildcard. `src` and `src/` both expand to "anything under top-level `src/`".
3. Wildcard match via `pathlib.PurePosixPath.full_match`. A single `*` matches one path segment and does not cross `/`, `?` matches one non-separator character, `**` matches any number of intermediate directories, and `[abc]` is a character class.

A wildcard pattern with no `/` is treated as recursive, so `*.py` keeps doing the natural thing and selects every Python file in the repository. Anchored patterns are precise:

| Pattern       | Matches                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `*.rs`        | every `.rs` file at any depth (slashless wildcard, recursive shorthand) |
| `src/*.rs`    | only direct `.rs` children of a top-level `src/` directory              |
| `src/**/*.rs` | every `.rs` file at any depth under a top-level `src/` directory        |
| `src/`        | every file under a top-level `src/` directory                           |

Anchored patterns never match suffixes: `src/*.rs` does *not* select `crates/foo/src/bar.rs`, because the leading `src/` is rooted at the repository top.

Multiple patterns can be combined in one run, for example `/audit '*.rs' '*.toml'`. Quote the pattern when invoking from a shell that would expand it before swival sees it.

## State and Storage

Audit state is persisted in `.swival/audit/<run_id>/state.json`. This includes:

- Scope (branch, commit, file list, focus)
- All triage records, including the LLM verdict, promotion reasons, any infrastructure-failure tag, and the confirmation-pass outcome
- Proposed and verified findings
- Verification status for each finding (pending, running, verified, discarded, failed)
- Metrics (parse failures, repair successes, analytical retries)
- Per-file attack-surface scores cached from Phase 1
- Current phase and per-finding artifact state (status, stable index, filenames, attempts, last error code, last patch budget used)
- `select_all` flag (whether the run was started with `--all`) and `measure_triage` flag (whether the run was started with `--measure-triage`)

When the outer session sets `--trace-dir`, every audit LLM call is written as a trace into that directory.

Temporary worktrees for verification and patch generation are created under `.swival/audit/<run_id>/verify/` and `.swival/audit/<run_id>/patch-gen/`, and cleaned up automatically.

Final artifacts go to `audit-findings/` in the project root.

## Interruption and Recovery

The audit is designed to be interrupted and resumed. `Ctrl+C` during any phase stops the audit gracefully. State is always saved before the interrupt is handled, so `/audit --resume` picks up where it left off.

If verification produces partial results (some findings verified, some failed), the audit reports the incomplete state and asks you to resume:

```text
Audit incomplete: 2 findings not verified after 3 attempts (1 failed). Use /audit --resume to retry.
```

If Phase 5 patch or report generation fails for some verified findings, the run stays in the `"artifacts"` phase with per-finding status recorded:

```text
Audit incomplete: artifact generation has 1 failed and 0 pending out of 10 verified finding(s). Use /audit --resume --patch-max-turns 75 to retry incomplete artifacts, or /audit --regen --finding 1 --patch-max-turns 75 to retry a specific finding.
```

A completed audit (phase `"done"`) is not resumable with `--resume`, but can be used with `--regen` to regenerate artifacts.

## Limitations

The audit depends heavily on the quality of the underlying LLM. Models with weak code understanding will produce lower-quality triage and more false negatives. The verification and adjudication phases catch many false positives, but a weak model may also drop real bugs or wave through speculative ones. The adjudication panel is deliberately biased toward dropping, so a borderline-but-genuine finding can be discarded. Dropped findings are listed with their reason in the README, and a full re-run is the way to revisit them if you disagree with the panel.

The audit sees only committed code. Runtime configuration, environment variables, deployment topology, and dynamic code paths that depend on external state are outside its view.

Large repositories with many auditable files can take significant time and LLM tokens to process.
