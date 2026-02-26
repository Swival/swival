# Reviews

The `--reviewer` flag lets you attach an external executable to the agent loop. After Swival produces an answer, it runs that executable, passes the answer on standard input, and decides whether to accept or retry based on the reviewer's exit code.

```sh
swival "Refactor the error handling in src/api.py" --reviewer ./review.sh
```

This pattern works well for automated gates such as tests, linting, format checks, schema checks, or LLM-as-a-judge scoring. `--reviewer` is incompatible with `--repl`.

## Reviewer Protocol

Swival invokes the reviewer as `reviewer_executable <base_dir>`. The first positional argument is the absolute base directory. The full assistant answer is written to reviewer standard input.

If the reviewer exits with code `0`, Swival accepts the answer immediately and ends normally. If the reviewer exits with code `1`, Swival treats reviewer standard output as feedback, appends that feedback as a new user message, resets turn budget for a new pass, and continues the loop. If the reviewer exits with code `2`, Swival treats that as reviewer failure, warns on standard error when diagnostics are enabled, and accepts the current answer unchanged. Any other nonzero exit code is handled the same way as `2`. Reviewer standard output is captured for all exit codes and recorded in the report timeline when `--report` is active.

Reviewer execution has a 60-minute timeout. Timeout or spawn failures are treated as reviewer errors and do not discard the agent's answer.

## Reviewer Environment Variables

Swival sets context variables on the reviewer subprocess for each round. `SWIVAL_TASK` contains the original user task and is always set. `SWIVAL_REVIEW_ROUND` contains the current review round number and is always set. `SWIVAL_MODEL` contains the resolved model identifier when available.

The reviewer inherits the parent environment too, but Swival's injected values override any same-named parent values.

## Writing A Reviewer Script

A minimal reviewer that accepts only when tests pass can look like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

base_dir="$1"
cd "$base_dir"

if python3 -m pytest tests/ -q 2>&1; then
    exit 0
else
    echo "Tests are failing. Fix the test failures and try again."
    exit 1
fi
```

A reviewer that requires valid JSON output can look like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

answer=$(cat)

if echo "$answer" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
    exit 0
else
    echo "Your answer is not valid JSON. Please output only valid JSON."
    exit 1
fi
```

The reviewer file must exist and be executable. Swival validates this before the run starts.

## Using Swival As The Reviewer

You can run a second Swival instance as an LLM judge. The outer Swival does the work and the inner Swival evaluates quality against the original task and current answer.

```bash
#!/usr/bin/env bash
# judge.sh -- use a second Swival instance to review the agent's answer
set -uo pipefail

base_dir="$1"
answer=$(cat)

judge_stderr=$(mktemp)
trap 'rm -f "$judge_stderr"' EXIT

judge_output=$(swival "You are reviewing a coding agent's output.

<task>$SWIVAL_TASK</task>

<answer>$answer</answer>

Evaluate whether the answer correctly and completely addresses the task.
Respond with exactly one of:
  VERDICT: ACCEPT
  VERDICT: RETRY followed by your feedback on the next line." \
    --base-dir "$base_dir" --quiet --no-history 2>"$judge_stderr")
judge_exit=$?

if [ $judge_exit -ne 0 ] || [ -z "$judge_output" ]; then
    echo "reviewer error: inner swival exited $judge_exit with no output"
    [ -s "$judge_stderr" ] && echo "stderr: $(cat "$judge_stderr")"
    exit 2
fi

if echo "$judge_output" | grep -qi "VERDICT: ACCEPT"; then
    echo "$judge_output"
    exit 0
elif echo "$judge_output" | grep -qi "VERDICT: RETRY"; then
    echo "$judge_output"
    exit 1
else
    echo "reviewer error: no VERDICT found in judge output"
    echo "$judge_output"
    exit 2
fi
```

This wrapper keeps reviewer behavior predictable even when the judge fails to return parseable output. In that case, returning exit code `2` tells the outer run to accept the current answer rather than failing hard. All three error paths output a diagnostic reason so it gets captured in the report's `feedback` field for debugging.

If both instances point at the same provider, they will use the same backend by default. You can direct the inner judge to a different provider or model by adding `--provider`, `--model`, and optionally `--base-url` inside the wrapper.

Using `set -uo pipefail` instead of `set -euo pipefail` avoids aborting the wrapper early when the inner run exits with code `2` but still returns parseable output.

## Retry And Round Limits

Every time the reviewer returns exit code `1`, Swival appends reviewer feedback as a user message and re-enters the loop with a fresh turn budget. The full conversation stays intact, so the model can build on prior work instead of restarting from scratch.

To prevent infinite cycles, Swival enforces a hard limit of five review rounds. If round five still returns code `1`, Swival accepts the latest answer and emits a warning when diagnostics are enabled.

## Failure Handling

Startup validation fails fast if the reviewer executable is missing or non-executable.

After startup, reviewer failures are non-fatal. Timeout failures, process spawn failures, and crash-style exits all degrade to reviewer error handling, which means the current answer is accepted and returned.

## Interaction With `--quiet` And `--report`

With `--quiet`, reviewer diagnostics are suppressed along with other diagnostic logging. Rejected intermediate answers are not printed to standard output; only the final accepted answer is printed.

With `--report`, each reviewer invocation is recorded as a `review` event in the timeline with the round number, exit code, and full reviewer output. `stats.review_rounds` records the total number of reviewer invocations. Turn numbers remain cumulative across rounds, so the timeline reads as one continuous run.

```sh
jq '.stats.review_rounds' report.json
jq '.timeline[] | select(.type == "review")' report.json
```

## Example CI Flow

A simple CI-style invocation can combine command access, reviewer retries, and report capture in one run.

```sh
swival "Fix the failing tests in tests/unit/" \
    --allowed-commands python3,pytest \
    --reviewer ./ci-review.sh \
    --report results.json \
    --quiet
```

In that setup, `ci-review.sh` should return `0` when checks pass and return `1` with actionable feedback when checks fail. Swival will retry with that feedback up to five rounds.
