# Reviews

The `--reviewer` flag lets you attach an external executable to the agent loop. After Swival produces an answer, it runs that executable, passes the answer on standard input, and decides whether to accept or retry based on the reviewer's exit code.

```sh
swival "Refactor the error handling in src/api.py" --reviewer ./review.sh
```

This pattern works well for automated gates such as tests, linting, format checks, schema checks, or LLM-as-a-judge scoring. `--reviewer` is incompatible with `--repl`.

## Reviewer Protocol

Swival invokes the reviewer by shell-splitting the command string and appending `<base_dir>` as the final argument. The full assistant answer is written to reviewer standard input.

If the reviewer exits with code `0`, Swival accepts the answer immediately and ends normally. If the reviewer exits with code `1`, Swival treats reviewer standard output as feedback, appends that feedback as a new user message, resets turn budget for a new pass, and continues the loop. If the reviewer exits with code `2`, Swival treats that as reviewer failure, warns on standard error when diagnostics are enabled, and accepts the current answer unchanged. Any other nonzero exit code is handled the same way as `2`. Reviewer standard output and standard error are both captured. Standard error is forwarded to the outer process when verbose, and recorded in the report timeline when `--report` is active.

Reviewer execution has a 60-minute timeout. Timeout or spawn failures are treated as reviewer errors and do not discard the agent's answer.

## Reviewer Environment Variables

Swival sets context variables on the reviewer subprocess for each round. `SWIVAL_TASK` contains the original user task and is always set. `SWIVAL_REVIEW_ROUND` contains the current review round number and is always set. `SWIVAL_MODEL` contains the resolved model identifier when available.

The reviewer inherits the parent environment too, but Swival's injected values override any same-named parent values.

## Using Swival As The Reviewer

The `--reviewer-mode` flag turns Swival into a reviewer process that speaks the reviewer protocol natively. No wrapper script needed:

```sh
swival "Refactor error handling in src/api.py" \
    --reviewer "swival --reviewer-mode"
```

When `--reviewer-mode` is active, Swival reads the base directory from the first positional argument (as passed by the outer instance), reads the agent's answer from standard input, reads the task from `SWIVAL_TASK` (or a `--objective` file), calls the LLM to evaluate the answer, parses the verdict, and exits with the appropriate code.

### Reviewer Mode Options

| Option                 | Description                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `--review-prompt TEXT` | Custom instructions appended to the built-in review prompt                              |
| `--objective FILE`     | Read the task description from a file instead of `SWIVAL_TASK` env var                  |
| `--verify FILE`        | Read verification/acceptance criteria from a file and include them in the review prompt |

All existing model and provider options (`--model`, `--provider`, `--base-url`, `--api-key`, `--temperature`) work normally in reviewer mode since they control the LLM call.

### Different Model For Review

```sh
swival "Fix the failing tests" \
    --model local/qwen-coder \
    --reviewer "swival --reviewer-mode --model qwen3-coder-next"
```

### Verification File

```sh
swival "Write a Python HTTP server on port 5000" \
    --reviewer "swival --reviewer-mode --verify verification/working.md"
```

The `--verify` file contains acceptance criteria that the reviewer checks the answer against.

### Custom Review Focus

```sh
swival "Add pagination to the API" \
    --reviewer "swival --reviewer-mode --review-prompt 'Verify that all endpoints return proper Link headers and that page_size defaults to 20'"
```

### CI With Report

```sh
swival "Fix the security vulnerability in auth.py" \
    --allowed-commands python3,pytest \
    --reviewer "swival --reviewer-mode --verify ci/security-criteria.md" \
    --report results.json \
    --quiet
```

### Project Config

These options can live in `swival.toml` so you don't repeat them on every invocation:

```toml
reviewer = "swival --reviewer-mode"
verify = "verification/working.md"
review_prompt = "Focus on correctness and test coverage"
```

The `reviewer` value is shell-split; the first token is resolved via PATH when it's a bare command name, or against the config directory when it starts with `./`, `../`, or `~`. Remaining tokens are preserved as-is. The `verify` and `objective` paths resolve relative to the config directory, consistent with `allowed_dirs` and `skills_dir`.

Note that `reviewer_mode` is deliberately not supported in config files. A config file with `reviewer_mode = true` would silently force every `swival` invocation into reviewer mode, breaking normal usage.

## Writing A Custom Reviewer Script

A minimal reviewer that accepts only when tests pass:

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

A reviewer that requires valid JSON output:

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

### Fully Custom LLM Reviewer

If you need complete control over the reviewer's prompt or LLM interaction, you can write a wrapper script instead of using `--reviewer-mode`:

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

## Retry And Round Limits

Every time the reviewer returns exit code `1`, Swival appends reviewer feedback as a user message and re-enters the loop with a fresh turn budget. The full conversation stays intact, so the model can build on prior work instead of restarting from scratch.

To prevent infinite cycles, Swival limits the number of review rounds. The default is five. You can change this with `--max-review-rounds` or `max_review_rounds` in config.

```sh
swival "task" --reviewer ./review.sh --max-review-rounds 10
```

Setting `--max-review-rounds 0` disables retries entirely: the reviewer still runs once, but a retry request is immediately capped and the answer is accepted.

If the limit is reached and the reviewer still returns code `1`, Swival accepts the latest answer and emits a warning when diagnostics are enabled.

## Failure Handling

Startup validation fails fast if the reviewer executable is missing or non-executable.

After startup, reviewer failures are non-fatal. Timeout failures, process spawn failures, and crash-style exits all degrade to reviewer error handling, which means the current answer is accepted and returned.

## Interaction With `--quiet` And `--report`

With `--quiet`, reviewer diagnostics are suppressed along with other diagnostic logging. Rejected intermediate answers are not printed to standard output; only the final accepted answer is printed.

With `--report`, each reviewer invocation is recorded as a `review` event in the timeline with the round number, exit code, full reviewer output, and reviewer standard error (when non-empty). `stats.review_rounds` records the total number of reviewer invocations. Turn numbers remain cumulative across rounds, so the timeline reads as one continuous run.

```sh
jq '.stats.review_rounds' report.json
jq '.timeline[] | select(.type == "review")' report.json
```
