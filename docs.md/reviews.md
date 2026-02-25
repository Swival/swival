# Reviews

The `--reviewer` flag hooks an external program into the agent loop. After the
agent produces an answer, Swival pipes it to your reviewer script, and the
script decides whether the answer is good enough or needs another pass.

```sh
swival "Refactor the error handling in src/api.py" --reviewer ./review.sh
```

This is useful for automated QA gates (run tests, check linting, validate
output format), for LLM-as-a-judge evaluation (have a second model score the
answer), and for running benchmarks where you need a programmatic pass/fail
signal. The reviewer is just an executable. It can do anything from a simple
`pytest` run to a full evaluation pipeline.

`--reviewer` is incompatible with `--repl`.

## The protocol

The reviewer executable receives:

- **Argument 1:** the base directory (absolute path).
- **stdin:** the agent's full text answer.

It communicates back through its exit code and stdout:

| Exit code | Meaning     | What Swival does                                                                                 |
| --------- | ----------- | ------------------------------------------------------------------------------------------------ |
| 0         | Accept      | Print the answer and exit normally                                                               |
| 1         | Retry       | Feed the reviewer's stdout back as a new prompt, reset the turn counter, re-enter the agent loop |
| 2         | Error       | Warn on stderr, accept the answer as-is                                                          |
| Other     | (same as 2) | Treated as a reviewer error                                                                      |

On exit code 1, the reviewer writes its feedback to stdout. Swival appends that
as a new user message -- the model sees the full conversation history plus the
review feedback, and gets a fresh turn budget to address it.

On exit code 0, stdout is ignored. On exit code 2 (or any other code), stdout
is ignored and Swival accepts the answer without modification.

## Environment variables

Swival sets environment variables on the reviewer subprocess so the script has
context beyond just the answer text:

| Variable              | Value                                      | Always set?    |
| --------------------- | ------------------------------------------ | -------------- |
| `SWIVAL_TASK`         | The original question passed to Swival     | Yes            |
| `SWIVAL_REVIEW_ROUND` | Current round number, 1-indexed            | Yes            |
| `SWIVAL_MODEL`        | Resolved model ID (e.g. `qwen3-coder-30b`) | When available |

The reviewer also inherits the parent process's full environment. If any of
these variables already exist in the parent environment, Swival's values take
precedence.

`SWIVAL_TASK` is the most useful one -- it tells an LLM-as-judge reviewer what
the agent was supposed to do, so it can evaluate the answer in context.
`SWIVAL_REVIEW_ROUND` lets a reviewer adjust its strictness or bail out based on
how many rounds have passed.

## Writing a reviewer script

A minimal reviewer that checks whether tests pass:

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

A reviewer that validates JSON output:

```bash
#!/usr/bin/env bash
set -euo pipefail

# The agent's answer arrives on stdin
answer=$(cat)

if echo "$answer" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
    exit 0
else
    echo "Your answer is not valid JSON. Please output only valid JSON."
    exit 1
fi
```

An LLM-as-judge reviewer using a second Swival instance:

```bash
#!/usr/bin/env bash
set -uo pipefail

base_dir="$1"
answer=$(cat)

judge_output=$(swival "You are reviewing a coding agent's output.

<task>$SWIVAL_TASK</task>
<answer>$answer</answer>

Respond with VERDICT: ACCEPT if the answer is correct, or
VERDICT: RETRY followed by feedback if it needs work." \
    --base-dir "$base_dir" --max-turns 3 --quiet --no-history 2>/dev/null)

if echo "$judge_output" | grep -qi "VERDICT: ACCEPT"; then
    exit 0
elif echo "$judge_output" | grep -qi "VERDICT: RETRY"; then
    echo "$judge_output" | sed '1,/VERDICT: RETRY/d'
    exit 1
else
    exit 2
fi
```

The script must be executable (`chmod +x review.sh`). Swival validates this at
startup and exits with an error if the file doesn't exist or isn't executable.

## Retry behavior

When the reviewer returns exit code 1, Swival:

1. Appends the reviewer's stdout as a new `user` message to the conversation.
2. Resets the turn counter (the agent gets a fresh `--max-turns` budget).
3. Re-enters the agent loop with the full conversation history intact.

The model sees everything from prior rounds -- its own reasoning, tool calls,
previous answers, and the review feedback -- so it can build on its earlier work
rather than starting from scratch.

There is a hard cap of 5 review rounds. If the reviewer keeps returning 1 after
5 rounds, Swival accepts the last answer with a warning. This prevents infinite
loops from a misconfigured reviewer.

## Failure handling

Swival validates the reviewer executable at startup. If it doesn't exist or
isn't executable, the run fails immediately with an error.

After startup, all reviewer failures are non-fatal:

- **Timeout** (120 seconds): warns on stderr, accepts the answer.
- **Spawn failure** (binary deleted, permissions changed): warns on stderr, accepts the answer.
- **Crash** (non-zero exit other than 1): warns on stderr, accepts the answer.

The agent's work is never lost due to a broken reviewer. If the reviewer can't
run, you still get the answer.

## Interaction with other flags

### `--quiet`

Reviewer diagnostics (round numbers, acceptance messages, warnings) go to
stderr and are gated on the verbose flag. `--quiet` suppresses all of them.
Intermediate answers rejected by the reviewer are never printed to stdout --
only the final accepted answer is.

### `--report`

When both `--reviewer` and `--report` are active, the JSON report captures the
full timeline across all review rounds. Turn numbers are cumulative (they don't
restart at 1 for each round), so the timeline reads as one continuous sequence.

The `stats` object includes a `review_rounds` field counting how many times the
reviewer was invoked. This is 0 when `--reviewer` is not used.

```sh
# Check how many review rounds happened
jq '.stats.review_rounds' report.json

# See the full timeline across rounds
jq '.timeline[] | {turn, type}' report.json
```

See [Reports](reports.md) for the full report schema.

## Example workflow

A CI pipeline that runs the agent and requires tests to pass:

```sh
#!/usr/bin/env bash
swival "Fix the failing tests in tests/unit/" \
    --allowed-commands python3,pytest \
    --reviewer ./ci-review.sh \
    --report results.json \
    --quiet
```

Where `ci-review.sh` runs the test suite and returns `0` on green, or`1` with
failure details on red. The agent gets up to 5 chances to fix the tests,
and the full timeline is captured in the report for later analysis.
