# Reviews

The `--reviewer` flag lets you attach an external executable to the agent loop. After Swival produces an answer, it runs that executable, passes the answer on standard input, and decides whether to accept or retry based on the reviewer's exit code.

```sh
swival "Refactor the error handling in src/api.py" --reviewer ./review.sh
```

This pattern works well for automated gates such as tests, linting, format checks, schema checks, or LLM-as-a-judge scoring.
It works in the REPL too, see Reviews In The REPL below.

## Reviewer Protocol

Swival invokes the reviewer by shell-splitting the command string and appending `<base_dir>` as the final argument. The full assistant answer is written to reviewer standard input.

If the reviewer exits with code `0`, Swival accepts the answer immediately and ends normally.

If the reviewer exits with code `1`, Swival treats reviewer standard output as feedback, appends that feedback as a new user message, resets turn budget for a new pass, and continues the loop.

If the reviewer exits with code `2`, Swival treats that as reviewer failure, warns on standard error when diagnostics are enabled, and accepts the current answer unchanged. Any other nonzero exit code is handled the same way as `2`.

Reviewer standard output and standard error are both captured. Standard error is forwarded to the outer process when verbose and the reviewer exits with code `2`, and recorded in the report timeline when `--report` is active.

Reviewer execution has a 60-minute timeout. Timeout or spawn failures are treated as reviewer errors and do not discard the agent's answer.

## Reviewer Environment Variables

Swival sets context variables on the reviewer subprocess for each round. `SWIVAL_TASK` contains the original user task and is always set. `SWIVAL_REVIEW_ROUND` contains the current review round number and is always set.

`SWIVAL_CONVERSATION` holds the session's recent requests and answers as `<user>` and `<agent>` blocks, oldest first.
It's only set in the REPL, once there's an earlier exchange.
Treat the answers in it as the agent's claims, not as proof.

`SWIVAL_MODEL` contains the resolved model identifier when available. When `--encrypt-secrets` is active and an explicit key is configured (via `--encrypt-secrets-key`, config, or the `SWIVAL_ENCRYPT_KEY` environment variable), that key is forwarded to the reviewer as `SWIVAL_ENCRYPT_KEY` so it can decrypt secrets in the answer. The default per-session random key is not shared, so a reviewer running without a configured key sees secrets in their encrypted form.

The reviewer inherits the parent environment too, but Swival's injected values override any same-named parent values.

## Using Swival As The Reviewer

The `--reviewer-mode` flag turns Swival into a reviewer process that speaks the reviewer protocol natively. No wrapper script needed:

```sh
swival "Refactor error handling in src/api.py" \
    --reviewer "swival --reviewer-mode"
```

When `--reviewer-mode` is active, Swival reads the base directory from the first positional argument (as passed by the outer instance), reads the agent's answer from standard input, and reads the task from `SWIVAL_TASK` (or a `--objective` file). If `SWIVAL_CONVERSATION` is set, it's added to the review prompt as background. The reviewer can read, list, search, and outline files under the base directory before returning its verdict. It cannot edit files, run commands or tests, or call external tools. These restrictions apply even with `--yolo` or broader file and command permissions in config.

Reviewer mode ignores the history file. The reviewer tools act as if `.swival/HISTORY.md` does not exist. Past answers are claims from the agent, not evidence. The review also adds no entry to the history file, even if the config enables history.

Other agents and applications can call it directly:

```sh
SWIVAL_TASK="Fix the login bug" swival --reviewer-mode . < answer.txt
swival --reviewer-mode --objective task.md . < answer.txt
```

Feedback goes to stdout; diagnostics go to stderr. Exit status `0` accepts, `1` requests changes, and `2` reports a reviewer error. Relative `--objective` and `--verify` paths on the command line resolve against the base directory. `--max-turns` limits the review loop; an unfinished review returns `2` and does not write a continuation file.

### Self-Review

`--self-review` has the session's own model check each answer, with no reviewer command to write:

```sh
swival --provider huggingface --model zai-org/GLM-5.2 --yolo --self-review "task"
```

The review uses the provider, model, keys, and settings the session is using at that moment.
In the REPL, that includes switches made with `/profile`, `/model`, `/reasoning`, or model routing.
If you set up an `llm_filter`, review requests go through it too.

The reviewer can only read files in the base directory.
It can't edit files, run commands, use MCP or A2A tools, or read `.swival/HISTORY.md`, even with `--yolo`.
It also can't see directories added with `--add-dir` or `/add-dir`.

`--review-prompt` and `--verify` work with self-review.
`--objective` doesn't: the answer is always checked against what you actually asked.

Each review round gets about 60 minutes.
A request already sent to the provider is allowed to finish, retries included, so a slow provider can push a round past the hour.
If a review times out or fails, the answer is accepted as it is.

Review cost shows up in the session cost.
`--report` records each round, but its usage, cost, and exposure totals don't include the reviewer's own requests.

You can't combine `--self-review` with `--reviewer` or `--reviewer-mode`.
To turn it on for a project, add this to `swival.toml`:

```toml
self_review = true
```

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
    --commands python3,pytest \
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

Or, to have the session's own model do the review:

```toml
self_review = true
verify = "verification/working.md"
review_prompt = "Focus on correctness and test coverage"
```

The `reviewer` value is shell-split; the first token is resolved via PATH when it's a bare command name, or against the config directory when it's a relative path containing a `/` (such as `./review.sh` or `scripts/review.sh`); `~` expands to your home directory. Remaining tokens are preserved as-is. The `verify` and `objective` paths resolve relative to the config directory, consistent with `allowed_dirs` and `skills_dir`.

Note that `reviewer_mode` is deliberately not supported in config files. A config file with `reviewer_mode = true` would silently force every `swival` invocation into reviewer mode, breaking normal usage. `self_review = true` is fine in config: a `swival --reviewer-mode` process reading the same config just ignores it.

## Reviews In The REPL

`--reviewer` and `--self-review` work in the REPL too.
These get reviewed before you see the answer:

- plain requests, including the first question in `swival --repl "task"`
- custom `!commands`, checked against the prompt they expand to
- `/continue`, when it picks up one of those requests

Other commands, like `/goal`, `/loop`, `/btw`, `/init`, `/learn`, `/simplify`, and `/audit`, aren't reviewed.

If the agent hits its turn limit, whatever it has so far is shown without a review.
Finish it with `/continue`, and that answer gets reviewed.

Follow-ups like "use option 2" only make sense with what came before.
So the reviewer also sees your recent requests and the agent's answers, leaving out the oldest ones when there are too many.
It still checks the files instead of taking those answers at face value.
External reviewers get the same text in `SWIVAL_CONVERSATION`.
`/clear` and `/init` start it over.

Pressing Ctrl-C during a review stops it, and the answer is shown with a warning that it wasn't reviewed.
In one-shot mode, Ctrl-C during a review stops the whole run, just like during the agent's own work.

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

To prevent infinite cycles, Swival limits the number of review rounds. The default is fifteen. You can change this with `--max-review-rounds` or `max_review_rounds` in config.

```sh
swival "task" --reviewer ./review.sh --max-review-rounds 10
```

Setting `--max-review-rounds 0` disables retries entirely: the reviewer still runs once, but a retry request is immediately capped and the answer is accepted.

If the limit is reached and the reviewer still returns code `1`, Swival accepts the latest answer and emits a warning when diagnostics are enabled.

## Failure Handling

Startup validation fails fast if the reviewer executable is missing or non-executable.

After startup, reviewer failures are non-fatal. Timeout failures, process spawn failures, and crash-style exits all degrade to reviewer error handling, which means the current answer is accepted and returned.
With `--self-review`, the same goes for provider errors, an unreadable `--verify` file, and a reply without a verdict.

## Interaction With `--quiet` And `--report`

With `--quiet`, reviewer diagnostics are suppressed along with other diagnostic logging. Rejected intermediate answers are not printed to standard output; only the final accepted answer is printed.

With `--report`, each finished review round is recorded as a `review` event in the timeline.
It has the round number, exit code, full reviewer output, reviewer standard error (when there is any), and a short reason when a review failed.
For an external reviewer, that reason is just its exit code.
A review interrupted with Ctrl-C isn't recorded.

`stats.review_rounds` counts those rounds, over the whole session in the REPL.
Turn numbers keep counting across rounds, so the timeline reads as one continuous run.

```sh
jq '.stats.review_rounds' report.json
jq '.timeline[] | select(.type == "review")' report.json
```
