# Reviews

The `--reviewer` flag lets you attach an external executable to the agent loop. After Swival produces an answer, it runs that executable, passes the answer on standard input, and decides whether to accept or retry based on the reviewer's exit code.

```sh
swival "Refactor the error handling in src/api.py" --reviewer ./review.sh
```

This pattern works well for automated gates such as tests, linting, format checks, schema checks, or LLM-as-a-judge scoring. `--reviewer` is incompatible with `--repl`.

## Reviewer Protocol

Swival invokes the reviewer by shell-splitting the command string and appending `<base_dir>` as the final argument. The full assistant answer is written to reviewer standard input.

If the reviewer exits with code `0`, Swival accepts the answer immediately and ends normally.

If the reviewer exits with code `1`, Swival treats reviewer standard output as feedback, appends that feedback as a new user message, resets turn budget for a new pass, and continues the loop.

If the reviewer exits with code `2`, Swival treats that as reviewer failure, warns on standard error when diagnostics are enabled, and accepts the current answer unchanged. Any other nonzero exit code is handled the same way as `2`.

Reviewer standard output and standard error are both captured. Standard error is forwarded to the outer process when verbose and the reviewer exits with code `2`, and recorded in the report timeline when `--report` is active.

Reviewer execution has a 60-minute timeout. Timeout or spawn failures are treated as reviewer errors and do not discard the agent's answer.

## Reviewer Environment Variables

Swival sets context variables on the reviewer subprocess for each round. `SWIVAL_TASK` contains the original user task and is always set. `SWIVAL_REVIEW_ROUND` contains the current review round number and is always set.

`SWIVAL_MODEL` contains the resolved model identifier when available. When `--encrypt-secrets` is active and an explicit key is configured (via `--encrypt-secrets-key`, config, or the `SWIVAL_ENCRYPT_KEY` environment variable), that key is forwarded to the reviewer as `SWIVAL_ENCRYPT_KEY` so it can decrypt secrets in the answer. The default per-session random key is not shared, so a reviewer running without a configured key sees secrets in their encrypted form.

The reviewer inherits the parent environment too, but Swival's injected values override any same-named parent values.

## Using Swival As The Reviewer

The `--reviewer-mode` flag turns Swival into a reviewer process that speaks the reviewer protocol natively. No wrapper script needed:

```sh
swival "Refactor error handling in src/api.py" \
    --reviewer "swival --reviewer-mode"
```

When `--reviewer-mode` is active, Swival reads the base directory from the first positional argument (as passed by the outer instance), reads the agent's answer from standard input, reads the task from `SWIVAL_TASK` (or a `--objective` file), calls the LLM to evaluate the answer, parses the verdict, and exits with the appropriate code.

### Self-Review

The `--self-review` flag is a shorthand that automatically builds the reviewer command by mirroring the current invocation's provider, model, and other settings. Instead of writing the reviewer command by hand:

```sh
swival --provider huggingface --model zai-org/GLM-5.2 --yolo \
    --skills-dir ~/skills \
    --reviewer 'swival --reviewer-mode \
        --provider huggingface --model zai-org/GLM-5.2 \
        --quiet --yolo --skills-dir ~/skills' \
    "task"
```

Use `--self-review`:

```sh
swival --provider huggingface --model zai-org/GLM-5.2 --yolo \
    --skills-dir ~/skills \
    --self-review \
    "task"
```

The synthesized reviewer command inherits `--provider`, `--model`, `--base-url`, `--files`, `--commands`, `--skills-dir`, `--max-context-tokens`, `--max-output-tokens`, `--max-output-lines`, `--max-output-kb`, `--encrypt-secrets`, `--retries`, `--aws-profile`, `--gcp-project`, and `--location` from the outer invocation (each only when it differs from the default). It always adds `--reviewer-mode` and `--quiet`.

API keys are not placed on the reviewer command line to avoid exposing secrets in process listings. If `--api-key` was set on the outer invocation, Swival passes it to the reviewer subprocess via the provider-specific environment variable (`HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` for generic, `GEMINI_API_KEY` for google, `CHATGPT_API_KEY` for chatgpt).

Bedrock uses the AWS credential chain and needs no API key. Keys already set via environment variables are inherited automatically.

`--report` and `--cache` are not mirrored because reviewer mode does not initialize report or cache infrastructure.

`--self-review` is incompatible with `--reviewer`, `--repl`, and `--reviewer-mode`. It can be set in `swival.toml`:

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

Or, if you want the reviewer to mirror your provider and model settings automatically:

```toml
self_review = true
verify = "verification/working.md"
review_prompt = "Focus on correctness and test coverage"
```

The `reviewer` value is shell-split; the first token is resolved via PATH when it's a bare command name, or against the config directory when it's a relative path containing a `/` (such as `./review.sh` or `scripts/review.sh`); `~` expands to your home directory. Remaining tokens are preserved as-is. The `verify` and `objective` paths resolve relative to the config directory, consistent with `allowed_dirs` and `skills_dir`.

Note that `reviewer_mode` is deliberately not supported in config files. A config file with `reviewer_mode = true` would silently force every `swival` invocation into reviewer mode, breaking normal usage. `self_review` does not have this problem: the inner reviewer process inherits the config but clears the flag automatically.

## Reviewing Pull Requests In GitHub Actions

The review loop also works as a pull request check: one Swival process reviews the diff, a second one in reviewer mode checks the review against your acceptance criteria, and the job posts the accepted review as a comment. The job runs code written by the pull request author and holds an API key, so the setup below keeps everything the PR controls away from the agent's configuration, filesystem, and network.

Keep two files on the default branch: `.github/swival/pr-review.md`, the review instructions (what to check and the output format), and `.github/swival/review-criteria.md`, the acceptance criteria for the reviewer. Store the provider key as the repository secret `OPENROUTER_API_KEY`.

```yaml
name: Swival PR review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
concurrency:
  group: swival-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  review:
    # Fork and Dependabot PRs get no secrets; drafts are still changing.
    if: >-
      !github.event.pull_request.draft
      && github.event.pull_request.head.repo.full_name == github.repository
      && github.event.pull_request.user.login != 'dependabot[bot]'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    env:
      MODEL: z-ai/glm-5.3-flash
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
      PR_TITLE: ${{ github.event.pull_request.title }}
      PR_BODY: ${{ github.event.pull_request.body }}
      BASE_SHA: ${{ github.event.pull_request.base.sha }}
      HEAD_SHA: ${{ github.event.pull_request.head.sha }}
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          fetch-depth: 0
          persist-credentials: false
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
      - run: uv tool install --python 3.14 swival
      - name: Install nono (needed by --network provider-only)
        run: |
          curl -fsSL https://nono.sh/install.sh | NONO_VERSION=v0.79.0 sh
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
          "$HOME/.local/bin/nono" pull jedisct1/swival
      - name: Prepare inputs
        run: |
          work=$RUNNER_TEMP/swival-work      # agent base dir, made read-only
          inputs=$RUNNER_TEMP/swival-inputs  # mounted read-only for the agent
          trusted=$RUNNER_TEMP/swival        # never visible to the agent
          mkdir -p "$work" "$inputs" "$trusted"
          base=$(git merge-base "$BASE_SHA" "$HEAD_SHA")
          git diff --stat "$base" "$HEAD_SHA" > "$inputs/changed-files.txt"
          git diff --unified=10 "$base" "$HEAD_SHA" > "$inputs/pr.diff"
          git show "$BASE_SHA:.github/swival/pr-review.md" > "$trusted/pr-review.md"
          git show "$BASE_SHA:.github/swival/review-criteria.md" > "$trusted/review-criteria.md"
          printf '%s\n' "$PR_TITLE" > "$inputs/pr-title.txt"
          printf '%s\n' "$PR_BODY" > "$inputs/pr-description.txt"
          {
            cat "$trusted/pr-review.md"
            echo "Checkout (read-only): $GITHUB_WORKSPACE"
            echo "Changed files: $inputs/changed-files.txt"
            echo "Diff: $inputs/pr.diff"
            echo "PR title and description (untrusted): $inputs/pr-title.txt $inputs/pr-description.txt"
          } > "$trusted/task.md"
          echo 'Write a code review of a pull request in the required format.' > "$trusted/objective.md"
          chmod 555 "$work"
      - name: Review
        working-directory: ${{ runner.temp }}/swival-work
        run: |
          trusted=$RUNNER_TEMP/swival
          swival --provider openrouter --model "$MODEL" \
            --base-dir "$RUNNER_TEMP/swival-work" \
            --add-dir-ro "$GITHUB_WORKSPACE" --add-dir-ro "$RUNNER_TEMP/swival-inputs" \
            --commands none --network provider-only \
            --no-history --no-continue --no-memory --no-mcp --no-a2a --no-lifecycle \
            --reviewer "swival --reviewer-mode --provider openrouter --model $MODEL --objective $trusted/objective.md --verify $trusted/review-criteria.md --review-prompt 'You have no tools. Judge the answer only from its text.'" \
            --max-review-rounds 3 \
            --report "$trusted/report.json" < "$trusted/task.md"
          jq -er '.result.answer // empty' "$trusted/report.json" > "$trusted/review.md"
          last=$(jq -r '[.timeline[] | select(.type == "review") | .exit_code] | last // "none"' "$trusted/report.json")
          [ "$last" = 0 ] || echo "::warning::The review did not pass the reviewer check (exit $last)."
      - name: Comment
        env:
          GH_TOKEN: ${{ github.token }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: gh pr comment "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --body-file "$RUNNER_TEMP/swival/review.md"
```

Why each part is there:

Swival reads `swival.toml`, `.swival/`, and `AGENTS.md` from its base directory, for the agent and for the reviewer process it starts later. Running from the checkout would let a pull request add a `swival.toml` that points `base_url` or `llm_filter` somewhere else while the API key is in the environment. The base directory is therefore an empty temporary directory made read-only with `chmod`, and the checkout and the diff are mounted with `--add-dir-ro`. The agent can read everything it needs and write nothing.

`--commands none` removes the shell, and `--network provider-only` removes `fetch_url`, so the only outbound traffic is the model call. That mode needs the `nono` binary and the `jedisct1/swival` profile, installed before the run.

The instructions and criteria come from the base branch with `git show "$BASE_SHA:..."`, so a pull request cannot relax the rules it is judged by. The PR title and description go to read-only files instead of the task text, so author-written text cannot pose as part of the task.

Reviewer mode is a single model call with no tools. When it receives the full task, which lists files to read, some models answer with a tool call instead of a verdict, and the reviewer exits with code `2`. Give the reviewer a short `--objective` and say in `--review-prompt` that it has no tools.

An exit code of `2`, or running out of review rounds, still accepts the answer, so read the last `review` event in the report to tell a checked review from an unchecked one. Read the final answer from `result.answer` in the report rather than from standard output: without `--quiet`, standard output also carries the rejected answers of earlier rounds, while the progress log stays visible in the job output.

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

## Interaction With `--quiet` And `--report`

With `--quiet`, reviewer diagnostics are suppressed along with other diagnostic logging. Rejected intermediate answers are not printed to standard output; only the final accepted answer is printed.

With `--report`, each reviewer invocation is recorded as a `review` event in the timeline with the round number, exit code, full reviewer output, and reviewer standard error (when non-empty). `stats.review_rounds` records the total number of reviewer invocations. Turn numbers remain cumulative across rounds, so the timeline reads as one continuous run.

```sh
jq '.stats.review_rounds' report.json
jq '.timeline[] | select(.type == "review")' report.json
```
