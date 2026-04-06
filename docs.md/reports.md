# Reports

The `--report` flag writes a structured JSON file that captures what happened during a run, including outcome, timing, tool usage, context-management events, and a full chronological timeline. This is designed for benchmarking and evaluation workflows where you want reproducible telemetry across model or prompt variants.

```sh
swival "Refactor the error handling in src/api.py" --report run1.json
```

When `--report` is enabled, Swival still prints the final answer to standard output when an answer exists, and also writes the same answer into the report JSON under `result.answer`. The flag is incompatible with `--repl`.

## Real Example

The JSON below is from a verified local run using `--model dummy-model --max-turns 0 --report run.json`, which produces an exhausted run with no LLM calls.

```json
{
  "version": 1,
  "timestamp": "2026-02-25T22:37:31.546022+00:00",
  "task": "No-op example",
  "model": "dummy-model",
  "provider": "lmstudio",
  "settings": {
    "temperature": null,
    "top_p": 1.0,
    "seed": null,
    "max_turns": 0,
    "max_output_tokens": 32768,
    "context_length": null,
    "files": "some",
    "commands": "none",
    "max_review_rounds": 15,
    "skills_discovered": [],
    "instructions_loaded": []
  },
  "sandbox": {
    "mode": "builtin"
  },
  "result": {
    "outcome": "exhausted",
    "answer": null,
    "exit_code": 2
  },
  "stats": {
    "turns": 0,
    "tool_calls_total": 0,
    "tool_calls_succeeded": 0,
    "tool_calls_failed": 0,
    "tool_calls_by_name": {},
    "compactions": 0,
    "turn_drops": 0,
    "guardrail_interventions": 0,
    "truncated_responses": 0,
    "llm_calls": 0,
    "total_llm_time_s": 0.0,
    "total_tool_time_s": 0.0,
    "skills_used": [],
    "review_rounds": 0
  },
  "timeline": []
}
```

## Report Structure

### Top-Level Fields

`version` is the schema version and is currently `1`. `timestamp` is the run completion time in UTC ISO 8601 format. `task` is the original question string passed on the command line. `model` is the resolved model identifier that was actually used. `provider` is one of `lmstudio`, `llamacpp`, `huggingface`, `openrouter`, `chatgpt`, `google`, `bedrock`, `generic`, or `command`.

`settings` captures run configuration. `sandbox` captures the sandbox backend in use. `result` captures outcome and exit semantics. `stats` captures aggregate counters. `timeline` captures ordered event records.

### `settings`

`temperature` stores the sampling temperature or `null` when omitted. `top_p` stores nucleus sampling. `seed` stores the random seed or `null`. `max_turns` and `max_output_tokens` store turn and output-token limits. `context_length` stores effective context length after provider resolution. `files` records the filesystem access policy: `"some"` (workspace only, the default), `"all"` (unrestricted), or `"none"` (`.swival/` only).

`commands` records the configured command policy: `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive approval per bucket), or a sorted list of whitelisted basenames. `max_review_rounds` records the reviewer retry limit. `skills_discovered` records skill names discovered at startup. `instructions_loaded` records loaded instruction files as absolute paths (e.g. the user-level `AGENTS.md` from `~/.config/swival/`, the cross-agent `~/.agents/AGENTS.md`, and the project-level files).

### `sandbox`

`mode` is always present and is `builtin` (application-layer path guards) or `agentfs` (OS-enforced write isolation). `session` appears when an AgentFS session ID is active. When `mode` is `agentfs`, `strict_read` is always present (whether strict read isolation is enabled). Additional fields may also appear: `agentfs_version` (the AgentFS binary version) and `diff_hint` (a hint for reviewing changes).

### `result`

`outcome` is `success`, `exhausted`, or `error`. `answer` contains final assistant text or `null` when unavailable. `exit_code` is the process exit code, which is `0` for success, `2` for turn exhaustion, and `1` for runtime failure. `error_message` appears only when `outcome` is `error`.

A `success` outcome means the model produced a final non-tool response. An `exhausted` outcome means the run reached `max_turns` before completing. An `error` outcome means runtime setup or execution failed.

### `stats`

`turns` is the highest completed turn number for the run. `llm_calls` is total logical LLM calls (each `call_llm()` invocation counts as one, regardless of internal provider retries), including retries after compaction. `total_llm_time_s` and `total_tool_time_s` are wall-clock totals in seconds.

`tool_calls_total`, `tool_calls_succeeded`, and `tool_calls_failed` are aggregate tool counters. `tool_calls_by_name` is a per-tool breakdown using `{succeeded, failed}` counts.

`compactions` counts `compact_messages` and `aggressive_drop` events. `turn_drops` counts `drop_middle_turns` events. `guardrail_interventions` counts injected correction prompts for repeated tool failures. `truncated_responses` counts model outputs that hit output-token limits.

`skills_used` records skill names successfully activated through `use_skill`. `review_rounds` records how many reviewer passes occurred when `--reviewer` is active. `todo` appears only when the `todo` tool was used and includes `added`, `completed`, and `remaining` counts. `snapshot` appears only when the `snapshot` tool was used and includes `saves`, `restores`, `cancels`, `blocked`, `force_restores`, and `tokens_saved` counts. `memory` appears when auto-memory was loaded and includes `total_entries`, `bootstrap_entries`, `retrievable_entries`, `bootstrap_tokens`, `retrieval_tokens`, `retrieved_ids`, and `mode` (either `budgeted` or `full`).

`prompt_cache` appears when at least one LLM call in the run returned cache stats. It is an object with `cached_tokens` (tokens served from the provider's prompt cache across the whole run) and `cache_write_tokens` (tokens written to the cache, i.e. the first-call population cost). Both fields are integers. Absent means no cache activity was reported by the provider.

`security` appears when at least one security-relevant event occurred during the run. It is an object with `command_policy_blocks` (commands denied by policy or by user), `command_policy_approvals` (commands approved by user or config), and `untrusted_inputs` (external content ingested from `fetch_url`, MCP, or A2A). All fields are integers. Absent when all counters are zero.

### `timeline`

`timeline` is an ordered array of event objects. Each event includes `type`, and most include `turn` (the turn number when the event occurred). Review events are an exception — they include `round` instead of `turn` since they occur between agent loop iterations.

For `llm_call`, fields include `duration_s`, `prompt_tokens_est`, `finish_reason`, `is_retry`, and optionally `provider_retries` (number of transient-error retries within this call; omitted when 0). Retry calls include `retry_reason`, which is one of `compact_messages`, `drop_middle_turns`, or `aggressive_drop`. When the provider returned prompt cache data, `cached_tokens` and `cache_write_tokens` are included (both integers; omitted when zero).

For `tool_call`, fields include `name`, `arguments`, `succeeded`, `duration_s`, and `result_length`. If arguments were invalid JSON, `arguments` is `null`. Failed tool calls include `error`.

For `compaction`, fields include `strategy`, `tokens_before`, and `tokens_after`. Strategy is one of `compact_messages`, `drop_middle_turns`, or `aggressive_drop`.

For `guardrail`, fields include `tool` and `level`, where `level` is `nudge` for repeated failures and `stop` for stronger intervention.

For `review`, fields include `round`, `exit_code`, and `feedback` (reviewer standard output). When the reviewer produces standard error output, `stderr` is also included.

For `truncated_response`, the event marks that an LLM response ended because of output token limits.

For `lifecycle`, fields include `event` (`startup` or `exit`), `exit_code`, `duration_s`, and optionally `error`. Lifecycle events appear when `--lifecycle-command` is configured. See [Lifecycle Hooks](lifecycle-hooks.md) for details.

For `command_policy`, fields include `bucket` (the normalized command bucket) and `decision` (`allow`, `persist`, `once`, `always_ask`, `deny`, or `block`). These events are emitted when `--commands ask` is active.

For `untrusted_input`, fields include `source` (the tool name, e.g. `fetch_url` or `mcp__server__tool`) and `origin` (the URL or empty string). These events are emitted when external content is successfully ingested.

## Benchmarking Workflow

A standard pattern is to run the same task set against multiple models or settings and then compare their report files. Passing `--seed` can reduce run-to-run variance for providers that support seeded sampling.

```sh
swival "task" --seed 42 --report run1.json
```

You can compare model variants like this:

```sh
for model in qwen3-coder-next deepseek-coder-v2; do
    swival "Fix the failing tests in tests/" \
        --model "$model" \
        --report "results/${model}.json"
done
```

You can compare sampling settings like this:

```sh
for temp in 0.2 0.55 0.8; do
    swival "Refactor src/api.py" \
        --temperature "$temp" \
        --report "results/temp-${temp}.json"
done
```

You can evaluate instruction variants like this:

```sh
for variant in minimal detailed strict; do
    cp "agent-variants/${variant}.md" project/AGENTS.md
    swival "Add input validation to the CLI" \
        --base-dir project \
        --report "results/agent-${variant}.json"
done
```

## Reading Reports With `jq`

Reports are plain JSON files, so `jq` works well for ad hoc analysis.

```sh
jq '{outcome: .result.outcome, turns: .stats.turns}' run1.json
jq '{llm: .stats.total_llm_time_s, tools: .stats.total_tool_time_s}' run1.json
jq '.stats.tool_calls_by_name' run1.json
jq '[.timeline[] | select(.type == "tool_call" and .succeeded == false)]' run1.json
jq '.stats.skills_used' run1.json
jq '{compactions: .stats.compactions, turn_drops: .stats.turn_drops}' run1.json
```

## Comparing Two Runs

You can produce quick side-by-side checks with shell tools.

```sh
paste <(jq -r '.result.outcome' a.json) <(jq -r '.result.outcome' b.json)

diff <(jq '{turns: .stats.turns, tools: .stats.tool_calls_total}' a.json) \
     <(jq '{turns: .stats.turns, tools: .stats.tool_calls_total}' b.json)
```

## What Reports Do Not Prove

The report captures behavior, not semantic correctness. It tells you whether the run completed cleanly, how the model spent time, which tools it used, and how context recovery behaved.

It does not prove that generated code compiles, passes tests, or satisfies business requirements. Those checks still belong in your evaluator, CI pipeline, or reviewer script.
