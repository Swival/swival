# Reports

The `--report` flag produces a JSON file that captures everything that happened
during an agent run: the outcome, timing, tool usage, context management events,
and a full action-by-action timeline. It's designed for benchmarking and
evaluation -- comparing models, tuning settings, or measuring how well an
AGENT.md file guides the agent on a set of tasks.

```sh
swival "Refactor the error handling in src/api.py" --report run1.json
```

When `--report` is active, the final answer goes into the JSON file instead of
stdout. Diagnostic output on stderr is unaffected.

`--report` is incompatible with `--repl`.

## Example

Here's a real report from a simple task. The agent called `python3` to compute
a square root, then delivered the answer:

```json
{
  "version": 1,
  "timestamp": "2026-02-25T10:36:28.312457+00:00",
  "task": "Compute the square root of 23847234",
  "model": "qwen3-coder-next",
  "provider": "lmstudio",
  "settings": {
    "temperature": 0.55,
    "top_p": 1.0,
    "seed": null,
    "max_turns": 100,
    "max_output_tokens": 32768,
    "context_length": 100000,
    "yolo": false,
    "allowed_commands": [
      "python3",
      "ruby"
    ],
    "skills_discovered": [],
    "instructions_loaded": [
      "AGENT.md"
    ]
  },
  "result": {
    "outcome": "success",
    "answer": "The square root of 23,847,234 is approximately **4,883.36**.",
    "exit_code": 0
  },
  "stats": {
    "turns": 2,
    "tool_calls_total": 1,
    "tool_calls_succeeded": 1,
    "tool_calls_failed": 0,
    "tool_calls_by_name": {
      "run_command": {
        "succeeded": 1,
        "failed": 0
      }
    },
    "compactions": 0,
    "turn_drops": 0,
    "guardrail_interventions": 0,
    "truncated_responses": 0,
    "llm_calls": 2,
    "total_llm_time_s": 48.25,
    "total_tool_time_s": 0.082,
    "skills_used": []
  },
  "timeline": [
    {
      "turn": 1,
      "type": "llm_call",
      "duration_s": 43.831,
      "prompt_tokens_est": 4758,
      "finish_reason": "tool_calls",
      "is_retry": false
    },
    {
      "turn": 1,
      "type": "tool_call",
      "name": "run_command",
      "arguments": {
        "command": [
          "python3",
          "-c",
          "import math; print(math.sqrt(23847234))"
        ]
      },
      "succeeded": true,
      "duration_s": 0.082,
      "result_length": 18
    },
    {
      "turn": 2,
      "type": "llm_call",
      "duration_s": 4.419,
      "prompt_tokens_est": 4797,
      "finish_reason": "stop",
      "is_retry": false
    }
  ]
}
```

## Report structure

### Top-level fields

| Field       | Type   | Description                                    |
| ----------- | ------ | ---------------------------------------------- |
| `version`   | int    | Schema version (currently `1`)                 |
| `timestamp` | string | UTC ISO 8601 timestamp of when the run ended   |
| `task`      | string | The question/task passed on the command line   |
| `model`     | string | Model identifier (auto-discovered or from CLI) |
| `provider`  | string | `lmstudio` or `huggingface`                    |
| `settings`  | object | Configuration snapshot (see below)             |
| `result`    | object | Outcome and answer (see below)                 |
| `stats`     | object | Aggregate counters (see below)                 |
| `timeline`  | array  | Ordered list of every event (see below)        |

### settings

A snapshot of the configuration used for the run.

| Field                 | Type        | Description                                  |
| --------------------- | ----------- | -------------------------------------------- |
| `temperature`         | float       | Sampling temperature                         |
| `top_p`               | float       | Top-p (nucleus) sampling                     |
| `seed`                | int or null | Random seed, if set                          |
| `max_turns`           | int         | Maximum agent loop iterations                |
| `max_output_tokens`   | int         | Max tokens per LLM response                  |
| `context_length`      | int or null | Effective context window size                |
| `yolo`                | bool        | Whether YOLO mode was active                 |
| `allowed_commands`    | string[]    | Whitelisted command basenames (sorted)       |
| `skills_discovered`   | string[]    | Skill names found at startup (sorted)        |
| `instructions_loaded` | string[]    | Instruction files loaded (`CLAUDE.md`, etc.) |

### result

| Field           | Type        | Description                                   |
| --------------- | ----------- | --------------------------------------------- |
| `outcome`       | string      | `success`, `exhausted`, or `error`            |
| `answer`        | string/null | The agent's final text answer (null on error) |
| `exit_code`     | int         | Process exit code (0, 1, or 2)                |
| `error_message` | string      | Present only when `outcome` is `error`        |

Outcomes:

- **success** -- the agent produced a final answer.
- **exhausted** -- the agent hit `max_turns` without finishing. `exit_code` is 2.
- **error** -- a runtime failure (couldn't connect, context overflow after all
  recovery attempts, bad model config, etc.). `exit_code` is 1.

### stats

Aggregate counters for the entire run.

| Field                     | Type     | Description                                                             |
| ------------------------- | -------- | ----------------------------------------------------------------------- |
| `turns`                   | int      | Number of turns completed                                               |
| `llm_calls`               | int      | Total LLM API calls (including retries)                                 |
| `total_llm_time_s`        | float    | Wall-clock seconds spent in LLM calls                                   |
| `total_tool_time_s`       | float    | Wall-clock seconds spent executing tools                                |
| `tool_calls_total`        | int      | Total tool invocations                                                  |
| `tool_calls_succeeded`    | int      | Tool calls that returned a result                                       |
| `tool_calls_failed`       | int      | Tool calls that returned an error                                       |
| `tool_calls_by_name`      | object   | Per-tool breakdown: `{"read_file": {"succeeded": 5, "failed": 0}, ...}` |
| `compactions`             | int      | Context compactions (truncating old results)                            |
| `turn_drops`              | int      | Aggressive context recovery (dropping turns)                            |
| `guardrail_interventions` | int      | Times the guardrail injected corrective messages                        |
| `truncated_responses`     | int      | LLM responses cut short by output token limit                           |
| `skills_used`             | string[] | Skill names that were successfully activated during the run             |

### timeline

An ordered array of every event. Each entry has a `turn` number and a `type`.

**`llm_call`** -- one per LLM API invocation, including failed attempts and
retries after context overflow. The example above shows two: the first with
`finish_reason: "tool_calls"` (the model decided to call a tool) and the second
with `"stop"` (the model produced its final answer). When a call is a retry
after compaction, `is_retry` is `true` and `retry_reason` is present
(`"compact_messages"` or `"drop_middle_turns"`). Failed calls use
`finish_reason` values like `"context_overflow"` or `"error"`.

**`tool_call`** -- one per tool invocation. The example shows a `run_command`
call with the full argument list, timing, and result size. `arguments` is `null`
when the model produced invalid JSON. `error` is present when `succeeded` is
`false`.

**`compaction`** -- context recovery. Recorded when the context window fills up
and Swival has to trim history. `strategy` is `"compact_messages"` (truncating
old tool results) or `"drop_middle_turns"` (removing entire middle turns).

```json
{"turn": 15, "type": "compaction", "strategy": "compact_messages",
 "tokens_before": 128000, "tokens_after": 64000}
```

**`guardrail`** -- injected when the agent repeats the same failing tool call.
`level` is `"nudge"` (2 consecutive identical errors) or `"stop"` (3+).

```json
{"turn": 7, "type": "guardrail", "tool": "edit_file", "level": "nudge"}
```

**`truncated_response`** -- the LLM hit its output token limit mid-response.

```json
{"turn": 4, "type": "truncated_response"}
```

## Benchmarking workflow

A typical benchmarking setup runs the same set of tasks across different
configurations and compares the reports. For more deterministic results, pass
`--seed` so the model samples consistently across runs:

```sh
swival "task" --seed 42 --report run1.json
```

Not all models guarantee identical outputs with the same seed, but it reduces
variance significantly. See [Customization](customization.md#seed) for details.

### Comparing models

```sh
for model in qwen3-coder-next deepseek-coder-v2; do
    swival "Fix the failing tests in tests/" \
        --model "$model" \
        --report "results/${model}.json"
done
```

### Comparing settings

```sh
for temp in 0.2 0.55 0.8; do
    swival "Refactor src/api.py" \
        --temperature "$temp" \
        --report "results/temp-${temp}.json"
done
```

### Evaluating AGENT.md files

```sh
for variant in minimal detailed strict; do
    cp "agent-variants/${variant}.md" project/AGENT.md
    swival "Add input validation to the CLI" \
        --base-dir project \
        --report "results/agent-${variant}.json"
done
```

### Reading reports

Reports are plain JSON. Use `jq` to pull out what you need:

```sh
# Outcome and turn count
jq '{outcome: .result.outcome, turns: .stats.turns}' run1.json

# Total time spent in LLM calls vs tool execution
jq '{llm: .stats.total_llm_time_s, tools: .stats.total_tool_time_s}' run1.json

# Which tools were used and how often
jq '.stats.tool_calls_by_name' run1.json

# All failed tool calls from the timeline
jq '[.timeline[] | select(.type == "tool_call" and .succeeded == false)]' run1.json

# Which skills were activated
jq '.stats.skills_used' run1.json

# Did context management kick in?
jq '{compactions: .stats.compactions, turn_drops: .stats.turn_drops}' run1.json
```

### Comparing two runs

```sh
# Side-by-side outcome summary
paste <(jq -r '.result.outcome' a.json) <(jq -r '.result.outcome' b.json)

# Turns and tool calls
diff <(jq '{turns: .stats.turns, tools: .stats.tool_calls_total}' a.json) \
     <(jq '{turns: .stats.turns, tools: .stats.tool_calls_total}' b.json)
```

## What the report doesn't include

The report captures the agent's behavior, not the correctness of its output.
Whether the agent actually solved the task -- whether the code compiles, tests
pass, the refactor is sound -- is up to your evaluation harness. The `answer`
field gives you the agent's final text, and the `outcome` tells you whether it
finished cleanly, but judging quality is outside the scope of the report.
