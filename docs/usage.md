# Usage

Swival operates in two modes: one-shot (the default) and interactive (REPL).

## One-shot mode

Give it a task, let it work, get the result:

```sh
swival "Review this codebase for security issues"
```

The agent loops through tool calls -- reading files, searching code, reasoning
-- until it has a final answer. That answer is printed to stdout. Everything else
(turn headers, tool call logs, timing) goes to stderr.

This separation is intentional. You can pipe the output:

```sh
swival -q "Summarize the API surface of src/" > api-summary.txt
```

The `-q` (or `--quiet`) flag suppresses all diagnostic output on stderr, so you
get just the answer.

### Giving the agent more power

By default, Swival can read, write, and search files within the current
directory. If you want it to run commands too:

```sh
swival --allowed-commands ls,git,python3 \
    "Create a tool that returns a random number between 0 and 42"
```

The available commands influence the agent's behavior -- if you allow `bun` and
`node`, it'll lean toward JavaScript. Allow `python3` and it'll write Python.

### Exit codes

- `0` -- success, the agent produced an answer
- `1` -- error (couldn't connect to LM Studio, invalid arguments, etc.)
- `2` -- max turns exhausted without a final answer

## Interactive mode (REPL)

```sh
swival --repl
```

This gives you an interactive session with conversation history that carries
across questions. It works like Claude Code or Codex -- ask a question, watch it
work, ask a follow-up.

You can also start a REPL with an initial question:

```sh
swival --repl "Look at the project structure and tell me what this does"
```

### REPL commands

- `/help` -- show available commands
- `/clear` -- reset conversation history (keeps system prompt)
- `/compact` -- compress context by truncating old tool results
- `/compact --drop` -- more aggressive: drops entire middle turns
- `/add-dir <path>` -- grant read/write access to an additional directory
- `/extend` -- double the current max turns limit
- `/extend <N>` -- set max turns to a specific number
- `/continue` -- reset the turn counter and continue the agent loop
- `/exit` or `/quit` -- exit (Ctrl-D works too)

The REPL uses `prompt-toolkit`, so you get command history, history search, and
line editing.

## CLI flags

### Model and provider

| Flag         | Default                 | Description                                |
| ------------ | ----------------------- | ------------------------------------------ |
| `--provider` | `lmstudio`              | LLM provider (`lmstudio` or `huggingface`) |
| `--model`    | auto-discovered         | Override model identifier                  |
| `--base-url` | `http://127.0.0.1:1234` | Server base URL                            |
| `--api-key`  | from env                | API key (overrides `HF_TOKEN`)             |

### Behavior tuning

| Flag                   | Default    | Description                                         |
| ---------------------- | ---------- | --------------------------------------------------- |
| `--max-turns`          | `100`      | Maximum agent loop iterations                       |
| `--max-output-tokens`  | `32768`    | Maximum output tokens per LLM call                  |
| `--max-context-tokens` | from model | Requested context length (may trigger model reload) |
| `--temperature`        | `0.55`     | Sampling temperature                                |
| `--top-p`              | `1.0`      | Top-p (nucleus) sampling                            |
| `--seed`               | none       | Random seed for reproducible outputs                |

### Sandboxing

| Flag                 | Default | Description                                    |
| -------------------- | ------- | ---------------------------------------------- |
| `--base-dir`         | `.`     | Base directory for file tools                  |
| `--allowed-commands` | none    | Comma-separated command whitelist              |
| `--allow-dir`        | none    | Grant access to extra directories (repeatable) |
| `--yolo`             | off     | Disable sandbox and command whitelist entirely |

### System prompt and instructions

| Flag                 | Default  | Description                             |
| -------------------- | -------- | --------------------------------------- |
| `--system-prompt`    | built-in | Custom system prompt (replaces default) |
| `--no-system-prompt` | off      | Omit system message entirely            |
| `--no-instructions`  | off      | Don't load CLAUDE.md or AGENT.md        |

### Skills

| Flag           | Default | Description                             |
| -------------- | ------- | --------------------------------------- |
| `--skills-dir` | none    | Additional skill directory (repeatable) |
| `--no-skills`  | off     | Disable skill discovery                 |

### Output and reporting

| Flag             | Default | Description                                           |
| ---------------- | ------- | ----------------------------------------------------- |
| `-q` / `--quiet` | off     | Suppress all diagnostics; only print the final result |
| `--report FILE`  | off     | Write a JSON evaluation report instead of stdout      |
| `--color`        | auto    | Force ANSI color on stderr                            |
| `--no-color`     | auto    | Disable ANSI color on stderr                          |

See [Reports](reports.md) for the full report schema, timeline event types, and
benchmarking workflows.
