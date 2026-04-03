# Python API

Swival exposes a small public API for embedding the agent loop in your own programs. Everything you need is importable from the top-level `swival` package.

```python
from swival import Session, Result, run
from swival import AgentError, ConfigError, ContextOverflowError, LifecycleError
```

## `run(question, *, base_dir=".", **kwargs) -> str`

One-call convenience function. Creates a `Session`, runs the question, and returns the answer string. Raises `AgentError` if the agent exhausts its turns or returns no answer. All keyword arguments are forwarded to `Session`.

```python
import swival

answer = swival.run("list the Python files in this directory", provider="lmstudio")
print(answer)
```

## `Session`

The main entry point. Stores configuration as plain attributes. Call `.run()` for single-shot questions or `.ask()` for multi-turn conversations.

### Constructor

```python
Session(
    *,
    base_dir: str = ".",
    provider: str = "lmstudio",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    aws_profile: str | None = None,
    max_turns: int = 100,
    max_output_tokens: int = 32768,
    max_context_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float = 1.0,
    seed: int | None = None,
    commands: list[str] | str | None = "all",  # "all", "none", "ask", or list (whitelist = run_command only)
    files: str = "some",
    yolo: bool = False,
    verbose: bool = False,
    system_prompt: str | None = None,
    no_system_prompt: bool = False,
    no_instructions: bool = False,
    no_skills: bool = False,
    skills_dir: list[str] | None = None,
    allowed_dirs: list[str] | None = None,
    allowed_dirs_ro: list[str] | None = None,
    sandbox: str = "builtin",
    sandbox_session: str | None = None,
    sandbox_strict_read: bool = False,
    sandbox_auto_session: bool = True,
    read_guard: bool = True,
    history: bool = True,
    memory: bool = True,
    memory_full: bool = False,
    config_dir: Path | None = None,
    proactive_summaries: bool = False,
    mcp_servers: dict | None = None,
    a2a_servers: dict | None = None,
    extra_body: dict | None = None,
    reasoning_effort: str | None = None,
    continue_here: bool = True,
    sanitize_thinking: bool | None = None,
    prompt_cache: bool = True,
    cache: bool = False,
    cache_dir: str | None = None,
    scratch_dir: str | None = None,
    retries: int = 5,
    encrypt_secrets: bool = False,
    encrypt_secrets_key: str | None = None,
    encrypt_secrets_tweak: str | None = None,
    encrypt_secrets_patterns: list | None = None,
    llm_filter: str | None = None,
    subagents: bool = False,
    lifecycle_command: str | None = None,
    lifecycle_timeout: int = 300,
    lifecycle_fail_closed: bool = False,
    lifecycle_enabled: bool = True,
)
```

All parameters are keyword-only. The important ones:

| Parameter               | Description                                                                                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base_dir`              | Project root. Tools resolve paths relative to this.                                                                                                                                                                                                                                                     |
| `provider`              | LLM provider: `"lmstudio"`, `"huggingface"`, `"openrouter"`, `"chatgpt"`, `"google"`, `"bedrock"`, `"generic"`, `"command"`, or a command string.                                                                                                                                                       |
| `model`                 | Model identifier. Required for most providers; LM Studio auto-discovers.                                                                                                                                                                                                                                |
| `api_key`               | API key. Can also be set via provider-specific env vars.                                                                                                                                                                                                                                                |
| `base_url`              | Override the provider's default endpoint.                                                                                                                                                                                                                                                               |
| `max_turns`             | Maximum agent loop iterations before returning `exhausted=True`.                                                                                                                                                                                                                                        |
| `max_output_tokens`     | Maximum tokens per LLM response.                                                                                                                                                                                                                                                                        |
| `max_context_tokens`    | Hard cap on context window size. `None` uses the provider's default.                                                                                                                                                                                                                                    |
| `temperature`           | Sampling temperature. `None` uses the provider's default.                                                                                                                                                                                                                                               |
| `files`                 | Filesystem access policy: `"some"` (workspace only, the default), `"all"` (unrestricted), or `"none"` (`.swival/` only).                                                                                                                                                                                |
| `commands`              | Command execution policy: `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive approval), or a list of whitelisted command names. In unrestricted modes (`"all"`, `"ask"`), both `run_command` and `run_shell_command` are available. A whitelist exposes only `run_command`. |
| `yolo`                  | Shorthand for `files="all"`. Explicit `files` takes precedence.                                                                                                                                                                                                                                         |
| `system_prompt`         | Override the default system prompt.                                                                                                                                                                                                                                                                     |
| `mcp_servers`           | MCP server configurations (see [MCP](mcp.html)).                                                                                                                                                                                                                                                        |
| `a2a_servers`           | A2A server configurations (see [A2A](a2a.html)).                                                                                                                                                                                                                                                        |
| `lifecycle_command`     | Shell command to run at startup and exit (see [Lifecycle Hooks](lifecycle-hooks.html)).                                                                                                                                                                                                                 |
| `lifecycle_fail_closed` | If `True`, hook failures raise `LifecycleError` instead of being silently ignored.                                                                                                                                                                                                                      |
| `llm_filter`            | Path to a filter script that can redact or block outbound LLM requests (see [Outbound LLM Filter](llm-filter.html)).                                                                                                                                                                                    |
| `subagents`             | Enable parallel subagent support (`spawn_subagent` / `check_subagents` tools).                                                                                                                                                                                                                          |
| `encrypt_secrets`       | Enable format-preserving secret encryption (see [Secret Encryption](secrets.html)).                                                                                                                                                                                                                     |
| `retries`               | Number of LLM call retries on transient failures. Must be >= 1.                                                                                                                                                                                                                                         |
| `history`               | Write `HISTORY.md` after successful runs.                                                                                                                                                                                                                                                               |
| `memory`                | Load memory files (`.swival/memory/`) into the system prompt.                                                                                                                                                                                                                                           |
| `prompt_cache`          | Inject explicit `cache_control` annotations for Anthropic/Gemini/Bedrock. Default `True`. Set `False` to opt out.                                                                                                                                                                                       |
| `cache`                 | Cache LLM responses to disk for deterministic replay.                                                                                                                                                                                                                                                   |
| `verbose`               | Print diagnostics to stderr.                                                                                                                                                                                                                                                                            |

Parameters not listed here correspond to the same-named CLI flags and config keys. See [Customization](customization.html) for the full config reference.

### Streaming and Cancellation Hooks

After constructing a `Session`, you can set two attributes for streaming events and cooperative cancellation. These are used by the A2A server internally, but are available to any library consumer.

```python
import threading
from collections.abc import Callable

session = Session(provider="lmstudio")

# Stream agent loop events (tool calls, text chunks, status changes).
def on_event(kind: str, data: dict) -> None:
    print(f"{kind}: {data}")

session.event_callback = on_event

# Cancel a running agent loop from another thread.
stop = threading.Event()
session.cancel_flag = stop

# In another thread: stop.set() to request graceful cancellation.
```

**`event_callback`** `Callable[[str, dict], None] | None` — called during the agent loop whenever something interesting happens. `kind` is one of:

| Kind              | Data keys                                     | Description                                                                                   |
| ----------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `"text_chunk"`    | `text`, `turn`                                | Final answer text (emitted when the assistant responds without tool calls).                   |
| `"tool_start"`    | `name`, `turn`                                | A tool call is about to execute.                                                              |
| `"tool_finish"`   | `name`, `turn`, `elapsed`                     | A tool call completed. `elapsed` is wall-clock seconds.                                       |
| `"tool_error"`    | `name`, `turn`, `error`                       | A tool call failed. `error` is the first 500 chars of the error message.                      |
| `"status_update"` | `turn`, `max_turns`, `elapsed`                | Emitted at the start of each turn with progress info.                                         |
| `"status_update"` | `turn`, `cancelled`                           | Emitted when the loop exits due to `cancel_flag`.                                             |
| `"status_update"` | `turn`, `type` (`"reasoning"`), `text_length` | Emitted when the assistant produces reasoning text alongside tool calls (not a final answer). |

Exceptions raised by the callback are silently swallowed — the agent loop never fails because of a callback error.

**`cancel_flag`** `threading.Event | None` — the agent loop checks this at the start of each turn and between tool calls. When set, the loop exits gracefully at the next check point. The loop does not interrupt a tool call that is already running.

Both default to `None` (no streaming, no external cancellation).

### `Session.run(question, *, report=False) -> Result`

Single-shot execution. Each call is independent — fresh message history, fresh state.

```python
session = Session(provider="lmstudio")
result = session.run("refactor the login handler")
print(result.answer)
```

If `report=True`, the returned `Result` includes a `report` dict with timing, token usage, and tool stats (see [Reports](reports.html)).

The exit lifecycle hook runs automatically after `run()`, even if the agent loop raises.

### `Session.ask(question) -> Result`

Multi-turn conversation. Shares message history across calls, like the REPL.

```python
session = Session(provider="lmstudio")
r1 = session.ask("read src/auth.py and explain the login flow")
r2 = session.ask("now add rate limiting to that handler")
session.close(outcome="success", exit_code=0)
```

On success, the assistant's reply is appended to the shared history so subsequent calls build on prior context.

On failure, the message list is rolled back to its state before the call — including any in-place mutations from compaction — so the session stays usable. State objects (thinking notes, todo items, file tracker) are not rolled back; partial progress from the failed turn is preserved.

Raises `AgentError` (or one of its subclasses) on LLM, tool, or infrastructure failures. In particular, `ContextOverflowError` is raised when the context window is exhausted even after all compaction strategies have been tried. The first `ask()` call triggers setup, so `LifecycleError` can also be raised if a fail-closed startup hook fails.

### `Session.reset()`

Clear the conversation state. The next `ask()` starts with a fresh message history. Setup (provider validation, MCP connections, etc.) is preserved.

### `Session.close(*, outcome=None, exit_code=None)`

Explicitly close the session and run the exit lifecycle hook. Pass `outcome` and `exit_code` to make them available to the hook as `SWIVAL_OUTCOME` and `SWIVAL_EXIT_CODE`.

Idempotent — safe to call after `run()` already ran the exit hook. Resources (MCP servers, cache, secrets) are always cleaned up.

Raises `LifecycleError` if the exit hook fails and `lifecycle_fail_closed` is `True`.

### Context Manager

`Session` supports `with` blocks. The exit hook runs on block exit; resources are cleaned up regardless.

```python
with Session(lifecycle_command="./sync.sh") as s:
    s.ask("first question")
    s.ask("follow-up")
# exit hook fires, resources cleaned up
```

If `lifecycle_fail_closed` is `True` and the exit hook fails, `LifecycleError` propagates from `__exit__` — but only when no other exception is already active.

## `Result`

Returned by `Session.run()` and `Session.ask()`.

```python
@dataclass
class Result:
    answer: str | None
    exhausted: bool
    messages: list[dict]
    report: dict | None
```

| Field       | Description                                                                    |
| ----------- | ------------------------------------------------------------------------------ |
| `answer`    | The agent's final text answer, or `None` if it never produced one.             |
| `exhausted` | `True` if the agent hit `max_turns` without finishing.                         |
| `messages`  | Deep copy of the full message history (system, user, assistant, tool results). |
| `report`    | Timing and token report dict if `report=True` was passed, otherwise `None`.    |

## Exceptions

Exceptions from the agent loop and configuration layer are subclasses of `AgentError`. Code that catches `AgentError` handles the common failure modes. Catch a subclass when you need finer control.

The constructor itself may raise standard Python exceptions for argument validation errors (e.g. `ValueError` for `retries < 1`). Unexpected runtime failures from underlying libraries can also escape as their original types — `AgentError` covers swival's own error paths, not every possible exception.

```text
AgentError
├── ConfigError
├── ContextOverflowError
└── LifecycleError
```

### `AgentError`

Base class for all swival errors that reach the caller. Raised on LLM failures, tool dispatch errors, provider connection problems, and any other runtime failure in the agent loop.

### `ConfigError`

Invalid configuration before the agent loop starts: missing model, bad API key format, malformed MCP server config, non-existent directories in `allowed_dirs`, conflicting options.

`ConfigError` is a subclass of `AgentError`, so `except AgentError` catches it too.

### `ContextOverflowError`

The context window is full and all compaction strategies (message shrinking, turn dropping, tool schema pruning, system prompt truncation) have been exhausted. This means the conversation grew too large for the model's context window and could not be recovered.

Callers that want to handle context exhaustion differently — retry with a larger-context model, split the task, or save partial progress — can catch this specifically:

```python
from swival import Session, ContextOverflowError

session = Session(provider="lmstudio")
try:
    result = session.ask("very large task")
except ContextOverflowError:
    session.reset()
    result = session.ask("smaller subtask")
```

### `LifecycleError`

A lifecycle hook (startup or exit) failed while `lifecycle_fail_closed` is `True`. When `lifecycle_fail_closed` is `False` (the default), hook failures are silently ignored and this exception is never raised.

Propagates from `Session.run()`, `Session.ask()` (startup hooks), `Session.close()`, and `Session.__exit__()` (exit hooks). See [Lifecycle Hooks](lifecycle-hooks.html) for details.

### Internal exceptions

`FilterError`, `McpShutdownError`, and `A2aShutdownError` are caught internally and either wrapped as `AgentError` or handled silently. They are not part of the public API.
