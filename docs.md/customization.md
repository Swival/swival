# Customization

## Configuration Files

Swival supports persistent settings through TOML config files at two levels: a user-global file and a project-local file. Both are optional.

The global config lives at `~/.config/swival/config.toml` (or `$XDG_CONFIG_HOME/swival/config.toml` if you set that variable). The project config is `swival.toml` in the base directory.

When both files exist, project settings override global settings. CLI flags override everything. The full precedence order is:

CLI flags > project config > global config > hardcoded defaults

To generate a starter config with all settings commented out, run one of:

```sh
swival --init-config            # global config
swival --init-config --project  # project config in current directory
```

Settings mostly use the same names as CLI flags, with a few exceptions: the config keys `allowed_dirs` and `allowed_dirs_ro` correspond to the CLI flags `--add-dir` and `--add-dir-ro`. Lists use TOML arrays instead of comma-separated strings.

```toml
provider = "openrouter"
model = "qwen/qwen3-coder-next"
# base_url = "http://127.0.0.1:1234"    # provider API base URL
# api_key = "..."                       # prefer env vars over config for credentials
# system_prompt = "..."                 # replaces the built-in system prompt
# no_system_prompt = false              # omit the system message entirely
max_turns = 250
# temperature = 0.3
# top_p = 0.9
# seed = 42
commands = ["ls", "git", "python3"]    # also accepts "all", "none", or "ask"
# approved_buckets = ["ls", "git push", "python3 -m pytest"]  # explicit pre-approvals for ask mode (runtime approvals go to .swival/approved_buckets)
allowed_dirs = ["/tmp"]
allowed_dirs_ro = ["/opt/zig/lib/std"]
proactive_summaries = true
# no_continue = false
quiet = false
extra_body = { chat_template_kwargs = { enable_thinking = false } }
reasoning_effort = "high"
cache = true
# cache_dir = ".swival"
# retries = 5              # max provider retries on transient network errors
# color = true             # true = force color, false = force no-color, absent = auto
# files = "some"           # filesystem access: "some" (workspace) | "all" (unrestricted) | "none" (.swival/ only)
# yolo = false             # shorthand for files = "all" + commands = "all"
# oneshot_commands = false # enable / and ! command dispatch in one-shot mode
# no_read_guard = false    # disable read-before-write guard
# no_mcp = false           # disable MCP server connections
# no_a2a = false           # disable A2A agent connections
# no_instructions = false  # skip CLAUDE.md and AGENTS.md loading
# no_skills = false        # disable skill discovery
# no_memory = false        # skip auto-memory loading
# memory_full = false      # inject entire MEMORY.md instead of budgeted retrieval
# no_history = false       # disable HISTORY.md writes
# sandbox = "builtin"      # "builtin" or "agentfs"
# sandbox_session = "..."  # AgentFS session ID
# sandbox_strict_read = false  # strict read isolation (agentfs only)
# sandbox_auto_session = true  # auto session ID from project dir (agentfs only)
# encrypt_secrets = false  # transparent credential encryption
# encrypt_secrets_key = "..." # hex 32-byte key for stable ciphertext
# sanitize_thinking = false   # strip leaked <think> tags
# trace_dir = "traces"        # directory for JSONL trace export (HuggingFace format)

# Reviewer settings
reviewer = "swival --reviewer-mode"
# self_review = true               # shorthand: mirrors provider/model into reviewer
review_prompt = "Focus on correctness and test coverage"
max_review_rounds = 15
objective = "objective.md"
verify = "verification/working.md"

# A2A agents (see docs.md/a2a.md)
[a2a_servers.research-agent]
url = "https://research.example.com"
auth_type = "bearer"
auth_token = "sk-..."
```

For the `chatgpt` provider (ChatGPT Plus/Pro subscriptions), no API key is needed since authentication is handled through OAuth:

```toml
provider = "chatgpt"
model = "gpt-5.4"
```

Relative paths in `allowed_dirs`, `allowed_dirs_ro`, `skills_dir`, `cache_dir`, `objective`, and `verify` resolve against the config file's parent directory, not the working directory. Tilde paths like `~/projects` expand to the home directory.

The `reviewer`, `llm_filter`, `lifecycle_command`, and `command_middleware` values are shell-split; only path-like first tokens — anything starting with `/`, `~`, or containing a `/` (e.g. `./`, `../`, `.rtk/`, `scripts/`) — are resolved against the config directory, while bare command names like `swival` are left for PATH lookup at runtime. The same resolution applies to `model` when `provider = "command"`.

If a project config contains `api_key` — at the top level or inside a profile — inside a git repository, Swival prints a warning because the key could be committed accidentally. Prefer environment variables for credentials.

The `--system-prompt` and `no_system_prompt` settings are mutually exclusive in config files, just as they are on the command line.

The library API (`Session` class) does not auto-load config files. If you want config file support in library code, call `load_config()` and `config_to_session_kwargs()` explicitly. Note that `config_to_session_kwargs()` drops `approved_buckets` — for `commands="ask"`, pass `approved_buckets` to `Session` directly, optionally using `load_persisted_buckets()` from `swival.command_policy` to include runtime-persisted approvals.

## Profiles

If you switch between multiple model setups — a local LM Studio model for quick tasks, a ChatGPT model for hard problems, an OpenRouter model for long-context work — profiles let you define each setup once and switch with a single flag.

```toml
active_profile = "fast-local"

[profiles.fast-local]
provider = "lmstudio"
model = "qwen3-coder-next"
max_context_tokens = 65536

[profiles.gpt5]
provider = "chatgpt"
model = "gpt-5.4"
reasoning_effort = "high"

[profiles.router-main]
provider = "openrouter"
model = "z-ai/glm-5"
max_context_tokens = 131072

[profiles.ollama]
provider = "generic"
base_url = "http://127.0.0.1:11434"
model = "qwen3:32b"
```

Switch with `--profile`:

```sh
swival --profile gpt5 "review this patch"
swival --profile fast-local "write tests"
```

Set `active_profile` in config to pick a default without typing `--profile` every time. Project config overrides global config, and `--profile` on the CLI overrides both.

List available profiles with `--list-profiles`:

```sh
swival --list-profiles
```

Each profile requires `provider`. The allowed keys are: `provider`, `model`, `api_key`, `base_url`, `aws_profile`, `max_output_tokens`, `max_context_tokens`, `temperature`, `top_p`, `seed`, `extra_body`, `reasoning_effort`, `sanitize_thinking`, and `description`. Keys outside this set — like `files`, `commands`, or `reviewer` — are rejected with an error listing the allowed keys. Profiles are for choosing a model stack, not for changing agent behavior. The `description` key is metadata — it appears in profile listings but is not passed to the provider.

If `--profile` is combined with explicit flags like `--provider` or `--model`, the explicit flags win on a per-key basis, just like CLI flags override config everywhere else in Swival.

```sh
swival --profile gpt5 --reasoning-effort medium "task"
```

This is particularly useful with `--model`. A profile can lock down the provider, base URL, API key, and other settings for a given service, while `--model` lets you swap models on the fly without duplicating the rest of the configuration. For example, with a HuggingFace profile:

```toml
active_profile = "hf"

[profiles.hf]
provider = "huggingface"
```

You get the default model most of the time, and override it for a specific task:

```sh
swival "everyday task"
swival --model "Qwen/Qwen3-Coder" "task needing a different model"
```

Any model supported by the inference endpoint works — no extra profile needed.

You can also switch profiles mid-session from the REPL without restarting:

```text
swival> /profile              # list profiles, active one marked with →
swival> /profile fast-local   # switch to the "fast-local" profile
swival> /profile -            # revert to the profile active at session start
```

Switching only changes LLM settings — conversation history, tools, files, and all other state are preserved. New subagents spawned after the switch use the new profile; existing running subagents are unaffected.

Profiles defined in global config and project config merge per-key for the same profile name. A project can refine a global profile by overriding just one or two keys without copying the whole table.

```toml
# In global config: defines the base profile
[profiles.shared]
provider = "openrouter"
model = "z-ai/glm-5"
max_context_tokens = 131072

# In project swival.toml: overrides just the model
[profiles.shared]
model = "z-ai/glm-5-mini"
```

## Instruction Files

Swival loads instruction files during startup and appends them to the system prompt. `CLAUDE.md` provides project rules (`<project-instructions>`), while `AGENTS.md` provides agent workflow commands and conventions (`<agent-instructions>`). Use `/init` in the REPL to auto-generate a project-level `AGENTS.md`, or `/remember <text>` to add individual facts to its `## Conventions` section.

### CLAUDE.md

Loaded from the project base directory only. Capped at 10,000 characters.

### AGENTS.md

Loaded from up to three locations, in this order:

1. **User-level**: `~/.config/swival/AGENTS.md` (or `$XDG_CONFIG_HOME/swival/AGENTS.md`)
2. **Global cross-agent**: `~/.agents/AGENTS.md`
3. **Project-level**: `<base-dir>/AGENTS.md`

All are optional. Content is concatenated inside a single `<agent-instructions>` block sharing a combined 10,000 character budget. Files are read in the order shown; earlier files get budget priority.

The user-level file is for swival-specific personal conventions. The global file (`~/.agents/AGENTS.md`) follows the cross-agent standard shared with OpenCode, OpenHands, and similar tools — put conventions here that should apply regardless of which agent you're using. The project-level file is for project-specific rules.

```markdown
This is a Go project using Chi for routing. Tests use testify.
Always run `go test ./...` after making changes.
Don't add dependencies without asking.
```

### Disabling instructions

Use `--no-instructions` to skip all instruction files (CLAUDE.md and AGENTS.md at all levels, including `~/.agents/AGENTS.md`).

```sh
swival --no-instructions "task"
```

Use `--no-memory` to skip loading auto-memory from `.swival/memory/`.

```sh
swival --no-memory "task"
```

Use `--memory-full` to inject the entire `MEMORY.md` file into the prompt instead of the default budgeted retrieval. This is the legacy behavior and serves as a fallback if the retrieval-based injection misses entries you need.

```sh
swival --memory-full "task"
```

If you set `--system-prompt`, instruction files are also skipped because you are providing the full prompt text directly.

## System Prompt Control

The built-in prompt is stored in `swival/system_prompt.txt` and defines default behavior, tool policy, and coding expectations.

You can replace it completely with `--system-prompt`.

```sh
swival --system-prompt "You are a security auditor. Only report vulnerabilities." "Audit src/"
```

You can also remove the system message entirely with `--no-system-prompt`.

```sh
swival --no-system-prompt "Just answer: what is 2+2?"
```

`--system-prompt` and `--no-system-prompt` are mutually exclusive.

When a system message is present, Swival appends the current local date and time to that system content.

## Sampling And Reproducibility

`--temperature` and `--top-p` control response sampling.

```sh
swival --temperature 0.3 --top-p 0.9 "task"
```

If you do not set `--temperature`, provider defaults apply. `--top-p` defaults to `1.0`.

`--seed` passes a deterministic seed when the provider supports it.

```sh
swival --seed 42 "task"
```

Seeded runs are usually more stable, but identical output is still not guaranteed across all providers, model versions, and hardware environments.

## Extra LLM Parameters

Some models accept parameters beyond the standard OpenAI chat completions API. The `extra_body` setting passes an arbitrary dictionary through to the underlying API call, so you can use provider-specific or model-specific options without Swival needing dedicated flags for each one.

On the command line, pass a JSON object:

```sh
swival --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' "task"
```

In a config file, use TOML inline table syntax:

```toml
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

In the library API:

```python
session = Session(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

The dictionary is forwarded as `extra_body` to the provider's API. Refer to your model's documentation for supported parameters.

## Reasoning Effort

Some models support a tunable reasoning level that controls how much effort the model puts into thinking before responding. This is a first-class parameter in Swival, separate from `extra_body`.

On the command line:

```sh
swival --provider chatgpt --model gpt-5.4 --reasoning-effort high "task"
```

In a config file:

```toml
reasoning_effort = "high"
```

In the library API:

```python
session = Session(reasoning_effort="high")
```

Valid levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `default`. Not all models support this parameter — when used with a model that doesn't support it, the behavior depends on the provider (it may be ignored or cause an error). Only set it when you know your model supports it.

## Thinking Tag Sanitization

Some open-weight models leak hidden-reasoning markers like `<think>` and `</think>` into their responses. This is especially common when using vLLM as the inference engine, where models may emit these tags even when thinking mode is disabled. The `sanitize_thinking` option strips these leaked tags from assistant content before returning it to the user.

This is off by default for all providers. Enable it in config if you're seeing leaked thinking tags:

```toml
sanitize_thinking = true
```

In the library API:

```python
session = Session(sanitize_thinking=True)
```

The sanitizer strips `<think>...</think>` blocks, standalone `<think>` / `</think>` lines, and special tokens like `<|start_header_id|>`. Inline mentions of these tags in code examples or backtick-quoted text are preserved.

## Outbound LLM Filter

Swival can run a user-defined script before every outbound LLM request to redact or block sensitive content. See [Outbound LLM Filter](llm-filter.md) for the script contract, configuration, and examples.

## Secret Encryption

Swival can transparently encrypt recognized credential tokens before they leave your machine. See [Secret Encryption](secrets.md) for the full documentation, including built-in token patterns, custom patterns, key management, and threat model.

## Lifecycle Hooks

Swival can run a user-defined command at startup and exit, useful for syncing `.swival/` state to and from remote storage. See [Lifecycle Hooks](lifecycle-hooks.md) for the full documentation, including environment variables, execution ordering, failure semantics, and a complete Hugging Face Buckets example.

## Command Middleware

Swival can run a user-defined command before each shell command the agent issues, allowing you to rewrite, block, or pass through commands before they execute. The primary use case is integrating with RTK to produce token-optimized command output. See [Command Middleware](command-middleware.md) for the full documentation, including the JSON contract, RTK setup, and a ready-to-use adapter script.

## Turn And Token Limits

`--max-turns` limits how many agent-loop iterations are allowed.

```sh
swival --max-turns 10 "quick task"
```

The default turn limit is `100`. If the loop reaches this limit without a final answer, Swival exits with code `2`.

`--max-output-tokens` limits tokens generated per model call.

```sh
swival --max-output-tokens 16384 "task"
```

The default is `32768`. If prompt size and context constraints require it, Swival clamps output budget downward automatically.

`--max-context-tokens` sets requested context length.

```sh
swival --max-context-tokens 65536 "task"
```

For LM Studio, this can trigger a model reload. When both `--max-context-tokens` and `--max-output-tokens` are set, `--max-output-tokens` must be less than or equal to context length.
