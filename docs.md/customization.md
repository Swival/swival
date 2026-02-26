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

Settings use the same names as CLI flags. Lists use TOML arrays instead of comma-separated strings.

```toml
provider = "openrouter"
model = "qwen/qwen3-235b-a22b"
max_turns = 250
allowed_commands = ["ls", "git", "python3"]
allowed_dirs = ["/opt/zig/lib/std"]
allowed_dirs_ro = ["/reference/docs", "~/datasets"]
quiet = false
```

Relative paths in `allowed_dirs`, `allowed_dirs_ro`, `skills_dir`, and `reviewer` resolve against the config file's parent directory, not the working directory. Tilde paths like `~/projects` expand to the home directory.

If a project config contains `api_key` inside a git repository, Swival prints a warning because the key could be committed accidentally. Prefer environment variables for credentials.

The `--system-prompt` and `no_system_prompt` settings are mutually exclusive in config files, just as they are on the command line.

The library API (`Session` class) does not auto-load config files. If you want config file support in library code, call `load_config()` and `config_to_session_kwargs()` explicitly.

## Project Instruction Files

Swival can load two project-local instruction files from the base directory during startup. `CLAUDE.md` is injected as `<project-instructions>...</project-instructions>`, and `AGENTS.md` is injected as `<agent-instructions>...</agent-instructions>`. If both files exist, Swival loads both in that order.

Each file is capped at 10,000 characters. These instructions are appended to the built-in system prompt, which makes them a practical place to encode house rules such as test commands, coding conventions, and dependency policies.

```markdown
This is a Go project using Chi for routing. Tests use testify.
Always run `go test ./...` after making changes.
Don't add dependencies without asking.
```

Use `--no-instructions` when you do not want Swival to read either file.

```sh
swival --no-instructions "task"
```

If you set `--system-prompt`, project instruction files are also skipped because you are providing the full prompt text directly.

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
