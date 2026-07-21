# Safety and Sandboxing

Swival's built-in sandbox is implemented at the application layer. It validates paths and enforces command policy in Python, but it is not an operating-system isolation boundary. You should treat it as a strong guardrail for normal use, not as a hard security perimeter against untrusted or adversarial models.

If you need stronger isolation, Swival integrates with two OS-level sandbox runtimes: AgentFS (copy-on-write filesystem overlay) and nono (Landlock/Seatbelt capability enforcement with optional network filtering and rollback).

## AgentFS Sandbox Mode

Pass `--sandbox agentfs` to run Swival inside an AgentFS overlay. At startup, Swival re-executes itself inside `agentfs run`, which provides copy-on-write filesystem isolation. The agent can edit files and run commands freely, but writes are confined to the overlay — your real project tree stays untouched until you copy changes back.

```sh
swival --sandbox agentfs "Refactor the auth module" --yolo
```

In this mode, `--base-dir` and each `--add-dir` path are mapped to AgentFS `--allow` rules so the agent can write to those directories inside the overlay. Everything else on the host filesystem is read-only to subprocesses.

Swival automatically generates a deterministic session ID from the project directory, so re-running `swival --sandbox agentfs` in the same directory reuses the overlay. You can see the session ID and a resume command when diagnostics are enabled (the default unless `--quiet` is set). To provide your own session ID instead:

```sh
swival --sandbox agentfs --sandbox-session my-feature "Continue the refactor" --yolo
```

To get a fresh, ephemeral overlay with no session reuse:

```sh
swival --sandbox agentfs --no-sandbox-auto-session "One-off task" --yolo
```

After a run, Swival prints a diff hint showing how to review changes (unless `--quiet` is set):

```text
  Review changes: agentfs diff swival-a1b2c3d4e5f6
```

This requires the `agentfs` binary on PATH. If it is not found, Swival exits with an actionable error. See [Using Swival With AgentFS](agentfs.md) for more workflows.

### Strict Read Mode

By default, the AgentFS sandbox only isolates writes — the agent can still read any file on the host filesystem. Pass `--sandbox-strict-read` to also restrict reads to explicitly allowed directories:

```sh
swival --sandbox agentfs --sandbox-strict-read "Analyze the project" --yolo
```

This flag requires an AgentFS version that supports strict read isolation. No current release supports it yet, so using the flag today produces a clear error with the installed version. When AgentFS ships the feature, Swival will detect it automatically and pass the appropriate flags through.

You can also combine `sandbox-exec` with AgentFS when you want additional kernel-level controls like network restriction:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
    swival --sandbox agentfs "task" --yolo
```

## nono Sandbox Mode

Pass `--sandbox nono` to run Swival inside a [nono](https://nono.sh) sandbox. Like the AgentFS path, Swival re-executes itself early in startup — this time as `nono run -- swival ...`. nono enforces filesystem and network boundaries at the kernel level, using Landlock on Linux and Seatbelt on macOS. The supervising parent process stays outside the sandbox and provides the audit trail, network proxy, and rollback machinery; the child Swival runs with the enforced capability set.

```sh
swival --sandbox nono "Refactor the auth module" --yolo
```

The base directory is granted read+write (so the agent can edit your project), and each `--add-dir` path becomes an additional grant. nono's default system-path allowances cover the Python interpreter and standard libraries, and Swival adds read-only grants for its own install location and the interpreter prefixes, so the re-exec'd process stays importable no matter how Swival was installed.

Unlike AgentFS, nono enforces by path rather than overlaying the working directory, so writes land on your real files. Enable rollback snapshots when you want a safety net:

```sh
swival --sandbox nono --nono-rollback "Try a risky migration" --yolo
```

After the run, Swival prints a hint for reviewing or undoing the changes (unless `--quiet` is set):

```text
  Review changes: nono rollback
```

nono can also filter the agent's network access. Block it entirely, or restrict it to an allowlist:

```sh
swival --sandbox nono --nono-block-net "Offline analysis" --yolo
swival --sandbox nono --nono-allow-domain api.openai.com --nono-allow-domain github.com "task" --yolo
```

The full set of nono knobs:

| Flag                            | Effect                                                         |
| ------------------------------- | -------------------------------------------------------------- |
| `--nono-profile <name>`         | Apply a named nono profile                                     |
| `--nono-rollback`               | Take atomic rollback snapshots for the session                 |
| `--nono-block-net`              | Deny all outbound network                                      |
| `--nono-allow-domain <host>`    | Add a domain to the proxy allowlist (repeatable)               |
| `--nono-network-profile <name>` | Apply a preset domain group                                    |
| `--nono-credential <service>`   | Inject credentials for a service via nono's proxy (repeatable) |
| `--nono-audit-integrity`        | Add filesystem-state hashing to the audit log                  |

All of these are accepted only with `--sandbox nono`; using one without it is an error. They can also be set in the config file (`nono_profile`, `nono_rollback`, `nono_block_net`, `nono_allow_domain`, `nono_network_profile`, `nono_credential`, `nono_audit_integrity`).

Some providers need access to credentials stored on disk. Using `--provider chatgpt` under nono grants the sandbox read/write access to LiteLLM's local OAuth state directory (`~/.config/litellm`) so the provider can authenticate. For stronger credential isolation, prefer nono credential proxy support via `--nono-credential` once it is available for your provider flow.

This requires the `nono` binary on PATH. If it is not found, Swival exits with an actionable error. See [Using Swival With nono](nono.md) for practical workflows.

`--sandbox nono` is a CLI feature: the automatic re-exec only happens for the `swival` command. Library callers using the `Session` API are not re-executed — instead, launch your own process under nono and pass `sandbox="nono"`:

```sh
nono run --allow . -- python my_agent.py
```

Inside that wrapped process, `Session(sandbox="nono")` detects the sandbox and proceeds; outside it, the session fails fast with a clear error rather than running unsandboxed.

## Network Access Modes

`--network` (config key `network`) is a top-level policy that separates the model transport from agent-controlled network access. It has three modes.

`full` is the default and today's behavior: no network restrictions beyond whatever sandbox you chose.

`provider-only` is the practical restricted mode. Swival's own process keeps normal network access, so provider calls work unchanged — hosted APIs, LM Studio, llama.cpp, anything. What changes is the agent's reach: every subprocess the agent can start (`run_command`, `run_shell_command`, the `run_python` tool, stdio MCP servers, and anything those spawn) is launched through `nono run --block-net`, so the operating system denies its entire subtree any outbound connection, loopback included. In-process network capability is removed at the application layer: `fetch_url` disappears from the tool schema and is rejected by dispatch even if the model hallucinates it, and URL-backed MCP servers and A2A clients are refused at startup. The result is an agent that can talk to its model but cannot browse, download, or exfiltrate through commands.

```sh
swival --network provider-only --provider openrouter --model qwen/qwen3-coder "Review this repo"
```

`provider-only` composes with `--sandbox builtin` (the default) and `--sandbox agentfs`, which keep owning filesystem policy. It cannot be combined with `--sandbox nono` (the wrapper would have to nest nono inside nono) and cannot run inside an external `nono run` wrapper for the same reason. Wrapped commands get the same filesystem grants a full nono run would give them — the base directory, `--add-dir` paths, temp directories, and nono's standard runtime allowances — which also means nono's built-in credential protections (`~/.ssh` and friends) apply inside them.

`none` is the air-gapped expert mode. The entire Swival process tree is re-executed under `nono --block-net`: no DNS, no public hosts, no loopback, enforced by the OS for Swival itself and every child. Because that includes the provider call, only the `command` provider — a local model process speaking over stdin/stdout — is compatible. `--serve`, remote MCP/A2A, and the nono proxy flags (`--nono-allow-domain`, `--nono-network-profile`, `--nono-credential`) are rejected up front. The air gap is verified, not assumed: if you wrap Swival in `nono run` yourself instead of letting it re-exec, startup checks nono's capability file and fails unless the surrounding sandbox really was started with `--block-net`.

```sh
swival --network none --provider command --model "llama-cli -m /models/model.gguf --simple-io" "Summarize this codebase"
```

A `network` value inherited from a config file is treated as security policy: combining it with an incompatible CLI `--sandbox` choice is an error rather than a silent downgrade, and the explicit escape hatch is passing `--network full` (or `provider-only`) yourself.

Two honest limitations. First, the boundary is what the agent executes *during* the run: code the agent wrote and you run later, or a lifecycle hook that executes agent-modified project files, is outside it. Lifecycle hooks, reviewers, and command middleware are user-authored configuration and run unwrapped. Second, on macOS, name resolution goes through the system's mDNSResponder daemon, so DNS lookups may appear to succeed inside a blocked sandbox — but no connection to any resolved address can be made.

## Base Directory Enforcement

All filesystem operations are anchored to `--base-dir`, which defaults to the auto-detected project root (the nearest ancestor directory containing `.git` or `swival.toml`, falling back to the directory you launched from). Path checks resolve both the base directory and target path through symlinks, then verify that the resolved target remains inside an allowed root. If a path escapes through traversal or symlink indirection, the operation fails.

Even with `--files all`, Swival blocks the filesystem root itself. You cannot grant the agent access to `/` by accident.

## Additional Allowed Directories

When the agent needs full access outside `--base-dir`, pass one or more `--add-dir` flags.

```sh
swival --add-dir ~/shared-data --add-dir /opt/configs "Update the config"
```

When the agent only needs to read files without modifying them, use `--add-dir-ro` instead.

```sh
swival --add-dir-ro ~/reference-docs --add-dir-ro /opt/datasets "Analyze the data"
```

Both flags can be combined. The agent gets read-write access to `--add-dir` paths and read-only access to `--add-dir-ro` paths.

```sh
swival --add-dir ./output --add-dir-ro ~/corpus "Summarize the corpus into output/"
```

Each allowed directory must already exist, must be a directory, and cannot be the filesystem root. In REPL mode, you can grant the same access dynamically with `/add-dir <path>` or `/add-dir-ro <path>`.

## Command Execution Policy

Command execution is unrestricted by default (`--commands all`). You can restrict or disable it.

In whitelist mode, you pass a comma-separated set of command basenames. Pass `"none"` to disable commands entirely. Pass `"ask"` to require interactive approval for every command bucket.

```sh
swival --commands ls,git,python3 "task"
swival --commands ask "task"
swival --commands none "task"
```

At startup, each basename is resolved to an absolute path using `which`. If a command cannot be found, Swival exits with an error. If a command resolves inside your base directory, Swival rejects it so the agent cannot modify and execute workspace binaries in one session.

At runtime in whitelist mode, only `run_command` is available and commands must be passed as argument arrays. This removes shell interpolation and injection risk from ordinary command calls. `run_shell_command` is not exposed in whitelist mode since shell strings bypass the whitelist entirely.

With `--commands all` or `--yolo`, both `run_command` and `run_shell_command` are available. The experimental `run_python` tool joins them when a Python interpreter is present and the context window is at least 100,000 tokens; it runs arbitrary Python in a `python -c` subprocess, the same trust level as shell access. In ask mode, only `run_command` is exposed — shell access requires `--commands all`.

### Ask Mode

In ask mode (`--commands ask`), Swival prompts you before running each new command category. Commands are grouped into buckets by their base name (e.g. `ls`, `git push`, `python3 -m pytest`).

Only `run_command` is available, so commands must be passed as argument arrays, which prevents shell injection. Once you approve a bucket, subsequent commands in the same bucket run without asking again.

High-risk buckets (`rm`, `git push`, `docker`, `curl`, interpreter inline-code like `bash -c` or `python3 -c`) default to deny — you must explicitly type `y` to allow them. Non-high-risk buckets default to allow on Enter.

Approval options:

- `Enter` — allow (non-high-risk) or deny (high-risk)
- `y` — allow this bucket for the rest of the session
- `n` — deny this bucket for the rest of the session
- `p` — allow and persist the approval to `.swival/approved_buckets`
- `o` — allow this one invocation only
- `a` — always re-prompt for this bucket

Subagents cannot prompt interactively. In ask mode, subagents can only run commands in buckets that are already approved — either pre-approved via `approved_buckets` in config, or runtime-persisted in `.swival/approved_buckets` from earlier interactive sessions.

```toml
# swival.toml — intentional pre-approvals (version-controllable)
commands = "ask"
approved_buckets = ["ls", "git status", "python3 -m pytest"]
```

## One-Shot Command Dispatch

In one-shot mode, `/` and `!` command dispatch is disabled by default. When input comes from a pipe, a file, or an upstream program, it may contain attacker-controlled content that starts with a slash or bang command. Executing that input as a command could run scripts from `~/.config/swival/commands/` or alter the agent's security posture (e.g. `/add-dir /`).

Pass `--oneshot-commands` to opt in when you trust the input source:

```sh
swival --oneshot-commands "/simplify swival/agent.py"
```

Or set it in config:

```toml
oneshot_commands = true
```

`--yolo` does not imply `--oneshot-commands`. They control different trust boundaries: `--yolo` governs what the agent can do (filesystem and shell access), while `--oneshot-commands` governs whether the input itself is trusted to contain commands.

In interactive mode, the user is typing directly, so commands are always enabled regardless of this flag.

## Untrusted Content Labels

Output from external sources — `fetch_url`, MCP tools, and A2A tools — is wrapped with a deterministic `[UNTRUSTED EXTERNAL CONTENT]` header before the model sees it. This header instructs the model to treat the content as data, not instructions, and to avoid changing tool-selection behavior based on it.

The label is baked into spill files too. When an external tool produces output too large for inline context (over 20 KB for MCP/A2A, over 50 KB for fetch_url), the content is saved to a temp file under `.swival/`. The untrusted header is prepended to the file contents, so the label survives when the agent reads the file back via `read_file`.

## Filesystem Access Policy

`--files` controls what the filesystem tools can access. It accepts `"some"` (the default), `"all"`, or `"none"`.

In the default mode (`--files some`), filesystem tools are restricted to the base directory and any `--add-dir` / `--add-dir-ro` paths. In `--files all` mode, the agent can read or write any non-root path. In `--files none` mode, only the `.swival/` directory is accessible through tools — the agent can still think, run commands, and fetch URLs.

```sh
swival --files all "do whatever you want"
swival --files none --commands ls,git "read-only analysis"
```

`--yolo` is shorthand for `--files all --commands all`. If you also pass an explicit `--files` or `--commands`, the explicit flag wins.

```sh
swival --yolo "unrestricted access"
swival --yolo --files none "commands only, no file access"
```

## Read-Before-Write Guard

By default, Swival blocks writes to existing files unless that file has already been read or previously written during the current session. This reduces accidental overwrites when the model has not inspected current file contents.

This guard also applies when `write_file` uses `move_from` and the destination already exists. The source path is exempt from the read requirement because renaming does not modify source content.

If you intentionally want direct write access without prior reads, disable the guard with `--no-read-guard`.

```sh
swival --no-read-guard "task"
```

## URL Fetching And SSRF Protections

The `fetch_url` tool only allows `http` and `https`. It resolves each hostname with `socket.getaddrinfo`, blocks private and internal address classes through `ipaddress`, and re-runs those checks on every redirect hop.

Loopback addresses are allowed when the hostname is explicitly `localhost`, `127.0.0.1`, or `::1`, so agents can test locally running servers. Arbitrary hostnames that resolve to loopback addresses are still blocked to prevent DNS rebinding attacks. Redirect chains are handled manually and capped at ten hops.

Binary MIME types are rejected. Response bodies are capped at 5 MB before conversion, and converted inline output is capped at 50 KB.

## Output Caps

Several hard caps keep the conversation bounded. File reads are limited to 50 KB per call and lines are truncated at 2,000 characters. Directory and grep-style listings are capped at 100 results. The 50 KB cap (which also bounds listings, grep, outline, and URL fetches) and the default 2000-line read limit can be changed with the `max_output_kb` and `max_output_lines` settings.

Command output is capped at 10 KB inline, with larger output written to `.swival/` (hard-capped at 1 MB) for paginated reads and auto-cleaned after roughly ten minutes.

MCP tool output uses higher thresholds: 20 KB inline, with larger output written to `.swival/` and hard-capped at 10 MB before writing. MCP error output is inline-capped at 20 KB without file save.

URL fetch output is capped at 50 KB inline, with larger output saved to files.

Response history is written to `.swival/HISTORY.md`, which is capped at 500 KB. When a new entry would exceed the cap, the oldest entries are trimmed to make room.
