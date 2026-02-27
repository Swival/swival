# Safety and Sandboxing

Swival's built-in sandbox is implemented at the application layer. It validates paths and enforces command policy in Python, but it is not an operating-system isolation boundary. You should treat it as a strong guardrail for normal use, not as a hard security perimeter against untrusted or adversarial models.

If you need stronger isolation, wrap Swival with an OS-level sandbox. [AgentFS](agentfs.md) gives you copy-on-write filesystem isolation so agent edits do not touch your real tree until you copy files back. On macOS, `sandbox-exec` can additionally limit network, process, and filesystem capabilities at the kernel policy level.

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
    swival "task" --yolo
```

AgentFS and `sandbox-exec` can be combined when you want both writable sandboxed development flow and stricter system-level controls.

## Base Directory Enforcement

All filesystem operations are anchored to `--base-dir`, which defaults to the current directory. Path checks resolve both the base directory and target path through symlinks, then verify that the resolved target remains inside an allowed root. If a path escapes through traversal or symlink indirection, the operation fails.

Even in YOLO mode, Swival blocks the filesystem root itself. You cannot grant the agent unrestricted access to `/` by accident.

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

Command execution is off by default. The agent only receives `run_command` when you explicitly enable it.

In whitelist mode, you pass a comma-separated set of command basenames.

```sh
swival --allowed-commands ls,git,python3 "task"
```

At startup, each basename is resolved to an absolute path using `which`. If a command cannot be found, Swival exits with an error. If a command resolves inside your base directory, Swival rejects it so the agent cannot modify and execute workspace binaries in one session.

At runtime in whitelist mode, commands must be passed as argument arrays, not shell strings. This removes shell interpolation and injection risk from ordinary command calls.

In YOLO mode, both the filesystem sandbox and the command whitelist are disabled. The agent can read or write any non-root path and run arbitrary commands.

```sh
swival --yolo "do whatever you want"
```

## Read-Before-Write Guard

By default, Swival blocks writes to existing files unless that file has already been read or previously written during the current session. This reduces accidental overwrites when the model has not inspected current file contents.

This guard also applies when `write_file` uses `move_from` and the destination already exists. The source path is exempt from the read requirement because renaming does not modify source content.

If you intentionally want direct write access without prior reads, disable the guard with `--no-read-guard`.

```sh
swival --no-read-guard "task"
```

## URL Fetching And SSRF Protections

The `fetch_url` tool only allows `http` and `https`. It resolves each hostname with `socket.getaddrinfo`, blocks private and internal address classes through `ipaddress`, and re-runs those checks on every redirect hop. Redirect chains are handled manually and capped at ten hops, which prevents public-to-private redirect abuse patterns.

Binary MIME types are rejected. Response bodies are capped at 5 MB before conversion, and converted inline output is capped at 50 KB.

## Output Caps

Several hard caps keep the conversation bounded. File reads are limited to 50 KB per call and lines are truncated at 2,000 characters. Directory and grep-style listings are capped at 100 results.

Command output is capped at 10 KB inline, with larger output written to `.swival/` (hard-capped at 1 MB) for paginated reads and auto-cleaned after roughly ten minutes. MCP tool output uses higher thresholds: 20 KB inline, with larger output written to `.swival/` and hard-capped at 10 MB before writing. MCP error output is inline-capped at 20 KB without file save.

URL fetch output is capped at 50 KB inline, with larger output saved to files. Response history is written to `.swival/HISTORY.md` until that file reaches 500 KB, after which new entries are skipped.
