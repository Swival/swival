# Safety and Sandboxing

Swival's sandboxing is application-level. It validates paths and whitelists
commands in Python, but it does not use OS-level isolation (namespaces,
seccomp, pledge, etc.). A sufficiently creative model or a bug in the sandbox
code could bypass these checks. Don't treat them as a security boundary for
untrusted models.

For stronger isolation, use an OS-level sandbox around Swival itself:

- **[AgentFS](agentfs.md)** provides a copy-on-write filesystem overlay. The
  agent can write freely, but your real files don't change until you explicitly
  copy them back. See [Using Swival with AgentFS](agentfs.md) for a full
  walkthrough.
- **sandbox-exec** (macOS) can restrict filesystem, network, and process
  access at the kernel level. For example, to deny network access:
  ```sh
  sandbox-exec -p '(version 1)(allow default)(deny network*)' \
      uvx swival "task" --yolo
  ```

Both can be combined. AgentFS handles filesystem isolation, sandbox-exec
handles everything else.

With that caveat, here's what Swival's built-in sandbox does and how to
configure it.

## Base directory

All file paths are resolved relative to `--base-dir` (defaults to the current
directory). The resolution function (`safe_resolve`) works like this:

1. Resolve the base directory to an absolute path, following symlinks.
2. Resolve the target path the same way.
3. Check that the resolved target is inside the resolved base directory.

If the target escapes the base directory at any point -- even through symlinks
-- the operation fails with an error. This means a symlink inside your project
that points to `/etc/passwd` won't be followed.

The filesystem root is always blocked, even in unrestricted mode. You can't
accidentally give the agent access to everything.

## Extra directories

Sometimes the agent needs to read or write files outside the base directory. Use
`--allow-dir` for that:

```sh
uvx swival --allow-dir ~/shared-data --allow-dir /opt/configs "Update the config"
```

Each `--allow-dir` path grants full read/write access. The flag is repeatable.
The path must exist, must be a directory, and can't be the filesystem root.

In the REPL, you can add directories on the fly with `/add-dir <path>`.

## Command execution

Command execution is disabled by default. The agent has no `run_command` tool
unless you explicitly enable it.

### Whitelisted commands

```sh
uvx swival --allowed-commands ls,git,python3 "task"
```

At startup, each command name is resolved to its absolute path via `which`. If a
command isn't found on PATH, Swival exits with an error. If a command resolves
to a path inside the base directory, it's also rejected -- this prevents the
agent from writing a script and then executing it.

At runtime, only the whitelisted basenames are accepted. Commands are passed as
arrays (`["ls", "-la"]`), not shell strings, so there's no shell injection. The
working directory is set to the base directory.

### YOLO mode

```sh
uvx swival --yolo "do whatever you want"
```

This disables both the filesystem sandbox and the command whitelist. The agent
can read and write any file (except the filesystem root) and run any command. No
questions asked.

Use this when you trust the model and want maximum capability. Combine it with
[AgentFS](agentfs.md) if you want an external safety net.

In YOLO mode, the `run_command` tool description changes to indicate any command
is allowed, and the system prompt is updated accordingly.

## URL fetching and SSRF protection

The `fetch_url` tool blocks requests to private, loopback, link-local, and
reserved IP addresses. It resolves the hostname to IP addresses using
`socket.getaddrinfo` and checks each address against Python's `ipaddress`
module before connecting.

Redirect chains are handled manually -- each hop is validated against the same
blocklist. This prevents SSRF attacks where a public URL redirects to an
internal service. The redirect limit is 10 hops.

Only HTTP and HTTPS schemes are allowed. Binary content types are rejected.
Response bodies are capped at 5 MB raw, with the converted output capped at
50 KB inline (larger responses are saved to files for pagination).

## Output caps

Several caps exist to prevent the agent from overwhelming the context window:

- File reads: 50 KB per read, with pagination for larger files
- Individual lines: truncated at 2,000 characters
- Directory listings and grep results: 100 entries max
- Command output: 10 KB inline, larger results saved to `.swival/` for
  pagination (auto-deleted after 10 minutes)
- URL fetch output: 50 KB inline, larger results saved to files
