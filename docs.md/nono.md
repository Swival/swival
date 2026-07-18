# Using Swival With nono

[nono](https://nono.sh) is a capability-based sandbox that confines Swival at the operating-system level: Landlock on Linux, Seatbelt on macOS. Where AgentFS overlays the filesystem and lets you copy changes back afterwards, nono enforces access by path. The agent process simply cannot read, write, or reach anything it was not granted, and the denial happens in the kernel rather than in Swival's own guards.

This makes nono the right choice for untrusted or destructive tasks, for runs where you want network egress under control, and for any situation where you would rather the operating system say no than rely on the agent behaving.

## Integrated Sandbox Mode

The simplest way to use nono with Swival is the built-in sandbox mode:

```sh
swival --sandbox nono "Refactor the auth module" --yolo
```

This re-executes Swival inside `nono run` automatically. Your working directory and any `--add-dir` paths are granted read+write, the platform temporary directory is writable so tools can scratch, and the Python interpreter and install paths are granted read-only so the re-exec'd process can still import and run. Everything else on the filesystem is denied.

See [Safety and Sandboxing](safety-and-sandboxing.md) for how nono fits alongside Swival's other safety layers.

The rest of this page covers how the integration works and the controls you can layer on top.

## Prerequisites

Install nono by following the instructions at [nono.sh](https://nono.sh), then run its one-time setup:

```sh
nono setup
```

Confirm the binary is on your `PATH`:

```sh
nono --version
```

You also need a working model provider for Swival itself, such as LM Studio, llama.cpp, or a remote provider like OpenRouter. See [Providers](providers.md).

## How It Works

When you pass `--sandbox nono`, Swival locates the `nono` binary and re-executes itself as:

```sh
nono run --allow <workspace> --allow <tmp> --read <interpreter> --profile swival -- swival ...
```

The parent process stays outside the sandbox and supervises the child: it provides the audit trail, the network proxy, and rollback services. The child Swival process runs under the enforced capabilities, so every tool it invokes — every `run_command`, every file write — inherits the same boundaries.

By default the integration uses nono's built-in `swival` profile, which grants the Python runtime and Swival's own config and state directories. You do not need to configure anything for a basic run.

## Filesystem Boundaries

Inside the sandbox, writes outside the granted directories fail. You can see this directly with nono before involving Swival:

```sh
nono run --profile swival -- sh -c 'echo nope > /etc/swival-test'
```

That write is denied by the kernel. The same boundary applies to the agent: a task that tries to modify a file outside the workspace cannot succeed, regardless of the `--files` level.

To grant additional directories, use Swival's `--add-dir` (read+write) — these become `--allow` grants inside nono:

```sh
swival --sandbox nono --add-dir ../shared-lib "Update the shared client" --yolo
```

## Network Controls

By default nono allows outbound network, so the call to your model provider works without any extra flags.

To restrict the agent to specific hosts, enable nono's filtering proxy with `--nono-network-profile` and allowlist the domains you trust with `--nono-allow-domain` (repeatable). Everything else is refused, which keeps your provider reachable while stopping the agent from reaching anything off the list:

```sh
swival --sandbox nono --nono-network-profile developer --nono-allow-domain openrouter.ai \
    "Refactor the parser" --yolo
```

The proxy runs in the supervising parent and terminates TLS for the allowlisted hosts; nono makes its certificate trusted inside the sandbox automatically. You can also broker a service's credentials through the proxy with `--nono-credential`, so API keys never enter the sandboxed process environment.

`--nono-block-net` goes further and refuses every outbound connection, including loopback. Because that also blocks the call to your model provider, it fits only fully offline work — for example a local `command`-provider model that needs no network at all. For ordinary runs against a hosted model or a local model server, reach for the allowlist above instead.

If what you actually want is "the model works, the agent's commands don't", you do not need to assemble that from these flags: the top-level `--network provider-only` option launches every agent subprocess through `nono run --block-net` while leaving the provider call alone, and `--network none` is the convenient spelling of the fully air-gapped setup (it selects this sandbox and `--nono-block-net` for you, and validates the provider and integrations up front). The `--nono-*` flags remain the expert layer for domain-scoped policies the top-level modes deliberately do not cover. See [Safety and Sandboxing](safety-and-sandboxing.md).

## Rollback

With `--nono-rollback`, nono records an atomic snapshot of the files the run touches so you can undo the whole thing afterwards.

```sh
swival --sandbox nono --nono-rollback "Migrate the test suite to pytest" --yolo
```

After the run, review and restore snapshots with nono directly:

```sh
nono rollback
```

This pairs well with high-autonomy runs: let the agent work freely, inspect the result, and roll back in one step if you do not want to keep it.

## Provider Credentials

The built-in `swival` profile deliberately denies credential and keychain locations, but Swival grants back exactly what your provider needs so authentication still works inside the sandbox:

- **chatgpt** — read+write to `~/.config/litellm`, the directory that holds its OAuth token cache, since it refreshes tokens on disk.
- **geap / vertexai** — read-only to the Google Cloud credentials directory (`~/.config/gcloud`, or `$CLOUDSDK_CONFIG`), plus the directory holding `$GOOGLE_APPLICATION_CREDENTIALS` when set.
- **bedrock** — read-only to `~/.aws`, plus the directories of `$AWS_SHARED_CREDENTIALS_FILE` and `$AWS_CONFIG_FILE` when set.

These grants are read-only where the provider only reads a long-lived credential and exchanges it for a short-lived token over the network, so the sandbox never gets write access to your credentials it does not need. No extra flags are required:

```sh
swival --sandbox nono --provider geap --gcp-project my-project --location us-central1 \
    --model gemini-3.1-pro "Explain this module" --yolo
```

## Audit

Every sandboxed run is recorded. Review the trail with nono:

```sh
nono audit
```

Add `--nono-audit-integrity` to include filesystem-state hashing in the audit log, which lets you verify exactly what changed:

```sh
swival --sandbox nono --nono-audit-integrity "Apply the lint fixes" --yolo
```

## Profiles

To use a profile other than the built-in `swival` one, pass `--nono-profile`:

```sh
swival --sandbox nono --nono-profile claude-code "Run the migration"
```

You can inspect and compare available profiles with `nono profile`.

## Manual Workflow

The integrated mode is the recommended path, but you can also wrap Swival with `nono run` yourself when you want full control over the grants. This mirrors how `nono` wraps any program.

```sh
cd ~/my-project

nono run --allow "$PWD" --profile swival -- \
    swival "Add a config module that reads from env vars" --yolo --max-turns 20
```

Because Swival detects that it is already inside nono (via nono's `NONO_CAP_FILE` marker), it does not try to re-exec a second time. Add `--read` and `--allow` grants as needed for extra paths.

One thing Swival does not take on faith in this setup is the network policy. If you combine an external wrapper with `--nono-block-net` or `--network none`, Swival checks the capability file nono hands the child and refuses to start unless it records that the network is actually blocked. An outer `nono run` without `--block-net` cannot masquerade as an air gap.

## REPL Sessions

The integrated sandbox mode works with REPL mode directly:

```sh
swival --sandbox nono --yolo
```

This gives you an interactive agent session under enforced capabilities. Everything the agent does for the rest of the session stays within the granted directories and network policy.

## Combining Layers

nono composes with Swival's own file and command controls. You can run under OS-level enforcement while still restricting the file tools to the workspace and the command tool to a whitelist:

```sh
swival --sandbox nono --files some --commands ls,git "Audit this code for security issues"
```

This gives you defense in depth: nono enforces the hard boundary at the kernel, and Swival's guards add a second layer inside it.

## Practical Guidance

For untrusted or destructive tasks, `--sandbox nono --yolo` is the combination to reach for: the agent gets full capability within the workspace while the operating system guarantees it cannot escape. Add `--nono-rollback` when you want a clean undo, and `--nono-block-net` or `--nono-allow-domain` when network egress matters. For everyday edits you trust, the lighter `builtin` or `agentfs` modes are usually enough.

## Further Reading

- [Safety and Sandboxing](safety-and-sandboxing.md)
- [Using Swival With AgentFS](agentfs.md)
- [Providers](providers.md)
