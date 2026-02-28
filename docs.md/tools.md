# Tools

Swival gives the model a fixed set of tools at runtime. Most tools are always available. `run_command` appears only when you enable command execution with `--allowed-commands` or `--yolo`, `use_skill` appears only when skills are discovered, and MCP tools appear when external MCP servers are configured.

## `read_file`

`read_file` can read text files and directory listings inside allowed roots. File output is line-numbered, which makes later edits precise. The default window starts at `offset=1` with `limit=2000` lines.

If output is truncated, Swival appends a continuation hint with the next offset. You can also request `tail=N` to start from the end of the file, which is useful for logs.

Large responses are capped at 50 KB per call, and individual long lines are truncated at 2,000 characters. Directory reads return sorted entries and mark subdirectories with a trailing `/`.

## `write_file`

`write_file` creates or overwrites files and automatically creates missing parent directories. It supports two mutually exclusive modes. In normal write mode, you provide `content`. In move mode, you provide `move_from` and Swival performs an atomic rename when possible.

When `move_from` is used and no `content` is provided, Swival moves the source path to the destination path without copying text content. This supports non-text files and symlinks as well. If the destination already exists, the read-before-write guard still applies to that destination, while the source path is exempt because rename does not modify source content.

## `edit_file`

`edit_file` is the main incremental editing tool. It replaces `old_string` with `new_string` in an existing file and supports `replace_all` when you intentionally want multiple replacements.

Matching is done in three passes. Swival tries an exact string match first. If that fails, it retries with per-line trimmed matching so leading and trailing whitespace differences do not break the edit. If that still fails, it retries with Unicode normalization so smart quotes, em dashes, and ellipsis variants map to ASCII equivalents. If multiple matches are found and `replace_all` is false, the call fails to prevent accidental bulk edits.

## `delete_file`

`delete_file` is a soft delete. Instead of removing files permanently, Swival moves them into `.swival/trash/<trash_id>/` and appends metadata to `.swival/trash/index.jsonl`. Directories are not allowed.

Trash retention is enforced automatically. Entries older than seven days are removed, and total trash size is capped at 50 MB with oldest-first eviction when needed.

## `list_files`

`list_files` recursively evaluates glob patterns such as `**/*.py` and returns matches sorted by file modification time, newest first. Results are capped at 100 files and output is still bounded by the same 50 KB response cap.

## `grep`

`grep` searches file contents with Python regular expressions. Matches are grouped by file, include line numbers, and are sorted by file recency so the newest files are surfaced first. You can narrow by directory with `path` and by filename glob with `include` (supports `**/*.ext` patterns).

Set `case_insensitive` to `true` for case-insensitive matching. Results are capped at 100 matches and long lines are truncated to 2,000 characters.

## `think`

`think` is structured scratchpad reasoning. It lets the model capture numbered thoughts, revise earlier thoughts, and branch from a prior thought to compare alternative approaches. This is especially helpful for debugging and multi-step refactors.

The only required parameter is `thought`. Everything else is optional. A `mode` parameter (`"new"`, `"revision"`, `"branch"`) selects the type of thought. Revision mode requires `revises_thought` to reference an earlier thought number. Branch mode requires `branch_from_thought` plus a `branch_id` label.

The tool applies tolerant coercion so models that send extra or contradictory fields don't get stuck in validation loops. Incompatible fields are stripped based on the inferred mode, and corrective error messages include valid thought numbers when a reference is wrong.

## `todo`

`todo` tracks work items during a run. The list is stored in `.swival/todo.md`, so the agent can recover state even after context compaction. Actions include `add`, `done`, `remove`, `clear`, and `list`, and each action returns the full current list.

Matching for `done` and `remove` is fuzzy in a controlled way, so exact wording is not required every time. Swival tries exact matching first, then prefix matching, then substring matching. The list allows up to 50 items, and each item can be up to 500 characters.

## History Logging

Every final answer is appended to `.swival/HISTORY.md` with a timestamp and the originating question. This file is capped at 500 KB. Once full, new entries are skipped instead of rotating older content. Use `--no-history` if you do not want history writes.

## `fetch_url`

`fetch_url` downloads HTTP or HTTPS content and returns it as markdown, plain text, or raw HTML. It is designed for documentation lookup and API reference pulls. Binary content types are rejected.

Raw response bodies are capped at 5 MB, and inline output is capped at 50 KB. Larger converted outputs are saved under `.swival/` so the agent can page through them with `read_file`.

SSRF protections are built in. Swival resolves every URL in the redirect chain and blocks private, loopback, link-local, and reserved addresses.

## `run_command`

`run_command` is disabled by default. You can enable it with a whitelist.

```sh
swival --allowed-commands ls,git,python3 "Run the tests"
```

In whitelist mode, the command must be passed as an array of arguments, not as a shell string. Swival resolves each allowed command to an absolute path at startup and rejects commands that resolve inside the base directory, so the model cannot edit and execute workspace scripts in one loop.

Timeout defaults to 30 seconds and is clamped to a maximum of 120 seconds. Inline command output is capped at 10 KB. Larger output is written to `.swival/cmd_output_*.txt` and hard-capped at 1 MB before writing to disk. Those files are cleaned up automatically after roughly ten minutes.

In YOLO mode, command execution is unrestricted and Swival also accepts shell command strings through `/bin/sh -c` on Unix or `cmd.exe /c` on Windows.

## `use_skill`

When skills are discovered, Swival exposes `use_skill` so the model can load full instructions on demand. The system prompt only includes a compact skill catalog at startup, and full skill instructions are injected only when the tool is called. This keeps the default prompt smaller while still allowing rich task-specific guidance.

## `snapshot`

`snapshot` is a context management tool for collapsing exploration into compact summaries. When the model spends many turns reading files, grepping, and reasoning before arriving at a conclusion, `snapshot` lets it collapse all of that into a single short message so the context window stays clean for the actual work.

The tool supports four actions: `save`, `restore`, `cancel`, and `status`.

`save` sets an explicit checkpoint to mark the start of a focused investigation. It takes a required `label` parameter (max 100 characters). Only one explicit checkpoint can exist at a time.

`restore` collapses all turns since the checkpoint into a single summary message. It takes a required `summary` parameter (max 4,000 characters) and an optional `force` parameter that defaults to false. If no explicit checkpoint exists, it collapses from the last implicit checkpoint instead.

`cancel` clears the explicit checkpoint without collapsing anything. `status` reports the current checkpoint state, dirty status, and history.

### Implicit Checkpoints

Calling `save` before `restore` is not required. The system automatically creates implicit checkpoints at every user message, after each successful restore, and on conversation reset. When `restore` is called without a prior `save`, it collapses everything since the last implicit checkpoint, which is typically the most recent user message.

### Dirty Scopes

Tools are classified as read-only or mutating. Read-only tools (`read_file`, `list_files`, `grep`, `glob`, `fetch_url`, `think`, `todo`, `snapshot`) are safe to collapse because they don't change anything on disk. Mutating tools (`write_file`, `edit_file`, `delete_file`, `run_command`, and unknown MCP tools) dirty the scope.

If the scope contains mutating tool calls, `restore` fails with a list of the dirty tools. Pass `force=true` to override when you are confident the summary captures the mutations.

### Snapshot History

Completed snapshots are preserved across context compaction. Up to 10 past summaries are retained and injected into the system prompt so knowledge survives aggressive compaction.

The collapsed message that replaces all intermediate turns looks like this:

```
[snapshot: <label>]
<your summary>
(collapsed N turns, saved ~K tokens)
```

### Example Workflow

1. User asks to debug a performance issue.
2. Agent reads six files, greps for bottlenecks, thinks through options.
3. Agent calls `snapshot` with `action=restore` and `summary="Bottleneck is in db/queries.py:89. The get_users() query does N+1 selects. Fix: add .select_related('profile') to the queryset."`.
4. All exploration collapses to roughly 100 tokens.
5. Agent proceeds to implement the fix with a clean context.

## MCP Tools

Swival can connect to external tool servers via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). MCP tools are discovered at startup and exposed alongside built-in tools. MCP tool output is size-guarded: results up to 20 KB are returned inline, larger results are saved to `.swival/` for paginated reads via `read_file`, and output is hard-capped at 10 MB. See [MCP](mcp.md) for configuration and details.
