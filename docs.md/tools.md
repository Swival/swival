# Tools

Swival gives the model a fixed set of tools. These are always available (except
`run_command` and `use_skill`, which are opt-in). The agent decides when and how
to use them.

## File reading

The agent can read any file within the base directory. Files come back with line
numbers, which helps the agent reference specific locations when editing. Large
files are automatically paginated at 50 KB per page, with continuation hints so
the agent can read the next chunk. There's also a `tail` parameter for reading
the end of log files or build output.

Directories can be read too -- the agent gets a listing with subdirectory
markers.

## File writing

Creates or overwrites a file. Parent directories are created automatically. The
agent uses this when it needs to create something from scratch. For modifying
existing files, it prefers `edit_file` (below), which is less destructive.

The optional `move_from` parameter lets the agent rename or move a file in a
single call: it writes the new content to the destination, then trashes the
source. This is "write + trash", not an atomic rename, so the source and
destination briefly coexist if interrupted. The source does not need to have
been read first — renaming doesn't change content. If the destination already
exists, it must have been read or written first, the same requirement that
`edit_file` enforces.

## File editing

This is the agent's primary tool for modifying code. It works by string
replacement: the agent specifies the exact text to find and what to replace it
with. Behind the scenes, the edit engine uses a 3-pass matching strategy:

1. Exact match first.
2. If that fails, line-trimmed matching (ignores leading/trailing whitespace per
   line).
3. If that fails too, Unicode-normalized matching (converts smart quotes, em
   dashes, and ellipsis to their ASCII equivalents before comparing).

This makes edits robust even when the model's output has slightly different
whitespace or punctuation than the original file. If multiple matches are found,
the edit fails unless `replace_all` is set -- this prevents accidental bulk
changes.

## File deletion

Moves a file to `.swival/trash/` inside the base directory rather than deleting
it outright. The agent gets a trash ID it can report back if you need to find the
file later. Directories cannot be deleted with this tool.

The trash is automatically cleaned up: entries older than 7 days are removed, and
the total trash size is capped at 50 MB (oldest entries removed first when the cap
is exceeded). The cleanup runs at the start of each delete operation.

## Directory listing

Recursively lists files matching a glob pattern (like `**/*.py` or
`src/**/*.ts`), sorted by modification time with newest first. Results are
capped at 100 entries. The agent uses this to orient itself in unfamiliar
codebases.

## Grep

Searches file contents with regex patterns, returning matches grouped by file
with line numbers. Supports filename filtering with globs (e.g., only search
`*.py` files) and directory scoping. Results are sorted by file modification
time and capped at 100 matches.

## Thinking

A structured reasoning tool. The agent uses this to break complex problems into
numbered steps, track hypotheses during debugging, and revise earlier
conclusions when new information surfaces. It supports revisions (correcting a
specific earlier thought) and branches (exploring alternative approaches in
parallel).

The think tool also has a persistent notes feature: the agent can save concise
summaries to `.swival/notes.md` during long sessions. When older conversation
turns get compacted to save context, these notes survive and can be re-read.

## Response history

Every final answer the agent produces is appended to `.swival/HISTORY.md` with a
timestamp and the original question. This is an append-only log that persists
across sessions -- the agent can read it back with `read_file` to recall what was
asked and answered earlier in the same project.

The file is capped at 500 KB. Once it reaches that size, new entries are skipped.
There's no automatic rotation or truncation; delete or trim the file manually if
it gets too large. Use `--no-history` to disable history logging entirely.

## Web fetching

Fetches URLs and returns the content as markdown (default), plain text, or raw
HTML. The agent uses this to read documentation, check API references, or pull
in information from the web. Raw responses are capped at 5 MB before conversion;
converted output is capped at 50 KB.

URL fetching has built-in SSRF protection -- it blocks requests to private,
loopback, and link-local addresses, checking every hop in the redirect chain.
See [Safety and Sandboxing](safety-and-sandboxing.md) for details.

## Command execution

Not available by default. You enable it with `--allowed-commands`:

```sh
swival --allowed-commands ls,git,python3 "Run the tests"
```

Commands are passed as arrays (not shell strings), so there's no shell
injection risk. Each command is resolved to its absolute path at startup, and
only whitelisted basenames are allowed. Commands that resolve to paths inside
the base directory are rejected -- this prevents the model from modifying a
script and then running it.

Command output over 10 KB is saved to `.swival/` and the agent paginates
through it with `read_file`. Timeout defaults to 30 seconds (max 120).

For unrestricted command access, see `--yolo` in
[Safety and Sandboxing](safety-and-sandboxing.md).

## Skills

When skills are discovered (see [Skills](skills.md)), the agent gets a
`use_skill` tool that loads detailed instructions for a specific task on
demand. The system prompt includes a compact catalog of available skills, and
the agent calls `use_skill` when it encounters a task that matches one.
