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

## Directory listing

Recursively lists files matching a glob pattern (like ``/*.py` or
`src/`/*.ts`), sorted by modification time with newest first. Results are
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

## Web fetching

Fetches URLs and returns the content as markdown (default), plain text, or raw
HTML. The agent uses this to read documentation, check API references, or pull
in information from the web. Output is capped at 50 KB; larger responses are
saved to a file for paginated reading.

URL fetching has built-in SSRF protection -- it blocks requests to private,
loopback, and link-local addresses, checking every hop in the redirect chain.
See [Safety and Sandboxing](safety-and-sandboxing.md) for details.

## Command execution

Not available by default. You enable it with `--allowed-commands`:

```sh
uvx swival --allowed-commands ls,git,python3 "Run the tests"
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
