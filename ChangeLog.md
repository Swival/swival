# Changelog

All notable user-facing changes to Swival.

## 0.1.19

- Added `/learn` command for interactive skill discovery.

## 0.1.18

- Added `read_multiple_files` tool for reading several files in a single call.
- Added continue-here feature: session state is saved on interruption (Ctrl+C,
  max turns, compaction failure) and resumed on next start.
- Made the `todo` tool accept multiple tasks in one call.
- Extended `grep` tool with additional options.
- Fixed context overflow detection for non-standard exception types.

## 0.1.17

- Added `--reasoning-effort` option.
- Added session memories that persist across runs.
- Added GPT-5.4 to the built-in model list.
- Added markdown formatting for agent responses.
- Improved spinner and progress display.
- Improved todo list UI.
- Listed all CLI options in `--help` and sorted them alphabetically.

## 0.1.16

- Added colored diff output in the `edit_file` tool.

## 0.1.15

- Made `write_file` coerce JSON content into a string instead of erroring.

## 0.1.14

- Added ChatGPT as a provider (direct OpenAI API).

## 0.1.13

- Integrated AgentFS sandbox support with auto-session IDs, diff hints, and
  strict read mode.
- Added "Did you mean?" suggestions for mistyped tool command names.
- Made MCP servers inherit the parent process environment variables.

## 0.1.12

- Added generic OpenAI-compatible provider for any server that speaks the
  OpenAI API.
- Added snapshot tool for proactive context collapse, with `/snapshot` and
  `/restore` REPL commands.
- Added `--extra-body` option to pass arbitrary JSON to the LLM request
  (useful for disabling thinking, etc.).
- Added OpenRouter documentation and setup instructions.

## 0.1.11

- Added MCP (Model Context Protocol) server support. Servers configured in
  `swival.toml` or `.mcp.json`; tools exposed as `mcp__<server>__<tool>`.
- Added configurable size limits for MCP tool output (`MCP_INLINE_LIMIT`,
  `MCP_FILE_LIMIT`).

## 0.1.10

- Added reviewer mode (`--reviewer-mode`): an LLM-as-judge loop that
  automatically evaluates agent output, with `--objective`, `--verify`,
  and `--review-prompt` options.
- Added `--max-review-rounds` to cap review iterations.

## 0.1.9

- Graduated context compaction: `compact_messages` -> `drop_middle_turns` ->
  `aggressive_drop_turns`, replacing the previous all-or-nothing approach.
- Suggested `/continue` when the agent hits the max turn limit.
- Improved clamping and retry messages.

## 0.1.8

- Made `grep` and `list_files` tools accept file paths in addition to
  directories.
- Improved `grep` tool output.
- Reported whether the model supports vision.
- Added global instructions via `~/.config/swival/AGENTS.md`.
- Clarified `--no-instructions` behavior.

## 0.1.7

- Added configuration file support (`swival.toml` and
  `~/.config/swival/config.toml`).
- Added `--add-dir-ro` for read-only additional directories (renamed from
  `--allow-dir`).
- Auto-corrected common command syntax mistakes in yolo mode.
- Switched instructions file from `ZOK.md` to `AGENT.md`.

## 0.1.6

- Redesigned `think` tool with numbered thoughts, revisions, and branches.
- Added CI pipeline.
- Added `Makefile` with common development commands.
- Fixed trash/undo handling.
- Improved error when the model sends a file size with units.

## 0.1.5

- Added `todo` tool: persistent checklist in `.swival/todo.md` that survives
  context compaction, with periodic reminders and duplicate detection.
- Added `/init` command for bootstrapping `AGENT.md`.
- Exposed a public Python API (`swival.Session`, `swival.run()`).
- Added loading spinner during LLM calls.
- Removed the unused `notes` tool.

## 0.1.4

- Added OpenRouter as a provider.
- Added `delete_file` tool.
- Added `move_file` / `rename_file` tools.
- Added external reviewer support for automated evaluation.
- Required read-before-write: the agent must read a file before editing or
  overwriting it (disable with `--no-read-guard`).
- Printed final output even when `--report` is enabled.
- Removed default values for `temperature` and `top_p` (let the provider
  decide).

## 0.1.3

- Renamed package from `swival-agent` to `swival`.
- Added `--version` flag.
- Deepened recursive skill discovery.
- Included skill activation events in reports.

## 0.1.2

- Added `--report` for JSON session reports.
- Added `--history` to replay previous sessions.
- Revamped thinking tool.
- Allowed absolute paths in yolo mode.
- Added full shell expansion in yolo mode.
- Increased default max turn limit.

## 0.1.1

- Added `--seed` option for deterministic output.

## 0.1.0

Initial release. Core agent loop with tool-use, LM Studio and HuggingFace
providers, file read/write/edit, grep, list_files, run_command, thinking tool,
skills system, and REPL mode.
