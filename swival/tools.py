"""Tool definitions and implementations for an LLM agent."""

import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file or list a directory. "
                "For files, returns lines prefixed with line numbers. "
                "Use offset/limit to paginate forward, or tail=N to start from the last N lines. "
                "If output is truncated, a continuation hint shows the offset for the next page. "
                "For directories, returns a listing with / suffix for subdirectories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file or directory to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "1-based line number to start reading from. Defaults to 1."
                        ),
                        "default": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to return. Defaults to 2000."
                        ),
                        "default": 2000,
                    },
                    "tail": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Return the last N lines of the file. "
                            "When set, offset is ignored. "
                            "To paginate within the tail, use the offset from the continuation hint "
                            "in a follow-up call (without tail)."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content, creating parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a targeted edit to an existing file by replacing old_string with new_string. "
                "For creating new files, use write_file instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences.",
                        "default": False,
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Recursively list files matching a glob pattern. "
                "Returns paths sorted by modification time (newest first), "
                "relative to the base directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            'Glob pattern to match files, e.g. "**/*.py", '
                            '"src/**/*.ts".'
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to search in, relative to base directory. "
                            'Defaults to "." (base directory).'
                        ),
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents for a regex pattern. "
                "Returns matches grouped by file with line numbers, "
                "sorted by file modification time (newest first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to search in, relative to base directory. "
                            'Defaults to "." (base directory).'
                        ),
                        "default": ".",
                    },
                    "include": {
                        "type": "string",
                        "description": (
                            'Glob pattern to filter filenames, e.g. "*.py".'
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Think step-by-step before acting. This is your scratchpad for reasoning — "
                "use it to plan, debug, weigh alternatives, or track what you've learned. "
                "Using think before complex actions leads to better outcomes. "
                "Pass a `note` to save important findings to disk — these persist even when "
                "older messages are dropped during context compaction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your current thinking step. Max 10000 characters.",
                    },
                    "thought_number": {
                        "type": "integer",
                        "description": (
                            "Current step number (1-based). "
                            "Optional — auto-increments if omitted."
                        ),
                        "minimum": 1,
                    },
                    "total_thoughts": {
                        "type": "integer",
                        "description": (
                            "Estimated total steps needed. "
                            "Optional — defaults to 3 on first call, then carries forward."
                        ),
                        "minimum": 1,
                    },
                    "next_thought_needed": {
                        "type": "boolean",
                        "description": (
                            "true if you need more thinking steps, false when done. "
                            "Optional — defaults to true."
                        ),
                    },
                    "is_revision": {
                        "type": "boolean",
                        "description": (
                            "Set true if this thought corrects or updates a previous one. "
                            "Must also set revises_thought."
                        ),
                    },
                    "revises_thought": {
                        "type": "integer",
                        "description": (
                            "Required when is_revision is true. "
                            "Which thought number is being revised."
                        ),
                        "minimum": 1,
                    },
                    "branch_from_thought": {
                        "type": "integer",
                        "description": (
                            "Thought number to branch from. "
                            "Must be set together with branch_id."
                        ),
                        "minimum": 1,
                    },
                    "branch_id": {
                        "type": "string",
                        "description": (
                            "Required when branch_from_thought is set. "
                            "Label for the branch (e.g. 'approach-b'). 1-50 characters."
                        ),
                        "minLength": 1,
                        "maxLength": 50,
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Optional. A concise summary to persist to disk for later retrieval. "
                            "Use this to save key findings, decisions, or context you'll need "
                            "after older messages are compacted. "
                            "Saved to .swival/notes.md — retrieve with read_file."
                        ),
                    },
                },
                "required": ["thought"],
            },
        },
    },
]

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the content of a URL and return it as markdown, plain text, or raw HTML. "
            "Use this to read documentation, APIs, web pages, or any HTTP-accessible resource."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must start with http:// or https://).",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "text", "html"],
                    "description": (
                        "Output format. 'markdown' converts HTML to readable markdown (default). "
                        "'text' extracts plain text. 'html' returns raw HTML."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (1-120, default 30).",
                },
            },
            "required": ["url"],
        },
    },
}

TOOLS.append(FETCH_URL_TOOL)

USE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "use_skill",
        "description": "Activate a skill to get detailed instructions for a specific task.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name to activate.",
                }
            },
            "required": ["name"],
        },
    },
}

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a command and return its output. Only whitelisted commands are allowed.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Command as an array of strings (NOT a single string). Each argument is a separate element. Correct: ["ls", "-la", "src/"]. Wrong: "ls -la src/".',
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-120). Defaults to 30.",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
}

MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB
MAX_LINE_LENGTH = 2000
BINARY_CHECK_BYTES = 8 * 1024  # 8 KB


def safe_resolve(
    file_path: str,
    base_dir: str,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> Path:
    """Resolve a file path, ensuring it stays within allowed roots.

    Resolves symlinks for both the base directory and the target path,
    then checks containment against base_dir first, then each extra_read_roots
    entry. extra_read_roots is only used for read operations.

    When unrestricted is True, resolves the path but skips containment checks.

    Raises:
        ValueError: If the resolved path escapes all allowed roots (when not unrestricted).
    """
    base = Path(base_dir).resolve()

    # For absolute paths, resolve directly; for relative, resolve against base_dir
    if Path(file_path).is_absolute():
        resolved = Path(file_path).resolve()
    else:
        resolved = (base / file_path).resolve()

    if unrestricted:
        # Even in unrestricted mode, block the filesystem root
        if resolved == Path(resolved.anchor):
            raise ValueError(
                f"Path {file_path!r} resolves to the filesystem root, "
                f"which is not allowed even in unrestricted mode"
            )
        return resolved

    # Check base_dir first
    if resolved.is_relative_to(base):
        return resolved

    # Check extra read roots
    for root in extra_read_roots:
        if resolved.is_relative_to(root):
            return resolved

    # Check extra write roots (--allow-dir paths, read+write access)
    for root in extra_write_roots:
        if resolved.is_relative_to(root):
            return resolved

    raise ValueError(
        f"Path {file_path!r} resolves to {resolved}, "
        f"which is outside base directory {base}"
    )


MAX_LIST_RESULTS = 100
MAX_GREP_MATCHES = 100


def _check_pattern(pattern: str) -> str | None:
    """Reject patterns that are absolute or contain '..'."""
    if PurePosixPath(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute():
        return f"error: pattern {pattern!r} must be relative, not absolute"
    # Check both POSIX and Windows path splitting so that both
    # "../foo" and "..\\foo" are caught.
    posix_parts = PurePosixPath(pattern).parts
    win_parts = PureWindowsPath(pattern).parts
    if ".." in posix_parts or ".." in win_parts:
        return f"error: pattern {pattern!r} contains '..', which is not allowed"
    return None


def _is_within_base(
    path: Path,
    base: Path,
    unrestricted: bool = False,
    extra_write_roots: list[Path] = (),
) -> bool:
    """Check that a resolved path is within the base directory or extra write roots."""
    if unrestricted:
        return True
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    if resolved.is_relative_to(base.resolve()):
        return True
    # extra_write_roots are already resolved at startup, no need to resolve again
    for root in extra_write_roots:
        if resolved.is_relative_to(root):
            return True
    return False


def _list_files(
    pattern: str,
    path: str,
    base_dir: str,
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> str:
    """Recursively list files matching a glob pattern."""
    err = _check_pattern(pattern)
    if err:
        return err

    try:
        root = safe_resolve(
            path,
            base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=unrestricted,
        )
    except ValueError as exc:
        return f"error: {exc}"

    if not root.exists():
        return f"error: path does not exist: {path}"
    if not root.is_dir():
        return f"error: path is not a directory: {path}"

    base = Path(base_dir).resolve()

    # Walk the tree, pruning .git directories
    matched: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for filename in files:
            filepath = Path(dirpath) / filename
            # Match the relative path (from root) against the glob pattern
            rel_to_root = filepath.relative_to(root)
            if not PurePath(rel_to_root).full_match(pattern):
                continue
            # Per-match containment check
            if not _is_within_base(
                filepath,
                base,
                unrestricted=unrestricted,
                extra_write_roots=extra_write_roots,
            ):
                continue
            matched.append(filepath)

    if not matched:
        return "No files matched the pattern."

    # Sort by modification time, newest first
    matched.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Cap results and format output
    truncated = len(matched) > MAX_LIST_RESULTS
    matched = matched[:MAX_LIST_RESULTS]

    output_parts: list[str] = []
    total_bytes = 0
    byte_truncated = False
    for filepath in matched:
        try:
            rel = str(filepath.relative_to(base))
        except ValueError:
            rel = str(filepath)
        encoded_len = len(rel.encode("utf-8")) + 1
        if total_bytes + encoded_len > MAX_OUTPUT_BYTES:
            byte_truncated = True
            break
        output_parts.append(rel)
        total_bytes += encoded_len

    result = "\n".join(output_parts)
    if truncated or byte_truncated:
        result += (
            "\n(Results truncated: showing first 100 results. "
            "Use a more specific pattern or path.)"
        )
    return result


def _grep(
    pattern: str,
    path: str,
    base_dir: str,
    include: str | None = None,
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> str:
    """Search file contents for a regex pattern."""
    # Validate include pattern
    if include is not None:
        err = _check_pattern(include)
        if err:
            return err

    # Compile regex
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid regex {pattern!r}: {exc}"

    try:
        root = safe_resolve(
            path,
            base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=unrestricted,
        )
    except ValueError as exc:
        return f"error: {exc}"

    if not root.exists():
        return f"error: path does not exist: {path}"
    if not root.is_dir():
        return f"error: path is not a directory: {path}"

    base = Path(base_dir).resolve()

    # Walk the tree, pruning .git directories
    # Collect ALL matches as (file_path, line_no, line_text, mtime),
    # then sort and cap — so the cap always picks the newest files.
    matches: list[tuple[Path, int, str, float]] = []

    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for filename in files:
            # Filter by include pattern
            if include and not fnmatch.fnmatch(filename, include):
                continue

            filepath = Path(dirpath) / filename

            # Per-file containment check
            if not _is_within_base(
                filepath,
                base,
                unrestricted=unrestricted,
                extra_write_roots=extra_write_roots,
            ):
                continue

            # Skip binary files
            try:
                with open(filepath, "rb") as f:
                    chunk = f.read(BINARY_CHECK_BYTES)
            except (PermissionError, OSError):
                continue
            if b"\x00" in chunk:
                continue

            # Read and search
            try:
                text = filepath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            mtime = filepath.stat().st_mtime
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append((filepath, line_no, line, mtime))

    if not matches:
        return "No matches found."

    # Sort by file mtime (newest first), then by line number within each file
    matches.sort(key=lambda m: (-m[3], m[0], m[1]))

    # Cap after sorting so the top-100 are truly the newest
    total_found = len(matches)
    truncated = total_found > MAX_GREP_MATCHES
    matches = matches[:MAX_GREP_MATCHES]

    from collections import OrderedDict

    grouped: OrderedDict[Path, list[tuple[int, str]]] = OrderedDict()
    for filepath, line_no, line_text, _ in matches:
        grouped.setdefault(filepath, []).append((line_no, line_text))

    output_parts: list[str] = []
    total_bytes = 0
    byte_truncated = False

    header = f"Found {total_found} matches"
    output_parts.append(header)
    total_bytes += len(header.encode("utf-8")) + 1

    for filepath, file_matches in grouped.items():
        try:
            rel = str(filepath.relative_to(base))
        except ValueError:
            rel = str(filepath)
        file_header = f"\n{rel}:"
        encoded_len = len(file_header.encode("utf-8")) + 1
        if total_bytes + encoded_len > MAX_OUTPUT_BYTES:
            byte_truncated = True
            break
        output_parts.append(file_header)
        total_bytes += encoded_len

        for line_no, line_text in file_matches:
            if len(line_text) > MAX_LINE_LENGTH:
                line_text = line_text[:MAX_LINE_LENGTH]
            entry = f"  Line {line_no}: {line_text}"
            encoded_len = len(entry.encode("utf-8")) + 1
            if total_bytes + encoded_len > MAX_OUTPUT_BYTES:
                byte_truncated = True
                break
            output_parts.append(entry)
            total_bytes += encoded_len
        if byte_truncated:
            break

    result = "\n".join(output_parts)
    if truncated or byte_truncated:
        result += (
            "\n(Results truncated: showing first 100 matches. "
            "Use a more specific pattern or path.)"
        )
    return result


def _read_file(
    file_path: str,
    base_dir: str,
    offset: int = 1,
    limit: int = 2000,
    tail: int | None = None,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> str:
    """Read a file or list a directory."""
    try:
        resolved = safe_resolve(
            file_path,
            base_dir,
            extra_read_roots=extra_read_roots,
            extra_write_roots=extra_write_roots,
            unrestricted=unrestricted,
        )
    except ValueError as exc:
        return f"error: {exc}"

    if not resolved.exists():
        return f"error: path does not exist: {file_path}"

    # Directory listing
    if resolved.is_dir():
        output_parts = []
        total_bytes = 0
        truncated = False
        try:
            for child in sorted(resolved.iterdir()):
                name = child.name + ("/" if child.is_dir() else "")
                encoded_len = len(name.encode("utf-8")) + 1  # +1 for newline
                if total_bytes + encoded_len > MAX_OUTPUT_BYTES:
                    truncated = True
                    break
                output_parts.append(name)
                total_bytes += encoded_len
        except PermissionError as exc:
            return f"error: {exc}"
        result = "\n".join(output_parts)
        if truncated:
            result += "\n[truncated at 50KB]"
        return result

    # Binary detection: check first 8 KB for null bytes
    try:
        with open(resolved, "rb") as f:
            chunk = f.read(BINARY_CHECK_BYTES)
    except PermissionError as exc:
        return f"error: {exc}"

    if b"\x00" in chunk:
        return f"error: binary file detected: {file_path}"

    # Read as UTF-8 text
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return f"error: failed to decode {file_path} as UTF-8: {exc}"
    except PermissionError as exc:
        return f"error: {exc}"

    lines = text.splitlines()

    # Apply tail or offset (1-based) and limit
    if tail is not None:
        if not isinstance(tail, int):
            return f"error: tail must be an integer, got {type(tail).__name__}"
        tail = max(tail, 1)
        start = max(len(lines) - tail, 0)
    else:
        start = max(offset - 1, 0)
    end = start + limit
    selected = lines[start:end]

    # Build output with line numbers, truncating long lines
    output_parts = []
    total_bytes = 0
    lines_emitted = 0

    for i, line in enumerate(selected, start=start + 1):
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH]
        numbered = f"{i}: {line}"
        encoded_len = len(numbered.encode("utf-8")) + 1  # +1 for newline
        if total_bytes + encoded_len > MAX_OUTPUT_BYTES:
            break
        output_parts.append(numbered)
        total_bytes += encoded_len
        lines_emitted += 1

    total_lines = len(lines)
    remaining = total_lines - (start + lines_emitted)

    result = "\n".join(output_parts)
    if remaining > 0:
        next_offset = start + lines_emitted + 1  # 1-based
        result += f"\n[{remaining} more lines, use offset={next_offset} to continue]"
    return result


def _write_file(
    file_path: str,
    content: str,
    base_dir: str,
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> str:
    """Create or overwrite a file with content."""
    try:
        resolved = safe_resolve(
            file_path,
            base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=unrestricted,
        )
    except ValueError as exc:
        return f"error: {exc}"

    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    resolved.write_bytes(data)
    return f"Wrote {len(data)} bytes to {file_path}"


def _edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    base_dir: str,
    replace_all: bool = False,
    extra_write_roots: list[Path] = (),
    unrestricted: bool = False,
) -> str:
    """Replace old_string with new_string in an existing file."""
    from .edit import replace

    try:
        resolved = safe_resolve(
            file_path,
            base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=unrestricted,
        )
    except ValueError as exc:
        return f"error: {exc}"

    if not resolved.exists():
        return f"error: file does not exist: {file_path}"

    if not old_string:
        return "error: old_string must not be empty"

    try:
        content = resolved.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError, OSError) as exc:
        return f"error: {exc}"

    try:
        new_content = replace(content, old_string, new_string, replace_all=replace_all)
    except ValueError as exc:
        return f"error: {exc}"

    resolved.write_text(new_content, encoding="utf-8")
    return f"Edited {file_path}"


MAX_INLINE_OUTPUT = 10 * 1024  # 10KB — max output returned inline
MAX_FILE_OUTPUT = 1 * 1024 * 1024  # 1MB — max output saved to file
SWIVAL_DIR = ".swival"
OUTPUT_FILE_TTL = 600  # seconds before temp file cleanup
MAX_TIMEOUT = 120


def cleanup_old_cmd_outputs(base_dir: str) -> int:
    """Remove cmd_output_* files older than OUTPUT_FILE_TTL from .swival/.

    Returns the number of files removed.
    """
    import time

    scratch = Path(base_dir) / SWIVAL_DIR
    if not scratch.is_dir():
        return 0
    cutoff = time.time() - OUTPUT_FILE_TTL
    removed = 0
    for f in scratch.glob("cmd_output_*.txt"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


_KILL_WAIT_TIMEOUT = 5  # seconds to wait for process to die after kill signals


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its descendants, then wait for exit.

    On Unix, uses process groups (via start_new_session=True) to kill the
    entire tree. On Windows, uses taskkill /T /F to kill the process tree.
    """
    if sys.platform != "win32":
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass  # already exited
    else:
        # taskkill /T kills the entire process tree rooted at the PID
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass  # best-effort
    try:
        proc.kill()
    except OSError:
        pass  # already dead
    try:
        proc.wait(timeout=_KILL_WAIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass  # give up — process is unkillable


def _save_large_output(output: str, base_dir: str, was_truncated: bool = False) -> str:
    """Save large command output to a temp file and return a summary message.

    Creates .swival/ inside base_dir, writes output there, and schedules
    cleanup after OUTPUT_FILE_TTL seconds.  Falls back to inline-truncated
    output on disk write failure.
    """
    size_bytes = len(output.encode("utf-8"))
    size_kb = size_bytes / 1024

    scratch = Path(base_dir) / SWIVAL_DIR
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Can't create dir — fall back to truncated inline
        truncated = output.encode("utf-8")[:MAX_INLINE_OUTPUT].decode(
            "utf-8", errors="replace"
        )
        return truncated + "\n[output truncated — failed to create .swival/ directory]"

    filename = f"cmd_output_{uuid.uuid4().hex[:12]}.txt"
    filepath = scratch / filename
    rel_path = f"{SWIVAL_DIR}/{filename}"

    try:
        filepath.write_text(output, encoding="utf-8")
    except OSError:
        truncated = output.encode("utf-8")[:MAX_INLINE_OUTPUT].decode(
            "utf-8", errors="replace"
        )
        return truncated + "\n[output truncated — failed to write temp file]"

    def _cleanup():
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass

    timer = threading.Timer(OUTPUT_FILE_TTL, _cleanup)
    timer.daemon = True
    timer.start()

    saved_label = (
        "Output (possibly truncated) saved to"
        if was_truncated
        else "Full output saved to"
    )

    return (
        f"Command output too large for context ({size_kb:.1f}KB).\n"
        f"{saved_label}: {rel_path}\n"
        f"Use read_file to examine the output (supports offset and limit for pagination)."
    )


def _capture_process(proc: subprocess.Popen, timeout: int, base_dir: str) -> str:
    """Capture output from a running subprocess with timeout enforcement."""
    output_chunks: list[bytes] = []
    output_total = 0
    output_truncated = False

    def _reader():
        nonlocal output_total, output_truncated
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                if output_truncated:
                    continue  # keep draining to prevent pipe backpressure
                remaining = MAX_FILE_OUTPUT - output_total
                output_chunks.append(chunk[:remaining])
                output_total += len(output_chunks[-1])
                if output_total >= MAX_FILE_OUTPUT:
                    output_truncated = True
        except (OSError, ValueError):
            pass  # pipe closed/broken after kill

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)

    reader_thread.join(timeout=2)
    proc.stdout.close()

    # Build result
    raw_output = b"".join(output_chunks).decode("utf-8", errors="replace")
    parts: list[str] = []

    if timed_out:
        parts.append(f"error: command timed out after {timeout}s")
    elif proc.returncode != 0:
        parts.append(f"Exit code: {proc.returncode}")

    if raw_output:
        parts.append(raw_output)

    if output_truncated:
        parts.append("[output truncated at 1MB]")

    result = "\n".join(parts) if parts else "(no output)"

    # Save large output to file instead of stuffing the context
    if len(result.encode("utf-8")) > MAX_INLINE_OUTPUT:
        exit_info = ""
        if timed_out:
            exit_info = f"\nerror: command timed out after {timeout}s"
        elif proc.returncode != 0:
            exit_info = f"\nExit code: {proc.returncode}"
        saved = _save_large_output(result, base_dir, was_truncated=output_truncated)
        result = saved + exit_info

    return result


def _run_shell_command(command: str, base_dir: str, timeout: int) -> str:
    """Execute a shell string via sh -c (Unix) or cmd.exe /c (Windows)."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return f"error: base directory does not exist: {base_dir}"
    if not base_path.is_dir():
        return f"error: base directory is not a directory: {base_dir}"

    timeout = max(1, min(timeout, MAX_TIMEOUT))

    if sys.platform == "win32":
        shell_cmd = ["cmd.exe", "/c", command]
    else:
        shell_cmd = ["/bin/sh", "-c", command]

    try:
        popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=base_dir,
        )
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(shell_cmd, **popen_kwargs)
    except OSError as e:
        return f"error: failed to start shell command: {e}"

    return _capture_process(proc, timeout, base_dir)


def _run_command(
    command: list[str] | str,
    base_dir: str,
    resolved_commands: dict[str, str],
    timeout: int = 30,
    unrestricted: bool = False,
) -> str:
    """Execute a command and return its output.

    When unrestricted is False, only whitelisted commands are allowed.
    When unrestricted is True, any command can be run.
    """
    import shutil as _shutil

    was_repaired = False

    def _finalize(result: str, repaired: bool) -> str:
        if not repaired:
            return result
        return (
            f"{result}\n"
            "(auto-corrected: command was passed as a string, converted to array)"
        )

    if isinstance(command, str):
        repaired_command = None
        try:
            parsed = json.loads(command)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                repaired_command = parsed
        except (json.JSONDecodeError, TypeError):
            pass

        if repaired_command is None:
            if unrestricted:
                return _run_shell_command(command, base_dir, timeout)
            return _finalize(
                'error: "command" must be a JSON array of strings, not a single string.\n'
                'Wrong: "command": "grep -n pattern file.py"\n'
                'Right: "command": ["grep", "-n", "pattern", "file.py"]\n'
                "Each argument must be a separate element in the array.\n"
                "Shell syntax (&&, |, >, 2>&1, etc.) is not supported — "
                "run one command at a time.",
                was_repaired,
            )
        command = repaired_command
        was_repaired = True

    if not command:
        return _finalize("error: command list is empty", was_repaired)

    base_path = Path(base_dir)
    if not base_path.exists():
        return _finalize(
            f"error: base directory does not exist: {base_dir}", was_repaired
        )
    if not base_path.is_dir():
        return _finalize(
            f"error: base directory is not a directory: {base_dir}", was_repaired
        )

    cmd_name = command[0]

    if unrestricted:
        # In unrestricted mode, resolve the command without whitelist checks
        if "/" in cmd_name or "\\" in cmd_name:
            # Absolute or relative path — resolve against base_dir
            candidate = Path(cmd_name)
            if not candidate.is_absolute():
                candidate = Path(base_dir) / candidate
            resolved_path = str(candidate.resolve())
        else:
            # Bare name — look up on PATH
            found = _shutil.which(cmd_name)
            if found is None:
                return _finalize(
                    f"error: command not found on PATH: {cmd_name!r}", was_repaired
                )
            resolved_path = str(Path(found).resolve())
    else:
        # Reject paths in command[0]
        if "/" in cmd_name or "\\" in cmd_name:
            allowed = ", ".join(sorted(resolved_commands)) or "(none)"
            return _finalize(
                f"error: command must be a bare name, not a path: {cmd_name!r}. "
                f"Allowed commands: {allowed}",
                was_repaired,
            )

        # Look up pinned path
        resolved_path = resolved_commands.get(cmd_name)
        if resolved_path is None:
            allowed = ", ".join(sorted(resolved_commands)) or "(none)"
            return _finalize(
                f"error: command {cmd_name!r} is not allowed. Allowed commands: {allowed}",
                was_repaired,
            )

    # Clamp timeout
    timeout = max(1, min(timeout, MAX_TIMEOUT))

    try:
        popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=base_dir,
        )
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen([resolved_path] + command[1:], **popen_kwargs)
    except FileNotFoundError:
        return _finalize(
            f"error: command executable not found: {resolved_path}", was_repaired
        )
    except PermissionError:
        return _finalize(
            f"error: permission denied executing: {resolved_path}", was_repaired
        )
    except OSError as e:
        return _finalize(f"error: failed to start command: {e}", was_repaired)

    return _finalize(_capture_process(proc, timeout, base_dir), was_repaired)


def dispatch(name: str, args: dict, base_dir: str, **kwargs) -> str:
    """Route a tool call to the appropriate implementation.

    Args:
        name: The tool name to invoke.
        args: Dictionary of arguments for the tool.
        base_dir: Base directory for path resolution.
        **kwargs: Extra context (e.g. thinking_state for the think tool).

    Returns:
        String result from the tool.

    Raises:
        KeyError: If the tool name is not recognized.
    """
    yolo = kwargs.get("yolo", False)
    extra_write_roots = kwargs.get("extra_write_roots", ())

    if name == "think":
        thinking_state = kwargs.get("thinking_state")
        if thinking_state is None:
            return "error: think tool is not available"
        return thinking_state.process(args)
    elif name == "read_file":
        return _read_file(
            file_path=args["file_path"],
            base_dir=base_dir,
            offset=args.get("offset", 1),
            limit=args.get("limit", 2000),
            tail=args.get("tail"),
            extra_read_roots=kwargs.get("skill_read_roots", ()),
            extra_write_roots=extra_write_roots,
            unrestricted=yolo,
        )
    elif name == "write_file":
        return _write_file(
            file_path=args["file_path"],
            content=args["content"],
            base_dir=base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=yolo,
        )
    elif name == "edit_file":
        return _edit_file(
            file_path=args["file_path"],
            old_string=args["old_string"],
            new_string=args["new_string"],
            base_dir=base_dir,
            replace_all=args.get("replace_all", False),
            extra_write_roots=extra_write_roots,
            unrestricted=yolo,
        )
    elif name == "list_files":
        return _list_files(
            pattern=args["pattern"],
            path=args.get("path", "."),
            base_dir=base_dir,
            extra_write_roots=extra_write_roots,
            unrestricted=yolo,
        )
    elif name == "grep":
        return _grep(
            pattern=args["pattern"],
            path=args.get("path", "."),
            base_dir=base_dir,
            include=args.get("include"),
            extra_write_roots=extra_write_roots,
            unrestricted=yolo,
        )
    elif name == "fetch_url":
        from .fetch import fetch_url as _fetch_url

        return _fetch_url(
            url=args.get("url", ""),
            format=args.get("format", "markdown"),
            timeout=args.get("timeout", 30),
            base_dir=base_dir,
        )
    elif name == "use_skill":
        from .skills import activate_skill

        catalog = kwargs.get("skills_catalog", {})
        read_roots = kwargs.get("skill_read_roots", [])
        return activate_skill(args["name"], catalog, read_roots)
    elif name == "run_command":
        resolved = kwargs.get("resolved_commands", {})
        return _run_command(
            command=args["command"],
            base_dir=base_dir,
            resolved_commands=resolved,
            timeout=args.get("timeout", 30),
            unrestricted=yolo,
        )
    else:
        raise KeyError(f"Unknown tool: {name!r}")
