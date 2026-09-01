"""Structural code outline tool — returns declarations without bodies."""

import ast
import bisect
import re
from dataclasses import dataclass
from pathlib import Path

from . import tools
from .codeparse import mask_noncode
from .tools import safe_resolve

_MAX_OUTLINE_FILES = 20


def outline(
    file_path: str,
    base_dir: str,
    depth: int | None = None,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    files_mode: str = "some",
    **kwargs,
) -> str:
    explicit_depth = depth is not None
    if explicit_depth:
        try:
            depth = int(depth)
        except (ValueError, TypeError):
            return "error: depth must be an integer"
        depth = max(1, min(depth, 3))

    try:
        resolved = safe_resolve(
            file_path,
            base_dir,
            extra_read_roots=extra_read_roots,
            extra_write_roots=extra_write_roots,
            files_mode=files_mode,
        )
    except ValueError as exc:
        return f"error: {exc}"

    if resolved.is_dir():
        return _outline_directory(
            resolved,
            requested_path=file_path,
            base_dir=base_dir,
            depth=depth if explicit_depth else 1,
            extra_read_roots=extra_read_roots,
            extra_write_roots=extra_write_roots,
            files_mode=files_mode,
        )

    try:
        raw = resolved.read_bytes()
    except FileNotFoundError:
        return f"error: file not found: {file_path}"
    except OSError as exc:
        return f"error: {exc}"

    if b"\x00" in raw[:8192]:
        return "error: binary file"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return "error: binary file"

    if not text.strip():
        return "empty file"

    depth = depth if explicit_depth else 2

    if resolved.suffix == ".py":
        try:
            return _outline_python(text, depth)
        except SyntaxError:
            pass

    return _outline_heuristic(text, depth)


def _outline_python(source: str, depth: int) -> str:
    tree = ast.parse(source)
    lines: list[str] = []
    _walk_python(tree.body, depth, 0, lines)
    return "\n".join(lines) if lines else "no declarations found"


def _walk_python(nodes: list[ast.stmt], depth: int, level: int, out: list[str]):
    if level >= depth:
        return
    indent = "    " * level
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            decorators = "".join(
                f"{d.lineno:<5}{indent}@{_expr_text(d)}\n" for d in node.decorator_list
            )
            bases = ", ".join(_expr_text(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            line = f"{node.lineno:<5}{indent}{sig}:"
            out.append(decorators + line if decorators else line)
            _walk_python(node.body, depth, level + 1, out)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = "".join(
                f"{d.lineno:<5}{indent}@{_expr_text(d)}\n" for d in node.decorator_list
            )
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sig = _func_signature(node)
            ret = f" -> {_expr_text(node.returns)}" if node.returns else ""
            line = f"{node.lineno:<5}{indent}{prefix} {node.name}({sig}){ret}"
            out.append(decorators + line if decorators else line)
            _walk_python(node.body, depth, level + 1, out)

        elif isinstance(node, ast.Assign) and level == 0:
            targets = ", ".join(_expr_text(t) for t in node.targets)
            out.append(f"{node.lineno:<5}{indent}{targets} = ...")

        elif isinstance(node, ast.AnnAssign) and level == 0:
            target = _expr_text(node.target)
            ann = _expr_text(node.annotation)
            if node.value is not None:
                out.append(f"{node.lineno:<5}{indent}{target}: {ann} = ...")
            else:
                out.append(f"{node.lineno:<5}{indent}{target}: {ann}")

        elif isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.With)):
            body = node.body if hasattr(node, "body") else []
            _walk_python(body, depth, level, out)
            for handler in getattr(node, "handlers", []):
                _walk_python(handler.body, depth, level, out)
            _walk_python(getattr(node, "orelse", []), depth, level, out)
            _walk_python(getattr(node, "finalbody", []), depth, level, out)


def _func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    # ``defaults`` covers positional-only and regular positional args together.
    positional = list(args.posonlyargs) + list(args.args)
    num_defaults = len(args.defaults)
    first_default = len(positional) - num_defaults

    for i, arg in enumerate(positional):
        p = arg.arg
        if arg.annotation:
            p += f": {_expr_text(arg.annotation)}"
        if i >= first_default:
            p += "=..."
        parts.append(p)
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        p = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            p += f": {_expr_text(args.vararg.annotation)}"
        parts.append(p)
    elif args.kwonlyargs:
        parts.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        p = arg.arg
        if arg.annotation:
            p += f": {_expr_text(arg.annotation)}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            p += "=..."
        parts.append(p)

    if args.kwarg:
        p = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            p += f": {_expr_text(args.kwarg.annotation)}"
        parts.append(p)

    return ", ".join(parts)


def _expr_text(node: ast.expr | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


_STRUCT_RE = re.compile(
    r"^(?:export\s+)?(?:default\s+)?"
    r"(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:abstract\s+)?(?:static\s+)?(?:async\s+)?"
    r"(?:inline\s+)?(?:comptime\s+)?"
    r"(?:public\s+|private\s+|protected\s+|internal\s+|fileprivate\s+)?"
    r"(?:static\s+)?(?:override\s+)?"
    r"(?:def|class|function|fn|func|type|typedef|struct|impl|interface|"
    r"module|package|namespace|protocol|"
    r"enum|trait|union|test|mod)\b"
)

_VALUE_RE = re.compile(
    r"^(?:export\s+)?(?:pub\s+)?"
    r"(?:const|var|let)\b"
)

_C_FUNC_RE = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?(?:extern\s+)?"
    r"(?:(?:unsigned|signed|long|short|const|volatile|struct|enum)\s+)*"
    r"(?:void|int|char|float|double|bool|size_t|ssize_t|u?int\d+_t"
    r"|[A-Z_]\w*(?:\s*\*)*)\s+"
    r"\*?(\w+)\s*\("
)


def _outline_heuristic(source: str, depth: int) -> str:
    lines_out: list[str] = []
    for lineno, raw_line in enumerate(source.splitlines(), 1):
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "<!--")):
            continue
        indent_chars = len(raw_line) - len(stripped)
        indent_level = indent_chars // 4 or (indent_chars // 2 if indent_chars else 0)
        if indent_level >= depth:
            continue
        is_structural = _STRUCT_RE.match(stripped)
        is_c_func = not is_structural and _C_FUNC_RE.match(stripped)
        is_value = not is_structural and not is_c_func and _VALUE_RE.match(stripped)
        if is_value and indent_level > 0:
            continue
        if is_structural or is_c_func or is_value:
            indent = "    " * indent_level
            display = stripped.rstrip()
            if len(display) > 120:
                display = display[:117] + "..."
            lines_out.append(f"{lineno:<5}{indent}{display}")
    return "\n".join(lines_out) if lines_out else "no declarations found"


# ---------------------------------------------------------------------------
# Symbol span extraction
# ---------------------------------------------------------------------------

_SPAN_MAX_LINES = 300

# How many lines past a declaration to look for a `{` or a top-level `;`
# before giving up on brace counting and falling back to indentation. Covers
# Allman-style braces and multi-line signatures without scanning a whole
# brace-less (Ruby-style) body character by character.
_BRACE_SEARCH_LINES = 5

_FUNC_KINDS = frozenset({"def", "function", "fn", "func", "fun", "sub"})

# Deliberately narrower than _STRUCT_RE above: span extraction needs a name
# capture and a kind classification, and skips nameless/system declarations
# (test, mod, package). Keep keyword additions to either grammar in sync.
_SPAN_DECL_RE = re.compile(
    r"^(?:(?:[\w()]+|\"[^\"]+\")\s+){0,6}?"
    r"(def|class|function|fn|func|fun|sub|type|typedef|struct|impl|interface|"
    r"module|namespace|protocol|enum|trait|union)\s+"
    r"(?:\w+\.)?(\w+)"
)

_VALUE_NAME_RE = re.compile(r"^(?:export\s+)?(?:pub\s+)?(?:const|var|let)\s+(\w+)")


@dataclass
class SymbolSpan:
    """Line span of one top-level symbol definition (1-based, inclusive).

    ``end`` is the true last line when known; ``render_end`` is where
    rendering should stop (equal to ``end`` unless the span hit
    ``_SPAN_MAX_LINES``, in which case ``truncated`` is ``"span-cap"``).
    """

    start: int
    end: int
    render_end: int
    kind: str  # "function" | "class" ("value" reserved for future constants)
    truncated: str = ""  # "" | "span-cap"

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "render_end": self.render_end,
            "kind": self.kind,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolSpan":
        start = int(d.get("start", 1))
        end = int(d.get("end", start))
        return cls(
            start=start,
            end=end,
            render_end=int(d.get("render_end", end)),
            kind=str(d.get("kind", "function")),
            truncated=str(d.get("truncated", "")),
        )


def _capped_span(start: int, end: int, kind: str) -> SymbolSpan:
    if end - start + 1 > _SPAN_MAX_LINES:
        return SymbolSpan(start, end, start + _SPAN_MAX_LINES - 1, kind, "span-cap")
    return SymbolSpan(start, end, end, kind)


def symbol_spans(content: str, file_path: str) -> dict[str, SymbolSpan]:
    """Map top-level symbol name -> :class:`SymbolSpan` for one file.

    v1 scope: top-level (module- or file-scope) functions and classes/types,
    plus top-level arrow/function-valued ``const``/``let``/``var`` bindings.
    Methods are not standalone keys; a class span covers its whole body.
    Underscore-prefixed names are kept (Python-internal helpers are real
    cross-file callees).
    """
    if not content.strip():
        return {}
    if file_path.endswith(".py"):
        try:
            return _python_symbol_spans(content)
        except SyntaxError:
            pass
    return _heuristic_symbol_spans(content)


def _python_symbol_spans(content: str) -> dict[str, SymbolSpan]:
    tree = ast.parse(content)
    spans: dict[str, SymbolSpan] = {}

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                if node.decorator_list:
                    start = min(start, min(d.lineno for d in node.decorator_list))
                end = node.end_lineno or start
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                spans.setdefault(node.name, _capped_span(start, end, kind))
            elif isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.With)):
                visit(getattr(node, "body", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))

    visit(tree.body)
    return spans


def _match_span_decl(line: str) -> tuple[str, str] | None:
    """Return ``(name, kind)`` when ``line`` opens a top-level definition."""
    m = _SPAN_DECL_RE.match(line)
    if m:
        keyword, name = m.group(1), m.group(2)
        return name, ("function" if keyword in _FUNC_KINDS else "class")
    m = _C_FUNC_RE.match(line)
    if m:
        return m.group(1), "function"
    m = _VALUE_NAME_RE.match(line)
    if m and ("=>" in line or "= function" in line or "= async" in line):
        return m.group(1), "function"
    return None


def _heuristic_symbol_spans(content: str) -> dict[str, SymbolSpan]:
    # Normalize CRLF and split on "\n" only so line offsets stay aligned
    # with ``masked`` (mask_noncode is length-preserving).
    masked = mask_noncode(content.replace("\r\n", "\n"))
    lines = masked.split("\n")
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    spans: dict[str, SymbolSpan] = {}
    for i, line in enumerate(lines):
        if not line or line[0] in " \t}":
            continue
        decl = _match_span_decl(line)
        if decl is None:
            continue
        name, kind = decl
        if name in spans:
            continue
        start = i + 1
        end = _find_span_end(masked, lines, offsets, i)
        if end is None:
            # Unbalanced braces: true end unknown, stop at the cap.
            capped = min(start + _SPAN_MAX_LINES - 1, len(lines))
            spans[name] = SymbolSpan(start, capped, capped, kind, "span-cap")
        else:
            spans[name] = _capped_span(start, end, kind)
    return spans


def _find_span_end(
    masked: str, lines: list[str], offsets: list[int], decl_idx: int
) -> int | None:
    """1-based end line of the definition opening at ``decl_idx`` (0-based).

    Brace counting on the masked text when a ``{`` (or a top-level ``;``)
    appears within ``_BRACE_SEARCH_LINES``; indentation fallback otherwise.
    Returns ``None`` when braces never balance (true end unknown).
    """
    n = len(lines)
    window_last = min(decl_idx + _BRACE_SEARCH_LINES, n) - 1
    window_end = offsets[window_last] + len(lines[window_last])

    depth = 0
    brace_seen = False
    i = offsets[decl_idx]
    while i < len(masked):
        c = masked[i]
        if c == "{":
            depth += 1
            brace_seen = True
        elif c == "}" and brace_seen:
            depth -= 1
            if depth <= 0:
                return bisect.bisect_right(offsets, i)
        elif c == ";" and not brace_seen:
            return bisect.bisect_right(offsets, i)
        elif c == "\n" and not brace_seen and i >= window_end:
            return _indent_fallback_end(lines, decl_idx)
        i += 1
    if brace_seen:
        return None
    return _indent_fallback_end(lines, decl_idx)


def _indent_fallback_end(lines: list[str], decl_idx: int) -> int:
    """Span end for brace-less bodies: last line indented past the declaration."""
    last_body = decl_idx
    j = decl_idx + 1
    while j < len(lines):
        line = lines[j]
        if line.strip():
            if line[0] not in " \t":
                # Ruby-style `end` closing a top-level body belongs to the span.
                if line.strip() == "end":
                    last_body = j
                break
            last_body = j
        j += 1
    return last_body + 1


_DIRECTORY_OUTLINE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".java",
        ".rb",
        ".zig",
        ".swift",
        ".kt",
        ".md",
        ".toml",
        ".sh",
        ".yaml",
        ".yml",
    }
)

_DIRECTORY_OUTLINE_NAMES = frozenset(
    {"Makefile", "Dockerfile", "README", "pyproject.toml", "Cargo.toml", "package.json"}
)

_DIRECTORY_OUTLINE_EXCLUDE = frozenset(
    {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "uv.lock"}
)

_DIRECTORY_OUTLINE_SKIP_DIRS = frozenset(
    {"__pycache__", "node_modules", ".git", "target", "dist", "build", ".venv"}
)

_DIRECTORY_OUTLINE_PRIORITY = frozenset(
    {
        "__init__.py",
        "mod.rs",
        "lib.rs",
        "readme.md",
        "pyproject.toml",
        "cargo.toml",
        "package.json",
    }
)


def _is_outline_source(name: str) -> bool:
    if name.startswith("."):
        return False
    if name in _DIRECTORY_OUTLINE_EXCLUDE:
        return False
    if name.endswith((".min.js", ".bundle.js")):
        return False
    if name in _DIRECTORY_OUTLINE_NAMES:
        return True
    return Path(name).suffix.lower() in _DIRECTORY_OUTLINE_SUFFIXES


def _outline_priority_key(name: str) -> tuple[int, str]:
    low = name.lower()
    is_priority = (
        low in _DIRECTORY_OUTLINE_PRIORITY
        or low.startswith("main.")
        or low.startswith("index.")
    )
    return (0 if is_priority else 1, low)


def _outline_directory(
    resolved: Path,
    requested_path: str,
    base_dir: str,
    depth: int,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    files_mode: str = "some",
) -> str:
    try:
        children = list(resolved.iterdir())
    except OSError as exc:
        return f"error: {exc}"

    subdirs: list[str] = []
    source_names: list[str] = []
    for c in children:
        if c.is_dir():
            if (
                not c.name.startswith(".")
                and c.name not in _DIRECTORY_OUTLINE_SKIP_DIRS
            ):
                subdirs.append(c.name)
        elif c.is_file() and _is_outline_source(c.name):
            source_names.append(c.name)
    subdirs.sort()
    source_names.sort(key=_outline_priority_key)

    try:
        dir_label = str(resolved.relative_to(Path(base_dir).resolve()))
    except ValueError:
        dir_label = requested_path.rstrip("/") or "."
    prefix = "" if dir_label == "." else f"{dir_label}/"
    dir_display = prefix or "."
    subdir_line = "subdirectories: " + " ".join(f"{d}/" for d in subdirs)

    if not source_names:
        if subdirs:
            return "\n".join(
                [
                    f"directory: {dir_display}",
                    f"source_files: 0 selected, {len(subdirs)} subdirectories",
                    subdir_line,
                ]
            )
        return f"directory: {dir_display}\nempty directory"

    selected = source_names[:_MAX_OUTLINE_FILES]
    omitted = source_names[_MAX_OUTLINE_FILES:]

    files = [{"file_path": f"{prefix}{name}"} for name in selected]
    body = outline_files(
        files=files,
        base_dir=base_dir,
        default_depth=depth,
        extra_read_roots=extra_read_roots,
        extra_write_roots=extra_write_roots,
        files_mode=files_mode,
    )

    summary = f"{len(selected)} selected"
    if omitted:
        summary += f", {len(omitted)} omitted (over {_MAX_OUTLINE_FILES}-file cap)"
    if subdirs:
        summary += f", {len(subdirs)} subdirectories"

    header = [f"directory: {dir_display}", f"source_files: {summary}"]
    if subdirs:
        header.append(subdir_line)
    if omitted:
        header.append("omitted_over_cap: " + " ".join(omitted))

    return "\n".join(header) + "\n\n" + body


def _build_outline_section(title: str, status: str, body: str) -> str:
    return "\n".join(
        [
            f"=== FILE: {title} ===",
            f"status: {status}",
            body,
        ]
    )


def outline_files(
    files: list[dict],
    base_dir: str,
    default_depth: int | None = None,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    files_mode: str = "some",
) -> str:
    """Outline multiple files using the same batch section envelope as read_multiple_files."""
    if not files:
        return "error: files list is empty"
    if len(files) > _MAX_OUTLINE_FILES:
        return f"error: too many files requested ({len(files)}), maximum is {_MAX_OUTLINE_FILES}"

    sections: list[str] = []
    files_succeeded = 0
    files_with_errors = 0
    skipped_files = 0
    total_bytes = 0

    def _try_append(section: str) -> bool:
        nonlocal total_bytes
        section_bytes = len(section.encode("utf-8")) + 2
        if total_bytes + section_bytes > tools.MAX_OUTPUT_BYTES and sections:
            return False
        sections.append(section)
        total_bytes += section_bytes
        return True

    def _record_section(title: str, status: str, body: str, index: int) -> bool:
        nonlocal files_succeeded, files_with_errors, skipped_files
        section = _build_outline_section(title, status, body)
        if not _try_append(section):
            skipped_files = len(files) - index
            return False
        if status == "error":
            files_with_errors += 1
        else:
            files_succeeded += 1
        return True

    for i, spec in enumerate(files):
        if isinstance(spec, str):
            spec = {"file_path": spec}
        if not isinstance(spec, dict):
            if not _record_section(
                f"file {i + 1}",
                "error",
                f"error: expected object or string, got {type(spec).__name__}",
                i,
            ):
                break
            continue

        file_path = spec.get("file_path")
        title = file_path or f"file {i + 1}"
        if not file_path:
            if not _record_section(title, "error", "error: missing file_path", i):
                break
            continue

        depth = spec.get("depth", default_depth)
        result = outline(
            file_path=file_path,
            base_dir=base_dir,
            depth=depth,
            extra_read_roots=extra_read_roots,
            extra_write_roots=extra_write_roots,
            files_mode=files_mode,
        )

        if not _record_section(
            title, "error" if result.startswith("error:") else "ok", result, i
        ):
            break

    header = "\n".join(
        [
            f"files_succeeded: {files_succeeded}",
            f"files_with_errors: {files_with_errors}",
            f"batch_truncated: {'true' if skipped_files > 0 else 'false'}",
        ]
    )
    output = header
    if sections:
        output += "\n\n" + "\n\n".join(sections)
    if skipped_files > 0:
        output += (
            f"\n\n[batch_truncated: {skipped_files} file(s) skipped due to size limit]"
        )
    return output
