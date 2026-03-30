"""Structural code outline tool — returns declarations without bodies."""

import ast
import re
from pathlib import Path

from .tools import MAX_OUTPUT_BYTES, safe_resolve

_MAX_OUTLINE_FILES = 20


def outline(
    file_path: str,
    base_dir: str,
    depth: int = 2,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    files_mode: str = "some",
    **kwargs,
) -> str:
    try:
        depth = int(depth)
    except (ValueError, TypeError):
        return "error: depth must be an integer"

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
        return f"error: path is a directory: {file_path}"

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

    depth = max(1, min(depth, 3))

    if resolved.suffix == ".py":
        try:
            return _outline_python(text, depth)
        except SyntaxError:
            return _outline_heuristic(text, depth)

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
                f"{node.decorator_list[i].lineno:<5}{indent}@{_decorator_text(d)}\n"
                for i, d in enumerate(node.decorator_list)
            )
            bases = ", ".join(_expr_text(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            line = f"{node.lineno:<5}{indent}{sig}:"
            out.append(decorators + line if decorators else line)
            _walk_python(node.body, depth, level + 1, out)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = "".join(
                f"{d.lineno:<5}{indent}@{_decorator_text(d)}\n"
                for d in node.decorator_list
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
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    first_default = num_args - num_defaults

    for i, arg in enumerate(args.args):
        p = arg.arg
        if arg.annotation:
            p += f": {_expr_text(arg.annotation)}"
        if i >= first_default:
            p += "=..."
        parts.append(p)

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


def _decorator_text(node: ast.expr) -> str:
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
    default_depth: int = 2,
    extra_read_roots: list[Path] = (),
    extra_write_roots: list[Path] = (),
    files_mode: str = "some",
) -> str:
    """Outline multiple files. Returns structured sections matching read_multiple_files format."""
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
        if total_bytes + section_bytes > MAX_OUTPUT_BYTES and sections:
            return False
        sections.append(section)
        total_bytes += section_bytes
        return True

    for i, spec in enumerate(files):
        if isinstance(spec, str):
            spec = {"file_path": spec}
        if not isinstance(spec, dict):
            section = _build_outline_section(
                f"file {i + 1}",
                "error",
                f"error: expected object or string, got {type(spec).__name__}",
            )
            if _try_append(section):
                files_with_errors += 1
            else:
                skipped_files = len(files) - i
                break
            continue

        file_path = spec.get("file_path")
        title = file_path or f"file {i + 1}"
        if not file_path:
            section = _build_outline_section(title, "error", "error: missing file_path")
            if _try_append(section):
                files_with_errors += 1
            else:
                skipped_files = len(files) - i
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

        is_error = result.startswith("error:")
        section = _build_outline_section(
            title,
            "error" if is_error else "ok",
            result,
        )

        if _try_append(section):
            if is_error:
                files_with_errors += 1
            else:
                files_succeeded += 1
        else:
            skipped_files = len(files) - i
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
