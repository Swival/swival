"""Skill discovery and activation for SKILL.md-based agent skills."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import config, fmt

MAX_SKILL_BODY_CHARS = 20_000
MAX_SKILL_DESCRIPTION_CHARS = 1024
MAX_SKILL_NAME_CHARS = 64
MAX_METASKILL_FILE_BYTES = 65_536

_KNOWN_METASKILL_LANGUAGES = {"starlark"}

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_markdown_comments(text: str) -> str:
    """Remove HTML/markdown comments (<!-- ... -->) from text."""
    return _COMMENT_RE.sub("", text)


@dataclass
class SkillInfo:
    name: str  # validated name from frontmatter
    description: str  # description from frontmatter
    path: Path  # resolved absolute path to skill directory
    is_local: bool  # True if under base_dir (no allowlist entry needed)
    metaskill_path: Path | None = None  # resolved path to metaskill program file
    metaskill_language: str | None = None  # language identifier (e.g. "starlark")


def validate_skill_name(name: str, dir_name: str) -> str | None:
    """Validate a skill name. Returns error string or None if valid."""
    if not name:
        return "name is empty"
    if len(name) > MAX_SKILL_NAME_CHARS:
        return f"name exceeds {MAX_SKILL_NAME_CHARS} characters"
    if not _NAME_RE.match(name):
        return f"name {name!r} must be lowercase alphanumeric with hyphens, no leading/trailing/consecutive hyphens"
    if "--" in name:
        return f"name {name!r} contains consecutive hyphens"
    if name != dir_name:
        return f"name {name!r} does not match directory name {dir_name!r}"
    return None


def parse_frontmatter(text: str) -> dict | str:
    """Parse YAML frontmatter from SKILL.md content.

    Returns a dict with 'name', 'description', and 'body' keys on success,
    or an error string on failure.

    Supports:
    - Plain scalar values: key: value
    - Quoted scalar values: key: "value" or key: 'value'
    - Multiline folded: indented continuation lines joined with spaces
    - Multiline literal: key: | followed by indented block, newlines preserved
    """
    lines = text.split("\n")

    # Must start with ---
    if not lines or lines[0].strip() != "---":
        return "missing opening '---' delimiter"

    # Find closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return "missing closing '---' delimiter"

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).strip()

    result: dict = {"body": body}

    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Must be a key: value line (not indented)
        if line[0] in (" ", "\t"):
            # Indented line outside a key context — skip
            i += 1
            continue

        colon_idx = line.find(":")
        if colon_idx < 0:
            i += 1
            continue

        key = line[:colon_idx].strip()
        raw_value = line[colon_idx + 1 :].strip()

        if key not in ("name", "description", "metaskill", "metaskill_language"):
            # Unknown key — skip it and any continuation lines
            i += 1
            while i < len(fm_lines) and fm_lines[i] and fm_lines[i][0] in (" ", "\t"):
                i += 1
            continue

        # Block scalars: "|" keeps newlines, ">" folds them.
        if raw_value in ("|", "|-", ">", ">-"):
            block_lines = []
            i += 1
            while i < len(fm_lines) and fm_lines[i] and fm_lines[i][0] in (" ", "\t"):
                block_lines.append(fm_lines[i].strip())
                i += 1
            joiner = "\n" if raw_value[0] == "|" else " "
            result[key] = joiner.join(block_lines)
            continue

        # Check for quoted value
        if raw_value and raw_value[0] in ('"', "'"):
            quote_char = raw_value[0]
            inner = raw_value[1:]
            # Find closing quote, skipping escaped ones
            pos = 0
            close_idx = -1
            while pos < len(inner):
                if inner[pos] == "\\" and pos + 1 < len(inner):
                    pos += 2  # skip escaped char
                    continue
                if inner[pos] == quote_char:
                    close_idx = pos
                    break
                pos += 1
            if close_idx < 0:
                return f"missing closing {quote_char} for {key}"
            if close_idx != len(inner) - 1:
                return f"trailing content after closing {quote_char} for {key}"
            inner = inner[:close_idx]
            # Unescape inner quotes
            inner = inner.replace(f"\\{quote_char}", quote_char)
            result[key] = inner
            i += 1
            continue

        # Plain scalar — check for multiline folded (continuation lines)
        value = raw_value
        i += 1
        while i < len(fm_lines) and fm_lines[i] and fm_lines[i][0] in (" ", "\t"):
            value += " " + fm_lines[i].strip()
            i += 1
        result[key] = value
        continue

    # Validate required fields
    if "name" not in result:
        return "missing 'name' field"
    if "description" not in result:
        return "missing 'description' field"
    if not result["name"]:
        return "name is empty"
    if not result["description"]:
        return "description is empty"

    return result


def _try_load_skill(
    entry: Path,
    base_resolved: Path,
    catalog: dict[str, "SkillInfo"],
    verbose: bool,
) -> None:
    """Try to load a single skill directory into the catalog.

    Validates frontmatter, name, description. Logs warnings and skips on
    any error. Deduplicates: first-seen name wins.
    """
    skill_md = entry / "SKILL.md"
    if not skill_md.is_file():
        return

    dir_name = entry.name

    try:
        content = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        if verbose:
            fmt.warning(f"failed to read {skill_md}: {e}")
        return

    parsed = parse_frontmatter(content)
    if isinstance(parsed, str):
        if verbose:
            fmt.warning(f"failed to parse SKILL.md frontmatter in {entry}: {parsed}")
        return

    name = parsed["name"]
    description = strip_markdown_comments(parsed["description"]).strip()

    # Validate name
    name_err = validate_skill_name(name, dir_name)
    if name_err:
        if verbose:
            fmt.warning(f"invalid skill in {entry}: {name_err}")
        return

    if not description:
        if verbose:
            fmt.warning(
                f"skill {name!r} description is empty after stripping comments, skipping"
            )
        return

    # Validate description length
    if len(description) > MAX_SKILL_DESCRIPTION_CHARS:
        if verbose:
            fmt.warning(
                f"skill {name!r} description exceeds {MAX_SKILL_DESCRIPTION_CHARS} chars, skipping"
            )
        return

    # Deduplication
    if name in catalog:
        if verbose:
            existing = catalog[name]
            fmt.warning(
                f"skill {name!r} in {entry} ignored; already loaded from {existing.path}"
            )
        return

    resolved_path = entry.resolve()
    skill_is_local = resolved_path.is_relative_to(base_resolved)

    # Resolve metaskill program file
    metaskill_path: Path | None = None
    metaskill_language: str | None = None
    metaskill_field = parsed.get("metaskill")
    if metaskill_field:
        ms_path = resolved_path / metaskill_field
        if not ms_path.resolve().is_relative_to(resolved_path):
            if verbose:
                fmt.warning(
                    f"skill {name!r}: metaskill path escapes skill directory, skipping metaskill"
                )
        elif not ms_path.is_file():
            if verbose:
                fmt.warning(
                    f"skill {name!r}: metaskill file {metaskill_field!r} not found"
                )
        else:
            metaskill_path = ms_path.resolve()
    elif (resolved_path / "SKILL.star").is_file():
        metaskill_path = (resolved_path / "SKILL.star").resolve()

    if metaskill_path is not None:
        lang = parsed.get("metaskill_language", "starlark")
        if lang not in _KNOWN_METASKILL_LANGUAGES:
            if verbose:
                fmt.warning(
                    f"skill {name!r}: unknown metaskill language {lang!r}, skipping metaskill"
                )
            metaskill_path = None
        else:
            try:
                size = metaskill_path.stat().st_size
                if size > MAX_METASKILL_FILE_BYTES:
                    if verbose:
                        fmt.warning(
                            f"skill {name!r}: metaskill file exceeds {MAX_METASKILL_FILE_BYTES} bytes"
                        )
                    metaskill_path = None
                else:
                    metaskill_language = lang
            except OSError:
                metaskill_path = None

    catalog[name] = SkillInfo(
        name=name,
        description=description,
        path=resolved_path,
        is_local=skill_is_local,
        metaskill_path=metaskill_path,
        metaskill_language=metaskill_language,
    )


def _scan_skills_dir(
    directory: Path,
    base_resolved: Path,
    catalog: dict[str, "SkillInfo"],
    verbose: bool,
    _depth: int = 0,
) -> None:
    """Scan a directory for skills, recursing up to 3 levels deep.

    At each level, if a subdirectory contains SKILL.md it's loaded as a skill.
    Otherwise we recurse into it to find nested skills (e.g. plugins/<name>/skills/<skill>/).
    """
    if _depth >= 3:
        return
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir() and not entry.name.startswith("."):
            if (entry / "SKILL.md").is_file():
                _try_load_skill(entry, base_resolved, catalog, verbose)
            else:
                _scan_skills_dir(entry, base_resolved, catalog, verbose, _depth + 1)


def _global_skill_dirs() -> list[Path]:
    """Return global skill directories to scan (testable seam)."""
    return [
        config.global_config_dir() / "skills",  # ~/.config/swival/skills/
        Path.home() / ".agents" / "skills",  # ~/.agents/skills/
    ]


def project_skills_dir(base_dir: str | Path) -> Path:
    """Active project skill directory: <base_dir>/.swival/skills/."""
    return Path(base_dir) / ".swival" / "skills"


def global_skills_dir() -> Path:
    """Active global skill directory: ~/.config/swival/skills/."""
    return config.global_config_dir() / "skills"


def skills_library_dir() -> Path:
    """Staging library for downloaded skill collections.

    Lives at ~/.config/swival/library/skills/ and is NOT scanned by
    discover_skills — it holds collections fetched from URLs until the user
    installs individual skills into an active location.
    """
    return config.global_config_dir() / "library" / "skills"


def discover_skills(
    base_dir: str,
    extra_dirs: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, SkillInfo]:
    """Discover skills from base_dir/skills/ and optional extra directories.

    Returns a dict keyed by skill name. Project-local skills take precedence
    over extra_dirs skills. Among extra_dirs, first occurrence wins.

    Each --skills-dir path can be either:
    - A directory that directly contains a SKILL.md (a single skill)
    - A parent directory whose subdirectories contain SKILL.md files
    """
    catalog: dict[str, SkillInfo] = {}
    base_resolved = Path(base_dir).resolve()
    scanned: set[Path] = set()

    # Scan project-local skills first (.swival/skills/)
    local_skills = base_resolved / ".swival" / "skills"
    if local_skills.is_dir():
        try:
            entries = sorted(local_skills.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_dir():
                _try_load_skill(entry, base_resolved, catalog, verbose)
        scanned.add(local_skills.resolve())

    # Scan .agents/skills/ (common agent standard, lower precedence than .swival/skills/)
    agents_skills = base_resolved / ".agents" / "skills"
    if agents_skills.is_dir():
        try:
            entries = sorted(agents_skills.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_dir():
                _try_load_skill(entry, base_resolved, catalog, verbose)
        scanned.add(agents_skills.resolve())

    # Process each --skills-dir path
    for extra in extra_dirs or []:
        p = Path(extra).resolve()
        if not p.exists():
            if verbose:
                fmt.warning(f"skills directory does not exist: {extra}")
            continue
        if not p.is_dir():
            if verbose:
                fmt.warning(f"skills path is not a directory: {extra}")
            continue
        if p in scanned:
            continue
        scanned.add(p)

        # If the path itself contains a SKILL.md, treat it as a single skill
        if (p / "SKILL.md").is_file():
            _try_load_skill(p, base_resolved, catalog, verbose)
        else:
            # Otherwise scan its subdirectories (and recurse into skills/ subdirs)
            _scan_skills_dir(p, base_resolved, catalog, verbose)

    # Scan global skills directories (lowest precedence)
    for global_skills in _global_skill_dirs():
        if global_skills.is_dir():
            resolved = global_skills.resolve()
            if resolved not in scanned:
                _scan_skills_dir(resolved, base_resolved, catalog, verbose)
                scanned.add(resolved)

    if verbose and catalog:
        names = ", ".join(sorted(catalog))
        fmt.info(f"Discovered {len(catalog)} skill(s): {names}")

    return catalog


_MAX_LISTING_FILES = 50


def _list_skill_files(skill_path: Path) -> list[str]:
    """List non-SKILL.md files in a skill directory, returning absolute paths."""
    files: list[str] = []
    try:
        for dirpath, dirs, filenames in os.walk(skill_path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for fname in sorted(filenames):
                if fname == "SKILL.md" and Path(dirpath) == skill_path:
                    continue
                if fname.startswith("."):
                    continue
                files.append(str(Path(dirpath) / fname))
                if len(files) >= _MAX_LISTING_FILES:
                    return files
    except OSError:
        pass
    return files


def activate_skill(
    name: str,
    catalog: dict[str, SkillInfo],
    read_roots: list[Path],
    enabled_metaskills: set[str] | None = None,
) -> str:
    """Load a skill's full instructions and update the read allowlist.

    Returns the formatted skill instructions or an error string.
    The metaskill warning is only emitted when the skill's name is in
    *enabled_metaskills*.
    """
    skill = catalog.get(name)
    if skill is None:
        return f"error: unknown skill: {name!r}"

    skill_md = skill.path / "SKILL.md"
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as e:
        return f"error: failed to read {skill_md}: {e}"

    parsed = parse_frontmatter(content)
    if isinstance(parsed, str):
        return f"error: failed to parse SKILL.md: {parsed}"

    body = strip_markdown_comments(parsed["body"])

    # Cap body size
    truncated = False
    if len(body) > MAX_SKILL_BODY_CHARS:
        body = body[:MAX_SKILL_BODY_CHARS]
        truncated = True

    # Update read allowlist for external skills
    if not skill.is_local:
        if skill.path not in read_roots:
            read_roots.append(skill.path)

    parts = [
        f"[Skill: {name} activated]",
        "",
        "<skill-instructions>",
        body,
    ]
    if truncated:
        parts.append(f"\n[truncated at {MAX_SKILL_BODY_CHARS} characters]")
    parts.append("</skill-instructions>")
    parts.append("")
    if enabled_metaskills and name in enabled_metaskills and skill.metaskill_path:
        parts.append(
            f"IMPORTANT: This is an executable metaskill. Do NOT follow the instructions "
            f"above manually. Instead, call the `run_metaskill` tool with "
            f'name="{name}" and an `input` object containing the task and constraints. '
            f"The metaskill program handles retries, validation, and tracing automatically."
        )
        parts.append("")
    parts.append(f"Skill directory: {skill.path}")

    # List supporting files so the LLM knows what references are available
    file_listing = _list_skill_files(skill.path)
    if file_listing:
        parts.append("")
        parts.append("Supporting files in this skill directory:")
        for f in file_listing:
            parts.append(f"  {f}")
        parts.append("")
        parts.append(
            "To read these files, use read_file with the absolute paths shown above."
        )
    else:
        parts.append(
            "To access supporting files, use read_file with absolute paths under this directory"
            f' (e.g. "{skill.path}/scripts/example.py").'
        )

    return "\n".join(parts)


def format_skill_catalog(
    catalog: dict[str, SkillInfo],
    metaskill_names: list[str] | None = None,
) -> str:
    """Format the skill catalog for inclusion in the system prompt.

    ``metaskill_names`` controls which skills get the ``(metaskill: …)`` tag
    and whether the ``### Metaskills`` section is emitted:
    - ``None`` — infer from catalog (backward compat for direct callers).
    - A list (including ``[]``) — use exactly that list.
    """
    if not catalog:
        return ""

    if metaskill_names is None:
        enabled_metaskills = {s.name for s in catalog.values() if s.metaskill_path}
    else:
        enabled_metaskills = set(metaskill_names)

    has_local = any(s.is_local for s in catalog.values())
    has_non_local = any(not s.is_local for s in catalog.values())

    lines = [
        "## Skills",
        "A skill is a set of local instructions stored in a `SKILL.md` file.",
        "### Available skills",
        "",
    ]
    for name in sorted(catalog):
        skill = catalog[name]
        meta_tag = ""
        if name in enabled_metaskills and skill.metaskill_path:
            meta_tag = f" (metaskill: {skill.metaskill_language})"
        if skill.is_local:
            path_str = str(skill.path / "SKILL.md")
            lines.append(f"- {name}: {skill.description}{meta_tag} (file: {path_str})")
        else:
            lines.append(f"- {name}: {skill.description}{meta_tag}")
    lines.append("")
    lines.append("### How to use skills")
    bullets = [
        "- Call the `use_skill` tool with the skill name to activate it and receive "
        "detailed instructions."
    ]
    if has_local:
        bullets.append(
            "- For local skills (those showing a file path above), you may also read the "
            "SKILL.md file directly."
        )
    if has_non_local:
        bullets.append(
            "- For skills without a path shown, `use_skill` is the only way to access "
            "them — do not search for their files, they are outside the project directory."
        )
    bullets.append(
        "- If the user mentions a skill with `$skill-name`, it is activated automatically."
    )
    if len(catalog) > 1 and enabled_metaskills:
        bullets.append(
            "- If multiple skills apply, activate the minimal set that covers the request."
        )
    lines.append("\n".join(bullets))
    if enabled_metaskills:
        lines.append(
            "\n### Metaskills\n"
            "Skills marked `(metaskill: starlark)` have an executable workflow program. "
            "When a metaskill applies, you MUST call `run_metaskill` — do not attempt "
            "the task manually. The metaskill program handles retries, validation, and "
            "tracing automatically. Pass the user's task and any constraints as the "
            '`input` object (e.g. `{"task": "...", "heading": "..."}`).'
        )
    return "\n".join(lines)


# =========================================================================
# $skill-name mention extraction
# =========================================================================

_MENTION_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9]")

_SKILL_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def find_skill_prefix(text: str) -> str | None:
    """Return the partial skill name being typed at the end of *text*.

    Applies the same boundary rules as :func:`extract_skill_mentions`:
    ``$`` must be preceded by a non-alphanumeric character (or be at
    position 0) and the partial name must consist of characters matching
    :data:`_MENTION_NAME_RE` (``[a-z0-9-]``, first char not ``-``).

    Returns the partial name (without ``$``), or ``None`` if the cursor
    is not in a valid skill-mention position.  An empty string means
    ``$`` was typed with no name yet -- all skills should be offered.
    """
    dollar = text.rfind("$")
    if dollar == -1:
        return None
    if dollar > 0 and text[dollar - 1].isalnum():
        return None

    partial = text[dollar + 1 :]
    if not partial:
        return ""
    if partial[0] not in _SKILL_NAME_CHARS or partial[0] == "-":
        return None
    for ch in partial:
        if ch not in _SKILL_NAME_CHARS:
            return None
    return partial


def extract_skill_mentions(text: str, catalog: dict[str, SkillInfo]) -> list[str]:
    """Extract $skill-name mentions from text that match catalog entries.

    Returns a deduplicated list of skill names in order of first appearance.
    Only matches names that exist in the catalog.
    """
    if not catalog or "$" not in text:
        return []

    mentioned: list[str] = []
    seen: set[str] = set()

    i = 0
    while i < len(text):
        if text[i] != "$":
            i += 1
            continue

        # Check character before $ is a word boundary
        if i > 0 and text[i - 1].isalnum():
            i += 1
            continue

        m = _MENTION_NAME_RE.match(text, i + 1)
        if not m:
            i += 1
            continue

        name = m.group(0)
        end = m.end()

        # Ensure the character after the name is a boundary (not alphanumeric/hyphen)
        if end < len(text) and (text[end].isalnum() or text[end] == "-"):
            i += 1
            continue

        if name in catalog and name not in seen:
            seen.add(name)
            mentioned.append(name)

        i = end

    return mentioned


def inject_skill_mentions(
    text: str,
    catalog: dict[str, SkillInfo],
    read_roots: list[Path],
    enabled_metaskills: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Extract $skill mentions from text and activate each skill.

    Returns a list of (skill_name, activation_result) tuples, one per
    mentioned skill. Returns an empty list if no skills were mentioned.
    """
    names = extract_skill_mentions(text, catalog)
    if not names:
        return []

    results: list[tuple[str, str]] = []
    for name in names:
        result = activate_skill(
            name, catalog, read_roots, enabled_metaskills=enabled_metaskills
        )
        results.append((name, result))

    return results
