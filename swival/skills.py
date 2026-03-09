"""Skill discovery and activation for SKILL.md-based agent skills."""

import re
from dataclasses import dataclass
from pathlib import Path

from . import fmt

MAX_SKILL_BODY_CHARS = 20_000
MAX_SKILL_DESCRIPTION_CHARS = 1024
MAX_SKILL_NAME_CHARS = 64

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


@dataclass
class SkillInfo:
    name: str  # validated name from frontmatter
    description: str  # description from frontmatter
    path: Path  # resolved absolute path to skill directory
    is_local: bool  # True if under base_dir (no allowlist entry needed)


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

        if key not in ("name", "description"):
            # Unknown key — skip it and any continuation lines
            i += 1
            while i < len(fm_lines) and fm_lines[i] and fm_lines[i][0] in (" ", "\t"):
                i += 1
            continue

        # Check for literal block scalar: key: |
        if raw_value == "|":
            # Collect indented block, preserving newlines
            block_lines = []
            i += 1
            while i < len(fm_lines) and fm_lines[i] and fm_lines[i][0] in (" ", "\t"):
                block_lines.append(fm_lines[i].strip())
                i += 1
            result[key] = "\n".join(block_lines)
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
    description = parsed["description"]

    # Validate name
    name_err = validate_skill_name(name, dir_name)
    if name_err:
        if verbose:
            fmt.warning(f"invalid skill in {entry}: {name_err}")
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

    catalog[name] = SkillInfo(
        name=name,
        description=description,
        path=resolved_path,
        is_local=skill_is_local,
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
    if _depth > 3:
        return
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            if (entry / "SKILL.md").is_file():
                _try_load_skill(entry, base_resolved, catalog, verbose)
            else:
                _scan_skills_dir(entry, base_resolved, catalog, verbose, _depth + 1)


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

    # Scan project-local skills first
    local_skills = base_resolved / ".swival" / "skills"
    if local_skills.is_dir():
        try:
            entries = sorted(local_skills.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_dir():
                _try_load_skill(entry, base_resolved, catalog, verbose)

    # Track already-scanned directories to avoid duplicate warnings
    scanned: set[Path] = set()
    if local_skills.is_dir():
        scanned.add(local_skills.resolve())

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

    if verbose and catalog:
        names = ", ".join(sorted(catalog))
        fmt.info(f"Discovered {len(catalog)} skill(s): {names}")

    return catalog


def activate_skill(
    name: str,
    catalog: dict[str, SkillInfo],
    read_roots: list[Path],
) -> str:
    """Load a skill's full instructions and update the read allowlist.

    Returns the formatted skill instructions or an error string.
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

    body = parsed["body"]

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
    parts.append(f"Skill directory: {skill.path}")
    parts.append(
        "To access supporting files, use read_file with absolute paths under this directory"
        f' (e.g. "{skill.path}/scripts/example.py").'
    )

    return "\n".join(parts)


def format_skill_catalog(catalog: dict[str, SkillInfo]) -> str:
    """Format the skill catalog for inclusion in the system prompt."""
    if not catalog:
        return ""

    lines = [
        "## Skills",
        "A skill is a set of local instructions stored in a `SKILL.md` file.",
        "### Available skills",
        "",
    ]
    for name in sorted(catalog):
        skill = catalog[name]
        if skill.is_local:
            path_str = str(skill.path / "SKILL.md")
            lines.append(f"- {name}: {skill.description} (file: {path_str})")
        else:
            lines.append(f"- {name}: {skill.description}")
    lines.append("")
    lines.append("### How to use skills")
    lines.append(
        "- **Trigger rules**: If the user mentions a skill with `$skill-name` in their message, "
        "the skill instructions are automatically injected. If the task clearly matches a "
        "skill's description above, you **must** use that skill by calling `use_skill` or "
        "reading the SKILL.md file directly.\n"
        "- **Progressive disclosure**: After deciding to use a skill, open its `SKILL.md` "
        "at the path listed above. Read only enough to follow the workflow.\n"
        "- **Supporting files**: If `SKILL.md` references relative paths (e.g. `scripts/foo.py`), "
        "resolve them relative to the skill directory.\n"
        "- **Multiple skills**: If multiple skills apply, use the minimal set that covers the "
        "request and state the order you'll use them.\n"
        "- **Missing/blocked**: If a named skill isn't in the list, say so briefly and continue "
        "with the best fallback."
    )
    return "\n".join(lines)


# =========================================================================
# $skill-name mention extraction
# =========================================================================

_MENTION_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9]")


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
        result = activate_skill(name, catalog, read_roots)
        results.append((name, result))

    return results
