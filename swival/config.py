"""Configuration file loading and merging for swival.

Reads TOML config from ~/.config/swival/config.toml (global) and
<base_dir>/swival.toml (project). Precedence: CLI > project > global > defaults.
"""

import argparse
import os
import re
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any


from .report import ConfigError  # noqa: F401 — re-export for convenience

_UNSET = object()  # Sentinel for "not set by CLI"


# --- Schema ---

CONFIG_KEYS: dict[str, type | tuple[type, ...]] = {
    "provider": str,
    "model": str,
    "api_key": str,
    "base_url": str,
    "max_output_tokens": int,
    "max_context_tokens": int,
    "temperature": (int, float),
    "top_p": (int, float),
    "seed": int,
    "max_turns": int,
    "system_prompt": str,
    "no_system_prompt": bool,
    "allowed_commands": list,
    "yolo": bool,
    "allowed_dirs": list,
    "allowed_dirs_ro": list,
    "no_read_guard": bool,
    "no_instructions": bool,
    "no_skills": bool,
    "skills_dir": list,
    "no_history": bool,
    "color": bool,
    "quiet": bool,
    "reviewer": str,
    "review_prompt": str,
    "objective": str,
    "verify": str,
    "max_review_rounds": int,
    "proactive_summaries": bool,
}

_LIST_OF_STR_KEYS = {
    "allowed_commands",
    "allowed_dirs",
    "allowed_dirs_ro",
    "skills_dir",
}

# Config key -> argparse dest (only where they differ)
_CONFIG_TO_ARGPARSE: dict[str, str] = {
    "allowed_dirs": "add_dir",
    "allowed_dirs_ro": "add_dir_ro",
}

# Argparse dest -> hardcoded default
_ARGPARSE_DEFAULTS: dict[str, Any] = {
    "provider": "lmstudio",
    "model": None,
    "api_key": None,
    "base_url": None,
    "max_output_tokens": 32768,
    "max_context_tokens": None,
    "temperature": None,
    "top_p": 1.0,
    "seed": None,
    "max_turns": 100,
    "system_prompt": None,
    "no_system_prompt": False,
    "allowed_commands": None,
    "yolo": False,
    "add_dir": [],
    "add_dir_ro": [],
    "no_read_guard": False,
    "no_instructions": False,
    "no_skills": False,
    "skills_dir": [],
    "no_history": False,
    "color": False,
    "no_color": False,
    "quiet": False,
    "reviewer": None,
    "review_prompt": None,
    "objective": None,
    "verify": None,
    "max_review_rounds": 5,
    "proactive_summaries": False,
}


# --- Internal helpers ---


def global_config_dir() -> Path:
    """Return the global config directory, respecting XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "swival"
    return Path.home() / ".config" / "swival"


def _type_name(expected: type | tuple[type, ...]) -> str:
    """Format an expected type spec as a human-readable string."""
    if expected is list:
        return "list"
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _validate_config(config: dict, source: str) -> None:
    """Validate types and mutual exclusions in a parsed config dict.

    Raises ConfigError for type mismatches or invalid combinations.
    Prints warnings for unknown keys.
    """
    for key, value in config.items():
        if key not in CONFIG_KEYS:
            print(f"warning: {source}: unknown config key {key!r}", file=sys.stderr)
            continue

        expected = CONFIG_KEYS[key]
        # bool is a subclass of int in Python, so isinstance(True, int) is True.
        # Reject bools for non-bool fields explicitly.
        if isinstance(value, bool) and expected is not bool:
            raise ConfigError(
                f"{source}: {key!r} expected {_type_name(expected)}, got bool"
            )
        if not isinstance(value, expected):
            raise ConfigError(
                f"{source}: {key!r} expected {_type_name(expected)}, got {type(value).__name__}"
            )

        # Validate list element types
        if key in _LIST_OF_STR_KEYS:
            for i, elem in enumerate(value):
                if not isinstance(elem, str):
                    raise ConfigError(
                        f"{source}: {key}[{i}]: expected string, got {type(elem).__name__}"
                    )

    # Mutual exclusion: system_prompt + no_system_prompt
    if config.get("system_prompt") and config.get("no_system_prompt"):
        raise ConfigError(
            f"{source}: 'system_prompt' and 'no_system_prompt' are mutually exclusive"
        )


_PATH_LIKE = re.compile(r"^(?:[/~]|\.\.?/)")


def _resolve_reviewer_command(config: dict, config_dir: Path, source: str) -> None:
    """Shell-split the reviewer value, resolve only path-like first tokens."""
    try:
        parts = shlex.split(config["reviewer"])
    except ValueError as e:
        raise ConfigError(f"{source}: malformed reviewer command: {e}")
    if not parts:
        raise ConfigError(f"{source}: reviewer command is empty")
    exe = parts[0]
    if _PATH_LIKE.match(exe):
        expanded = Path(exe).expanduser()
        if expanded.is_absolute():
            parts[0] = str(expanded)
        else:
            parts[0] = str(config_dir / exe)
    config["reviewer"] = shlex.join(parts)


def _resolve_paths(config: dict, config_dir: Path, source: str = "") -> None:
    """Resolve relative paths in config against the config file's parent directory.

    Applies expanduser() before checking is_absolute(), so that ~/... paths
    expand to the user's home directory instead of becoming <config_dir>/~/...
    """
    for key in ("allowed_dirs", "allowed_dirs_ro", "skills_dir"):
        if key in config:
            resolved = []
            for p in config[key]:
                expanded = Path(p).expanduser()
                if expanded.is_absolute():
                    resolved.append(str(expanded))
                else:
                    resolved.append(str(config_dir / p))
            config[key] = resolved

    if "reviewer" in config:
        _resolve_reviewer_command(config, config_dir, source)

    for key in ("objective", "verify"):
        if key in config:
            p = Path(config[key]).expanduser()
            if p.is_absolute():
                config[key] = str(p)
            else:
                config[key] = str(config_dir / config[key])


def _check_api_key_in_git(config: dict, config_path: Path) -> None:
    """Warn if api_key is set in a project config inside a git repo."""
    if "api_key" not in config:
        return
    # Walk up from config file looking for .git
    parent = config_path.parent
    while parent != parent.parent:
        if (parent / ".git").exists():
            print(
                f"warning: {config_path}: 'api_key' in a git-tracked project config "
                f"may be committed accidentally. Consider using an environment variable.",
                file=sys.stderr,
            )
            return
        parent = parent.parent


def _load_single(path: Path, label: str) -> dict:
    """Load and validate a single TOML config file. Returns empty dict if missing."""
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{label}: invalid TOML: {e}") from e

    # Strip unknown keys after warning (keep only known ones for downstream)
    _validate_config(config, label)
    known = {k: v for k, v in config.items() if k in CONFIG_KEYS}
    return known


# --- Public API ---


def load_config(base_dir: Path) -> dict:
    """Load and merge global + project config.

    Returns a flat dict with config-canonical keys. Only keys that were
    actually set in config files are included (no defaults injected).
    Validates types and mutual exclusions. Resolves relative paths
    against each config file's parent directory.

    The returned dict also contains ``config_dir`` (a ``Path``) pointing
    to the resolved global config directory (e.g. ``~/.config/swival``).
    """
    # Global config
    config_dir = global_config_dir()
    global_path = config_dir / "config.toml"
    global_config = _load_single(global_path, str(global_path))
    if global_config:
        _resolve_paths(global_config, global_path.parent, str(global_path))

    # Project config
    project_path = Path(base_dir).resolve() / "swival.toml"
    project_config = _load_single(project_path, str(project_path))
    if project_config:
        _check_api_key_in_git(project_config, project_path)
        _resolve_paths(project_config, project_path.parent, str(project_path))

    # Merge: project overrides global (shallow)
    merged = {**global_config, **project_config}

    # Re-validate mutual exclusion on merged result (could conflict across files)
    if merged.get("system_prompt") and merged.get("no_system_prompt"):
        raise ConfigError(
            "'system_prompt' and 'no_system_prompt' are mutually exclusive "
            "(set across global and project config)"
        )

    # Attach resolved config directory so callers don't re-derive it.
    merged["config_dir"] = config_dir

    return merged


def apply_config_to_args(args: argparse.Namespace, config: dict) -> None:
    """Apply config values to argparse namespace where CLI didn't set a value.

    For each config key, maps to the argparse dest name and checks if
    the value is still _UNSET. If so, applies the config value. After
    processing all config keys, sweeps remaining _UNSET sentinels and
    replaces them with hardcoded defaults from _ARGPARSE_DEFAULTS.
    """
    # Dests that use None as sentinel (argparse append actions can't use _UNSET)
    _NONE_SENTINEL_DESTS = {"add_dir", "add_dir_ro", "skills_dir"}

    def _is_unset(dest: str) -> bool:
        val = getattr(args, dest, _UNSET)
        if dest in _NONE_SENTINEL_DESTS:
            return val is None
        return val is _UNSET

    # Special handling for color: single config key controls mutual-exclusive pair
    if "color" in config:
        color_val = config["color"]
        if _is_unset("color") and _is_unset("no_color"):
            args.color = color_val
            args.no_color = not color_val

    # Apply all other config keys
    for key, value in config.items():
        if key == "color":
            continue  # Already handled above

        dest = _CONFIG_TO_ARGPARSE.get(key, key)
        if _is_unset(dest):
            setattr(args, dest, value)

    # Sweep: replace remaining sentinels with hardcoded defaults
    for dest, default in _ARGPARSE_DEFAULTS.items():
        if _is_unset(dest):
            setattr(args, dest, default)


def config_to_session_kwargs(config: dict) -> dict:
    """Convert config dict to Session constructor kwargs.

    Translates config-canonical keys to Session's naming conventions:
    no_read_guard -> read_guard (inverted), no_history -> history (inverted),
    quiet -> verbose (inverted). Drops keys that aren't Session concerns
    (color, reviewer).
    """
    kwargs = {}
    _DROP_KEYS = {
        "color",
        "reviewer",
        "review_prompt",
        "objective",
        "verify",
        "max_review_rounds",
    }
    _INVERT_KEYS = {
        "no_read_guard": "read_guard",
        "no_history": "history",
        "quiet": "verbose",
    }

    for key, value in config.items():
        if key in _DROP_KEYS:
            continue
        if key in _INVERT_KEYS:
            kwargs[_INVERT_KEYS[key]] = not value
        else:
            kwargs[key] = value

    return kwargs


def generate_config(project: bool = False) -> str:
    """Return a commented-out template config string."""
    lines = [
        "# Swival configuration file",
        f"# {'Project' if project else 'Global'} config — "
        f"{'<project>/swival.toml' if project else '~/.config/swival/config.toml'}",
        "#",
        "# CLI flags override these values. Only uncomment what you need.",
        "",
        "# --- Provider / model ---",
        '# provider = "lmstudio"          # "lmstudio" | "huggingface" | "openrouter"',
        '# model = "qwen/qwen3-235b-a22b"',
        '# api_key = "sk-or-..."            # prefer env vars; this is a fallback',
        '# base_url = "https://..."',
        "",
        "# --- Generation parameters ---",
        "# max_output_tokens = 32768",
        "# max_context_tokens = 131072",
        "# temperature = 0.7",
        "# top_p = 1.0",
        "# seed = 42",
        "",
        "# --- Agent behaviour ---",
        "# max_turns = 50",
        '# system_prompt = "You are a helpful assistant."',
        "# no_system_prompt = false",
        "",
        "# --- Sandbox / security ---",
        '# allowed_commands = ["ls", "git", "python3"]',
        "# yolo = false",
        '# allowed_dirs = ["../shared-lib", "/data/assets"]',
        '# allowed_dirs_ro = ["/reference/docs", "~/datasets"]',
        "# no_read_guard = false",
        "",
        "# --- Features ---",
        "# no_instructions = false",
        "# no_skills = false",
        '# skills_dir = ["../my-skills"]',
        "# no_history = false",
        "",
        "# --- UI ---",
        "# color = true       # true = force color, false = force no-color, absent = auto",
        "# quiet = false",
        "",
        "# --- External ---",
        "# max_review_rounds = 5",
        '# reviewer = "./review.sh"',
        '# review_prompt = "Focus on correctness"',
        '# objective = "objective.md"',
        '# verify = "verification/working.md"',
        "",
    ]
    return "\n".join(lines)
