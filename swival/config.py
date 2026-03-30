"""Configuration file loading and merging for swival.

Reads TOML config from ~/.config/swival/config.toml (global) and
<base_dir>/swival.toml (project). Precedence: CLI > project > global > defaults.
"""

import argparse
import json
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

SANDBOX_MODES = ("builtin", "agentfs")
REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "default")

PROFILE_KEYS: set[str] = {
    "provider",
    "model",
    "api_key",
    "base_url",
    "aws_profile",
    "max_output_tokens",
    "max_context_tokens",
    "temperature",
    "top_p",
    "seed",
    "extra_body",
    "reasoning_effort",
    "sanitize_thinking",
}

CONFIG_KEYS: dict[str, type | tuple[type, ...]] = {
    "provider": str,
    "model": str,
    "api_key": str,
    "base_url": str,
    "aws_profile": str,
    "max_output_tokens": int,
    "max_context_tokens": int,
    "temperature": (int, float),
    "top_p": (int, float),
    "seed": int,
    "max_turns": int,
    "retries": int,
    "system_prompt": str,
    "no_system_prompt": bool,
    "files": str,
    "commands": (str, list),
    "yolo": bool,
    "allowed_dirs": list,
    "allowed_dirs_ro": list,
    "sandbox": str,
    "sandbox_session": str,
    "sandbox_strict_read": bool,
    "sandbox_auto_session": bool,
    "no_read_guard": bool,
    "no_instructions": bool,
    "no_skills": bool,
    "skills_dir": list,
    "no_history": bool,
    "no_memory": bool,
    "memory_full": bool,
    "no_continue": bool,
    "color": bool,
    "quiet": bool,
    "llm_filter": str,
    "reviewer": str,
    "self_review": bool,
    "review_prompt": str,
    "objective": str,
    "verify": str,
    "max_review_rounds": int,
    "proactive_summaries": bool,
    "no_mcp": bool,
    "no_a2a": bool,
    "extra_body": dict,
    "reasoning_effort": str,
    "cache": bool,
    "prompt_cache": bool,
    "sanitize_thinking": bool,
    "cache_dir": str,
    "serve_name": str,
    "serve_description": str,
    "encrypt_secrets": bool,
    "encrypt_secrets_key": str,
    "encrypt_secrets_tweak": str,
    "lifecycle_command": str,
    "lifecycle_timeout": int,
    "lifecycle_fail_closed": bool,
    "no_lifecycle": bool,
    "subagents": bool,
}

_LIST_OF_STR_KEYS = {
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
    "retries": 5,
    "system_prompt": None,
    "no_system_prompt": False,
    "files": "some",
    "commands": "all",
    "yolo": False,
    "add_dir": [],
    "add_dir_ro": [],
    "sandbox": "builtin",
    "sandbox_session": None,
    "sandbox_strict_read": False,
    "no_sandbox_auto_session": False,
    "no_read_guard": False,
    "no_instructions": False,
    "no_skills": False,
    "skills_dir": [],
    "no_history": False,
    "no_memory": False,
    "memory_full": False,
    "no_continue": False,
    "color": False,
    "no_color": False,
    "quiet": False,
    "llm_filter": None,
    "reviewer": None,
    "self_review": False,
    "review_prompt": None,
    "objective": None,
    "verify": None,
    "max_review_rounds": 15,
    "proactive_summaries": False,
    "no_mcp": False,
    "mcp_config": None,
    "no_a2a": False,
    "a2a_config": None,
    "extra_body": None,
    "reasoning_effort": None,
    "sanitize_thinking": False,
    "cache": False,
    "prompt_cache": True,
    "cache_dir": None,
    "serve_name": None,
    "serve_description": None,
    "encrypt_secrets": False,
    "no_encrypt_secrets": False,
    "encrypt_secrets_key": None,
    "encrypt_secrets_tweak": None,
    "lifecycle_command": None,
    "lifecycle_timeout": 300,
    "lifecycle_fail_closed": False,
    "no_lifecycle": False,
    "aws_profile": None,
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
        if key == "files":
            if value not in ("none", "some", "all"):
                raise ConfigError(
                    f"{source}: 'files' must be 'none', 'some', or 'all', got {value!r}"
                )
        if key == "commands":
            if isinstance(value, str) and value not in ("all", "none"):
                raise ConfigError(
                    f"{source}: 'commands' must be 'all', 'none', or a list of command names, "
                    f"got {value!r}"
                )
            if isinstance(value, list):
                for i, elem in enumerate(value):
                    if not isinstance(elem, str):
                        raise ConfigError(
                            f"{source}: commands[{i}]: expected string, got {type(elem).__name__}"
                        )

        # Validate list element types
        if key in _LIST_OF_STR_KEYS:
            for i, elem in enumerate(value):
                if not isinstance(elem, str):
                    raise ConfigError(
                        f"{source}: {key}[{i}]: expected string, got {type(elem).__name__}"
                    )

    # Validate sandbox enum value
    if "sandbox" in config and config["sandbox"] not in SANDBOX_MODES:
        raise ConfigError(
            f"{source}: 'sandbox' must be one of {SANDBOX_MODES!r}, "
            f"got {config['sandbox']!r}"
        )

    # Validate reasoning_effort enum value
    if (
        "reasoning_effort" in config
        and config["reasoning_effort"] not in REASONING_LEVELS
    ):
        raise ConfigError(
            f"{source}: 'reasoning_effort' must be one of {REASONING_LEVELS!r}, "
            f"got {config['reasoning_effort']!r}"
        )

    # Mutual exclusion: system_prompt + no_system_prompt
    if config.get("system_prompt") and config.get("no_system_prompt"):
        raise ConfigError(
            f"{source}: 'system_prompt' and 'no_system_prompt' are mutually exclusive"
        )


_PATH_LIKE = re.compile(r"^(?:[/~]|\.\.?/)")


def _resolve_command_string(
    value: str, config_dir: Path, source: str, label: str
) -> str:
    """Shell-split *value*, resolve path-like first tokens against *config_dir*."""
    try:
        parts = shlex.split(value)
    except ValueError as e:
        raise ConfigError(f"{source}: malformed {label}: {e}")
    if not parts:
        raise ConfigError(f"{source}: {label} is empty")
    exe = parts[0]
    if _PATH_LIKE.match(exe):
        expanded = Path(exe).expanduser()
        if expanded.is_absolute():
            parts[0] = str(expanded)
        else:
            parts[0] = str((config_dir / expanded).resolve())
    return shlex.join(parts)


def _resolve_config_command(
    config: dict, key: str, config_dir: Path, source: str
) -> None:
    """Shell-split a command config value, resolve only path-like first tokens."""
    config[key] = _resolve_command_string(
        config[key], config_dir, source, f"{key} command"
    )


def _resolve_command_model(config: dict, config_dir: Path, source: str) -> None:
    """Shell-split the model value when provider=command, resolve path-like first tokens."""
    if config.get("provider") != "command" or "model" not in config:
        return
    config["model"] = _resolve_command_string(
        config["model"], config_dir, source, "command model"
    )


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

    for cmd_key in ("llm_filter", "reviewer", "lifecycle_command"):
        if cmd_key in config:
            _resolve_config_command(config, cmd_key, config_dir, source)

    _resolve_command_model(config, config_dir, source)

    for key in ("objective", "verify", "cache_dir"):
        if key in config:
            p = Path(config[key]).expanduser()
            if p.is_absolute():
                config[key] = str(p)
            else:
                config[key] = str(config_dir / config[key])


def _validate_profiles(profiles: dict, source: str) -> None:
    """Validate the [profiles.*] tables from a config file.

    Each profile must be a table containing only PROFILE_KEYS.
    """
    if not isinstance(profiles, dict):
        raise ConfigError(f"{source}: 'profiles' must be a table of tables")
    for name, body in profiles.items():
        if not isinstance(body, dict):
            raise ConfigError(
                f"{source}: profiles.{name} must be a table, got {type(body).__name__}"
            )
        for key, value in body.items():
            if key not in PROFILE_KEYS:
                allowed = ", ".join(sorted(PROFILE_KEYS))
                raise ConfigError(
                    f"{source}: profiles.{name}: '{key}' is not allowed in a profile. "
                    f"Profiles only support LLM-related keys: {allowed}"
                )
            expected = CONFIG_KEYS[key]
            if isinstance(value, bool) and expected is not bool:
                raise ConfigError(
                    f"{source}: profiles.{name}.{key} expected "
                    f"{_type_name(expected)}, got bool"
                )
            if not isinstance(value, expected):
                raise ConfigError(
                    f"{source}: profiles.{name}.{key} expected "
                    f"{_type_name(expected)}, got {type(value).__name__}"
                )
        if (
            "reasoning_effort" in body
            and body["reasoning_effort"] not in REASONING_LEVELS
        ):
            raise ConfigError(
                f"{source}: profiles.{name}.reasoning_effort must be one of "
                f"{REASONING_LEVELS!r}, got {body['reasoning_effort']!r}"
            )


def _check_api_key_in_git(config: dict, config_path: Path) -> None:
    """Warn if api_key is set in a project config (or its profiles) inside a git repo."""
    has_top_level = "api_key" in config
    has_in_profile = any(
        "api_key" in body
        for body in config.get("profiles", {}).values()
        if isinstance(body, dict)
    )
    if not has_top_level and not has_in_profile:
        return
    # Walk up from config file looking for .git
    parent = config_path.parent
    while parent != parent.parent:
        if (parent / ".git").exists():
            where = "api_key"
            if has_in_profile and not has_top_level:
                where = "api_key in a profile"
            print(
                f"warning: {config_path}: '{where}' in a git-tracked project config "
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

    # Extract nested tables before flat-key validation
    mcp_servers = config.pop("mcp_servers", None)
    a2a_servers = config.pop("a2a_servers", None)
    serve_skills = config.pop("serve_skills", None)
    encrypt_patterns = config.pop("encrypt_secrets_patterns", None)
    profiles = config.pop("profiles", None)
    active_profile = config.pop("active_profile", None)

    # Strip unknown keys after warning (keep only known ones for downstream)
    _validate_config(config, label)
    known = {k: v for k, v in config.items() if k in CONFIG_KEYS}

    # Re-attach mcp_servers if present
    if mcp_servers is not None:
        if not isinstance(mcp_servers, dict):
            raise ConfigError(f"{label}: 'mcp_servers' must be a table")
        _validate_mcp_server_configs(mcp_servers, label)
        known["mcp_servers"] = mcp_servers

    # Re-attach a2a_servers if present
    if a2a_servers is not None:
        if not isinstance(a2a_servers, dict):
            raise ConfigError(f"{label}: 'a2a_servers' must be a table")
        _validate_a2a_server_configs(a2a_servers, label)
        known["a2a_servers"] = a2a_servers

    # Re-attach serve_skills if present
    if serve_skills is not None:
        if not isinstance(serve_skills, list):
            raise ConfigError(f"{label}: 'serve_skills' must be an array of tables")
        _validate_serve_skills(serve_skills, label)
        known["serve_skills"] = serve_skills

    # Re-attach encrypt_secrets_patterns if present
    if encrypt_patterns is not None:
        if not isinstance(encrypt_patterns, list):
            raise ConfigError(
                f"{label}: 'encrypt_secrets_patterns' must be an array of tables"
            )
        for i, pat in enumerate(encrypt_patterns):
            if not isinstance(pat, dict):
                raise ConfigError(
                    f"{label}: encrypt_secrets_patterns[{i}]: expected a table"
                )
            if "name" not in pat:
                raise ConfigError(
                    f"{label}: encrypt_secrets_patterns[{i}]: missing required key 'name'"
                )
        known["encrypt_secrets_patterns"] = encrypt_patterns

    # Re-attach profiles if present
    if profiles is not None:
        _validate_profiles(profiles, label)
        known["profiles"] = profiles

    # Re-attach active_profile if present
    if active_profile is not None:
        if not isinstance(active_profile, str):
            raise ConfigError(
                f"{label}: 'active_profile' must be a string, "
                f"got {type(active_profile).__name__}"
            )
        known["active_profile"] = active_profile

    return known


# --- MCP config helpers ---


_MCP_SERVER_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "command": str,
    "url": str,
    "args": list,
    "env": dict,
    "headers": dict,
}


def _validate_mcp_server_configs(servers: dict, source: str) -> None:
    """Validate structure and field types of MCP server configurations."""
    from .mcp_client import validate_server_name

    for name, cfg in servers.items():
        validate_server_name(name)
        if not isinstance(cfg, dict):
            raise ConfigError(f"{source}: mcp_servers.{name} must be a table")
        has_command = "command" in cfg
        has_url = "url" in cfg
        if not has_command and not has_url:
            raise ConfigError(
                f"{source}: mcp_servers.{name} must have 'command' or 'url'"
            )
        if has_command and has_url:
            raise ConfigError(
                f"{source}: mcp_servers.{name} cannot have both 'command' and 'url'"
            )

        # Validate field types
        prefix = f"{source}: mcp_servers.{name}"
        for field, expected in _MCP_SERVER_FIELD_TYPES.items():
            if field in cfg:
                if not isinstance(cfg[field], expected):
                    exp_name = (
                        expected.__name__
                        if isinstance(expected, type)
                        else " or ".join(t.__name__ for t in expected)
                    )
                    raise ConfigError(
                        f"{prefix}.{field}: expected {exp_name}, "
                        f"got {type(cfg[field]).__name__}"
                    )

        # Validate list element types
        if "args" in cfg:
            for i, elem in enumerate(cfg["args"]):
                if not isinstance(elem, str):
                    raise ConfigError(
                        f"{prefix}.args[{i}]: expected string, "
                        f"got {type(elem).__name__}"
                    )

        # Validate dict value types
        for dict_field in ("env", "headers"):
            if dict_field in cfg:
                for k, v in cfg[dict_field].items():
                    if not isinstance(v, str):
                        raise ConfigError(
                            f"{prefix}.{dict_field}.{k}: expected string, "
                            f"got {type(v).__name__}"
                        )


def load_mcp_json(path: Path) -> dict[str, dict]:
    """Load MCP server configs from an MCP JSON file.

    Returns a dict of server_name -> server_config.
    Raises ConfigError on invalid JSON or structure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"{path}: cannot read file: {e}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object at top level")

    servers_raw = data.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        raise ConfigError(f"{path}: 'mcpServers' must be a JSON object")

    _validate_mcp_server_configs(servers_raw, str(path))
    return servers_raw


def merge_mcp_configs(
    toml_servers: dict[str, dict] | None,
    json_servers: dict[str, dict] | None,
) -> dict[str, dict]:
    """Merge MCP server configs. TOML wins on name collision."""
    merged: dict[str, dict] = {}
    if json_servers:
        merged.update(json_servers)
    if toml_servers:
        merged.update(toml_servers)  # toml wins
    return merged


# --- A2A config helpers ---


_A2A_SERVER_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "url": str,
    "card_url": str,
    "auth_type": str,
    "auth_token": str,
    "timeout": (int, float),
}


def _validate_a2a_server_configs(servers: dict, source: str) -> None:
    """Validate structure and field types of A2A server configurations."""
    from .a2a_types import validate_server_name

    for name, cfg in servers.items():
        validate_server_name(name)
        if not isinstance(cfg, dict):
            raise ConfigError(f"{source}: a2a_servers.{name} must be a table")
        if "url" not in cfg:
            raise ConfigError(f"{source}: a2a_servers.{name} must have 'url'")

        prefix = f"{source}: a2a_servers.{name}"
        for field, expected in _A2A_SERVER_FIELD_TYPES.items():
            if field in cfg:
                if isinstance(cfg[field], bool) and expected is not bool:
                    raise ConfigError(
                        f"{prefix}.{field}: expected {_type_name(expected)}, got bool"
                    )
                if not isinstance(cfg[field], expected):
                    raise ConfigError(
                        f"{prefix}.{field}: expected {_type_name(expected)}, "
                        f"got {type(cfg[field]).__name__}"
                    )


def load_a2a_config(path: Path) -> dict[str, dict]:
    """Load A2A server configs from a TOML file.

    Expects [a2a_servers.*] tables. Returns a dict of name -> config.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: cannot read file: {e}")

    servers = data.get("a2a_servers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"{path}: 'a2a_servers' must be a table")

    _validate_a2a_server_configs(servers, str(path))
    return servers


# --- Serve skills validation ---


_SERVE_SKILL_KNOWN_KEYS = {"id", "name", "description", "examples"}


def _validate_serve_skills(skills: list, source: str) -> None:
    """Validate structure of serve_skills entries."""
    from .a2a_types import sanitize_skill_id

    seen_ids: set[str] = set()
    for i, skill in enumerate(skills):
        prefix = f"{source}: serve_skills[{i}]"
        if not isinstance(skill, dict):
            raise ConfigError(f"{prefix}: expected a table, got {type(skill).__name__}")

        # id is required
        if "id" not in skill:
            raise ConfigError(f"{prefix}: missing required key 'id'")

        skill_id = skill["id"]
        if not isinstance(skill_id, str):
            raise ConfigError(
                f"{prefix}.id: expected string, got {type(skill_id).__name__}"
            )

        # id must be stable under sanitization
        sanitized = sanitize_skill_id(skill_id)
        if sanitized != skill_id:
            raise ConfigError(
                f"{prefix}.id: {skill_id!r} is not a valid skill ID "
                f"(would be sanitized to {sanitized!r}). Use the sanitized form directly."
            )

        # id must be unique
        if skill_id in seen_ids:
            raise ConfigError(f"{prefix}.id: duplicate skill ID {skill_id!r}")
        seen_ids.add(skill_id)

        # Optional field types
        for key in ("name", "description"):
            if key in skill and not isinstance(skill[key], str):
                raise ConfigError(
                    f"{prefix}.{key}: expected string, got {type(skill[key]).__name__}"
                )

        if "examples" in skill:
            if not isinstance(skill["examples"], list):
                raise ConfigError(
                    f"{prefix}.examples: expected list, got {type(skill['examples']).__name__}"
                )
            for j, ex in enumerate(skill["examples"]):
                if not isinstance(ex, str):
                    raise ConfigError(
                        f"{prefix}.examples[{j}]: expected string, "
                        f"got {type(ex).__name__}"
                    )

        # Warn about unknown keys
        unknown = set(skill.keys()) - _SERVE_SKILL_KNOWN_KEYS
        if unknown:
            print(
                f"warning: {prefix}: unknown keys {unknown}",
                file=sys.stderr,
            )


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
    # Handle nested tables separately before flat merge
    global_mcp = global_config.pop("mcp_servers", None)
    project_mcp = project_config.pop("mcp_servers", None)

    # Handle profiles separately (per-key merge within same-name profiles)
    global_profiles = global_config.pop("profiles", None)
    project_profiles = project_config.pop("profiles", None)
    global_active_profile = global_config.pop("active_profile", None)
    project_active_profile = project_config.pop("active_profile", None)

    merged = {**global_config, **project_config}

    mcp_servers = merge_mcp_configs(project_mcp, global_mcp)
    if mcp_servers:
        merged["mcp_servers"] = mcp_servers

    # Handle a2a_servers separately (merge by server name, not overwrite)
    global_a2a = global_config.pop("a2a_servers", None)
    project_a2a = project_config.pop("a2a_servers", None)
    a2a_merged: dict[str, dict] = {}
    if global_a2a:
        a2a_merged.update(global_a2a)
    if project_a2a:
        a2a_merged.update(project_a2a)  # project wins
    if a2a_merged:
        merged["a2a_servers"] = a2a_merged

    # Handle serve_skills separately (project replaces global wholesale)
    global_serve_skills = global_config.pop("serve_skills", None)
    project_serve_skills = project_config.pop("serve_skills", None)
    serve_skills = (
        project_serve_skills
        if project_serve_skills is not None
        else global_serve_skills
    )
    if serve_skills is not None:
        merged["serve_skills"] = serve_skills
    else:
        merged.pop("serve_skills", None)  # remove stale value from shallow merge

    # Handle encrypt_secrets_patterns separately (project replaces global wholesale)
    global_enc_pats = global_config.pop("encrypt_secrets_patterns", None)
    project_enc_pats = project_config.pop("encrypt_secrets_patterns", None)
    enc_pats = project_enc_pats if project_enc_pats is not None else global_enc_pats
    if enc_pats is not None:
        merged["encrypt_secrets_patterns"] = enc_pats
    else:
        merged.pop("encrypt_secrets_patterns", None)

    # Merge profiles: per-key merge within same-name profiles (project wins)
    profiles_merged: dict[str, dict] = {}
    if global_profiles:
        for name, body in global_profiles.items():
            profiles_merged[name] = dict(body)
    if project_profiles:
        for name, body in project_profiles.items():
            if name in profiles_merged:
                profiles_merged[name].update(body)
            else:
                profiles_merged[name] = dict(body)
    if profiles_merged:
        for name, body in profiles_merged.items():
            if "provider" not in body:
                raise ConfigError(
                    f"profiles.{name}: 'provider' is required "
                    f"(after merging global and project config)"
                )
        merged["profiles"] = profiles_merged

    # active_profile: project wins over global
    active_profile = project_active_profile or global_active_profile
    if active_profile is not None:
        merged["active_profile"] = active_profile
        if project_active_profile:
            merged["_active_profile_source"] = "via project config"
        else:
            merged["_active_profile_source"] = "via global config"

    # Re-validate mutual exclusion on merged result (could conflict across files)
    if merged.get("system_prompt") and merged.get("no_system_prompt"):
        raise ConfigError(
            "'system_prompt' and 'no_system_prompt' are mutually exclusive "
            "(set across global and project config)"
        )

    # Attach resolved config directory so callers don't re-derive it.
    merged["config_dir"] = config_dir

    return merged


def resolve_profile_config(args: argparse.Namespace, config: dict) -> str | None:
    """Resolve the active profile and overlay it onto *config* in place.

    Selection precedence: ``--profile`` CLI flag > project ``active_profile``
    > global ``active_profile``.

    Returns the profile name if one was activated, or None.
    Removes ``profiles``, ``active_profile``, and ``_active_profile_source``
    from *config* so downstream code sees only flat LLM keys.
    """
    profiles = config.pop("profiles", None) or {}
    cfg_active = config.pop("active_profile", None)
    config.pop("_active_profile_source", None)

    cli_profile = getattr(args, "profile", None)
    name = cli_profile or cfg_active

    if name is None:
        return None

    if name not in profiles:
        known = ", ".join(sorted(profiles)) if profiles else "(none defined)"
        raise ConfigError(
            f"profile {name!r} not found. Known profiles: {known}. "
            f"Use --list-profiles to see available profiles."
        )

    # Overlay profile values onto config (config values win only if not in profile)
    for key, value in profiles[name].items():
        config[key] = value

    return name


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

    # Special handling for encrypt_secrets: single config key controls mutual-exclusive pair
    if "encrypt_secrets" in config:
        enc_val = config["encrypt_secrets"]
        if _is_unset("encrypt_secrets") and _is_unset("no_encrypt_secrets"):
            args.encrypt_secrets = enc_val
            args.no_encrypt_secrets = not enc_val

    # Special handling: positive config key -> negative argparse dest
    if "sandbox_auto_session" in config:
        if _is_unset("no_sandbox_auto_session"):
            args.no_sandbox_auto_session = not config["sandbox_auto_session"]

    # Apply all other config keys
    _SKIP_KEYS = {"color", "sandbox_auto_session", "encrypt_secrets"}
    for key, value in config.items():
        if key in _SKIP_KEYS:
            continue

        dest = _CONFIG_TO_ARGPARSE.get(key, key)
        if _is_unset(dest):
            setattr(args, dest, value)

    # Sweep: replace remaining sentinels with hardcoded defaults
    for dest, default in _ARGPARSE_DEFAULTS.items():
        if _is_unset(dest):
            setattr(args, dest, default)


def args_to_session_kwargs(args, base_dir: str) -> dict:
    """Convert an argparse namespace to Session constructor kwargs.

    Handles the argparse-dest -> Session-kwarg mapping including boolean
    inversions (no_read_guard -> read_guard, etc.) and key renames
    (add_dir -> allowed_dirs). Filters None values so Session defaults apply.
    """
    # Argparse dest -> Session kwarg name (where they differ)
    _RENAME = {
        "add_dir": "allowed_dirs",
        "add_dir_ro": "allowed_dirs_ro",
    }
    _INVERT = {
        "no_read_guard": "read_guard",
        "no_history": "history",
        "no_memory": "memory",
        "no_continue": "continue_here",
        "no_sandbox_auto_session": "sandbox_auto_session",
        "no_lifecycle": "lifecycle_enabled",
        "quiet": "verbose",
    }
    # Argparse dests that map directly to Session kwargs
    _DIRECT = [
        "provider",
        "model",
        "api_key",
        "base_url",
        "max_turns",
        "max_output_tokens",
        "max_context_tokens",
        "temperature",
        "top_p",
        "seed",
        "files",
        "yolo",
        "commands",
        "system_prompt",
        "no_system_prompt",
        "no_instructions",
        "no_skills",
        "sandbox",
        "sandbox_session",
        "sandbox_strict_read",
        "memory_full",
        "config_dir",
        "proactive_summaries",
        "extra_body",
        "reasoning_effort",
        "sanitize_thinking",
        "prompt_cache",
        "cache",
        "cache_dir",
        "retries",
        "llm_filter",
        "encrypt_secrets_key",
        "encrypt_secrets_tweak",
        "encrypt_secrets_patterns",
        "lifecycle_command",
        "lifecycle_timeout",
        "lifecycle_fail_closed",
    ]

    kwargs: dict = {"base_dir": base_dir}

    for dest in _DIRECT:
        val = getattr(args, dest, None)
        if val is not None:
            kwargs[dest] = val

    for dest, kwarg in _RENAME.items():
        val = getattr(args, dest, None) or []
        kwargs[kwarg] = val

    for dest, kwarg in _INVERT.items():
        val = getattr(args, dest, False)
        kwargs[kwarg] = not val

    # encrypt_secrets: resolve --encrypt-secrets / --no-encrypt-secrets pair
    if getattr(args, "encrypt_secrets", False):
        kwargs["encrypt_secrets"] = True
    elif getattr(args, "no_encrypt_secrets", False):
        kwargs["encrypt_secrets"] = False
    # Also check env var for key (used by reviewer subprocess)
    from .secrets import ENCRYPT_KEY_ENV

    env_key = os.environ.get(ENCRYPT_KEY_ENV)
    if env_key and "encrypt_secrets_key" not in kwargs:
        kwargs["encrypt_secrets_key"] = env_key
        if "encrypt_secrets" not in kwargs:
            kwargs["encrypt_secrets"] = True

    # subagents: resolve --subagents / --no-subagents pair
    _sa = getattr(args, "subagents", False)
    _no_sa = getattr(args, "no_subagents", False)
    if _sa is True:
        kwargs["subagents"] = True
    elif _no_sa is True:
        kwargs["subagents"] = False

    # skills_dir uses None as sentinel for "not set"
    skills_dir = getattr(args, "skills_dir", None)
    if skills_dir is not None:
        kwargs["skills_dir"] = skills_dir

    # verbose is derived from quiet (already handled by _INVERT)
    # but args.verbose may have been set directly
    if hasattr(args, "verbose") and "verbose" not in kwargs:
        kwargs["verbose"] = args.verbose

    return kwargs


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
        "self_review",
        "review_prompt",
        "objective",
        "verify",
        "max_review_rounds",
        "no_mcp",
        "mcp_config",
        "no_a2a",
        "a2a_config",
        "serve_name",
        "serve_description",
        "serve_skills",
    }
    _INVERT_KEYS = {
        "no_read_guard": "read_guard",
        "no_history": "history",
        "no_memory": "memory",
        "no_continue": "continue_here",
        "no_sandbox_auto_session": "sandbox_auto_session",
        "no_lifecycle": "lifecycle_enabled",
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
        '# provider = "lmstudio"          # "lmstudio" | "huggingface" | "openrouter" | "google" | "generic" | "chatgpt" | "bedrock" | "command"',
        '# model = "qwen/qwen3-coder-next"',
        '# api_key = "sk-or-..."            # prefer env vars; this is a fallback',
        '# base_url = "https://..."         # server URL; for bedrock: region name or endpoint URL',
        '# aws_profile = "bedrock"          # AWS profile name for bedrock provider (from ~/.aws/config)',
        "",
        "# --- Generation parameters ---",
        "# max_output_tokens = 32768",
        "# max_context_tokens = 131072",
        "# temperature = 0.7",
        "# top_p = 1.0",
        "# seed = 42",
        "# extra_body = { chat_template_kwargs = { enable_thinking = false } }",
        '# reasoning_effort = "medium"     # "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "default"',
        "# sanitize_thinking = true        # strip leaked <think> tags from assistant responses (default: off)",
        "",
        "# --- Agent behaviour ---",
        "# max_turns = 50",
        "# retries = 5                     # max provider retries on transient network errors (1 = no retry)",
        '# system_prompt = "You are a helpful assistant."',
        "# no_system_prompt = false",
        "",
        "# --- Sandbox / security ---",
        '# sandbox = "builtin"             # "builtin" | "agentfs"',
        '# sandbox_session = "my-session"  # agentfs session ID (optional)',
        "# sandbox_strict_read = false",
        "# sandbox_auto_session = true",
        '# files = "some"                  # "some" (default, workspace) | "all" (unrestricted) | "none" (.swival/ only)',
        '# commands = "all"                # "all" (default) | "none" | ["ls", "git", "python3"]',
        "# yolo = false                    # shorthand for files = all + commands = all",
        '# allowed_dirs = ["../shared-lib", "/data/assets"]',
        '# allowed_dirs_ro = ["/reference/docs", "~/datasets"]',
        "# no_read_guard = false",
        "",
        "# --- Features ---",
        "# no_instructions = false",
        "# no_skills = false",
        "# Global skills are auto-discovered from:",
        "#   $XDG_CONFIG_HOME/swival/skills/ (defaults to ~/.config/swival/skills/)",
        "#   ~/.agents/skills/",
        "# Additional skills directories (override global skills of the same name):",
        '# skills_dir = ["../my-skills"]',
        "# no_history = false",
        "# no_memory = false",
        "# memory_full = false             # inject entire MEMORY.md instead of smart BM25-budgeted excerpts",
        "# no_continue = false",
        "# subagents = false               # enable parallel subagent support (spawn_subagent / check_subagents)",
        "# proactive_summaries = false     # auto-summarize context to prevent overflow on long runs",
        "",
        "# --- UI ---",
        "# color = true       # true = force color, false = force no-color, absent = auto",
        "# quiet = false",
        "",
        "# --- Cache ---",
        "# cache = false                   # enable LLM response caching (.swival/cache.db)",
        '# cache_dir = ".swival"           # custom cache database directory',
        "# prompt_cache = true             # provider-side prompt caching (disable with false)",
        "",
        "# --- MCP ---",
        "# no_mcp = false",
        "",
        "# --- Profiles ---",
        "# Named LLM profiles for quick switching with --profile NAME.",
        "# Set active_profile to use one by default.",
        "#",
        '# active_profile = "fast-local"',
        "",
        "# --- External ---",
        "# max_review_rounds = 15",
        '# llm_filter = "./filter.py"    # outbound message filter script (stdin/stdout JSON)',
        '# reviewer = "./review.sh"',
        "# self_review = false              # use self as reviewer (mirrors provider/model flags)",
        '# review_prompt = "Focus on correctness"',
        '# objective = "objective.md"',
        '# verify = "verification/working.md"',
        "",
        "# --- Lifecycle hooks ---",
        '# lifecycle_command = "./scripts/swival-sync"  # command invoked as: <command> startup|exit <base_dir>',
        "# lifecycle_timeout = 300          # seconds before hook is killed",
        "# lifecycle_fail_closed = false     # true = abort run on hook failure",
        "# no_lifecycle = false              # disable hooks entirely",
        "",
        "# --- Secret encryption ---",
        "# encrypt_secrets = false         # encrypt credential tokens before sending to LLM provider",
        '# encrypt_secrets_key = "hex..."  # optional persistent 32-byte key (hex-encoded)',
        "",
        "# --- A2A serve ---",
        '# serve_name = "My Agent"',
        '# serve_description = "What this agent does"',
        '# serve_skills = [{id = "ask", name = "Ask", description = "Send a question"}]',
        "",
        "# --- Profile examples ---",
        "# [profiles.fast-local]",
        '# provider = "lmstudio"',
        '# model = "qwen3-coder-next"',
        "#",
        "# [profiles.gpt5]",
        '# provider = "chatgpt"',
        '# model = "gpt-5.4"',
        '# reasoning_effort = "high"',
        "",
        "# --- MCP server examples ---",
        "# [mcp_servers.brave-search]",
        '# command = "npx"',
        '# args = ["-y", "@modelcontextprotocol/server-brave-search"]',
        '# env = { BRAVE_API_KEY = "your-key-here" }',
        "",
        "# [mcp_servers.remote-api]",
        '# url = "https://api.example.com/mcp"',
        '# headers = { Authorization = "Bearer token123" }',
        "",
    ]
    return "\n".join(lines)
