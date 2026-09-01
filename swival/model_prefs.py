"""Per-user model favorites and recents, stored in ~/.config/swival/models.toml.

Deliberately separate from config.toml: that file is user-authored and
onboarding refuses to overwrite it, while this one is program-managed state
that /model rewrites on every favorite toggle and switch. It stays
human-editable TOML so hand-fixing a typo is trivial.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import _toml_format, global_config_dir
from .model_catalog import normalize_provider

_PREFS_FILENAME = "models.toml"
RECENTS_CAP = 10


@dataclass
class ModelPrefs:
    favorites: dict[str, list[str]] = field(default_factory=dict)
    recents: dict[str, list[str]] = field(default_factory=dict)
    # Set when the file existed but could not be parsed; callers show it once.
    warning: str | None = None

    def favorites_for(self, provider: str) -> list[str]:
        return list(self.favorites.get(normalize_provider(provider), []))

    def recents_for(self, provider: str) -> list[str]:
        return list(self.recents.get(normalize_provider(provider), []))

    def is_favorite(self, provider: str, model_id: str) -> bool:
        return model_id in self.favorites.get(normalize_provider(provider), [])

    def preferred_for(self, provider: str) -> list[str]:
        """Favorites then recents, deduped case-insensitively, order kept.

        The priority pool for /model fuzzy matching and TAB completion.
        """
        seen = set()
        names = []
        for name in self.favorites_for(provider) + self.recents_for(provider):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                names.append(name)
        return names


def prefs_path() -> Path:
    return global_config_dir() / _PREFS_FILENAME


def load_prefs() -> ModelPrefs:
    path = prefs_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return ModelPrefs()
    except OSError as e:
        return ModelPrefs(warning=f"could not read {path}: {e}")

    try:
        data = tomllib.loads(raw.decode("utf-8", errors="replace"))
    except tomllib.TOMLDecodeError as e:
        return ModelPrefs(warning=f"ignoring malformed {path}: {e}")

    return ModelPrefs(
        favorites=_clean_section(data.get("favorites")),
        recents=_clean_section(data.get("recents")),
    )


def _preserve_malformed(prefs: ModelPrefs) -> None:
    """Move an unparseable prefs file aside before saving over it, so the
    user's favorites are not silently wiped."""
    if not prefs.warning:
        return
    path = prefs_path()
    try:
        os.replace(path, path.with_name(path.name + ".bad"))
    except OSError:
        pass


def toggle_favorite(provider: str, model_id: str) -> bool:
    """Flip favorite status for (provider, model_id); returns the new status."""
    provider = normalize_provider(provider)
    prefs = load_prefs()
    _preserve_malformed(prefs)
    favs = prefs.favorites.setdefault(provider, [])
    if model_id in favs:
        favs.remove(model_id)
        now_favorite = False
    else:
        favs.append(model_id)
        now_favorite = True
    _save(prefs)
    return now_favorite


def record_recent(provider: str, model_id: str) -> None:
    """Push (provider, model_id) to the front of the MRU recents list."""
    provider = normalize_provider(provider)
    prefs = load_prefs()
    _preserve_malformed(prefs)
    recents = prefs.recents.setdefault(provider, [])
    if model_id in recents:
        recents.remove(model_id)
    recents.insert(0, model_id)
    del recents[RECENTS_CAP:]
    _save(prefs)


def _clean_section(section) -> dict[str, list[str]]:
    if not isinstance(section, dict):
        return {}
    cleaned = {}
    for provider, models in section.items():
        if not isinstance(models, list):
            continue
        cleaned[str(provider)] = [m for m in models if isinstance(m, str)]
    return cleaned


def _save(prefs: ModelPrefs) -> None:
    lines = ["# Model favorites and recents, managed by /model. Safe to edit.", ""]
    for section_name, section in (
        ("favorites", prefs.favorites),
        ("recents", prefs.recents),
    ):
        lines.append(f"[{section_name}]")
        for provider in sorted(section):
            if section[provider]:
                lines.append(f"{provider} = {_toml_format(section[provider])}")
        lines.append("")
    content = "\n".join(lines)

    path = prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".models-", suffix=".toml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
