"""TAB completion for the Swival REPL."""

from __future__ import annotations

import sys

from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document

from .input_commands import INPUT_COMMANDS
from .skills import find_skill_prefix

_FILE_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/~-"
)


def find_file_prefix(text: str) -> str | None:
    """Return the partial file path being typed after ``@`` at the end of *text*.

    Boundary rule mirrors :func:`~swival.skills.find_skill_prefix`: ``@`` must
    be at position 0 or preceded by a non-alphanumeric character.  Characters
    after ``@`` must be valid path characters (``[a-zA-Z0-9._/~-]``).

    Returns the partial path (without ``@``), or ``None`` if the cursor is not
    in a valid ``@``-mention position.  An empty string means ``@`` was just
    typed — all cwd entries should be offered.
    """
    at = text.rfind("@")
    if at == -1:
        return None
    if at > 0 and text[at - 1].isalnum():
        return None
    partial = text[at + 1 :]
    for ch in partial:
        if ch not in _FILE_PATH_CHARS:
            return None
    return partial


class SwivalCompleter(Completer):
    """Context-aware completer for the Swival REPL.

    Completes slash commands, directory paths for ``/add-dir`` and
    ``/add-dir-ro``, custom commands (``!`` prefix), skill mentions
    (``$`` prefix), and file-path mentions (``@`` prefix).
    """

    def __init__(self, skills_catalog: dict[str, object]) -> None:
        self._skills_catalog = skills_catalog
        self._path_completer = PathCompleter(only_directories=True, expanduser=True)
        self._file_completer = PathCompleter(only_directories=False, expanduser=True)

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        if text.startswith("/") and " " not in text:
            yield from self._complete_slash_commands(text)
            return

        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd = parts[0].lower()
            if cmd in INPUT_COMMANDS and INPUT_COMMANDS[cmd].arg_type == "dir_path":
                arg_text = parts[1] if len(parts) > 1 else ""
                sub_doc = Document(arg_text, len(arg_text))
                yield from self._path_completer.get_completions(sub_doc, complete_event)
            return

        if text.startswith("!") and " " not in text:
            yield from self._complete_custom_commands(text)
            return

        prefix = find_skill_prefix(text)
        if prefix is not None:
            yield from self._complete_skills(prefix)
            return

        file_prefix = find_file_prefix(text)
        if file_prefix is not None:
            sub_doc = Document(file_prefix, len(file_prefix))
            yield from self._file_completer.get_completions(sub_doc, complete_event)

    # ------------------------------------------------------------------

    def _complete_slash_commands(self, text: str):
        prefix = text.lower()
        for cmd in sorted(INPUT_COMMANDS):
            if cmd.lower().startswith(prefix):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=INPUT_COMMANDS[cmd].desc,
                )

    def _complete_custom_commands(self, text: str):
        from .agent import discover_custom_commands

        prefix = text[1:]
        ci = sys.platform == "win32"
        _prefix = prefix.lower() if ci else prefix
        for name in discover_custom_commands():
            _name = name.lower() if ci else name
            if _name.startswith(_prefix):
                yield Completion("!" + name, start_position=-len(text))

    def _complete_skills(self, prefix: str):
        for name in sorted(self._skills_catalog):
            if name.startswith(prefix):
                info = self._skills_catalog[name]
                yield Completion(
                    "$" + name,
                    start_position=-(len(prefix) + 1),
                    display_meta=info.description,
                )
