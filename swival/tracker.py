"""File access tracker for read-before-write guard."""

from __future__ import annotations


class FileAccessTracker:
    """Tracks which files have been read or written, enforcing read-before-write.

    Prevents the agent from overwriting existing files it hasn't read first.
    New file creation is always allowed. Files the agent created via write_file
    are re-writable without a read.
    """

    def __init__(self):
        self.read_files: set[str] = set()
        self.written_files: set[str] = set()

    def record_read(self, path: str) -> None:
        self.read_files.add(path)

    def record_write(self, path: str) -> None:
        self.written_files.add(path)

    def check_write_allowed(self, path: str, exists: bool) -> str | None:
        """Return an error string if the write should be blocked, None if OK."""
        if not exists:
            return None
        if path in self.read_files or path in self.written_files:
            return None
        return (
            "error: cannot write to an existing file that hasn't been read first. "
            "Use read_file to inspect the current contents before overwriting or editing."
        )

    def reset(self) -> None:
        """Clear all tracked state (used by /clear)."""
        self.read_files.clear()
        self.written_files.clear()
