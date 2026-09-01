from __future__ import annotations

import os
import re
import sys

from rich.console import Console
from rich.text import Text

_WRAPPER_PATTERNS: list[tuple[list[str], int | None]] = [
    (["uv", "run"], 1),
    (["uv", "pip", "install"], None),
    (["python3", "-m"], 1),
    (["python", "-m"], 1),
    (["cargo", "test"], None),
    (["go", "test"], None),
    (["npm", "run"], 1),
    (["npm", "test"], None),
    (["npm", "install"], None),
    (["pip", "install"], None),
    (["git", "push"], None),
    (["git", "reset"], None),
    (["git", "clean"], None),
]

_INTERPRETERS = {"python3", "python", "bash", "sh", "node", "bun", "ruby", "perl"}

_TEMP_SCRIPT_RE = re.compile(r"(/tmp/swival-|\.swival/tmp/|/tmp/)")

# Flags that make an interpreter execute inline code, in priority order.
# -c: bash, sh, python, python3, ruby, perl
# -e: node, bun, ruby, perl
# Tuple, not set — iteration order must be deterministic so that combined
# flags like -ec always resolve to the same bucket regardless of hash seed.
_INLINE_CODE_FLAGS = ("c", "e")

# node/bun run code with -e (-c is --check), so prefer e for them.
_INLINE_FLAG_PRIORITY = {"node": ("e", "c"), "bun": ("e", "c")}

# Options that consume the next token, per interpreter; their value must
# not end the flag scan as if it were the script name.
_VALUE_TAKING_FLAGS: dict[str, frozenset[str]] = {
    "bash": frozenset({"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}),
    "sh": frozenset({"-o", "+o"}),
    "python": frozenset({"-X", "-W", "-Q"}),
    "python3": frozenset({"-X", "-W", "-Q"}),
    "perl": frozenset({"-I", "-M", "-m"}),
    "ruby": frozenset({"-I", "-r", "-C", "-E", "-F", "-K"}),
    "node": frozenset(
        {
            "-r",
            "--require",
            "--import",
            "--loader",
            "--experimental-loader",
            "--input-type",
            "--title",
            "--env-file",
        }
    ),
    "bun": frozenset({"-r", "--preload", "--loader", "--env-file", "--cwd"}),
}


def _has_inline_code_flag(argv: list[str]) -> str | None:
    """Return the inline-code flag letter among *argv*'s leading options.

    Handles combined short flags like ``-lc``, skips values of known
    value-taking options, and scans every leading option before choosing so
    ``bash -e -c code`` resolves like ``bash -ec``.
    """
    interp = os.path.basename(argv[0])
    value_flags = _VALUE_TAKING_FLAGS.get(interp, frozenset())
    found: set[str] = set()
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--" or arg == "-":
            break
        if arg in value_flags:
            i += 2
            continue
        if arg.startswith("--") or arg.startswith("+"):
            i += 1
            continue
        if not arg.startswith("-"):
            break
        found.update(arg[1:])
        i += 1
    for ch in _INLINE_FLAG_PRIORITY.get(interp, _INLINE_CODE_FLAGS):
        if ch in found:
            return ch
    return None


_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Wrappers that run whatever follows them, and their options that consume
# the next token.
_TRANSPARENT_WRAPPERS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "-C", "-S", "--unset", "--chdir", "--split-string"}),
    "command": frozenset(),
    "builtin": frozenset(),
    "exec": frozenset({"-a"}),
}

# Shell builtins that evaluate their arguments as shell code.
_SHELL_EVAL_BUILTINS = frozenset({"eval", "source", "."})


def _strip_transparent_prefix(argv: list[str]) -> list[str]:
    """Drop leading ``NAME=value`` assignments and ``env``-style wrappers so
    the wrapped command is what gets bucketed.  Returns *argv* unchanged when
    nothing would be left.
    """
    i = 0
    n = len(argv)
    while i < n:
        if _ENV_ASSIGNMENT_RE.match(argv[i]):
            i += 1
            continue
        value_opts = _TRANSPARENT_WRAPPERS.get(os.path.basename(argv[i]))
        if value_opts is None:
            break
        i += 1
        while i < n:
            tok = argv[i]
            if tok == "--":
                i += 1
                break
            if tok in value_opts:
                i += 2
                continue
            if tok.startswith("-") or _ENV_ASSIGNMENT_RE.match(tok):
                i += 1
                continue
            break
    if 0 < i < n:
        return argv[i:]
    return argv


HIGH_RISK_BUCKETS = {
    "rm",
    "git push",
    "git reset",
    "git clean",
    "docker",
    "kubectl",
    "curl",
    "wget",
    "npm install",
    "pip install",
    "uv pip install",
    "uv run -c",
    "<shell>",
    "bash -c",
    "sh -c",
    "python3 -c",
    "python -c",
    "node -e",
    "bun -e",
    "ruby -e",
    "ruby -c",
    "perl -e",
    "perl -c",
}


_SHELL_BUCKET = "<shell>"


def normalize_bucket(argv: list[str]) -> str:
    if not argv:
        return ""
    argv = _strip_transparent_prefix(argv)

    for prefix, extra_count in _WRAPPER_PATTERNS:
        plen = len(prefix)
        if len(argv) >= plen and argv[:plen] == prefix:
            if extra_count is not None and len(argv) > plen:
                return " ".join(prefix + argv[plen : plen + extra_count])
            return " ".join(prefix)

    cmd = argv[0]
    is_path = "/" in cmd or "\\" in cmd
    basename = os.path.basename(cmd)
    # The bucket prefix is the full path for path-invoked commands, so that
    # approving "ls" does not also approve "/tmp/ls" or "./ls".
    prefix = cmd if is_path else basename

    if basename in _SHELL_EVAL_BUILTINS:
        return _SHELL_BUCKET

    # Interpreter inline-code detection applies regardless of path form,
    # so that /bin/bash -c and bash -c get equivalent treatment.
    if basename in _INTERPRETERS and len(argv) >= 2:
        if _TEMP_SCRIPT_RE.search(argv[1]):
            return f"{prefix} <temp-script>"
        flag = _has_inline_code_flag(argv)
        if flag is not None:
            return f"{prefix} -{flag}"

    return prefix


def is_high_risk(bucket: str) -> bool:
    if bucket in HIGH_RISK_BUCKETS:
        return True
    # Path-prefixed buckets like "/bin/bash -c" should match "bash -c".
    if "/" in bucket or "\\" in bucket:
        # Split into command part and suffix (e.g. "/bin/bash -c" -> "bash -c")
        parts = bucket.split(None, 1)  # split on first space
        basename_form = os.path.basename(parts[0])
        if len(parts) > 1:
            basename_form += " " + parts[1]
        return basename_form in HIGH_RISK_BUCKETS
    return False


_VALID_MODES = frozenset({"full", "none", "allowlist", "ask"})


def _read_bucket_file(path: str) -> set[str]:
    with open(path) as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


class CommandPolicy:
    def __init__(
        self,
        mode: str,
        allowed_basenames: set[str] | None = None,
        approved_buckets: set[str] | None = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"invalid CommandPolicy mode {mode!r}, "
                f"expected one of {sorted(_VALID_MODES)}"
            )
        self.mode = mode
        self.allowed_basenames = allowed_basenames or set()
        self.approved_buckets = set(approved_buckets or ())
        self.denied_buckets: set[str] = set()
        self.always_ask_buckets: set[str] = set()

    @property
    def shell_allowed(self) -> bool:
        return self.mode == "full"

    def check(self, argv: list[str], is_subagent: bool = False) -> str | None:
        if self.mode == "full":
            return None
        if self.mode == "none":
            return "error: commands are disabled (commands=none). Adjust your plan."
        if self.mode == "allowlist":
            basename = os.path.basename(_strip_transparent_prefix(argv)[0])
            if basename in self.allowed_basenames:
                return None
            return (
                f"error: command {basename!r} is not in the allowed list. "
                f"Allowed: {', '.join(sorted(self.allowed_basenames))}."
            )

        # mode == "ask": every command requires approval via bucket
        bucket = normalize_bucket(argv)

        if bucket in self.denied_buckets:
            return (
                f"error: user denied command bucket {bucket!r}. "
                f"Do not retry this command or any equivalent variant. Adjust your plan."
            )

        if bucket in self.approved_buckets and bucket not in self.always_ask_buckets:
            return None

        if is_subagent:
            return (
                f"error: command bucket {bucket!r} is not approved. "
                f"Subagents cannot prompt for approval. "
                f"Run this command from the main agent, or pre-approve the bucket in config."
            )

        return f"needs_approval:{bucket}"

    def approve_bucket(self, bucket: str) -> None:
        self.approved_buckets.add(bucket)
        self.denied_buckets.discard(bucket)

    def deny_bucket(self, bucket: str) -> None:
        self.denied_buckets.add(bucket)
        self.approved_buckets.discard(bucket)

    def mark_always_ask(self, bucket: str) -> None:
        self.always_ask_buckets.add(bucket)
        self.approved_buckets.discard(bucket)


def prompt_approval(bucket: str, high_risk: bool = False) -> str:
    from .fmt import suspend_live

    console = Console(stderr=True)

    with suspend_live():
        label = Text(bucket, style="bold")
        if high_risk:
            console.print(Text("⚠ high-risk ", style="bold red"), label, end="")
        else:
            console.print(Text("? ", style="bold yellow"), label, end="")

        if high_risk:
            hint = " [enter=deny / y=allow / p=persist / o=once / a=always-ask]: "
        else:
            hint = " [enter=allow / n=deny / p=persist / o=once / a=always-ask]: "

        sys.stderr.write(hint)
        sys.stderr.flush()

        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"

    if answer == "":
        return "deny" if high_risk else "allow"
    if answer in ("y", "yes"):
        return "allow"
    if answer == "p":
        return "persist"
    if answer in ("n", "no"):
        return "deny"
    if answer == "o":
        return "once"
    if answer in ("a", "always"):
        return "always_ask"
    return "deny" if high_risk else "allow"


def load_persisted_buckets(base_dir: str) -> set[str]:
    """Load runtime-persisted approved buckets from .swival/approved_buckets."""
    path = os.path.join(base_dir, ".swival", "approved_buckets")
    if not os.path.isfile(path):
        return set()
    return _read_bucket_file(path)


def persist_approved_bucket(bucket: str, base_dir: str) -> None:
    dir_path = os.path.join(base_dir, ".swival")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "approved_buckets")

    existing = _read_bucket_file(file_path) if os.path.isfile(file_path) else set()
    if bucket in existing:
        return

    with open(file_path, "a") as f:
        f.write(bucket + "\n")
