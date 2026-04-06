"""Interactive first-run onboarding wizard.

Guides the user through provider selection and config creation on first run.
All output goes to stderr via Rich. Never writes to stdout.
"""

import shlex
import shutil
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.output import create_output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import global_config_dir

_console = Console(stderr=True)
_session = PromptSession(output=create_output(sys.stderr))

_PROVIDERS = [
    (
        "lmstudio",
        "LM Studio",
        "Local models on your machine",
        "fast local runs and low-cost experimentation",
    ),
    (
        "llamacpp",
        "llama.cpp",
        "Self-hosted llama-server",
        "running llama.cpp directly with OpenAI-compatible APIs",
    ),
    (
        "chatgpt",
        "ChatGPT",
        "Use your ChatGPT Plus or Pro subscription",
        "getting started fast with an existing subscription",
    ),
    (
        "openrouter",
        "OpenRouter",
        "Hosted models with one API key",
        "easy model sampling and cost/performance tradeoffs",
    ),
    (
        "google",
        "Google Gemini",
        "Gemini through Google's API",
        "Gemini models with a Google API key",
    ),
    (
        "generic",
        "OpenAI-compatible",
        "A local or remote server you already run",
        "Ollama, vLLM, or any other OpenAI-compatible server",
    ),
    (
        "huggingface",
        "HuggingFace",
        "Hosted inference API",
        "HuggingFace-hosted models and endpoints",
    ),
    (
        "bedrock",
        "AWS Bedrock",
        "Models through AWS",
        "enterprise setups with AWS credentials",
    ),
    (
        "command",
        "Command",
        "Use an external program as the backend",
        "advanced: piping through another CLI",
    ),
]

_CONFIG_KEY_ORDER = [
    "provider",
    "model",
    "base_url",
    "api_key",
    "aws_profile",
    "max_context_tokens",
    "max_output_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
    "seed",
]

_SKIP_MARKER = ".onboarding-skipped"

_SUCCESS_SECTIONS = [
    (
        "Start here",
        [
            ("swival", "Open the REPL"),
            ('swival "summarize this repo"', "Run a one-shot task"),
        ],
    ),
    (
        "Want stronger review?",
        [
            ('swival --self-review "fix the login bug"', "Extra quality pass"),
            ('swival --reviewer ./review.sh "add tests"', "Your own review script"),
        ],
    ),
    (
        "Want privacy controls?",
        [
            ('swival --encrypt-secrets "refactor auth"', "Protect secrets in context"),
            ("llm_filter in config", "Filter outbound LLM requests"),
        ],
    ),
    (
        "Want the REPL superpowers?",
        [
            ("/init", "Generate an AGENTS.md for your project"),
            ("/learn", "Review mistakes and persist notes for future sessions"),
            ("/remember", "Add a convention to your project's AGENTS.md"),
            ("/simplify", "Clean up recently changed code"),
            ("/copy", "Copy the last assistant output to clipboard"),
            ("/save", "Set a context checkpoint"),
            ("/restore", "Collapse context back to the last checkpoint"),
        ],
    ),
    (
        "Want to switch model stacks quickly?",
        [
            ('swival --profile gpt5 "review this patch"', "Named profile"),
            ("swival --list-profiles", "See configured profiles"),
            ("swival --init-config --project", "Project-local config template"),
        ],
    ),
    (
        "Want agent-to-agent collaboration?",
        [("See the A2A section at https://swival.dev/", None)],
    ),
    (
        "Want the docs?",
        [
            ("https://swival.dev/", None),
            ("Start with Getting Started, then Providers and Customization.", None),
        ],
    ),
]


def _skip_marker_path() -> Path:
    return global_config_dir() / _SKIP_MARKER


def _global_config_path() -> Path:
    return global_config_dir() / "config.toml"


def run_onboarding() -> Path | None:
    """Run the interactive onboarding wizard.

    Returns the path to the created config file, or None if canceled.
    """
    try:
        return _onboarding_flow()
    except (KeyboardInterrupt, EOFError):
        _console.print()
        _console.print(
            Text("No worries! Run swival again whenever you're ready.", style="dim")
        )
        return None


def _step(current: int, total: int) -> str:
    return f"[dim]\\[{current}/{total}][/dim]"


def _onboarding_flow() -> Path | None:
    """The main onboarding flow. Raises KeyboardInterrupt/EOFError on Ctrl-C."""

    _console.print()
    _console.print(
        Panel(
            "[bold cyan]Swival[/bold cyan] [dim]-- a coding agent for any model[/dim]",
            style="cyan",
            expand=False,
        )
    )
    _console.print()
    _console.print(
        "Swival is a coding agent that lives in your terminal. It can dig through\n"
        "your codebase, edit files, run commands, and pair with you in a live REPL."
    )
    _console.print()
    _console.print(
        Text(
            "Looks like this is your first time here. How would you like to start?",
            style="bold",
        )
    )
    _console.print()

    entry = _prompt_choice(
        "Start",
        [
            "Guided tour + setup [bold green](recommended)[/bold green]",
            "Quick setup",
            "Not right now",
            "Don't show this again",
        ],
        rich_labels=True,
    )

    if entry == 2:
        return None
    if entry == 3:
        _write_skip_marker()
        return None

    guided = entry == 0

    if guided:
        _show_guided_intro()

    total = 4 if guided else 3
    offset = 1 if guided else 0

    while True:
        _console.print()
        _console.print(
            f"  {_step(1 + offset, total)}  [bold]Pick your LLM provider[/bold]"
        )
        settings = _collect_settings()
        if settings is None:
            return None

        _console.print()
        _console.print(f"  {_step(2 + offset, total)}  [bold]Review your config[/bold]")
        result = _preview_and_confirm(settings)
        if result == "yes":
            return _write_config(settings, step_label=_step(3 + offset, total))
        elif result == "start_over":
            continue
        else:
            return None


def _show_guided_intro() -> None:
    _console.print()
    _console.print(f"  {_step(1, 4)}  [bold]Why Swival feels different[/bold]")
    _console.print()
    _console.print(
        "  Many coding agents optimize for speed and plausible output.\n"
        "  Swival leans toward [bold]correctness[/bold]: review loops, workflow capture,\n"
        "  and context discipline are part of the product, not bolted on afterward."
    )
    _console.print()
    _console.print(
        "  Even with the same model, you may get a different result here:\n"
        "  better review feedback, better use of context, or a clearer path\n"
        "  through a messy codebase."
    )
    _console.print()

    table = Table(
        show_header=True,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
    )
    table.add_column("What makes it different", style="white")
    table.add_column("Try it", style="green")
    table.add_row(
        "Favors correctness over plausible output", "--self-review, --reviewer"
    )
    table.add_row("Works well across many model stacks", "providers, profiles")
    table.add_row(
        "Privacy controls at the provider boundary", "llm_filter, --encrypt-secrets"
    )
    table.add_row(
        "The REPL is a workspace, not a chat box", "/learn, /remember, /simplify"
    )

    _console.print(table)
    _console.print()
    _console.print(Text("  Press Enter to continue to setup...", style="dim"))
    _session.prompt("")


def _collect_settings() -> dict | None:
    """Provider selection + provider-specific questions.

    Returns a settings dict or None if canceled.
    """
    _console.print()
    _console.print(
        Text(
            "  You can switch later with profiles. Setup first, tune later.",
            style="dim",
        )
    )
    _console.print()

    labels = []
    for _, display, desc, _ in _PROVIDERS:
        labels.append(f"{display:<20s}{desc}")

    idx = _prompt_choice("Provider", labels)
    provider_name, provider_display, _, best_for = _PROVIDERS[idx]

    settings = {"provider": provider_name}

    _console.print()
    _console.print(Text(f"Nice! Let's configure {provider_display}.", style="bold"))
    _console.print(Text(f"  Best for: {best_for}.", style="dim"))
    _console.print()

    if provider_name == "lmstudio":
        _ask_lmstudio(settings)
    elif provider_name == "llamacpp":
        _ask_llamacpp(settings)
    elif provider_name == "chatgpt":
        _ask_chatgpt(settings)
    elif provider_name == "openrouter":
        _ask_openrouter(settings)
    elif provider_name == "google":
        _ask_google(settings)
    elif provider_name == "generic":
        _ask_generic(settings)
    elif provider_name == "huggingface":
        _ask_huggingface(settings)
    elif provider_name == "bedrock":
        _ask_bedrock(settings)
    elif provider_name == "command":
        _ask_command(settings)

    return settings


def _preview_and_confirm(settings: dict) -> str:
    """Show preview and ask for confirmation.

    Returns "yes", "start_over", or "cancel".
    """
    _console.print()

    dest = _global_config_path()
    _console.print(f"  [dim]Location:[/dim] {dest}")
    for key in _CONFIG_KEY_ORDER:
        if key not in settings:
            continue
        val = settings[key]
        display_key = key.replace("_", " ").title()
        if key == "api_key":
            val = _mask_secret(val)
        _console.print(f"  [dim]{display_key}:[/dim] {val}")
    _console.print()

    idx = _prompt_choice(
        "Write this config?", ["Looks good, write it!", "Start over", "Cancel"]
    )
    if idx == 0:
        return "yes"
    elif idx == 1:
        return "start_over"
    else:
        return "cancel"


def _write_config(settings: dict, *, step_label: str) -> Path | None:
    """Write the config file and show the success screen."""
    dest = _global_config_path()

    if dest.exists():
        _console.print()
        _console.print(
            Text(
                f"A config file already exists at {dest}. Not overwriting.",
                style="yellow",
            )
        )
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = dest.open("x")
    except FileExistsError:
        _console.print()
        _console.print(
            Text(
                f"A config file already exists at {dest}. Not overwriting.",
                style="yellow",
            )
        )
        return None
    with fd:
        fd.write(render_minimal_config(settings))

    _console.print()
    _console.print(
        Panel(
            f"  {step_label}  [bold green]You're all set![/bold green]\n\n"
            f"  Config saved to: [bold]{dest}[/bold]",
            style="green",
            expand=False,
        )
    )

    _render_success_screen()

    return dest


def _render_success_screen() -> None:
    for heading, items in _SUCCESS_SECTIONS:
        _console.print()
        _console.print(f"[bold]{heading}[/bold]")
        for cmd, desc in items:
            if desc:
                _console.print(f"  [green]{cmd}[/green]  {desc}")
            else:
                _console.print(f"  {cmd}")

    _console.print()
    _console.print(
        Text(
            "You don't need to switch tools completely.\n"
            "Swival is worth trying alongside whatever you already use.",
            style="dim",
        )
    )
    _console.print()


def _write_skip_marker() -> None:
    """Write the global skip marker so onboarding doesn't re-prompt."""
    marker = _skip_marker_path()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    except OSError:
        pass


def _ask_lmstudio(s: dict) -> None:
    _console.print(
        Text(
            "Great pick! LM Studio is the fastest way to get going locally.\n"
            "Leave the model blank and Swival will auto-detect whatever you\n"
            "have loaded.",
            style="dim",
        )
    )
    _console.print()

    use_default = _prompt_confirm(
        "Use the default server at http://127.0.0.1:1234?", default=True
    )
    if not use_default:
        url = _prompt_text("Server URL", default="http://127.0.0.1:1234")
        if url and url != "http://127.0.0.1:1234":
            s["base_url"] = url

    model = _prompt_text("Model name (blank for auto-discovery)", default="")
    if model:
        s["model"] = model


def _ask_llamacpp(s: dict) -> None:
    _console.print(
        Text(
            "Nice. Swival can talk directly to llama-server. Leave the model blank\n"
            "and Swival will try to auto-discover what the server is already serving.",
            style="dim",
        )
    )
    _console.print()

    use_default = _prompt_confirm(
        "Use the default server at http://127.0.0.1:8080?", default=True
    )
    if not use_default:
        url = _prompt_text("Server URL", default="http://127.0.0.1:8080")
        if url and url != "http://127.0.0.1:8080":
            s["base_url"] = url

    model = _prompt_text("Model name (blank for auto-discovery)", default="")
    if model:
        s["model"] = model


def _ask_chatgpt(s: dict) -> None:
    _console.print(
        Text(
            "On first use Swival will pop open a quick device login in your browser.\n"
            "After that it remembers you automatically.",
            style="dim",
        )
    )
    _console.print()

    model = _prompt_text("Model name", default="gpt-5.4")
    if model:
        s["model"] = model

    effort = _prompt_text(
        "Reasoning effort (none/low/medium/high, blank to skip)", default=""
    )
    if effort:
        s["reasoning_effort"] = effort


def _ask_openrouter(s: dict) -> None:
    s["model"] = _prompt_text_required("Model (e.g. openai/gpt-5.4)")

    _ask_api_key(s, env_var="OPENROUTER_API_KEY")

    ctx = _prompt_int("Max context tokens (blank to skip)", default=None)
    if ctx is not None:
        s["max_context_tokens"] = ctx


def _ask_google(s: dict) -> None:
    s["model"] = _prompt_text_required("Model (e.g. gemini-2.5-flash)")

    _ask_api_key(s, env_var="GEMINI_API_KEY or OPENAI_API_KEY")


def _ask_generic(s: dict) -> None:
    s["base_url"] = _prompt_text_required("Base URL (e.g. http://127.0.0.1:11434)")
    s["model"] = _prompt_text_required("Model name")

    _ask_api_key(s, env_var="OPENAI_API_KEY")

    ctx = _prompt_int("Max context tokens (blank to skip)", default=None)
    if ctx is not None:
        s["max_context_tokens"] = ctx


def _ask_huggingface(s: dict) -> None:
    while True:
        model = _prompt_text_required("Model (org/model, e.g. zai-org/GLM-5)")
        if "/" in model:
            break
        _console.print(Text("  Must be in org/model format.", style="red"))
    s["model"] = model

    _ask_api_key(s, env_var="HF_TOKEN", label="HuggingFace token")

    url = _prompt_text("Endpoint URL override (blank to skip)", default="")
    if url:
        s["base_url"] = url


def _ask_bedrock(s: dict) -> None:
    s["model"] = _prompt_text_required(
        "Model (e.g. global.anthropic.claude-opus-4-6-v1)"
    )

    region = _prompt_text("AWS region (blank for default)", default="")
    if region:
        s["base_url"] = region

    profile = _prompt_text("AWS profile name (blank for default)", default="")
    if profile:
        s["aws_profile"] = profile


def _ask_command(s: dict) -> None:
    _console.print(
        Text(
            "This shells out to an external program as the LLM backend.\n"
            "The model value is the command Swival will run.",
            style="dim",
        )
    )
    _console.print()

    while True:
        cmd = _prompt_text_required("Command to run as the backend")
        try:
            parts = shlex.split(cmd)
        except ValueError:
            _console.print(
                Text("  Invalid command syntax (check quoting).", style="red")
            )
            continue
        if parts and shutil.which(parts[0]):
            break
        _console.print(
            Text(
                f"  Command not found: {parts[0] if parts else cmd}",
                style="red",
            )
        )
    s["model"] = cmd


def _ask_api_key(s: dict, *, env_var: str, label: str = "API key") -> None:
    """Ask whether to store an API key in config or use an env var."""
    idx = _prompt_choice(
        label,
        [f"I'll set {env_var} myself", "Enter it now (stored in config)"],
    )
    if idx == 1:
        s["api_key"] = _prompt_text_required(label, secret=True)


def render_minimal_config(settings: dict) -> str:
    """Render a minimal TOML config string from onboarding settings."""
    from .config import _toml_format

    lines = [
        "# Swival config, created by first-run setup.",
        "# Run `swival --init-config` to see all available options.",
        "",
    ]
    for key in _CONFIG_KEY_ORDER:
        if key not in settings:
            continue
        lines.append(f"{key} = {_toml_format(settings[key])}")
    lines.append("")
    return "\n".join(lines)


def _mask_secret(val: str) -> str:
    """Mask all but the last 4 characters of a secret."""
    if len(val) <= 4:
        return "****"
    return "*" * (len(val) - 4) + val[-4:]


def _prompt_choice(label: str, choices: list[str], *, rich_labels: bool = False) -> int:
    """Present a numbered list and return the 0-based index of the selection."""
    for i, c in enumerate(choices, 1):
        if rich_labels:
            _console.print(f"  [bold]{i}.[/bold] {c}")
        else:
            _console.print(f"  {i}. {c}")
    _console.print()

    while True:
        raw = _session.prompt(
            HTML(f"<b>{label}</b> [1-{len(choices)}]: "),
        ).strip()
        try:
            n = int(raw)
            if 1 <= n <= len(choices):
                return n - 1
        except ValueError:
            pass
        _console.print(f"  Please enter a number between 1 and {len(choices)}.")


def _prompt_confirm(label: str, *, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    hint = "Y/n" if default else "y/N"
    raw = (
        _session.prompt(
            HTML(f"<b>{label}</b> [{hint}]: "),
        )
        .strip()
        .lower()
    )
    if not raw:
        return default
    return raw in ("y", "yes")


def _prompt_text(label: str, *, default: str = "", secret: bool = False) -> str:
    """Free-text prompt with optional default."""
    if default:
        result = _session.prompt(
            HTML(f"<b>{label}</b> [{default}]: "),
            is_password=secret,
        ).strip()
        return result or default
    return _session.prompt(
        HTML(f"<b>{label}</b>: "),
        is_password=secret,
    ).strip()


def _prompt_text_required(label: str, *, secret: bool = False) -> str:
    """Like _prompt_text but repeats until non-empty."""
    while True:
        val = _prompt_text(label, secret=secret)
        if val:
            return val
        _console.print(Text(f"  {label} is required.", style="red"))


def _prompt_int(label: str, *, default: int | None = None) -> int | None:
    """Prompt for an integer, returning None on blank."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = _session.prompt(
            HTML(f"<b>{label}</b>{hint}: "),
        ).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            _console.print("  Please enter a number.")
