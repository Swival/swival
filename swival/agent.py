import argparse
import copy
from datetime import datetime
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import tiktoken

from . import fmt
from .thinking import ThinkingState, _safe_notes_path
from .tools import (
    TOOLS,
    RUN_COMMAND_TOOL,
    USE_SKILL_TOOL,
    dispatch,
    cleanup_old_cmd_outputs,
)

DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
MAX_ARG_LOG = 1000
MAX_INSTRUCTIONS_CHARS = 10_000

_encoder = tiktoken.get_encoding("cl100k_base")

_CONTEXT_OVERFLOW_RE = re.compile(
    r"context.{0,10}(length|window|limit)"
    r"|maximum.{0,10}(context|token)"
    r"|token.{0,10}limit"
    r"|exceed.{0,10}(context|token|max)",
    re.IGNORECASE,
)


class ContextOverflowError(Exception):
    """Raised when the LLM call fails due to context window overflow."""

    pass


def _canonical_error(error: str) -> str:
    """Extract a stable error fingerprint for repeat detection."""
    return error.split("\n", 1)[0]


def estimate_tokens(messages: list, tools: list | None = None) -> int:
    """Count tokens across all messages using tiktoken."""
    total = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content", "") or ""
            tool_calls = m.get("tool_calls", None)
        else:
            content = getattr(m, "content", "") or ""
            tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    content += tc.function.name + (tc.function.arguments or "")
                elif isinstance(tc, dict):
                    fn = tc.get("function", {})
                    content += fn.get("name", "") + (fn.get("arguments", "") or "")
        total += len(_encoder.encode(content))
    if tools:
        total += len(_encoder.encode(json.dumps(tools)))
    # Per-message overhead (role, separators) — ~4 tokens each
    total += 4 * len(messages)
    return total


def group_into_turns(messages: list) -> list[list]:
    """Group messages into atomic turns.

    A turn is one of:
    - A single message (system, user, or assistant without tool_calls)
    - An assistant message with tool_calls + all its matching tool results
    """
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        tool_calls = (
            msg.get("tool_calls", None)
            if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )

        if role == "assistant" and tool_calls:
            # Collect this assistant msg + all following tool results
            turn = [msg]
            tc_ids = {tc.id if hasattr(tc, "id") else tc["id"] for tc in tool_calls}
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                next_role = (
                    next_msg.get("role")
                    if isinstance(next_msg, dict)
                    else getattr(next_msg, "role", None)
                )
                tc_id = (
                    next_msg.get("tool_call_id")
                    if isinstance(next_msg, dict)
                    else getattr(next_msg, "tool_call_id", None)
                )
                if next_role == "tool" and tc_id in tc_ids:
                    turn.append(next_msg)
                    j += 1
                else:
                    break
            turns.append(turn)
            i = j
        else:
            turns.append([msg])
            i += 1
    return turns


def compact_messages(messages: list) -> list:
    """Truncate large tool results in older turns, preserving turn atomicity."""
    turns = group_into_turns(messages)
    # Skip the most recent 2 turns
    cutoff = max(0, len(turns) - 2)
    for turn in turns[:cutoff]:
        for msg in turn:
            role = (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            if role == "tool":
                content = (
                    msg.get("content", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                )
                if content and len(content) > 1000:
                    replacement = f"[compacted — originally {len(content)} chars]"
                    if isinstance(msg, dict):
                        msg["content"] = replacement
                    else:
                        msg.content = replacement
    # Flatten turns back to message list
    return [msg for turn in turns for msg in turn]


def drop_middle_turns(messages: list) -> list:
    """Drop oldest turns from middle; keep leading system/user block + last 3 turns."""
    turns = group_into_turns(messages)

    # Find leading block: scan forward while role is system or user
    leading_count = 0
    for turn in turns:
        msg = turn[0]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role in ("system", "user"):
            leading_count += 1
        else:
            break

    keep_tail = 3
    # If there's no middle to drop, return unchanged
    if leading_count + keep_tail >= len(turns):
        return [msg for turn in turns for msg in turn]

    leading = turns[:leading_count]
    tail = turns[-keep_tail:]
    splice_marker = [
        {
            "role": "user",
            "content": "[context compacted — older tool calls and results were removed to fit context window]",
        }
    ]

    result = []
    for turn in leading:
        result.extend(turn)
    result.extend(splice_marker)
    for turn in tail:
        result.extend(turn)
    return result


def clamp_output_tokens(
    messages: list,
    tools: list | None,
    context_length: int | None,
    requested_max_output: int,
) -> int:
    """Reduce max_output_tokens if prompt + output would exceed context."""
    if context_length is None:
        return requested_max_output
    prompt_tokens = estimate_tokens(messages, tools)
    available = context_length - prompt_tokens
    if available < 1:
        return 1  # Nearly full; use minimal budget and let overflow retry handle it
    return min(requested_max_output, available)


def load_instructions(base_dir: str, verbose: bool) -> str:
    """Load CLAUDE.md and/or AGENT.md from base_dir, if present.

    Returns the combined content as XML-tagged sections, or "" if
    no instruction files are found. Reads at most MAX_INSTRUCTIONS_CHARS+1
    characters per file to avoid unbounded memory use.
    """
    sections = []
    for filename, tag in [
        ("CLAUDE.md", "project-instructions"),
        ("AGENT.md", "agent-instructions"),
    ]:
        path = Path(base_dir).resolve() / filename
        if not path.is_file():
            continue
        try:
            file_size = path.stat().st_size
            with path.open(encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_INSTRUCTIONS_CHARS + 1)
        except OSError:
            continue
        if len(content) > MAX_INSTRUCTIONS_CHARS:
            content = (
                content[:MAX_INSTRUCTIONS_CHARS]
                + f"\n[truncated — {filename} exceeds {MAX_INSTRUCTIONS_CHARS} character limit]"
            )
        if verbose:
            fmt.info(f"Loaded {filename} ({file_size} bytes) from {path.parent}")
        sections.append(f"<{tag}>\n{content}\n</{tag}>")
    return "\n\n".join(sections)


def handle_tool_call(
    tool_call,
    base_dir,
    thinking_state,
    verbose,
    resolved_commands=None,
    skills_catalog=None,
    skill_read_roots=None,
    extra_write_roots=None,
    yolo=False,
):
    """Execute a single tool call and return the tool result message dict.

    Handles JSON parsing, logging (skipped for think), dispatch, and error catching.
    """
    name = tool_call.function.name
    raw_args = tool_call.function.arguments

    try:
        parsed_args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError) as e:
        if verbose:
            fmt.tool_error(name, f"invalid JSON: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"error: invalid JSON in tool arguments: {e}",
        }

    if name != "think" and verbose:
        pretty = json.dumps(parsed_args, indent=2)
        if len(pretty) > MAX_ARG_LOG:
            pretty = pretty[:MAX_ARG_LOG] + "\n... (truncated)"
        fmt.tool_call(name, pretty)

    t0 = time.monotonic()
    try:
        result = dispatch(
            name,
            parsed_args,
            base_dir,
            thinking_state=thinking_state,
            resolved_commands=resolved_commands or {},
            skills_catalog=skills_catalog or {},
            skill_read_roots=skill_read_roots if skill_read_roots is not None else [],
            extra_write_roots=extra_write_roots
            if extra_write_roots is not None
            else [],
            yolo=yolo,
        )
    except Exception as e:
        result = f"error: {e}"
    elapsed = time.monotonic() - t0

    if name != "think" and verbose:
        if result.startswith("error:"):
            fmt.tool_error(name, result)
        else:
            fmt.tool_result(name, elapsed, result[:500])

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }


def discover_model(base_url, verbose):
    """Query LM Studio's native API to find the currently loaded LLM."""
    url = f"{base_url}/api/v1/models"
    if verbose:
        fmt.model_info(f"Querying {url} for loaded models...")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        fmt.error(f"could not connect to LM Studio at {base_url}: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        fmt.error(f"invalid JSON from {url}: {e}")
        sys.exit(1)

    # Find first entry with type=="llm" and non-empty loaded_instances
    # LM Studio uses "data" (OpenAI-compat) or "models" (native API) as the top-level key
    entries = data.get("data") or data.get("models") or []
    for entry in entries:
        if entry.get("type") == "llm" and entry.get("loaded_instances"):
            instance = entry["loaded_instances"][0]
            context_length = instance.get("config", {}).get("context_length")
            model_key = entry.get("id", entry.get("key"))
            if verbose:
                fmt.model_info(
                    f"Discovered loaded model: {model_key} (context={context_length})"
                )
            return model_key, context_length

    return None, None


def configure_context(base_url, model_key, requested_context, current_context, verbose):
    """Reload the model with a different context size if needed."""
    if requested_context == current_context:
        if verbose:
            fmt.model_info(
                f"Requested context {requested_context} matches current context, no reload needed."
            )
        return

    url = f"{base_url}/api/v1/models/load"
    payload = json.dumps(
        {"model": model_key, "context_length": requested_context}
    ).encode()
    if verbose:
        fmt.model_info(
            f"Reloading model {model_key} with context_length={requested_context}..."
        )
        fmt.model_info("Note: this may take a while as the model reloads.")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except urllib.error.URLError as e:
        fmt.error(f"failed to reload model with new context size: {e}")
        sys.exit(1)

    if verbose:
        fmt.model_info("Model reloaded successfully.")


def call_llm(
    base_url,
    model_id,
    messages,
    max_output_tokens,
    temperature,
    top_p,
    seed,
    tools,
    verbose,
    *,
    provider="lmstudio",
    api_key=None,
):
    """Call LiteLLM with the appropriate provider. Returns (message, finish_reason)."""
    import litellm

    litellm.suppress_debug_info = True

    if provider == "lmstudio":
        model_str = f"openai/{model_id}"
        kwargs = {"api_base": f"{base_url}/v1", "api_key": "lm-studio"}
    elif provider == "huggingface":
        bare_id = model_id.removeprefix("huggingface/")
        model_str = f"huggingface/{bare_id}"
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["api_base"] = base_url
    else:
        fmt.error(f"unknown provider {provider!r}")
        sys.exit(1)

    if verbose:
        seed_info = f", seed={seed}" if seed is not None else ""
        fmt.model_info(
            f"Calling model {model_str} with max_tokens={max_output_tokens}, temperature={temperature}, top_p={top_p}{seed_info}"
        )

    completion_kwargs = dict(
        model=model_str,
        messages=messages,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        tools=tools,
        tool_choice="auto",
        **kwargs,
    )
    if seed is not None:
        completion_kwargs["seed"] = seed

    try:
        response = litellm.completion(**completion_kwargs)
    except litellm.ContextWindowExceededError:
        raise ContextOverflowError("context window exceeded (typed)")
    except litellm.BadRequestError as e:
        msg_text = str(e)
        if _CONTEXT_OVERFLOW_RE.search(msg_text):
            raise ContextOverflowError(f"context window exceeded (inferred): {e}")
        fmt.error(f"LLM call failed: {e}")
        sys.exit(1)
    except Exception as e:
        fmt.error(f"LLM call failed: {e}")
        sys.exit(1)

    choice = response.choices[0]
    return choice.message, choice.finish_reason


def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="swival",
        usage="%(prog)s [options] <question>\n       %(prog)s --repl [options] [question]",
        description="A CLI coding agent with tool-calling, sandboxed file access, and multi-provider LLM support.",
    )
    parser.add_argument(
        "question", nargs="?", default=None, help="The question or task for the model."
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start an interactive session instead of answering a single question.",
    )
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "huggingface"],
        default="lmstudio",
        help="LLM provider (default: lmstudio).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the provider (overrides env var).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Server base URL (default: http://127.0.0.1:1234 for lmstudio).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Requested context length for the model (may trigger a reload).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        help="Maximum output tokens (default: 32768).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.55,
        help="Sampling temperature (default: 0.55).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible outputs (optional, model support varies).",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to include.",
    )
    prompt_group.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit the system message entirely.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override auto-discovered model with a specific model identifier.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all diagnostics; only print the final result.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum agent loop iterations (default: 50).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for file tools (default: current directory).",
    )
    parser.add_argument(
        "--allowed-commands",
        type=str,
        default=None,
        help='Comma-separated list of allowed command basenames (e.g. "ls,git,python3").',
    )
    parser.add_argument(
        "--no-instructions",
        action="store_true",
        help="Don't load CLAUDE.md or AGENT.md from the base directory.",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=[],
        help="Additional directory to scan for skills (can be repeated).",
    )
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="Don't load or discover any skills.",
    )
    parser.add_argument(
        "--allow-dir",
        type=str,
        action="append",
        default=[],
        help="Grant read/write access to an extra directory (repeatable).",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Disable filesystem sandbox and command whitelist (unrestricted mode).",
    )

    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        action="store_true",
        help="Force ANSI color even when stderr is not a TTY.",
    )
    color_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color even when stderr is a TTY.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.verbose = not args.quiet

    if not args.repl and args.question is None:
        parser.error("question is required (or use --repl)")

    fmt.init(color=args.color, no_color=args.no_color)

    # Validation: max_output_tokens <= max_context_tokens
    if (
        args.max_context_tokens is not None
        and args.max_output_tokens > args.max_context_tokens
    ):
        parser.error(
            "--max-output-tokens must be <= --max-context-tokens when both are specified."
        )

    # Provider-specific model discovery and context configuration
    if args.provider == "lmstudio":
        api_base = args.base_url or "http://127.0.0.1:1234"
        if args.model:
            model_id = args.model
            current_context = None
            if args.verbose:
                fmt.model_info(f"Using user-specified model: {model_id}")
        else:
            model_id, current_context = discover_model(api_base, args.verbose)
            if not model_id:
                fmt.error(
                    "no loaded LLM found in LM Studio. "
                    "Load a model in LM Studio or use --model to specify one."
                )
                sys.exit(1)
        if args.max_context_tokens is not None:
            configure_context(
                api_base,
                model_id,
                args.max_context_tokens,
                current_context,
                args.verbose,
            )
        api_key = None

    elif args.provider == "huggingface":
        if not args.model:
            parser.error("--model is required when --provider is huggingface")
        bare_model = args.model.removeprefix("huggingface/")
        if "/" not in bare_model:
            parser.error(
                "HuggingFace model must be in org/model format (e.g. meta-llama/Llama-3.3-70B-Instruct)"
            )
        api_base = args.base_url  # None unless user set it (dedicated endpoint)
        model_id = args.model
        current_context = args.max_context_tokens  # None if not given
        api_key = args.api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            parser.error(
                "--api-key or HF_TOKEN env var required for huggingface provider"
            )

    llm_kwargs = {
        "provider": args.provider,
        "api_key": api_key,
    }

    # Resolve --allow-dir paths
    allowed_dirs: list[Path] = []
    for d in args.allow_dir:
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            fmt.error(f"--allow-dir path is not a directory: {d}")
            sys.exit(1)
        if p == Path(p.anchor):
            fmt.error(f"--allow-dir cannot be the filesystem root: {d}")
            sys.exit(1)
        allowed_dirs.append(p)

    # Resolve allowed commands
    base_dir = args.base_dir
    yolo = args.yolo

    if yolo:
        # In yolo mode, skip all command resolution — any command can run
        resolved_commands: dict[str, str] = {}
    else:
        allowed_names = (
            {c.strip() for c in args.allowed_commands.split(",") if c.strip()}
            if args.allowed_commands
            else set()
        )
        resolved_commands = {}
        base_resolved = Path(base_dir).resolve()
        for name in sorted(allowed_names):
            cmd_path = shutil.which(name)
            if cmd_path is None:
                fmt.error(f"allowed command {name!r} not found on PATH")
                sys.exit(1)
            abs_path = Path(cmd_path).resolve()
            if abs_path.is_relative_to(base_resolved):
                fmt.error(
                    f"allowed command {name!r} resolves to {abs_path}, "
                    f"which is inside base directory {base_resolved}. "
                    f"Commands inside the workspace can be modified by the model."
                )
                sys.exit(1)
            resolved_commands[name] = str(abs_path)

    # Discover skills
    from .skills import discover_skills, format_skill_catalog

    skills_catalog: dict = {}
    skill_read_roots: list[Path] = []
    if not args.no_skills:
        skills_catalog = discover_skills(base_dir, args.skills_dir, args.verbose)

    # Build tools list (conditionally include run_command and use_skill)
    tools = list(TOOLS)
    if skills_catalog:
        tools.append(USE_SKILL_TOOL)
    if yolo:
        tool = copy.deepcopy(RUN_COMMAND_TOOL)
        tool["function"]["description"] = "Run any command and return its output."
        tools.append(tool)
    elif resolved_commands:
        tool = copy.deepcopy(RUN_COMMAND_TOOL)
        tool["function"]["description"] = (
            f"Run a command and return its output. Allowed commands: {', '.join(sorted(resolved_commands))}."
        )
        tools.append(tool)

    # Build messages
    messages = []
    if not args.no_system_prompt:
        if args.system_prompt:
            system_content = args.system_prompt
        else:
            system_content = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
            # Append project/agent instructions (only with default prompt)
            if not args.no_instructions:
                instructions = load_instructions(base_dir, args.verbose)
                if instructions:
                    system_content += "\n\n" + instructions
        # Include current date and time
        now = datetime.now().astimezone()
        system_content += (
            f"\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M %Z')}"
        )
        # Append skill catalog (only with default prompt)
        if skills_catalog and not args.system_prompt:
            catalog_text = format_skill_catalog(skills_catalog)
            if catalog_text:
                system_content += "\n\n" + catalog_text
        if yolo:
            system_content += (
                "\n\n**Command execution tool:**\n"
                "- `run_command`: Run any command and return its output. "
                'Pass the command and arguments as a list (e.g. `["ls", "-la"]`). '
                "Optional `timeout` (1-120s, default 30)."
            )
        elif resolved_commands:
            cmd_list = ", ".join(sorted(resolved_commands))
            system_content += (
                "\n\n**Command execution tool:**\n"
                f"- `run_command`: Run a whitelisted command and return its output. "
                f'Pass the command and arguments as a list (e.g. `["ls", "-la"]`). '
                f"Optional `timeout` (1-120s, default 30). "
                f"Allowed commands: {cmd_list}."
            )
        messages.append({"role": "system", "content": system_content})
    # Determine context length for output clamping
    context_length = args.max_context_tokens or current_context

    # Clean up stale cmd_output files from previous sessions
    removed = cleanup_old_cmd_outputs(base_dir)
    if removed and args.verbose:
        fmt.info(f"Cleaned up {removed} stale cmd_output file(s) from .swival/")

    import atexit

    atexit.register(cleanup_old_cmd_outputs, base_dir)

    thinking_state = ThinkingState(verbose=args.verbose, notes_dir=base_dir)

    loop_kwargs = dict(
        api_base=api_base,
        model_id=model_id,
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        context_length=context_length,
        base_dir=base_dir,
        thinking_state=thinking_state,
        resolved_commands=resolved_commands,
        skills_catalog=skills_catalog,
        skill_read_roots=skill_read_roots,
        extra_write_roots=allowed_dirs,
        yolo=yolo,
        verbose=args.verbose,
        llm_kwargs=llm_kwargs,
    )

    if not args.repl:
        # Single-shot path
        messages.append({"role": "user", "content": args.question})
        answer, exhausted = run_agent_loop(messages, tools, **loop_kwargs)
        if answer is not None:
            print(answer)
        if exhausted:
            fmt.warning("max turns reached, agent stopped.")
            sys.exit(2)
        return

    # REPL path
    if args.question:
        messages.append({"role": "user", "content": args.question})
        answer, exhausted = run_agent_loop(messages, tools, **loop_kwargs)
        if answer is not None:
            print(answer)
        if exhausted:
            fmt.warning("max turns reached for initial question.")

    repl_loop(messages, tools, **loop_kwargs)


def run_agent_loop(
    messages: list,
    tools: list,
    *,
    api_base: str,
    model_id: str,
    max_turns: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    context_length: int | None,
    base_dir: str,
    thinking_state: ThinkingState,
    resolved_commands: dict,
    skills_catalog: dict,
    skill_read_roots: list,
    extra_write_roots: list,
    yolo: bool,
    verbose: bool,
    llm_kwargs: dict,
) -> tuple[str | None, bool]:
    """Run the tool-calling loop until a final answer or max turns.

    Mutates `messages` in place (appends assistant/tool messages,
    in-place compaction on overflow).
    Returns (final_answer, exhausted). final_answer is the last
    assistant text (may be None). exhausted is True if max_turns hit.
    """
    consecutive_errors: dict[str, tuple[str, int]] = {}
    turns = 0

    while turns < max_turns:
        turns += 1
        token_est = estimate_tokens(messages, tools)
        if verbose:
            fmt.turn_header(turns, max_turns, token_est)

        effective_max_output = clamp_output_tokens(
            messages, tools, context_length, max_output_tokens
        )
        if effective_max_output != max_output_tokens and verbose:
            fmt.info(
                f"Output tokens clamped: {max_output_tokens} -> {effective_max_output} (context_length={context_length}, prompt=~{token_est})"
            )

        t0 = time.monotonic()
        try:
            msg, finish_reason = call_llm(
                api_base,
                model_id,
                messages,
                effective_max_output,
                temperature,
                top_p,
                seed,
                tools,
                verbose,
                **llm_kwargs,
            )
        except ContextOverflowError:
            fmt.warning("context window exceeded, compacting history...")
            messages[:] = compact_messages(messages)
            effective_max_output = clamp_output_tokens(
                messages, tools, context_length, max_output_tokens
            )
            if verbose:
                fmt.context_stats(
                    "Context after compaction", estimate_tokens(messages, tools)
                )
            try:
                msg, finish_reason = call_llm(
                    api_base,
                    model_id,
                    messages,
                    effective_max_output,
                    temperature,
                    top_p,
                    tools,
                    verbose,
                    **llm_kwargs,
                )
            except ContextOverflowError:
                fmt.warning("still too large, dropping older turns...")
                messages[:] = drop_middle_turns(messages)
                effective_max_output = clamp_output_tokens(
                    messages, tools, context_length, max_output_tokens
                )
                if verbose:
                    fmt.context_stats(
                        "Context after drop", estimate_tokens(messages, tools)
                    )
                try:
                    msg, finish_reason = call_llm(
                        api_base,
                        model_id,
                        messages,
                        effective_max_output,
                        temperature,
                        top_p,
                        tools,
                        verbose,
                        **llm_kwargs,
                    )
                except ContextOverflowError:
                    fmt.error("context window exceeded even after compaction.")
                    sys.exit(1)
        elapsed = time.monotonic() - t0
        if verbose:
            fmt.llm_timing(elapsed, finish_reason)
        messages.append(msg)

        # Log intermediate assistant text (reasoning before tool calls, or truncated responses)
        if msg.content and (msg.tool_calls or finish_reason == "length") and verbose:
            fmt.assistant_text(msg.content)

        if not msg.tool_calls:
            if finish_reason == "length":
                # Output was truncated before the model could finish;
                # nudge it to continue using tools instead of quitting.
                if verbose:
                    fmt.info(
                        "Response truncated (finish_reason=length), prompting continuation."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "Your response was cut off. Please use the provided tools to complete the task step by step.",
                    }
                )
                continue
            # Model produced a final text answer
            if verbose:
                fmt.completion(turns, "ok")
            return msg.content or "", False

        interventions: list[str] = []
        for tool_call in msg.tool_calls:
            tool_msg = handle_tool_call(
                tool_call,
                base_dir,
                thinking_state,
                verbose,
                resolved_commands=resolved_commands,
                skills_catalog=skills_catalog,
                skill_read_roots=skill_read_roots,
                extra_write_roots=extra_write_roots,
                yolo=yolo,
            )
            messages.append(tool_msg)

            tool_name = tool_call.function.name
            result = tool_msg["content"]
            if result.startswith("error:"):
                canonical_error = _canonical_error(result)
                prev_error, prev_count = consecutive_errors.get(tool_name, ("", 0))
                if canonical_error == prev_error:
                    count = prev_count + 1
                else:
                    count = 1
                consecutive_errors[tool_name] = (canonical_error, count)

                if count >= 2:
                    if count >= 3:
                        interventions.append(
                            f"STOP: You have failed to use `{tool_name}` correctly {count} times in a row "
                            "with the same error. Do NOT call "
                            f"`{tool_name}` again with the same arguments. "
                            "Either fix the arguments or use a completely different approach to accomplish your task."
                        )
                    else:
                        interventions.append(
                            f"IMPORTANT: You have called `{tool_name}` {count} times with the same error. "
                            f"The error is: {canonical_error}\n"
                            "Please carefully re-read the error message and fix your tool call. "
                            "If you cannot use this tool correctly, use a different approach."
                        )
                    if verbose:
                        fmt.guardrail(tool_name, count, canonical_error)
            else:
                consecutive_errors.pop(tool_name, None)
        if interventions:
            messages.append({"role": "user", "content": "\n\n".join(interventions)})
        if verbose:
            fmt.context_stats(
                f"Context after turn {turns}", estimate_tokens(messages, tools)
            )

    # max_turns exhausted — extract last assistant text
    if verbose:
        fmt.completion(turns, "max_turns")
    last_text = None
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "assistant":
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            )
            if content:
                last_text = content
                break
    return last_text, True


# ---------------------------------------------------------------------------
# REPL command helpers
# ---------------------------------------------------------------------------


def _repl_help() -> None:
    """Print available REPL commands."""
    fmt.info(
        "Available commands:\n"
        "  /help              Show this help message\n"
        "  /clear             Reset conversation to initial state\n"
        "  /compact [--drop]  Compress context (--drop removes middle turns)\n"
        "  /add-dir <path>    Grant read+write access to a directory\n"
        "  /exit, /quit       Exit the REPL"
    )


def _repl_clear(messages: list, thinking_state: ThinkingState) -> None:
    """Clear conversation history, keeping only the leading system messages."""
    leading = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            leading.append(msg)
        else:
            break

    dropped = len(messages) - len(leading)
    messages[:] = leading

    # Fully reset ThinkingState
    thinking_state.history.clear()
    thinking_state.branches.clear()
    thinking_state.note_count = 0
    if thinking_state.notes_dir is not None:
        try:
            notes_path = _safe_notes_path(thinking_state.notes_dir)
            notes_path.unlink(missing_ok=True)
        except ValueError:
            pass  # symlink escape — don't touch it

    fmt.info(f"context cleared ({dropped} messages removed)")


def _repl_add_dir(path_str: str, extra_write_roots: list) -> None:
    """Add a directory to the write-access whitelist."""
    path_str = path_str.strip()
    if not path_str:
        fmt.warning("/add-dir requires a path argument")
        return

    p = Path(path_str).expanduser().resolve()
    if not p.is_dir():
        fmt.warning(f"not a directory: {path_str}")
        return
    if p == Path(p.anchor):
        fmt.warning("cannot add filesystem root")
        return
    if p in extra_write_roots:
        fmt.info(f"already in whitelist: {p}")
        return

    extra_write_roots.append(p)
    fmt.info(f"added to whitelist: {p}")


def _repl_compact(
    messages: list, tools: list, context_length: int | None, arg: str
) -> None:
    """Manually compact conversation context."""
    before = estimate_tokens(messages, tools)

    messages[:] = compact_messages(messages)
    if arg.strip() == "--drop":
        messages[:] = drop_middle_turns(messages)

    after = estimate_tokens(messages, tools)
    saved = before - after
    fmt.info(f"compacted: {before} -> {after} tokens ({saved} saved)")


def repl_loop(
    messages: list,
    tools: list,
    *,
    api_base: str,
    model_id: str,
    max_turns: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    context_length: int | None,
    base_dir: str,
    thinking_state: ThinkingState,
    resolved_commands: dict,
    skills_catalog: dict,
    skill_read_roots: list,
    extra_write_roots: list,
    yolo: bool,
    verbose: bool,
    llm_kwargs: dict,
) -> None:
    """Interactive read-eval-print loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory

    history_path = os.path.join(base_dir, ".swival", "repl_history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(
        history=FileHistory(history_path),
        enable_history_search=True,
    )
    prompt_text = FormattedText([("bold fg:ansigreen", "swival> ")])

    if verbose:
        fmt.repl_banner()

    while True:
        try:
            print(file=sys.stderr)  # blank line before prompt
            line = session.prompt(prompt_text)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)  # newline after ^D / ^C
            break

        line = line.strip()
        if not line:
            continue

        # REPL commands — only intercept known commands; unknown /foo passes through
        if line in ("/exit", "/quit"):
            break

        cmd_parts = line.split(None, 1)
        cmd = cmd_parts[0].lower()
        cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd == "/help":
            _repl_help()
            continue
        elif cmd == "/clear":
            _repl_clear(messages, thinking_state)
            continue
        elif cmd == "/add-dir":
            _repl_add_dir(cmd_arg, extra_write_roots)
            continue
        elif cmd == "/compact":
            _repl_compact(messages, tools, context_length, cmd_arg)
            continue

        messages.append({"role": "user", "content": line})
        try:
            answer, exhausted = run_agent_loop(
                messages,
                tools,
                api_base=api_base,
                model_id=model_id,
                max_turns=max_turns,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                context_length=context_length,
                base_dir=base_dir,
                thinking_state=thinking_state,
                resolved_commands=resolved_commands,
                skills_catalog=skills_catalog,
                skill_read_roots=skill_read_roots,
                extra_write_roots=extra_write_roots,
                yolo=yolo,
                verbose=verbose,
                llm_kwargs=llm_kwargs,
            )
        except KeyboardInterrupt:
            fmt.warning("interrupted, question aborted.")
            continue

        if answer is not None:
            print(answer)
        if exhausted:
            fmt.warning("max turns reached for this question.")


if __name__ == "__main__":
    main()
