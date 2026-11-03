"""Reviewer mode: run swival as an LLM-as-judge reviewer."""

import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from time import monotonic

from .config import ONESHOT_MAX_TURNS
from .cost import SessionCost
from .report import AgentError
from .thinking import ThinkingState
from .todo import TodoState
from .tools import TOOLS, _history_path
from .worktree import READ_ONLY_AGENT_TOOLS, ToolPolicy, filter_tool_schemas

_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(ACCEPT|RETRY)\s*$", re.IGNORECASE)

REVIEW_PROMPT_TEMPLATE = """\
You are reviewing a coding agent's work.

{conversation_section}<task>
{task}
</task>

{verification_section}\
<answer>
{answer}
</answer>

Evaluate whether the answer correctly and completely addresses the task.
You have read-only tools to inspect files in the base directory. Read relevant
code before judging code changes; do not rely only on the agent's claims.
You cannot edit files or run commands, including tests.
{custom_instructions}
You MUST end your response with exactly one of these lines:
  VERDICT: ACCEPT
  VERDICT: RETRY

If RETRY, explain what needs to be fixed above the verdict line. Be specific and actionable."""


def _read_file(path: str, label: str) -> str:
    """Read a file, raising a descriptive error on failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise AgentError(f"cannot read {label} file: {e}")


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve a relative path against base_dir."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p)


def _parse_verdict(text: str) -> str | None:
    """Return 'ACCEPT' or 'RETRY' from the last VERDICT line, or None."""
    verdict = None
    for line in text.splitlines():
        m = _VERDICT_RE.match(line)
        if m:
            verdict = m.group(1).upper()
    return verdict


def _build_prompt(
    task: str,
    answer: str,
    verification: str | None = None,
    custom_instructions: str | None = None,
    conversation: str | None = None,
) -> str:
    """Build the review prompt from components."""
    conversation_section = ""
    if conversation:
        conversation_section = (
            f"<conversation>\n{conversation}\n</conversation>\n\n"
            "The conversation is context for the task. "
            "Answers in it are the agent's claims, not evidence.\n\n"
        )
    verification_section = ""
    if verification:
        verification_section = (
            f"<verification>\n{verification}\n</verification>\n\n"
            "The answer must satisfy these verification criteria.\n\n"
        )

    custom = ""
    if custom_instructions:
        custom = f"{custom_instructions}\n\n"

    return REVIEW_PROMPT_TEMPLATE.format(
        task=task,
        answer=answer,
        verification_section=verification_section,
        custom_instructions=custom,
        conversation_section=conversation_section,
    )


CONVERSATION_BUDGET = 4000
# Same ratio as _estimate_tokens(), so a rendered conversation never goes over.
_CHARS_PER_TOKEN = 4
_OLDER_ANSWER_CHARS = 480
_MIN_ANSWER_CHARS = 800
_EXCHANGE = "<user>\n{}\n</user>\n<agent>\n{}\n</agent>"
_CLIP_MARKER = "\n[...]\n"
_OPENING_MARKER = "\n[...]"


def _clip(text: str, chars: int) -> str:
    """Keep the head and tail of *text* within *chars*."""
    if len(text) <= chars:
        return text
    if chars < len(_CLIP_MARKER) + 2:
        return text[:chars]
    half = (chars - len(_CLIP_MARKER)) // 2
    return text[:half] + _CLIP_MARKER + text[-half:]


def _opening(text: str, chars: int) -> str:
    """Keep the opening lines of *text* within *chars*."""
    if len(text) <= chars:
        return text
    cut = text[: chars - len(_OPENING_MARKER)]
    if "\n" in cut:
        cut = cut[: cut.rindex("\n")]
    return cut.rstrip() + _OPENING_MARKER


def render_conversation(
    exchanges: list[tuple[str, str | None]], budget: int = CONVERSATION_BUDGET
) -> str | None:
    """Render earlier exchanges within *budget* tokens, oldest first.

    When they don't all fit, the oldest ones are left out.
    Returns None when there is nothing to show.
    """
    limit = budget * _CHARS_PER_TOKEN
    blocks: list[str] = []
    used = 0
    for i, (request, answer) in enumerate(reversed(exchanges)):
        answer = "(no answer)" if answer is None else answer
        if i == 0:
            room = max(0, limit - len(_EXCHANGE.format("", "")))
            floor = min(len(answer), _MIN_ANSWER_CHARS)
            request = _clip(request, max(0, room - floor))
            answer = _clip(answer, room - len(request))
        else:
            answer = _opening(answer, _OLDER_ANSWER_CHARS)
        block = _EXCHANGE.format(request, answer)
        cost = len(block) + (1 if blocks else 0)
        if blocks and used + cost > limit:
            break
        blocks.append(block)
        used += cost
    if not blocks:
        return None
    return "\n".join(reversed(blocks))


REVIEW_TIMEOUT = 3600
_EXIT_CODES = {"accept": 0, "retry": 1}


@dataclass
class ReviewVerdict:
    """What one review decided, and why when it failed (``error``)."""

    status: Literal["accept", "retry", "error"]
    text: str = ""
    error: str | None = None
    stderr: str = ""
    # Set by an external reviewer, which may exit with any code.
    exit_code: int | None = None

    def __post_init__(self):
        if self.exit_code is None:
            self.exit_code = _EXIT_CODES.get(self.status, 2)


@dataclass
class ReviewRequest:
    task: str
    answer: str
    review_round: int
    conversation: str | None = None


def _deadline_timer(seconds: float, flag: threading.Event) -> threading.Timer:
    timer = threading.Timer(seconds, flag.set)
    timer.daemon = True
    return timer


def review_answer(
    task: str,
    answer: str,
    *,
    base_dir: str,
    runtime: dict,
    verification: str | None = None,
    custom_instructions: str | None = None,
    deadline: float | None = None,
    conversation: str | None = None,
) -> ReviewVerdict:
    """Review *answer* against *task* with read-only access to *base_dir*.

    *runtime* holds the provider settings for ``run_agent_loop()``.
    Tools, file access and cancellation are fixed here, so a runtime key that
    tries to change them raises ``TypeError``.

    The *deadline* stops the review at its next step, but a provider request
    already sent still finishes.
    A verdict that arrives late counts as a timeout.

    The caller owns the secret shield and must destroy it.
    """
    from .agent import run_agent_loop

    prompt = _build_prompt(
        task=task,
        answer=answer,
        verification=verification,
        custom_instructions=custom_instructions,
        conversation=conversation,
    )
    verbose = runtime.get("verbose", False)
    cancel_flag = threading.Event()
    timer = expires_at = None
    if deadline is not None:
        expires_at = monotonic() + deadline
        timer = _deadline_timer(deadline, cancel_flag)
    try:
        # Started here, so an interrupt can't leave it running.
        if timer is not None:
            timer.start()
        response_text, exhausted = run_agent_loop(
            [{"role": "user", "content": prompt}],
            filter_tool_schemas(TOOLS, READ_ONLY_AGENT_TOOLS),
            base_dir=base_dir,
            thinking_state=ThinkingState(verbose=verbose),
            todo_state=TodoState(verbose=verbose),
            resolved_commands={},
            skills_catalog={},
            skill_read_roots=[],
            extra_write_roots=[],
            files_mode="some",
            continue_here=False,
            network_mode="provider-only",
            # Past answers are only the agent's own claims, so the reviewer shouldn't see them.
            tool_policy=ToolPolicy(
                allowed_tools=READ_ONLY_AGENT_TOOLS,
                hidden_paths=frozenset({_history_path(base_dir)}),
            ),
            cancel_flag=cancel_flag,
            **runtime,
        )
    except AgentError as e:
        return ReviewVerdict("error", error=f"LLM call failed: {e}")
    finally:
        if timer is not None:
            timer.cancel()

    # The timer can fire late, so check the clock too.
    if cancel_flag.is_set() or (expires_at is not None and monotonic() >= expires_at):
        return ReviewVerdict("error", error=f"review timed out after {deadline:g}s")
    response_text = response_text or ""
    if exhausted:
        return ReviewVerdict("error", error="review did not finish within its limits")
    verdict = _parse_verdict(response_text)
    if verdict is None:
        return ReviewVerdict(
            "error", text=response_text, error="no VERDICT found in LLM response"
        )
    return ReviewVerdict(verdict.lower(), text=response_text)


# What a self-review takes from the agent. New loop kwargs stay out unless listed.
_SHARED_RUNTIME_KEYS = (
    "api_base",
    "model_id",
    "max_output_tokens",
    "temperature",
    "top_p",
    "seed",
    "context_length",
    "secret_shield",
    "llm_filter",
    "session_cost",
)


def _reviewer_runtime(loop_kwargs: dict) -> dict:
    runtime = {k: loop_kwargs[k] for k in _SHARED_RUNTIME_KEYS if k in loop_kwargs}
    # Not the agent's session id: the review loop starts its own.
    runtime["llm_kwargs"] = {
        k: v for k, v in loop_kwargs["llm_kwargs"].items() if k != "session_id"
    }
    runtime["max_turns"] = ONESHOT_MAX_TURNS
    runtime["verbose"] = False
    return runtime


@dataclass
class ExternalReviewer:
    """A ``--reviewer`` command speaking the reviewer protocol."""

    command: str
    base_dir: str
    max_rounds: int
    encrypt_key: str | None = None

    def review(self, request: ReviewRequest, loop_kwargs: dict) -> ReviewVerdict:
        from .agent import run_reviewer
        from .secrets import ENCRYPT_KEY_ENV

        # Environment values can't hold NUL bytes, which answers sometimes do.
        env = {
            "SWIVAL_TASK": request.task.replace("\0", ""),
            "SWIVAL_REVIEW_ROUND": str(request.review_round),
        }
        if loop_kwargs.get("model_id"):
            env["SWIVAL_MODEL"] = loop_kwargs["model_id"]
        if request.conversation:
            env["SWIVAL_CONVERSATION"] = request.conversation.replace("\0", "")
        if self.encrypt_key:
            env[ENCRYPT_KEY_ENV] = self.encrypt_key
        code, stdout, stderr = run_reviewer(
            self.command,
            self.base_dir,
            request.answer,
            loop_kwargs.get("verbose", False),
            env_extra=env,
        )
        status = {v: k for k, v in _EXIT_CODES.items()}.get(code, "error")
        error = f"exited with code {code}" if status == "error" else None
        return ReviewVerdict(
            status, text=stdout, error=error, stderr=stderr, exit_code=code
        )


@dataclass
class SelfReviewer:
    """``--self-review``: the session's own model reviews in-process.

    It reads the runtime from the loop kwargs of each call, so it follows
    ``/profile``, ``/model``, ``/reasoning`` and routing.
    """

    base_dir: str
    max_rounds: int
    review_prompt: str | None = None
    verify: str | None = None

    def review(self, request: ReviewRequest, loop_kwargs: dict) -> ReviewVerdict:
        verification = None
        if self.verify:
            try:
                verification = _read_file(
                    _resolve_path(self.verify, self.base_dir), "--verify"
                )
            except AgentError as e:
                return ReviewVerdict("error", error=str(e))
        return review_answer(
            request.task,
            request.answer,
            base_dir=self.base_dir,
            runtime=_reviewer_runtime(loop_kwargs),
            verification=verification,
            custom_instructions=self.review_prompt,
            deadline=REVIEW_TIMEOUT,
            conversation=request.conversation,
        )


def run_as_reviewer(args, base_dir: str) -> int:
    """Execute reviewer mode. Returns exit code (0, 1, or 2)."""
    from .agent import resolve_provider

    # Read answer from stdin
    answer = sys.stdin.read()
    if not answer.strip():
        print("reviewer error: empty answer on stdin", file=sys.stderr)
        return 2

    # Get task description
    task = None
    if args.objective:
        obj_path = _resolve_path(args.objective, base_dir)
        try:
            task = _read_file(obj_path, "--objective")
        except AgentError as e:
            print(f"reviewer error: {e}", file=sys.stderr)
            return 2
    if not task:
        task = os.environ.get("SWIVAL_TASK")
    if not task:
        print(
            "reviewer error: no task description (set SWIVAL_TASK or use --objective)",
            file=sys.stderr,
        )
        return 2

    # Get optional verification criteria
    verification = None
    if args.verify:
        verify_path = _resolve_path(args.verify, base_dir)
        try:
            verification = _read_file(verify_path, "--verify")
        except AgentError as e:
            print(f"reviewer error: {e}", file=sys.stderr)
            return 2

    # Resolve provider
    try:
        model_id, api_base, api_key, context_length, llm_kwargs = resolve_provider(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_context_tokens=args.max_context_tokens,
            verbose=args.verbose,
            aws_profile=getattr(args, "aws_profile", None),
            project=getattr(args, "gcp_project", None),
            location=getattr(args, "location", None),
        )
    except (AgentError, SystemExit) as e:
        print(f"reviewer error: provider resolution failed: {e}", file=sys.stderr)
        return 2

    # Set up secret encryption if configured
    secret_shield = None
    if getattr(args, "encrypt_secrets", False):
        from .secrets import ENCRYPT_KEY_ENV, SecretShield

        key_hex = getattr(args, "encrypt_secrets_key", None)
        if not key_hex:
            key_hex = os.environ.get(ENCRYPT_KEY_ENV)
        secret_shield = SecretShield.from_config(
            key_hex=key_hex,
            tweak_str=getattr(args, "encrypt_secrets_tweak", None),
            extra_patterns=getattr(args, "encrypt_secrets_patterns", None),
        )

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        max_turns = ONESHOT_MAX_TURNS
    runtime = dict(
        api_base=api_base,
        model_id=model_id,
        max_turns=max_turns,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        context_length=context_length,
        verbose=args.verbose,
        llm_kwargs={
            **llm_kwargs,
            "api_key": api_key,
            "max_retries": getattr(args, "retries", 5),
            "provider_timeout": getattr(args, "provider_timeout", 900),
        },
        # The reviewer is a separate process, so it keeps its own subtotal.
        session_cost=SessionCost(),
        secret_shield=secret_shield,
    )

    try:
        verdict = review_answer(
            task,
            answer,
            base_dir=base_dir,
            runtime=runtime,
            verification=verification,
            custom_instructions=args.review_prompt,
            conversation=os.environ.get("SWIVAL_CONVERSATION") or None,
        )
    finally:
        if secret_shield is not None:
            secret_shield.destroy()

    if verdict.status == "error":
        print(f"reviewer error: {verdict.error}", file=sys.stderr)
    if verdict.text:
        print(verdict.text)
    return verdict.exit_code
