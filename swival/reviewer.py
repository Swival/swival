"""Reviewer mode: run swival as an LLM-as-judge reviewer."""

import os
import re
import sys
from pathlib import Path

from .report import AgentError

_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(ACCEPT|RETRY)\s*$", re.IGNORECASE)

REVIEW_PROMPT_TEMPLATE = """\
You are reviewing a coding agent's work.

<task>
{task}
</task>

{verification_section}\
<answer>
{answer}
</answer>

Evaluate whether the answer correctly and completely addresses the task.
{custom_instructions}
You MUST end your response with exactly one of these lines:
  VERDICT: ACCEPT
  VERDICT: RETRY

If RETRY, explain what needs to be fixed above the verdict line. Be specific and actionable."""


def _read_file(path: str, label: str) -> str:
    """Read a file, raising a descriptive error on failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
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
) -> str:
    """Build the review prompt from components."""
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
    )


def run_as_reviewer(args, base_dir: str) -> int:
    """Execute reviewer mode. Returns exit code (0, 1, or 2)."""
    from .agent import call_llm, resolve_provider

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

    # Build the review prompt
    prompt = _build_prompt(
        task=task,
        answer=answer,
        verification=verification,
        custom_instructions=args.review_prompt,
    )

    # Resolve provider and call LLM
    try:
        model_id, api_base, api_key, context_length, llm_kwargs = resolve_provider(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_context_tokens=args.max_context_tokens,
            verbose=args.verbose,
        )
    except (AgentError, SystemExit) as e:
        print(f"reviewer error: provider resolution failed: {e}", file=sys.stderr)
        return 2

    messages = [{"role": "user", "content": prompt}]

    try:
        msg, finish_reason = call_llm(
            api_base,
            model_id,
            messages,
            args.max_output_tokens,
            args.temperature,
            args.top_p,
            args.seed,
            None,  # no tools
            args.verbose,
            provider=llm_kwargs.get("provider", args.provider),
            api_key=api_key,
        )
    except AgentError as e:
        print(f"reviewer error: LLM call failed: {e}", file=sys.stderr)
        return 2

    response_text = msg.content or ""

    # Parse verdict
    verdict = _parse_verdict(response_text)

    if verdict == "ACCEPT":
        print(response_text)
        return 0
    elif verdict == "RETRY":
        print(response_text)
        return 1
    else:
        print("reviewer error: no VERDICT found in LLM response", file=sys.stderr)
        print(response_text)
        return 2
