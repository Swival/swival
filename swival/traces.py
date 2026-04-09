"""JSONL trace export compatible with HuggingFace agent-traces format."""

import json
import os
import uuid
from datetime import datetime, timezone
from importlib import metadata

from ._msg import _msg_get, _msg_role, _msg_content, _msg_tool_calls


def _swival_version() -> str:
    try:
        return metadata.version("swival")
    except metadata.PackageNotFoundError:
        return "unknown"


def write_trace(
    messages: list,
    *,
    path: str,
    session_id: str,
    base_dir: str,
    model: str,
    task: str | None = None,
    secret_shield=None,
) -> None:
    enc = secret_shield.encrypt_text if secret_shield is not None else (lambda s: s)
    enc_obj = secret_shield.encrypt_obj if secret_shield is not None else (lambda o: o)

    version = _swival_version()
    last_uuid = None
    lines: list[dict] = []

    def _append(type_: str, **kwargs) -> str:
        nonlocal last_uuid
        uid = str(uuid.uuid4())
        line = {
            "uuid": uid,
            "parentUuid": last_uuid,
            "sessionId": session_id,
            "harness": "swival",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": version,
            "cwd": base_dir,
            "type": type_,
            "isSidechain": False,
            "userType": "external",
            **kwargs,
        }
        lines.append(line)
        last_uuid = uid
        return uid

    for msg in messages:
        role = _msg_role(msg)

        if role == "system":
            _append(
                "system", content=enc(_msg_content(msg)), level="info", isMeta=False
            )

        elif role == "user":
            content = _msg_content(msg)
            _append(
                "user",
                message={"role": "user", "content": enc(content)},
                promptId=str(uuid.uuid4()),
            )

        elif role == "assistant":
            tool_calls = _msg_tool_calls(msg)
            content_blocks = []

            text = _msg_content(msg)
            if text:
                content_blocks.append({"type": "text", "text": enc(text)})

            if tool_calls:
                for tc in tool_calls:
                    fn = _msg_get(tc, "function")
                    raw_args = _msg_get(fn, "arguments", "{}")
                    try:
                        parsed = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed = {"_raw": raw_args}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": _msg_get(tc, "id"),
                            "name": _msg_get(fn, "name"),
                            "input": enc_obj(parsed),
                        }
                    )

            stop = "tool_use" if tool_calls else "end_turn"
            usage = _msg_get(msg, "usage")
            if usage and not isinstance(usage, dict):
                usage = {
                    k: getattr(usage, k, None)
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                    )
                    if getattr(usage, k, None) is not None
                }

            _append(
                "assistant",
                message={
                    "model": model,
                    "id": _msg_get(msg, "id") or str(uuid.uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": content_blocks,
                    "stop_reason": stop,
                    "stop_sequence": None,
                    "usage": usage or {},
                },
            )

        elif role == "tool":
            raw = _msg_content(msg)
            tc_id = _msg_get(msg, "tool_call_id")
            is_error = raw.startswith("error:")
            encrypted_raw = enc(raw)
            _append(
                "user",
                message={
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc_id,
                            "content": encrypted_raw,
                            "is_error": is_error,
                        }
                    ],
                },
                toolUseResult=encrypted_raw,
            )

    if task:
        lines.append(
            {"type": "last-prompt", "lastPrompt": enc(task), "sessionId": session_id}
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_trace_to_dir(
    messages: list,
    *,
    trace_dir: str,
    base_dir: str,
    model: str,
    task: str | None = None,
    session_id: str | None = None,
    verbose: bool = False,
    secret_shield=None,
) -> None:
    if not messages:
        return
    if session_id is None:
        session_id = str(uuid.uuid4())
    path = os.path.join(trace_dir, f"{session_id}.jsonl")

    from .secrets import SecretShield

    with SecretShield.ensure(secret_shield) as shield:
        try:
            write_trace(
                messages,
                path=path,
                session_id=session_id,
                base_dir=base_dir,
                model=model,
                task=task,
                secret_shield=shield,
            )
            if verbose:
                from . import fmt

                fmt.info(f"Trace written to {path}")
        except OSError as exc:
            from . import fmt

            fmt.error(f"Failed to write trace to {path}: {exc}")
