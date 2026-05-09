"""Compatibility patches for LiteLLM's ChatGPT subscription adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any


StripSseData = Callable[[str], str | None]


def _sse_json_events(body_text: str, strip_sse_data: StripSseData) -> Iterator[dict]:
    for line in body_text.splitlines():
        stripped = strip_sse_data(line)
        if not stripped:
            continue
        stripped = stripped.strip()
        if not stripped:
            continue
        if stripped == "[DONE]":
            break
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            yield parsed


def _event_index(event: dict, fallback: int) -> int:
    raw = event.get("output_index", fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _content_index(event: dict) -> int:
    raw = event.get("content_index", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def output_items_from_sse(body_text: str, strip_sse_data: StripSseData) -> list[dict]:
    """Recover Responses API output items from ChatGPT SSE events."""
    added_items: dict[int, dict] = {}
    done_items: dict[int, dict] = {}
    text_chunks: dict[tuple[int, int], list[str]] = {}
    text_done: dict[tuple[int, int], str] = {}
    text_annotations: dict[tuple[int, int], list[Any]] = {}
    arg_chunks: dict[int, list[str]] = {}
    arg_done: dict[int, str] = {}

    for event in _sse_json_events(body_text, strip_sse_data):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                added_items[_event_index(event, len(added_items))] = dict(item)
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                done_items[_event_index(event, len(done_items))] = dict(item)
        elif event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                key = (_event_index(event, 0), _content_index(event))
                text_chunks.setdefault(key, []).append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str):
                key = (_event_index(event, 0), _content_index(event))
                text_done[key] = text
        elif event_type == "response.content_part.done":
            part = event.get("part")
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                key = (_event_index(event, 0), _content_index(event))
                if isinstance(text, str):
                    text_done[key] = text
                annotations = part.get("annotations")
                if isinstance(annotations, list):
                    text_annotations[key] = annotations
        elif event_type == "response.function_call_arguments.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                idx = _event_index(event, 0)
                arg_chunks.setdefault(idx, []).append(delta)
        elif event_type == "response.function_call_arguments.done":
            arguments = event.get("arguments")
            if isinstance(arguments, str):
                arg_done[_event_index(event, 0)] = arguments

    def text_content(output_index: int) -> list[dict]:
        keys = sorted(
            key for key in set(text_chunks) | set(text_done) if key[0] == output_index
        )
        return [
            {
                "type": "output_text",
                "text": text_done.get(key, "".join(text_chunks.get(key, []))),
                "annotations": text_annotations.get(key, []),
            }
            for key in keys
        ]

    def fill_item(output_index: int, item: dict) -> dict:
        item = dict(item)
        if item.get("type") == "message":
            content = item.get("content")
            has_text = (
                isinstance(content, list)
                and any(
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and part.get("text")
                    for part in content
                )
            )
            recovered_content = text_content(output_index)
            if recovered_content and not has_text:
                item["content"] = recovered_content
            item.setdefault("role", "assistant")
        elif item.get("type") == "function_call":
            arguments = arg_done.get(output_index)
            if arguments is None and output_index in arg_chunks:
                arguments = "".join(arg_chunks[output_index])
            if arguments is not None and not item.get("arguments"):
                item["arguments"] = arguments
        return item

    source_items = {**added_items, **done_items}
    for output_index in {key[0] for key in set(text_chunks) | set(text_done)}:
        if output_index not in source_items:
            source_items[output_index] = {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": text_content(output_index),
            }

    return [
        fill_item(output_index, source_items[output_index])
        for output_index in sorted(source_items)
    ]


def patch_chatgpt_responses_empty_output() -> None:
    """Patch LiteLLM to recover ChatGPT SSE output when completion output is empty."""
    try:
        from litellm.llms.chatgpt.responses.transformation import (
            ChatGPTResponsesAPIConfig,
        )
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    except ImportError:
        return

    if getattr(ChatGPTResponsesAPIConfig, "_swival_patched", False):
        return
    ChatGPTResponsesAPIConfig._swival_patched = True

    original = ChatGPTResponsesAPIConfig.transform_response_api_response
    strip_sse_data = CustomStreamWrapper._strip_sse_data_from_chunk

    def patched_transform(self, model, raw_response, logging_obj):
        result = original(self, model, raw_response, logging_obj)
        if getattr(result, "output", None):
            return result

        body_text = getattr(raw_response, "text", None) or ""
        output_items = output_items_from_sse(body_text, strip_sse_data)
        if output_items:
            result.output = output_items
        return result

    ChatGPTResponsesAPIConfig.transform_response_api_response = patched_transform
