"""Regression test for the ChatGPT empty-output backfill patch.

ChatGPT streams each output item through an ``output_item.done`` event but
emits an empty ``output`` list on the final ``response.completed`` event. Since
litellm 1.90 a chat-completions call against a Responses-API model routes
through the responses streaming iterator, which keeps the completed event
verbatim, so the empty output reaches the chat transform and raises
"Unknown items in responses API response: []".

``_patch_chatgpt_responses_empty_output`` wraps the iterator's chunk processor
to fold the streamed items back into the completed response when it arrives
empty. These tests drive that wrapper with a stubbed original so they stay
deterministic and need no live LLM.
"""

import types

import pytest

from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIStreamEvents

from swival.agent import _patch_chatgpt_responses_empty_output


@pytest.fixture
def patched_iterator(monkeypatch):
    """Apply the patch over a stub original and yield a fake iterator factory.

    The stub mimics the real ``_process_chunk``: it returns the chunk and, on a
    completed event, records it as ``self.completed_response`` (line 249 of the
    real iterator). The class-level guard and method are restored afterwards.
    """

    def _stub_process_chunk(self, chunk):
        if getattr(chunk, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            self.completed_response = chunk
        return chunk

    monkeypatch.setattr(
        BaseResponsesAPIStreamingIterator,
        "_process_chunk",
        _stub_process_chunk,
        raising=False,
    )
    monkeypatch.setattr(
        BaseResponsesAPIStreamingIterator, "_swival_patched", False, raising=False
    )

    _patch_chatgpt_responses_empty_output()
    process = BaseResponsesAPIStreamingIterator._process_chunk

    def _new_iterator():
        it = types.SimpleNamespace(completed_response=None)
        return it, process

    yield _new_iterator


def _item(text, index, item_type="message"):
    payload = {
        "type": item_type,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return types.SimpleNamespace(model_dump=lambda payload=payload: dict(payload))


def _output_item_done(text, index):
    return types.SimpleNamespace(
        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        output_index=index,
        item=_item(text, index),
    )


def _completed(output):
    response = types.SimpleNamespace(output=output)
    return types.SimpleNamespace(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=response
    )


def test_backfills_empty_completed_response(patched_iterator):
    it, process = patched_iterator()
    process(it, _output_item_done("pong", 0))
    process(it, _completed([]))

    output = it.completed_response.response.output
    assert len(output) == 1
    assert output[0]["content"][0]["text"] == "pong"


def test_preserves_streaming_order(patched_iterator):
    it, process = patched_iterator()
    process(it, _output_item_done("second", 1))
    process(it, _output_item_done("first", 0))
    process(it, _completed([]))

    output = it.completed_response.response.output
    assert [o["content"][0]["text"] for o in output] == ["first", "second"]


def test_recovered_items_are_plain_dicts(patched_iterator):
    it, process = patched_iterator()
    process(it, _output_item_done("pong", 0))
    process(it, _completed([]))

    # The chat transform only converts raw dicts via its raw-dict callback;
    # pydantic objects would be dropped and re-raise "Unknown items".
    assert all(isinstance(o, dict) for o in it.completed_response.response.output)


def test_leaves_populated_completed_response_untouched(patched_iterator):
    it, process = patched_iterator()
    process(it, _output_item_done("ignored", 0))
    existing = [
        {"type": "message", "content": [{"type": "output_text", "text": "real"}]}
    ]
    process(it, _completed(existing))

    assert it.completed_response.response.output is existing


def test_no_streamed_items_leaves_output_empty(patched_iterator):
    it, process = patched_iterator()
    process(it, _completed([]))

    assert it.completed_response.response.output == []
