from swival._chatgpt_compat import output_items_from_sse


def _strip_sse_test_line(line):
    return line[5:] if line.startswith("data:") else ""


def test_recovers_output_item_done():
    body = (
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","output_index":0,'
        '"item":{"type":"message","role":"assistant","content":'
        '[{"type":"output_text","text":"ok"}]}}\n'
    )

    assert output_items_from_sse(body, _strip_sse_test_line) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ]


def test_recovers_text_when_completed_output_is_empty():
    body = (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","output_index":0,'
        '"content_index":0,"delta":"Hello "}\n'
        'event: response.output_text.done\n'
        'data: {"type":"response.output_text.done","output_index":0,'
        '"content_index":0,"text":"Hello world"}\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"output":[]}}\n'
    )

    assert output_items_from_sse(body, _strip_sse_test_line) == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello world",
                    "annotations": [],
                }
            ],
        }
    ]


def test_fills_empty_message_content_from_text_stream():
    body = (
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","output_index":0,'
        '"item":{"type":"message","role":"assistant","content":[]}}\n'
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","output_index":0,'
        '"content_index":0,"delta":"Recovered"}\n'
    )

    assert output_items_from_sse(body, _strip_sse_test_line) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Recovered", "annotations": []}
            ],
        }
    ]


def test_recovers_function_call_arguments_from_stream():
    body = (
        'event: response.output_item.added\n'
        'data: {"type":"response.output_item.added","output_index":0,'
        '"item":{"type":"function_call","call_id":"call_1","name":"read_file",'
        '"arguments":""}}\n'
        'event: response.function_call_arguments.delta\n'
        'data: {"type":"response.function_call_arguments.delta",'
        '"output_index":0,"delta":"{\\"path\\":"}\n'
        'event: response.function_call_arguments.delta\n'
        'data: {"type":"response.function_call_arguments.delta",'
        '"output_index":0,"delta":"\\"README.md\\"}"}\n'
    )

    assert output_items_from_sse(body, _strip_sse_test_line) == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]
