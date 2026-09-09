"""Literal text must remain valid input to token budgets."""

import subprocess
import sys

import pytest
import tiktoken

from swival._msg import _msg_content
from swival.agent import _call_command, estimate_tokens, load_memory
from swival.tokens import count_tokens, truncate_to_tokens


@pytest.mark.parametrize(
    "text",
    [
        "ordinary text café",
        *sorted(tiktoken.get_encoding("cl100k_base").special_tokens_set),
    ],
)
def test_literal_text_token_budget(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    expected = encoder.encode_ordinary(text)
    assert count_tokens(text) == len(expected)
    assert truncate_to_tokens(text, len(expected)) == text
    assert truncate_to_tokens(text, 1) == encoder.decode(expected[:1])


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "Explain <|endoftext|>"},
        {"role": "assistant", "reasoning_content": "Explain <|endoftext|>"},
        {
            "role": "tool",
            "content": [{"type": "text", "text": "Explain <|endoftext|>"}],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": "<|endoftext|>"}}
            ],
        },
    ],
)
def test_message_token_budget_accepts_literal_special_tokens(message):
    assert estimate_tokens([message]) > 4


def test_tool_schema_token_budget_accepts_literal_special_tokens():
    tools = [
        {
            "type": "function",
            "function": {"name": "test", "description": "<|endoftext|>"},
        }
    ]
    assert estimate_tokens([], tools) > 0


def test_memory_with_literal_special_token(tmp_path):
    path = tmp_path / ".swival" / "memory" / "MEMORY.md"
    path.parent.mkdir(parents=True)
    path.write_text("## Tokenization\nLiteral <|endoftext|> appears in output.\n")
    assert "Literal <|endoftext|> appears in output." in load_memory(str(tmp_path))


def test_command_output_with_literal_special_token(monkeypatch):
    monkeypatch.setattr("swival.agent._run_command_once", lambda *args: "<|endoftext|>")
    message, reason = _call_command("test-command", [], False, max_output_tokens=100)
    assert _msg_content(message) == "<|endoftext|>"
    assert reason == "stop"


def test_offline_tokenizer_supports_memory_and_command_output(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from unittest.mock import patch

with patch("tiktoken.get_encoding", side_effect=OSError("offline")):
    from swival.agent import estimate_tokens, load_memory, _call_command
    from swival._msg import _msg_content

    base = Path(sys.argv[1])
    assert load_memory(str(base)) == ""
    memory = base / ".swival" / "memory" / "MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("## Fact\\nCafé uses UTF-8.\\n", encoding="utf-8")
    assert "Café uses UTF-8." in load_memory(str(base))

    from swival.tokens import count_tokens, truncate_to_tokens

    text = "Café 🙂 <|endoftext|>"
    assert count_tokens(text) == len(text.encode("utf-8"))
    assert estimate_tokens([{"role": "user", "content": text}]) == count_tokens(text) + 4
    for budget in range(count_tokens(text) + 1):
        truncated = truncate_to_tokens(text, budget)
        assert text.startswith(truncated)
        assert count_tokens(truncated) <= budget
    assert truncate_to_tokens(text, count_tokens(text)) == text

    with patch("swival.agent._run_command_once", return_value=text):
        message, reason = _call_command("test-command", [], False, max_output_tokens=10)
    assert _msg_content(message) == truncate_to_tokens(text, 10)
    assert reason == "stop"
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
