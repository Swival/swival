"""Regression tests for Gemini thought_signature preservation (issue #13).

Ensures that request-shaped extras (extra_content) on tool_calls survive
canonicalization, history writeback, and filter serialization.
"""

import types

from swival._msg import _canonicalize_tool_calls, _find_current_turn_boundary
from swival.agent import _msg_to_dict
from swival.filter import _message_to_dict as _filter_message_to_dict


class TestCurrentTurnBoundary:
    def test_finds_last_user_message(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "a2"},
        ]
        assert _find_current_turn_boundary(messages) == 3

    def test_skips_swival_synthetic_nudges(self):
        """Synthetic user nudges (think, todo, guardrail) must not become
        the turn boundary — they are injected mid-turn by the agent loop."""
        messages = [
            {"role": "user", "content": "real question"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig1"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "user",
                "content": "IMPORTANT: You have repeated the same error...",
                "_swival_synthetic": True,
            },
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig2"}},
                    }
                ],
            },
        ]
        assert _find_current_turn_boundary(messages) == 0

    def test_skips_always_synthetic_prefixes(self):
        """Bracket-prefixed content that is always synthetic by construction."""
        messages = [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "[image] screenshot.png"},
            {"role": "assistant", "content": "a2"},
        ]
        assert _find_current_turn_boundary(messages) == 0

    def test_real_bracket_user_message_is_boundary(self):
        """A real user message that starts with [ is a valid turn boundary."""
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "[1, 2, 3] convert this to yaml"},
            {"role": "assistant", "content": "a2"},
        ]
        assert _find_current_turn_boundary(messages) == 2

    def test_returns_zero_with_no_user_messages(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
        ]
        assert _find_current_turn_boundary(messages) == 0


class TestCanonicalizePreservesCurrentTurnExtras:
    def test_both_current_turn_steps_keep_extra_content(self):
        """All current-turn tool calls preserve extra_content."""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "do_it", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig1"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "do_more", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig2"}},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        tc1 = messages[2]["tool_calls"][0]
        assert tc1.get("extra_content") == {"google": {"thought_signature": "sig1"}}

        tc2 = messages[4]["tool_calls"][0]
        assert tc2.get("extra_content") == {"google": {"thought_signature": "sig2"}}

    def test_prior_turn_tool_calls_are_stripped(self):
        """Tool calls before the current user turn lose extra_content."""
        messages = [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "old",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "old_tool", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "old_sig"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "ok"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "second question"},
            {
                "role": "assistant",
                "content": "new",
                "tool_calls": [
                    {
                        "id": "call_new",
                        "type": "function",
                        "function": {"name": "new_tool", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "new_sig"}},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        assert "extra_content" not in messages[1]["tool_calls"][0]
        assert messages[5]["tool_calls"][0].get("extra_content") == {
            "google": {"thought_signature": "new_sig"}
        }

    def test_other_provider_extras_still_stripped_in_current_turn(self):
        """Non-request-shaped fields like provider_specific_fields and index
        are stripped even on current-turn tool calls."""
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                        "provider_specific_fields": {"thought_signature": "x"},
                        "index": 0,
                        "extra_content": {"google": {"thought_signature": "sig"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        tc = messages[1]["tool_calls"][0]
        assert "provider_specific_fields" not in tc
        assert "index" not in tc
        assert tc.get("extra_content") == {"google": {"thought_signature": "sig"}}

    def test_namespace_tool_call_preserves_extra_content(self):
        """Namespace-style tool calls in current turn also preserve extra_content."""
        tc_obj = types.SimpleNamespace(
            id="call_ns",
            type="function",
            function=types.SimpleNamespace(name="tool_ns", arguments="{}"),
            extra_content={"google": {"thought_signature": "ns_sig"}},
        )
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [tc_obj],
            },
            {"role": "tool", "tool_call_id": "call_ns", "content": "ok"},
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        tc = messages[1]["tool_calls"][0]
        assert isinstance(tc, dict)
        assert tc["extra_content"] == {"google": {"thought_signature": "ns_sig"}}

    def test_synthetic_nudge_does_not_split_turn(self):
        """A synthetic user nudge mid-turn must not cause earlier tool calls
        to lose their extra_content. This is the exact scenario from the
        agent loop: real user -> tool_call(sig1) -> synthetic nudge -> tool_call(sig2)."""
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig1"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "user",
                "content": "IMPORTANT: The same tool error has occurred 2 times.",
                "_swival_synthetic": True,
            },
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig2"}},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        tc1 = messages[1]["tool_calls"][0]
        assert tc1.get("extra_content") == {"google": {"thought_signature": "sig1"}}, (
            "synthetic nudge must not cause prior tool call to lose extra_content"
        )

        tc2 = messages[4]["tool_calls"][0]
        assert tc2.get("extra_content") == {"google": {"thought_signature": "sig2"}}

    def test_no_extra_content_is_fine(self):
        """Tool calls without extra_content work normally (non-Google providers)."""
        messages = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "a",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
        ]
        _canonicalize_tool_calls(messages)

        tc = messages[1]["tool_calls"][0]
        assert "extra_content" not in tc
        assert tc == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "t", "arguments": "{}"},
        }


class TestCallLlmStripsSwivalSynthetic:
    """Regression: _swival_synthetic must not reach the provider API (HuggingFace
    rejects extra fields) but must remain on the live transcript for
    _is_synthetic() / _find_current_turn_boundary()."""

    def test_outbound_messages_lack_synthetic_marker(self):
        """The messages list passed to litellm.completion must not contain
        _swival_synthetic, but the caller's original list must keep it."""
        from unittest.mock import patch
        from swival import agent

        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "real question"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "user",
                "content": "Your response was empty. Please continue working on the task using the available tools.",
                "_swival_synthetic": True,
            },
            {"role": "assistant", "content": "done"},
        ]

        captured = {}

        def fake_completion(**kwargs):
            captured["messages"] = kwargs["messages"]

            class FakeChoice:
                def __init__(self):
                    self.message = types.SimpleNamespace(
                        role="assistant", content="ok", tool_calls=None
                    )
                    self.finish_reason = "stop"

            class FakeResponse:
                choices = [FakeChoice()]
                usage = types.SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    prompt_tokens_details=None,
                )

            return FakeResponse()

        with patch("litellm.completion", fake_completion):
            agent.call_llm(
                base_url="http://localhost:1234",
                model_id="test-model",
                messages=original,
                max_output_tokens=100,
                temperature=None,
                top_p=None,
                seed=None,
                tools=None,
                verbose=False,
                provider="generic",
                api_key="test",
            )

        # Outbound must not contain _swival_synthetic
        for msg in captured["messages"]:
            if isinstance(msg, dict):
                assert "_swival_synthetic" not in msg, (
                    f"_swival_synthetic leaked to provider: {msg}"
                )

        # Original transcript must still have it
        assert original[4].get("_swival_synthetic") is True

    def test_synthetic_continuation_survives_for_turn_boundary(self):
        """After call_llm, a synthetic continuation prompt must still be
        recognized by _is_synthetic() so _find_current_turn_boundary()
        does not treat it as a real user turn."""
        from swival._msg import _is_synthetic, _find_current_turn_boundary

        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "step1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig1"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "user",
                "content": "Your response was cut off. Please use the provided tools to complete the task step by step.",
                "_swival_synthetic": True,
            },
            {
                "role": "assistant",
                "content": "step2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig2"}},
                    }
                ],
            },
        ]

        # The synthetic continuation must still be recognized
        assert _is_synthetic(messages[3])

        # Turn boundary must be the real user message, not the synthetic one
        assert _find_current_turn_boundary(messages) == 0

        # Therefore canonicalization must preserve extra_content on both tool calls
        from swival._msg import _canonicalize_tool_calls

        _canonicalize_tool_calls(messages)

        tc1 = messages[1]["tool_calls"][0]
        assert tc1.get("extra_content") == {"google": {"thought_signature": "sig1"}}

        tc2 = messages[4]["tool_calls"][0]
        assert tc2.get("extra_content") == {"google": {"thought_signature": "sig2"}}


class TestMsgToDictPreservesToolCallExtras:
    def test_strips_provider_specific_fields_preserves_tool_call_extra_content(self):
        """_msg_to_dict strips message-level provider_specific_fields but
        tool-call extra_content survives through model_dump."""

        class FakeMsg:
            def __init__(self):
                self.role = "assistant"
                self.content = "check"
                self.tool_calls = None

            def model_dump(self, exclude_none=False):
                return {
                    "role": self.role,
                    "content": self.content,
                    "provider_specific_fields": {"something": "internal"},
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "t", "arguments": "{}"},
                            "extra_content": {"google": {"thought_signature": "sig"}},
                        }
                    ],
                }

        d = _msg_to_dict(FakeMsg())
        assert "provider_specific_fields" not in d
        assert d["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "sig"}
        }


class TestFilterPreservesToolCallExtras:
    def test_dict_tool_call_extra_content(self):
        """Filter serializer preserves extra_content on dict tool calls."""
        msg = {
            "role": "assistant",
            "content": "need tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_it", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "filter_sig"}},
                }
            ],
        }
        d = _filter_message_to_dict(msg)
        assert d["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "filter_sig"}
        }

    def test_namespace_tool_call_extra_content(self):
        """Filter serializer preserves extra_content on namespace tool calls."""
        tc = types.SimpleNamespace(
            id="call_ns",
            type="function",
            function=types.SimpleNamespace(name="tool_ns", arguments="{}"),
            extra_content={"google": {"thought_signature": "ns_sig"}},
        )
        msg = types.SimpleNamespace(
            role="assistant",
            content="need tool",
            tool_calls=[tc],
        )
        d = _filter_message_to_dict(msg)
        assert d["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "ns_sig"}
        }

    def test_no_extra_content_omitted(self):
        """Filter serializer does not add extra_content when absent."""
        msg = {
            "role": "assistant",
            "content": "need tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_it", "arguments": "{}"},
                }
            ],
        }
        d = _filter_message_to_dict(msg)
        assert "extra_content" not in d["tool_calls"][0]
