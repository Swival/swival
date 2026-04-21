"""Integration tests for secret encryption in call_llm and Session."""

import types
from unittest.mock import MagicMock

import pytest

from swival.secrets import SecretShield

# Valid GitHub PAT for testing (ghp_ + 36 alphanumeric chars)
GHP_TOKEN = "ghp_" + "A" * 36


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(content="ok", tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump = lambda exclude_none=True: {"content": content, "role": "assistant"}
    return msg


def _make_response(content="ok", finish_reason="stop", tool_calls=None):
    choice = types.SimpleNamespace()
    choice.message = _make_message(content=content, tool_calls=tool_calls)
    choice.finish_reason = finish_reason
    resp = types.SimpleNamespace()
    resp.choices = [choice]
    return resp


@pytest.fixture
def shield():
    s = SecretShield(key=b"\x00" * 32)
    yield s
    if not s.destroyed:
        s.destroy()


# ---------------------------------------------------------------------------
# call_llm integration
# ---------------------------------------------------------------------------


class TestCallLlmEncryption:
    def test_encrypted_messages_sent_to_litellm(self, shield, monkeypatch):
        """Verify messages sent to litellm.completion have encrypted tokens."""
        import litellm
        from swival import agent, fmt

        fmt.init(color=False)

        captured_messages = []

        def fake_completion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_response("The token ghp_ABCDEF12 was used.")

        monkeypatch.setattr(litellm, "completion", fake_completion)

        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": f"Use {GHP_TOKEN} to clone."},
        ]

        msg, finish, *_ = agent.call_llm(
            "http://fake",
            "test-model",
            messages,
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="fake",
            secret_shield=shield,
        )

        # LiteLLM received encrypted messages
        assert len(captured_messages) == 1
        sent = captured_messages[0]
        user_content = sent[1]["content"]
        assert "ghp_" in user_content
        assert GHP_TOKEN not in user_content  # original token not sent

        # Original messages not mutated
        assert messages[1]["content"] == f"Use {GHP_TOKEN} to clone."

    def test_response_decrypted_for_caller(self, shield, monkeypatch):
        """Verify response content is decrypted before returning."""
        import litellm
        from swival import agent, fmt

        fmt.init(color=False)

        # First encrypt so the registry has entries
        shield.encrypt_messages([{"role": "user", "content": f"token: {GHP_TOKEN}"}])
        # Find the encrypted token
        ct = None
        for k in shield._registry:
            ct = k
            break
        assert ct is not None

        def fake_completion(**kwargs):
            # Model echoes back the encrypted token
            return _make_response(f"Found {ct}")

        monkeypatch.setattr(litellm, "completion", fake_completion)

        msg, finish, *_ = agent.call_llm(
            "http://fake",
            "test-model",
            [{"role": "user", "content": "test"}],
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="fake",
            secret_shield=shield,
        )

        # Response should contain the decrypted token
        assert GHP_TOKEN in msg.content

    def test_tool_call_args_decrypted(self, shield, monkeypatch):
        """Verify tool_call arguments are decrypted in the response."""
        import litellm
        from swival import agent, fmt

        fmt.init(color=False)

        # Encrypt to populate registry
        shield.encrypt_messages([{"role": "user", "content": f"key: {GHP_TOKEN}"}])
        ct = list(shield._registry.keys())[0]

        tc = types.SimpleNamespace()
        tc.id = "tc_1"
        tc.type = "function"
        tc.function = types.SimpleNamespace()
        tc.function.name = "write_file"
        tc.function.arguments = f'{{"content": "{ct}"}}'

        def fake_completion(**kwargs):
            return _make_response(content=None, tool_calls=[tc])

        monkeypatch.setattr(litellm, "completion", fake_completion)

        msg, finish, *_ = agent.call_llm(
            "http://fake",
            "test-model",
            [{"role": "user", "content": "test"}],
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="fake",
            secret_shield=shield,
        )

        assert GHP_TOKEN in msg.tool_calls[0].function.arguments

    def test_cache_disabled_with_shield(self, shield, monkeypatch):
        """Verify cache is disabled when secret_shield is active."""
        import litellm
        from swival import agent, fmt

        fmt.init(color=False)

        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=None)
        mock_cache.put = MagicMock()

        def fake_completion(**kwargs):
            return _make_response("ok")

        monkeypatch.setattr(litellm, "completion", fake_completion)

        agent.call_llm(
            "http://fake",
            "test-model",
            [{"role": "user", "content": "test"}],
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="fake",
            secret_shield=shield,
            cache=mock_cache,
        )

        # Cache should not have been consulted at all
        mock_cache.get.assert_not_called()
        mock_cache.put.assert_not_called()

    def test_command_provider_bypasses_encryption(self, shield, monkeypatch):
        """Command provider should not encrypt (local subprocess)."""
        from swival import agent

        call_count = 0

        def fake_call_command(model_id, messages, verbose, max_output_tokens):
            nonlocal call_count
            call_count += 1
            # Verify messages are NOT encrypted (contain original tokens)
            user_msg = [m for m in messages if m["role"] == "user"][0]
            assert GHP_TOKEN in user_msg["content"]
            return _make_message("done"), "stop"

        monkeypatch.setattr(agent, "_call_command", fake_call_command)

        messages = [{"role": "user", "content": f"Use {GHP_TOKEN}"}]

        msg, finish, *_ = agent.call_llm(
            "http://fake",
            "cmd",
            messages,
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="command",
            secret_shield=shield,
        )

        assert call_count == 1

    def test_sanitization_runs_before_encryption(self, shield, monkeypatch):
        """_sanitize_assistant_messages runs before encryption copy."""
        import litellm
        from swival import agent, fmt

        fmt.init(color=False)

        def fake_completion(**kwargs):
            return _make_response("ok")

        monkeypatch.setattr(litellm, "completion", fake_completion)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": None, "tool_calls": None},
            {"role": "user", "content": "hello"},
        ]

        agent.call_llm(
            "http://fake",
            "test-model",
            messages,
            1024,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="fake",
            secret_shield=shield,
        )

        # The empty assistant message should have been sanitized in the
        # canonical list (content set to "")
        assert messages[1]["content"] == ""


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_session_exit_destroys_shield(self):
        """Session.__exit__() should call destroy() on the shield."""
        shield = SecretShield(key=b"\x00" * 32)
        assert not shield.destroyed

        from swival.session import Session

        session = Session(base_dir="/tmp")
        session._secret_shield = shield
        session.__exit__(None, None, None)

        assert shield.destroyed
        assert session._secret_shield is None


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_encrypt_secrets_key_from_env(self, monkeypatch):
        """SWIVAL_ENCRYPT_KEY env var enables encryption and sets key."""
        from swival.config import args_to_session_kwargs

        monkeypatch.setenv("SWIVAL_ENCRYPT_KEY", "aa" * 32)

        args = types.SimpleNamespace(
            provider="generic",
            model=None,
            api_key=None,
            base_url=None,
            max_turns=100,
            max_output_tokens=32768,
            max_context_tokens=None,
            temperature=None,
            top_p=None,
            seed=None,
            commands=None,
            files="some",
            yolo=False,
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_skills=False,
            sandbox="builtin",
            sandbox_session=None,
            sandbox_strict_read=False,
            no_sandbox_auto_session=False,
            no_read_guard=False,
            no_history=False,
            no_memory=False,
            memory_full=False,
            no_continue=False,
            quiet=False,
            proactive_summaries=False,
            extra_body=None,
            reasoning_effort=None,
            sanitize_thinking=False,
            cache=False,
            cache_dir=None,
            config_dir=None,
            encrypt_secrets=False,
            no_encrypt_secrets=False,
            encrypt_secrets_key=None,
            encrypt_secrets_tweak=None,
            add_dir=[],
            add_dir_ro=[],
            skills_dir=None,
        )

        kwargs = args_to_session_kwargs(args, "/tmp")

        assert kwargs["encrypt_secrets"] is True
        assert kwargs["encrypt_secrets_key"] == "aa" * 32

    def test_build_self_review_cmd_includes_encrypt_flag(self):
        """_build_self_review_cmd should include --encrypt-secrets when enabled."""
        from swival.agent import _build_self_review_cmd

        args = types.SimpleNamespace(
            files="some",
            yolo=False,
            provider="generic",
            model="test-model",
            base_url=None,
            skills_dir=None,
            max_context_tokens=None,
            max_output_tokens=32768,
            encrypt_secrets=True,
            commands=None,
        )

        cmd = _build_self_review_cmd(args)
        assert "--encrypt-secrets" in cmd
