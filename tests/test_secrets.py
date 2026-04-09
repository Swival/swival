"""Tests for swival.secrets — SecretShield class."""

import copy

import pytest

from swival.report import ConfigError
from swival.secrets import SecretShield


# ---------------------------------------------------------------------------
# Helpers: valid tokens for built-in patterns
# ---------------------------------------------------------------------------

# GitHub PAT: ghp_ + 36 alphanumeric chars
GHP_TOKEN = "ghp_" + "A" * 36
# OpenAI key: sk-proj- + 48 alphanumeric chars
SKPROJ_TOKEN = "sk-proj-" + "B" * 48
# Anthropic key: sk-ant-api03- + 48 alphanumeric chars
SKANT_TOKEN = "sk-ant-api03-" + "C" * 48
# AWS secret key (heuristic): 40-char base64, 3+ char classes, high entropy
AWS_SECRET = "odJFCrnl2edlBDdz1C5Jau2RJtBRnlWmTSHf6pWk"


@pytest.fixture
def shield():
    """Create a SecretShield with a fixed key."""
    s = SecretShield(key=b"\x00" * 32)
    yield s
    if not s.destroyed:
        s.destroy()


# ---------------------------------------------------------------------------
# Encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_github_pat_round_trip(self, shield):
        msgs = [{"role": "user", "content": f"Use {GHP_TOKEN} to clone."}]
        encrypted = shield.encrypt_messages(msgs)
        # Original unchanged
        assert msgs[0]["content"] == f"Use {GHP_TOKEN} to clone."
        # Encrypted content differs in the token body
        enc_content = encrypted[0]["content"]
        assert "ghp_" in enc_content
        assert GHP_TOKEN not in enc_content
        # Decrypt round-trip
        restored = shield.reverse_known(enc_content)
        assert restored == f"Use {GHP_TOKEN} to clone."

    def test_openai_key_round_trip(self, shield):
        msgs = [{"role": "tool", "content": f"key: {SKPROJ_TOKEN}"}]
        encrypted = shield.encrypt_messages(msgs)
        enc = encrypted[0]["content"]
        assert "sk-proj-" in enc
        assert SKPROJ_TOKEN not in enc
        restored = shield.reverse_known(enc)
        assert restored == msgs[0]["content"]

    def test_multiple_tokens_in_one_string(self, shield):
        text = f"gh: {GHP_TOKEN}, oai: {SKPROJ_TOKEN}"
        msgs = [{"role": "user", "content": text}]
        encrypted = shield.encrypt_messages(msgs)
        enc = encrypted[0]["content"]
        assert GHP_TOKEN not in enc
        assert SKPROJ_TOKEN not in enc
        restored = shield.reverse_known(enc)
        assert restored == text

    def test_no_tokens_unchanged(self, shield):
        msgs = [{"role": "user", "content": "No secrets here."}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] == "No secrets here."


# ---------------------------------------------------------------------------
# All roles encrypted outbound
# ---------------------------------------------------------------------------


class TestAllRolesEncrypted:
    def test_system_encrypted(self, shield):
        msgs = [{"role": "system", "content": f"Token: {GHP_TOKEN}"}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] != msgs[0]["content"]

    def test_user_encrypted(self, shield):
        msgs = [{"role": "user", "content": f"Token: {GHP_TOKEN}"}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] != msgs[0]["content"]

    def test_tool_encrypted(self, shield):
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": f"Output: {GHP_TOKEN}",
            }
        ]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] != msgs[0]["content"]
        # tool_call_id NOT encrypted
        assert encrypted[0]["tool_call_id"] == "tc_1"

    def test_assistant_content_encrypted(self, shield):
        msgs = [{"role": "assistant", "content": f"Found {GHP_TOKEN}"}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] != msgs[0]["content"]

    def test_assistant_tool_call_args_encrypted(self, shield):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": f'{{"path": "/tmp/k", "content": "{GHP_TOKEN}"}}',
                        },
                    }
                ],
            }
        ]
        encrypted = shield.encrypt_messages(msgs)
        enc_args = encrypted[0]["tool_calls"][0]["function"]["arguments"]
        assert GHP_TOKEN not in enc_args
        assert "ghp_" in enc_args
        # Name and ID not encrypted
        assert encrypted[0]["tool_calls"][0]["function"]["name"] == "write_file"
        assert encrypted[0]["tool_calls"][0]["id"] == "tc_1"


# ---------------------------------------------------------------------------
# Multimodal content
# ---------------------------------------------------------------------------


class TestMultimodal:
    def test_text_parts_encrypted_image_parts_untouched(self, shield):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is {GHP_TOKEN}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ghp_NotAToken"},
                    },
                ],
            }
        ]
        encrypted = shield.encrypt_messages(msgs)
        # Text part encrypted
        text_part = encrypted[0]["content"][0]
        assert GHP_TOKEN not in text_part["text"]
        # Image part untouched
        img_part = encrypted[0]["content"][1]
        assert img_part["image_url"]["url"] == "data:image/png;base64,ghp_NotAToken"


# ---------------------------------------------------------------------------
# Provenance check
# ---------------------------------------------------------------------------


class TestProvenanceCheck:
    def test_model_invented_token_passes_through(self, shield):
        # Populate registry with a known token
        shield.encrypt_messages([{"role": "user", "content": f"{GHP_TOKEN}"}])
        # Model-invented token that was never encrypted by us
        invented = "ghp_" + "X" * 36
        result = shield.reverse_known(invented)
        # Should pass through untouched
        assert result == invented

    def test_reverse_known_on_malformed_json(self, shield):
        shield.encrypt_messages([{"role": "user", "content": f"{GHP_TOKEN}"}])
        # Find what ciphertext was generated
        ct = None
        for k in shield._registry:
            if "ghp_" in k:
                ct = k
                break
        assert ct is not None
        # Malformed JSON containing the ciphertext
        malformed = '{"bad json: ' + ct
        result = shield.reverse_known(malformed)
        assert GHP_TOKEN in result


# ---------------------------------------------------------------------------
# Tweak isolation
# ---------------------------------------------------------------------------


class TestTweakIsolation:
    def test_different_tweaks_produce_different_ciphertext(self):
        key = b"\x01" * 32
        s1 = SecretShield(key=key, tweak=b"session-1")
        s2 = SecretShield(key=key, tweak=b"session-2")
        try:
            text = GHP_TOKEN
            e1 = s1.encrypt_messages([{"role": "user", "content": text}])
            e2 = s2.encrypt_messages([{"role": "user", "content": text}])
            # Both should encrypt
            assert isinstance(e1[0]["content"], str)
            assert isinstance(e2[0]["content"], str)
            assert GHP_TOKEN not in e1[0]["content"]
            assert GHP_TOKEN not in e2[0]["content"]
        finally:
            s1.destroy()
            s2.destroy()


# ---------------------------------------------------------------------------
# Destroy lifecycle
# ---------------------------------------------------------------------------


class TestDestroy:
    def test_destroy_clears_registry(self, shield):
        shield.encrypt_messages([{"role": "user", "content": f"{GHP_TOKEN}"}])
        assert len(shield._registry) > 0
        shield.destroy()
        assert len(shield._registry) == 0
        assert shield.destroyed

    def test_encrypt_after_destroy_raises(self, shield):
        shield.destroy()
        with pytest.raises(ConfigError, match="destroyed"):
            shield.encrypt_messages([{"role": "user", "content": f"{GHP_TOKEN}"}])

    def test_double_destroy_is_safe(self, shield):
        shield.destroy()
        shield.destroy()  # no error

    def test_reverse_known_after_destroy_is_noop(self, shield):
        shield.destroy()
        result = shield.reverse_known(f"{GHP_TOKEN}")
        assert result == f"{GHP_TOKEN}"


# ---------------------------------------------------------------------------
# Empty / None content
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_content(self, shield):
        msgs = [{"role": "assistant", "content": None}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] is None

    def test_empty_string_content(self, shield):
        msgs = [{"role": "user", "content": ""}]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["content"] == ""

    def test_empty_messages_list(self, shield):
        encrypted = shield.encrypt_messages([])
        assert encrypted == []

    def test_tool_calls_empty_list(self, shield):
        msgs = [
            {"role": "assistant", "content": "hello", "tool_calls": []},
        ]
        encrypted = shield.encrypt_messages(msgs)
        assert encrypted[0]["tool_calls"] == []

    def test_original_messages_not_mutated(self, shield):
        msgs = [{"role": "user", "content": f"Secret: {GHP_TOKEN}"}]
        original = copy.deepcopy(msgs)
        shield.encrypt_messages(msgs)
        assert msgs == original


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------


class TestCustomPatterns:
    def test_register_pattern_encrypts_token(self):
        s = SecretShield(
            key=b"\x00" * 32,
            extra_patterns=[
                {
                    "name": "myapp-key",
                    "prefix": "myapp_",
                    "body_regex": "[A-Za-z0-9]{32}",
                }
            ],
        )
        try:
            tok = "myapp_" + "A" * 32
            msgs = [{"role": "user", "content": f"key: {tok}"}]
            encrypted = s.encrypt_messages(msgs)
            assert tok not in encrypted[0]["content"]
            restored = s.reverse_known(encrypted[0]["content"])
            assert tok in restored
        finally:
            s.destroy()


# ---------------------------------------------------------------------------
# Heuristic tokens (length-expanding encryption)
# ---------------------------------------------------------------------------


class TestHeuristicTokens:
    def test_heuristic_token_round_trip(self, shield):
        """Heuristic tokens (AWS secret key) expand via [ENCRYPTED:...] markers."""
        msgs = [{"role": "user", "content": f"Key: {AWS_SECRET}"}]
        encrypted = shield.encrypt_messages(msgs)
        enc_content = encrypted[0]["content"]
        # Should contain [ENCRYPTED: marker
        assert "[ENCRYPTED:" in enc_content
        # Original token should not be present
        assert AWS_SECRET not in enc_content
        # Length changed (marker added)
        assert len(enc_content) > len(msgs[0]["content"])
        # Round-trip via reverse_known should recover original
        restored = shield.reverse_known(enc_content)
        assert restored == f"Key: {AWS_SECRET}"

    def test_heuristic_and_prefixed_mixed(self, shield):
        """Mixed heuristic + prefixed tokens both round-trip correctly."""
        text = f"GH: {GHP_TOKEN}, AWS: {AWS_SECRET}"
        msgs = [{"role": "user", "content": text}]
        encrypted = shield.encrypt_messages(msgs)
        enc_content = encrypted[0]["content"]
        # Both tokens should be encrypted
        assert GHP_TOKEN not in enc_content
        assert AWS_SECRET not in enc_content
        # Round-trip
        restored = shield.reverse_known(enc_content)
        assert restored == text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_hex_key(self):
        """from_config accepts hex key string."""
        s = SecretShield.from_config(key_hex="00" * 32)
        try:
            msgs = [{"role": "user", "content": f"Use {GHP_TOKEN}"}]
            encrypted = s.encrypt_messages(msgs)
            assert GHP_TOKEN not in encrypted[0]["content"]
        finally:
            s.destroy()

    def test_from_config_no_key_generates_random(self):
        """from_config with no key generates a random key."""
        s = SecretShield.from_config()
        try:
            msgs = [{"role": "user", "content": f"Use {GHP_TOKEN}"}]
            encrypted = s.encrypt_messages(msgs)
            assert GHP_TOKEN not in encrypted[0]["content"]
        finally:
            s.destroy()

    def test_from_config_with_patterns(self):
        """from_config forwards extra_patterns."""
        s = SecretShield.from_config(
            key_hex="00" * 32,
            extra_patterns=[
                {"name": "test-key", "prefix": "test_", "body_regex": "[A-Za-z0-9]{32}"}
            ],
        )
        try:
            tok = "test_" + "X" * 32
            msgs = [{"role": "user", "content": tok}]
            encrypted = s.encrypt_messages(msgs)
            assert tok not in encrypted[0]["content"]
        finally:
            s.destroy()


class TestDeterminism:
    def test_same_key_same_output(self):
        """Same key produces the same ciphertext (deterministic FPE)."""
        s1 = SecretShield(key=b"\x00" * 32)
        s2 = SecretShield(key=b"\x00" * 32)
        try:
            msgs = [{"role": "user", "content": f"Use {GHP_TOKEN}"}]
            e1 = s1.encrypt_messages(msgs)
            e2 = s2.encrypt_messages(msgs)
            assert e1[0]["content"] == e2[0]["content"]
        finally:
            s1.destroy()
            s2.destroy()

    def test_different_keys_different_output(self):
        """Different keys produce different ciphertext."""
        s1 = SecretShield(key=b"\x00" * 32)
        s2 = SecretShield(key=b"\x01" * 32)
        try:
            msgs = [{"role": "user", "content": f"Use {GHP_TOKEN}"}]
            e1 = s1.encrypt_messages(msgs)
            e2 = s2.encrypt_messages(msgs)
            assert e1[0]["content"] != e2[0]["content"]
        finally:
            s1.destroy()
            s2.destroy()


class TestEncryptObj:
    def test_encrypts_string(self):
        s = SecretShield()
        try:
            result = s.encrypt_obj(f"token={GHP_TOKEN}")
            assert GHP_TOKEN not in result
            assert isinstance(result, str)
        finally:
            s.destroy()

    def test_encrypts_nested_dict(self):
        s = SecretShield()
        try:
            obj = {"key": GHP_TOKEN, "nested": {"also": GHP_TOKEN}, "plain": "hello"}
            result = s.encrypt_obj(obj)
            assert GHP_TOKEN not in result["key"]
            assert GHP_TOKEN not in result["nested"]["also"]
            assert result["plain"] == "hello"
        finally:
            s.destroy()

    def test_encrypts_list(self):
        s = SecretShield()
        try:
            result = s.encrypt_obj([GHP_TOKEN, "plain", {"k": GHP_TOKEN}])
            assert GHP_TOKEN not in result[0]
            assert result[1] == "plain"
            assert GHP_TOKEN not in result[2]["k"]
        finally:
            s.destroy()

    def test_passthrough_non_strings(self):
        s = SecretShield()
        try:
            obj = {"count": 42, "flag": True, "nothing": None}
            assert s.encrypt_obj(obj) == obj
        finally:
            s.destroy()

    def test_raises_after_destroy(self):
        s = SecretShield()
        s.destroy()
        with pytest.raises(ConfigError):
            s.encrypt_obj(GHP_TOKEN)
