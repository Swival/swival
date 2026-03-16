"""Tests for vision/image support."""

import base64
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swival import fmt
from swival._msg import _msg_content
from swival.agent import (
    SYNTHETIC_USER_PREFIXES,
    _IMAGE_SYNTHETIC_PREFIX,
    _IMAGE_TOKEN_ESTIMATE,
    _is_vision_rejection,
    _model_supports_vision,
    _resolve_model_str,
    _strip_image_content,
    compact_messages,
    estimate_tokens,
    is_pinned,
)
from swival._msg import _has_image_content
from swival.report import AgentError
from swival.tools import (
    IMAGE_EXTENSIONS,
    IMAGE_MIME,
    MAX_IMAGE_BYTES,
    _view_image,
    dispatch,
)


@pytest.fixture(autouse=True)
def _init_fmt():
    fmt.init(color=False, no_color=False)


def _make_1x1_png():
    """Create a minimal valid 1x1 PNG image."""
    # Minimal PNG: signature + IHDR + IDAT + IEND
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(ctype, data):
        c = ctype + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture
def png_file(tmp_path):
    """Create a 1x1 PNG file in tmp_path."""
    p = tmp_path / "test.png"
    p.write_bytes(_make_1x1_png())
    return p


# ---------------------------------------------------------------------------
# _view_image tests
# ---------------------------------------------------------------------------


class TestViewImage:
    def test_basic_load(self, tmp_path, png_file):
        stash = []
        result = _view_image(str(png_file), str(tmp_path), image_stash=stash)
        assert "Image loaded" in result
        assert "test.png" in result
        assert len(stash) == 1
        assert stash[0]["data_url"].startswith("data:image/png;base64,")
        assert stash[0]["question"] == ""
        assert stash[0]["path"] == str(png_file)

    def test_with_question(self, tmp_path, png_file):
        stash = []
        result = _view_image(
            str(png_file),
            str(tmp_path),
            image_stash=stash,
            question="What is in this image?",
        )
        assert "Image loaded" in result
        assert stash[0]["question"] == "What is in this image?"

    def test_file_not_found(self, tmp_path):
        stash = []
        result = _view_image(
            str(tmp_path / "nope.png"), str(tmp_path), image_stash=stash
        )
        assert result.startswith("error:")
        assert "not found" in result
        assert len(stash) == 0

    def test_directory_rejected(self, tmp_path):
        stash = []
        result = _view_image(str(tmp_path), str(tmp_path), image_stash=stash)
        assert result.startswith("error:")
        assert "directory" in result

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "test.tiff"
        f.write_bytes(b"\x00\x00")
        stash = []
        result = _view_image(str(f), str(tmp_path), image_stash=stash)
        assert result.startswith("error:")
        assert "unsupported" in result.lower()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")
        stash = []
        result = _view_image(str(f), str(tmp_path), image_stash=stash)
        assert result.startswith("error:")
        assert "empty" in result

    def test_file_too_large(self, tmp_path):
        f = tmp_path / "big.png"
        # Write just enough to trigger the size check (use a sparse approach)
        f.write_bytes(b"\x89PNG" + b"\x00" * (MAX_IMAGE_BYTES + 1))
        stash = []
        result = _view_image(str(f), str(tmp_path), image_stash=stash)
        assert result.startswith("error:")
        assert "too large" in result

    def test_base64_correctness(self, tmp_path, png_file):
        stash = []
        _view_image(str(png_file), str(tmp_path), image_stash=stash)
        data_url = stash[0]["data_url"]
        assert data_url.startswith("data:image/png;base64,")
        b64_part = data_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == png_file.read_bytes()

    def test_supported_extensions(self, tmp_path):
        """All IMAGE_EXTENSIONS should be accepted (when file exists)."""
        for ext in IMAGE_EXTENSIONS:
            f = tmp_path / f"test{ext}"
            f.write_bytes(_make_1x1_png())  # content doesn't matter for ext check
            stash = []
            result = _view_image(str(f), str(tmp_path), image_stash=stash)
            assert not result.startswith("error:"), f"Failed for {ext}: {result}"

    def test_path_outside_base_dir(self, tmp_path):
        """Paths outside base_dir should be rejected."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_1x1_png())
            outside = f.name
        try:
            stash = []
            result = _view_image(outside, str(tmp_path), image_stash=stash)
            assert result.startswith("error:")
        finally:
            Path(outside).unlink()


# ---------------------------------------------------------------------------
# dispatch routing tests
# ---------------------------------------------------------------------------


class TestDispatchViewImage:
    def test_dispatch_routes_to_view_image(self, tmp_path, png_file):
        stash = []
        result = dispatch(
            "view_image",
            {"image_path": str(png_file)},
            str(tmp_path),
            image_stash=stash,
        )
        assert "Image loaded" in result
        assert len(stash) == 1

    def test_dispatch_no_stash(self, tmp_path, png_file):
        result = dispatch(
            "view_image",
            {"image_path": str(png_file)},
            str(tmp_path),
        )
        assert result.startswith("error:")
        assert "not available" in result


# ---------------------------------------------------------------------------
# _msg_content with list content
# ---------------------------------------------------------------------------


class TestMsgContentList:
    def test_string_content(self):
        assert _msg_content({"role": "user", "content": "hello"}) == "hello"

    def test_list_content_text_only(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
            ],
        }
        assert _msg_content(msg) == "What is this?"

    def test_list_content_with_image(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "[image] Describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
        assert _msg_content(msg) == "[image] Describe"

    def test_empty_list(self):
        msg = {"role": "user", "content": []}
        assert _msg_content(msg) == ""

    def test_none_content(self):
        msg = {"role": "user", "content": None}
        assert _msg_content(msg) == ""


# ---------------------------------------------------------------------------
# Image injection tests
# ---------------------------------------------------------------------------


class TestImageInjection:
    def test_synthetic_prefix_in_prefixes(self):
        assert "[image]" in SYNTHETIC_USER_PREFIXES

    def test_image_synthetic_prefix_value(self):
        assert _IMAGE_SYNTHETIC_PREFIX == "[image]"


# ---------------------------------------------------------------------------
# is_pinned with image turns
# ---------------------------------------------------------------------------


class TestIsPinnedImage:
    def test_normal_user_turn_pinned(self):
        turn = [{"role": "user", "content": "Hello"}]
        assert is_pinned(turn) is True

    def test_image_user_turn_not_pinned(self):
        turn = [{"role": "user", "content": "[image] Describe this"}]
        assert is_pinned(turn) is False

    def test_image_multimodal_not_pinned(self):
        turn = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[image] Describe"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        assert is_pinned(turn) is False

    def test_assistant_turn_not_pinned(self):
        turn = [{"role": "assistant", "content": "Sure"}]
        assert is_pinned(turn) is False


# ---------------------------------------------------------------------------
# Compaction with image messages
# ---------------------------------------------------------------------------


class TestOverflowAfterImageInjection:
    """Verify that context overflow right after image injection produces
    an explanatory fallback, not a silent strip."""

    def _make_messages_with_image(self):
        return [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Describe this image"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "view_image",
                            "arguments": '{"image_path": "test.png"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": "Image loaded: test.png (1 KB). The image has been attached.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[image] Describe and analyze the attached image(s).",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 5000},
                    },
                ],
            },
        ]

    def test_vision_pending_replaces_image_before_compaction(self):
        """Simulate the overflow path: when _vision_pending is True,
        the image message should be replaced with an explanatory fallback
        BEFORE compaction strips it silently."""
        messages = self._make_messages_with_image()

        # This is the logic from run_agent_loop's compaction path:
        # when _vision_pending is True, scan backward and replace.
        _vision_pending = True
        if _vision_pending:
            for i in range(len(messages) - 1, -1, -1):
                if (
                    isinstance(messages[i], dict)
                    and isinstance(messages[i].get("content"), list)
                    and any(
                        p.get("type") == "image_url"
                        for p in messages[i]["content"]
                        if isinstance(p, dict)
                    )
                ):
                    messages[i] = {
                        "role": "user",
                        "content": (
                            _IMAGE_SYNTHETIC_PREFIX
                            + " The image was dropped during context compaction "
                            "and could not be analyzed. Inform the user that the "
                            "image could not be processed due to context limits."
                        ),
                    }
                    break
            _vision_pending = False

        # Now compact — should not have any list content left
        result = compact_messages(messages)
        for msg in result:
            content = msg.get("content", "")
            assert not isinstance(content, list), "List content survived"

        # Find the replacement message
        fallback = [
            m
            for m in result
            if isinstance(m.get("content"), str)
            and "dropped during context compaction" in m["content"]
        ]
        assert len(fallback) == 1
        assert fallback[0]["content"].startswith(_IMAGE_SYNTHETIC_PREFIX)

    def test_without_vision_pending_strips_silently(self):
        """When _vision_pending is False (image was already processed),
        compaction strips the base64 data as a plain cleanup."""
        messages = self._make_messages_with_image()
        result = compact_messages(messages)
        # Should strip image data but not add the explanatory fallback
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str) and "[image]" in content:
                assert "image data removed during compaction" in content
                assert "dropped during context compaction" not in content


class TestCompactionImageStripping:
    def test_strip_image_content(self):
        messages = [
            {"role": "user", "content": "describe"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[image] What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {"role": "assistant", "content": "It's a cat."},
        ]
        _strip_image_content(messages)
        # Image message should be flattened to text
        assert isinstance(messages[1]["content"], str)
        assert "[image] What is this?" in messages[1]["content"]
        assert "image data removed" in messages[1]["content"]
        # Non-image messages unchanged
        assert messages[0]["content"] == "describe"
        assert messages[2]["content"] == "It's a cat."

    def test_compact_messages_strips_images(self):
        """compact_messages should strip base64 from image messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[image] test"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 1000},
                    },
                ],
            },
            {"role": "assistant", "content": "Result"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You're welcome"},
        ]
        result = compact_messages(messages)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Should not happen — all lists should be flattened
                assert False, "List content survived compaction"
            assert "base64," not in str(content) or "A" * 100 not in str(content)


# ---------------------------------------------------------------------------
# Token estimation with images
# ---------------------------------------------------------------------------


class TestEstimateTokensImage:
    def test_text_only(self):
        messages = [{"role": "user", "content": "hello world"}]
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_image_content_adds_estimate(self):
        text_messages = [{"role": "user", "content": "hello"}]
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
        text_tokens = estimate_tokens(text_messages)
        image_tokens = estimate_tokens(image_messages)
        # Image version should add _IMAGE_TOKEN_ESTIMATE
        assert (
            image_tokens >= text_tokens + _IMAGE_TOKEN_ESTIMATE - 10
        )  # allow small variance

    def test_multiple_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,a"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,b"},
                    },
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        assert tokens >= 2 * _IMAGE_TOKEN_ESTIMATE


# ---------------------------------------------------------------------------
# Cache skip for vision
# ---------------------------------------------------------------------------


class TestHasImageContent:
    def test_no_images(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _has_image_content(messages) is False

    def test_with_images(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        assert _has_image_content(messages) is True

    def test_list_without_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "just text"},
                ],
            },
        ]
        assert _has_image_content(messages) is False


# ---------------------------------------------------------------------------
# _model_supports_vision
# ---------------------------------------------------------------------------


class TestModelSupportsVision:
    def test_returns_true(self):
        with patch("litellm.supports_vision", return_value=True):
            assert _model_supports_vision("gpt-4o") is True

    def test_returns_false(self):
        with patch("litellm.supports_vision", return_value=False):
            assert _model_supports_vision("gpt-3.5-turbo") is False

    def test_returns_none_on_exception(self):
        with patch("litellm.supports_vision", side_effect=Exception("unknown model")):
            assert _model_supports_vision("custom/local-model") is None


# ---------------------------------------------------------------------------
# _is_vision_rejection
# ---------------------------------------------------------------------------


class TestIsVisionRejection:
    def test_matches_image_input(self):
        assert _is_vision_rejection(
            AgentError("This model does not support image input")
        )

    def test_matches_image_content(self):
        assert _is_vision_rejection(AgentError("invalid image content in request"))

    def test_matches_vision_keyword(self):
        assert _is_vision_rejection(AgentError("vision is not supported"))

    def test_matches_multimodal(self):
        assert _is_vision_rejection(AgentError("multimodal content not allowed"))

    def test_matches_image_url(self):
        assert _is_vision_rejection(AgentError("image_url is not supported"))

    def test_no_match_does_not_support_generic(self):
        """'does not support' alone (without image/vision keywords) should NOT match."""
        assert not _is_vision_rejection(AgentError("model does not support seed"))

    def test_no_match_invalid_content_generic(self):
        """'invalid content' alone should NOT match."""
        assert not _is_vision_rejection(AgentError("invalid content length"))

    def test_no_match_auth(self):
        assert not _is_vision_rejection(AgentError("authentication failed"))

    def test_no_match_rate_limit(self):
        assert not _is_vision_rejection(AgentError("rate limit exceeded"))

    def test_no_match_timeout(self):
        assert not _is_vision_rejection(AgentError("connection timeout"))

    def test_no_match_generic(self):
        assert not _is_vision_rejection(AgentError("server error 500"))


# ---------------------------------------------------------------------------
# _resolve_model_str
# ---------------------------------------------------------------------------


class TestResolveModelStr:
    def test_lmstudio(self):
        assert _resolve_model_str("lmstudio", "qwen2-vl") == "openai/qwen2-vl"

    def test_huggingface(self):
        assert (
            _resolve_model_str("huggingface", "huggingface/my-model")
            == "huggingface/my-model"
        )

    def test_huggingface_no_prefix(self):
        assert _resolve_model_str("huggingface", "my-model") == "huggingface/my-model"

    def test_openrouter(self):
        assert (
            _resolve_model_str("openrouter", "meta-llama/llama-3")
            == "openrouter/meta-llama/llama-3"
        )

    def test_openrouter_double_prefix(self):
        assert (
            _resolve_model_str("openrouter", "openrouter/openrouter/free")
            == "openrouter/openrouter/free"
        )

    def test_generic(self):
        assert _resolve_model_str("generic", "my-model") == "openai/my-model"

    def test_chatgpt(self):
        assert _resolve_model_str("chatgpt", "gpt-4o") == "chatgpt/gpt-4o"

    def test_chatgpt_double_prefix(self):
        assert (
            _resolve_model_str("chatgpt", "chatgpt/chatgpt/gpt-4o") == "chatgpt/gpt-4o"
        )

    def test_unknown_provider(self):
        assert _resolve_model_str("other", "model-x") == "model-x"


# ---------------------------------------------------------------------------
# read_file hint for images
# ---------------------------------------------------------------------------


class TestReadFileImageHint:
    def test_read_file_suggests_view_image(self, tmp_path, png_file):
        from swival.tools import _read_file

        result = _read_file(str(png_file), str(tmp_path))
        assert result.startswith("error:")
        assert "view_image" in result

    def test_read_file_binary_no_image_ext(self, tmp_path):
        from swival.tools import _read_file

        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = _read_file(str(f), str(tmp_path))
        assert result.startswith("error:")
        assert "binary file" in result
        assert "view_image" not in result


# ---------------------------------------------------------------------------
# continue_here synthetic detection
# ---------------------------------------------------------------------------


class TestContinueHereSyntheticDetection:
    def test_image_prefix_is_synthetic(self):
        from swival.continue_here import _is_synthetic

        assert _is_synthetic("[image] Describe and analyze the attached image(s).")

    def test_image_prefix_with_question(self):
        from swival.continue_here import _is_synthetic

        assert _is_synthetic("[image] What is in this photo?")
