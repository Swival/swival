"""Tests for the /model REPL command."""

import types
from unittest.mock import patch

import pytest

from swival import model_catalog as mc
from swival import model_prefs as mp
from swival.agent import _repl_model, execute_input
from swival.input_dispatch import InputContext, parse_input_line
from swival.model_catalog import Catalog, ModelEntry
from swival.report import ConfigError
from swival.thinking import ThinkingState
from swival.todo import TodoState

BASELINE = {
    "provider": "huggingface",
    "model": "org/old-model",
    "api_key": "hf-key",
    "base_url": None,
    "max_context_tokens": None,
}

CATALOG = Catalog(
    entries=[
        ModelEntry(
            id="zai-org/GLM-5.2",
            context_length=1_048_576,
            supports_tools=True,
            price_in=0.9,
            price_out=3.0,
        ),
        ModelEntry(id="Qwen/Qwen3.6-35B", context_length=262_144, supports_tools=True),
        ModelEntry(id="acme/no-tools", supports_tools=False),
    ],
    source="test catalog",
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def _resolve_return(model="zai-org/GLM-5.2", context=128000):
    return (
        model,
        None,
        "hf-key",
        context,
        {"provider": "huggingface", "api_key": "hf-key"},
    )


def _make_repl_kwargs():
    return {
        "model_id": "org/old-model",
        "api_base": None,
        "context_length": 8192,
        "llm_kwargs": {
            "provider": "huggingface",
            "api_key": "hf-key",
            "reasoning_effort": "high",
            "prompt_cache": False,
        },
        "max_output_tokens": 4096,
        "temperature": 0.7,
        "top_p": None,
        "seed": None,
    }


def _call_model(
    cmd_arg,
    *,
    repl_kwargs=None,
    last_model=None,
    catalog=CATALOG,
    resolve_return=None,
    resolve_side_effect=None,
    subagent_manager=None,
    baseline=None,
    monkeypatch=None,
):
    if repl_kwargs is None:
        repl_kwargs = _make_repl_kwargs()

    def fake_list_models(provider, base_url=None, api_key=None, **_):
        if catalog is None:
            raise mc.CatalogUnavailable("listing off", hint="pass an id")
        return catalog

    monkeypatch.setattr(mc, "list_models", fake_list_models)

    patch_kwargs = {}
    if resolve_side_effect is not None:
        patch_kwargs["side_effect"] = resolve_side_effect
    else:
        patch_kwargs["return_value"] = resolve_return or _resolve_return()

    with patch("swival.agent.resolve_provider", **patch_kwargs) as mock_rp:
        new_last, msg, is_error = _repl_model(
            cmd_arg,
            profiles={},
            current_profile=None,
            startup_profile=None,
            raw_baseline=dict(baseline or BASELINE),
            pre_profile_baseline={},
            repl_kwargs=repl_kwargs,
            subagent_manager=subagent_manager,
            last_model=last_model,
            interactive=False,
            verbose=False,
        )
    return new_last, msg, is_error, repl_kwargs, mock_rp


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------


class TestModelSwitch:
    def test_exact_id_switch_updates_kwargs(self, monkeypatch):
        new_last, msg, err, kw, mock_rp = _call_model(
            "zai-org/GLM-5.2", monkeypatch=monkeypatch
        )
        assert not err
        assert new_last == ("huggingface", "org/old-model")  # revert slot
        assert kw["model_id"] == "zai-org/GLM-5.2"
        assert kw["context_length"] == 128000
        call_kw = mock_rp.call_args.kwargs
        assert call_kw["provider"] == "huggingface"
        assert call_kw["model"] == "zai-org/GLM-5.2"
        assert "model: zai-org/GLM-5.2" in msg
        assert "128,000 tokens" in msg
        assert "$0.9 in / $3 out" in msg

    def test_exact_match_is_case_insensitive(self, monkeypatch):
        _, _, err, kw, _ = _call_model("zai-org/glm-5.2", monkeypatch=monkeypatch)
        assert not err
        assert kw["model_id"] == "zai-org/GLM-5.2"

    def test_fuzzy_single_match(self, monkeypatch):
        _, _, err, kw, mock_rp = _call_model("qwen", monkeypatch=monkeypatch)
        assert not err
        assert mock_rp.call_args.kwargs["model"] == "Qwen/Qwen3.6-35B"

    def test_ambiguous_match_errors_when_not_interactive(self, monkeypatch):
        catalog = Catalog(
            entries=[ModelEntry(id="acme/model-a"), ModelEntry(id="acme/model-b")],
            source="t",
        )
        new_last, msg, err, kw, _ = _call_model(
            "model", catalog=catalog, monkeypatch=monkeypatch
        )
        assert err
        assert "matches several models" in msg
        assert "acme/model-a" in msg and "acme/model-b" in msg
        assert kw["model_id"] == "org/old-model"

    def test_no_match_lists_available(self, monkeypatch):
        new_last, msg, err, kw, _ = _call_model("mistral", monkeypatch=monkeypatch)
        assert err
        assert "was not found" in msg
        assert "zai-org/GLM-5.2" in msg
        assert kw["model_id"] == "org/old-model"

    def test_catalog_unavailable_trusts_the_id(self, monkeypatch):
        _, msg, err, kw, mock_rp = _call_model(
            "org/whatever", catalog=None, monkeypatch=monkeypatch
        )
        assert not err
        assert mock_rp.call_args.kwargs["model"] == "org/whatever"

    def test_same_model_is_a_noop(self, monkeypatch):
        kw = _make_repl_kwargs()
        kw["model_id"] = "zai-org/GLM-5.2"
        new_last, msg, err, kw, _ = _call_model(
            "zai-org/GLM-5.2",
            repl_kwargs=kw,
            last_model=("huggingface", "keep-me"),
            monkeypatch=monkeypatch,
        )
        assert not err
        assert "already using" in msg
        assert new_last == ("huggingface", "keep-me")

    def test_switch_failure_is_transactional(self, monkeypatch):
        kw = _make_repl_kwargs()
        original = dict(kw)
        new_last, msg, err, kw, _ = _call_model(
            "zai-org/GLM-5.2",
            repl_kwargs=kw,
            last_model=("huggingface", "prior"),
            resolve_side_effect=ConfigError("bad key"),
            monkeypatch=monkeypatch,
        )
        assert err
        assert "model switch failed" in msg
        assert "Still using org/old-model" in msg
        assert kw == original
        assert new_last == ("huggingface", "prior")

    def test_context_falls_back_to_entry_metadata(self, monkeypatch):
        _, msg, err, kw, _ = _call_model(
            "zai-org/GLM-5.2",
            resolve_return=("zai-org/GLM-5.2", None, "hf-key", None, {}),
            monkeypatch=monkeypatch,
        )
        assert kw["context_length"] == 1_048_576

    def test_session_llm_kwargs_survive(self, monkeypatch):
        _, _, _, kw, _ = _call_model("zai-org/GLM-5.2", monkeypatch=monkeypatch)
        assert kw["llm_kwargs"]["reasoning_effort"] == "high"
        assert kw["llm_kwargs"]["prompt_cache"] is False
        assert kw["llm_kwargs"]["provider"] == "huggingface"

    def test_sampling_knobs_untouched(self, monkeypatch):
        _, _, _, kw, _ = _call_model("zai-org/GLM-5.2", monkeypatch=monkeypatch)
        assert kw["temperature"] == 0.7
        assert kw["max_output_tokens"] == 4096

    def test_no_tools_model_warns_but_switches(self, monkeypatch):
        _, msg, err, kw, _ = _call_model(
            "acme/no-tools",
            resolve_return=("acme/no-tools", None, "hf-key", 4096, {}),
            monkeypatch=monkeypatch,
        )
        assert not err
        assert kw["model_id"] == "acme/no-tools"
        assert "no tool-calling support" in msg

    def test_recents_recorded(self, monkeypatch):
        _call_model("zai-org/GLM-5.2", monkeypatch=monkeypatch)
        assert mp.load_prefs().recents_for("huggingface") == ["zai-org/GLM-5.2"]

    def test_subagent_template_updated(self, monkeypatch):
        mgr = types.SimpleNamespace(_template={"model_id": "old"})
        _call_model("zai-org/GLM-5.2", subagent_manager=mgr, monkeypatch=monkeypatch)
        assert mgr._template["model_id"] == "zai-org/GLM-5.2"
        assert mgr._template["context_length"] == 128000


# ---------------------------------------------------------------------------
# Revert (/model -)
# ---------------------------------------------------------------------------


class TestModelRevert:
    def test_revert_without_history_errors(self, monkeypatch):
        new_last, msg, err, _, _ = _call_model("-", monkeypatch=monkeypatch)
        assert err
        assert "no previous model" in msg

    def test_revert_switches_back(self, monkeypatch):
        new_last, msg, err, kw, mock_rp = _call_model(
            "-",
            last_model=("huggingface", "org/previous"),
            resolve_return=("org/previous", None, "hf-key", 32000, {}),
            monkeypatch=monkeypatch,
        )
        assert not err
        assert mock_rp.call_args.kwargs["model"] == "org/previous"
        # The revert slot now holds the model we just left, so - toggles.
        assert new_last == ("huggingface", "org/old-model")

    def test_revert_ignores_other_providers_slot(self, monkeypatch):
        # A slot recorded under a different provider must not resolve; the
        # slot is self-validating, so no /profile-side reset is needed.
        new_last, msg, err, _, mock_rp = _call_model(
            "-",
            last_model=("lmstudio", "local-model"),
            monkeypatch=monkeypatch,
        )
        assert err
        assert "no previous model" in msg
        assert new_last == ("lmstudio", "local-model")  # slot kept, not wiped
        mock_rp.assert_not_called()


# ---------------------------------------------------------------------------
# Favorites (--fav)
# ---------------------------------------------------------------------------


class TestModelFavorites:
    def test_fav_toggles_current_model(self, monkeypatch):
        _, msg, err, _, _ = _call_model("--fav", monkeypatch=monkeypatch)
        assert not err
        assert "org/old-model added to huggingface favorites" in msg
        assert mp.load_prefs().is_favorite("huggingface", "org/old-model")

        _, msg, err, _, _ = _call_model("--fav", monkeypatch=monkeypatch)
        assert "removed from" in msg
        assert not mp.load_prefs().is_favorite("huggingface", "org/old-model")

    def test_fav_with_explicit_id(self, monkeypatch):
        _, msg, err, _, _ = _call_model("--fav org/other", monkeypatch=monkeypatch)
        assert not err
        assert mp.load_prefs().is_favorite("huggingface", "org/other")

    def test_unknown_option(self, monkeypatch):
        _, msg, err, _, _ = _call_model("--bogus", monkeypatch=monkeypatch)
        assert err
        assert "unknown option" in msg

    def test_favorite_outranks_catalog_in_fuzzy_match(self, monkeypatch):
        mp.toggle_favorite("huggingface", "acme/no-tools")
        # "o" alone matches several catalog ids; the favorite must win only
        # for queries where it scores at least as well, so use a query that
        # matches both a favorite and a non-favorite by substring.
        catalog = Catalog(
            entries=[ModelEntry(id="acme/shared-name"), ModelEntry(id="acme/no-tools")],
            source="t",
        )
        _, _, err, _, mock_rp = _call_model(
            "acme/no", catalog=catalog, monkeypatch=monkeypatch
        )
        assert not err
        assert mock_rp.call_args.kwargs["model"] == "acme/no-tools"


# ---------------------------------------------------------------------------
# Listing (non-interactive, no argument)
# ---------------------------------------------------------------------------


class TestModelListing:
    def test_listing_shows_catalog(self, monkeypatch):
        new_last, msg, err, kw, _ = _call_model("", monkeypatch=monkeypatch)
        assert not err
        assert "test catalog" in msg
        assert "zai-org/GLM-5.2" in msg
        assert "switch with /model <id>" in msg
        assert kw["model_id"] == "org/old-model"

    def test_listing_marks_current_and_favorites(self, monkeypatch):
        mp.toggle_favorite("huggingface", "Qwen/Qwen3.6-35B")
        kw = _make_repl_kwargs()
        kw["model_id"] = "zai-org/GLM-5.2"
        _, msg, _, _, _ = _call_model("", repl_kwargs=kw, monkeypatch=monkeypatch)
        assert "★ Qwen/Qwen3.6-35B" in msg
        assert "(current)" in msg
        # Favorite is pinned to the top of the listing.
        lines = msg.splitlines()
        assert "Qwen" in lines[1]

    def test_listing_unavailable(self, monkeypatch):
        _, msg, err, _, _ = _call_model("", catalog=None, monkeypatch=monkeypatch)
        assert err
        assert "cannot list models" in msg
        assert "pass an id" in msg


# ---------------------------------------------------------------------------
# HuggingFace long-tail status check
# ---------------------------------------------------------------------------


class TestHFStatusCheck:
    def test_hub_model_with_live_provider_switches(self, monkeypatch):
        hub_entry = ModelEntry(
            id="org/long-tail", context_length=65536, supports_tools=True
        )
        monkeypatch.setattr(
            mc, "hf_model_status", lambda *a, **k: (True, hub_entry, [])
        )
        _, msg, err, kw, mock_rp = _call_model(
            "org/long-tail",
            resolve_return=("org/long-tail", None, "hf-key", None, {}),
            monkeypatch=monkeypatch,
        )
        assert not err
        assert mock_rp.call_args.kwargs["model"] == "org/long-tail"
        assert kw["context_length"] == 65536

    def test_hub_model_without_live_provider_errors_nicely(self, monkeypatch):
        monkeypatch.setattr(
            mc,
            "hf_model_status",
            lambda *a, **k: (True, None, ["fireworks-ai (staging)"]),
        )
        _, msg, err, kw, _ = _call_model("org/long-tail", monkeypatch=monkeypatch)
        assert err
        assert "cannot use org/long-tail" in msg
        assert "fireworks-ai (staging)" in msg
        assert kw["model_id"] == "org/old-model"

    def test_hub_model_missing_falls_back_to_not_found(self, monkeypatch):
        monkeypatch.setattr(mc, "hf_model_status", lambda *a, **k: (False, None, []))
        _, msg, err, _, _ = _call_model("org/long-tail", monkeypatch=monkeypatch)
        assert err
        assert "was not found" in msg


# ---------------------------------------------------------------------------
# execute_input integration
# ---------------------------------------------------------------------------


def _make_ctx():
    return InputContext(
        messages=[],
        tools=[],
        base_dir="/tmp",
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=True,
        continue_here=False,
        verbose=False,
        interactive=False,
        raw_llm_baseline=dict(BASELINE),
        loop_kwargs=_make_repl_kwargs(),
    )


class TestModelDispatch:
    def test_dispatch_switch_updates_last_model(self, monkeypatch):
        ctx = _make_ctx()
        monkeypatch.setattr(mc, "list_models", lambda *a, **k: CATALOG)
        with patch("swival.agent.resolve_provider", return_value=_resolve_return()):
            result = execute_input(parse_input_line("/model zai-org/GLM-5.2"), ctx)
        assert result.kind == "state_change"
        assert not result.is_error
        assert ctx.loop_kwargs["model_id"] == "zai-org/GLM-5.2"
        assert ctx.last_model == ("huggingface", "org/old-model")

    def test_revert_survives_profile_round_trip(self, monkeypatch):
        # The pair-based slot self-validates on provider, so a /profile
        # excursion to another provider does not need to clear it: /model -
        # simply refuses while the provider differs and works again after
        # switching back.
        ctx = _make_ctx()
        ctx.last_model = ("lmstudio", "local-model")
        monkeypatch.setattr(
            mc,
            "list_models",
            lambda *a, **k: (_ for _ in ()).throw(mc.CatalogUnavailable("off")),
        )
        with patch("swival.agent.resolve_provider", return_value=_resolve_return()):
            result = execute_input(parse_input_line("/model -"), ctx)
        assert result.is_error
        assert "no previous model" in result.text
        assert ctx.last_model == ("lmstudio", "local-model")

    def test_help_mentions_model(self):
        from swival.agent import _repl_help

        text = _repl_help()
        assert "/model" in text
        assert "--fav" in text
