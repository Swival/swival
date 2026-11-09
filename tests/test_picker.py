"""Tests for swival/picker.py: pure logic headless, app logic via pipe input."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from swival import model_catalog as mc
from swival import picker
from swival.model_catalog import Catalog, ModelEntry
from swival.picker import (
    PickerState,
    format_entry,
    match_score,
    pick_model,
    rank_entries,
)


def E(model_id, **kw):
    return ModelEntry(id=model_id, **kw)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


# ---------------------------------------------------------------- pure logic


def test_match_score_ordering():
    e = E("zai-org/GLM-5.3")
    assert match_score("", e) == 0
    assert match_score("zai-org/glm-5.3", e) == 0  # exact, case-insensitive
    assert match_score("zai", e) == 1  # prefix
    assert match_score("glm", e) == 2  # substring
    assert match_score("zgl", e) == 3  # subsequence
    assert match_score("mistral", e) is None


def test_match_score_uses_display_name():
    e = E("acme/xy-1", display_name="Big Friendly Model")
    assert match_score("friendly", e) == 2


def test_rank_entries_priority_groups():
    entries = [E("plain-1"), E("recent-b"), E("fav"), E("cur"), E("recent-a")]
    ranked = rank_entries(
        entries,
        favorites={"fav"},
        recents=["recent-a", "recent-b"],
        current="cur",
    )
    assert [e.id for e in ranked] == ["fav", "cur", "recent-a", "recent-b", "plain-1"]


def test_rank_entries_filters_and_keeps_catalog_order():
    entries = [E("b-match"), E("nope"), E("a-match")]
    ranked = rank_entries(entries, query="match")
    assert [e.id for e in ranked] == ["b-match", "a-match"]


def test_state_tools_filter_hides_only_explicit_false():
    entries = [
        E("yes", supports_tools=True),
        E("no", supports_tools=False),
        E("unknown"),
    ]
    state = PickerState(entries=entries, tools_only=True)
    assert [e.id for e in state.visible()] == ["yes", "unknown"]
    assert state.hidden_by_tools() == 1
    state.toggle_tools_only()
    assert len(state.visible()) == 3
    assert state.hidden_by_tools() == 0


def test_state_rows_has_manual_tail_and_move_clamps():
    state = PickerState(entries=[E("a"), E("b")])
    rows = state.rows()
    assert rows[-1] is None  # manual row
    state.move(99)
    assert state.on_manual_row()
    state.move(-99)
    assert state.cursor == 0
    assert state.selected().id == "a"


def test_state_set_query_resets_cursor():
    state = PickerState(entries=[E("a"), E("ab")])
    state.move(1)
    state.set_query("a")
    assert state.cursor == 0


def test_state_matching_is_memoized_and_invalidated():
    state = PickerState(entries=[E("alpha"), E("bravo")])
    first = state.matching()
    assert state.matching() is first  # cached between reads

    state.set_query("bra")
    assert [e.id for e in state.matching()] == ["bravo"]

    state.set_query("")
    state.set_favorite("bravo", True)
    assert [e.id for e in state.matching()] == ["bravo", "alpha"]  # re-ranked

    state.set_entries([E("charlie")])
    assert [e.id for e in state.matching()] == ["charlie"]


def test_format_entry_columns():
    line = format_entry(
        E(
            "org/model",
            context_length=1_048_576,
            supports_tools=True,
            price_in=0.9,
            price_out=3.0,
            loaded=True,
            tags=("free",),
        ),
        is_favorite=True,
        is_current=True,
        id_width=30,
    )
    assert line.startswith("★ ")
    assert "org/model (current)" in line
    assert "1M" in line
    assert "$0.9/$3" in line
    assert "tools" in line
    assert "loaded" in line and "free" in line

    bare = format_entry(E("just-a-model"))
    assert bare.strip().startswith("just-a-model")
    assert "$" not in bare


# ------------------------------------------------------------- app behavior


def _run(entries, keys, *, state_kw=None, **kw):
    state = PickerState(entries=entries, **(state_kw or {}))
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        result = pick_model(state, title="test", input=pipe, output=DummyOutput(), **kw)
    return result, state


def test_app_enter_selects_first_row():
    res, _ = _run([E("first"), E("second")], "\r")
    assert res.kind == "selected"
    assert res.entry.id == "first"


def test_app_typing_filters_then_selects():
    res, state = _run([E("alpha"), E("bravo")], "bra\r")
    assert res.kind == "selected"
    assert res.entry.id == "bravo"
    assert state.query == "bra"


def test_app_arrow_moves_selection():
    res, _ = _run([E("one"), E("two")], "\x1b[B\r")
    assert res.entry.id == "two"


def test_app_star_toggles_favorite_without_typing():
    toggled = []

    def on_toggle(model_id):
        toggled.append(model_id)
        return True

    res, state = _run([E("m1"), E("m2")], "*\r", on_toggle_favorite=on_toggle)
    assert toggled == ["m1"]
    assert res.kind == "selected"
    assert res.entry.id == "m1"
    assert state.query == ""  # '*' did not land in the filter
    assert "m1" in state.favorites


def test_app_ctrl_t_toggles_tools_filter():
    entries = [E("no-tools", supports_tools=False), E("tooled", supports_tools=True)]
    res, state = _run(entries, "\x14\r")
    assert state.tools_only is True
    assert res.entry.id == "tooled"


def test_app_ctrl_c_cancels():
    res, _ = _run([E("m")], "\x03")
    assert res.kind == "cancel"


def test_app_manual_row_via_cursor():
    res, _ = _run([E("m")], "\x1b[B\r")
    assert res.kind == "manual"


def test_app_no_matches_enter_means_manual_with_query():
    res, state = _run([E("alpha")], "zzz\r")
    assert res.kind == "manual"
    assert state.query == "zzz"


def test_app_ctrl_r_requests_refresh():
    res, _ = _run([E("m")], "\x12")
    assert res.kind == "refresh"


def test_app_ctrl_s_requests_search_only_when_allowed():
    res, _ = _run([E("m")], "\x13", allow_search=True)
    assert res.kind == "search"
    # Without allow_search, ^s is unbound and ignored.
    res, _ = _run([E("m")], "\x13\r", allow_search=False)
    assert res.kind == "selected"


# ---------------------------------------------------------- basic fallback


def _run_basic(entries, responses, **state_kw):
    from swival.picker import _pick_basic

    remaining = list(responses)

    class FakeSession:
        def __init__(self, *a, **k):
            pass

        def prompt(self, *a, **k):
            return remaining.pop(0)

    import prompt_toolkit

    original = prompt_toolkit.PromptSession
    prompt_toolkit.PromptSession = FakeSession
    try:
        return _pick_basic(PickerState(entries=entries, **state_kw), title="basic")
    finally:
        prompt_toolkit.PromptSession = original


def test_basic_number_selects():
    res = _run_basic([E("one"), E("two")], ["2"])
    assert res.kind == "selected"
    assert res.entry.id == "two"


def test_basic_text_filters_then_selects():
    res = _run_basic([E("alpha"), E("bravo")], ["bra", "1"])
    assert res.entry.id == "bravo"


def test_basic_out_of_range_reprompts():
    res = _run_basic([E("only")], ["99", "1"])
    assert res.entry.id == "only"


def test_basic_m_means_manual_and_blank_cancels():
    res = _run_basic([E("x")], ["m"])
    assert res.kind == "manual"
    res = _run_basic([E("x")], [""])
    assert res.kind == "cancel"


def test_basic_no_matches_still_recovers():
    res = _run_basic([E("alpha")], ["zzz", "alp", "1"])
    assert res.kind == "selected"
    assert res.entry.id == "alpha"


# ------------------------------------------------------------- choose_model


def _fake_catalog(entries):
    return Catalog(list(entries), source="test source")


def _choose(keys, monkeypatch, provider="lmstudio", entries=None, catalog=None, **kw):
    """*catalog* can also be an exception for the listing to raise."""
    catalog = catalog or _fake_catalog(
        entries if entries is not None else [E("m1"), E("m2")]
    )
    calls = []

    def fake_list_models(prov, base_url=None, api_key=None, *, refresh=False, **_):
        calls.append(refresh)
        if isinstance(catalog, Exception):
            raise catalog
        return catalog

    monkeypatch.setattr(mc, "list_models", Mock(side_effect=fake_list_models))
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        result = picker.choose_model(provider, input=pipe, output=DummyOutput(), **kw)
    return result, calls


def test_choose_model_select(monkeypatch):
    result, _ = _choose("\r", monkeypatch)
    assert result == ("m1", result[1])
    assert result[1].id == "m1"


def test_choose_model_cancel(monkeypatch):
    result, _ = _choose("\x03", monkeypatch)
    assert result is None


def test_choose_model_manual_entry(monkeypatch):
    # Down to the manual row, enter, then type the id at the mini prompt.
    result, _ = _choose("\x1b[B\x1b[B\r" + "org/custom\r", monkeypatch)
    assert result == ("org/custom", None)


def test_choose_model_refresh_refetches(monkeypatch):
    result, calls = _choose("\x12\r", monkeypatch)
    assert result[0] == "m1"
    assert calls == [False, True]


NO_LISTING = mc.CatalogUnavailable("not supported")


@pytest.mark.parametrize(
    "provider, expected_warnings",
    [("geap", []), ("lmstudio", ["cannot list models: not supported"])],
)
def test_choose_model_falls_back_to_known_models(
    provider, expected_warnings, monkeypatch
):
    warnings = []
    monkeypatch.setattr(picker.fmt, "warning", warnings.append)
    (model_id, _), _ = _choose(
        "\x1b[B\r",
        monkeypatch,
        provider=provider,
        catalog=NO_LISTING,
        known_models=["a", "b"],
    )
    assert model_id == "b"
    assert warnings == expected_warnings

    with pytest.raises(mc.CatalogUnavailable):
        _choose("\r", monkeypatch, provider=provider, catalog=NO_LISTING)


def test_choose_model_manual_entry_without_a_listing(monkeypatch):
    result, _ = _choose(
        "\x1b[B\r" + "gemini-typed\r",
        monkeypatch,
        provider="geap",
        catalog=NO_LISTING,
        known_models=["gemini-a"],
    )
    assert result == ("gemini-typed", None)


def test_choose_model_failed_refresh_keeps_known_models(monkeypatch):
    result, _ = _choose(
        "\x12\r",
        monkeypatch,
        provider="geap",
        catalog=NO_LISTING,
        known_models=["gemini-a"],
    )
    assert result[0] == "gemini-a"


def test_choose_model_refresh_keeps_known_models_on_partial_catalog(monkeypatch):
    result, calls = _choose(
        "\x12\r",
        monkeypatch,
        provider="bedrock",
        catalog=Catalog([E("listed")], source="partial", exhaustive=False),
        current="unlisted",
        known_models=["unlisted"],
        aws_profile="prod",
    )
    assert result[0] == "unlisted"
    assert calls == [False, True]
    assert [c.kwargs["aws_profile"] for c in mc.list_models.call_args_list] == [
        "prod",
        "prod",
    ]


def test_choose_model_hf_search_flow(monkeypatch):
    searched = {}

    def fake_search(query, api_key=None, **_):
        searched["query"] = query
        return [E("org/tail-model", supports_tools=True)]

    monkeypatch.setattr(mc, "search_hf_models", fake_search)
    result, _ = _choose(
        "\x13" + "qwen\r" + "\r",
        monkeypatch,
        provider="huggingface",
        entries=[E("org/main", supports_tools=True)],
    )
    assert searched["query"] == "qwen"
    assert result[0] == "org/tail-model"


def test_choose_model_records_favorite_toggle(monkeypatch):
    from swival import model_prefs as mp

    result, _ = _choose("*\r", monkeypatch, provider="lmstudio")
    assert result[0] == "m1"
    assert mp.load_prefs().is_favorite("lmstudio", "m1")
