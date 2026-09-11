"""Tests for the recorded regions inside the system message."""

from swival import prompt_spans as ps


def _msg(content, spans=None):
    m = {"role": "system", "content": content}
    ps.set_spans(m, spans or [])
    return m


class TestSpanBasics:
    def test_make_span_is_half_open(self):
        span = ps.make_span("instructions", 3, "abc")
        assert span["start"] == 3
        assert span["end"] == 6
        assert span["text"] == "abc"

    def test_set_spans_marks_the_message_even_when_empty(self):
        m = _msg("hello")
        assert ps.is_tracked(m)
        assert ps.get_spans(m) == []

    def test_untracked_message(self):
        assert not ps.is_tracked({"role": "system", "content": "hi"})

    def test_valid_when_offsets_still_hold(self):
        content = "prefixBODYsuffix"
        spans = [ps.make_span("instructions", 6, "BODY")]
        assert ps.spans_valid(content, spans)

    def test_invalid_after_an_untracked_shift(self):
        spans = [ps.make_span("instructions", 6, "BODY")]
        assert not ps.spans_valid("XXprefixBODYsuffix", spans)

    def test_invalid_when_past_the_end(self):
        spans = [ps.make_span("instructions", 6, "BODY")]
        assert not ps.spans_valid("short", spans)


class TestSplice:
    def test_prefix_change_moves_a_later_span(self):
        content = "AAABBBCCC"
        spans = [ps.make_span("instructions", 3, "BBB")]
        out, spans = ps.splice(content, spans, 0, 3, "ZZZZZ")
        assert out == "ZZZZZBBBCCC"
        assert out[spans[0]["start"] : spans[0]["end"]] == "BBB"

    def test_exact_replacement_keeps_identity(self):
        content = "AAABBBCCC"
        spans = [ps.make_span("instructions", 3, "BBB")]
        out, spans = ps.splice(content, spans, 3, 6, "NEW")
        assert out == "AAANEWCCC"
        assert spans[0]["kind"] == "instructions"
        assert spans[0]["text"] == "NEW"
        assert ps.is_protected(spans[0])

    def test_replacing_with_nothing_drops_the_span(self):
        content = "AAABBBCCC"
        spans = [ps.make_span("snapshot", 3, "BBB")]
        out, spans = ps.splice(content, spans, 3, 6, "")
        assert out == "AAACCC"
        assert spans == []

    def test_partial_overlap_drops_the_span(self):
        content = "AAABBBCCC"
        spans = [ps.make_span("instructions", 3, "BBB")]
        _out, spans = ps.splice(content, spans, 4, 8, "x")
        assert spans == []

    def test_insertion_at_a_zero_width_span(self):
        content = "AAACCC"
        spans = [ps.make_span("instructions", 3, "")]
        out, spans = ps.splice(content, spans, 3, 3, "\n\nBBB")
        assert out == "AAA\n\nBBBCCC"
        assert out[spans[0]["start"] : spans[0]["end"]] == "\n\nBBB"

    def test_repeated_splices_stay_aligned(self):
        content = "head" + "RULES" + "tail"
        spans = [ps.make_span("instructions", 4, "RULES")]
        for prefix in ("a", "bb", "ccc"):
            content, spans = ps.splice(content, spans, 0, 0, prefix)
        assert ps.spans_valid(content, spans)
        assert content[spans[0]["start"] : spans[0]["end"]] == "RULES"


class TestMapSegments:
    def test_length_changing_rewrite_keeps_the_region(self):
        content = "aa<|eot|>bb" + "RULES"
        spans = [ps.make_span("instructions", 11, "RULES")]
        out, spans = ps.map_segments(content, spans, lambda t: t.replace("<|", "< |"))
        assert out.endswith("RULES")
        assert out[spans[0]["start"] : spans[0]["end"]] == "RULES"

    def test_rewrite_inside_a_span_updates_its_text(self):
        content = "head" + "A<|x|>B"
        spans = [ps.make_span("instructions", 4, "A<|x|>B")]
        out, spans = ps.map_segments(content, spans, lambda t: t.replace("<|", "< |"))
        assert out[spans[0]["start"] : spans[0]["end"]] == spans[0]["text"]
        assert "< |" in spans[0]["text"]


class TestTruncateOutsideSpans:
    def test_free_text_shrinks_and_the_span_survives(self):
        content = "x" * 100 + "RULES" + "y" * 100
        spans = [ps.make_span("instructions", 100, "RULES")]
        out, spans = ps.truncate_outside_spans(content, spans, 55)
        assert "RULES" in out
        assert out[spans[0]["start"] : spans[0]["end"]] == "RULES"
        assert len(out) <= 60

    def test_unprotected_span_is_not_spared(self):
        content = "x" * 50 + "NOTES" + "y" * 50
        spans = [ps.make_span("snapshot", 50, "NOTES")]
        out, _spans = ps.truncate_outside_spans(content, spans, 20)
        assert len(out) <= 21

    def test_protected_text_alone_is_the_floor(self):
        content = "x" * 20 + "RULES" * 10 + "y" * 20
        spans = [ps.make_span("instructions", 20, "RULES" * 10)]
        out, spans = ps.truncate_outside_spans(content, spans, 5)
        assert out == "RULES" * 10
        assert ps.spans_valid(out, spans)

    def test_no_change_when_already_small(self):
        content = "abcRULESdef"
        spans = [ps.make_span("instructions", 3, "RULES")]
        out, out_spans = ps.truncate_outside_spans(content, spans, 100)
        assert out == content
        assert out_spans == spans


class TestProtectionFollowsFromKind:
    """No record carries a flag that could disagree with what it is."""

    def test_an_instruction_region_is_protected(self):
        assert ps.is_protected(ps.make_span(ps.KIND_INSTRUCTIONS, 0, "rules"))

    def test_a_snapshot_region_is_not(self):
        assert not ps.is_protected(ps.make_span(ps.KIND_SNAPSHOT, 0, "notes"))

    def test_a_replacement_keeps_its_kind(self):
        spans = [ps.make_span(ps.KIND_INSTRUCTIONS, 3, "old")]
        _out, spans = ps.splice("abcoldxyz", spans, 3, 6, "new", target=spans[0])
        assert spans[0]["kind"] == ps.KIND_INSTRUCTIONS
        assert ps.is_protected(spans[0])


class TestShift:
    def test_it_moves_every_region_by_the_prefix(self):
        spans = [
            ps.make_span(ps.KIND_INSTRUCTIONS, 4, "RULES"),
            ps.make_span(ps.KIND_SNAPSHOT, 20, "NOTES"),
        ]
        moved = ps.shift(spans, 7)
        assert [(s["start"], s["end"]) for s in moved] == [(11, 16), (27, 32)]

    def test_it_leaves_the_originals_alone(self):
        spans = [ps.make_span(ps.KIND_INSTRUCTIONS, 4, "RULES")]
        ps.shift(spans, 7)
        assert spans[0]["start"] == 4

    def test_a_shifted_region_still_reads_back(self):
        content = "RULEStail"
        spans = [ps.make_span(ps.KIND_INSTRUCTIONS, 0, "RULES")]
        prefix = "preamble\n\n"
        moved = ps.shift(spans, len(prefix))
        assert ps.spans_valid(prefix + content, moved)

    def test_nothing_to_shift_is_fine(self):
        assert ps.shift([], 10) == []


class TestTrackedness:
    def test_an_empty_list_still_marks_the_message(self):
        m = {"role": "system", "content": "hi"}
        ps.set_spans(m, [])
        assert ps.is_tracked(m)
        assert ps.get_spans(m) == []

    def test_a_message_we_did_not_assemble_is_untracked(self):
        assert not ps.is_tracked({"role": "system", "content": "hi"})

    def test_a_namespace_message_is_untracked(self):
        import types

        assert not ps.is_tracked(types.SimpleNamespace(role="system", content="hi"))
