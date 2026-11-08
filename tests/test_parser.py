"""Parser behaviour on small synthetic scripts (no confidential data needed)."""

from __future__ import annotations

import re

import pytest

from conftest import FOOTER, HEADER, SCENE_1, SCENE_2, codes, make_script
from promoai.models import LineKind, Severity, TimestampStatus
from promoai.parser import episode_info_from_filename, split_source_lines


# ---------------------------------------------------------------- basic structure
def test_clean_script_parses_without_warnings(parse, good_script):
    ep = parse(good_script)
    assert ep.warnings == []
    assert ep.unparsed == []
    assert len(ep.scenes) == 2
    assert [len(s.lines) for s in ep.scenes] == [5, 1]
    assert ep.coverage.fully_accounted


def test_kinds_speakers_and_delivery(parse, good_script):
    lines = parse(good_script).lines
    assert [l.kind for l in lines[:5]] == [
        LineKind.CAMERA, LineKind.ACTION, LineKind.DIALOGUE, LineKind.SOUND_FX, LineKind.DIALOGUE,
    ]
    ann = lines[2]
    assert ann.speaker == "ANN"
    assert ann.delivery == ["Calm, Quiet"]
    assert ann.text == "Hello there."            # quotes stripped
    assert ann.text_raw == '"Hello there."'      # original preserved
    assert lines[0].speaker is None                # camera has no speaker


def test_stacked_parentheticals(parse, good_script):
    bob = parse(good_script).lines[4]
    assert bob.speaker == "BOB"
    assert bob.delivery == ["O.S.", "Shouting"]
    assert bob.speaker_raw == "BOB (O.S.) (Shouting)"


def test_scene_heading_split(parse, good_script):
    s1, s2 = parse(good_script).scenes
    assert (s1.location_type, s1.location, s1.time_of_day) == ("INT", "TEST HALL", "DAY")
    assert (s2.location_type, s2.location, s2.time_of_day) == ("EXT", "GARDEN", "NIGHT")  # em dash


def test_heading_interior_is_not_int(parse):
    text = make_script(1, SCENE_1.replace("INT. TEST HALL – DAY", "INTERIOR HALL – DAY"))
    sc = parse(text).scenes[0]
    assert sc.location_type is None
    assert "unrecognised_heading" in codes(parse(text))


def test_header_fields(parse, good_script):
    ep = parse(good_script)
    assert ep.logline == "A test logline."
    assert ep.continuity_notes == "Test notes."


# ---------------------------------------------------------------- unique IDs
def test_ids_unique_and_well_formed(parse, good_script):
    ep = parse(good_script)
    scene_ids = [s.scene_id for s in ep.scenes]
    line_ids = [l.line_id for l in ep.lines]
    assert scene_ids == ["E001-S01", "E001-S02"]
    assert len(set(line_ids)) == len(line_ids) == 6
    assert all(re.fullmatch(r"E001-S0[12]-L\d{3}", i) for i in line_ids)


def test_line_numbers_continue_across_scenes(parse, good_script):
    ep = parse(good_script)
    assert [l.seq for l in ep.lines] == [1, 2, 3, 4, 5, 6]
    assert ep.scenes[1].lines[0].line_id == "E001-S02-L006"
    assert ep.scenes[1].lines[0].index_in_scene == 1


def test_ids_embed_scene_and_episode_relationships(parse, good_script):
    ep = parse(good_script)
    for sc in ep.scenes:
        assert sc.episode_id == ep.episode_id
        for ln in sc.lines:
            assert ln.scene_id == sc.scene_id
            assert ln.episode_id == ep.episode_id
            assert ln.line_id.startswith(sc.scene_id + "-L")


def test_ids_are_deterministic(parse, good_script):
    a = [l.line_id for l in parse(good_script).lines]
    b = [l.line_id for l in parse(good_script).lines]
    assert a == b


def test_duplicate_scene_numbers_are_renumbered_not_merged(parse):
    text = make_script(1, SCENE_1, SCENE_2.replace("SCENE 02", "SCENE 01"))
    ep = parse(text)
    assert [s.scene_id for s in ep.scenes] == ["E001-S01", "E001-S02"]
    assert "scene_number_conflict" in codes(ep)
    assert len({l.line_id for l in ep.lines}) == len(ep.lines)


# ---------------------------------------------------------------- timestamps
def test_timestamps_preserved_raw_and_ms(parse, good_script):
    l = parse(good_script).lines[2]  # ANN "Hello there."
    assert (l.start_raw, l.end_raw) == ("00:05.00", "00:06.50")
    assert (l.start_ms, l.end_ms) == (5000, 6500)
    assert l.timestamp_status is TimestampStatus.OK


def test_scene_timestamps(parse, good_script):
    s1, s2 = parse(good_script).scenes
    assert (s1.start_ms, s1.end_ms, s1.times_derived) == (0, 20000, False)
    assert (s2.start_raw, s2.end_raw) == ("00:20.00", "00:30.00")
    assert parse(good_script).duration_ms == 30000


def test_every_line_keeps_timestamp_association(parse, good_script):
    for l in parse(good_script).lines:
        assert l.timestamp_status is TimestampStatus.OK
        assert l.start_ms is not None and l.end_ms is not None and l.start_ms <= l.end_ms


def test_missing_timestamp_line_keeps_the_content(parse):
    text = make_script(1, SCENE_1.replace("Start: (00:05.00) | End: (00:06.50)\n", ""))
    ep = parse(text)
    ann = [l for l in ep.lines if l.speaker == "ANN"][0]
    assert ann.timestamp_status is TimestampStatus.MISSING
    assert ann.start_ms is None and ann.start_raw is None
    assert ann.text == "Hello there."            # not dropped
    assert "missing_timestamp" in codes(ep)
    assert ep.coverage.fully_accounted and ep.unparsed == []
    assert len(ep.lines) == 5                     # nothing lost, other lines unaffected


def test_label_directly_followed_by_label_is_kept(parse):
    text = make_script(1, SCENE_1.replace('Start: (00:05.00) | End: (00:06.50)\n"Hello there."\n', ""))
    ep = parse(text)
    ann = [l for l in ep.lines if l.speaker == "ANN"][0]
    assert ann.timestamp_status is TimestampStatus.MISSING and ann.text == ""
    assert {"missing_timestamp", "empty_body"} <= set(codes(ep))


def test_unreadable_timestamp_is_flagged_not_fatal(parse):
    text = make_script(1, SCENE_1.replace("Start: (00:05.00) | End: (00:06.50)", "Start: (--:--) | End: (00:06.50)"))
    ep = parse(text)
    ann = [l for l in ep.lines if l.speaker == "ANN"][0]
    assert ann.timestamp_status is TimestampStatus.INVALID
    assert ann.start_ms is None
    assert ann.text == "Hello there."
    assert "invalid_timestamp" in codes(ep)


def test_end_before_start_is_flagged(parse):
    text = make_script(1, SCENE_1.replace("Start: (00:05.00) | End: (00:06.50)", "Start: (00:07.00) | End: (00:06.50)"))
    ep = parse(text)
    ann = [l for l in ep.lines if l.speaker == "ANN"][0]
    assert ann.timestamp_status is TimestampStatus.INCONSISTENT
    assert (ann.start_ms, ann.end_ms) == (7000, 6500)    # raw values kept, not "fixed"
    assert "end_before_start" in codes(ep)


def test_line_outside_scene_window_warns(parse):
    text = make_script(1, SCENE_1.replace("End: (00:09.25)", "End: (00:21.00)"))
    assert "line_outside_scene" in codes(parse(text))


def test_missing_scene_times_derived_from_lines(parse):
    scene = SCENE_1.replace("Scene start: 00:00.00\n", "").replace("Scene end: 00:20.00\n", "")
    ep = parse(make_script(1, scene))
    sc = ep.scenes[0]
    assert sc.times_derived is True
    assert (sc.start_ms, sc.end_ms) == (0, 9250)
    assert "scene_times_derived" in codes(ep)


def test_scene_discontinuity_warns(parse):
    text = make_script(1, SCENE_1, SCENE_2.replace("00:20.00", "00:25.00", 1))
    assert "scene_time_discontinuity" in codes(parse(text))


# ---------------------------------------------------------------- episode identification
def test_episode_number_from_header(parse):
    ep = parse(make_script(7), name="whatever.txt")
    assert (ep.episode_id, ep.episode_number, ep.episode_number_source) == ("E007", 7, "header")


def test_episode_number_falls_back_to_filename(parse):
    text = make_script(1).replace("EPISODE NUMBER: Episode 1\n", "")
    ep = parse(text, name="episode-012-Episode 99.txt", fallback_number=12)
    assert (ep.episode_id, ep.episode_number_source) == ("E012", "filename")
    assert "missing_header_field" in codes(ep)


def test_header_filename_mismatch_prefers_header_and_warns(parse):
    ep = parse(make_script(5), name="episode-009-Episode 1.txt", fallback_number=9)
    assert ep.episode_number == 5
    assert "episode_number_mismatch" in codes(ep)


def test_number_override_used_for_duplicates(parse):
    ep = parse(make_script(5), number_override=77)
    assert ep.episode_id == "E077" and ep.episode_number_source == "sequence"
    assert all(l.line_id.startswith("E077-") for l in ep.lines)
    assert all(w.episode_id == "E077" for w in ep.warnings)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("episode-001-Episode 1671703.txt", (1, "1671703")),
        ("episode-050-Episode 1697563.txt", (50, "1697563")),
        ("Episode_12.txt", (12, None)),
        ("notes.txt", (None, None)),
    ],
)
def test_filename_parsing(name, expected):
    assert episode_info_from_filename(name) == expected


def test_episode_count_is_not_hardcoded(parse):
    for n in (1, 3, 120):
        ep = parse(make_script(n))
        assert ep.episode_id == f"E{n:03d}"


# ---------------------------------------------------------------- formatting variations
def test_uppercase_scene_markers_like_episode_6(parse):
    scene = (
        SCENE_1.replace("Scene heading:", "SCENE HEADING:")
        .replace("Scene start:", "SCENE START:")
        .replace("Scene end:", "SCENE END:")
        .replace("Actions and dialogue:", "ACTIONS AND DIALOGUE:")
    )
    ep = parse(make_script(1, scene))
    assert ep.warnings == []
    assert ep.scenes[0].heading.startswith("INT. TEST HALL")
    assert (ep.scenes[0].start_ms, ep.scenes[0].end_ms) == (0, 20000)
    assert len(ep.scenes[0].lines) == 5


def test_missing_actions_and_dialogue_marker_is_harmless(parse):
    ep = parse(make_script(1, SCENE_1.replace("Actions and dialogue:\n", "")))
    assert ep.warnings == [] and len(ep.lines) == 5


def test_crlf_line_endings(parse, good_script):
    a = parse(good_script)
    b = parse(good_script.replace("\n", "\r\n"))
    assert [l.model_dump() for l in a.lines] == [l.model_dump() for l in b.lines]


def test_bom_is_stripped_by_decoder():
    from promoai.pipeline import decode_script

    text, note = decode_script(b"\xef\xbb\xbfEPISODE NUMBER: Episode 1\n")
    assert text.startswith("EPISODE") and note is None


def test_non_utf8_bytes_decoded_with_note():
    from promoai.pipeline import decode_script

    text, note = decode_script("caf\xe9".encode("cp1252"))
    assert text == "caf\xe9" and "cp1252" in note


def test_unicode_line_separator_does_not_shift_line_numbers():
    text = "a b\nc"
    assert split_source_lines(text) == ["a b", "c"]


def test_multiline_body_is_joined(parse):
    scene = SCENE_1.replace("Ann enters.", "Ann enters,\nshaking off the rain.")
    ep = parse(make_script(1, scene))
    act = ep.lines[1]
    assert act.text == "Ann enters, shaking off the rain."
    assert act.text_raw == "Ann enters,\nshaking off the rain."
    assert act.source_end_line == act.source_line + 3
    assert ep.warnings == []


def test_blank_line_between_timestamp_and_body(parse):
    scene = SCENE_1.replace("Start: (00:01.00) | End: (00:04.80)\nAnn enters.", "Start: (00:01.00) | End: (00:04.80)\n\nAnn enters.")
    ep = parse(make_script(1, scene))
    assert ep.lines[1].text == "Ann enters." and ep.warnings == []


def test_inline_body_after_label(parse):
    scene = SCENE_1.replace('**ANN (Calm, Quiet):**\nStart: (00:05.00) | End: (00:06.50)\n"Hello there."',
                            '**ANN (Calm, Quiet):** "Hello there."\nStart: (00:05.00) | End: (00:06.50)')
    ep = parse(make_script(1, scene))
    ann = [l for l in ep.lines if l.speaker == "ANN"][0]
    assert ann.text == "Hello there." and ann.timestamp_status is TimestampStatus.OK
    assert "inline_body" in codes(ep)


def test_label_with_colon_outside_bold(parse):
    scene = SCENE_1.replace("**ACTION:**", "**ACTION**:")
    ep = parse(make_script(1, scene))
    assert ep.lines[1].kind is LineKind.ACTION and ep.warnings == []


def test_curly_quotes_and_unquoted_dialogue(parse):
    scene = SCENE_1.replace('"Hello there."', "“Hello there.”").replace('"Who goes there?"', "Who goes there?")
    ep = parse(make_script(1, scene))
    assert ep.lines[2].text == "Hello there."
    assert ep.lines[4].text == "Who goes there?"
    assert "unquoted_dialogue" in codes(ep)


def test_apostrophes_are_not_stripped_as_quotes(parse):
    scene = SCENE_1.replace('"Hello there."', "'Tis the season, ma'am'")
    ep = parse(make_script(1, scene))
    assert ep.lines[2].text == "'Tis the season, ma'am'"


def test_wrapped_logline_is_joined(parse):
    text = make_script(1).replace("A test logline.", "A test logline\nthat wraps.")
    ep = parse(text)
    assert ep.logline == "A test logline that wraps."
    assert ep.coverage.fully_accounted and ep.unparsed == []


def test_missing_markers_and_header_fields_warn(parse):
    text = make_script(1, footer="").replace("--- SCRIPT START ---\n", "").replace("EPISODE LOGLINE: A test logline.\n", "")
    c = codes(parse(text))
    assert {"missing_script_start", "missing_script_end", "missing_header_field"} <= set(c)


def test_content_before_first_scene_creates_implicit_scene(parse):
    text = HEADER.format(n=1) + "\n**ANN:**\nStart: (00:00.00) | End: (00:01.00)\n\"Hi.\"\n" + SCENE_1 + FOOTER
    ep = parse(text)
    assert "implicit_scene" in codes(ep)
    assert len(ep.scenes) == 2 and ep.scenes[0].lines[0].speaker == "ANN"
    assert ep.scenes[1].scene_number == 2           # explicit SCENE 01 renumbered, not merged
    assert "scene_number_conflict" in codes(ep)


# ---------------------------------------------------------------- nothing silently lost
def test_unrecognised_lines_go_to_unparsed_with_location(parse):
    text = make_script(1, SCENE_1 + "\nA stray note from an editor.\n")
    ep = parse(text)
    assert len(ep.unparsed) == 1
    u = ep.unparsed[0]
    assert u.text == "A stray note from an editor."
    assert text.split("\n")[u.source_line - 1] == "A stray note from an editor."
    assert "unparsed_line" in codes(ep)
    assert ep.coverage.fully_accounted


def test_orphan_timestamp_goes_to_unparsed(parse):
    text = make_script(1, SCENE_1 + "\nStart: (00:10.00) | End: (00:11.00)\n")
    ep = parse(text)
    assert [u.reason for u in ep.unparsed] == ["timestamp line without a preceding label"]


def test_content_after_script_end_is_kept_and_flagged(parse):
    ep = parse(make_script(1, footer="\n--- SCRIPT END ---\nTrailing text\n"))
    assert "content_after_script_end" in codes(ep)
    assert [u.text for u in ep.unparsed] == ["Trailing text"]


def test_coverage_accounts_for_every_nonblank_line(parse, good_script):
    for text in (good_script, good_script + "\nrandom\njunk\n", good_script.replace("\n", "\r\n")):
        ep = parse(text)
        nb = sum(1 for l in split_source_lines(text) if l.strip())
        assert ep.coverage.non_blank_lines == nb
        assert ep.coverage.parsed_lines + ep.coverage.unparsed_lines == nb
        assert ep.coverage.fully_accounted


def test_empty_file_reports_an_error(parse):
    ep = parse("")
    assert ep.lines == [] and ep.scenes == []
    assert any(w.code == "no_scenes" and w.severity is Severity.ERROR for w in ep.warnings)


def test_garbage_file_does_not_raise_and_keeps_everything(parse):
    text = "\x00\x01 binary-ish\nnot a script at all\n\n12345\n"
    ep = parse(text)
    assert ep.lines == []
    assert len(ep.unparsed) == 3 and ep.coverage.fully_accounted


# ---------------------------------------------------------------- mapping IDs back to source
def test_every_id_maps_back_to_its_source_location(parse, good_script):
    ep = parse(good_script)
    src = split_source_lines(good_script)
    for l in ep.lines:
        label_line = src[l.source_line - 1]
        assert label_line.startswith("**") and l.speaker_raw in label_line
        ts_line = src[l.source_line]                      # timestamp is the very next line
        assert l.start_raw in ts_line and l.end_raw in ts_line
        body = "\n".join(src[l.source_line + 1 : l.source_end_line])
        assert body == l.text_raw


def test_roundtrip_through_json_and_index(parse, good_script):
    from promoai.index import ScreenplayIndex
    from promoai.models import Episode

    ep = parse(good_script)
    ep2 = Episode.model_validate_json(ep.model_dump_json())
    assert ep2 == ep
    idx = ScreenplayIndex([ep2])
    first = ep.lines[0]
    assert idx.get_line(first.line_id).start_ms == first.start_ms
    assert idx.get_scene("E001-S02").scene_number == 2
    with pytest.raises(KeyError):
        idx.get_line("E999-S01-L001")


def test_index_rejects_duplicate_ids(parse, good_script):
    from promoai.index import ScreenplayIndex

    ep = parse(good_script)
    with pytest.raises(ValueError, match="duplicate episode id"):
        ScreenplayIndex([ep, ep])
