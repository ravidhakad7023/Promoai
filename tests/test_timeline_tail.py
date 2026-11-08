"""Cliffhanger tail limit (default 2.0 s, configurable): trimming, boundary choice, revalidation."""

from __future__ import annotations

import random

import pytest

from tl_helpers import build_index, lid, ms, one_seg_episode, stored
from promoai.timeline import CLIFF_TAIL_MAX_MS, MAX_CLIP_MS, MIN_CLIP_MS, TimelineConfig, TimelineResolver


def gate(outcome, name):
    return next(g for g in outcome.gates if g.gate == name)


def tail_episode(tail_s: float, cliff_end: float = 30.0, trailing: str = "ACTION") -> list[tuple]:
    """hook (L1) ... cliffhanger dialogue ending at `cliff_end` (L2) ... trailing footage lasting `tail_s` after it (L3)."""
    return [("ANN", 0.0, 1.0, "hook"), ("BOB", cliff_end - 1.5, cliff_end, "the question"),
            (trailing, cliff_end, round(cliff_end + tail_s, 3), "aftermath")]


def tail_resolve(tail_s, config=None, **kw):
    idx = build_index({1: tail_episode(tail_s, **kw)})
    c = stored(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 1), cliff=lid(1, 2))
    return idx, c, TimelineResolver(idx, config).resolve(c)


def trim_of(o):
    return next((a for a in o.clip.adjustments if a.kind == "cliffhanger_trim"), None)


# ------------------------------------------------------------------ the four required tail lengths
@pytest.mark.parametrize("tail_s", [1.9, 2.0])
def test_tail_up_to_and_including_two_seconds_is_left_alone(tail_s):
    _, _, o = tail_resolve(tail_s)
    assert o.accepted and trim_of(o) is None and o.clip.adjustments == []
    assert o.clip.cliffhanger.tail_slack_ms == ms(tail_s)
    assert o.clip.segments[0].end.time_ms == 30_000 + ms(tail_s) and not o.clip.segments[0].end.adjusted
    assert o.clip.total_duration_ms == 30_000 + ms(tail_s)


@pytest.mark.parametrize("tail_s", [2.1, 4.7])
def test_tail_over_two_seconds_is_trimmed_to_a_complete_boundary(tail_s):
    _, _, o = tail_resolve(tail_s)
    t = trim_of(o)
    assert o.accepted and t is not None
    assert (t.side, t.from_ms, t.to_ms) == ("end", 30_000 + ms(tail_s), 30_000)
    assert t.new_boundary_line_id == lid(1, 2)                          # the cliffhanger line defines the new end
    assert t.detail == {"trailing_before_ms": ms(tail_s), "trailing_after_ms": 0, "complete_boundary": True, "limit_ms": 2_000}
    assert f"{tail_s:.1f}s of footage followed cliffhanger" in t.reason and "limit 2.0s" in t.reason
    assert o.clip.total_duration_ms == 30_000 and o.clip.original_total_duration_ms == 30_000 + ms(tail_s)   # recalculated
    assert o.clip.cliffhanger.tail_slack_ms == 0
    assert o.clip.segments[0].dropped_selected_line_ids == [lid(1, 3)]
    g = gate(o, "cliffhanger_reference")
    assert g.passed and "after trimming" in g.reason and g.details["trimmed"] is True


def test_all_mandatory_gates_are_revalidated_after_the_trim():
    _, _, o = tail_resolve(4.7)
    assert [g.gate for g in o.gates] == ["timestamps_valid", "duration_20_60s", "hook_timing_3s", "cliffhanger_reference",
                                         "boundary_integrity", "segments_distinct"]
    assert all(g.passed for g in o.gates)
    d = gate(o, "duration_20_60s")
    assert d.details["total_ms"] == 30_000 and d.details["proposed_total_ms"] == 34_700   # judged on the trimmed length


# ------------------------------------------------------------------ dialogue is never cut
def test_trimming_never_cuts_dialogue_it_keeps_talk_over_whole():
    # CLEO talks over the end of BOB's line (29.0-31.5); a cut at 30.0 would split her, so the first safe end is 31.5
    idx = build_index({1: [("ANN", 0.5, 1.5, "hook"), ("BOB", 28.5, 30.0, "question"), ("CLEO", 29.0, 31.5, "talks over"),
                           ("ACTION", 31.5, 36.0, "aftermath")]})
    c = stored(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 1), cliff=lid(1, 2))
    o = TimelineResolver(idx).resolve(c)
    t = trim_of(o)
    assert o.accepted and t.to_ms == 31_500 and t.new_boundary_line_id == lid(1, 3)
    assert o.clip.segments[0].cut_dialogue_line_ids == [] and gate(o, "boundary_integrity").passed
    assert o.clip.cliffhanger.tail_slack_ms == 1_500 <= 2_000


def test_candidate_is_rejected_when_dialogue_prevents_a_trim_within_the_limit():
    # a long talk-over (29.0-33.5) means no dialogue-safe end exists within 2 s of the cliffhanger (ends at 30.0)
    idx = build_index({1: [("ANN", 0.5, 1.5, "hook"), ("BOB", 28.5, 30.0, "question"), ("CLEO", 29.0, 33.5, "long talk-over"),
                           ("ACTION", 33.5, 38.0, "aftermath")]})
    c = stored(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 1), cliff=lid(1, 2))
    o = TimelineResolver(idx).resolve(c)
    assert not o.accepted
    g = gate(o, "cliffhanger_reference")
    assert not g.passed and "cannot be trimmed to a valid boundary within the limit without cutting dialogue" in g.reason
    assert o.clip.segments[0].cut_dialogue_line_ids == []                # even the rejected version never cuts dialogue
    assert any(a.kind == "cliffhanger_trim" for a in o.clip.adjustments)  # the attempt is on record
    assert any("cliffhanger_reference" in r for r in o.rejection_reasons)


# ------------------------------------------------------------------ choosing the boundary
def test_a_complete_shot_boundary_is_preferred_over_cutting_a_shot():
    # a camera shot 28.0-31.5 spans the cliffhanger end (30.0): cutting at 30.0 would slice the shot; 31.5 is clean
    idx = build_index({1: [("ANN", 0.5, 1.5, "hook"), ("BOB", 28.5, 30.0, "question"), ("CAMERA", 28.0, 31.5, "reaction shot"),
                           ("ACTION", 31.5, 36.0, "aftermath")]})
    c = stored(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 1), cliff=lid(1, 2))
    o = TimelineResolver(idx).resolve(c)
    t = trim_of(o)
    assert o.accepted and t.to_ms == 31_500 and t.detail["complete_boundary"] is True
    assert o.clip.segments[0].partially_cut_line_ids == []
    assert "complete boundary (no block cut)" in t.reason


def test_falls_back_to_cutting_a_shot_when_no_complete_boundary_exists_in_the_limit():
    idx = build_index({1: [("ANN", 0.5, 1.5, "hook"), ("BOB", 28.5, 30.0, "question"), ("CAMERA", 28.0, 40.0, "long shot"),
                           ("ACTION", 31.0, 36.0, "aftermath")]})
    c = stored(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 1), cliff=lid(1, 2))
    o = TimelineResolver(idx).resolve(c)
    t = trim_of(o)
    assert o.accepted and t.to_ms == 30_000 and t.detail["complete_boundary"] is False
    assert o.clip.segments[0].partially_cut_line_ids == [lid(1, 3)] and o.clip.segments[0].cut_dialogue_line_ids == []


# ------------------------------------------------------------------ duration after trimming
def test_trim_that_pushes_the_clip_below_20_seconds_is_rejected_with_both_durations():
    _, c, o = tail_resolve(4.7, cliff_end=18.0)                  # 22.7 s as proposed, 18.0 s once trimmed
    assert not o.accepted
    g = gate(o, "duration_20_60s")
    assert not g.passed and "18.0s after boundary adjustment (was 22.7s as proposed)" in g.reason
    assert o.clip.original_total_duration_ms == 22_700 and o.clip.total_duration_ms == 18_000
    assert trim_of(o) is not None                                  # the attempt is recorded even though the candidate is rejected


def test_trim_can_rescue_an_over_long_clip():
    _, _, o = tail_resolve(4.7, cliff_end=57.0)                    # 61.7 s as proposed -> 57.0 s
    assert o.accepted and o.clip.total_duration_ms == 57_000 and o.clip.original_total_duration_ms == 61_700


def test_exactly_20_seconds_after_the_trim_is_valid_and_19_9_is_not():
    assert tail_resolve(4.7, cliff_end=20.0)[2].accepted
    assert not tail_resolve(4.7, cliff_end=19.9)[2].accepted


# ------------------------------------------------------------------ records
def test_original_candidate_data_is_never_modified_and_every_adjustment_is_recorded():
    idx, c, _ = tail_resolve(4.7)
    before = c.model_dump()
    o = TimelineResolver(idx).resolve(c)
    assert c.model_dump() == before
    a = o.clip.adjustments[0]
    assert a.kind == "cliffhanger_trim" and a.reason and a.new_boundary_line_id and a.segment_index == 1
    assert o.clip.segments[0].end.adjusted and o.clip.segments[0].end.original_ms == 34_700
    assert o.clip.segments[0].end.source_line > 0


def test_adjustments_have_a_kind_for_every_boundary_change():
    idx = build_index({1: [("CAMERA", 0, 31.1), ("SOUND FX", 1.5, 2.5), ("ANN", 3.1, 4.6, "hook"), ("BOB", 27.0, 28.5, "question"),
                           ("ACTION", 28.5, 31.1, "tail")]})
    c = stored(idx, segs=[((1, 1), (1, 5))], hook=lid(1, 3), cliff=lid(1, 4))
    o = TimelineResolver(idx).resolve(c)
    assert sorted(a.kind for a in o.clip.adjustments) == ["cliffhanger_trim", "hook_alignment"]


# ------------------------------------------------------------------ configuration
def test_tail_limit_is_configurable():
    _, _, old = tail_resolve(4.7, TimelineConfig(cliff_tail_max_ms=5_000))
    assert trim_of(old) is None and old.accepted and old.clip.cliffhanger.tail_slack_ms == 4_700     # the former 5 s behaviour
    _, _, tight = tail_resolve(1.5, TimelineConfig(cliff_tail_max_ms=1_000))
    assert trim_of(tight).detail["limit_ms"] == 1_000 and tight.clip.cliffhanger.tail_slack_ms == 0
    _, _, none = tail_resolve(0.5, TimelineConfig(cliff_tail_max_ms=0))
    assert trim_of(none) is not None and none.clip.cliffhanger.tail_slack_ms == 0


def test_default_cliffhanger_tail_limit_is_two_seconds():
    assert CLIFF_TAIL_MAX_MS == 2_000 == TimelineConfig().cliff_tail_max_ms
    assert TimelineConfig().to_dict() == {"min_clip_ms": 20_000, "max_clip_ms": 60_000, "hook_max_ms": 3_000,
                                          "hook_preferred_lead_ms": 2_000, "cliff_tail_max_ms": 2_000}


def test_other_limits_are_configurable_too():
    idx = build_index({1: one_seg_episode(25.0)})
    c = stored(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert TimelineResolver(idx).resolve(c).accepted
    assert not TimelineResolver(idx, TimelineConfig(min_clip_ms=30_000)).resolve(c).accepted
    assert not TimelineResolver(idx, TimelineConfig(max_clip_ms=24_000)).resolve(c).accepted


def test_trim_applies_only_to_the_last_segment_of_a_multi_segment_clip():
    idx = build_index({1: [("ANN", 0.5, 1.5, "hook"), ("BOB", 13.5, 15.0, "mid"), ("ACTION", 15.0, 19.0, "gap footage"),
                           ("CLEO", 40.0, 41.5, "second"), ("DAN", 58.5, 60.0, "question"), ("ACTION", 60.0, 66.0, "aftermath")]})
    c = stored(idx, segs=[((1, 1), (1, 3)), ((1, 4), (1, 6))], hook=lid(1, 1), cliff=lid(1, 5))
    o = TimelineResolver(idx).resolve(c)
    assert o.accepted
    trims = [a for a in o.clip.adjustments if a.kind == "cliffhanger_trim"]
    assert len(trims) == 1 and trims[0].segment_index == 2 and trims[0].to_ms == 60_000
    assert o.clip.segments[0].duration_ms == 18_500 and o.clip.segments[1].duration_ms == 20_000   # 0.5-19.0 and 40.0-60.0
    assert o.clip.total_duration_ms == 38_500


@pytest.mark.parametrize("seed", range(40))
def test_property_after_trim_the_tail_is_within_limit_and_dialogue_is_whole(seed):
    rnd = random.Random(1000 + seed)
    blocks = [("ANN", 0.5, 1.5, "hook")]
    cliff_end = round(rnd.uniform(22, 50), 1)
    blocks.append(("BOB", cliff_end - 1.5, cliff_end, "q"))
    for _ in range(rnd.randint(1, 5)):
        s = round(cliff_end - rnd.uniform(0, 2) + rnd.uniform(0, 3), 1)
        blocks.append((rnd.choice(["CLEO", "ACTION", "CAMERA", "SOUND FX"]), s, round(s + rnd.uniform(0.5, 4), 1), "t"))
    idx = build_index({1: blocks})
    c = stored(idx, segs=[((1, 1), (1, len(blocks)))], hook=lid(1, 1), cliff=lid(1, 2))
    o = TimelineResolver(idx).resolve(c)
    assert o.clip.segments[0].cut_dialogue_line_ids == []                 # holds for rejected candidates too
    if o.accepted:
        assert 0 <= o.clip.cliffhanger.tail_slack_ms <= CLIFF_TAIL_MAX_MS
        assert MIN_CLIP_MS <= o.clip.total_duration_ms <= MAX_CLIP_MS
