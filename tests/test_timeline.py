"""Deterministic timeline resolution + mandatory deterministic gates (no LLM, no network)."""

from __future__ import annotations

import random

import pytest

from conftest import cand
from tl_helpers import build_index, lid, ms, one_seg_episode, stored
from promoai.models import LineKind
from promoai.timeline import (
    CLIFF_TAIL_MAX_MS, HOOK_MAX_MS, MAX_CLIP_MS, MIN_CLIP_MS, TimelineConfig, TimelineResolver,
)


def resolve(index, **kw):
    return TimelineResolver(index).resolve(stored(index, **kw))


def gate(outcome, name):
    return next(g for g in outcome.gates if g.gate == name)


# ------------------------------------------------------------------ overlapping blocks
def test_overlapping_blocks_are_a_span_not_a_sum():
    # camera 0-10, action 1-4, dialogue 2-5, closing 8.5-10: durations sum to 10+3+3+1.5 = 17.5 but the span is 10.0
    idx = build_index({1: [("CAMERA", 0, 10), ("ACTION", 1, 4), ("ANN", 2, 5), ("BOB", 8.5, 10)]})
    o = resolve(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 3), cliff=lid(1, 4))
    seg = o.clip.segments[0]
    assert seg.selected_window_ms == 10_000 and seg.duration_ms == 10_000
    assert sum(l.end_ms - l.start_ms for l in idx.episodes["E001"].lines) == 17_500   # what a naive sum would give
    assert o.clip.total_duration_ms == 10_000


def test_file_order_is_not_timeline_order():
    # lines listed out of time order: the span still uses min start / max end, wherever they sit in the file
    idx = build_index({1: [("BOB", 30, 31.5), ("CAMERA", 0, 31.5), ("ANN", 0.5, 2)]})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 3), cliff=lid(1, 1))
    assert (o.clip.segments[0].start.time_ms, o.clip.segments[0].end.time_ms) == (0, 31_500)


# ------------------------------------------------------------------ id -> timestamp mapping
def test_boundaries_map_to_the_defining_block_and_source_location():
    idx = build_index({1: one_seg_episode(30.0)})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    s = o.clip.segments[0]
    first, last = idx.lines[s.start.line_id], idx.lines[s.end.line_id]
    assert (first.start_ms, last.end_ms) == (s.start.time_ms, s.end.time_ms) == (0, 30_000)
    assert s.start.timecode == "00:00.00" and s.end.timecode == "00:30.00"
    assert (s.start.source_file, s.start.source_line) == (first.source_file, first.source_line)
    assert s.start.line_id == lid(1, 1) and s.end.line_id in (lid(1, 1), lid(1, 3))   # camera or closing line both end at 30.0
    assert not s.start.adjusted and not s.end.adjusted and o.clip.adjustments == []


# ------------------------------------------------------------------ 20 s / 60 s boundaries
@pytest.mark.parametrize("dur,ok", [(19.5, False), (19.9, False), (20.0, True), (20.1, True),
                                    (59.9, True), (60.0, True), (60.1, False), (75.8, False)])
def test_duration_boundaries(dur, ok):
    idx = build_index({1: one_seg_episode(dur)})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert o.clip.total_duration_ms == ms(dur)
    assert gate(o, "duration_20_60s").passed is ok and o.accepted is ok
    assert MIN_CLIP_MS == 20_000 and MAX_CLIP_MS == 60_000
    if not ok:
        assert any("duration_20_60s" in r for r in o.rejection_reasons)
        assert "must be" in gate(o, "duration_20_60s").reason


def test_over_long_clip_can_be_rescued_by_trimming_after_the_cliffhanger():
    # 70 s selected, but the cliffhanger line ends at 45 s: 25 s of trailing footage is trimmed
    idx = build_index({1: [("CAMERA", 0, 70), ("ANN", 0.5, 2), ("BOB", 43.5, 45), ("ANN", 60, 61.5, "after")]})
    o = resolve(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 2), cliff=lid(1, 3))
    assert o.accepted and o.clip.total_duration_ms == 45_000          # ends where the cliffhanger line ends
    assert o.clip.original_total_duration_ms == 70_000                 # as proposed it would have failed
    adj = [a for a in o.clip.adjustments if a.side == "end"]
    assert len(adj) == 1 and adj[0].new_boundary_line_id == lid(1, 3) and "cliffhanger" in adj[0].reason
    assert o.clip.segments[0].dropped_selected_line_ids == [lid(1, 4)]


# ------------------------------------------------------------------ hook timing
@pytest.mark.parametrize("hook_at,expect_pass,moved", [(0.0, True, False), (2.9, True, False), (3.0, True, False)])
def test_hook_at_or_before_three_seconds_needs_no_adjustment(hook_at, expect_pass, moved):
    idx = build_index({1: one_seg_episode(30.0, hook_at=hook_at)})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert gate(o, "hook_timing_3s").passed is expect_pass
    assert o.clip.hook.offset_ms == ms(hook_at) and not o.clip.adjustments
    assert HOOK_MAX_MS == 3_000


def test_hook_at_3_1s_is_not_trusted_and_the_start_is_moved():
    # blocks start at 0 (camera) and at 1.5 (filler) and 3.1 (hook); LLM claimed the hook was fine
    idx = build_index({1: [("CAMERA", 0, 31.1), ("SOUND FX", 1.5, 2.5), ("ANN", 3.1, 4.6, "hook"), ("BOB", 29.6, 31.1, "end")]})
    o = resolve(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 3), cliff=lid(1, 4),
                gates={"hook_in_first_3s": True, "hook_note": "n", "unresolved_cliffhanger": True, "cliffhanger_note": "n",
                       "standalone_clear": True, "standalone_note": "n", "duration_plausible_20_60s": True, "duration_note": "n"})
    assert o.clip.hook.offset_before_ms == 3_100 and o.clip.hook.offset_ms == 1_600
    assert gate(o, "hook_timing_3s").passed and o.accepted
    s = o.clip.segments[0]
    assert s.start.time_ms == 1_500 and s.start.adjusted and s.start.original_ms == 0
    assert s.start.line_id == lid(1, 2)                                   # source reference of the new boundary
    assert o.clip.total_duration_ms == 29_600 and o.clip.original_total_duration_ms == 31_100   # recalculated
    adj = o.clip.adjustments[0]
    assert (adj.side, adj.from_ms, adj.to_ms, adj.new_boundary_line_id) == ("start", 0, 1_500, lid(1, 2))
    assert "3.1s" in adj.reason and "limit 3.0s" in adj.reason


def test_hook_move_prefers_a_two_second_lead_in_when_available():
    idx = build_index({1: [("CAMERA", 0, 35), ("ACTION", 1.0, 2.0), ("ACTION", 4.0, 5.0), ("ANN", 6.0, 7.5, "hook"), ("BOB", 33.5, 35)]})
    o = resolve(idx, segs=[((1, 1), (1, 5))], hook=lid(1, 4), cliff=lid(1, 5))
    assert o.clip.segments[0].start.time_ms == 4_000 and o.clip.hook.offset_ms == 2_000      # earliest start within 2.0 s


def test_hook_that_cannot_be_fixed_is_rejected():
    # long speech 0-3.5 straddles the only start inside the 3.0 s window (the hook at 3.1) -> no valid boundary
    idx = build_index({1: [("CAMERA", 0, 30), ("OLD MAN", 0.0, 3.5, "long speech"), ("ANN", 3.1, 4.6, "hook"), ("BOB", 28.5, 30)]})
    o = resolve(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 3), cliff=lid(1, 4))
    assert not o.accepted and not gate(o, "hook_timing_3s").passed
    assert "no valid earlier-start adjustment" in gate(o, "hook_timing_3s").reason


def test_hook_without_a_timestamp_is_rejected():
    blocks = [("CAMERA", 0, 30), ("ANN", None, None, "no time"), ("BOB", 28.5, 30)]
    idx = build_index({1: blocks})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert not o.accepted and not gate(o, "timestamps_valid").passed
    assert "no valid timestamp" in o.rejection_reasons[0]


# ------------------------------------------------------------------ boundary snapping
def test_dialogue_straddling_the_start_is_included_whole():
    # selected range starts at the camera block (4.0) but a dialogue line outside the range runs 3.0-5.0
    idx = build_index({1: [("ANN", 3.0, 5.0, "in progress"), ("CAMERA", 4.0, 34.0), ("ACTION", 4.5, 6.0), ("BOB", 32.5, 34.0)]})
    o = resolve(idx, segs=[((1, 2), (1, 4))], hook=lid(1, 3), cliff=lid(1, 4))
    s = o.clip.segments[0]
    assert s.start.time_ms == 3_000 and s.start.adjusted and s.start.line_id == lid(1, 1)
    assert "start fell inside dialogue" in s.start.adjustment_reason
    assert s.cut_dialogue_line_ids == [] and gate(o, "boundary_integrity").passed


def test_dialogue_straddling_the_end_is_included_whole():
    idx = build_index({1: [("CAMERA", 0, 30), ("ANN", 0.5, 2), ("BOB", 28.5, 30), ("OLD MAN", 29.0, 32.0, "keeps talking")]})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    s = o.clip.segments[0]
    assert s.end.time_ms == 32_000 and s.end.adjusted and s.end.line_id == lid(1, 4)
    assert o.clip.total_duration_ms == 32_000 and gate(o, "boundary_integrity").passed


def test_camera_blocks_may_straddle_an_edge_but_are_reported():
    idx = build_index({1: [("CAMERA", 0, 40), ("ANN", 10.0, 11.0, "hook"), ("ACTION", 9.0, 12.0), ("BOB", 38.5, 40.0)]})
    o = resolve(idx, segs=[((1, 2), (1, 4))], hook=lid(1, 2), cliff=lid(1, 4))
    s = o.clip.segments[0]
    assert s.start.time_ms == 9_000 and lid(1, 1) in s.partially_cut_line_ids       # camera shot crosses the start edge
    assert s.cut_dialogue_line_ids == []


# ------------------------------------------------------------------ multiple segments / cross-episode
def test_multiple_segments_are_summed_independently():
    idx = build_index({1: [("CAMERA", 0, 12), ("ANN", 0.5, 2), ("BOB", 10.5, 12),
                           ("CAMERA", 40, 55), ("ANN", 41, 42), ("BOB", 53.5, 55)]})
    o = resolve(idx, segs=[((1, 1), (1, 3)), ((1, 4), (1, 6))], hook=lid(1, 2), cliff=lid(1, 6))
    assert [s.duration_ms for s in o.clip.segments] == [12_000, 15_000]
    assert o.clip.total_duration_ms == 27_000 and o.accepted        # the 28 s gap between the segments is not counted
    assert o.clip.segments[1].start.time_ms == 40_000               # timestamps preserved, not rebased


def test_overlapping_segments_on_the_timeline_are_rejected():
    # the two ID ranges are disjoint but their time windows overlap (camera 10-30 vs action 20-40)
    idx = build_index({1: [("CAMERA", 10, 30), ("ANN", 10.5, 12), ("ACTION", 20, 40), ("BOB", 38.5, 40)]})
    o = resolve(idx, segs=[((1, 1), (1, 2)), ((1, 3), (1, 4))], hook=lid(1, 2), cliff=lid(1, 4))
    assert not o.accepted and not gate(o, "segments_distinct").passed


def test_cross_episode_timestamps_reset_and_are_measured_independently():
    idx = build_index({
        1: [("CAMERA", 50, 62), ("ANN", 50.5, 52), ("BOB", 60.5, 62)],     # late in episode 1
        2: [("CAMERA", 0, 15), ("CLEO", 0.5, 2), ("DAN", 13.5, 15)],       # starts again at 0 in episode 2
    })
    o = resolve(idx, segs=[((1, 1), (1, 3)), ((2, 1), (2, 3))], hook=lid(1, 2), cliff=lid(2, 3))
    assert o.accepted and o.clip.total_duration_ms == 27_000
    assert [(s.episode_id, s.start.time_ms, s.end.time_ms) for s in o.clip.segments] == [("E001", 50_000, 62_000), ("E002", 0, 15_000)]
    assert o.clip.episode_ids == ["E001", "E002"]
    assert o.clip.hook.episode_id == "E001" and o.clip.cliffhanger.inside_final_window


def test_cross_episode_hook_measured_only_against_first_segment():
    idx = build_index({1: [("CAMERA", 0, 12), ("ANN", 0.5, 2), ("BOB", 10.5, 12)],
                       2: [("CAMERA", 0, 15), ("CLEO", 0.5, 2), ("DAN", 13.5, 15)]})
    o = resolve(idx, segs=[((1, 1), (1, 3)), ((2, 1), (2, 3))], hook=lid(1, 2), cliff=lid(2, 3))
    assert o.clip.hook.offset_ms == 500          # not 12.5 s (which would be wrong if second-episode time were added)


# ------------------------------------------------------------------ cliffhanger references
def test_cliffhanger_far_before_the_end_is_trimmed_to_the_cliffhanger():
    idx = build_index({1: [("CAMERA", 0, 50), ("ANN", 0.5, 2), ("BOB", 28.5, 30, "question"), ("ACTION", 40, 50, "resolution")]})
    o = resolve(idx, segs=[((1, 1), (1, 4))], hook=lid(1, 2), cliff=lid(1, 3))
    s = o.clip.segments[0]
    assert s.end.time_ms == 30_000 and s.end.line_id == lid(1, 3)       # the cliffhanger line's own end
    assert o.clip.original_total_duration_ms == 50_000 and o.clip.total_duration_ms == 30_000
    assert o.clip.cliffhanger.tail_slack_ms == 0
    assert lid(1, 1) in s.partially_cut_line_ids                         # the 0-50 camera shot is cut, and reported
    assert lid(1, 4) in s.dropped_selected_line_ids and o.accepted


def test_tail_trim_ends_exactly_on_the_cliffhanger_when_a_block_ends_there():
    idx = build_index({1: [("ANN", 0.5, 2), ("BOB", 28.5, 30, "question"), ("ACTION", 40, 50, "resolution")]})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 1), cliff=lid(1, 2))
    s = o.clip.segments[0]
    assert (s.end.time_ms, s.end.adjusted, s.end.line_id) == (30_000, True, lid(1, 2))
    assert o.clip.original_total_duration_ms == 49_500 and o.clip.total_duration_ms == 29_500
    assert o.clip.cliffhanger.tail_slack_ms == 0 and lid(1, 3) in s.dropped_selected_line_ids
    assert gate(o, "cliffhanger_reference").passed


def test_default_cliffhanger_tail_limit_is_two_seconds():
    assert CLIFF_TAIL_MAX_MS == 2_000 == TimelineConfig().cliff_tail_max_ms
    assert TimelineConfig().to_dict() == {"min_clip_ms": 20_000, "max_clip_ms": 60_000, "hook_max_ms": 3_000,
                                          "hook_preferred_lead_ms": 2_000, "cliff_tail_max_ms": 2_000}


def test_cliffhanger_without_a_timestamp_is_rejected():
    idx = build_index({1: [("CAMERA", 0, 30), ("ANN", 0.5, 2), ("BOB", None, None, "no time")]})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert not o.accepted and "cliffhanger" in " ".join(o.rejection_reasons).lower()


def test_cliffhanger_that_ends_up_before_the_adjusted_start_is_rejected():
    # the "cliffhanger" line (listed last) actually sits at 1.0-2.0 s, long before the hook at 20 s
    idx = build_index({1: [("CAMERA", 0, 40), ("ANN", 20.0, 21.5, "hook"), ("BOB", 1.0, 2.0, "early")]})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert not o.accepted
    assert not gate(o, "cliffhanger_reference").passed
    assert "not inside the final clip window" in gate(o, "cliffhanger_reference").reason


# ------------------------------------------------------------------ misc invariants
def test_rejected_candidates_still_carry_full_detail():
    idx = build_index({1: one_seg_episode(19.5)})
    o = resolve(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert not o.accepted and o.clip is not None
    names = [g.gate for g in o.gates]
    assert names == ["timestamps_valid", "duration_20_60s", "hook_timing_3s", "cliffhanger_reference", "boundary_integrity", "segments_distinct"]
    assert all(g.reason for g in o.gates) and all(g.kind == "deterministic" for g in o.gates)


def test_llm_claims_are_ignored_by_the_resolver():
    idx = build_index({1: one_seg_episode(75.8)})
    c = stored(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 2), cliff=lid(1, 3))
    assert c.llm_gate_claims.duration_plausible_20_60s is True and c.gates_verified is False
    assert not TimelineResolver(idx).resolve(c).accepted


@pytest.mark.parametrize("seed", range(40))
def test_property_accepted_clips_always_satisfy_every_deterministic_rule(seed):
    """Random overlapping episodes: whatever the resolver accepts must be 20-60 s, hook <= 3 s, no cut dialogue,
    and every boundary must be a real block boundary."""
    rnd = random.Random(seed)
    T = rnd.uniform(25, 70)
    blocks = [("CAMERA", 0.0, T, "shot")]
    for _ in range(rnd.randint(6, 14)):
        s = round(rnd.uniform(0, T - 2), 1)
        e = round(min(T, s + rnd.uniform(0.5, 4.0)), 1)
        blocks.append((rnd.choice(["ANN", "BOB", "ACTION", "CAMERA", "SOUND FX"]), s, e, "x"))
    n = len(blocks)
    idx = build_index({1: blocks})
    lines = idx.episodes["E001"].lines
    dialogue = [l for l in lines if l.kind is LineKind.DIALOGUE]
    if not dialogue:
        return
    hook, cliff = rnd.choice(dialogue), rnd.choice(dialogue)
    try:
        c = stored(idx, segs=[((1, 1), (1, n))], hook=hook.line_id, cliff=cliff.line_id)
    except AssertionError:
        return
    o = TimelineResolver(idx).resolve(c)
    if not o.accepted:
        return
    clip = o.clip
    assert MIN_CLIP_MS <= clip.total_duration_ms <= MAX_CLIP_MS
    assert 0 <= clip.hook.offset_ms <= HOOK_MAX_MS
    starts = {l.start_ms for l in lines}
    ends = {l.end_ms for l in lines}
    for s in clip.segments:
        assert s.start.time_ms in starts and s.end.time_ms in ends
        assert not any(d.start_ms < t < d.end_ms for d in dialogue for t in (s.start.time_ms, s.end.time_ms))
        assert s.cut_dialogue_line_ids == []
    assert clip.total_duration_ms == sum(s.end.time_ms - s.start.time_ms for s in clip.segments)
