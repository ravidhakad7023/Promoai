"""Deterministic Phase 3 checks on the real Phase 2 v2 sample (offline, no API).

Skipped when the confidential outputs are absent."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from promoai.index import ScreenplayIndex
from promoai.models import LineKind
from promoai.timeline import HOOK_MAX_MS, MAX_CLIP_MS, MIN_CLIP_MS
from promoai.validate import load_phase2_run, run_deterministic

ROOT = Path(__file__).resolve().parent.parent / "data" / "output"
SERIES = [Path(p) for p in glob.glob(str(ROOT / "*" / "phase2" / "runs" / "v2-sample-e001-e005"))]
if not SERIES:
    pytest.skip("Phase 2 v2 sample not present", allow_module_level=True)

pytestmark = pytest.mark.real_data
SERIES_DIR = SERIES[0].parents[2]


@pytest.fixture(scope="module")
def data():
    index = ScreenplayIndex.from_output_dir(SERIES_DIR)
    cands, _, _ = load_phase2_run(SERIES_DIR, "v2-sample-e001-e005")
    return index, cands, run_deterministic(index, cands)


def test_every_candidate_is_resolved_and_ids_map_to_real_timestamps(data):
    index, cands, outs = data
    assert len(outs) == len(cands) == 4
    for o in outs:
        assert o.clip is not None
        for s in o.clip.segments:
            st, en = index.lines[s.start.line_id], index.lines[s.end.line_id]
            assert st.start_ms == s.start.time_ms and en.end_ms == s.end.time_ms
            assert st.episode_id == s.episode_id == en.episode_id
            assert (st.source_file, st.source_line) == (s.start.source_file, s.start.source_line)


def test_total_is_the_sum_of_independent_segment_windows(data):
    for o in data[2]:
        assert o.clip.total_duration_ms == sum(s.end.time_ms - s.start.time_ms for s in o.clip.segments)


def test_accepted_clips_satisfy_the_hard_rules_and_never_cut_dialogue(data):
    index, _, outs = data
    for o in outs:
        if not o.accepted:
            continue
        assert MIN_CLIP_MS <= o.clip.total_duration_ms <= MAX_CLIP_MS
        assert 0 <= o.clip.hook.offset_ms <= HOOK_MAX_MS
        for s in o.clip.segments:
            for d in (l for l in index.episodes[s.episode_id].lines if l.kind is LineKind.DIALOGUE):
                for t in (s.start.time_ms, s.end.time_ms):
                    assert not (d.start_ms < t < d.end_ms)


def test_decision_equals_conjunction_of_gates_and_reasons_are_kept(data):
    for o in data[2]:
        assert o.accepted == all(g.passed for g in o.gates)
        assert all(g.reason for g in o.gates)
        assert len(o.rejection_reasons) == sum(not g.passed for g in o.gates)


def test_llm_claims_were_not_trusted(data):
    _, cands, outs = data
    # the model claimed every duration and hook gate was met; Python rejects the 19.5 s one and moves two hooks
    assert all(c.llm_gate_claims.duration_plausible_20_60s and c.llm_gate_claims.hook_in_first_3s for c in cands)
    rejected = [o for o in outs if not o.accepted]
    assert len(rejected) >= 1 and all(any("duration" in r for r in o.rejection_reasons) for o in rejected)
    assert any(o.clip.hook.offset_before_ms and o.clip.hook.offset_before_ms > HOOK_MAX_MS for o in outs)
    assert all(o.clip.hook.offset_ms <= HOOK_MAX_MS for o in outs)       # ...and every hook was brought inside 3 s
