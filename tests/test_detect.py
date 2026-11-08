"""Window planning, ID validation and the orchestration loop (fake provider, no network)."""

from __future__ import annotations

import json
import logging
import re

import pytest

from conftest import cand, llm_reply, make_index
from promoai.detect import (
    parse_episode_selection, plan_windows, run_detection, validate_candidate, write_outputs,
)
from promoai.llm import CachedProvider, FakeProvider, LLMBlockedError, LLMQuotaError
from promoai.schemas import PromoCandidate, RejectedCandidate, StoredCandidate

EP_IDS = ["E001", "E002", "E003", "E004"]


# ------------------------------------------------------------------ windows
@pytest.mark.parametrize("n,size,ov,expected", [
    (5, 3, 1, [[1, 2, 3], [3, 4, 5]]),
    (4, 2, 1, [[1, 2], [2, 3], [3, 4]]),
    (4, 6, 1, [[1, 2, 3, 4]]),
    (1, 3, 1, [[1]]),
    (6, 3, 0, [[1, 2, 3], [4, 5, 6]]),
    (7, 3, 0, [[1, 2, 3], [4, 5, 6], [7]]),
    (0, 3, 1, []),
])
def test_plan_windows(n, size, ov, expected):
    assert plan_windows(list(range(1, n + 1)), size, ov) == expected


def test_plan_windows_50_episodes_like_the_real_series():
    w = plan_windows(list(range(1, 51)), 6, 1)
    assert len(w) == 10 and w[0][0] == 1 and w[-1][-1] == 50
    pairs = {(a, b) for win in w for a, b in zip(win, win[1:])}
    assert all((i, i + 1) in pairs for i in range(1, 50))      # every neighbouring pair shares a window


@pytest.mark.parametrize("size,ov", [(0, 0), (3, 3), (3, -1)])
def test_plan_windows_rejects_bad_args(size, ov):
    with pytest.raises(ValueError):
        plan_windows([1, 2, 3], size, ov)


def test_episode_selection():
    avail = list(range(1, 11))
    assert parse_episode_selection("all", avail) == avail
    assert parse_episode_selection("1-3", avail) == [1, 2, 3]
    assert parse_episode_selection("2,5-6,99", avail) == [2, 5, 6]


# ------------------------------------------------------------------ ID validation
def check(c_dict, window=("E001", "E002"), idx=None):
    idx = idx or make_index(4)
    return validate_candidate(PromoCandidate.model_validate(c_dict), 1, "W01", set(window), idx)


def test_valid_single_segment_candidate():
    r = check(cand())
    assert isinstance(r, StoredCandidate)
    assert r.candidate_id == "W01-C01" and r.llm_candidate_id == "C1" and r.status == "llm_proposed"
    s = r.segments[0]
    assert (s.episode_id, s.start_scene_id, s.end_scene_id, s.n_lines) == ("E001", "E001-S01", "E001-S01", 4)
    assert r.cross_episode is False


def test_segment_spanning_two_scenes_resolves_both_scene_ids():
    r = check(cand(segs=(("E001-S01-L004", "E001-S02-L006"),)))
    assert isinstance(r, StoredCandidate)
    assert (r.segments[0].start_scene_id, r.segments[0].end_scene_id) == ("E001-S01", "E001-S02")


def test_multiple_joined_segments_and_cross_episode():
    r = check(cand(segs=(("E001-S01-L003", "E001-S01-L005"), ("E002-S01-L001", "E002-S01-L002")),
                   hook="E001-S01-L003", cliff="E002-S01-L002"))
    assert isinstance(r, StoredCandidate) and r.cross_episode and len(r.segments) == 2
    assert r.segments[0].episode_id == "E001" and r.segments[1].episode_id == "E002"   # order preserved


def test_unknown_line_id_is_rejected_with_reason():
    r = check(cand(segs=(("E001-S01-L001", "E001-S01-L099"),)))
    assert isinstance(r, RejectedCandidate) and any("E001-S01-L099 does not exist" in x for x in r.reasons)


def test_unknown_episode_is_rejected():
    r = check(cand(segs=(("E077-S01-L001", "E077-S01-L002"),)))
    assert isinstance(r, RejectedCandidate)


def test_real_id_outside_the_window_is_rejected():
    r = check(cand(segs=(("E003-S01-L001", "E003-S01-L002"),)))
    assert isinstance(r, RejectedCandidate) and "outside this window" in r.reasons[0]


def test_start_after_end_is_rejected():
    r = check(cand(segs=(("E001-S01-L005", "E001-S01-L002"),), hook="E001-S01-L005", cliff="E001-S01-L002"))
    assert isinstance(r, RejectedCandidate) and "comes after" in r.reasons[0]


def test_segment_must_not_span_episodes():
    r = check(cand(segs=(("E001-S01-L005", "E002-S01-L002"),), hook="E001-S01-L005", cliff="E002-S01-L002"))
    assert isinstance(r, RejectedCandidate) and "spans two episodes" in r.reasons[0]


def test_hook_must_be_in_first_segment_and_cliffhanger_in_last():
    r = check(cand(hook="E001-S02-L006"))
    assert isinstance(r, RejectedCandidate) and any("hook_line_id" in x and "first segment" in x for x in r.reasons)
    r = check(cand(cliff="E001-S02-L006"))
    assert isinstance(r, RejectedCandidate) and any("cliffhanger_line_id" in x and "last segment" in x for x in r.reasons)
    r = check(cand(hook="E002-S01-L001"))
    assert isinstance(r, RejectedCandidate)


def test_hook_or_cliffhanger_that_does_not_exist_is_rejected():
    assert isinstance(check(cand(hook="E001-S01-L099")), RejectedCandidate)


def test_overlapping_segments_are_rejected():
    r = check(cand(segs=(("E001-S01-L001", "E001-S01-L004"), ("E001-S01-L003", "E001-S01-L005")),
                   hook="E001-S01-L001", cliff="E001-S01-L005"))
    assert isinstance(r, RejectedCandidate) and any("overlap" in x for x in r.reasons)


def test_out_of_story_order_segments_warn_but_are_kept():
    r = check(cand(segs=(("E002-S01-L001", "E002-S01-L002"), ("E001-S01-L001", "E001-S01-L002")),
                   hook="E002-S01-L001", cliff="E001-S01-L002"))
    assert isinstance(r, StoredCandidate) and r.warnings


def test_stored_candidate_contains_no_timestamp_keys():
    r = check(cand(start_time="00:03.0", end_seconds=40))
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    ks = set(keys(r.model_dump(mode="json")))
    assert not ({"start_time", "end_seconds", "start_ms", "end_ms", "duration"} & ks)
    assert r.dropped_fields == ["end_seconds", "start_time"]


def test_gates_are_stored_as_unverified_claims():
    r = check(cand())
    assert r.llm_gate_claims.hook_in_first_3s is True          # the model's claim, kept as-is
    assert r.status == "llm_proposed" and r.gates_verified is False
    dumped = r.model_dump(mode="json")
    assert "llm_gate_claims" in dumped and "llm_gate_assessment" not in dumped


# ------------------------------------------------------------------ orchestration
def two_window_provider():
    """Window 1 (E001-E002): Ann+Bob, one good + one bad candidate.  Window 2 (E002-E003) omits Bob."""
    w1 = llm_reply([cand("C1"), cand("C2", segs=(("E001-S01-L001", "E001-S01-L099"),))],
                   characters=(("Ann", ()), ("Bob", ())), revelations=("r1",))
    w2 = llm_reply([cand("C1", segs=(("E003-S01-L001", "E003-S01-L003"),)),
                    cand("C2")],                                       # E001 range: outside window 2
                   characters=(("Ann", ()), ("Cleo", ())), revelations=("r2",))
    return FakeProvider([w1, w2])


def test_run_two_windows_end_to_end():
    idx, prov = make_index(3), two_window_provider()
    run = run_detection(idx, prov, ["E001", "E002", "E003"], window_size=2, overlap=1)
    assert [w.status for w in run.windows] == ["ok", "ok"]
    assert [c.candidate_id for c in run.accepted] == ["W01-C01", "W02-C01"]
    assert len(run.rejected) == 2
    assert run.windows[0].candidates_returned == 2 and run.windows[0].candidates_accepted == 1
    assert run.windows[0].api_calls == 1 and run.windows[0].prompt_tokens > 0


def test_story_context_is_updated_and_reused_in_the_next_prompt():
    idx, prov = make_index(3), two_window_provider()
    run = run_detection(idx, prov, ["E001", "E002", "E003"], window_size=2, overlap=1)
    assert "start of the series" in prov.prompts[0]
    assert '"Bob"' in prov.prompts[1] and '"Ann"' in prov.prompts[1] and "r1" in prov.prompts[1]
    final = run.final_context
    assert {c.name for c in final.characters} == {"Ann", "Bob", "Cleo"}     # Bob kept though window 2 omitted him
    assert final.revelations == ["r1", "r2"] and final.through_episode_id == "E003"
    assert run.contexts["W01"].through_episode_id == "E002"
    assert any("kept" in n for n in run.windows[1].context_notes)


def test_prompts_contain_only_window_episodes_with_valid_ids():
    idx, prov = make_index(4), FakeProvider([llm_reply()])
    run_detection(idx, prov, EP_IDS, window_size=2, overlap=1)
    assert len(prov.prompts) == 3
    for prompt, win in zip(prov.prompts, [["E001", "E002"], ["E002", "E003"], ["E003", "E004"]]):
        ids = re.findall(r"^(E\d{3}-S\d{2}-L\d{3}) \[", prompt, re.M)
        assert ids and all(i in idx.lines for i in ids)
        assert {i[:4] for i in ids} == set(win)


def test_duplicate_clip_from_overlapping_window_is_dropped():
    same = cand("C1", segs=(("E002-S01-L001", "E002-S01-L004"),))
    prov = FakeProvider([llm_reply([same]), llm_reply([same])])
    run = run_detection(make_index(3), prov, ["E001", "E002", "E003"], window_size=2, overlap=1)
    assert len(run.accepted) == 1 and run.accepted[0].window_id == "W01"
    assert "exact duplicate of W01-C01" in run.rejected[0].reasons[0]


def test_invalid_json_triggers_one_retry_inside_a_window():
    prov = FakeProvider(["not json at all", llm_reply([cand()])])
    run = run_detection(make_index(2), prov, ["E001", "E002"], window_size=2, overlap=0)
    w = run.windows[0]
    assert w.status == "ok" and w.attempts == 2 and w.api_calls == 2
    assert prov.stats.json_retries == 1 and len(run.accepted) == 1


def test_window_fails_cleanly_after_two_invalid_replies_and_run_continues():
    prov = FakeProvider(["bad", "worse", llm_reply([cand("C1", segs=(("E003-S01-L001", "E003-S01-L002"),))])])
    run = run_detection(make_index(4), prov, EP_IDS, window_size=2, overlap=0)
    assert [w.status for w in run.windows] == ["failed", "ok"]
    assert run.windows[0].error == "LLMResponseError" and len(run.accepted) == 1
    assert run.stopped_early is None


def test_two_consecutive_failures_stop_the_run():
    prov = FakeProvider(["bad"])
    run = run_detection(make_index(6), prov, [f"E00{i}" for i in range(1, 7)], window_size=2, overlap=0)
    assert [w.status for w in run.windows] == ["failed", "failed", "skipped"]
    assert "consecutive" in run.stopped_early


def test_daily_quota_stops_run_but_keeps_partial_results():
    def script(prompt, _n=[0]):
        _n[0] += 1
        if _n[0] == 2:
            raise LLMQuotaError("daily quota")
        return llm_reply([cand()])
    run = run_detection(make_index(6), FakeProvider(script), [f"E00{i}" for i in range(1, 7)], window_size=2, overlap=0)
    assert [w.status for w in run.windows] == ["ok", "failed", "skipped"]
    assert run.windows[1].error == "quota_exhausted" and run.stopped_early.startswith("stopped at W02")
    assert len(run.accepted) == 1 and run.final_context is not None


def test_blocked_window_is_recorded_not_fatal():
    def script(prompt, _n=[0]):
        _n[0] += 1
        if _n[0] == 1:
            raise LLMBlockedError("safety")
        return llm_reply()
    run = run_detection(make_index(4), FakeProvider(script), EP_IDS, window_size=2, overlap=0)
    assert [w.status for w in run.windows] == ["failed", "ok"] and run.windows[0].error == "LLMBlockedError"


def test_no_candidates_is_a_valid_outcome_not_fabricated():
    run = run_detection(make_index(2), FakeProvider([llm_reply([])]), ["E001", "E002"], window_size=2)
    assert run.accepted == [] and run.rejected == [] and run.windows[0].status == "ok"


def test_caching_makes_a_rerun_free(tmp_path):
    idx = make_index(3)
    first = CachedProvider(two_window_provider(), tmp_path)
    r1 = run_detection(idx, first, ["E001", "E002", "E003"], window_size=2, overlap=1)
    assert first.stats.api_calls == 2 and first.stats.cache_hits == 0

    inner2 = FakeProvider(["MUST NOT BE CALLED"])
    second = CachedProvider(inner2, tmp_path)
    r2 = run_detection(idx, second, ["E001", "E002", "E003"], window_size=2, overlap=1)
    assert inner2.prompts == [] and second.stats.api_calls == 0 and second.stats.cache_hits == 2
    assert [c.model_dump() for c in r2.accepted] == [c.model_dump() for c in r1.accepted]
    assert r2.final_context == r1.final_context
    assert second.stats.tokens_saved_by_cache > 0


def test_outputs_written_and_contain_no_screenplay_text_or_secrets(tmp_path):
    idx, prov = make_index(3), two_window_provider()
    run = run_detection(idx, prov, ["E001", "E002", "E003"], window_size=2, overlap=1)
    out = write_outputs(run, prov, tmp_path, {"provider": "fake", "model": "fake-1"})
    names = sorted(p.name for p in out.rglob("*.json"))
    assert {"candidates.json", "run_report.json", "story_context.json", "W01.json", "W02.json"} <= set(names)
    report = json.loads((out / "run_report.json").read_text())
    assert report["totals"]["candidates_accepted"] == 2 and report["usage"]["api_calls"] == 2
    blob = (out / "run_report.json").read_text()
    for secret in ("Hello there.", "Wide shot of the hall", "GEMINI_API_KEY"):
        assert secret not in blob
    cands = json.loads((out / "candidates.json").read_text())
    assert len(cands["accepted"]) == 2 and len(cands["rejected"]) == 2


def test_logs_never_contain_screenplay_text(caplog):
    caplog.set_level(logging.DEBUG)
    run_detection(make_index(3), two_window_provider(), ["E001", "E002", "E003"], window_size=2, overlap=1)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Hello there" not in text and "Wide shot" not in text and "hook" not in text.lower().replace("hook_", "")
