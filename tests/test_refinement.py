"""Phase 2 refinement: prompt requirements, structural checks, versioned outputs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import cand, llm_reply, make_index
from promoai.detect import main, run_detection, validate_candidate, write_outputs
from promoai.llm import FakeProvider
from promoai.prompts import OUTPUT_SKELETON, PROMPT_VERSION, build_prompt
from promoai.schemas import PromoCandidate, RejectedCandidate, StoredCandidate


@pytest.fixture(scope="module")
def prompt() -> str:
    return build_prompt(None, [make_index(1).episodes["E001"]], max_candidates=3)


# ------------------------------------------------------------------ prompt requirements
def test_length_preference_and_hard_limit(prompt):
    assert "between 20 and 60 seconds" in prompt
    assert "60 is a hard limit: never exceed it" in prompt
    assert "AIM for 30-45 seconds" in prompt


def test_shorten_instruction_preserves_coherence(prompt):
    assert "do NOT include all of it" in prompt
    assert "tightest sub-moment that still makes sense on its own" in prompt
    assert "cutting off mid-thought" in prompt


def test_length_is_measured_as_a_span_not_a_sum(prompt):
    assert "latest end - earliest start" in prompt
    assert "NEVER add up individual line lengths" in prompt
    assert "[4.0-7.0] and [5.0-9.0]" in prompt and "5.0 s" in prompt      # worked overlap example
    assert "add the segment lengths" in prompt                             # multi-segment rule
    assert "NOT a timeline" in prompt and "ARE the timeline" in prompt


def test_hook_must_be_within_3s_of_the_actual_clip_start(prompt):
    assert "HOOK WITHIN 3 SECONDS" in prompt
    assert "earliest start time among the lines of its FIRST segment" in prompt
    assert "no later than 3.0 seconds after that moment" in prompt
    assert "hook's own [start]" in prompt


def test_prompt_does_not_force_the_hook_to_be_the_first_listed_line(prompt):
    assert "CAMERA and ACTION lines may be listed before the dialogue" in prompt
    low = prompt.lower()
    for forbidden in ("hook_line_id must be the first", "hook must be the first line", "first line id", "first source"):
        assert forbidden not in low


def test_cliffhanger_must_name_a_specific_open_question(prompt):
    assert "REAL CLIFFHANGER" in prompt
    assert '"open_question"' in prompt
    assert "is NOT automatically a cliffhanger" in prompt
    assert "a scene ending" in prompt.lower() and "a character leaving" in prompt
    assert "cut the clip earlier, before the answer" in prompt


def test_gates_are_presented_as_unverified(prompt):
    assert "UNVERIFIED claims" in prompt and "Answer false whenever you are unsure" in prompt


def test_output_shape_requests_open_question_and_still_forbids_times(prompt):
    assert '"open_question"' in OUTPUT_SKELETON
    assert "NEVER output timestamps" in prompt
    assert "Segments must not overlap" in prompt


def test_voice_labels_are_not_aliases(prompt):
    assert "Off-Screen Voice" in prompt and "never treat a voice label" in prompt


def test_prompt_version_is_recorded_and_new():
    assert PROMPT_VERSION == "v2-refined"


def test_new_prompt_has_a_different_cache_key_than_old_text(tmp_path):
    """Old cached replies stay on disk but can never be served for the new prompt."""
    from promoai.llm import CachedProvider

    c = CachedProvider(FakeProvider(["x"]), tmp_path)
    assert c._key("old prompt text") != c._key(build_prompt(None, [make_index(1).episodes["E001"]]))


# ------------------------------------------------------------------ schema: open_question
def test_open_question_is_required_by_schema():
    c = cand()
    del c["open_question"]
    with pytest.raises(ValidationError):
        PromoCandidate.model_validate(c)


def test_missing_open_question_triggers_the_json_retry():
    bad = json.loads(llm_reply([cand()]))
    del bad["candidates"][0]["open_question"]
    prov = FakeProvider([json.dumps(bad), llm_reply([cand()])])
    run = run_detection(make_index(2), prov, ["E001", "E002"], window_size=2, overlap=0)
    assert run.windows[0].attempts == 2 and len(run.accepted) == 1
    assert "open_question" in prov.prompts[1]                  # the repair prompt names the missing field


@pytest.mark.parametrize("q", ["", "Why?", "Unclear."])
def test_vague_open_question_rejects_only_that_candidate(q):
    run = run_detection(make_index(2), FakeProvider([llm_reply([cand("C1", open_question=q), cand("C2")])]),
                        ["E001", "E002"], window_size=2, overlap=0)
    assert [c.llm_candidate_id for c in run.accepted] == ["C2"]
    assert any("open_question" in r for r in run.rejected[0].reasons)


def test_open_question_is_stored():
    idx = make_index(2)
    r = validate_candidate(PromoCandidate.model_validate(cand()), 1, "W01", {"E001", "E002"}, idx)
    assert r.open_question == "Will Ann find out who sent the letter?"


# ------------------------------------------------------------------ structural checks
def check(c, window=("E001", "E002")):
    return validate_candidate(PromoCandidate.model_validate(c), 1, "W01", set(window), make_index(4))


def test_hook_may_be_any_line_of_the_first_segment_not_just_the_first():
    r = check(cand(segs=(("E001-S01-L001", "E001-S01-L004"),), hook="E001-S01-L003"))   # dialogue after CAMERA/ACTION
    assert isinstance(r, StoredCandidate) and r.warnings == []


def test_hook_must_belong_to_first_segment_in_multi_segment_clip():
    segs = (("E001-S01-L001", "E001-S01-L003"), ("E002-S01-L001", "E002-S01-L003"))
    ok = check(cand(segs=segs, hook="E001-S01-L002", cliff="E002-S01-L003"))
    assert isinstance(ok, StoredCandidate)
    bad = check(cand(segs=segs, hook="E002-S01-L002", cliff="E002-S01-L003"))           # hook in the 2nd segment
    assert isinstance(bad, RejectedCandidate) and any("first segment" in x for x in bad.reasons)


def test_cliffhanger_must_belong_to_last_segment_in_multi_segment_clip():
    segs = (("E001-S01-L001", "E001-S01-L003"), ("E002-S01-L001", "E002-S01-L003"))
    bad = check(cand(segs=segs, hook="E001-S01-L001", cliff="E001-S01-L003"))           # cliffhanger in the 1st segment
    assert isinstance(bad, RejectedCandidate) and any("last segment" in x for x in bad.reasons)


def test_hook_inside_the_right_episode_but_outside_the_range_is_rejected():
    bad = check(cand(segs=(("E001-S01-L001", "E001-S01-L003"),), hook="E001-S01-L005", cliff="E001-S01-L003"))
    assert isinstance(bad, RejectedCandidate)


def test_existing_id_checks_still_apply():
    assert isinstance(check(cand(segs=(("E001-S01-L001", "E001-S01-L099"),))), RejectedCandidate)         # unknown id
    assert isinstance(check(cand(segs=(("E003-S01-L001", "E003-S01-L002"),))), RejectedCandidate)         # outside window
    assert isinstance(check(cand(segs=(("E001-S01-L005", "E001-S01-L002"),), hook="E001-S01-L005", cliff="E001-S01-L002")), RejectedCandidate)
    assert isinstance(check(cand(segs=(("E001-S01-L005", "E002-S01-L002"),), hook="E001-S01-L005", cliff="E002-S01-L002")), RejectedCandidate)


def test_same_line_for_hook_and_cliffhanger_is_flagged_not_rejected():
    r = check(cand(segs=(("E001-S01-L003", "E001-S01-L003"),), hook="E001-S01-L003", cliff="E001-S01-L003"))
    assert isinstance(r, StoredCandidate) and "hook and cliffhanger are the same line" in r.warnings


def test_hook_listed_after_cliffhanger_is_only_a_warning():
    r = check(cand(segs=(("E001-S01-L001", "E001-S01-L005"),), hook="E001-S01-L004", cliff="E001-S01-L002"))
    assert isinstance(r, StoredCandidate) and any("check on the timeline" in w for w in r.warnings)


def test_candidates_never_gain_a_verified_flag_in_phase2():
    run = run_detection(make_index(2), FakeProvider([llm_reply([cand()])]), ["E001", "E002"], window_size=2)
    assert run.accepted and all(c.gates_verified is False and c.status == "llm_proposed" for c in run.accepted)


# ------------------------------------------------------------------ versioned outputs keep old files
def test_run_folder_is_separate_and_never_overwritten(tmp_path):
    legacy = tmp_path / "phase2"
    legacy.mkdir()
    (legacy / "candidates.json").write_text('{"v1": true}')
    run = run_detection(make_index(2), FakeProvider([llm_reply([cand()])]), ["E001", "E002"], window_size=2)

    out = write_outputs(run, None, tmp_path, {"prompt_version": PROMPT_VERSION}, run_name="r1")
    assert out == tmp_path / "phase2" / "runs" / "r1"
    assert (out / "candidates.json").is_file()
    assert (legacy / "candidates.json").read_text() == '{"v1": true}'            # v1 file untouched
    assert json.loads((out / "run_report.json").read_text())["prompt_version"] == "v2-refined"
    with pytest.raises(FileExistsError):
        write_outputs(run, None, tmp_path, {}, run_name="r1")


def test_cli_checks_run_name_before_spending_api_calls(tmp_path, monkeypatch):
    series = tmp_path / "series"
    (series / "episodes").mkdir(parents=True)
    idx = make_index(2)
    for ep in idx.episodes.values():
        (series / "episodes" / f"{ep.episode_id}.json").write_text(ep.model_dump_json(), encoding="utf-8")
    legacy = series / "phase2" / "candidates.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("old results")

    from promoai import detect
    from promoai.config import Settings

    made = []
    def fake_create(settings, cache=True):
        p = FakeProvider([llm_reply([cand()])])
        made.append(p)
        return p
    monkeypatch.setattr(detect, "create_provider", fake_create)
    monkeypatch.setattr(detect, "load_settings", lambda: Settings(model="m", api_key="k"))

    args = ["--series-dir", str(series), "--episodes", "1-2", "--window-size", "2", "--run-name", "r1"]
    assert main(args) == 0
    assert (series / "phase2" / "runs" / "r1" / "candidates.json").is_file()
    assert legacy.read_text() == "old results"

    calls_before = sum(len(p.prompts) for p in made)
    assert main(args) == 2                                                       # same name again: refused...
    assert sum(len(p.prompts) for p in made) == calls_before                    # ...before any API call
