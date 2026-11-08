"""Trimmed clips: coherence gate, prompt note (only for trimmed clips), CLI option."""

from __future__ import annotations

import json

import pytest

from tl_helpers import build_index, lid, stored
from test_semantic_rank import A, make_ev, nums, pick
from test_timeline_tail import tail_episode
from promoai import validate
from promoai.config import Settings
from promoai.llm import FakeProvider
from promoai.ranking import Evaluated
from promoai.semantic import PromoAssessment, build_semantic_prompt, semantic_gates, trim_adjustment
from promoai.timeline import TimelineResolver


def trimmed_ev(tail_s: float, assessment: dict | None, cid="T"):
    idx = build_index({1: tail_episode(tail_s)})
    c = stored(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 1), cliff=lid(1, 2), cid=cid).model_copy(update={"candidate_id": cid})
    ev = Evaluated(cand=c, outcome=TimelineResolver(idx).resolve(c))
    if assessment:
        ev.assessment = PromoAssessment.model_validate(assessment)
        ev.semantic_gates = semantic_gates(ev.assessment, trimmed=trim_adjustment(ev.clip) is not None)
    return idx, ev


# ------------------------------------------------------------------ prompt
def test_only_trimmed_clips_get_the_trim_note():
    idx, trimmed = trimmed_ev(4.7, None, "T")
    _, untrimmed = trimmed_ev(1.5, None, "U")
    pt = build_semantic_prompt(idx, [(trimmed.cand, trimmed.clip)], None, 50)
    pu = build_semantic_prompt(idx, [(untrimmed.cand, untrimmed.clip)], None, 50)
    assert "NOTE: an editor trimmed 4.7s of footage from the END" in pt and "still plays as a complete" in pt
    assert "NOTE:" not in pu.split("<clip")[1]                       # untrimmed prompts are unchanged -> existing cache entries stay valid


# ------------------------------------------------------------------ gate
def test_trimmed_clip_gets_a_coherence_gate_and_untrimmed_does_not():
    a = PromoAssessment.model_validate(A("X", coherence="strong"))
    assert [g.gate for g in semantic_gates(a)] == ["hook_strength", "cliffhanger_strength", "standalone_clarity"]
    gates = semantic_gates(a, trimmed=True)
    assert [g.gate for g in gates][-1] == "coherence_after_trim" and gates[-1].passed and gates[-1].kind == "semantic_assessment"


@pytest.mark.parametrize("coh,ok", [("strong", True), ("adequate", True), ("weak", False)])
def test_coherence_after_trim_thresholds(coh, ok):
    g = semantic_gates(PromoAssessment.model_validate(A("X", coherence=coh)), trimmed=True)[-1]
    assert g.passed is ok and g.details["narrative_coherence"] == coh
    assert ("the trim broke the story" in g.reason) is (not ok) and "model's view:" in g.reason


def test_trimmed_clip_with_weak_coherence_is_rejected_and_the_reason_is_saved():
    idx, ev = trimmed_ev(4.7, A("T", coherence="weak"))
    sel = pick(idx, [ev])
    assert sel.ranked == []
    d = sel.decisions["T"]
    assert d.stage == "semantic" and d.status == "rejected" and d.reasons[0].startswith("coherence_after_trim:")
    assert "the trim broke the story" in d.reasons[0]


def test_trimmed_clip_with_coherent_story_is_ranked():
    idx, ev = trimmed_ev(4.7, A("T", coherence="adequate"))
    assert [e.cid for e in pick(idx, [ev]).ranked] == ["T"]


def test_untrimmed_clip_with_weak_coherence_is_not_rejected_for_it():
    # coherence only decides ranking order for untrimmed clips; it is not a mandatory gate there
    idx, ev = trimmed_ev(1.5, A("U", coherence="weak"), "U")
    assert trim_adjustment(ev.clip) is None
    assert [e.cid for e in pick(idx, [ev]).ranked] == ["U"]


# ------------------------------------------------------------------ CLI
@pytest.fixture
def series(tmp_path):
    idx = build_index({1: tail_episode(4.7), 2: tail_episode(1.9)})
    root = tmp_path / "series"
    (root / "episodes").mkdir(parents=True)
    for ep in idx.episodes.values():
        (root / "episodes" / f"{ep.episode_id}.json").write_text(ep.model_dump_json(), encoding="utf-8")
    run = root / "phase2" / "runs" / "r1"
    (run / "story_context_history").mkdir(parents=True)
    cands = []
    for ep, cid, win in ((1, "W01-C01", "W01"), (2, "W02-C01", "W02")):
        c = stored(idx, segs=[((ep, 1), (ep, 3))], hook=lid(ep, 1), cliff=lid(ep, 2), cid=cid)
        cands.append(c.model_copy(update={"candidate_id": cid, "window_id": win}).model_dump(mode="json"))
    (run / "candidates.json").write_text(json.dumps({"accepted": cands, "rejected": []}), encoding="utf-8")
    (run / "run_report.json").write_text(json.dumps({"prompt_version": "v2-refined"}), encoding="utf-8")
    return root


def run_cli(series, name, *extra):
    return validate.main(["--series-dir", str(series), "--phase2-run", "r1", "--out-name", name, *extra])


def test_cli_default_limit_is_two_seconds_and_results_are_recorded(series, monkeypatch):
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: pytest.fail("no LLM expected"))
    assert run_cli(series, "d") == 0
    out = series / "phase3" / "d"
    rep = json.loads((out / "report.json").read_text())
    assert rep["timeline_config"]["cliff_tail_max_ms"] == 2_000 and rep["deterministic"]["adjusted"] == 1
    rows = {r["candidate_id"]: r for r in json.loads((out / "timeline_results.json").read_text())["before_after"]}
    t = rows["W01-C01"]["after_python"]["adjustments_detail"][0]
    assert t["kind"] == "cliffhanger_trim" and t["from_ms"] == 34_700 and t["to_ms"] == 30_000
    assert rows["W01-C01"]["measured_as_proposed"]["total_duration_s"] == 34.7 and rows["W01-C01"]["after_python"]["total_duration_s"] == 30.0
    assert rows["W01-C01"]["original_candidate"]["segments"][0]["end_line_id"] == lid(1, 3)       # original proposal preserved
    assert rows["W02-C01"]["after_python"]["adjustments"] == []                                 # 1.9 s tail: untouched
    assert all(g["reason"] for r in rows.values() for g in r["gates"])


def test_cli_option_changes_the_limit(series):
    assert run_cli(series, "d5", "--cliff-tail-max-s", "5") == 0
    rep = json.loads((series / "phase3" / "d5" / "report.json").read_text())
    assert rep["timeline_config"]["cliff_tail_max_ms"] == 5_000 and rep["deterministic"]["adjusted"] == 0
    assert run_cli(series, "d1", "--cliff-tail-max-s", "1") == 0
    assert json.loads((series / "phase3" / "d1" / "report.json").read_text())["deterministic"]["adjusted"] == 2


def test_cli_rejects_a_negative_limit(series, capsys):
    assert run_cli(series, "neg", "--cliff-tail-max-s", "-1") == 2
    assert "must be >= 0" in capsys.readouterr().err
    assert not (series / "phase3" / "neg").exists()


def test_cli_semantic_mode_sends_the_trimmed_clip_with_its_note(series, monkeypatch):
    def reply(prompt):
        ids = [i for i in ("W01-C01", "W02-C01") if f'candidate_id="{i}"' in prompt]
        return json.dumps({"assessments": [A(i, coherence="weak" if i == "W01-C01" else "strong") for i in ids]})

    prov = FakeProvider(reply)
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: prov)
    monkeypatch.setattr(validate, "load_settings", lambda: Settings(model="fake", api_key="k"))
    assert run_cli(series, "s", "--semantic") == 0
    sent = "".join(prov.prompts)
    assert sent.count("NOTE: an editor trimmed") == 1               # only the trimmed clip carries the note
    out = series / "phase3" / "s"
    ranked = [p["candidate_id"] for p in json.loads((out / "ranked_promos.json").read_text())["promos"]]
    assert ranked == ["W02-C01"]
    rej = {r["candidate_id"]: r for r in json.loads((out / "rejected_candidates.json").read_text())["candidates"]}
    assert "coherence_after_trim" in rej["W01-C01"]["decision_reasons"][0]
    assert rej["W01-C01"]["adjustments"][0]["kind"] == "cliffhanger_trim"     # adjustment kept on the rejected record
