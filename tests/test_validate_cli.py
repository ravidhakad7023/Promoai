"""Phase 3 CLI end to end on a synthetic series, with a fake LLM."""

from __future__ import annotations

import json

import pytest

from tl_helpers import build_index, lid, stored
from promoai import validate
from promoai.config import Settings
from promoai.llm import FakeProvider
from promoai.schemas import CharacterInfo, StoryContext
from test_semantic_rank import A, clip_blocks


@pytest.fixture
def series(tmp_path):
    """Episode 1 holds three candidates: good (30 s), too short (19.5 s) and an exact duplicate of the good one.
    Episode 2 holds another good one."""
    blocks = clip_blocks(1) + [("CAMERA", 200, 219.5, "short"), ("ANN", 200.5, 201.5, "hook"), ("BOB", 218.0, 219.5, "end")]
    idx = build_index({1: blocks, 2: clip_blocks(1)})
    root = tmp_path / "series"
    (root / "episodes").mkdir(parents=True)
    for ep in idx.episodes.values():
        (root / "episodes" / f"{ep.episode_id}.json").write_text(ep.model_dump_json(), encoding="utf-8")
    run = root / "phase2" / "runs" / "r1"
    (run / "story_context_history").mkdir(parents=True)

    def c(cid, ep, first, window="W01"):
        x = stored(idx, segs=[((ep, first), (ep, first + 2))], hook=lid(ep, first + 1), cliff=lid(ep, first + 2), cid=cid)
        return x.model_copy(update={"candidate_id": cid, "window_id": window}).model_dump(mode="json")

    cands = [c("W01-C01", 1, 1), c("W01-C02", 1, 4), c("W01-C03", 1, 1), c("W02-C01", 2, 1, "W02")]
    (run / "candidates.json").write_text(json.dumps({"accepted": cands, "rejected": []}), encoding="utf-8")
    ctx = StoryContext(characters=[CharacterInfo(name="Ann", aliases=["Off-Screen Voice"], role="lead")], through_episode_id="E002")
    for w in ("W01", "W02"):
        (run / "story_context_history" / f"{w}.json").write_text(ctx.model_dump_json(), encoding="utf-8")
    (run / "run_report.json").write_text(json.dumps({"prompt_version": "v2-refined"}), encoding="utf-8")
    return root


def args(series, name, *extra):
    return ["--series-dir", str(series), "--phase2-run", "r1", "--out-name", name, *extra]


def test_deterministic_mode_makes_no_llm_calls_and_writes_results(series, monkeypatch, capsys):
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: pytest.fail("LLM must not be created without --semantic"))
    assert validate.main(args(series, "d1")) == 0
    out = series / "phase3" / "d1"
    assert {p.name for p in out.glob("*.json")} == {"timeline_results.json", "rejected_timeline.json", "report.json"}
    rep = json.loads((out / "report.json").read_text())
    assert rep["mode"] == "deterministic" and rep["deterministic"] == {"candidates": 4, "pass": 3, "rejected": 1, "adjusted": 0}
    rej = json.loads((out / "rejected_timeline.json").read_text())
    assert [r["candidate_id"] for r in rej] == ["W01-C02"] and "19.5s" in rej[0]["reasons"][0]
    shown = capsys.readouterr().out
    assert "LLM claimed" in shown and "measured" in shown and "REJECTED" in shown


def test_output_folder_is_never_overwritten(series):
    assert validate.main(args(series, "d1")) == 0
    assert validate.main(args(series, "d1")) == 2


def test_missing_phase2_run_is_a_clean_error(series, capsys):
    assert validate.main(["--series-dir", str(series), "--phase2-run", "nope"]) == 2
    assert "Phase 2 run not found" in capsys.readouterr().err


def semantic_run(series, monkeypatch, replies, name="s1", *extra):
    prov = FakeProvider(replies)
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: prov)
    monkeypatch.setattr(validate, "load_settings", lambda: Settings(model="fake", api_key="k"))
    code = validate.main(args(series, name, "--semantic", *extra))
    return code, prov, series / "phase3" / name


def test_semantic_mode_only_sends_clips_that_passed_the_deterministic_gates(series, monkeypatch):
    def reply(prompt):
        ids = [i for i in ("W01-C01", "W01-C02", "W01-C03", "W02-C01") if f'candidate_id="{i}"' in prompt]
        return json.dumps({"assessments": [A(i, pull="high" if i == "W02-C01" else "medium") for i in ids]})

    code, prov, out = semantic_run(series, monkeypatch, reply)
    assert code == 0
    sent = "".join(prov.prompts)
    assert 'candidate_id="W01-C01"' in sent and 'candidate_id="W02-C01"' in sent
    assert 'candidate_id="W01-C02"' not in sent          # 19.5 s: failed Python's gate, so no API spend on it
    assert 'candidate_id="W01-C03"' not in sent          # exact duplicate of W01-C01: dropped before the LLM
    assert len(prov.prompts) == 2                        # one call per window


def test_semantic_outputs_ranked_rejected_and_reasons(series, monkeypatch):
    def reply(prompt):
        ids = [i for i in ("W01-C01", "W02-C01") if f'candidate_id="{i}"' in prompt]
        return json.dumps({"assessments": [A(i, pull="high" if i == "W02-C01" else "medium") for i in ids]})

    _, _, out = semantic_run(series, monkeypatch, reply)
    ranked = json.loads((out / "ranked_promos.json").read_text())
    assert [p["candidate_id"] for p in ranked["promos"]] == ["W02-C01", "W01-C01"] and [p["rank"] for p in ranked["promos"]] == [1, 2]
    first = ranked["promos"][0]
    assert first["status"] == "recommended" and first["one_line_justification"]
    assert {g["kind"] for g in first["gates"]} == {"deterministic", "semantic_assessment"}
    assert all(g["reason"] for g in first["gates"])
    assert "opinions" in ranked["disclaimer"] and first["llm_original_claims"]["hook_in_first_3s"] is True
    assert first["clip"]["segments"][0]["start"]["source_line"] > 0                       # source reference kept

    rej = {r["candidate_id"]: r for r in json.loads((out / "rejected_candidates.json").read_text())["candidates"]}
    assert rej["W01-C02"]["decision_stage"] == "deterministic" and "duration_20_60s" in rej["W01-C02"]["decision_reasons"][0]
    assert rej["W01-C03"]["decision_stage"] == "duplicate" and "exact duplicate of W01-C01" in rej["W01-C03"]["decision_reasons"][0]

    sem = json.loads((out / "semantic_assessments.json").read_text())
    assert set(sem["assessments"]) == {"W01-C01", "W02-C01"} and sem["prompt_version"]
    assert any("Off-Screen Voice" in n for notes in sem["context_notes"].values() for n in notes)   # alias fix applied on load
    rep = json.loads((out / "report.json").read_text())
    assert rep["semantic"]["recommended"] == 2 and rep["semantic"]["usage"]["api_calls"] == 2


def test_semantic_failures_are_recorded_and_never_promoted(series, monkeypatch):
    code, _, out = semantic_run(series, monkeypatch, "this is not json")
    assert code == 0
    ranked = json.loads((out / "ranked_promos.json").read_text())
    assert ranked["count"] == 0                                           # nothing passes by default
    rej = {r["candidate_id"]: r for r in json.loads((out / "rejected_candidates.json").read_text())["candidates"]}
    assert rej["W01-C01"]["decision_stage"] == "semantic" and "no semantic assessment" in rej["W01-C01"]["decision_reasons"][0]
    sem = json.loads((out / "semantic_assessments.json").read_text())
    assert all(b["status"] == "failed" for b in sem["batches"])


def test_a_semantic_fail_is_saved_with_the_models_reason(series, monkeypatch):
    def reply(prompt):
        ids = [i for i in ("W01-C01", "W02-C01") if f'candidate_id="{i}"' in prompt]
        return json.dumps({"assessments": [A(i, hook=False, hook_level="weak") if i == "W01-C01" else A(i) for i in ids]})

    _, _, out = semantic_run(series, monkeypatch, reply)
    rej = {r["candidate_id"]: r for r in json.loads((out / "rejected_candidates.json").read_text())["candidates"]}
    assert rej["W01-C01"]["decision_stage"] == "semantic" and "reason about the opening seconds" in rej["W01-C01"]["decision_reasons"][0]
    assert [p["candidate_id"] for p in json.loads((out / "ranked_promos.json").read_text())["promos"]] == ["W02-C01"]
