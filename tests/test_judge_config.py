"""Judge temperature (default 0.0) and judge_votes (default 1; majority voting when > 1)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from test_llm import Script, ok_body, settings as llm_settings
from test_semantic_rank import A
from test_trim_semantic import series  # noqa: F401  (fixture)
from promoai import validate
from promoai.config import ConfigError, Settings, load_settings
from promoai.llm import CachedProvider, FakeProvider
from promoai.llm_gemini import GeminiProvider
from promoai.semantic import PromoAssessment, aggregate_votes


# ------------------------------------------------------------------ configuration
def test_judge_defaults_and_overrides(tmp_path):
    s = load_settings(tmp_path / "none", environ={})
    assert (s.judge_temperature, s.judge_votes, s.temperature) == (0.0, 1, 0.4)        # judge 0.0, proposer unchanged
    s = load_settings(tmp_path / "none", environ={"LLM_JUDGE_TEMPERATURE": "0.4", "LLM_JUDGE_VOTES": "3"})
    assert (s.judge_temperature, s.judge_votes, s.temperature) == (0.4, 3, 0.4)


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", "abc"])
def test_invalid_judge_votes_is_a_config_error(tmp_path, bad):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "none", environ={"LLM_JUDGE_VOTES": bad})


def test_temperature_reaches_the_request_and_the_cache_key(tmp_path):
    judge_t, prop_t = Script((200, ok_body())), Script((200, ok_body()))
    base = llm_settings(temperature=0.4)
    judge = GeminiProvider(dataclasses.replace(base, temperature=0.0), transport=judge_t, sleep=lambda s: None)
    prop = GeminiProvider(base, transport=prop_t, sleep=lambda s: None)
    judge.generate("p"), prop.generate("p")
    assert judge_t.requests[0][3]["generationConfig"]["temperature"] == 0.0
    assert prop_t.requests[0][3]["generationConfig"]["temperature"] == 0.4
    ck = lambda p: CachedProvider(p, tmp_path)._key("p")
    assert ck(judge) != ck(prop)                                    # a temperature-0 answer is never served for 0.4


def test_cache_key_format_is_unchanged_without_a_salt(tmp_path):
    c = CachedProvider(FakeProvider(["x"]), tmp_path)
    assert c._key("prompt") == hashlib.sha256(f"{c.fingerprint}\0prompt".encode()).hexdigest()   # old entries still match
    assert c.salted("vote2")._key("prompt") != c._key("prompt")


def test_cli_judge_temperature_defaults_to_zero_and_can_be_overridden(series, monkeypatch):
    seen = []
    prov = FakeProvider(lambda p: json.dumps({"assessments": [A(i) for i in ("W01-C01", "W02-C01") if f'candidate_id="{i}"' in p]}))
    monkeypatch.setattr(validate, "create_provider", lambda s, *a, **k: (seen.append(s), prov)[1])
    monkeypatch.setattr(validate, "load_settings", lambda: Settings(model="m", api_key="k"))
    base = ["--series-dir", str(series), "--phase2-run", "r1", "--semantic"]
    assert validate.main([*base, "--out-name", "a"]) == 0
    assert validate.main([*base, "--out-name", "b", "--judge-temperature", "0.4"]) == 0
    assert [s.temperature for s in seen] == [0.0, 0.4]
    rep = json.loads((series / "phase3" / "a" / "report.json").read_text())
    assert rep["judge_temperature"] == 0.0 and rep["judge_votes"] == 1


# ------------------------------------------------------------------ judge_votes = 1 keeps the single-judge behaviour
def test_one_vote_means_one_call_per_batch_and_the_old_output_shape(series, monkeypatch):
    prov = FakeProvider(lambda p: json.dumps({"assessments": [A(i) for i in ("W01-C01", "W02-C01") if f'candidate_id="{i}"' in p]}))
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: prov)
    monkeypatch.setattr(validate, "load_settings", lambda: Settings(model="m", api_key="k"))
    assert validate.main(["--series-dir", str(series), "--phase2-run", "r1", "--semantic", "--out-name", "one"]) == 0
    assert len(prov.prompts) == 2                                           # two windows, one call each
    sem = json.loads((series / "phase3" / "one" / "semantic_assessments.json").read_text())
    assert sem["judge_votes"] == 1 and "votes" not in sem
    ranked = json.loads((series / "phase3" / "one" / "ranked_promos.json").read_text())["promos"]
    assert all("votes" not in g["details"] for p in ranked for g in p["gates"] if g["kind"] == "semantic_assessment")


# ------------------------------------------------------------------ majority voting
def vote(**kw) -> PromoAssessment:
    return PromoAssessment.model_validate(A("X", **kw))


def test_majority_needs_a_strict_majority_of_the_configured_votes():
    passing, failing = vote(), vote(hook=False, hook_level="weak")
    agg, gates = aggregate_votes([passing, passing, failing])
    hook = next(g for g in gates if g.gate == "hook_strength")
    assert hook.passed and hook.details["passes"] == 2 and hook.details["judge_votes"] == 3 and "2/3 votes passed" in hook.reason
    assert agg.hook_strength.passed is True
    _, gates = aggregate_votes([passing, failing, failing])
    assert not next(g for g in gates if g.gate == "hook_strength").passed
    _, gates = aggregate_votes([passing, failing])                           # even split -> fail (conservative)
    assert not next(g for g in gates if g.gate == "hook_strength").passed


def test_a_missing_vote_counts_as_not_passing():
    p = vote()
    _, g2 = aggregate_votes([p, None])                                       # 1 of 2: not a strict majority
    assert not next(g for g in g2 if g.gate == "hook_strength").passed
    _, g3 = aggregate_votes([p, p, None])                                    # 2 of 3 still passes
    hook = next(g for g in g3 if g.gate == "hook_strength")
    assert hook.passed and hook.details["votes"][2] == {"vote": 3, "passed": None, "level": None,
                                                        "reason": "no usable verdict (counts as not passing)"}
    assert aggregate_votes([None, None, None]) == (None, [])


def test_every_individual_verdict_and_justification_is_preserved():
    votes = [vote(hook_level="strong"), vote(hook_level="adequate"), vote(hook=False, hook_level="weak")]
    for v, text in zip(votes, ("first reason text", "second reason text", "third reason text")):
        v.hook_strength.reason = text
    _, gates = aggregate_votes(votes)
    rows = next(g for g in gates if g.gate == "hook_strength").details["votes"]
    assert [(r["vote"], r["passed"], r["level"], r["reason"]) for r in rows] == [
        (1, True, "strong", "first reason text"), (2, True, "adequate", "second reason text"), (3, False, "weak", "third reason text")]


def test_qualitative_tiers_use_the_mode_and_break_ties_conservatively():
    agg, _ = aggregate_votes([vote(pull="high"), vote(pull="low")])
    assert agg.emotional_pull == "low"
    agg, _ = aggregate_votes([vote(pull="high"), vote(pull="high"), vote(pull="low")])
    assert agg.emotional_pull == "high"
    agg, _ = aggregate_votes([vote(spoiler="none"), vote(spoiler="major")])
    assert agg.spoiler_risk == "major"


def test_trimmed_clips_get_the_coherence_gate_in_each_vote():
    _, gates = aggregate_votes([vote(coherence="strong"), vote(coherence="weak"), vote(coherence="weak")], trimmed=True)
    g = next(x for x in gates if x.gate == "coherence_after_trim")
    assert not g.passed and g.details["passes"] == 1


def test_votes_get_distinct_cache_entries_and_a_rerun_is_free(tmp_path):
    fake = FakeProvider(["v1", "v2", "v3"])
    cached = CachedProvider(fake, tmp_path)
    got = [p.generate("same prompt") for p in validate.vote_providers(cached, 3)]
    assert got == ["v1", "v2", "v3"] and len(fake.prompts) == 3              # NOT collapsed into one cached answer
    again = FakeProvider(["never"])
    assert [p.generate("same prompt") for p in validate.vote_providers(CachedProvider(again, tmp_path), 3)] == got
    assert again.prompts == []


def test_cli_majority_voting_end_to_end_keeps_all_votes(series, monkeypatch):
    calls: dict[str, int] = {}
    plan = {"W01-C01": [True, True, False], "W02-C01": [True, False, False]}    # per vote: does the hook pass?

    def reply(prompt):
        cid = next(i for i in plan if f'candidate_id="{i}"' in prompt)
        k = calls[cid] = calls.get(cid, 0) + 1
        ok = plan[cid][k - 1]
        a = A(cid, hook=ok, hook_level="strong" if ok else "weak")
        a["hook_strength"]["reason"] = f"{cid} vote {k} hook reasoning"
        return json.dumps({"assessments": [a]})

    prov = FakeProvider(reply)
    monkeypatch.setattr(validate, "create_provider", lambda *a, **k: prov)
    monkeypatch.setattr(validate, "load_settings", lambda: Settings(model="m", api_key="k"))
    assert validate.main(["--series-dir", str(series), "--phase2-run", "r1", "--semantic", "--judge-votes", "3", "--out-name", "v3"]) == 0
    assert len(prov.prompts) == 6                                            # 2 batches x 3 votes
    out = series / "phase3" / "v3"
    ranked = [p["candidate_id"] for p in json.loads((out / "ranked_promos.json").read_text())["promos"]]
    assert ranked == ["W01-C01"]                                             # 2/3 passes; 1/3 fails
    sem = json.loads((out / "semantic_assessments.json").read_text())
    assert sem["judge_votes"] == 3 and set(sem["votes"]) == {"W01-C01", "W02-C01"}
    assert [v["hook_strength"]["reason"] for v in sem["votes"]["W02-C01"]] == [f"W02-C01 vote {k} hook reasoning" for k in (1, 2, 3)]
    rej = {r["candidate_id"]: r for r in json.loads((out / "rejected_candidates.json").read_text())["candidates"]}
    assert rej["W02-C01"]["decision_stage"] == "semantic" and "1/3 votes passed" in rej["W02-C01"]["decision_reasons"][0]
    assert json.loads((out / "report.json").read_text())["judge_votes"] == 3
