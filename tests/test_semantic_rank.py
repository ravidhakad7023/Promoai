"""Semantic gates, ranking, dedupe, spoilers, diversity (fake LLM only; no network)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tl_helpers import build_index, lid, stored
from promoai.llm import FakeProvider
from promoai.ranking import (
    Evaluated, drop_exact_duplicates, overlap_fraction, select_promos, window_signature,
)
from promoai.schemas import StoryContext, CharacterInfo
from promoai.semantic import (
    AssessmentBatch, PromoAssessment, assess_batch, build_semantic_prompt, render_clip, semantic_gates,
)
from promoai.timeline import TimelineResolver


# ------------------------------------------------------------------ builders
def A(cid, *, pull="high", standalone="clear", coherence="strong", spoiler="none", hook=True, hook_level="strong",
      cliff=True, cliff_level="strong", sa_pass=True, note="") -> dict:
    return {
        "candidate_id": cid,
        "hook_strength": {"passed": hook, "level": hook_level, "reason": "reason about the opening seconds"},
        "cliffhanger_strength": {"passed": cliff, "level": cliff_level, "reason": "reason about the ending beat"},
        "standalone_clarity": {"passed": sa_pass, "level": standalone, "reason": "reason about newcomer clarity", "missing_context": []},
        "emotional_pull": pull, "narrative_coherence": coherence, "spoiler_risk": spoiler, "spoiler_note": note,
        "one_line_justification": f"Why {cid} works or does not work as a promo.",
    }


def clip_blocks(n_clips: int, gap: float = 100.0) -> list[tuple]:
    """n well-formed 30 s clips at 100 s spacing in one episode: lines 3k+1..3k+3 for clip k."""
    blocks = []
    for k in range(n_clips):
        o = k * gap
        blocks += [("CAMERA", o, o + 30.0, "wide"), ("ANN", o + 0.5, o + 1.5, "hook"), ("BOB", o + 28.5, o + 30.0, "ending")]
    return blocks


def make_ev(index, ep, k, cid, assessment: dict | None = None, window="W01", extra_segs=None) -> Evaluated:
    segs = [((ep, 3 * k + 1), (ep, 3 * k + 3))] + (extra_segs or [])
    cand = stored(index, segs=segs, hook=lid(ep, 3 * k + 2), cliff=lid(*segs[-1][1]), cid=cid)
    cand = cand.model_copy(update={"window_id": window, "candidate_id": cid})
    outcome = TimelineResolver(index).resolve(cand)
    ev = Evaluated(cand=cand, outcome=outcome)
    if assessment is not None:
        ev.assessment = PromoAssessment.model_validate(assessment)
        ev.semantic_gates = semantic_gates(ev.assessment)
    return ev


def nums(index):
    return {eid: ep.episode_number for eid, ep in index.episodes.items()}


def pick(index, evs, **kw):
    return select_promos(evs, nums(index), kw.pop("series_length", len(index.episodes)), **kw)


# ------------------------------------------------------------------ semantic schema + gates
def test_assessment_schema_accepts_case_and_spacing_variants():
    d = A("X", pull="HIGH", standalone="Needs Minor Context", coherence="Strong", spoiler="None")
    d["hook_strength"]["level"] = "Adequate"
    a = PromoAssessment.model_validate(d)
    assert (a.emotional_pull, a.standalone_clarity.level, a.hook_strength.level) == ("high", "needs_minor_context", "adequate")


@pytest.mark.parametrize("mutate", [
    lambda d: d["hook_strength"].update(reason="short"),            # reasons are mandatory (min 10 chars)
    lambda d: d.update(emotional_pull="amazing"),
    lambda d: d["hook_strength"].update(level="superb"),
    lambda d: d.pop("one_line_justification"),
])
def test_assessment_schema_rejects_bad_values(mutate):
    d = A("X")
    mutate(d)
    with pytest.raises(ValidationError):
        PromoAssessment.model_validate(d)


def test_semantic_gates_are_marked_as_assessments_and_keep_reasons():
    gates = semantic_gates(PromoAssessment.model_validate(A("X")))
    assert [g.gate for g in gates] == ["hook_strength", "cliffhanger_strength", "standalone_clarity"]
    assert all(g.kind == "semantic_assessment" and g.passed and "reason about" in g.reason for g in gates)


def test_a_pass_at_a_failing_level_is_treated_as_a_fail():
    gates = semantic_gates(PromoAssessment.model_validate(A("X", hook=True, hook_level="weak", sa_pass=True, standalone="needs_prior_knowledge")))
    by = {g.gate: g for g in gates}
    assert not by["hook_strength"].passed and "inconsistent" in by["hook_strength"].reason
    assert not by["standalone_clarity"].passed and by["standalone_clarity"].details["llm_passed"] is True


def test_explicit_llm_fail_is_a_fail():
    gates = {g.gate: g for g in semantic_gates(PromoAssessment.model_validate(A("X", cliff=False, cliff_level="weak")))}
    assert not gates["cliffhanger_strength"].passed and gates["hook_strength"].passed


# ------------------------------------------------------------------ prompt: the evaluator sees the FINAL clip
def test_prompt_renders_the_final_clip_in_timeline_order_with_relative_times():
    idx = build_index({1: [("CAMERA", 50, 80), ("ANN", 50.5, 52, "hello"), ("BOB", 78.5, 80, "bye")]})
    ev = make_ev(idx, 1, 0, "W01-C01")
    text = render_clip(idx, ev.clip)
    assert text.splitlines()[1].startswith("[0.0-30.0] CAMERA") and "[0.5-2.0] ANN" in text and "[28.5-30.0] BOB" in text
    assert "50.0" not in text and "80.0" not in text            # absolute episode times are hidden


def test_prompt_orders_by_time_not_by_file_order():
    idx = build_index({1: [("CAMERA", 0, 30), ("BOB", 28.5, 30, "late"), ("ANN", 0.5, 2, "early")]})
    cand = stored(idx, segs=[((1, 1), (1, 3))], hook=lid(1, 3), cliff=lid(1, 2))
    clip = TimelineResolver(idx).resolve(cand).clip
    lines = render_clip(idx, clip).splitlines()
    assert "ANN" in lines[2] and "BOB" in lines[3]


def test_prompt_marks_blocks_cut_by_the_clip_edge_and_numbers_segments():
    idx = build_index({1: [("CAMERA", 0, 40), ("ANN", 10, 11, "hook"), ("ACTION", 9, 12), ("BOB", 38.5, 40)],
                       2: [("CAMERA", 0, 15), ("CLEO", 0.5, 2), ("DAN", 13.5, 15)]})
    cand = stored(idx, segs=[((1, 2), (1, 4)), ((2, 1), (2, 3))], hook=lid(1, 2), cliff=lid(2, 3))
    text = render_clip(idx, TimelineResolver(idx).resolve(cand).clip)
    assert "(cut by clip edge)" in text and "segment 1 (from episode 1)" in text and "segment 2 (from episode 2)" in text
    assert "plays at" in text


def test_prompt_contains_rubric_notes_and_hides_the_proposers_claims():
    idx = build_index({1: clip_blocks(1)})
    ev = make_ev(idx, 1, 0, "W01-C01")
    ctx = StoryContext(characters=[CharacterInfo(name="Zelda", role="queen")], conflicts=["feud"])
    p = build_semantic_prompt(idx, [(ev.cand, ev.clip)], ctx, series_length=50)
    for phrase in ("first 3 seconds", "never seen the show", "FAIL", "NOT a cliffhanger", "Do not output timestamps",
                   "Zelda", "episode 1 of 50", 'candidate_id="W01-C01"'):
        assert phrase in p, phrase
    assert ev.cand.hook_description not in p or ev.cand.hook_description == "hook"      # proposer's text is not injected
    assert "why_it_works" not in p and "because" not in p.split("<clip")[1]


# ------------------------------------------------------------------ assess_batch
def run_batch(idx, evs, reply, ctx=None):
    prov = FakeProvider(reply if isinstance(reply, list) else [reply])
    got, problems = assess_batch(prov, idx, [(e.cand, e.clip) for e in evs], ctx, len(idx.episodes))
    return got, problems, prov


def test_assess_batch_returns_one_assessment_per_clip():
    idx = build_index({1: clip_blocks(2)})
    evs = [make_ev(idx, 1, 0, "W01-C01"), make_ev(idx, 1, 1, "W01-C02")]
    got, problems, prov = run_batch(idx, evs, json.dumps({"assessments": [A("W01-C01"), A("W01-C02", pull="low")]}))
    assert set(got) == {"W01-C01", "W01-C02"} and problems == [] and len(prov.prompts) == 1
    assert got["W01-C02"].emotional_pull == "low"


def test_missing_unknown_and_duplicate_assessments_are_reported_not_passed():
    idx = build_index({1: clip_blocks(2)})
    evs = [make_ev(idx, 1, 0, "W01-C01"), make_ev(idx, 1, 1, "W01-C02")]
    got, problems, _ = run_batch(idx, evs, json.dumps({"assessments": [A("W01-C01"), A("W01-C01"), A("W99-C99")]}))
    assert set(got) == {"W01-C01"}
    assert any("no assessment returned for W01-C02" in p for p in problems)
    assert any("unknown candidate" in p for p in problems) and any("duplicate" in p for p in problems)


def test_invalid_json_is_retried_once():
    idx = build_index({1: clip_blocks(1)})
    ev = make_ev(idx, 1, 0, "W01-C01")
    got, _, prov = run_batch(idx, [ev], ["not json", json.dumps({"assessments": [A("W01-C01")]})])
    assert "W01-C01" in got and len(prov.prompts) == 2 and prov.stats.json_retries == 1


# ------------------------------------------------------------------ ranking: mandatory gates
def test_only_candidates_passing_every_gate_are_ranked():
    idx = build_index({1: clip_blocks(1), 2: clip_blocks(1), 3: clip_blocks(1), 4: [("CAMERA", 0, 19.5), ("ANN", 0.5, 1.5), ("BOB", 18.0, 19.5)]})
    good = make_ev(idx, 1, 0, "G", A("G"))
    sem_fail = make_ev(idx, 2, 0, "F", A("F", hook=False, hook_level="weak"))
    unassessed = make_ev(idx, 3, 0, "U", None)
    too_short = make_ev(idx, 4, 0, "S", A("S"))
    sel = pick(idx, [good, sem_fail, unassessed, too_short])
    assert [e.cid for e in sel.ranked] == ["G"]
    d = sel.decisions
    assert d["F"].stage == "semantic" and "hook_strength" in d["F"].reasons[0]
    assert d["U"].stage == "semantic" and "no semantic assessment" in d["U"].reasons[0]
    assert d["S"].stage == "deterministic" and "duration_20_60s" in d["S"].reasons[0]
    assert all(d[x].status == "rejected" for x in "FUS") and d["G"].status == "recommended"


def test_a_candidate_with_a_high_llm_pull_cannot_bypass_a_failed_deterministic_gate():
    idx = build_index({1: [("CAMERA", 0, 75.8), ("ANN", 0.5, 2), ("BOB", 74.3, 75.8)]})
    ev = make_ev(idx, 1, 0, "LONG", A("LONG", pull="high"))
    assert pick(idx, [ev]).ranked == []


# ------------------------------------------------------------------ ranking order (qualitative tiers)
def test_ranking_follows_pull_then_standalone_then_coherence():
    idx = build_index({n: clip_blocks(1) for n in range(1, 8)})
    evs = [
        make_ev(idx, 1, 0, "low_pull", A("low_pull", pull="low")),
        make_ev(idx, 2, 0, "med_clear_strong", A("med_clear_strong", pull="medium")),
        make_ev(idx, 3, 0, "high_minor", A("high_minor", pull="high", standalone="needs_minor_context")),
        make_ev(idx, 4, 0, "high_clear_adequate", A("high_clear_adequate", pull="high", coherence="adequate")),
        make_ev(idx, 5, 0, "high_clear_strong", A("high_clear_strong", pull="high")),
    ]
    order = [e.cid for e in pick(idx, evs).ranked]
    assert order == ["high_clear_strong", "high_clear_adequate", "high_minor", "med_clear_strong", "low_pull"]


def test_ties_break_deterministically_and_no_numeric_score_is_emitted():
    idx = build_index({n: clip_blocks(1) for n in (1, 2, 3)})
    evs = [make_ev(idx, n, 0, f"C{n}", A(f"C{n}")) for n in (3, 1, 2)]
    assert [e.cid for e in pick(idx, evs).ranked] == ["C1", "C2", "C3"]          # earlier episode first
    from promoai.validate import promo_record
    s = pick(idx, evs)
    rec = promo_record(idx, s.ranked[0], 1, s.decisions["C1"])
    flat = json.dumps(rec).lower()
    assert "score" not in flat and "weight" not in flat
    assert rec["qualitative_assessment"]["emotional_pull"] == "high" and rec["one_line_justification"]


# ------------------------------------------------------------------ duplicates / overlaps
def test_exact_duplicate_windows_are_dropped_before_semantic_spend():
    idx = build_index({1: clip_blocks(1)})
    a, b = make_ev(idx, 1, 0, "A"), make_ev(idx, 1, 0, "B")
    assert window_signature(a.clip) == window_signature(b.clip)
    kept, dropped = drop_exact_duplicates([a, b])
    assert [e.cid for e in kept] == ["A"] and dropped["B"].stage == "duplicate" and "exact duplicate of A" in dropped["B"].reasons[0]


def test_overlapping_promos_keep_only_the_higher_ranked():
    # clip B starts 20 s into clip A's footage: 10 s shared of 30 s (33% >= 25%)
    idx = build_index({1: [("CAMERA", 0, 30), ("ANN", 0.5, 1.5), ("BOB", 28.5, 30),
                           ("CAMERA", 20, 50), ("CLEO", 20.5, 21.5), ("DAN", 48.5, 50)]})
    lo, hi = make_ev(idx, 1, 0, "A", A("A", pull="medium")), make_ev(idx, 1, 1, "B", A("B", pull="high"))
    assert overlap_fraction(lo.clip, hi.clip) == pytest.approx(10 / 30)
    sel = pick(idx, [lo, hi])
    assert [e.cid for e in sel.ranked] == ["B"]
    assert sel.decisions["A"].stage == "overlap" and "overlaps higher-ranked B" in sel.decisions["A"].reasons[0]


def test_small_overlap_is_tolerated_and_same_footage_in_different_episodes_is_not_overlap():
    idx = build_index({1: [("CAMERA", 0, 30), ("ANN", 0.5, 1.5), ("BOB", 28.5, 30),
                           ("CAMERA", 25, 55), ("CLEO", 25.5, 26.5), ("DAN", 53.5, 55)],      # 5 s shared = 17%
                       2: clip_blocks(1)})
    a, b, c = make_ev(idx, 1, 0, "A", A("A")), make_ev(idx, 1, 1, "B", A("B")), make_ev(idx, 2, 0, "C", A("C"))
    assert overlap_fraction(a.clip, b.clip) < 0.25 and overlap_fraction(a.clip, c.clip) == 0.0
    assert {e.cid for e in pick(idx, [a, b, c]).ranked} == {"A", "B", "C"}


# ------------------------------------------------------------------ diversity / list size
def test_episode_diversity_caps_promos_per_episode():
    idx = build_index({1: clip_blocks(4), 2: clip_blocks(1), 3: clip_blocks(1)})
    evs = [make_ev(idx, 1, k, f"E1-{k}", A(f"E1-{k}", pull="high")) for k in range(4)]
    evs += [make_ev(idx, 2, 0, "E2", A("E2", pull="medium")), make_ev(idx, 3, 0, "E3", A("E3", pull="medium"))]
    sel = pick(idx, evs, max_per_episode=2)
    chosen = [e.cid for e in sel.ranked]
    assert sum(c.startswith("E1") for c in chosen) == 2 and {"E2", "E3"} <= set(chosen)
    capped = [d for d in sel.decisions.values() if d.stage == "diversity_cap"]
    assert len(capped) == 2 and all(d.status == "passed_not_selected" for d in capped)


def test_top_ten_is_a_cap_not_a_target():
    idx = build_index({n: clip_blocks(1) for n in range(1, 16)})
    evs = [make_ev(idx, n, 0, f"C{n:02d}", A(f"C{n:02d}")) for n in range(1, 16)]
    sel = pick(idx, evs)
    assert len(sel.ranked) == 10 and sum(d.stage == "list_cap" for d in sel.decisions.values()) == 5
    few = pick(idx, evs[:3])
    assert len(few.ranked) == 3                                   # not padded to 10
    assert pick(idx, []).ranked == []


def test_cross_episode_and_multi_segment_promos_are_included_when_strong_ones_exist():
    eps = {n: clip_blocks(1) for n in range(1, 12)}
    eps[12] = [("CAMERA", 0, 12), ("ANN", 0.5, 2), ("BOB", 10.5, 12)]
    eps[13] = [("CAMERA", 0, 15), ("CLEO", 0.5, 2), ("DAN", 13.5, 15)]
    idx = build_index(eps)
    singles = [make_ev(idx, n, 0, f"S{n:02d}", A(f"S{n:02d}", pull="high")) for n in range(1, 11)]
    cross = make_ev(idx, 12, 0, "X", A("X", pull="medium"), extra_segs=None)
    cross_cand = stored(idx, segs=[((12, 1), (12, 3)), ((13, 1), (13, 3))], hook=lid(12, 2), cliff=lid(13, 3), cid="X").model_copy(update={"window_id": "W01", "candidate_id": "X"})
    cross = Evaluated(cand=cross_cand, outcome=TimelineResolver(idx).resolve(cross_cand))
    cross.assessment = PromoAssessment.model_validate(A("X", pull="medium"))
    cross.semantic_gates = semantic_gates(cross.assessment)
    assert cross.outcome.accepted and len(cross.clip.episode_ids) == 2
    sel = pick(idx, [*singles, cross])
    ids = [e.cid for e in sel.ranked]
    assert "X" in ids and len(ids) == 10
    assert sel.decisions["X"].reasons == ["included to represent cross-episode promos"]
    assert sum(d.stage == "list_cap" for d in sel.decisions.values()) == 1        # exactly one single-episode clip was displaced


def test_weak_cross_episode_promo_is_not_forced_in():
    eps = {n: clip_blocks(1) for n in range(1, 12)}
    eps[12] = [("CAMERA", 0, 12), ("ANN", 0.5, 2), ("BOB", 10.5, 12)]
    eps[13] = [("CAMERA", 0, 15), ("CLEO", 0.5, 2), ("DAN", 13.5, 15)]
    idx = build_index(eps)
    singles = [make_ev(idx, n, 0, f"S{n:02d}", A(f"S{n:02d}", pull="high")) for n in range(1, 11)]
    c = stored(idx, segs=[((12, 1), (12, 3)), ((13, 1), (13, 3))], hook=lid(12, 2), cliff=lid(13, 3), cid="X").model_copy(update={"window_id": "W01", "candidate_id": "X"})
    x = Evaluated(cand=c, outcome=TimelineResolver(idx).resolve(c))
    x.assessment = PromoAssessment.model_validate(A("X", pull="low"))
    x.semantic_gates = semantic_gates(x.assessment)
    assert "X" not in [e.cid for e in pick(idx, [*singles, x]).ranked]


# ------------------------------------------------------------------ spoilers
def test_major_late_season_spoiler_is_rejected_but_early_or_minor_is_not():
    idx = build_index({n: clip_blocks(1) for n in range(1, 51)})
    early = make_ev(idx, 5, 0, "early_major", A("early_major", spoiler="major", note="reveals the villain"))
    late_major = make_ev(idx, 45, 0, "late_major", A("late_major", spoiler="major", note="reveals the villain"))
    late_minor = make_ev(idx, 46, 0, "late_minor", A("late_minor", spoiler="minor"))
    sel = pick(idx, [early, late_major, late_minor])
    assert {e.cid for e in sel.ranked} == {"early_major", "late_minor"}
    d = sel.decisions["late_major"]
    assert d.stage == "spoiler" and "episode 45 of 50" in d.reasons[0] and "reveals the villain" in d.reasons[0]


def test_late_season_threshold_follows_the_series_length_not_a_constant():
    idx = build_index({n: clip_blocks(1) for n in range(1, 11)})                  # a 10-episode series
    e9 = make_ev(idx, 9, 0, "e9", A("e9", spoiler="major", note="twist"))
    e8 = make_ev(idx, 8, 0, "e8", A("e8", spoiler="major", note="twist"))
    sel = pick(idx, [e9, e8])
    assert sel.decisions["e9"].stage == "spoiler" and sel.decisions["e8"].status == "recommended"   # 8 is not > 0.8*10


# ------------------------------------------------------------------ rejected candidates keep detailed reasons
def test_rejection_records_include_every_failed_gate_with_its_reason():
    idx = build_index({1: clip_blocks(1)})
    ev = make_ev(idx, 1, 0, "F", A("F", hook=False, hook_level="weak", cliff=False, cliff_level="weak"))
    d = pick(idx, [ev]).decisions["F"]
    assert d.stage == "semantic" and len(d.reasons) == 2
    assert d.reasons[0].startswith("hook_strength:") and d.reasons[1].startswith("cliffhanger_strength:")
