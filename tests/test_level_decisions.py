from __future__ import annotations

from tl_helpers import build_index
from test_semantic_rank import A, clip_blocks, make_ev, pick
from promoai.semantic import PromoAssessment, semantic_gates


def test_gate_decisions_come_from_the_level_and_the_judge_boolean_is_only_recorded():
    def gates(**kw):
        return {g.gate: g for g in semantic_gates(PromoAssessment.model_validate(A("X", **kw)))}

    # judge says passed=False, but every level is a passing level -> gates PASS, boolean preserved, disagreement flagged
    g = gates(hook=False, hook_level="adequate", cliff=False, cliff_level="strong", sa_pass=False, standalone="needs_minor_context")
    for name in ("hook_strength", "cliffhanger_strength", "standalone_clarity"):
        assert g[name].passed is True
        assert g[name].details["llm_passed"] is False and g[name].details["disagreement"] is True
        assert g[name].details["derived_decision"] is True and "the level decides" in g[name].reason

    # judge says passed=True, but every level is a failing level -> gates FAIL, flagged
    g = gates(hook=True, hook_level="weak", cliff=True, cliff_level="weak", sa_pass=True, standalone="needs_prior_knowledge")
    assert all(not x.passed and x.details["llm_passed"] is True and x.details["disagreement"] for x in g.values())

    # boolean and level agree -> no flag, and the rubric's boundary levels land where the rubric says
    g = gates(hook=True, hook_level="adequate", cliff=False, cliff_level="weak", sa_pass=True, standalone="needs_minor_context")
    assert [g[n].passed for n in ("hook_strength", "cliffhanger_strength", "standalone_clarity")] == [True, False, True]
    assert not any(x.details["disagreement"] for x in g.values())

    # downstream: ranking follows the level-derived decision, the judge's all-False booleans do not block the clip
    idx = build_index({1: clip_blocks(1)})
    ev = make_ev(idx, 1, 0, "X", A("X", hook=False, hook_level="strong", cliff=False, cliff_level="strong",
                                    sa_pass=False, standalone="clear"))
    assert ev.assessment.hook_strength.passed is False                       # the original boolean is still in the assessment
    assert [e.cid for e in pick(idx, [ev]).ranked] == ["X"]
