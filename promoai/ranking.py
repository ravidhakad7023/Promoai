"""Selection of final promos: gates -> dedupe -> rank -> diversity.

Only a candidate that passes EVERY mandatory gate can be ranked:
  deterministic (Python): duration 20-60 s, hook within 3 s, cliffhanger reference valid,
                          no dialogue cut, segments distinct, timestamps valid
  semantic (LLM rubric) : hook strength, cliffhanger strength, standalone clarity

Ranking is a lexicographic ordering of qualitative tiers - there is deliberately no
weighted numeric score, which would be false precision on top of LLM opinions:
    1. emotional / curiosity pull   (high > medium > low)
    2. standalone clarity           (clear > needs_minor_context)
    3. narrative coherence          (strong > adequate > weak)
  then, only as tie-breakers: lower spoiler risk, then earlier episode, then candidate id.
Tie-breaking by episode is a convenience, not a quality judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from .schemas import StoredCandidate
from .semantic import PromoAssessment
from .timeline import ClipTimeline, GateResult, TimelineOutcome

PULL = {"high": 0, "medium": 1, "low": 2}
STANDALONE = {"clear": 0, "needs_minor_context": 1, "needs_prior_knowledge": 2}
COHERENCE = {"strong": 0, "adequate": 1, "weak": 2}
SPOILER = {"none": 0, "minor": 1, "major": 2}

DEFAULT_MAX_PROMOS = 10          # an upper bound, never a target
DEFAULT_MAX_PER_EPISODE = 2
OVERLAP_MAX_FRACTION = 0.25      # overlap >= 25% of the shorter clip counts as a duplicate
LATE_SEASON_FRACTION = 0.80      # episodes after 80% of the series are "late season"


class Decision(BaseModel):
    candidate_id: str
    status: str                  # recommended | rejected | passed_not_selected
    stage: str                   # deterministic | duplicate | semantic | spoiler | overlap | diversity_cap | list_cap | selected
    reasons: list[str] = Field(default_factory=list)


@dataclass
class Evaluated:
    cand: StoredCandidate
    outcome: TimelineOutcome
    assessment: Optional[PromoAssessment] = None
    semantic_gates: list[GateResult] = field(default_factory=list)
    votes: list = field(default_factory=list)        # individual judge samples (only filled when judge_votes > 1)

    @property
    def clip(self) -> Optional[ClipTimeline]:
        return self.outcome.clip

    @property
    def cid(self) -> str:
        return self.cand.candidate_id

    @property
    def all_gates(self) -> list[GateResult]:
        return [*self.outcome.gates, *self.semantic_gates]


@dataclass
class Selection:
    ranked: list[Evaluated]
    decisions: dict[str, Decision]


# --------------------------------------------------------------------------- helpers
def window_signature(clip: ClipTimeline) -> tuple:
    return tuple(sorted(clip.windows))


def overlap_fraction(a: ClipTimeline, b: ClipTimeline) -> float:
    """Shared footage / duration of the shorter clip.  Different episodes never overlap
    (every episode has its own clock)."""
    shared = 0
    for ea, sa, xa in a.windows:
        for eb, sb, xb in b.windows:
            if ea == eb:
                shared += max(0, min(xa, xb) - max(sa, sb))
    shorter = min(a.total_duration_ms, b.total_duration_ms)
    return shared / shorter if shorter else 0.0


# Plain-language description of EXACTLY what rank_key sorts by, in order (smaller sorts first).  The run
# report prints this list, so keep it in sync with rank_key below.
RANK_KEY_DESCRIPTION = [
    "emotional / curiosity pull (judge tier): high, then medium, then low",
    "standalone clarity (judge level): clear, then needs_minor_context",
    "narrative coherence (judge tier): strong, then adequate, then weak",
    "spoiler risk (judge tier): none, then minor, then major",
    "episode number of the clip's first segment: earlier episode first",
    "candidate id: plain string order",
]


def rank_key(ev: Evaluated, episode_numbers: dict[str, int]) -> tuple:
    a = ev.assessment
    assert a is not None and ev.clip is not None
    first_ep = episode_numbers[ev.clip.segments[0].episode_id]
    return (PULL[a.emotional_pull], STANDALONE[a.standalone_clarity.level], COHERENCE[a.narrative_coherence],
            SPOILER[a.spoiler_risk], first_ep, ev.cid)


def is_late_season(clip: ClipTimeline, episode_numbers: dict[str, int], series_length: int, fraction: float) -> bool:
    return max(episode_numbers[e] for e in clip.episode_ids) > fraction * series_length


def drop_exact_duplicates(items: list[Evaluated]) -> tuple[list[Evaluated], dict[str, Decision]]:
    """Before any LLM spend: two deterministic-pass candidates with identical final windows are the same clip."""
    seen: dict[tuple, str] = {}
    keep: list[Evaluated] = []
    dropped: dict[str, Decision] = {}
    for ev in items:
        if not (ev.outcome.accepted and ev.clip):
            keep.append(ev)
            continue
        sig = window_signature(ev.clip)
        if sig in seen:
            dropped[ev.cid] = Decision(candidate_id=ev.cid, status="rejected", stage="duplicate",
                                       reasons=[f"exact duplicate of {seen[sig]} (identical final clip windows)"])
        else:
            seen[sig] = ev.cid
            keep.append(ev)
    return keep, dropped


# --------------------------------------------------------------------------- selection
def select_promos(
    items: list[Evaluated],
    episode_numbers: dict[str, int],
    series_length: int,
    *,
    max_promos: int = DEFAULT_MAX_PROMOS,
    max_per_episode: int = DEFAULT_MAX_PER_EPISODE,
    overlap_max: float = OVERLAP_MAX_FRACTION,
    late_fraction: float = LATE_SEASON_FRACTION,
) -> Selection:
    decisions: dict[str, Decision] = {}
    eligible: list[Evaluated] = []

    for ev in items:
        if ev.cid in decisions:
            continue
        if not ev.outcome.accepted or ev.clip is None:
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="rejected", stage="deterministic",
                                         reasons=list(ev.outcome.rejection_reasons))
        elif ev.assessment is None:
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="rejected", stage="semantic",
                                         reasons=["no semantic assessment was obtained, so the mandatory semantic gates cannot be passed"])
        elif any(not g.passed for g in ev.semantic_gates):
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="rejected", stage="semantic",
                                         reasons=[f"{g.gate}: {g.reason}" for g in ev.semantic_gates if not g.passed])
        elif ev.assessment.spoiler_risk == "major" and is_late_season(ev.clip, episode_numbers, series_length, late_fraction):
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="rejected", stage="spoiler",
                                         reasons=[f"major late-season spoiler (episode {max(episode_numbers[e] for e in ev.clip.episode_ids)}"
                                                  f" of {series_length}): {ev.assessment.spoiler_note or 'no note given'}"])
        else:
            eligible.append(ev)

    eligible.sort(key=lambda e: rank_key(e, episode_numbers))

    selected: list[Evaluated] = []
    per_ep: dict[str, int] = {}

    def episode_cap_ok(ev: Evaluated, counts: dict[str, int]) -> bool:
        return all(counts.get(e, 0) < max_per_episode for e in ev.clip.episode_ids)  # type: ignore[union-attr]

    def add(ev: Evaluated) -> None:
        selected.append(ev)
        for e in ev.clip.episode_ids:  # type: ignore[union-attr]
            per_ep[e] = per_ep.get(e, 0) + 1

    def remove(ev: Evaluated) -> None:
        selected.remove(ev)
        for e in ev.clip.episode_ids:  # type: ignore[union-attr]
            per_ep[e] -= 1

    for ev in eligible:
        clash = next((s for s in selected if overlap_fraction(ev.clip, s.clip) >= overlap_max), None)   # type: ignore[arg-type]
        if clash is not None:
            frac = overlap_fraction(ev.clip, clash.clip)  # type: ignore[arg-type]
            exact = window_signature(ev.clip) == window_signature(clash.clip)  # type: ignore[arg-type]
            decisions[ev.cid] = Decision(
                candidate_id=ev.cid, status="rejected", stage="duplicate" if exact else "overlap",
                reasons=[f"{'exact duplicate' if exact else 'overlaps'} higher-ranked {clash.cid} "
                         f"({frac:.0%} of the shorter clip; limit {overlap_max:.0%})"])
        elif not episode_cap_ok(ev, per_ep):
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="passed_not_selected", stage="diversity_cap",
                                         reasons=[f"episode diversity: at most {max_per_episode} promos per episode"])
        elif len(selected) >= max_promos:
            decisions[ev.cid] = Decision(candidate_id=ev.cid, status="passed_not_selected", stage="list_cap",
                                         reasons=[f"list is limited to {max_promos} promos"])
        else:
            add(ev)

    # ---- make sure strong multi-segment / cross-episode promos are represented when they exist
    def is_cross(e: Evaluated) -> bool:
        return len(e.clip.episode_ids) > 1  # type: ignore[union-attr]

    def is_multi(e: Evaluated) -> bool:
        return len(e.clip.segments) > 1  # type: ignore[union-attr]

    for pred, label in ((is_cross, "cross-episode"), (is_multi, "multi-segment")):
        if any(pred(s) for s in selected):
            continue
        pool = [e for e in eligible if pred(e) and e not in selected and e.cid in decisions
                and decisions[e.cid].stage in ("diversity_cap", "list_cap") and e.assessment.emotional_pull != "low"]  # type: ignore[union-attr]
        if not pool:
            continue
        best = pool[0]
        displaced: Optional[Evaluated] = None
        if len(selected) >= max_promos:
            displaced = next((s for s in reversed(selected) if not is_cross(s) and not is_multi(s)), None)
            if displaced is None:
                continue
            remove(displaced)
        if not episode_cap_ok(best, per_ep) or any(overlap_fraction(best.clip, s.clip) >= overlap_max for s in selected):  # type: ignore[arg-type]
            if displaced is not None:
                add(displaced)
            continue
        add(best)
        decisions[best.cid] = Decision(candidate_id=best.cid, status="recommended", stage="selected",
                                       reasons=[f"included to represent {label} promos"])
        if displaced is not None:
            decisions[displaced.cid] = Decision(candidate_id=displaced.cid, status="passed_not_selected", stage="list_cap",
                                                reasons=[f"displaced by a {label} promo to keep the list varied"])

    selected.sort(key=lambda e: rank_key(e, episode_numbers))
    for ev in selected:
        decisions.setdefault(ev.cid, Decision(candidate_id=ev.cid, status="recommended", stage="selected"))
        decisions[ev.cid] = decisions[ev.cid].model_copy(update={"status": "recommended", "stage": "selected"})
    return Selection(ranked=selected, decisions=decisions)
