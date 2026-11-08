"""Deterministic timeline resolution and mandatory *deterministic* quality gates.

Input : a Phase-2 :class:`StoredCandidate` (line-ID references proposed by the LLM)
Output: a :class:`TimelineOutcome` - the real clip windows, every boundary adjustment
        with its source reference, and pass/fail results with reasons.

Model of the screenplay (see Phase 1): blocks (CAMERA / ACTION / SOUND FX / dialogue)
overlap in time, and each episode has its own clock starting at 0.  Therefore:

* A segment's window is ``[min(start_ms), max(end_ms)]`` over the blocks selected by its
  line-ID range.  Block durations are NEVER summed.
* Windows from different episodes are measured independently; the clip duration is the
  sum of the segment window lengths.
* A window boundary may only sit on a real block boundary (a block's start_ms for a
  start, a block's end_ms for an end) and must never fall in the middle of a dialogue
  line.  Other blocks (camera/action/sound) may straddle an edge - a cut through a shot
  is normal editing - but those are listed in ``partially_cut_line_ids``.
* The LLM's hook timing claim is ignored.  Python measures it, and if the hook is later
  than 3.0 s it moves the start boundary to a valid block start shortly before the hook.

All arithmetic is integer milliseconds, so 3000 and 60000 are exact thresholds.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .index import ScreenplayIndex
from .models import Line, LineKind, TimestampStatus
from .schemas import StoredCandidate
from .timecode import format_timecode


# ---- thresholds ----------------------------------------------------------------------
@dataclass(frozen=True)
class TimelineConfig:
    """Every tunable rule of the timeline resolver, in one place (milliseconds, all inclusive limits).

    Pass a custom instance to :class:`TimelineResolver`; the CLI exposes ``--cliff-tail-max-s``."""

    min_clip_ms: int = 20_000           # shortest valid promo
    max_clip_ms: int = 60_000           # longest valid promo
    hook_max_ms: int = 3_000            # the hook must START <= 3.0 s after the clip start
    hook_preferred_lead_ms: int = 2_000  # when moving the start, prefer <= 2.0 s of lead-in (safety margin)
    cliff_tail_max_ms: int = 2_000      # footage allowed AFTER the cliffhanger line ends (exceeding it triggers a trim)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_CONFIG = TimelineConfig()
# Module-level aliases of the defaults (kept for callers and tests that import them).
MIN_CLIP_MS = DEFAULT_CONFIG.min_clip_ms
MAX_CLIP_MS = DEFAULT_CONFIG.max_clip_ms
HOOK_MAX_MS = DEFAULT_CONFIG.hook_max_ms
HOOK_PREFERRED_LEAD_MS = DEFAULT_CONFIG.hook_preferred_lead_ms
CLIFF_TAIL_MAX_MS = DEFAULT_CONFIG.cliff_tail_max_ms     # was 5 s until Phase 3 refinement 2 (see report)


def _s(ms: int) -> str:
    return f"{ms / 1000:.1f}s"


# --------------------------------------------------------------------------- models
class GateResult(BaseModel):
    gate: str
    passed: bool
    kind: str = "deterministic"      # "deterministic" (Python enforced) | "semantic_assessment" (LLM opinion)
    reason: str
    details: dict = Field(default_factory=dict)


class Boundary(BaseModel):
    """A clip edge, tied to the real block that defines it."""

    side: str                        # "start" | "end"
    episode_id: str
    time_ms: int
    timecode: str                    # MM:SS.ff of time_ms
    line_id: str                     # the block whose start_ms / end_ms defines this boundary
    source_file: str
    source_line: int
    original_ms: int                 # where the boundary was before any adjustment
    adjusted: bool = False
    adjustment_reason: Optional[str] = None


class SegmentTimeline(BaseModel):
    segment_index: int
    episode_id: str
    start_line_id: str               # the LLM's references, preserved
    end_line_id: str
    selected_line_count: int
    selected_window_ms: int          # unadjusted window = max end - min start of the selected blocks
    start: Boundary
    end: Boundary
    duration_ms: int
    lines_in_window: int             # blocks overlapping the final window
    dropped_selected_line_ids: list[str] = Field(default_factory=list)   # selected but trimmed away
    partially_cut_line_ids: list[str] = Field(default_factory=list)      # non-dialogue blocks crossing an edge
    cut_dialogue_line_ids: list[str] = Field(default_factory=list)       # must be empty


class Adjustment(BaseModel):
    """One change Python made to the LLM-proposed window.  The original proposal is never overwritten."""

    segment_index: int
    side: str
    kind: str = ""                   # dialogue_snap | hook_alignment | cliffhanger_trim
    from_ms: int
    to_ms: int
    new_boundary_line_id: str
    reason: str
    detail: dict = Field(default_factory=dict)


class HookCheck(BaseModel):
    line_id: str
    episode_id: str
    start_ms: int
    offset_before_ms: Optional[int]  # measured against the unadjusted clip start
    offset_ms: Optional[int]         # measured against the final clip start
    within_limit: bool


class CliffhangerCheck(BaseModel):
    line_id: str
    end_ms: Optional[int]
    inside_final_window: bool
    tail_slack_ms: Optional[int]     # clip end - cliffhanger line end


class ClipTimeline(BaseModel):
    candidate_id: str
    segments: list[SegmentTimeline]
    total_duration_ms: int
    original_total_duration_ms: int  # sum of unadjusted segment windows
    hook: HookCheck
    cliffhanger: CliffhangerCheck
    adjustments: list[Adjustment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def windows(self) -> list[tuple[str, int, int]]:
        return [(s.episode_id, s.start.time_ms, s.end.time_ms) for s in self.segments]

    @property
    def episode_ids(self) -> list[str]:
        out: list[str] = []
        for s in self.segments:
            if s.episode_id not in out:
                out.append(s.episode_id)
        return out


class TimelineOutcome(BaseModel):
    candidate_id: str
    window_id: str
    accepted: bool
    clip: Optional[ClipTimeline] = None
    gates: list[GateResult] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- per-episode helper
class _EpisodeTimeline:
    """Boundary lookups for one episode (timestamps are episode-relative)."""

    def __init__(self, lines: list[Line]) -> None:
        self.ok = [l for l in lines if l.timestamp_status is TimestampStatus.OK]
        self.start_line: dict[int, Line] = {}
        self.end_line: dict[int, Line] = {}
        for l in self.ok:                    # first block in file order wins ties
            self.start_line.setdefault(l.start_ms, l)  # type: ignore[arg-type]
            self.end_line.setdefault(l.end_ms, l)      # type: ignore[arg-type]
        self.starts = sorted(self.start_line)
        self.ends = sorted(self.end_line)
        self.dialogue = [l for l in self.ok if l.kind is LineKind.DIALOGUE]

    def straddling_dialogue(self, t: int) -> Optional[Line]:
        """A dialogue line that is in progress at time t (so cutting at t would split it)."""
        for d in self.dialogue:
            if d.start_ms < t < d.end_ms:  # type: ignore[operator]
                return d
        return None

    def valid_start(self, t: int) -> bool:
        return t in self.start_line and self.straddling_dialogue(t) is None

    def valid_end(self, t: int) -> bool:
        return t in self.end_line and self.straddling_dialogue(t) is None

    def prev_valid_start(self, t: int) -> int:
        """Greatest valid start boundary <= t."""
        for s in reversed(self.starts[: bisect_right(self.starts, t)]):
            if self.valid_start(s):
                return s
        return self.starts[0]

    def next_valid_end(self, t: int) -> int:
        """Smallest valid end boundary >= t."""
        for e in self.ends[bisect_left(self.ends, t):]:
            if self.valid_end(e):
                return e
        return self.ends[-1]

    def valid_starts_between(self, lo: int, hi: int) -> list[int]:
        return [s for s in self.starts[bisect_left(self.starts, lo): bisect_right(self.starts, hi)] if self.valid_start(s)]

    def crosses(self, t: int) -> bool:
        """True if ANY block (dialogue, camera, action, sound) is in progress at t, i.e. a cut at t splits it."""
        return any(l.start_ms < t < l.end_ms for l in self.ok)  # type: ignore[operator]

    def clean_valid_ends_between(self, lo: int, hi: int) -> list[int]:
        """Block ends in [lo, hi] where no block at all is cut (a 'complete line boundary')."""
        return [e for e in self.ends[bisect_left(self.ends, lo): bisect_right(self.ends, hi)] if not self.crosses(e)]


# --------------------------------------------------------------------------- resolver
class TimelineResolver:
    def __init__(self, index: ScreenplayIndex, config: Optional[TimelineConfig] = None) -> None:
        self.index = index
        self.cfg = config or DEFAULT_CONFIG
        self._eps: dict[str, _EpisodeTimeline] = {}

    def _episode(self, episode_id: str) -> _EpisodeTimeline:
        if episode_id not in self._eps:
            self._eps[episode_id] = _EpisodeTimeline(self.index.episodes[episode_id].lines)
        return self._eps[episode_id]

    def _boundary(self, ep: _EpisodeTimeline, episode_id: str, side: str, t: int, original: int, reason: Optional[str]) -> Boundary:
        line = (ep.start_line if side == "start" else ep.end_line)[t]
        raw = line.start_raw if side == "start" else line.end_raw
        return Boundary(side=side, episode_id=episode_id, time_ms=t, timecode=format_timecode(t),
                        line_id=line.line_id, source_file=line.source_file, source_line=line.source_line,
                        original_ms=original, adjusted=(t != original), adjustment_reason=reason if t != original else None)

    # ------------------------------------------------------------------ main entry
    def resolve(self, cand: StoredCandidate) -> TimelineOutcome:
        idx, cfg = self.index, self.cfg
        problems: list[str] = []          # hard failures that prevent building a timeline
        warnings: list[str] = []
        adjustments: list[Adjustment] = []

        # ---- 1. unadjusted windows from the selected blocks (never summing durations) ----
        work: list[dict] = []
        for i, seg in enumerate(cand.segments):
            ep = self._episode(seg.episode_id)
            a, b = idx.lines[seg.start_line_id], idx.lines[seg.end_line_id]
            selected = [l for l in idx.episodes[seg.episode_id].lines if a.seq <= l.seq <= b.seq]
            ok = [l for l in selected if l.timestamp_status is TimestampStatus.OK]
            bad = [l.line_id for l in selected if l.timestamp_status is not TimestampStatus.OK]
            if bad:
                warnings.append(f"segment {i + 1}: {len(bad)} block(s) without a valid timestamp were ignored ({', '.join(bad[:3])})")
            if not ok:
                problems.append(f"segment {i + 1} has no block with a valid timestamp")
                continue
            s0, e0 = min(l.start_ms for l in ok), max(l.end_ms for l in ok)   # type: ignore[type-var]
            work.append({"i": i, "seg": seg, "ep": ep, "selected": selected, "s0": s0, "e0": e0, "s": s0, "e": e0,
                         "s_reason": None, "e_reason": None, "s_kind": "", "e_kind": "", "e_detail": {}})
        if problems or not work:
            return self._fail(cand, problems or ["no usable segments"], gates_extra=[])

        original_total = sum(w["e0"] - w["s0"] for w in work)

        # ---- 2. snap edges onto block boundaries that do not cut dialogue ---------------
        for w in work:
            ep: _EpisodeTimeline = w["ep"]
            d = ep.straddling_dialogue(w["s"])
            if d is not None:
                new = ep.prev_valid_start(w["s"])
                w["s"], w["s_reason"], w["s_kind"] = new, f"start fell inside dialogue {d.line_id}; extended to include the whole line", "dialogue_snap"
            d = ep.straddling_dialogue(w["e"])
            if d is not None:
                new = ep.next_valid_end(w["e"])
                w["e"], w["e_reason"], w["e_kind"] = new, f"end fell inside dialogue {d.line_id}; extended to include the whole line", "dialogue_snap"

        # ---- 3. hook: measure, then (if needed) move the clip start ---------------------
        hook = idx.lines[cand.hook_line_id]
        first = work[0]
        hook_ok = hook.timestamp_status is TimestampStatus.OK
        hook_before = hook_after = None
        if not hook_ok:
            problems.append(f"hook line {hook.line_id} has no valid timestamp")
        elif hook.episode_id != first["seg"].episode_id:
            problems.append(f"hook line {hook.line_id} is not in the first segment's episode")
        else:
            hook_before = hook.start_ms - first["s0"]            # type: ignore[operator]
            offset = hook.start_ms - first["s"]                  # type: ignore[operator]
            if offset > cfg.hook_max_ms:
                ep = first["ep"]
                lo = max(hook.start_ms - cfg.hook_max_ms, first["s"])  # type: ignore[operator]
                options = ep.valid_starts_between(lo, hook.start_ms)
                preferred = [t for t in options if hook.start_ms - t <= cfg.hook_preferred_lead_ms]  # type: ignore[operator]
                pick = (preferred or options or [None])[0]       # earliest valid start inside the allowed lead-in
                if pick is not None:
                    first["s"], first["s_kind"] = pick, "hook_alignment"
                    first["s_reason"] = (f"hook {hook.line_id} was {_s(offset)} into the clip (limit {_s(cfg.hook_max_ms)}); "
                                         f"start moved to a block start {_s(hook.start_ms - pick)} before the hook")
            hook_after = hook.start_ms - first["s"]               # type: ignore[operator]

        # ---- 4. cliffhanger: must be inside the final window and (nearly) the end -------
        last = work[-1]
        cliff = idx.lines[cand.cliffhanger_line_id]
        cliff_ok = cliff.timestamp_status is TimestampStatus.OK and cliff.episode_id == last["seg"].episode_id
        if not cliff_ok:
            problems.append(f"cliffhanger line {cliff.line_id} has no valid timestamp or is not in the last segment's episode")
        elif cliff.start_ms >= last["s"] and cliff.end_ms <= last["e"] + 0:   # type: ignore[operator]
            slack = last["e"] - cliff.end_ms                       # type: ignore[operator]
            if slack > cfg.cliff_tail_max_ms:                     # exactly the limit is allowed; MORE than it triggers a trim
                ep = last["ep"]
                # Prefer a COMPLETE boundary: a block end within the allowed tail where no shot/line is cut at all.
                clean = ep.clean_valid_ends_between(cliff.end_ms, cliff.end_ms + cfg.cliff_tail_max_ms)  # type: ignore[operator]
                new_e = clean[0] if clean else ep.next_valid_end(cliff.end_ms)   # else: first end that at least keeps dialogue whole
                if new_e < last["e"]:
                    last["e"], last["e_kind"] = new_e, "cliffhanger_trim"
                    kind = "a complete boundary (no block cut)" if clean else "the first end that keeps every dialogue line whole"
                    last["e_reason"] = (f"{_s(slack)} of footage followed cliffhanger {cliff.line_id} (limit {_s(cfg.cliff_tail_max_ms)}); "
                                        f"end trimmed to {kind}, leaving {_s(new_e - cliff.end_ms)} after it")
                    last["e_detail"] = {"trailing_before_ms": slack, "trailing_after_ms": new_e - cliff.end_ms,
                                        "complete_boundary": bool(clean), "limit_ms": cfg.cliff_tail_max_ms}

        # ---- 5. build segment objects, adjustments, integrity checks ------------------
        segments: list[SegmentTimeline] = []
        for w in work:
            ep = w["ep"]
            seg = w["seg"]
            s, e = w["s"], w["e"]
            if e <= s:
                problems.append(f"segment {w['i'] + 1} collapsed to zero length after adjustment")
                continue
            sb = self._boundary(ep, seg.episode_id, "start", s, w["s0"], w["s_reason"])
            eb = self._boundary(ep, seg.episode_id, "end", e, w["e0"], w["e_reason"])
            for b, frm, kind, detail in ((sb, w["s0"], w["s_kind"], {}), (eb, w["e0"], w["e_kind"], w["e_detail"])):
                if b.adjusted:
                    adjustments.append(Adjustment(segment_index=w["i"] + 1, side=b.side, kind=kind, from_ms=frm, to_ms=b.time_ms,
                                                  new_boundary_line_id=b.line_id, reason=b.adjustment_reason or "", detail=detail))
            in_window = [l for l in ep.ok if l.end_ms > s and l.start_ms < e]          # type: ignore[operator]
            partial = [l.line_id for l in in_window if l.kind is not LineKind.DIALOGUE
                       and (l.start_ms < s < l.end_ms or l.start_ms < e < l.end_ms)]   # type: ignore[operator]
            cut_dialogue = [l.line_id for l in ep.dialogue if l.start_ms < s < l.end_ms or l.start_ms < e < l.end_ms]  # type: ignore[operator]
            dropped = [l.line_id for l in w["selected"] if l.timestamp_status is TimestampStatus.OK
                       and (l.end_ms <= s or l.start_ms >= e)]                          # type: ignore[operator]
            segments.append(SegmentTimeline(
                segment_index=w["i"] + 1, episode_id=seg.episode_id, start_line_id=seg.start_line_id,
                end_line_id=seg.end_line_id, selected_line_count=len(w["selected"]),
                selected_window_ms=w["e0"] - w["s0"], start=sb, end=eb, duration_ms=e - s,
                lines_in_window=len(in_window), dropped_selected_line_ids=dropped,
                partially_cut_line_ids=partial, cut_dialogue_line_ids=cut_dialogue))

        if not segments:
            return self._fail(cand, problems or ["no usable segments"], gates_extra=[])

        # segments in the same episode must not play the same footage twice
        overlaps: list[str] = []
        for x in range(len(segments)):
            for y in range(x + 1, len(segments)):
                p, q = segments[x], segments[y]
                if p.episode_id == q.episode_id and p.start.time_ms < q.end.time_ms and q.start.time_ms < p.end.time_ms:
                    overlaps.append(f"segments {p.segment_index} and {q.segment_index} overlap on the timeline")

        total = sum(sg.duration_ms for sg in segments)
        first_seg, last_seg = segments[0], segments[-1]
        cliff_inside = bool(cliff_ok and cliff.start_ms >= last_seg.start.time_ms and cliff.end_ms <= last_seg.end.time_ms)  # type: ignore[operator]
        tail = (last_seg.end.time_ms - cliff.end_ms) if cliff_ok else None  # type: ignore[operator]
        hook_inside = bool(hook_ok and hook.episode_id == first_seg.episode_id
                           and first_seg.start.time_ms <= hook.start_ms < first_seg.end.time_ms)  # type: ignore[operator]
        hook_offset = (hook.start_ms - first_seg.start.time_ms) if hook_ok else None  # type: ignore[operator]

        clip = ClipTimeline(
            candidate_id=cand.candidate_id, segments=segments, total_duration_ms=total,
            original_total_duration_ms=original_total,
            hook=HookCheck(line_id=hook.line_id, episode_id=hook.episode_id, start_ms=hook.start_ms or 0,
                           offset_before_ms=hook_before, offset_ms=hook_offset,
                           within_limit=bool(hook_inside and hook_offset is not None and 0 <= hook_offset <= cfg.hook_max_ms)),
            cliffhanger=CliffhangerCheck(line_id=cliff.line_id, end_ms=cliff.end_ms, inside_final_window=cliff_inside,
                                         tail_slack_ms=tail),
            adjustments=adjustments, warnings=warnings)

        gates = self._gates(clip, problems)
        gates.append(GateResult(gate="segments_distinct", passed=not overlaps,
                                reason="segments cover different footage" if not overlaps else "; ".join(overlaps)))
        rejected = [g for g in gates if not g.passed]
        return TimelineOutcome(candidate_id=cand.candidate_id, window_id=cand.window_id, accepted=not rejected,
                               clip=clip, gates=gates, rejection_reasons=[f"{g.gate}: {g.reason}" for g in rejected])

    # ------------------------------------------------------------------ gates
    def _gates(self, clip: ClipTimeline, problems: list[str]) -> list[GateResult]:
        total = clip.total_duration_ms
        gates: list[GateResult] = [GateResult(
            gate="timestamps_valid", passed=not problems,
            reason="all referenced blocks needed for timing have valid timestamps" if not problems else "; ".join(problems))]

        cfg = self.cfg
        ok_len = cfg.min_clip_ms <= total <= cfg.max_clip_ms
        why = f"clip is {_s(total)}"
        if clip.original_total_duration_ms != total:
            why += f" after boundary adjustment (was {_s(clip.original_total_duration_ms)} as proposed)"
        if not ok_len:
            why += f"; must be {_s(cfg.min_clip_ms)}-{_s(cfg.max_clip_ms)} inclusive"
            if any(a.kind == "cliffhanger_trim" for a in clip.adjustments):
                why += " (the cliffhanger trim shortened it below the minimum; narrative-safe extension is not attempted)" if total < cfg.min_clip_ms else ""
        gates.append(GateResult(gate="duration_20_60s", passed=ok_len, reason=why,
                                details={"total_ms": total, "proposed_total_ms": clip.original_total_duration_ms,
                                         "segment_ms": [s.duration_ms for s in clip.segments]}))

        h = clip.hook
        if h.offset_ms is None or not h.within_limit:
            why = (f"hook {h.line_id} starts {_s(h.offset_ms)} into the clip (limit {_s(cfg.hook_max_ms)}) and no valid earlier-start "
                   f"adjustment exists" if h.offset_ms is not None else f"hook {h.line_id} cannot be placed on the timeline")
        else:
            why = f"hook {h.line_id} starts {_s(h.offset_ms)} into the clip"
            if h.offset_before_ms is not None and h.offset_before_ms > cfg.hook_max_ms:
                why += f" (was {_s(h.offset_before_ms)} before the start boundary was moved)"
        gates.append(GateResult(gate="hook_timing_3s", passed=h.within_limit, reason=why,
                                details={"offset_ms": h.offset_ms, "offset_before_ms": h.offset_before_ms}))

        c = clip.cliffhanger
        cliff_ok = c.inside_final_window and c.tail_slack_ms is not None and 0 <= c.tail_slack_ms <= cfg.cliff_tail_max_ms
        trim = next((a for a in clip.adjustments if a.kind == "cliffhanger_trim"), None)
        if cliff_ok:
            why = f"cliffhanger line {c.line_id} ends {_s(c.tail_slack_ms)} before the clip end"   # type: ignore[arg-type]
            if trim:
                why += f" (after trimming {_s(trim.detail.get('trailing_before_ms', 0))} of trailing footage)"
        elif not c.inside_final_window:
            why = f"cliffhanger line {c.line_id} is not inside the final clip window"
        else:
            why = (f"{_s(c.tail_slack_ms)} of footage follows cliffhanger line {c.line_id} (limit {_s(cfg.cliff_tail_max_ms)}) "   # type: ignore[arg-type]
                   f"and it cannot be trimmed to a valid boundary within the limit without cutting dialogue")
        gates.append(GateResult(gate="cliffhanger_reference", passed=cliff_ok, reason=why,
                                details={"tail_slack_ms": c.tail_slack_ms, "limit_ms": cfg.cliff_tail_max_ms,
                                         "trimmed": trim is not None}))

        cuts = [d for s in clip.segments for d in s.cut_dialogue_line_ids]
        gates.append(GateResult(gate="boundary_integrity", passed=not cuts,
                                reason="no dialogue line is cut by a clip boundary" if not cuts else f"boundary cuts dialogue: {', '.join(cuts)}"))
        return gates

    # ------------------------------------------------------------------ failure helper
    def _fail(self, cand: StoredCandidate, problems: list[str], gates_extra: list[GateResult]) -> TimelineOutcome:
        g = GateResult(gate="timestamps_valid", passed=False, reason="; ".join(problems))
        return TimelineOutcome(candidate_id=cand.candidate_id, window_id=cand.window_id, accepted=False, clip=None,
                               gates=[g, *gates_extra], rejection_reasons=[f"timestamps_valid: {g.reason}"])
