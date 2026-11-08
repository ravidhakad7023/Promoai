"""Phase 2 orchestrator: story understanding + promo candidate detection.

    python -m promoai.detect --dry-run                       # sizes only, no API, no key needed
    python -m promoai.detect --episodes 1-5 --window-size 3  # small live sample

Flow per window of consecutive episodes::

    previous StoryContext + rendered episodes --LLM--> updated StoryContext + candidates
                                                         |            |
                                          merge_context (Python)   validate every ID (Python)

Outputs go to ``data/output/<series>/phase2/`` (git-ignored: they contain screenplay-
derived text).  Candidates are *proposals*: ``status`` stays ``llm_proposed``.
Durations / hook timing / cliffhanger checks are Phase 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import ConfigError, load_settings
from .index import ScreenplayIndex
from .llm import (
    LLMBlockedError,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    LLMResponseError,
    create_provider,
    generate_validated,
)
from .models import Episode
from .prompts import PROMPT_VERSION, build_prompt, estimate_tokens
from .schemas import (
    DetectionResponse,
    PromoCandidate,
    RejectedCandidate,
    ResolvedSegment,
    StoredCandidate,
    StoryContext,
)
from .story import merge_context

log = logging.getLogger("promoai.detect")

MAX_CONSECUTIVE_FAILURES = 2
DEFAULT_MAX_CANDIDATES = 4      # candidates requested per window (the long-standing default)


# --------------------------------------------------------------------------- windows
def plan_windows(episode_ids: Sequence[str], size: int, overlap: int = 1) -> list[list[str]]:
    """Split ordered episode IDs into windows of ``size`` that share ``overlap`` episodes.

    The overlap guarantees every pair of neighbouring episodes appears together in at
    least one window, so cross-episode clips are possible."""
    if size < 1:
        raise ValueError("window size must be >= 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    ids = list(episode_ids)
    if not ids:
        return []
    stride, windows, start = size - overlap, [], 0
    while True:
        windows.append(ids[start : start + size])
        if start + size >= len(ids):
            return windows
        start += stride


# --------------------------------------------------------------------------- validation
def _collect_dropped(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dropped_fields" and isinstance(v, list):
                out.extend(str(x) for x in v)
            else:
                out.extend(_collect_dropped(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_dropped(v))
    return sorted(set(out))


def validate_candidate(
    cand: PromoCandidate, position: int, window_id: str, window_episode_ids: set[str], index: ScreenplayIndex
) -> StoredCandidate | RejectedCandidate:
    """Check every ID the LLM returned against the real screenplay.

    Deliberately NOT done here: any duration / timing check (overlapping blocks make
    that a Phase-3 timeline problem).  We never "repair" IDs - an invalid reference
    means the candidate is rejected and the reason recorded."""
    reasons: list[str] = []
    warnings: list[str] = []
    resolved: list[ResolvedSegment] = []

    def known(line_id: str, what: str) -> bool:
        if line_id not in index.lines:
            reasons.append(f"{what} {line_id} does not exist")
            return False
        ep = index.lines[line_id].episode_id
        if ep not in window_episode_ids:
            reasons.append(f"{what} {line_id} is in {ep}, outside this window")
            return False
        return True

    for i, seg in enumerate(cand.segments, 1):
        ok_s, ok_e = known(seg.start_line_id, f"segment {i} start"), known(seg.end_line_id, f"segment {i} end")
        if not (ok_s and ok_e):
            continue
        a, b = index.lines[seg.start_line_id], index.lines[seg.end_line_id]
        if a.episode_id != b.episode_id:
            reasons.append(f"segment {i} spans two episodes ({a.episode_id}, {b.episode_id}); use one segment per episode")
        elif a.seq > b.seq:
            reasons.append(f"segment {i} start {a.line_id} comes after end {b.line_id}")
        else:
            resolved.append(ResolvedSegment(
                start_line_id=a.line_id, end_line_id=b.line_id, episode_id=a.episode_id,
                start_scene_id=a.scene_id, end_scene_id=b.scene_id, n_lines=b.seq - a.seq + 1))

    if not reasons and len(resolved) == len(cand.segments):
        # no two segments may overlap each other
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                x, y = resolved[i], resolved[j]
                if x.episode_id == y.episode_id:
                    xs, xe = index.lines[x.start_line_id].seq, index.lines[x.end_line_id].seq
                    ys, ye = index.lines[y.start_line_id].seq, index.lines[y.end_line_id].seq
                    if xs <= ye and ys <= xe:
                        reasons.append(f"segments {i + 1} and {j + 1} overlap")

        def inside(line_id: str, seg: ResolvedSegment, what: str) -> None:
            if line_id not in index.lines:
                reasons.append(f"{what} {line_id} does not exist")
                return
            ln = index.lines[line_id]
            lo, hi = index.lines[seg.start_line_id].seq, index.lines[seg.end_line_id].seq
            if ln.episode_id != seg.episode_id or not lo <= ln.seq <= hi:
                reasons.append(f"{what} {line_id} is not inside the {'first' if what.startswith('hook') else 'last'} segment")

        inside(cand.hook_line_id, resolved[0], "hook_line_id")
        inside(cand.cliffhanger_line_id, resolved[-1], "cliffhanger_line_id")

        # The hook may be ANY line of the first segment (CAMERA/ACTION blocks can precede
        # dialogue), so its position in file order is deliberately not constrained.
        # Whether it is really within 3 s of the clip start is a timeline question: Phase 3.
        if len(resolved) == 1 and cand.hook_line_id == cand.cliffhanger_line_id:
            warnings.append("hook and cliffhanger are the same line")
        elif len(resolved) == 1 and not reasons:
            h, c = index.lines[cand.hook_line_id], index.lines[cand.cliffhanger_line_id]
            if h.seq > c.seq:   # file order is not a timeline, so only flag - never reject
                warnings.append("hook is listed after the cliffhanger in file order; check on the timeline")

        order = [(index.episodes[s.episode_id].episode_number, index.lines[s.start_line_id].seq) for s in resolved]
        if order != sorted(order):
            warnings.append("segments are not in story order (earlier material comes after later material)")

    if len(cand.open_question.split()) < 4:
        reasons.append("open_question is missing or too vague to name a specific unresolved question")

    if reasons:
        return RejectedCandidate(window_id=window_id, llm_candidate_id=cand.candidate_id, reasons=sorted(set(reasons)))

    return StoredCandidate(
        candidate_id=f"{window_id}-C{position:02d}",
        llm_candidate_id=cand.candidate_id,
        window_id=window_id,
        segments=resolved,
        cross_episode=len({s.episode_id for s in resolved}) > 1,
        hook_line_id=cand.hook_line_id,
        hook_description=cand.hook_description,
        cliffhanger_line_id=cand.cliffhanger_line_id,
        cliffhanger_description=cand.cliffhanger_description,
        open_question=cand.open_question,
        why_it_works=cand.why_it_works,
        tags=cand.tags,
        llm_gate_claims=cand.gates,
        dropped_fields=_collect_dropped(cand.model_dump()),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- run
@dataclass
class WindowResult:
    window_id: str
    episode_ids: list[str]
    status: str = "pending"                  # ok | failed | skipped
    prompt_chars: int = 0
    estimated_prompt_tokens: int = 0
    attempts: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    candidates_returned: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    context_notes: list[str] = field(default_factory=list)
    dropped_fields: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DetectionRun:
    windows: list[WindowResult] = field(default_factory=list)
    accepted: list[StoredCandidate] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    contexts: dict[str, StoryContext] = field(default_factory=dict)   # window_id -> context after it
    final_context: Optional[StoryContext] = None
    stopped_early: Optional[str] = None


def run_detection(
    index: ScreenplayIndex,
    provider: LLMProvider,
    episode_ids: Sequence[str],
    *,
    window_size: int = 6,
    overlap: int = 1,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> DetectionRun:
    run = DetectionRun()
    context: Optional[StoryContext] = None
    seen_signatures: dict[tuple, str] = {}
    plans = plan_windows(episode_ids, window_size, overlap)
    consecutive_failures = 0

    for n, ids in enumerate(plans, 1):
        wid = f"W{n:02d}"
        wr = WindowResult(window_id=wid, episode_ids=ids)
        run.windows.append(wr)
        if run.stopped_early:
            wr.status, wr.error = "skipped", run.stopped_early
            continue

        episodes = [index.episodes[i] for i in ids]
        prompt = build_prompt(context, episodes, max_candidates=max_candidates)
        wr.prompt_chars, wr.estimated_prompt_tokens = len(prompt), estimate_tokens(prompt)

        before = provider.stats.as_dict()
        try:
            resp, wr.attempts = generate_validated(provider, prompt, DetectionResponse)
        except LLMQuotaError as exc:
            wr.status, wr.error = "failed", "quota_exhausted"
            run.stopped_early = f"stopped at {wid}: {exc}"
            log.error("%s", run.stopped_early)
            _fill_usage(wr, before, provider)
            continue
        except (LLMResponseError, LLMBlockedError, LLMError) as exc:
            wr.status, wr.error = "failed", type(exc).__name__
            _fill_usage(wr, before, provider)
            consecutive_failures += 1
            log.error("%s failed: %s", wid, type(exc).__name__)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                run.stopped_early = f"stopped after {consecutive_failures} consecutive window failures"
            continue
        consecutive_failures = 0
        _fill_usage(wr, before, provider)

        context, wr.context_notes = merge_context(context, resp.story_context, ids[-1])
        run.contexts[wid] = context
        wr.dropped_fields = _collect_dropped(resp.model_dump())

        wr.candidates_returned = len(resp.candidates)
        window_set = set(ids)
        for pos, cand in enumerate(resp.candidates, 1):
            result = validate_candidate(cand, pos, wid, window_set, index)
            if isinstance(result, RejectedCandidate):
                run.rejected.append(result)
                continue
            sig = tuple((s.start_line_id, s.end_line_id) for s in result.segments)
            if sig in seen_signatures:      # same clip proposed again in an overlapping window
                run.rejected.append(RejectedCandidate(
                    window_id=wid, llm_candidate_id=cand.candidate_id,
                    reasons=[f"exact duplicate of {seen_signatures[sig]}"]))
                continue
            seen_signatures[sig] = result.candidate_id
            run.accepted.append(result)
        wr.candidates_accepted = sum(1 for c in run.accepted if c.window_id == wid)
        wr.candidates_rejected = sum(1 for c in run.rejected if c.window_id == wid)
        wr.status = "ok"
        log.info("%s ok: %d returned, %d accepted", wid, wr.candidates_returned, wr.candidates_accepted)

    run.final_context = context
    return run


def _fill_usage(wr: WindowResult, before: dict, provider: LLMProvider) -> None:
    after = provider.stats.as_dict()
    for k in ("api_calls", "cache_hits", "prompt_tokens", "output_tokens", "thinking_tokens"):
        setattr(wr, k, after[k] - before[k])


# --------------------------------------------------------------------------- outputs
def write_outputs(
    run: DetectionRun, provider: Optional[LLMProvider], series_dir: Path, meta: dict[str, Any], run_name: Optional[str] = None
) -> Path:
    """Write a run's files.

    With ``run_name`` they go to ``phase2/runs/<run_name>/`` and an existing run folder is
    never overwritten, so earlier results (including the unversioned v1 files directly
    under ``phase2/``) stay intact."""
    out = series_dir / "phase2" / "runs" / run_name if run_name else series_dir / "phase2"
    if run_name and out.exists():
        raise FileExistsError(f"run folder already exists: {out} (pick another --run-name)")
    (out / "story_context_history").mkdir(parents=True, exist_ok=True)
    for wid, ctx in run.contexts.items():
        (out / "story_context_history" / f"{wid}.json").write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    if run.final_context is not None:
        (out / "story_context.json").write_text(run.final_context.model_dump_json(indent=2), encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps({"accepted": [c.model_dump(mode="json") for c in run.accepted],
                    "rejected": [r.model_dump(mode="json") for r in run.rejected]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    report = {
        **meta,
        "usage": provider.stats.as_dict() if provider else None,
        "windows": [w.__dict__ for w in run.windows],
        "totals": {
            "windows": len(run.windows),
            "windows_ok": sum(w.status == "ok" for w in run.windows),
            "windows_failed": sum(w.status == "failed" for w in run.windows),
            "windows_skipped": sum(w.status == "skipped" for w in run.windows),
            "candidates_returned": sum(w.candidates_returned for w in run.windows),
            "candidates_accepted": len(run.accepted),
            "candidates_rejected": len(run.rejected),
        },
        "stopped_early": run.stopped_early,
    }
    (out / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- CLI
def parse_episode_selection(spec: str, available: Sequence[int]) -> list[int]:
    """'all' | '1-5' | '1,3,7-9'  ->  sorted episode numbers that exist."""
    if spec.strip().lower() == "all":
        return sorted(available)
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        elif part:
            wanted.add(int(part))
    return sorted(n for n in wanted if n in set(available))


def _find_series_dir(root: Path) -> Path:
    dirs = sorted(p for p in root.iterdir() if (p / "episodes").is_dir()) if root.is_dir() else []
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected exactly one parsed series in {root} (found {len(dirs)}); run `python -m promoai.pipeline` or pass --series-dir")
    return dirs[0]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PromoAI phase 2: story context + promo candidate detection")
    ap.add_argument("--series-dir", type=Path, help="parsed series folder (default: the only one under data/output)")
    ap.add_argument("--episodes", default="all", help="'all', '1-5' or '1,3,7-9' (episode numbers)")
    ap.add_argument("--window-size", type=int, default=6)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    ap.add_argument("--dry-run", action="store_true", help="build prompts and print sizes; no API call, no key needed")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--run-name", help="output folder name under phase2/runs/ (default: <prompt version>-<timestamp>)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        series_dir = args.series_dir or _find_series_dir(Path("data/output"))
        index = ScreenplayIndex.from_output_dir(series_dir)
        nums = parse_episode_selection(args.episodes, [e.episode_number for e in index.episodes.values()])
        if not nums:
            print("error: no episodes match --episodes", file=sys.stderr)
            return 2
        ep_ids = [e.episode_id for e in sorted(index.episodes.values(), key=lambda e: e.episode_number) if e.episode_number in nums]
        plans = plan_windows(ep_ids, args.window_size, args.overlap)

        if args.dry_run:
            tot = 0
            print(f"{len(ep_ids)} episodes -> {len(plans)} windows (size {args.window_size}, overlap {args.overlap}); dry run, no API calls")
            for n, ids in enumerate(plans, 1):
                p = build_prompt(None, [index.episodes[i] for i in ids], max_candidates=args.max_candidates)
                t = estimate_tokens(p)
                tot += t
                print(f"  W{n:02d} {ids[0]}..{ids[-1]}  prompt ~{t:,} tokens ({len(p):,} chars, before story notes grow)")
            print(f"  total ~{tot:,} input tokens (rough chars/4 estimate; story notes add ~1-2k per call after the first)")
            return 0

        run_name = args.run_name or f"{PROMPT_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}"
        if (series_dir / "phase2" / "runs" / run_name).exists():     # fail BEFORE spending API calls
            raise FileExistsError(f"run folder already exists: {run_name} (pick another --run-name)")
        settings = load_settings()
        provider = create_provider(settings, cache=not args.no_cache)
        print(f"provider={settings.provider} model={settings.model} windows={len(plans)} episodes={len(ep_ids)}")
        run = run_detection(index, provider, ep_ids, window_size=args.window_size, overlap=args.overlap,
                            max_candidates=args.max_candidates)
        out = write_outputs(run, provider, series_dir, {
            "prompt_version": PROMPT_VERSION, "provider": settings.provider, "model": settings.model,
            "temperature": settings.temperature, "window_size": args.window_size, "overlap": args.overlap,
            "max_candidates": args.max_candidates, "episodes": ep_ids}, run_name)
    except (ConfigError, FileNotFoundError, FileExistsError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    s = provider.stats
    print(f"\nwindows ok/failed/skipped: {sum(w.status=='ok' for w in run.windows)}/{sum(w.status=='failed' for w in run.windows)}/{sum(w.status=='skipped' for w in run.windows)}")
    print(f"api calls: {s.api_calls}  cache hits: {s.cache_hits}  json retries: {s.json_retries}")
    print(f"tokens: prompt {s.prompt_tokens:,}  output {s.output_tokens:,}  thinking {s.thinking_tokens:,}  total {s.total_tokens:,}")
    print(f"candidates: accepted {len(run.accepted)}  rejected {len(run.rejected)}  (accepted = IDs valid, NOT quality-approved)")
    if s.failures:
        print(f"failures recorded: {len(s.failures)} -> see run_report.json")
    if run.stopped_early:
        print(f"STOPPED EARLY: {run.stopped_early}")
    print(f"report: {out / 'run_report.json'}")
    return 0 if not run.stopped_early and all(w.status == "ok" for w in run.windows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
