"""Phase 3 orchestrator: deterministic validation, semantic gates, ranking.

    python -m promoai.validate --phase2-run v2-sample-e001-e005               # deterministic only, NO API calls
    python -m promoai.validate --phase2-run v2-sample-e001-e005 --semantic    # + LLM rubric assessments + ranking

Reads  : data/output/<series>/phase2/runs/<run>/{candidates.json, story_context_history/}
Writes : data/output/<series>/phase3/<out-name>/   (git-ignored; an existing folder is never overwritten)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .config import ConfigError, load_settings
from .index import ScreenplayIndex
from .llm import LLMError, LLMProvider, LLMQuotaError, create_provider
from .ranking import (
    DEFAULT_MAX_PER_EPISODE, DEFAULT_MAX_PROMOS, Decision, Evaluated, Selection, drop_exact_duplicates, select_promos,
)
from .schemas import StoredCandidate, StoryContext
from .semantic import SEMANTIC_PROMPT_VERSION, aggregate_votes, assess_batch, semantic_gates, trim_adjustment
from .story import sanitize_context
from .timeline import DEFAULT_CONFIG, TimelineConfig, TimelineOutcome, TimelineResolver

log = logging.getLogger("promoai.validate")

ASSESSMENT_DISCLAIMER = (
    "Deterministic gates (duration, hook timing, cliffhanger reference, boundary integrity) are enforced by Python on the real "
    "timeline. Semantic gates (hook strength, cliffhanger strength, standalone clarity) and the qualitative tiers are "
    "assessments by a language model against a written rubric - opinions with stated reasons, not objective facts."
)


def _find_series_dir(root: Path) -> Path:
    dirs = sorted(p for p in root.iterdir() if (p / "episodes").is_dir()) if root.is_dir() else []
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected exactly one parsed series in {root} (found {len(dirs)}); pass --series-dir")
    return dirs[0]


def load_phase2_run(series_dir: Path, run_name: str) -> tuple[list[StoredCandidate], dict[str, StoryContext], dict[str, Any]]:
    """Candidates + per-window story contexts + the run report of a Phase 2 run."""
    run_dir = series_dir / "phase2" / "runs" / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Phase 2 run not found: {run_dir}")
    raw = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    candidates = [StoredCandidate.model_validate(c) for c in raw["accepted"]]
    contexts = {p.stem: StoryContext.model_validate_json(p.read_text(encoding="utf-8"))
                for p in sorted((run_dir / "story_context_history").glob("W*.json"))}
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    return candidates, contexts, report


def run_deterministic(index: ScreenplayIndex, candidates: list[StoredCandidate],
                      config: Optional[TimelineConfig] = None) -> list[TimelineOutcome]:
    resolver = TimelineResolver(index, config)
    return [resolver.resolve(c) for c in candidates]


# --------------------------------------------------------------------------- before / after
def before_after(c: StoredCandidate, o: TimelineOutcome) -> dict[str, Any]:
    """What the LLM proposed/claimed vs what Python measured and decided."""
    clip = o.clip
    claims = c.llm_gate_claims
    return {
        "candidate_id": c.candidate_id,
        "original_candidate": c.model_dump(mode="json"),     # the Phase 2 record, preserved untouched
        "before_llm_proposal": {
            "segments": [{"start_line_id": s.start_line_id, "end_line_id": s.end_line_id, "episode_id": s.episode_id} for s in c.segments],
            "hook_line_id": c.hook_line_id,
            "cliffhanger_line_id": c.cliffhanger_line_id,
            "claimed_duration_ok": claims.duration_plausible_20_60s,
            "claimed_hook_within_3s": claims.hook_in_first_3s,
            "claimed_unresolved_cliffhanger": claims.unresolved_cliffhanger,
            "claimed_standalone_clear": claims.standalone_clear,
        },
        "measured_as_proposed": None if clip is None else {
            "total_duration_s": clip.original_total_duration_ms / 1000,
            "hook_offset_s": None if clip.hook.offset_before_ms is None else clip.hook.offset_before_ms / 1000,
        },
        "after_python": None if clip is None else {
            "accepted": o.accepted,
            "total_duration_s": clip.total_duration_ms / 1000,
            "hook_offset_s": None if clip.hook.offset_ms is None else clip.hook.offset_ms / 1000,
            "windows": [{"episode": s.episode_id, "start": s.start.timecode, "end": s.end.timecode,
                         "duration_s": s.duration_ms / 1000} for s in clip.segments],
            "adjustments": [a.reason for a in clip.adjustments],
            "adjustments_detail": [a.model_dump() for a in clip.adjustments],
        },
        "gates": [{"gate": g.gate, "passed": g.passed, "reason": g.reason} for g in o.gates],
        "rejection_reasons": o.rejection_reasons,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        b, m, a = r["before_llm_proposal"], r["measured_as_proposed"], r["after_python"]
        print(f"\n{r['candidate_id']}")
        print(f"  LLM claimed : duration ok={b['claimed_duration_ok']}  hook<=3s={b['claimed_hook_within_3s']}")
        if m:
            hk = f", hook at {m['hook_offset_s']:.1f}s" if m["hook_offset_s"] is not None else ""
            print(f"  measured    : {m['total_duration_s']:.1f}s as proposed{hk}")
        if a:
            ho = f"{a['hook_offset_s']:.1f}s" if a["hook_offset_s"] is not None else "n/a"
            print(f"  after Python: {'PASSES deterministic gates' if a['accepted'] else 'REJECTED'}  {a['total_duration_s']:.1f}s, hook at {ho}")
            for adj in a["adjustments"]:
                print(f"     adjusted : {adj}")
        for reason in r["rejection_reasons"]:
            print(f"     reject   : {reason}")


# --------------------------------------------------------------------------- semantic stage
def run_semantic(
    provider: LLMProvider,
    index: ScreenplayIndex,
    items: list[Evaluated],
    contexts: dict[str, StoryContext],
    series_length: int,
    batch_size: int,
    judge_votes: int = 1,
    salt_prefix: str = "",
) -> dict[str, Any]:
    """Assess every deterministic-pass candidate (batched per window).  Mutates ``items``.

    ``judge_votes == 1`` is the plain single-judge behaviour.  With more votes every batch is judged
    ``judge_votes`` independent times and each gate is decided by strict majority (see aggregate_votes)."""
    todo = [e for e in items if e.outcome.accepted and e.clip is not None]
    by_window: dict[str, list[Evaluated]] = {}
    for e in todo:
        by_window.setdefault(e.cand.window_id, []).append(e)

    info: dict[str, Any] = {"batches": [], "problems": [], "context_notes": {}, "stopped": None, "judge_votes": judge_votes}
    for wid, group in sorted(by_window.items()):
        ctx = contexts.get(wid)
        if ctx is not None:
            ctx, notes = sanitize_context(ctx)
            info["context_notes"][wid] = notes
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            if info["stopped"]:
                info["batches"].append({"window": wid, "candidates": [e.cid for e in chunk], "status": "skipped"})
                continue
            before = provider.stats.as_dict()
            if judge_votes > 1:
                _vote_batch(provider, index, chunk, ctx, series_length, judge_votes, info, wid, before, salt_prefix)
                continue
            try:
                got, problems = assess_batch(provider, index, [(e.cand, e.clip) for e in chunk], ctx, series_length)  # type: ignore[misc]
            except LLMQuotaError as exc:
                info["stopped"] = f"quota exhausted: {exc}"
                info["batches"].append({"window": wid, "candidates": [e.cid for e in chunk], "status": "failed", "error": "quota_exhausted"})
                continue
            except LLMError as exc:
                info["batches"].append({"window": wid, "candidates": [e.cid for e in chunk], "status": "failed", "error": type(exc).__name__})
                continue
            after = provider.stats.as_dict()
            for e in chunk:
                a = got.get(e.cid)
                if a is not None:
                    e.assessment = a
                    e.semantic_gates = semantic_gates(a, trimmed=trim_adjustment(e.clip) is not None)  # type: ignore[arg-type]
            info["problems"].extend(problems)
            info["batches"].append({"window": wid, "candidates": [e.cid for e in chunk], "status": "ok",
                                    "api_calls": after["api_calls"] - before["api_calls"],
                                    "cache_hits": after["cache_hits"] - before["cache_hits"],
                                    "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                                    "output_tokens": after["output_tokens"] - before["output_tokens"]})
    return info


def vote_providers(provider: LLMProvider, n: int, salt_prefix: str = "") -> list[LLMProvider]:
    """n providers that answer the SAME prompt independently.

    A caching provider gets its own cache salt for EVERY vote, ``<salt_prefix>vote<k>`` (k = 1..n), so no vote can
    replay another vote's answer or an older unsalted entry.  The salt is fully determined by the prefix and the vote
    index, so re-creating the providers (e.g. when a run resumes with the same prefix) finds the same entries again.
    A non-caching provider already samples independently."""
    salted = getattr(provider, "salted", None)
    if salted is None:
        return [provider] * n
    return [salted(f"{salt_prefix}vote{k}") for k in range(1, n + 1)]


def _vote_batch(provider: LLMProvider, index: ScreenplayIndex, chunk: list[Evaluated], ctx: Optional[StoryContext],
                series_length: int, n: int, info: dict[str, Any], wid: str, before: dict[str, Any], salt_prefix: str = "") -> None:
    samples: list[dict[str, Any]] = []          # per vote: {cid: assessment}
    errors: list[str] = []
    for k, p in enumerate(vote_providers(provider, n, salt_prefix), 1):
        if info["stopped"]:
            samples.append({})
            continue
        try:
            got, problems = assess_batch(p, index, [(e.cand, e.clip) for e in chunk], ctx, series_length)  # type: ignore[misc]
            info["problems"].extend(f"vote {k}: {x}" for x in problems)
            samples.append(got)
        except LLMQuotaError as exc:
            info["stopped"] = f"quota exhausted: {exc}"
            errors.append(f"vote {k}: quota_exhausted")
            samples.append({})
        except LLMError as exc:
            errors.append(f"vote {k}: {type(exc).__name__}")
            samples.append({})
    for e in chunk:
        e.votes = [s.get(e.cid) for s in samples]
        agg, gates = aggregate_votes(e.votes, trimmed=trim_adjustment(e.clip) is not None)  # type: ignore[arg-type]
        if agg is not None:
            e.assessment, e.semantic_gates = agg, gates
    after = provider.stats.as_dict()
    info["batches"].append({"window": wid, "candidates": [e.cid for e in chunk], "status": "ok" if not errors else "partial",
                            "judge_votes": n, "vote_errors": errors,
                            "api_calls": after["api_calls"] - before["api_calls"],
                            "cache_hits": after["cache_hits"] - before["cache_hits"],
                            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                            "output_tokens": after["output_tokens"] - before["output_tokens"]})


# --------------------------------------------------------------------------- records
def _line_view(index: ScreenplayIndex, line_id: str) -> dict[str, Any]:
    ln = index.lines[line_id]
    return {"line_id": line_id, "kind": ln.kind.value, "speaker": ln.speaker, "text": ln.text}


def promo_record(index: ScreenplayIndex, ev: Evaluated, rank: Optional[int], decision: Decision) -> dict[str, Any]:
    clip, a = ev.clip, ev.assessment
    assert clip is not None
    return {
        "rank": rank,
        "candidate_id": ev.cid,
        "status": decision.status,
        "decision_stage": decision.stage,
        "decision_reasons": decision.reasons,
        "clip": {
            "total_duration_s": clip.total_duration_ms / 1000,
            "segments": [{
                "episode_id": s.episode_id,
                "episode_number": index.episodes[s.episode_id].episode_number,
                "start": s.start.model_dump(), "end": s.end.model_dump(), "duration_s": s.duration_ms / 1000,
                "llm_line_range": [s.start_line_id, s.end_line_id],
                "partially_cut_line_ids": s.partially_cut_line_ids,
            } for s in clip.segments],
            "cross_episode": len(clip.episode_ids) > 1,
            "multi_segment": len(clip.segments) > 1,
        },
        "hook": {**_line_view(index, ev.cand.hook_line_id), "offset_s": None if clip.hook.offset_ms is None else clip.hook.offset_ms / 1000},
        "cliffhanger": {**_line_view(index, ev.cand.cliffhanger_line_id), "open_question_llm_claim": ev.cand.open_question},
        "qualitative_assessment": None if a is None else {
            "emotional_pull": a.emotional_pull, "standalone_clarity": a.standalone_clarity.level,
            "narrative_coherence": a.narrative_coherence, "spoiler_risk": a.spoiler_risk, "spoiler_note": a.spoiler_note},
        "one_line_justification": None if a is None else a.one_line_justification,
        "gates": [g.model_dump() for g in ev.all_gates],
        "adjustments": [x.model_dump() for x in clip.adjustments],
        "llm_original_claims": ev.cand.llm_gate_claims.model_dump(),
        "disclaimer": ASSESSMENT_DISCLAIMER,
    }


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], outcomes: list[TimelineOutcome], meta: dict[str, Any],
                  evaluated: Optional[list[Evaluated]] = None, selection: Optional[Selection] = None,
                  index: Optional[ScreenplayIndex] = None, semantic_info: Optional[dict[str, Any]] = None,
                  early_decisions: Optional[dict[str, Decision]] = None, usage: Optional[dict[str, Any]] = None) -> None:
    out_dir.mkdir(parents=True)
    (out_dir / "timeline_results.json").write_text(
        json.dumps({"before_after": rows, "outcomes": [o.model_dump(mode="json") for o in outcomes]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    report: dict[str, Any] = {
        **meta,
        "deterministic": {"candidates": len(outcomes), "pass": sum(o.accepted for o in outcomes),
                          "rejected": sum(not o.accepted for o in outcomes),
                          "adjusted": sum(bool(o.clip and o.clip.adjustments) for o in outcomes)},
    }
    if evaluated is not None and selection is not None and index is not None:
        decisions = {**selection.decisions, **(early_decisions or {})}
        ranked = [promo_record(index, ev, i, decisions[ev.cid]) for i, ev in enumerate(selection.ranked, 1)]
        others = [promo_record(index, ev, None, decisions[ev.cid]) for ev in evaluated
                  if ev.cid in decisions and decisions[ev.cid].status != "recommended" and ev.clip is not None]
        no_clip = [{"candidate_id": ev.cid, "status": "rejected", "decision_stage": "deterministic",
                    "decision_reasons": ev.outcome.rejection_reasons, "gates": [g.model_dump() for g in ev.outcome.gates]}
                   for ev in evaluated if ev.clip is None]
        (out_dir / "ranked_promos.json").write_text(
            json.dumps({"disclaimer": ASSESSMENT_DISCLAIMER, "count": len(ranked), "promos": ranked}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "rejected_candidates.json").write_text(
            json.dumps({"count": len(others) + len(no_clip), "candidates": [*others, *no_clip]}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        sem_doc: dict[str, Any] = {
            "disclaimer": ASSESSMENT_DISCLAIMER, "prompt_version": SEMANTIC_PROMPT_VERSION,
            "judge_votes": (semantic_info or {}).get("judge_votes", 1),
            "assessments": {e.cid: e.assessment.model_dump() for e in evaluated if e.assessment},
            "batches": (semantic_info or {}).get("batches"), "problems": (semantic_info or {}).get("problems"),
            "context_notes": (semantic_info or {}).get("context_notes")}
        if sem_doc["judge_votes"] > 1:      # every individual verdict + justification, not only the majority result
            sem_doc["assessments_note"] = "with judge_votes > 1 each entry is the majority aggregate; all samples are under 'votes'"
            sem_doc["votes"] = {e.cid: [v.model_dump() if v is not None else None for v in e.votes] for e in evaluated if e.votes}
        (out_dir / "semantic_assessments.json").write_text(json.dumps(sem_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        report["semantic"] = {"stopped": (semantic_info or {}).get("stopped"), "usage": usage,
                              "recommended": len(ranked), "not_recommended": len(others) + len(no_clip),
                              "by_stage": _count_stages(decisions)}
    else:
        (out_dir / "rejected_timeline.json").write_text(
            json.dumps([{"candidate_id": o.candidate_id, "reasons": o.rejection_reasons,
                         "gates": [g.model_dump() for g in o.gates if not g.passed]} for o in outcomes if not o.accepted],
                       indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_stages(decisions: dict[str, Decision]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in decisions.values():
        out[d.stage] = out.get(d.stage, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PromoAI phase 3: deterministic validation, semantic gates, ranking")
    ap.add_argument("--series-dir", type=Path)
    ap.add_argument("--phase2-run", required=True, help="name of a folder under phase2/runs/")
    ap.add_argument("--out-name", help="output folder under phase3/ (default p3-<timestamp>)")
    ap.add_argument("--semantic", action="store_true", help="also run the LLM rubric and ranking (uses API calls)")
    ap.add_argument("--eval-batch", type=int, default=5, help="clips per LLM call (same window only)")
    ap.add_argument("--max-promos", type=int, default=DEFAULT_MAX_PROMOS)
    ap.add_argument("--max-per-episode", type=int, default=DEFAULT_MAX_PER_EPISODE)
    ap.add_argument("--judge-temperature", type=float, default=None,
                    help="temperature of the semantic judge (default: LLM_JUDGE_TEMPERATURE or 0.0)")
    ap.add_argument("--judge-votes", type=int, default=None,
                    help="independent judge samples per clip; >1 enables majority voting (default: LLM_JUDGE_VOTES or 1)")
    ap.add_argument("--cliff-tail-max-s", type=float, default=DEFAULT_CONFIG.cliff_tail_max_ms / 1000,
                    help="max footage allowed after the cliffhanger line before the end is trimmed (default 2.0)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    provider: Optional[LLMProvider] = None
    try:
        series_dir = args.series_dir or _find_series_dir(Path("data/output"))
        out_dir = series_dir / "phase3" / (args.out_name or f"p3-{time.strftime('%Y%m%d-%H%M%S')}")
        if out_dir.exists():                                   # fail BEFORE spending API calls
            raise FileExistsError(f"output folder already exists: {out_dir}")
        index = ScreenplayIndex.from_output_dir(series_dir)
        candidates, contexts, p2_report = load_phase2_run(series_dir, args.phase2_run)
        if args.cliff_tail_max_s < 0:
            raise ValueError("--cliff-tail-max-s must be >= 0")
        tl_config = TimelineConfig(cliff_tail_max_ms=round(args.cliff_tail_max_s * 1000))
        outcomes = run_deterministic(index, candidates, tl_config)
        if args.semantic:
            settings = load_settings()
            judge_temperature = settings.judge_temperature if args.judge_temperature is None else args.judge_temperature
            judge_votes = settings.judge_votes if args.judge_votes is None else args.judge_votes
            if judge_votes < 1 or judge_temperature < 0:
                raise ValueError("--judge-votes must be >= 1 and --judge-temperature >= 0")
            # The judge is the same provider abstraction with its own temperature: only the Settings differ.
            provider = create_provider(dataclasses.replace(settings, temperature=judge_temperature))
    except (FileNotFoundError, FileExistsError, ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = [before_after(c, o) for c, o in zip(candidates, outcomes)]
    print_table(rows)
    meta = {"phase2_run": args.phase2_run, "phase2_prompt_version": p2_report.get("prompt_version"),
            "mode": "semantic" if args.semantic else "deterministic", "timeline_config": tl_config.to_dict()}
    acc = sum(o.accepted for o in outcomes)
    print(f"\n{len(outcomes)} candidates: {acc} pass the deterministic gates, {len(outcomes) - acc} rejected")

    if not args.semantic:
        write_outputs(out_dir, rows, outcomes, meta)
        print(f"-> {out_dir}  (deterministic only; add --semantic for LLM assessments and ranking)")
        return 0

    assert provider is not None
    all_ev = [Evaluated(cand=c, outcome=o) for c, o in zip(candidates, outcomes)]
    kept_ev, early = drop_exact_duplicates(all_ev)             # identical final windows: don't pay twice
    print(f"semantic stage: model={provider.model}, {sum(e.outcome.accepted for e in kept_ev)} clip(s) to assess")
    info = run_semantic(provider, index, kept_ev, contexts, len(index.episodes), max(1, args.eval_batch), judge_votes)

    episode_numbers = {eid: ep.episode_number for eid, ep in index.episodes.items()}
    selection = select_promos(kept_ev, episode_numbers, len(index.episodes), max_promos=args.max_promos,
                              max_per_episode=args.max_per_episode)
    usage = provider.stats.as_dict()
    meta.update({"semantic_prompt_version": SEMANTIC_PROMPT_VERSION, "model": provider.model,
                 "judge_temperature": judge_temperature, "judge_votes": judge_votes})
    write_outputs(out_dir, rows, outcomes, meta, all_ev, selection, index, info, early, usage)

    print(f"\napi calls: {usage['api_calls']}  cache hits: {usage['cache_hits']}  json retries: {usage['json_retries']}  "
          f"tokens: prompt {usage['prompt_tokens']:,} / output {usage['output_tokens']:,}  failures: {len(usage['failures'])}")
    if info["stopped"]:
        print(f"STOPPED EARLY: {info['stopped']}")
    print(f"recommended: {len(selection.ranked)}   (assessments are model opinions; see ranked_promos.json for reasons)")
    for i, ev in enumerate(selection.ranked, 1):
        print(f"  #{i} {ev.cid}  {ev.clip.total_duration_ms / 1000:.1f}s  pull={ev.assessment.emotional_pull}  "  # type: ignore[union-attr]
              f"standalone={ev.assessment.standalone_clarity.level}  coherence={ev.assessment.narrative_coherence}")  # type: ignore[union-attr]
        print(f"      {ev.assessment.one_line_justification}")  # type: ignore[union-attr]
    for cid, d in sorted(selection.decisions.items()):
        if d.status != "recommended":
            print(f"  - {cid}: {d.status} [{d.stage}] {'; '.join(d.reasons)[:230]}")
    for cid, d in early.items():
        print(f"  - {cid}: {d.status} [{d.stage}] {'; '.join(d.reasons)}")
    print(f"-> {out_dir}")
    return 0 if not info["stopped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
