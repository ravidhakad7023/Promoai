"""Judge stability experiment: how repeatable are the semantic verdicts at a given temperature?

    python -m promoai.stability --phase2-run v2-sample-e001-e005 --out-name judge-stability

For each temperature the SAME judging job (same candidates, same window batching as the real
pipeline, same prompt, same model) is run several times.  Only the temperature differs between the
two groups.  Each run gets its own cache salt inside a separate cache folder (``data/cache/stability``),
so every run is a genuine independent API call while the normal cache and all earlier outputs stay
untouched; re-running the experiment replays from that folder at no cost.

Majority voting is NOT used here (judge_votes=1): this measures raw single-judge stability.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .config import ConfigError, load_settings
from .index import ScreenplayIndex
from .llm import CachedProvider, LLMError, LLMProvider
from .llm_gemini import GeminiProvider
from .ranking import Evaluated, drop_exact_duplicates
from .semantic import (SEMANTIC_PROMPT_VERSION, PromoAssessment, RUBRIC, assess_batch, build_semantic_prompt, semantic_gates)
from .story import sanitize_context
from .timeline import DEFAULT_CONFIG
from .validate import _find_series_dir, load_phase2_run, run_deterministic, run_semantic

GATES = ("hook_strength", "cliffhanger_strength", "standalone_clarity", "narrative_coherence")


# --------------------------------------------------------------------------- verdicts
def verdicts_of(a: Optional[PromoAssessment]) -> dict[str, Optional[dict[str, Any]]]:
    """The four gate verdicts of one assessment.  ``decision`` is pass/fail (after Python's consistency rule),
    ``label`` is the finer level the judge chose.  Narrative coherence has no pass flag of its own: 'fail' = level weak."""
    if a is None:
        return {g: None for g in GATES}
    sg = {g.gate: g for g in semantic_gates(a)}
    return {
        "hook_strength": {"decision": "pass" if sg["hook_strength"].passed else "fail", "label": a.hook_strength.level,
                          "justification": a.hook_strength.reason},
        "cliffhanger_strength": {"decision": "pass" if sg["cliffhanger_strength"].passed else "fail", "label": a.cliffhanger_strength.level,
                                 "justification": a.cliffhanger_strength.reason},
        "standalone_clarity": {"decision": "pass" if sg["standalone_clarity"].passed else "fail", "label": a.standalone_clarity.level,
                               "justification": a.standalone_clarity.reason, "missing_context": a.standalone_clarity.missing_context},
        "narrative_coherence": {"decision": "fail" if a.narrative_coherence == "weak" else "pass", "label": a.narrative_coherence,
                                "justification": a.one_line_justification + "  [overall justification; coherence has no separate reason field]"},
    }


def flip_stats(runs: list[dict[str, dict]]) -> dict[str, Any]:
    """Compare runs 2..n against run 1.  A comparison needs both verdicts to exist.

    decision flip = pass/fail differs;  label change = the finer level differs (a superset of decision flips)."""
    cids = sorted(runs[0])
    out: dict[str, Any] = {"per_gate": {}, "overall": {}, "by_comparison": {}}
    tot = {"comparisons": 0, "decision_flips": 0, "label_changes": 0}
    by_cmp: dict[str, dict[str, int]] = {}
    stable_all = stable_decisions = cells = 0
    for g in GATES:
        c = {"comparisons": 0, "decision_flips": 0, "label_changes": 0, "by_candidate": {}}
        for cid in cids:
            vals = [r.get(cid, {}).get(g) if r.get(cid) else None for r in runs]
            first = vals[0]
            flips = []
            for k, v in enumerate(vals[1:], 2):
                name = f"run{k}_vs_run1"
                by_cmp.setdefault(name, {"comparisons": 0, "decision_flips": 0, "label_changes": 0})
                if first is None or v is None:
                    continue
                d, l = v["decision"] != first["decision"], v["label"] != first["label"]
                for bucket in (c, tot, by_cmp[name]):
                    bucket["comparisons"] += 1
                    bucket["decision_flips"] += d
                    bucket["label_changes"] += l
                flips.append({"comparison": name, "decision_flip": d, "label_change": l})
            c["by_candidate"][cid] = {"labels": [v["label"] if v else None for v in vals],
                                      "decisions": [v["decision"] if v else None for v in vals], "comparisons": flips}
            if all(v is not None for v in vals):
                cells += 1
                stable_all += len({v["label"] for v in vals}) == 1
                stable_decisions += len({v["decision"] for v in vals}) == 1
        c["decision_flip_rate"] = c["decision_flips"] / c["comparisons"] if c["comparisons"] else None
        c["label_change_rate"] = c["label_changes"] / c["comparisons"] if c["comparisons"] else None
        out["per_gate"][g] = c
    out["overall"] = {**tot, "decision_flip_rate": tot["decision_flips"] / tot["comparisons"] if tot["comparisons"] else None,
                      "label_change_rate": tot["label_changes"] / tot["comparisons"] if tot["comparisons"] else None,
                      "candidate_gate_cells": cells, "cells_identical_label_in_all_runs": stable_all,
                      "cells_identical_decision_in_all_runs": stable_decisions}
    for name, b in by_cmp.items():
        b["decision_flip_rate"] = b["decision_flips"] / b["comparisons"] if b["comparisons"] else None
        b["label_change_rate"] = b["label_changes"] / b["comparisons"] if b["comparisons"] else None
    out["by_comparison"] = by_cmp
    return out


# --------------------------------------------------------------------------- recording shim
class _Recorder(LLMProvider):
    """Records a hash of every prompt (never the text) so we can PROVE the inputs were identical across runs."""

    def __init__(self, inner: LLMProvider) -> None:
        super().__init__()
        self.inner, self.name, self.model, self.stats = inner, inner.name, inner.model, inner.stats
        self.prompt_hashes: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompt_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        return self.inner.generate(prompt)

    def invalidate(self, prompt: str) -> None:
        self.inner.invalidate(prompt)

    def remember(self, prompt: str, text: str) -> None:
        self.inner.remember(prompt, text)


def _usage_delta(before: dict, after: dict) -> dict[str, Any]:
    keys = ("api_calls", "cache_hits", "json_retries", "prompt_tokens", "output_tokens", "thinking_tokens")
    d = {k: after[k] - before[k] for k in keys}
    d["failures"] = after["failures"][len(before["failures"]):]
    return d


# --------------------------------------------------------------------------- experiment
def run_experiment(index: ScreenplayIndex, candidates, contexts, settings, cache_dir: Path, temps: list[float], n_runs: int,
                   batch_size: int) -> dict[str, Any]:
    outcomes = run_deterministic(index, candidates, DEFAULT_CONFIG)

    def fresh() -> list[Evaluated]:
        evs = [Evaluated(cand=c, outcome=o) for c, o in zip(candidates, outcomes)]
        return drop_exact_duplicates(evs)[0]

    base = fresh()
    judged = [e for e in base if e.outcome.accepted and e.clip is not None]
    not_judged = [{"candidate_id": e.cid, "reason": "; ".join(e.outcome.rejection_reasons) or "duplicate"} for e in base if e not in judged]
    n_series = len(index.episodes)

    inners = {t: GeminiProvider(dataclasses.replace(settings, temperature=t)) for t in temps}
    runs: list[dict[str, Any]] = []
    stopped: Optional[str] = None
    for k in range(1, n_runs + 1):                              # interleaved: t0 r1, t1 r1, t0 r2, t1 r2 ...
        for t in temps:
            if stopped:
                break
            rec = _Recorder(CachedProvider(inners[t], cache_dir, salt=f"stability-run{k}"))
            items = fresh()
            before = rec.stats.as_dict()
            info = run_semantic(rec, index, items, contexts, n_series, batch_size, judge_votes=1)
            runs.append({
                "temperature": t, "run": k,
                "assessments": {e.cid: (e.assessment.model_dump() if e.assessment else None) for e in items if e.outcome.accepted},
                "usage": _usage_delta(before, rec.stats.as_dict()), "problems": info["problems"],
                "batches": info["batches"], "prompt_sha256": rec.prompt_hashes,
            })
            if info["stopped"]:
                stopped = info["stopped"]
    return {"runs": runs, "judged": [e.cid for e in judged], "not_judged": not_judged, "stopped": stopped,
            "_judged_objects": judged, "_inners": inners, "_n_series": n_series}


def solo_check(index, exp, contexts, settings, cache_dir: Path, batch_size: int) -> list[dict[str, Any]]:
    """For clips whose temperature-0 verdicts changed between runs: judge each ALONE once at temperature 0."""
    t0 = [r for r in exp["runs"] if r["temperature"] == 0.0]
    if len(t0) < 2:
        return []
    judged: list[Evaluated] = exp["_judged_objects"]
    by_window: dict[str, list[Evaluated]] = {}
    for e in judged:
        by_window.setdefault(e.cand.window_id, []).append(e)
    out = []
    solo_provider = _Recorder(CachedProvider(exp["_inners"][0.0], cache_dir, salt="stability-solo"))
    for e in judged:
        seqs = [verdicts_of(PromoAssessment.model_validate(r["assessments"][e.cid]) if r["assessments"].get(e.cid) else None) for r in t0]
        changed = [g for g in GATES if len({(s[g]["decision"], s[g]["label"]) for s in seqs if s[g]}) > 1]
        if not changed:
            continue
        wid = e.cand.window_id
        ctx = sanitize_context(contexts[wid])[0] if wid in contexts else None
        group = by_window[wid]
        batch = group[(group.index(e) // batch_size) * batch_size:][:batch_size]
        same_prompt = (build_semantic_prompt(index, [(e.cand, e.clip)], ctx, exp["_n_series"])
                       == build_semantic_prompt(index, [(x.cand, x.clip) for x in batch], ctx, exp["_n_series"]))
        before = solo_provider.stats.as_dict()
        try:
            got, problems = assess_batch(solo_provider, index, [(e.cand, e.clip)], ctx, exp["_n_series"])
        except LLMError as exc:
            out.append({"candidate_id": e.cid, "gates_that_changed_in_batch": changed, "error": type(exc).__name__})
            continue
        a = got.get(e.cid)
        out.append({"candidate_id": e.cid, "gates_that_changed_in_batch": changed, "prompt_identical_to_batch_prompt": same_prompt,
                    "batch_size_in_pipeline": len(batch), "solo_verdicts": verdicts_of(a),
                    "batch_run_verdicts": [{g: ({"decision": s[g]["decision"], "label": s[g]["label"]} if s[g] else None) for g in GATES} for s in seqs],
                    "assessment": a.model_dump() if a else None, "problems": problems,
                    "prompt_sha256": solo_provider.prompt_hashes[-1],
                    "usage": _usage_delta(before, solo_provider.stats.as_dict())})
    return out


# --------------------------------------------------------------------------- reporting
def summarize(exp: dict[str, Any], temps: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_temperature": {}}
    for t in temps:
        rs = sorted((r for r in exp["runs"] if r["temperature"] == t), key=lambda r: r["run"])
        per_run = [{cid: verdicts_of(PromoAssessment.model_validate(a) if a else None) for cid, a in r["assessments"].items()} for r in rs]
        if per_run:
            summary["by_temperature"][str(t)] = flip_stats(per_run)
    return summary


def totals(runs: list[dict[str, Any]], extra: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("api_calls", "cache_hits", "json_retries", "prompt_tokens", "output_tokens", "thinking_tokens")
    t = {k: sum(r["usage"][k] for r in runs) + sum(s.get("usage", {}).get(k, 0) for s in extra) for k in keys}
    t["failures"] = [f for r in runs for f in r["usage"]["failures"]] + [f for s in extra for f in s.get("usage", {}).get("failures", [])]
    return t


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_report(exp, summary, solo, temps, meta) -> str:
    L = [f"# Judge stability experiment", "", f"Model `{meta['model']}`, semantic prompt `{SEMANTIC_PROMPT_VERSION}`, rubric sha256 `{meta['rubric_sha256'][:12]}`.",
         f"Judged clips: {', '.join(exp['judged'])}. Not judged (failed Python's deterministic gates, so the pipeline never sends them): "
         f"{'; '.join(n['candidate_id'] + ' (' + n['reason'][:90] + ')' for n in exp['not_judged']) or 'none'}.", ""]
    L += ["## Flip rates (runs 2 and 3 compared with run 1)", "",
          "| temperature | gate | comparisons | decision flips | label changes |", "|---|---|---|---|---|"]
    for t in temps:
        st = summary["by_temperature"].get(str(t))
        if not st:
            continue
        for g in GATES:
            c = st["per_gate"][g]
            L.append(f"| {t} | {g} | {c['comparisons']} | {c['decision_flips']} ({pct(c['decision_flip_rate'])}) | {c['label_changes']} ({pct(c['label_change_rate'])}) |")
        o = st["overall"]
        L.append(f"| **{t}** | **overall** | {o['comparisons']} | **{o['decision_flips']} ({pct(o['decision_flip_rate'])})** | **{o['label_changes']} ({pct(o['label_change_rate'])})** |")
    L += ["", "## Every verdict (label per run 1 / 2 / 3, decision in brackets)", ""]
    for t in temps:
        st = summary["by_temperature"].get(str(t))
        if not st:
            continue
        L += [f"### temperature {t}", "", "| clip | gate | run 1 | run 2 | run 3 |", "|---|---|---|---|---|"]
        for cid in exp["judged"]:
            for g in GATES:
                b = st["per_gate"][g]["by_candidate"][cid]
                L.append(f"| {cid} | {g} | " + " | ".join(f"{lb} ({d})" for lb, d in zip(b["labels"], b["decisions"])) + " |")
        L.append("")
    L += ["## All justifications", ""]
    for r in sorted(exp["runs"], key=lambda r: (r["temperature"], r["run"])):
        L += [f"### temperature {r['temperature']} - run {r['run']}", ""]
        for cid, a in r["assessments"].items():
            v = verdicts_of(PromoAssessment.model_validate(a) if a else None)
            L.append(f"**{cid}**" + ("" if a else " - NO ASSESSMENT"))
            for g in GATES:
                if v[g]:
                    L.append(f"- {g}: {v[g]['decision']} / {v[g]['label']} - {v[g]['justification']}")
            if a:
                L.append(f"- emotional_pull: {a['emotional_pull']}; spoiler_risk: {a['spoiler_risk']}")
            L.append("")
    if solo:
        L += ["## Solo check at temperature 0 (clips whose batch verdicts changed)", ""]
        for s in solo:
            L.append(f"**{s['candidate_id']}** - changed in batch: {', '.join(s['gates_that_changed_in_batch'])}; "
                     f"prompt identical to its batch prompt: {s.get('prompt_identical_to_batch_prompt')}")
            for g in GATES:
                sv = (s.get("solo_verdicts") or {}).get(g)
                if sv:
                    L.append(f"- {g}: solo = {sv['decision']} / {sv['label']} | batch runs = " +
                             ", ".join(f"{b[g]['decision']}/{b[g]['label']}" for b in s["batch_run_verdicts"] if b[g]) + f" | solo justification: {sv['justification']}")
            L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PromoAI: semantic-judge stability experiment")
    ap.add_argument("--series-dir", type=Path)
    ap.add_argument("--phase2-run", required=True)
    ap.add_argument("--out-name", required=True, help="folder under phase3/stability/")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.4])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--eval-batch", type=int, default=5)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/stability"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        series_dir = args.series_dir or _find_series_dir(Path("data/output"))
        out_dir = series_dir / "phase3" / "stability" / args.out_name
        if out_dir.exists():
            raise FileExistsError(f"output folder already exists: {out_dir}")
        settings = load_settings()
        index = ScreenplayIndex.from_output_dir(series_dir)
        candidates, contexts, _ = load_phase2_run(series_dir, args.phase2_run)
        GeminiProvider(settings)             # fail early on a missing key / model
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    exp = run_experiment(index, candidates, contexts, settings, args.cache_dir, args.temps, args.runs, args.eval_batch)
    solo = solo_check(index, exp, contexts, settings, args.cache_dir, args.eval_batch) if 0.0 in args.temps else []
    summary = summarize(exp, args.temps)
    used = totals(exp["runs"], solo)
    meta = {"model": settings.model, "temperatures": args.temps, "runs_per_temperature": args.runs,
            "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(), "semantic_prompt_version": SEMANTIC_PROMPT_VERSION,
            "judge_votes": 1, "cache_dir": str(args.cache_dir)}
    hashes = {(r["temperature"], r["run"]): tuple(r["prompt_sha256"]) for r in exp["runs"]}
    meta["prompts_identical_across_all_runs"] = len(set(hashes.values())) == 1
    meta["distinct_prompt_hashes"] = sorted({h for hs in hashes.values() for h in hs})
    public = {k: v for k, v in exp.items() if not k.startswith("_")}

    out_dir.mkdir(parents=True)
    (out_dir / "verdicts.json").write_text(json.dumps({"meta": meta, **public, "solo_check": solo}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "flip_stats": summary, "usage": used}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(exp, summary, solo, args.temps, meta), encoding="utf-8")

    print(f"model={settings.model}  judged clips={exp['judged']}  not judged={[n['candidate_id'] for n in exp['not_judged']]}")
    print(f"prompts identical across all runs: {meta['prompts_identical_across_all_runs']}")
    print(f"API: calls={used['api_calls']} cache_hits={used['cache_hits']} json_retries={used['json_retries']} "
          f"tokens in/out={used['prompt_tokens']:,}/{used['output_tokens']:,} thinking={used['thinking_tokens']:,} failures={len(used['failures'])}")
    if exp["stopped"]:
        print(f"STOPPED EARLY: {exp['stopped']}")
    print(f"-> {out_dir}")
    return 0 if not exp["stopped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
