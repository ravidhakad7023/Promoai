"""End-to-end runner:  python run.py --input <zip-or-folder> [options]

Stages (each writes artifacts, then - last - a completion record):
    01_parse     parse + index the screenplay
    02_detect    story context + promo candidate detection (LLM)
    03_timeline  deterministic timeline validation / quality gates (no LLM)
    04_semantic  semantic gates (LLM judge)
    05_rank      ranking + final deliverables

A stage is "complete" only if its record exists AND every artifact it lists is present with the recorded
checksum.  Half-written files are never mistaken for results (all writes are atomic: temp file + rename).

Every invocation creates a NEW run directory and refuses to reuse an existing one.  After a crash,
``--resume RUN_DIR`` verifies finished stages, quarantines (moves, never deletes) incomplete ones and
re-runs them; model calls already answered are served from the response cache.

Secrets: the API key lives only in the config system; nothing here writes it anywhere.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .config import ConfigError, Settings, load_settings
from .detect import DEFAULT_MAX_CANDIDATES, run_detection
from .index import ScreenplayIndex
from .ingest import ExtractResult, discover_folder, extract_archive, find_archives, slugify
from .llm import LLMError, LLMProvider, create_provider
from .pipeline import build_summary, parse_extracted
from .prompts import PROMPT_VERSION
from .ranking import (
    DEFAULT_MAX_PER_EPISODE, DEFAULT_MAX_PROMOS, LATE_SEASON_FRACTION, OVERLAP_MAX_FRACTION, RANK_KEY_DESCRIPTION,
    Decision, Evaluated, Selection, drop_exact_duplicates, rank_key, select_promos,
)
from .schemas import RejectedCandidate, StoredCandidate, StoryContext
from .semantic import SEMANTIC_PROMPT_VERSION, PromoAssessment
from .timeline import DEFAULT_CONFIG, GateResult, TimelineConfig, TimelineOutcome
from .validate import ASSESSMENT_DISCLAIMER, before_after, promo_record, run_deterministic, run_semantic

RUN_FORMAT_VERSION = 1
RUN_DEFAULT_JUDGE_VOTES = 3       # this entry point judges every clip with 3 independent votes (majority decides)
STAGES = ("01_parse", "02_detect", "03_timeline", "04_semantic", "05_rank")
DELIVERABLES = ("ranked_promos.json", "ranked_promos.csv", "rejected_candidates.json", "report.md", "usage_report.json")


class RunError(RuntimeError):
    """A problem with the request itself (bad input, mismatching resume ...)."""


class StageIncomplete(RuntimeError):
    """A stage could not finish (model failures, quota ...).  It gets NO completion record."""

    def __init__(self, stage: str, reason: str, usage: Optional[dict] = None) -> None:
        super().__init__(reason)
        self.stage, self.reason, self.usage = stage, reason, usage


# =========================================================================== small utilities
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same folder, fsync, then rename: readers see the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================== failure log (raw replies + errors)
def _prompt_refs(prompt: Optional[str]) -> dict[str, Any]:
    if not prompt:
        return {}
    return {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "candidate_ids": sorted(set(re.findall(r'candidate_id="([^"]+)"', prompt))),
            "episode_ids": sorted(set(re.findall(r"^## (E\d+)", prompt, re.M)))}


class FailureLog:
    """Numbered, exclusive-create files: an earlier raw reply is never overwritten, even across resumes."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _next(self) -> int:
        nums = [int(p.name[:4]) for p in self.dir.glob("[0-9][0-9][0-9][0-9]-*.json")]
        return max(nums, default=0) + 1

    def record(self, stage: str, kind: str, error: str, raw: Optional[str] = None, prompt: Optional[str] = None,
               attempt: Optional[int] = None) -> Path:
        payload = {"recorded_at": utc_now(), "stage": stage, "kind": kind, "attempt": attempt, "error": error,
                   **_prompt_refs(prompt), "raw_reply": raw}
        while True:
            path = self.dir / f"{self._next():04d}-{stage}-{kind}.json"
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"sequence": int(path.name[:4]), **payload}, fh, indent=2, ensure_ascii=False)
            return path

    def sink_for(self, stage: str) -> Callable[..., None]:
        def sink(prompt: str, attempt: int, raw: str, error: str) -> None:
            self.record(stage, "invalid_reply", error, raw=raw, prompt=prompt, attempt=attempt)
        return sink

    def files(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("[0-9][0-9][0-9][0-9]-*.json"))


# =========================================================================== stage store (completion records)
class StageStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.state = run_dir / "state"
        self.state.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / "stages" / stage

    def record_path(self, stage: str) -> Path:
        return self.state / f"{stage}.json"

    def valid_record(self, stage: str) -> Optional[dict]:
        """The completion record, only if every listed artifact exists with the recorded checksum."""
        p = self.record_path(stage)
        if not p.is_file():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            for rel, digest in rec["artifacts"].items():
                f = self.run_dir / rel
                if not f.is_file() or sha256_file(f) != digest:
                    return None
            return rec
        except (OSError, ValueError, KeyError):
            return None

    def write_record(self, stage: str, artifacts: list[Path], usage: dict, counts: dict, seconds: float) -> dict:
        rec = {"stage": stage, "completed_at": utc_now(), "seconds": round(seconds, 2), "usage": usage, "counts": counts,
               "artifacts": {p.relative_to(self.run_dir).as_posix(): sha256_file(p) for p in artifacts}}
        atomic_write_json(self.record_path(stage), rec)      # LAST: its existence means the stage finished
        return rec

    def quarantine(self, stage: str) -> Optional[Path]:
        """Move (never delete) a stage's leftovers and record so an incomplete stage cannot be mistaken for finished work."""
        src_dir, src_rec = self.stage_dir(stage), self.record_path(stage)
        if not src_dir.exists() and not src_rec.exists():
            return None
        dest = self.state / "quarantine" / f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        dest.mkdir(parents=True)
        if src_dir.exists():
            shutil.move(str(src_dir), str(dest / "stage"))
        if src_rec.exists():
            shutil.move(str(src_rec), str(dest / "record.json"))
        return dest


# =========================================================================== context carried between stages
@dataclass
class Ctx:
    run_dir: Path
    cfg: dict[str, Any]
    settings: Settings
    factory: Callable[[Settings, str], LLMProvider]
    store: StageStore
    failures: FailureLog
    tl_config: TimelineConfig
    index: Optional[ScreenplayIndex] = None
    series_length: int = 0
    ep_ids: list[str] = field(default_factory=list)
    parse_info: dict[str, Any] = field(default_factory=dict)
    candidates: list[StoredCandidate] = field(default_factory=list)
    structural_rejected: list[RejectedCandidate] = field(default_factory=list)
    contexts: dict[str, StoryContext] = field(default_factory=dict)
    windows: list[dict[str, Any]] = field(default_factory=list)
    detect_info: dict[str, Any] = field(default_factory=dict)
    outcomes: list[TimelineOutcome] = field(default_factory=list)
    all_ev: list[Evaluated] = field(default_factory=list)
    kept_ev: list[Evaluated] = field(default_factory=list)
    early: dict[str, Decision] = field(default_factory=dict)
    semantic_info: dict[str, Any] = field(default_factory=dict)
    records: dict[str, dict] = field(default_factory=dict)
    executed: list[str] = field(default_factory=list)

    def sdir(self, stage: str) -> Path:
        return self.store.stage_dir(stage)


def usage_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Actual API calls vs cache hits, tokens, retries and failures from a provider's counters."""
    failures = list(stats["failures"])
    transport_retries = sum(1 for f in failures if f.get("detail") == "will retry")
    return {
        "api_calls": stats["api_calls"], "cache_hits": stats["cache_hits"],
        "json_retries": stats["json_retries"], "transport_retries": transport_retries,
        "retries_total": stats["json_retries"] + transport_retries,
        "prompt_tokens": stats["prompt_tokens"], "output_tokens": stats["output_tokens"],
        "thinking_tokens": stats["thinking_tokens"], "total_tokens": stats["total_tokens"],
        "tokens_saved_by_cache": stats["tokens_saved_by_cache"],
        "failure_events": len(failures), "failure_kinds": dict(Counter(f["kind"] for f in failures)), "failures": failures,
    }


ZERO_USAGE = usage_summary({"api_calls": 0, "cache_hits": 0, "json_retries": 0, "prompt_tokens": 0, "output_tokens": 0,
                            "thinking_tokens": 0, "total_tokens": 0, "tokens_saved_by_cache": 0, "failures": []})


# =========================================================================== stage 1: parse
def run_parse(ctx: Ctx) -> tuple[list[Path], dict, dict]:
    d = ctx.sdir("01_parse")
    if ctx.cfg["input_kind"] == "zip":
        # Copy the input under a short name (pins the exact input AND keeps extracted paths well under Windows' 260-char limit).
        src = Path(ctx.cfg["input_path"])
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / "input.zip")
        ex: ExtractResult = extract_archive(d / "input.zip", d / "x")
        if ex.series_title == "input":                       # the zip had no single top folder: fall back to the original name
            ex.series_title = src.stem.removesuffix("-script").strip()
    else:
        ex = discover_folder(Path(ctx.cfg["input_path"]))
    total = len(ex.script_files)
    if ctx.cfg["max_episodes"]:
        ex.script_files = ex.script_files[: ctx.cfg["max_episodes"]]      # first N discovered, in episode order
    series = parse_extracted(ex)
    if not series.episodes:
        raise RunError(f"no episode could be parsed from the input ({len(series.failed)} file(s) failed)")
    for ep in series.episodes:
        atomic_write_text(d / "episodes" / f"{ep.episode_id}.json", ep.model_dump_json(indent=2))
    summary = build_summary(series)
    atomic_write_json(d / "dataset_summary.json", {**summary.model_dump(mode="json"), "episodes_discovered_in_input": total,
                                                    "episodes_selected": len(ex.script_files),
                                                    "series_title": ex.series_title})
    load_parse(ctx)
    files = [p for p in d.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    return sorted(files), ZERO_USAGE, {"episodes_discovered": total, "episodes_processed": len(series.episodes),
                                       "scenes": summary.scenes, "lines": summary.lines, "parse_warnings": summary.warnings,
                                       "unparsed_entries": summary.unparsed_entries, "failed_files": len(summary.failed_files)}


def load_parse(ctx: Ctx) -> None:
    d = ctx.sdir("01_parse")
    ctx.index = ScreenplayIndex.from_output_dir(d)
    info = json.loads((d / "dataset_summary.json").read_text(encoding="utf-8"))
    ctx.parse_info = info
    ctx.series_length = int(info["episodes_discovered_in_input"])       # spoiler guard / prompts use the WHOLE series size
    ctx.ep_ids = [e.episode_id for e in sorted(ctx.index.episodes.values(), key=lambda e: e.episode_number)]


# =========================================================================== stage 2: story context + detection
def run_detect(ctx: Ctx) -> tuple[list[Path], dict, dict]:
    d = ctx.sdir("02_detect")
    provider = ctx.factory(ctx.settings, "proposer")
    provider.on_invalid = ctx.failures.sink_for("detect")
    run = run_detection(ctx.index, provider, ctx.ep_ids, window_size=ctx.cfg["window_size"], overlap=ctx.cfg["overlap"],
                        max_candidates=ctx.cfg["candidates_per_window"])
    usage = {**usage_summary(provider.stats.as_dict()), "provider": provider.name, "model": provider.model}
    atomic_write_json(d / "candidates.json", {"accepted": [c.model_dump(mode="json") for c in run.accepted],
                                              "rejected": [r.model_dump(mode="json") for r in run.rejected]})
    for wid, c in run.contexts.items():
        atomic_write_text(d / "story_context_history" / f"{wid}.json", c.model_dump_json(indent=2))
    if run.final_context is not None:
        atomic_write_text(d / "story_context.json", run.final_context.model_dump_json(indent=2))
    atomic_write_json(d / "detect_report.json", {"prompt_version": PROMPT_VERSION, "windows": [dataclasses.asdict(w) for w in run.windows],
                                                 "stopped_early": run.stopped_early})
    bad = [w for w in run.windows if w.status != "ok"]
    if run.stopped_early or bad:
        raise StageIncomplete("02_detect", run.stopped_early or f"{len(bad)} window(s) failed: " + ", ".join(
            f"{w.window_id} ({w.error})" for w in bad), usage)
    load_detect(ctx)
    files = sorted(p for p in d.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    counts = {"windows": len(run.windows), "candidates_requested_per_window": ctx.cfg["candidates_per_window"],
              "candidates_returned": sum(w.candidates_returned for w in run.windows),
              "structurally_valid": len(run.accepted), "structurally_rejected": len(run.rejected)}
    return files, usage, counts


def load_detect(ctx: Ctx) -> None:
    d = ctx.sdir("02_detect")
    raw = json.loads((d / "candidates.json").read_text(encoding="utf-8"))
    ctx.candidates = [StoredCandidate.model_validate(c) for c in raw["accepted"]]
    ctx.structural_rejected = [RejectedCandidate.model_validate(r) for r in raw["rejected"]]
    ctx.contexts = {p.stem: StoryContext.model_validate_json(p.read_text(encoding="utf-8"))
                    for p in sorted((d / "story_context_history").glob("W*.json"))}
    ctx.detect_info = json.loads((d / "detect_report.json").read_text(encoding="utf-8"))
    ctx.windows = ctx.detect_info["windows"]


# =========================================================================== stage 3: deterministic timeline gates
def run_timeline(ctx: Ctx) -> tuple[list[Path], dict, dict]:
    d = ctx.sdir("03_timeline")
    outcomes = run_deterministic(ctx.index, ctx.candidates, ctx.tl_config)
    rows = [before_after(c, o) for c, o in zip(ctx.candidates, outcomes)]
    atomic_write_json(d / "timeline_results.json", {"timeline_config": ctx.tl_config.to_dict(), "before_after": rows,
                                                    "outcomes": [o.model_dump(mode="json") for o in outcomes]})
    load_timeline(ctx)
    return [d / "timeline_results.json"], ZERO_USAGE, {"candidates": len(outcomes), "pass": sum(o.accepted for o in outcomes),
                                                       "fail": sum(not o.accepted for o in outcomes),
                                                       "with_adjustments": sum(bool(o.clip and o.clip.adjustments) for o in outcomes)}


def load_timeline(ctx: Ctx) -> None:
    raw = json.loads((ctx.sdir("03_timeline") / "timeline_results.json").read_text(encoding="utf-8"))
    ctx.outcomes = [TimelineOutcome.model_validate(o) for o in raw["outcomes"]]


# =========================================================================== stage 4: semantic gates
def _fresh_evaluated(ctx: Ctx) -> None:
    ctx.all_ev = [Evaluated(cand=c, outcome=o) for c, o in zip(ctx.candidates, ctx.outcomes)]
    ctx.kept_ev, ctx.early = drop_exact_duplicates(ctx.all_ev)


def run_semantic_stage(ctx: Ctx) -> tuple[list[Path], dict, dict]:
    d = ctx.sdir("04_semantic")
    _fresh_evaluated(ctx)
    provider = ctx.factory(ctx.settings, "judge")
    provider.on_invalid = ctx.failures.sink_for("semantic")
    # salt = run folder name: unique per run (new runs make fresh calls), identical on resume (finished votes are reused)
    info = run_semantic(provider, ctx.index, ctx.kept_ev, ctx.contexts, ctx.series_length, ctx.cfg["eval_batch"], ctx.cfg["judge_votes"],
                        salt_prefix=f"{ctx.run_dir.name}/")
    usage = {**usage_summary(provider.stats.as_dict()), "provider": provider.name, "model": provider.model}
    doc: dict[str, Any] = {
        "disclaimer": ASSESSMENT_DISCLAIMER, "prompt_version": SEMANTIC_PROMPT_VERSION,
        "judge_temperature": ctx.cfg["judge_temperature"], "judge_votes": ctx.cfg["judge_votes"],
        "assessments": {e.cid: e.assessment.model_dump() for e in ctx.kept_ev if e.assessment},
        "semantic_gates": {e.cid: [g.model_dump() for g in e.semantic_gates] for e in ctx.kept_ev if e.assessment},
        "early_duplicates": {cid: dec.model_dump() for cid, dec in ctx.early.items()},
        "batches": info["batches"], "problems": info["problems"], "context_notes": info["context_notes"], "stopped": info["stopped"]}
    if ctx.cfg["judge_votes"] > 1:
        doc["votes"] = {e.cid: [v.model_dump() if v is not None else None for v in e.votes] for e in ctx.kept_ev if e.votes}
    atomic_write_json(d / "semantic_assessments.json", doc)
    bad = [b for b in info["batches"] if b["status"] != "ok"]
    if info["stopped"] or bad:
        raise StageIncomplete("04_semantic", info["stopped"] or f"{len(bad)} judging batch(es) did not complete", usage)
    load_semantic(ctx)
    return [d / "semantic_assessments.json"], usage, {
        "assessed": len(doc["assessments"]), "judge_votes": ctx.cfg["judge_votes"], "batches": len(info["batches"]),
        "judge_flag_disagreements": count_disagreements(doc["semantic_gates"])}


def count_disagreements(semantic_gates: dict[str, list[dict]]) -> int:
    """Gate verdicts where the judge's own pass flag conflicted with its level (counted per individual vote when voting)."""
    n = 0
    for gates in semantic_gates.values():
        for g in gates:
            d = g["details"]
            n += sum(1 for v in d["votes"] if v.get("disagreement")) if "votes" in d else int(bool(d.get("disagreement")))
    return n


def load_semantic(ctx: Ctx) -> None:
    doc = json.loads((ctx.sdir("04_semantic") / "semantic_assessments.json").read_text(encoding="utf-8"))
    _fresh_evaluated(ctx)
    for e in ctx.kept_ev:
        if e.cid in doc["assessments"]:
            e.assessment = PromoAssessment.model_validate(doc["assessments"][e.cid])
            e.semantic_gates = [GateResult.model_validate(g) for g in doc["semantic_gates"][e.cid]]
            if doc.get("votes") and e.cid in doc["votes"]:
                e.votes = [PromoAssessment.model_validate(v) if v else None for v in doc["votes"][e.cid]]
    ctx.semantic_info = doc


# =========================================================================== rejection accounting
STRUCTURAL_CODES = [
    (re.compile(r"does not exist"), "unknown_line_id"),
    (re.compile(r"outside this window"), "id_outside_window"),
    (re.compile(r"comes after"), "segment_start_after_end"),
    (re.compile(r"spans two episodes"), "segment_spans_two_episodes"),
    (re.compile(r"segments \d+ and \d+ overlap"), "segments_overlap_in_ids"),
    (re.compile(r"hook_line_id.*first segment"), "hook_not_in_first_segment"),
    (re.compile(r"cliffhanger_line_id.*last segment"), "cliffhanger_not_in_last_segment"),
    (re.compile(r"open_question"), "vague_open_question"),
]


def classify_structural(reason: str) -> str:
    return next((code for rx, code in STRUCTURAL_CODES if rx.search(reason)), "other_structural")


def classify_deterministic(g: GateResult) -> str:
    d = g.details
    if g.gate == "timestamps_valid":
        return "missing_or_invalid_timestamp"
    if g.gate == "duration_20_60s":
        return "too_short" if d.get("total_ms", 0) < DEFAULT_CONFIG.min_clip_ms else "too_long"
    if g.gate == "hook_timing_3s":
        return "hook_not_placeable" if d.get("offset_ms") is None else "hook_after_limit_no_valid_boundary"
    if g.gate == "cliffhanger_reference":
        return "cliffhanger_outside_clip" if "not inside" in g.reason else "trailing_footage_untrimmable"
    if g.gate == "boundary_integrity":
        return "dialogue_cut"
    if g.gate == "segments_distinct":
        return "segments_overlap_in_time"
    return "other"


def classify_semantic(g: GateResult, a: Optional[PromoAssessment] = None) -> str:
    """Reason code = the quality level behind the failure.  With voting, the gate holds no single level, so use the
    aggregate assessment's level for that gate (the most common level among the votes that agreed with the decision)."""
    if g.details.get("level"):
        return str(g.details["level"])
    if g.details.get("narrative_coherence"):
        return str(g.details["narrative_coherence"])
    if a is not None:
        lv = {"hook_strength": a.hook_strength.level, "cliffhanger_strength": a.cliffhanger_strength.level,
              "standalone_clarity": a.standalone_clarity.level, "coherence_after_trim": a.narrative_coherence}.get(g.gate)
        if lv:
            return str(lv)
    return "failed"


def collect_rejections(ctx: Ctx, selection: Selection) -> dict[str, Any]:
    """Every (candidate, stage, gate, reason code) counted ONCE, however many times it was observed."""
    seen: set[tuple] = set()
    events: list[dict[str, Any]] = []

    def add(cid: str, stage: str, gate: str, code: str, reason: str) -> None:
        key = (cid, stage, gate, code)
        if key not in seen:
            seen.add(key)
            events.append({"candidate_id": cid, "stage": stage, "gate": gate, "code": code, "reason": reason})

    for r in ctx.structural_rejected:
        cid = f"{r.window_id}:{r.llm_candidate_id}"
        for reason in r.reasons:
            add(cid, "structural", "id_validation", classify_structural(reason), reason)

    decisions = {**selection.decisions, **ctx.early}
    passed_not_selected: list[str] = []
    for ev in ctx.all_ev:
        d = decisions.get(ev.cid)
        if ev.cid in ctx.early:
            add(ev.cid, "selection", "exact_duplicate", "identical_final_windows", ctx.early[ev.cid].reasons[0])
        elif not ev.outcome.accepted:
            for g in ev.outcome.gates:
                if not g.passed:
                    add(ev.cid, "deterministic", g.gate, classify_deterministic(g), g.reason)
        elif d is None:
            continue
        elif d.stage == "semantic":
            failed = [g for g in ev.semantic_gates if not g.passed]
            for g in failed:
                add(ev.cid, "semantic", g.gate, classify_semantic(g, ev.assessment), g.reason)
            if not failed:
                add(ev.cid, "semantic", "semantic_assessment", "no_assessment", "; ".join(d.reasons))
        elif d.stage == "spoiler":
            add(ev.cid, "semantic", "spoiler_guard", "major_late_season_spoiler", "; ".join(d.reasons))
        elif d.stage in ("duplicate", "overlap"):
            add(ev.cid, "selection", d.stage, "overlaps_higher_ranked_promo", "; ".join(d.reasons))
        elif d.status == "passed_not_selected":
            passed_not_selected.append(ev.cid)

    table: dict[tuple, set[str]] = {}
    example: dict[tuple, str] = {}
    for e in events:
        k = (e["stage"], e["gate"], e["code"])
        table.setdefault(k, set()).add(e["candidate_id"])
        example.setdefault(k, e["reason"])
    rows = [{"stage": s, "gate": g, "reason_code": c, "candidates": len(ids), "candidate_ids": sorted(ids), "example_reason": example[(s, g, c)]}
            for (s, g, c), ids in sorted(table.items())]
    stages: dict[str, set[str]] = {}
    for e in events:
        stages.setdefault(e["stage"], set()).add(e["candidate_id"])
    return {"events": events, "rows": rows,
            "unique_candidates_by_stage": {k: len(v) for k, v in sorted(stages.items())},
            "unique_candidates_rejected": len({e["candidate_id"] for e in events}),
            "passed_gates_not_selected": passed_not_selected}


# =========================================================================== final deliverables
def _tier_key(ev: Evaluated, nums: dict[str, int]) -> tuple:
    return rank_key(ev, nums)


def _ordering_note(i: int, ranked: list[Evaluated], nums: dict[str, int]) -> str:
    """What actually decided this promo's position relative to the one above it (from rank_key itself)."""
    if i == 0:
        return "first in the ordering"
    a, b = _tier_key(ranked[i - 1], nums), _tier_key(ranked[i], nums)
    pos = next(k for k in range(len(a)) if a[k] != b[k])
    # rank_key's first three entries are the quality tiers (pull, standalone, coherence); the rest are tie-breakers
    kind = ("tied with the promo above on all three quality tiers; placed after it by a tie-break" if pos >= 3
            else "ranked below the promo above on the quality tiers")
    return f"{kind} - first differing sort key: {RANK_KEY_DESCRIPTION[pos]}"


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def run_rank(ctx: Ctx, command: str) -> tuple[list[Path], dict, dict]:
    run_dir, index = ctx.run_dir, ctx.index
    nums = {eid: ep.episode_number for eid, ep in index.episodes.items()}
    selection = select_promos(ctx.kept_ev, nums, ctx.series_length, max_promos=ctx.cfg["max_promos"], max_per_episode=ctx.cfg["max_per_episode"])
    decisions = {**selection.decisions, **ctx.early}
    rej = collect_rejections(ctx, selection)

    promos = []
    for i, ev in enumerate(selection.ranked):
        rec = promo_record(index, ev, i + 1, decisions[ev.cid])
        rec["ranking_evidence"] = {"sort_key": list(_tier_key(ev, nums)), "sort_key_meaning": RANK_KEY_DESCRIPTION,
                                   "ordering_note": _ordering_note(i, selection.ranked, nums)}
        promos.append(rec)

    # ---- ranked_promos.json / .csv
    atomic_write_json(run_dir / "ranked_promos.json", {
        "run_id": run_dir.name, "disclaimer": ASSESSMENT_DISCLAIMER, "count": len(promos),
        "ranking_rule": {"sort_order": RANK_KEY_DESCRIPTION, "max_promos": ctx.cfg["max_promos"], "max_per_episode": ctx.cfg["max_per_episode"],
                         "overlap_limit_fraction": OVERLAP_MAX_FRACTION, "late_season_fraction": LATE_SEASON_FRACTION},
        "promos": promos})
    cols = ["rank", "candidate_id", "episodes", "clip_windows", "total_duration_s", "hook_offset_s", "hook_line_id", "cliffhanger_line_id",
            "emotional_pull", "standalone_clarity", "narrative_coherence", "spoiler_risk", "one_line_justification", "open_question_llm_claim",
            "cross_episode", "multi_segment", "adjustments"]
    rows = []
    for p in promos:
        q = p["qualitative_assessment"]
        rows.append({"rank": p["rank"], "candidate_id": p["candidate_id"],
                     "episodes": ";".join(sorted({s["episode_id"] for s in p["clip"]["segments"]})),
                     "clip_windows": " | ".join(f"{s['episode_id']} {s['start']['timecode']}-{s['end']['timecode']}" for s in p["clip"]["segments"]),
                     "total_duration_s": p["clip"]["total_duration_s"], "hook_offset_s": p["hook"]["offset_s"],
                     "hook_line_id": p["hook"]["line_id"], "cliffhanger_line_id": p["cliffhanger"]["line_id"],
                     "emotional_pull": q["emotional_pull"], "standalone_clarity": q["standalone_clarity"],
                     "narrative_coherence": q["narrative_coherence"], "spoiler_risk": q["spoiler_risk"],
                     "one_line_justification": p["one_line_justification"], "open_question_llm_claim": p["cliffhanger"]["open_question_llm_claim"],
                     "cross_episode": p["clip"]["cross_episode"], "multi_segment": p["clip"]["multi_segment"],
                     "adjustments": " | ".join(a["reason"] for a in p["adjustments"])})
    atomic_write_text(run_dir / "ranked_promos.csv", _csv_text(rows, cols))

    # ---- rejected_candidates.json
    by_cid: dict[str, list[dict]] = {}
    for e in rej["events"]:
        by_cid.setdefault(e["candidate_id"], []).append(e)
    rejected_docs = []
    for ev in ctx.all_ev:
        d = decisions.get(ev.cid)
        if d is None or d.status == "recommended":
            continue
        base = promo_record(index, ev, None, d) if ev.clip is not None else {"candidate_id": ev.cid, "status": d.status, "decision_stage": d.stage,
                                                                           "decision_reasons": d.reasons, "gates": [g.model_dump() for g in ev.outcome.gates]}
        base["rejection_events"] = by_cid.get(ev.cid, [])
        base["original_candidate"] = ev.cand.model_dump(mode="json")
        rejected_docs.append(base)
    structural_docs = [{"candidate_id": f"{r.window_id}:{r.llm_candidate_id}", "status": "rejected", "decision_stage": "structural",
                        "decision_reasons": r.reasons, "rejection_events": by_cid.get(f"{r.window_id}:{r.llm_candidate_id}", [])}
                       for r in ctx.structural_rejected]
    atomic_write_json(run_dir / "rejected_candidates.json", {
        "disclaimer": ASSESSMENT_DISCLAIMER, "count": len(rejected_docs) + len(structural_docs),
        "rejection_counts": {"unique_candidates_rejected": rej["unique_candidates_rejected"], "by_stage": rej["unique_candidates_by_stage"],
                             "by_gate_and_reason": rej["rows"]},
        "passed_gates_not_selected": rej["passed_gates_not_selected"],
        "candidates": [*structural_docs, *rejected_docs]})

    # ---- usage_report.json
    usage_doc = build_usage_report(ctx)
    atomic_write_json(run_dir / "usage_report.json", usage_doc)

    # ---- report.md
    funnel = build_funnel(ctx, selection, rej)
    atomic_write_text(run_dir / "report.md", render_report(ctx, command, promos, rej, funnel, usage_doc, selection, nums))
    files = [run_dir / n for n in DELIVERABLES]
    return files, ZERO_USAGE, {"ranked": len(promos), **funnel}


def build_funnel(ctx: Ctx, selection: Selection, rej: dict[str, Any]) -> dict[str, Any]:
    ws = ctx.windows
    det_pass = sum(o.accepted for o in ctx.outcomes)
    assessed = [e for e in ctx.kept_ev if e.assessment is not None]
    sem_pass = [e for e in assessed if e.outcome.accepted and all(g.passed for g in e.semantic_gates)]
    return {
        "episodes_discovered_in_input": ctx.series_length, "episodes_processed": len(ctx.ep_ids),
        "windows": len(ws), "candidates_requested_per_window": ctx.cfg["candidates_per_window"],
        "candidates_requested_total": len(ws) * ctx.cfg["candidates_per_window"],
        "candidates_returned_by_llm": sum(w["candidates_returned"] for w in ws),
        "structurally_valid": len(ctx.candidates), "structurally_rejected": len(ctx.structural_rejected),
        "deterministic_pass": det_pass, "deterministic_fail": len(ctx.outcomes) - det_pass,
        "exact_duplicates_dropped": len(ctx.early), "semantic_assessed": len(assessed), "semantic_all_gates_pass": len(sem_pass),
        "ranked": len(selection.ranked), "passed_gates_not_selected": len(rej["passed_gates_not_selected"]),
    }


def build_usage_report(ctx: Ctx) -> dict[str, Any]:
    per_stage = {s: ctx.records[s]["usage"] for s in STAGES if s in ctx.records}
    keys = ("api_calls", "cache_hits", "json_retries", "transport_retries", "retries_total", "prompt_tokens", "output_tokens",
            "thinking_tokens", "total_tokens", "tokens_saved_by_cache", "failure_events")
    totals = {k: sum(u.get(k, 0) for u in per_stage.values()) for k in keys}
    kinds: Counter = Counter()
    for u in per_stage.values():
        kinds.update(u.get("failure_kinds", {}))
    # Work paid for by earlier attempts whose stage was later quarantined and re-run (still real spend).
    earlier: list[dict[str, Any]] = []
    for rec_file in sorted((ctx.run_dir / "state" / "quarantine").glob("*/record.json")):
        try:
            r = json.loads(rec_file.read_text(encoding="utf-8"))
            earlier.append({"stage": r["stage"], "quarantined_as": rec_file.parent.name,
                            **{k: r["usage"].get(k, 0) for k in keys if k != "failure_events"}})
        except (OSError, ValueError, KeyError):
            continue
    n_eps = max(1, len(ctx.ep_ids))
    per_episode = {
        "episodes_processed": len(ctx.ep_ids),
        "note": "actual tokens spent in this run divided by episodes processed; stages served from the cache spent none here",
        "prompt_tokens": round(totals["prompt_tokens"] / n_eps, 1), "output_tokens": round(totals["output_tokens"] / n_eps, 1),
        "total_tokens": round(totals["total_tokens"] / n_eps, 1),
        "by_stage": {s: {"prompt_tokens": round(u["prompt_tokens"] / n_eps, 1), "output_tokens": round(u["output_tokens"] / n_eps, 1)}
                     for s, u in per_stage.items() if u["prompt_tokens"] or u["output_tokens"]}}
    created = json.loads((ctx.run_dir / "run_config.json").read_text(encoding="utf-8")).get("created_at")
    used = sorted({(u["provider"], u["model"]) for u in per_stage.values() if u.get("model")})
    return {
        "run_id": ctx.run_dir.name,
        "run_started_at_utc": created, "report_generated_at_utc": utc_now(),
        "note": "api_calls = real network calls that returned a model answer; cache_hits are counted separately and cost nothing. "
                "Stage usage is taken from each stage's completion record (the invocation that produced it).",
        "provider": ctx.cfg["settings"]["provider"], "model": ctx.cfg["settings"]["model"],
        "provider_and_model_used_by_providers": [{"provider": a, "model": b} for a, b in used],
        "tokens_per_episode": per_episode,
        "proposer_temperature": ctx.cfg["settings"]["temperature"], "judge_temperature": ctx.cfg["judge_temperature"],
        "judge_votes": ctx.cfg["judge_votes"],
        "per_stage": per_stage, "totals": totals, "failure_kinds": dict(kinds),
        "earlier_attempts_usage": earlier,
        "earlier_attempts_note": "Usage of stage attempts that were later quarantined and re-run (not included in totals). A hard crash before "
                                 "a stage finished leaves no usage record, so calls made by such an attempt cannot be reported here." ,
        "failures": [dict(stage=s, **f) for s, u in per_stage.items() for f in u.get("failures", [])],
        "raw_failure_files": ctx.failures.files(),
        "stage_seconds": {s: ctx.records[s]["seconds"] for s in STAGES if s in ctx.records},
        "executed_in_this_invocation": ctx.executed,
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_report(ctx: Ctx, command: str, promos: list[dict], rej: dict, funnel: dict, usage: dict, selection: Selection,
                  nums: dict[str, int]) -> str:
    t, L = usage["totals"], []
    s = ctx.cfg["settings"]
    L += [f"# PromoAI run report - {ctx.run_dir.name}", "",
          f"- Command: `{command}`", f"- Input: `{ctx.cfg['input_path']}` ({ctx.cfg['input_kind']}), fingerprint `{ctx.cfg['input_fingerprint'][:16]}`",
          f"- Run started (UTC): {usage['run_started_at_utc']}",
          f"- Provider / model: `{s['provider']}` / `{s['model']}` (proposer temperature {s['temperature']}, judge temperature {ctx.cfg['judge_temperature']}, judge votes {ctx.cfg['judge_votes']}, majority decides)",
          f"- Prompt versions: detection `{PROMPT_VERSION}`, semantic `{SEMANTIC_PROMPT_VERSION}`; run format v{RUN_FORMAT_VERSION}",
          f"- Settings: window size {ctx.cfg['window_size']}, overlap {ctx.cfg['overlap']}, candidates requested per window {ctx.cfg['candidates_per_window']}, "
          f"max episodes {ctx.cfg['max_episodes'] or 'all'}, cliffhanger tail limit {ctx.tl_config.cliff_tail_max_ms / 1000:.1f}s", "",
          "> " + ASSESSMENT_DISCLAIMER, "", "## Funnel", "",
          _md_table(["step", "count"], [
              ["episodes in input / processed", f"{funnel['episodes_discovered_in_input']} / {funnel['episodes_processed']}"],
              ["windows", funnel["windows"]],
              ["candidates requested (windows x per-window)", f"{funnel['candidates_requested_total']} ({funnel['candidates_requested_per_window']} per window)"],
              ["candidates returned by the model", funnel["candidates_returned_by_llm"]],
              ["valid line references (structural check)", funnel["structurally_valid"]],
              ["rejected by structural check", funnel["structurally_rejected"]],
              ["pass deterministic timeline gates", funnel["deterministic_pass"]],
              ["fail deterministic timeline gates", funnel["deterministic_fail"]],
              ["exact duplicates dropped before judging", funnel["exact_duplicates_dropped"]],
              ["judged by the semantic judge", funnel["semantic_assessed"]],
              ["pass every semantic gate", funnel["semantic_all_gates_pass"]],
              ["**ranked (recommended)**", f"**{funnel['ranked']}**"],
              ["passed all gates but not selected", funnel["passed_gates_not_selected"]]]), ""]

    L += ["## Windows", "", _md_table(["window", "episodes", "requested", "returned", "valid refs", "rejected (structure)", "status"],
          [[w["window_id"], f"{w['episode_ids'][0]}..{w['episode_ids'][-1]}", ctx.cfg["candidates_per_window"], w["candidates_returned"],
            w["candidates_accepted"], w["candidates_rejected"], w["status"]] for w in ctx.windows]), ""]

    # ---- top 3
    L += ["## Recommended to post first", ""]
    top = promos[:3]
    if not top:
        L += ["**No candidate qualified.** Nothing passed every mandatory gate, so nothing is recommended. This report does not fill the list with weaker clips.", ""]
    else:
        if len(top) < 3:
            L += [f"**Only {len(top)} promo{'s' if len(top) != 1 else ''} qualified** (fewer than the three requested); no weaker clip was added to fill the list.", ""]
        for p in top:
            q, clip = p["qualitative_assessment"], p["clip"]
            segs = "; ".join(f"episode {x['episode_number']} {x['start']['timecode']}-{x['end']['timecode']} ({x['duration_s']:.1f}s)" for x in clip["segments"])
            det = {g["gate"]: g for g in p["gates"] if g["kind"] == "deterministic"}
            sem = {g["gate"]: g for g in p["gates"] if g["kind"] == "semantic_assessment"}
            L += [f"### #{p['rank']} - {p['candidate_id']} ({clip['total_duration_s']:.1f}s)", "",
                  f"- **Clip:** {segs}",
                  f"- **Ranking evidence (judge tiers{', most common level across the votes' if ctx.cfg['judge_votes'] > 1 else ''}):** emotional pull **{q['emotional_pull']}**, standalone clarity **{q['standalone_clarity']}**, "
                  f"narrative coherence **{q['narrative_coherence']}**, spoiler risk **{q['spoiler_risk']}**",
                  f"- **Position:** {p['ranking_evidence']['ordering_note']}",
                  f"- **Python-verified:** duration - {det['duration_20_60s']['reason']}; hook - {det['hook_timing_3s']['reason']}; "
                  f"ending - {det['cliffhanger_reference']['reason']}; {det['boundary_integrity']['reason']}",
                  f"- **Adjustments by Python:** {'; '.join(a['reason'] for a in p['adjustments']) or 'none'}",
                  f"- **Hook ({p['hook']['kind']}{', ' + p['hook']['speaker'] if p['hook']['speaker'] else ''}):** \"{p['hook']['text'][:160]}\"",
                  f"- **Ending line ({p['cliffhanger']['kind']}{', ' + p['cliffhanger']['speaker'] if p['cliffhanger']['speaker'] else ''}):** \"{p['cliffhanger']['text'][:160]}\"; "
                  f"open question claimed by the proposer: {p['cliffhanger']['open_question_llm_claim']}",
                  f"- **Judge votes per gate:** " + "; ".join(
                      f"{g} {[v['level'] for v in sem[g]['details']['votes']]} -> {'pass' if sem[g]['passed'] else 'fail'} ({sem[g]['details']['passes']}/{sem[g]['details']['judge_votes']})"
                      for g in ("hook_strength", "cliffhanger_strength", "standalone_clarity")) if ctx.cfg["judge_votes"] > 1 else
                  f"- **Judge's gate reasons:** hook - {sem['hook_strength']['reason']} / cliffhanger - {sem['cliffhanger_strength']['reason']} / standalone - {sem['standalone_clarity']['reason']}",
                  f"- **Judge's one-line justification:** {p['one_line_justification']}", ""]
    if len(promos) > 3:
        L += ["## Remaining ranked promos", "", _md_table(["rank", "id", "length", "pull", "standalone", "coherence"],
              [[p["rank"], p["candidate_id"], f"{p['clip']['total_duration_s']:.1f}s", p["qualitative_assessment"]["emotional_pull"],
                p["qualitative_assessment"]["standalone_clarity"], p["qualitative_assessment"]["narrative_coherence"]] for p in promos[3:]]), ""]

    # ---- ranking rule
    L += ["## How the ranking works (from the implementation)", "",
          "Only candidates that pass **every** mandatory gate are ranked. Candidates are ordered lexicographically by this sort key (smaller first). "
          "Items 1-3 are the quality tiers; items 4-6 only break ties between promos that are equal on all three tiers - that is the actual tie-break rule:", ""]
    L += [f"{i}. {d}" for i, d in enumerate(RANK_KEY_DESCRIPTION, 1)]
    L += ["", f"There is no weighted numeric score. After ordering, a promo is skipped if it overlaps >= {OVERLAP_MAX_FRACTION:.0%} of a higher-ranked clip's footage, "
          f"at most {ctx.cfg['max_per_episode']} promos may touch one episode, and the list holds at most {ctx.cfg['max_promos']} (a cap, not a target). "
          f"A strong cross-episode or multi-segment promo is included when one exists. Major spoilers are rejected after {LATE_SEASON_FRACTION:.0%} of the series "
          f"({ctx.series_length} episodes in the input).", ""]

    # ---- rejections
    L += ["## Rejections by gate and reason", "",
          "Each candidate is counted once per (stage, gate, reason). A candidate that fails several gates appears in several rows but only once in the stage totals. "
          "Deterministic failures are Python-enforced on the real timeline; semantic failures are the judge's level-derived assessments.", ""]
    if rej["rows"]:
        L += [_md_table(["stage", "gate", "reason", "candidates", "ids", "example"],
                        [[r["stage"], r["gate"], r["reason_code"], r["candidates"], ", ".join(r["candidate_ids"]), r["example_reason"][:140]] for r in rej["rows"]]), ""]
    else:
        L += ["No candidate was rejected.", ""]
    L += [f"Unique candidates rejected: **{rej['unique_candidates_rejected']}** - by stage: {rej['unique_candidates_by_stage'] or 'none'}. "
          f"Passed every gate but not selected: {rej['passed_gates_not_selected'] or 'none'}.", ""]

    # ---- usage
    L += ["## Usage", "", _md_table(["stage", "API calls", "cache hits", "JSON retries", "transport retries", "input tokens", "output tokens", "failure events"],
          [[st, u["api_calls"], u["cache_hits"], u["json_retries"], u["transport_retries"], u["prompt_tokens"], u["output_tokens"], u["failure_events"]]
           for st, u in usage["per_stage"].items()] +
          [["**total**", t["api_calls"], t["cache_hits"], t["json_retries"], t["transport_retries"], t["prompt_tokens"], t["output_tokens"], t["failure_events"]]]),
          "", f"Tokens per episode ({usage['tokens_per_episode']['episodes_processed']} processed): "
          f"input {usage['tokens_per_episode']['prompt_tokens']:,.0f}, output {usage['tokens_per_episode']['output_tokens']:,.0f} "
          f"(actual spend in this run; cached stages spent nothing). Judge: temperature {ctx.cfg['judge_temperature']}, {ctx.cfg['judge_votes']} vote(s) per judgment; "
          f"provider/model used: {', '.join(m['provider'] + '/' + m['model'] for m in usage['provider_and_model_used_by_providers'])}.", "",
          f"Thinking tokens: {t['thinking_tokens']}. Tokens that cache hits avoided: {t['tokens_saved_by_cache']}. "
          f"Failure kinds: {usage['failure_kinds'] or 'none'}. Raw failed replies saved: {len(usage['raw_failure_files'])} file(s) in `raw_failures/`."
          + (f" Earlier quarantined stage attempts also spent {sum(e['api_calls'] for e in usage['earlier_attempts_usage'])} API call(s) "
             f"(see usage_report.json)." if usage["earlier_attempts_usage"] else ""), ""]

    # ---- errors / limitations
    notes = []
    if ctx.parse_info.get("failed_files"):
        notes.append(f"{len(ctx.parse_info['failed_files'])} input file(s) could not be parsed.")
    if ctx.parse_info.get("episodes_discovered_in_input", 0) > len(ctx.ep_ids):
        notes.append(f"Limited run: only the first {len(ctx.ep_ids)} of {ctx.series_length} episodes were processed; candidates cannot span past them.")
    notes += [("Semantic verdicts are the majority of %d independent judge votes (temperature %s); still model opinions, and an earlier stability experiment saw single-vote verdicts change between identical temperature-0 runs." % (ctx.cfg["judge_votes"], ctx.cfg["judge_temperature"]))
              if ctx.cfg["judge_votes"] > 1 else
              ("Semantic verdicts are single-judge model opinions (judge votes = 1); an earlier stability experiment saw 1 of 24 gate decisions change between identical temperature-0 runs."),
              "Ranking quality has not been validated against human judgement.",
              f"Spoiler checks only see what a clip itself reveals (the judge does not know the ending); the late-season rule needs a 'major' verdict past {LATE_SEASON_FRACTION:.0%} of the series."]
    if usage["failure_kinds"]:
        notes.append(f"Recorded failure events: {usage['failure_kinds']}.")
    L += ["## Errors and limitations", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


# =========================================================================== engine
def _public_settings(s: Settings) -> dict[str, Any]:
    return {"provider": s.provider, "model": s.model, "temperature": s.temperature, "max_output_tokens": s.max_output_tokens}


def fingerprint_input(kind: str, path: Path) -> str:
    if kind == "zip":
        return sha256_file(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.txt")):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def resolve_input(raw: str) -> tuple[str, Path]:
    p = Path(raw).expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".zip":
        return "zip", p
    if p.is_dir():
        zips = find_archives(p)
        if len(zips) == 1:
            return "zip", zips[0].resolve()
        if len(zips) > 1:
            raise RunError(f"{p} contains {len(zips)} ZIP archives; pass the one you want with --input")
        if any(True for _ in p.rglob("*.txt")):
            return "folder", p
    raise RunError(f"--input must be a .zip file, a folder containing one .zip, or a folder of episode .txt files: {raw}")


def default_factory(settings: Settings, role: str) -> LLMProvider:
    """Same provider abstraction for both roles; the judge only differs in temperature."""
    return create_provider(settings)


def execute(ctx: Ctx, command: str) -> None:
    stages = [("01_parse", run_parse, load_parse), ("02_detect", run_detect, load_detect),
              ("03_timeline", run_timeline, load_timeline), ("04_semantic", run_semantic_stage, load_semantic)]
    force = False
    for name, run_fn, load_fn in stages:
        rec = None if force else ctx.store.valid_record(name)
        if rec is not None:
            load_fn(ctx)
            ctx.records[name] = rec
            print(f"[{name}] already complete - verified and loaded")
            continue
        q = ctx.store.quarantine(name)
        if q:
            print(f"[{name}] incomplete leftovers moved to {q.relative_to(ctx.run_dir).as_posix()}")
        force = True
        print(f"[{name}] running ...", flush=True)
        t0 = time.time()
        files, usage, counts = run_fn(ctx)
        ctx.records[name] = ctx.store.write_record(name, files, usage, counts, time.time() - t0)
        ctx.executed.append(name)
        print(f"[{name}] done in {time.time() - t0:.1f}s  {counts}")
    rec = None if force else ctx.store.valid_record("05_rank")
    if rec is not None:
        ctx.records["05_rank"] = rec
        print("[05_rank] already complete - verified")
        return
    ctx.store.quarantine("05_rank")
    for n in DELIVERABLES:       # a half-finished earlier attempt may have left some deliverables; keep them, don't trust them
        f = ctx.run_dir / n
        if f.exists():
            dest = ctx.store.state / "quarantine" / f"deliverable-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / n))
    print("[05_rank] running ...", flush=True)
    t0 = time.time()
    # usage_report needs this stage's own (zero) record to exist in memory before it is written
    ctx.records["05_rank"] = {"usage": ZERO_USAGE, "seconds": 0.0}
    ctx.executed.append("05_rank")
    files, usage, counts = run_rank(ctx, command)
    ctx.records["05_rank"] = ctx.store.write_record("05_rank", files, usage, counts, time.time() - t0)
    print(f"[05_rank] done in {time.time() - t0:.1f}s")


def print_summary(ctx: Ctx) -> None:
    rep = json.loads((ctx.run_dir / "usage_report.json").read_text(encoding="utf-8"))
    rj = json.loads((ctx.run_dir / "rejected_candidates.json").read_text(encoding="utf-8"))
    ranked = json.loads((ctx.run_dir / "ranked_promos.json").read_text(encoding="utf-8"))
    f = build_funnel_from_files(ctx)
    t = rep["totals"]
    print("\n" + "=" * 72)
    print(f"RUN COMPLETE  {ctx.run_dir}")
    print(f"episodes processed {f['episodes_processed']} of {f['episodes_discovered_in_input']} | windows {f['windows']} | "
          f"candidates: requested {f['candidates_requested_total']}, returned {f['candidates_returned_by_llm']}, valid refs {f['structurally_valid']}")
    print(f"deterministic: {f['deterministic_pass']} pass / {f['deterministic_fail']} fail | semantic: {f['semantic_assessed']} judged, "
          f"{f['semantic_all_gates_pass']} pass all gates | RANKED: {ranked['count']}")
    print("rejections by gate/reason (candidates counted once per row):")
    for r in rj["rejection_counts"]["by_gate_and_reason"]:
        print(f"  [{r['stage']}] {r['gate']} / {r['reason_code']}: {r['candidates']}  ({', '.join(r['candidate_ids'])})")
    if not rj["rejection_counts"]["by_gate_and_reason"]:
        print("  none")
    for w in ctx.windows:
        print(f"  window {w['window_id']}: {w['episode_ids'][0]}..{w['episode_ids'][-1]} | requested {ctx.cfg['candidates_per_window']}, returned {w['candidates_returned']}, "
              f"valid {w['candidates_accepted']}, structurally rejected {w['candidates_rejected']}")
    te = rep["tokens_per_episode"]
    print(f"judge: temperature {ctx.cfg['judge_temperature']}, {ctx.cfg['judge_votes']} votes | model {', '.join(m['model'] for m in rep['provider_and_model_used_by_providers'])}")
    print(f"tokens per episode ({te['episodes_processed']} processed): in {te['prompt_tokens']:,.0f} / out {te['output_tokens']:,.0f}")
    print(f"usage: API calls {t['api_calls']} | cache hits {t['cache_hits']} | tokens in/out {t['prompt_tokens']:,}/{t['output_tokens']:,} "
          f"(thinking {t['thinking_tokens']}) | retries {t['retries_total']} (json {t['json_retries']}, transport {t['transport_retries']}) | failure events {t['failure_events']}")
    print("files: " + ", ".join(DELIVERABLES))
    print("=" * 72)


def build_funnel_from_files(ctx: Ctx) -> dict[str, Any]:
    ws = ctx.windows
    det_pass = sum(o.accepted for o in ctx.outcomes)
    assessed = [e for e in ctx.kept_ev if e.assessment is not None]
    return {"episodes_discovered_in_input": ctx.series_length, "episodes_processed": len(ctx.ep_ids), "windows": len(ws),
            "candidates_requested_total": len(ws) * ctx.cfg["candidates_per_window"],
            "candidates_returned_by_llm": sum(w["candidates_returned"] for w in ws), "structurally_valid": len(ctx.candidates),
            "deterministic_pass": det_pass, "deterministic_fail": len(ctx.outcomes) - det_pass, "semantic_assessed": len(assessed),
            "semantic_all_gates_pass": sum(1 for e in assessed if e.outcome.accepted and all(g.passed for g in e.semantic_gates))}


# =========================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run.py", description="PromoAI end to end: parse -> story context -> detect -> validate -> judge -> rank")
    ap.add_argument("--input", required=True, help="a .zip, a folder containing one .zip, or a folder of episode .txt files")
    ap.add_argument("--max-episodes", type=int, default=None, help="process only the first N discovered episodes (limited dry run)")
    ap.add_argument("--window-size", type=int, default=6)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--candidates-per-window", type=int, default=DEFAULT_MAX_CANDIDATES,
                    help=f"promo candidates requested from the model per window (default {DEFAULT_MAX_CANDIDATES})")
    ap.add_argument("--eval-batch", type=int, default=5, help="clips per judge call (same window only)")
    ap.add_argument("--max-promos", type=int, default=DEFAULT_MAX_PROMOS)
    ap.add_argument("--max-per-episode", type=int, default=DEFAULT_MAX_PER_EPISODE)
    ap.add_argument("--cliff-tail-max-s", type=float, default=DEFAULT_CONFIG.cliff_tail_max_ms / 1000)
    ap.add_argument("--judge-temperature", type=float, default=None, help="default: LLM_JUDGE_TEMPERATURE or 0.0")
    ap.add_argument("--judge-votes", type=int, default=None,
                    help=f"independent judge votes per clip, majority decides (default: LLM_JUDGE_VOTES if set, else {RUN_DEFAULT_JUDGE_VOTES})")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--resume", type=Path, default=None, help="continue a crashed run directory (finished stages are verified, not redone)")
    return ap


def main(argv: Optional[list[str]] = None, *, provider_factory: Optional[Callable[[Settings, str], LLMProvider]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")       # never crash on a character the console can't show
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    command = "python run.py " + " ".join(f'"{a}"' if " " in a else a for a in (sys.argv[1:] if argv is None else argv))
    try:
        kind, path = resolve_input(args.input)
        settings = load_settings()
        if provider_factory is None and (not settings.model or not settings.api_key):
            raise ConfigError("LLM_MODEL and the provider API key must be set in .env or the environment "
                              "(see .env.example; run `python -m promoai.llm --list-models` to pick a model)")
        judge_t = settings.judge_temperature if args.judge_temperature is None else args.judge_temperature
        judge_v = args.judge_votes if args.judge_votes is not None else (settings.judge_votes if settings.judge_votes_set else RUN_DEFAULT_JUDGE_VOTES)
        if min(args.window_size, args.candidates_per_window, args.eval_batch, args.max_promos, args.max_per_episode, judge_v) < 1 \
                or not 0 <= args.overlap < args.window_size or judge_t < 0 or args.cliff_tail_max_s < 0 \
                or (args.max_episodes is not None and args.max_episodes < 1):
            raise RunError("invalid numeric option (sizes/counts must be >= 1, 0 <= overlap < window size, temperatures/limits >= 0)")
        fingerprint = fingerprint_input(kind, path)
        factory = provider_factory or (lambda s, role: create_provider(dataclasses.replace(s, temperature=judge_t) if role == "judge" else s))
        cfg = {"run_format_version": RUN_FORMAT_VERSION, "input_path": str(path), "input_kind": kind, "input_fingerprint": fingerprint,
               "max_episodes": args.max_episodes, "window_size": args.window_size, "overlap": args.overlap,
               "candidates_per_window": args.candidates_per_window, "eval_batch": args.eval_batch, "max_promos": args.max_promos,
               "max_per_episode": args.max_per_episode, "cliff_tail_max_s": args.cliff_tail_max_s,
               "judge_temperature": judge_t, "judge_votes": judge_v, "settings": _public_settings(settings)}

        if args.resume:
            run_dir = args.resume.resolve()
            if not (run_dir / "run_config.json").is_file():
                raise RunError(f"not a run directory (no run_config.json): {run_dir}")
            old = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["config"]
            if old["input_fingerprint"] != fingerprint:
                raise RunError("the input differs from the one this run was started with; refusing to resume")
            if old["settings"] != cfg["settings"] or old["judge_temperature"] != judge_t:
                raise RunError("provider/model/temperature changed since this run started; refusing to resume")
            cfg = old                                  # resume with the ORIGINAL parameters
            print(f"resuming {run_dir}")
        else:
            base = args.runs_dir / slugify(path.stem.removesuffix("-script"))[:24].strip("-")     # short: keeps Windows paths safe
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = (base / f"run-v{RUN_FORMAT_VERSION}-{stamp}-{secrets.token_hex(2)}").resolve()
            run_dir.mkdir(parents=True, exist_ok=False)      # refuses to reuse an existing run directory
            atomic_write_json(run_dir / "run_config.json", {"created_at": utc_now(), "command": command, "config": cfg})
    except (RunError, ConfigError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ctx = Ctx(run_dir=run_dir, cfg=cfg, settings=settings, factory=factory, store=StageStore(run_dir),
              failures=FailureLog(run_dir / "raw_failures"), tl_config=TimelineConfig(cliff_tail_max_ms=round(cfg["cliff_tail_max_s"] * 1000)))
    print(f"run directory: {run_dir}")
    try:
        execute(ctx, command)
    except StageIncomplete as exc:
        atomic_write_json(run_dir / "state" / "last_failure.json", {"at": utc_now(), "stage": exc.stage, "reason": exc.reason, "usage": exc.usage,
                                                                      "raw_failure_files": ctx.failures.files()})
        print(f"\nSTOPPED in {exc.stage}: {exc.reason}", file=sys.stderr)
        if exc.usage:
            u = exc.usage
            print(f"usage so far in this stage: API calls {u['api_calls']}, cache hits {u['cache_hits']}, tokens {u['prompt_tokens']:,}/{u['output_tokens']:,}", file=sys.stderr)
        print(f"nothing was marked complete for {exc.stage}. Resume with:  python run.py --input \"{cfg['input_path']}\" --resume \"{run_dir}\"", file=sys.stderr)
        return 1
    except (RunError, LLMError, ConfigError) as exc:
        atomic_write_json(run_dir / "state" / "last_failure.json", {"at": utc_now(), "error": type(exc).__name__, "reason": str(exc)})
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print_summary(ctx)
    return 0
