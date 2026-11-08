"""Phase 1 batch runner: ZIP -> extracted scripts -> parsed episodes -> JSON + summary.

    python -m promoai.pipeline                       # uses data/input, data/extracted, data/output

Outputs (all git-ignored, they contain confidential screenplay text):
    data/output/<series-slug>/episodes/E001.json ...
    data/output/<series-slug>/dataset_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ingest import ExtractResult, extract_archive, find_archives, slugify
from .models import (
    DatasetSummary,
    Episode,
    FailedFile,
    ParseWarning,
    Severity,
    TimestampStatus,
)
from .parser import episode_info_from_filename, parse_episode_text


@dataclass
class ParsedSeries:
    extract: ExtractResult
    episodes: list[Episode] = field(default_factory=list)
    failed: list[FailedFile] = field(default_factory=list)


def decode_script(data: bytes) -> tuple[str, Optional[str]]:
    """bytes -> (text, note).  ``note`` is set when we had to leave UTF-8."""
    try:
        return data.decode("utf-8-sig"), None  # utf-8-sig also strips a BOM
    except UnicodeDecodeError:
        pass
    for enc in ("cp1252",):
        try:
            return data.decode(enc), f"file is not valid UTF-8; decoded as {enc}"
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1"), "file is not valid UTF-8; decoded as latin-1 (lossless but may show wrong characters)"


def parse_archive(archive: Path, extract_root: Path) -> ParsedSeries:
    return parse_extracted(extract_archive(archive, extract_root))


def parse_extracted(ex: ExtractResult) -> ParsedSeries:
    """Parse the script files listed in ``ex`` (from a ZIP extraction or a plain folder)."""
    out = ParsedSeries(extract=ex)

    loaded: list[tuple[Path, str, str, str, Optional[str]]] = []  # path, rel, text, sha, note
    for path in ex.script_files:
        rel = path.relative_to(ex.extract_dir).as_posix()
        try:
            data = path.read_bytes()
            text, note = decode_script(data)
            loaded.append((path, rel, text, hashlib.sha256(data).hexdigest(), note))
        except Exception as exc:  # unreadable file must not stop the batch
            out.failed.append(FailedFile(source_file=rel, reason=f"{type(exc).__name__}: {exc}"))

    parsed: list[tuple[tuple, Episode]] = []
    for item in loaded:
        path, rel, text, sha, note = item
        number, platform_id = episode_info_from_filename(path.name)
        try:
            ep = parse_episode_text(
                text, rel, fallback_number=number, platform_id=platform_id,
                series_title=ex.series_title, source_sha256=sha,
            )
        except Exception as exc:
            out.failed.append(FailedFile(source_file=rel, reason=f"{type(exc).__name__}: {exc}"))
            continue
        if note:
            ep.warnings.append(ParseWarning(code="non_utf8_encoding", message=note, source_file=rel, episode_id=ep.episode_id))
        parsed.append((item, ep))

    # ---- duplicate episode numbers: first file keeps the number, later ones get fresh ones
    seen: set[int] = set()
    next_free = max((ep.episode_number for _, ep in parsed), default=0) + 1
    for item, ep in parsed:
        if ep.episode_number in seen:
            path, rel, text, sha, note = item
            number, platform_id = episode_info_from_filename(path.name)
            old = ep.episode_number
            ep = parse_episode_text(
                text, rel, fallback_number=number, number_override=next_free,
                platform_id=platform_id, series_title=ex.series_title, source_sha256=sha,
            )
            ep.warnings.append(ParseWarning(
                code="duplicate_episode_number",
                message=f"episode number {old} already used by another file; assigned {next_free}",
                severity=Severity.WARNING, source_file=rel, episode_id=ep.episode_id,
            ))
            next_free += 1
        seen.add(ep.episode_number)
        out.episodes.append(ep)

    out.episodes.sort(key=lambda e: e.episode_number)

    # ---- files that produced nothing usable are reported as failed (but still saved for debugging)
    for ep in out.episodes:
        if not ep.lines:
            out.failed.append(FailedFile(source_file=ep.source_file, reason="no dialogue/action lines could be extracted"))
    return out


def build_summary(series: ParsedSeries) -> DatasetSummary:
    usable = [e for e in series.episodes if e.lines]
    lines = [ln for e in series.episodes for ln in e.lines]
    kinds = Counter(ln.kind.value for ln in lines)
    ts = Counter(ln.timestamp_status.value for ln in lines)
    warns = [w for e in series.episodes for w in e.warnings]
    return DatasetSummary(
        series_title=series.extract.series_title,
        source_archives=[series.extract.archive.name],
        files_discovered=len(series.extract.script_files),
        episodes_parsed=len(usable),
        scenes=sum(len(e.scenes) for e in series.episodes),
        lines=len(lines),
        lines_by_kind=dict(sorted(kinds.items())),
        lines_with_valid_timestamps=ts.get(TimestampStatus.OK.value, 0),
        lines_without_valid_timestamps=len(lines) - ts.get(TimestampStatus.OK.value, 0),
        timestamp_status_counts=dict(sorted(ts.items())),
        warnings=len(warns),
        warnings_by_code=dict(sorted(Counter(w.code for w in warns).items())),
        unparsed_entries=sum(len(e.unparsed) for e in series.episodes),
        total_non_blank_source_lines=sum(e.coverage.non_blank_lines for e in series.episodes),
        total_accounted_source_lines=sum(e.coverage.parsed_lines + e.coverage.unparsed_lines for e in series.episodes),
        all_files_fully_accounted=all(e.coverage.fully_accounted for e in series.episodes) and bool(series.episodes),
        total_duration_ms=sum(e.duration_ms or 0 for e in series.episodes),
        failed_files=series.failed,
        ignored_files=series.extract.ignored,
    )


def write_outputs(series: ParsedSeries, summary: DatasetSummary, output_root: Path) -> Path:
    out_dir = output_root / slugify(series.extract.series_title)
    ep_dir = out_dir / "episodes"
    if ep_dir.exists():
        shutil.rmtree(ep_dir)  # our own generated folder; avoids stale episodes
    ep_dir.mkdir(parents=True)
    for ep in series.episodes:
        (ep_dir / f"{ep.episode_id}.json").write_text(ep.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "dataset_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return out_dir


def run(input_dir: Path, extract_root: Path, output_root: Path) -> list[tuple[ParsedSeries, DatasetSummary, Path]]:
    archives = find_archives(input_dir)
    if not archives:
        raise FileNotFoundError(f"no .zip archive found in {input_dir}")
    results = []
    for archive in archives:
        series = parse_archive(archive, extract_root)
        summary = build_summary(series)
        out_dir = write_outputs(series, summary, output_root)
        results.append((series, summary, out_dir))
    return results


def _fmt(ms: int) -> str:
    return f"{ms / 60000:.1f} min"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PromoAI phase 1: parse screenplay ZIP into structured JSON")
    ap.add_argument("--input-dir", type=Path, default=Path("data/input"))
    ap.add_argument("--extract-dir", type=Path, default=Path("data/extracted"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/output"))
    args = ap.parse_args(argv)

    try:
        results = run(args.input_dir, args.extract_dir, args.output_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for series, s, out_dir in results:
        print(f"\n=== {s.series_title} ===  ({s.source_archives[0]})")
        print(f"  files discovered           : {s.files_discovered}")
        print(f"  episodes parsed            : {s.episodes_parsed}")
        print(f"  scenes                     : {s.scenes}")
        print(f"  dialogue/action lines      : {s.lines}  {s.lines_by_kind}")
        print(f"  lines with valid timestamps: {s.lines_with_valid_timestamps}/{s.lines}")
        print(f"  parsing warnings           : {s.warnings}  {s.warnings_by_code}")
        print(f"  unparsed entries           : {s.unparsed_entries}")
        print(f"  source lines accounted     : {s.total_accounted_source_lines}/{s.total_non_blank_source_lines}"
              f"  (all files fully accounted: {s.all_files_fully_accounted})")
        print(f"  total runtime              : {_fmt(s.total_duration_ms)}")
        print(f"  failed files               : {[f.model_dump() for f in s.failed_files] or 'none'}")
        print(f"  ignored archive members    : {s.ignored_files or 'none'}")
        print(f"  output written to          : {out_dir}")
    return 0 if all(not s.failed_files for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
