"""Verification against the real (confidential) screenplay pack.

These tests are skipped automatically when data/input has no ZIP, so the repository's
public test-suite stays green without the private data.

Independence matters: the checks below deliberately use their own tiny regexes and
arithmetic instead of the parser's code, otherwise a parser bug would also hide in
the test.
"""

from __future__ import annotations

import re
import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

from promoai.index import ScreenplayIndex
from promoai.ingest import find_archives
from promoai.models import TimestampStatus
from promoai.pipeline import run
from promoai.parser import split_source_lines

pytestmark = pytest.mark.real_data

INPUT = Path(__file__).resolve().parent.parent / "data" / "input"
ARCHIVES = find_archives(INPUT) if INPUT.is_dir() else []

if not ARCHIVES:
    pytest.skip("no screenplay ZIP in data/input", allow_module_level=True)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("real")
    out = run(INPUT, root / "extracted", root / "output")
    return out[0]  # (series, summary, out_dir)


@pytest.fixture(scope="module")
def raw_texts(result):
    series, _, _ = result
    ex = series.extract
    return {p.relative_to(ex.extract_dir).as_posix(): p.read_text(encoding="utf-8") for p in ex.script_files}


def _sec(ts: str) -> int:
    """Independent timestamp -> ms conversion (Decimal, no shared code)."""
    mm, rest = ts.split(":")
    return int((Decimal(mm) * 60 + Decimal(rest)) * 1000)


def test_episode_count_matches_the_zip_itself(result):
    series, summary, _ = result
    with zipfile.ZipFile(ARCHIVES[0]) as z:
        txt_members = [n for n in z.namelist() if n.lower().endswith(".txt")]
    assert summary.files_discovered == len(txt_members)
    assert summary.episodes_parsed == len(txt_members)
    assert summary.failed_files == []


def test_no_file_failed_and_every_source_line_is_accounted_for(result, raw_texts):
    series, summary, _ = result
    assert summary.all_files_fully_accounted
    for ep in series.episodes:
        lines = split_source_lines(raw_texts[ep.source_file])
        assert ep.coverage.total_lines == len(lines)
        assert ep.coverage.non_blank_lines == sum(1 for l in lines if l.strip())  # independent count
        assert ep.coverage.parsed_lines + ep.coverage.unparsed_lines == ep.coverage.non_blank_lines


def test_parsed_line_count_equals_independent_label_count(result, raw_texts):
    """Count '**...**' label lines straight from the raw files; parser must agree."""
    series, _, _ = result
    for ep in series.episodes:
        n_labels = sum(1 for l in raw_texts[ep.source_file].split("\n") if re.fullmatch(r"\*\*.+\*\*", l.strip()) or re.fullmatch(r"\*\*[^*]+:\*\*", l.strip()))
        assert len(ep.lines) == n_labels, ep.episode_id
    n_scene_markers = {ep.source_file: len(re.findall(r"^SCENE \d+\s*$", raw_texts[ep.source_file], re.M)) for ep in series.episodes}
    for ep in series.episodes:
        assert len(ep.scenes) == n_scene_markers[ep.source_file], ep.episode_id


def test_unique_ids_across_whole_dataset(result):
    series, _, _ = result
    scene_ids = [s.scene_id for e in series.episodes for s in e.scenes]
    line_ids = [l.line_id for e in series.episodes for l in e.lines]
    assert len(set(scene_ids)) == len(scene_ids)
    assert len(set(line_ids)) == len(line_ids)
    assert len({e.episode_id for e in series.episodes}) == len(series.episodes)


def test_episode_identification(result):
    series, _, _ = result
    for ep in series.episodes:
        m = re.match(r".*episode-(\d+)-Episode (\d+)\.txt$", ep.source_file)
        assert ep.episode_number == int(m.group(1))
        assert ep.platform_id == m.group(2)
        assert ep.episode_number_source == "header"
        assert ep.episode_id == f"E{ep.episode_number:03d}"
    nums = [e.episode_number for e in series.episodes]
    assert nums == sorted(nums) and len(set(nums)) == len(nums)


def test_every_line_id_maps_back_to_original_source(result, raw_texts):
    series, _, out_dir = result
    idx = ScreenplayIndex.from_output_dir(out_dir)     # load from the JSON files on disk
    total = 0
    for ep in series.episodes:
        src = split_source_lines(raw_texts[ep.source_file])
        for l in ep.lines:
            total += 1
            got = idx.get_line(l.line_id)
            label = src[got.source_line - 1].strip()
            assert label.startswith("**") and got.speaker_raw in label
            m = re.fullmatch(r"Start: \((\S+)\) \| End: \((\S+)\)", src[got.source_line].strip())
            assert m, (l.line_id, src[got.source_line])
            assert (got.start_raw, got.end_raw) == (m.group(1), m.group(2))         # original timestamp preserved
            assert (got.start_ms, got.end_ms) == (_sec(m.group(1)), _sec(m.group(2)))  # converted correctly
            assert got.timestamp_status is TimestampStatus.OK
            body = "\n".join(src[got.source_line + 1 : got.source_end_line])
            assert body == got.text_raw
            assert got.source_file == ep.source_file
    assert total == sum(len(e.lines) for e in series.episodes) > 0


def test_scene_timestamps_preserved(result, raw_texts):
    series, _, _ = result
    for ep in series.episodes:
        txt = raw_texts[ep.source_file]
        starts = re.findall(r"^Scene start:\s*(\S+)", txt, re.M | re.I)
        ends = re.findall(r"^Scene end:\s*(\S+)", txt, re.M | re.I)
        assert [s.start_raw for s in ep.scenes] == starts
        assert [s.end_raw for s in ep.scenes] == ends
        assert all(not s.times_derived for s in ep.scenes)


def test_dialogue_text_comes_from_the_source(result, raw_texts):
    series, _, _ = result
    for ep in series.episodes:
        txt = raw_texts[ep.source_file]
        for l in ep.lines:
            if l.kind.value == "dialogue":
                assert l.speaker and f'"{l.text}"' in txt


def test_known_quirks_are_surfaced_not_hidden(result):
    """The pack has exactly one block that runs past its scene end (episode 22)."""
    series, summary, _ = result
    odd = [w for e in series.episodes for w in e.warnings if w.code == "line_outside_scene"]
    assert summary.warnings == len(sum((e.warnings for e in series.episodes), []))
    assert all(w.code in {"line_outside_scene"} for e in series.episodes for w in e.warnings), summary.warnings_by_code
    assert len(odd) == 1 and odd[0].episode_id == "E022"


def test_output_json_exists_for_every_episode(result):
    series, _, out_dir = result
    files = sorted(p.name for p in (out_dir / "episodes").glob("*.json"))
    assert files == [f"{e.episode_id}.json" for e in series.episodes]
    assert (out_dir / "dataset_summary.json").is_file()
