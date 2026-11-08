"""ZIP extraction + batch pipeline on synthetic archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from conftest import make_script
from promoai.ingest import ArchiveError, extract_archive, find_archives
from promoai.pipeline import build_summary, parse_archive, run, write_outputs


def make_zip(path: Path, files: dict[str, str | bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return path


def episode_files(n: int, folder: str = "Show") -> dict[str, str]:
    return {f"{folder}/episode-{i:03d}-Episode {1000 + i}.txt": make_script(i) for i in range(1, n + 1)}


@pytest.fixture
def dirs(tmp_path):
    d = {k: tmp_path / k for k in ("input", "extracted", "output")}
    d["input"].mkdir()
    return d


# ------------------------------------------------------------------ discovery
@pytest.mark.parametrize("n", [1, 4, 13])
def test_episode_count_is_discovered_dynamically(dirs, n):
    make_zip(dirs["input"] / "show.zip", episode_files(n))
    [(series, summary, _)] = run(dirs["input"], dirs["extracted"], dirs["output"])
    assert summary.files_discovered == n
    assert summary.episodes_parsed == n
    assert [e.episode_id for e in series.episodes] == [f"E{i:03d}" for i in range(1, n + 1)]


def test_no_zip_gives_clear_error(dirs):
    with pytest.raises(FileNotFoundError, match="no .zip"):
        run(dirs["input"], dirs["extracted"], dirs["output"])


def test_find_archives_only_returns_zips(dirs):
    (dirs["input"] / "readme.txt").write_text("x")
    make_zip(dirs["input"] / "A.ZIP", {"a.txt": "x"})
    assert [p.name for p in find_archives(dirs["input"])] == ["A.ZIP"]


def test_non_script_members_and_macos_junk_are_ignored(dirs):
    files = episode_files(2)
    files["Show/cover.jpg"] = b"\xff\xd8"
    files["__MACOSX/Show/._episode-001-Episode 1001.txt"] = b"junk"
    z = make_zip(dirs["input"] / "show.zip", files)
    res = extract_archive(z, dirs["extracted"])
    assert len(res.script_files) == 2
    assert res.ignored == ["Show/cover.jpg"]


def test_natural_sort_orders_episode_10_after_9(dirs):
    z = make_zip(dirs["input"] / "show.zip", episode_files(12))
    names = [p.name for p in extract_archive(z, dirs["extracted"]).script_files]
    assert names[8].startswith("episode-009") and names[9].startswith("episode-010")


# ------------------------------------------------------------------ safety
def test_zip_slip_is_rejected(dirs):
    z = make_zip(dirs["input"] / "evil.zip", {"../../evil.txt": "x"})
    with pytest.raises(ArchiveError, match="unsafe path"):
        extract_archive(z, dirs["extracted"])
    assert not (dirs["extracted"].parent / "evil.txt").exists()


def test_corrupt_zip_is_reported(dirs):
    (dirs["input"] / "bad.zip").write_bytes(b"this is not a zip")
    with pytest.raises(ArchiveError, match="not a valid ZIP"):
        extract_archive(dirs["input"] / "bad.zip", dirs["extracted"])


def test_stale_extraction_is_wiped(dirs):
    z = make_zip(dirs["input"] / "show.zip", episode_files(3))
    extract_archive(z, dirs["extracted"])
    z = make_zip(dirs["input"] / "show.zip", episode_files(2))  # smaller archive replaces it
    assert len(extract_archive(z, dirs["extracted"]).script_files) == 2


# ------------------------------------------------------------------ robustness
def test_bad_files_are_reported_but_do_not_stop_the_batch(dirs):
    files = episode_files(3)
    files["Show/episode-004-Episode 1004.txt"] = ""                                # empty
    files["Show/episode-005-Episode 1005.txt"] = "garbage\nwith no structure\n"   # unparseable
    make_zip(dirs["input"] / "show.zip", files)
    [(series, summary, _)] = run(dirs["input"], dirs["extracted"], dirs["output"])
    assert summary.files_discovered == 5
    assert summary.episodes_parsed == 3
    failed = {f.source_file.split("/")[-1] for f in summary.failed_files}
    assert failed == {"episode-004-Episode 1004.txt", "episode-005-Episode 1005.txt"}
    assert summary.unparsed_entries == 2          # the garbage lines were kept, not dropped


def test_non_utf8_file_still_parses_with_warning(dirs):
    files = episode_files(1)
    raw = make_script(2).replace("Ann enters.", "Ann café enters.").encode("cp1252")
    z = dirs["input"] / "show.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Show/episode-001-Episode 1001.txt", files["Show/episode-001-Episode 1001.txt"])
        zf.writestr("Show/episode-002-Episode 1002.txt", raw)
    series = parse_archive(z, dirs["extracted"])
    ep2 = series.episodes[1]
    assert any(w.code == "non_utf8_encoding" for w in ep2.warnings)
    assert "café" in ep2.lines[1].text


def test_duplicate_episode_numbers_get_distinct_ids(dirs):
    files = {
        "Show/episode-001-Episode 1001.txt": make_script(1),
        "Show/episode-002-Episode 1002.txt": make_script(1),   # header also says 1
    }
    make_zip(dirs["input"] / "show.zip", files)
    series = parse_archive(dirs["input"] / "show.zip", dirs["extracted"])
    ids = [e.episode_id for e in series.episodes]
    assert len(set(ids)) == 2
    all_line_ids = [l.line_id for e in series.episodes for l in e.lines]
    assert len(set(all_line_ids)) == len(all_line_ids)
    assert any(w.code == "duplicate_episode_number" for e in series.episodes for w in e.warnings)


# ------------------------------------------------------------------ outputs
def test_outputs_and_summary_are_consistent(dirs):
    make_zip(dirs["input"] / "show.zip", episode_files(3, folder="My Show"))
    [(series, summary, out_dir)] = run(dirs["input"], dirs["extracted"], dirs["output"])

    assert out_dir.name == "my-show"
    assert sorted(p.name for p in (out_dir / "episodes").glob("*.json")) == ["E001.json", "E002.json", "E003.json"]
    on_disk = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(summary.model_dump_json())

    assert summary.scenes == 6
    assert summary.lines == 18                      # 6 lines per synthetic episode
    assert summary.lines_by_kind == {"action": 3, "camera": 3, "dialogue": 9, "sound_fx": 3}
    assert summary.lines_with_valid_timestamps == 18
    assert summary.warnings == 0 and summary.failed_files == []
    assert summary.all_files_fully_accounted
    assert summary.total_duration_ms == 3 * 30000


def test_summary_counts_missing_timestamps(dirs):
    bad = make_script(2).replace("Start: (00:05.00) | End: (00:06.50)\n", "")
    files = {"Show/episode-001-Episode 1001.txt": make_script(1), "Show/episode-002-Episode 1002.txt": bad}
    make_zip(dirs["input"] / "show.zip", files)
    [(_, summary, _)] = run(dirs["input"], dirs["extracted"], dirs["output"])
    assert summary.lines == 12
    assert summary.lines_with_valid_timestamps == 11
    assert summary.lines_without_valid_timestamps == 1
    assert summary.timestamp_status_counts == {"missing": 1, "ok": 11}
    assert summary.warnings_by_code == {"missing_timestamp": 1}


def test_json_files_load_back_into_models(dirs):
    from promoai.index import ScreenplayIndex

    make_zip(dirs["input"] / "show.zip", episode_files(2))
    [(series, _, out_dir)] = run(dirs["input"], dirs["extracted"], dirs["output"])
    idx = ScreenplayIndex.from_output_dir(out_dir)
    assert len(idx.lines) == sum(len(e.lines) for e in series.episodes)
