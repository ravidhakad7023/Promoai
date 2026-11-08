"""Locate, safely extract and enumerate screenplay files.

The number of episodes is *never* hardcoded: whatever ``*.txt`` scripts the
archive contains are discovered dynamically.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .parser import episode_info_from_filename

# Guard against zip bombs: refuse archives that would expand beyond this.
MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
SCRIPT_SUFFIXES = {".txt"}


class ArchiveError(RuntimeError):
    pass


@dataclass
class ExtractResult:
    archive: Path
    extract_dir: Path
    series_title: str
    script_files: list[Path] = field(default_factory=list)   # absolute paths, naturally sorted
    ignored: list[str] = field(default_factory=list)          # archive members that are not scripts


def find_archives(input_dir: Path) -> list[Path]:
    """All ``*.zip`` files directly inside ``input_dir`` (sorted, case-insensitive)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    return sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"), key=lambda p: p.name.lower())


def _is_junk(member: str) -> bool:
    parts = Path(member).parts
    return any(p == "__MACOSX" for p in parts) or Path(member).name.startswith(".") or Path(member).name.lower() == "thumbs.db"


def extract_archive(archive: Path, extract_root: Path) -> ExtractResult:
    """Extract ``archive`` into ``extract_root/<zip name>/`` and list its scripts.

    * Rejects members that would escape the target directory ("zip slip").
    * Rejects archives whose declared size is absurd.
    * Wipes the target folder first so stale files from an older archive can
      never be mistaken for current episodes.
    """
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"{archive.name} is not a valid ZIP: {exc}") from exc

    with zf:
        infos = zf.infolist()
        if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
            raise ArchiveError(f"{archive.name} expands to more than {MAX_UNCOMPRESSED_BYTES} bytes; refusing")

        dest = (extract_root / archive.stem).resolve()
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        tops = {Path(i.filename).parts[0] for i in infos if Path(i.filename).parts and not _is_junk(i.filename)}
        # If everything sits in one folder inside the zip, that folder is the series title.
        series_title = next(iter(tops)) if len(tops) == 1 and any("/" in i.filename for i in infos) else archive.stem
        series_title = series_title.removesuffix("-script").strip()

        result = ExtractResult(archive=archive, extract_dir=dest, series_title=series_title)
        for info in infos:
            if info.is_dir() or _is_junk(info.filename):
                continue
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ArchiveError(f"unsafe path in archive: {info.filename!r}")
            if Path(info.filename).suffix.lower() not in SCRIPT_SUFFIXES:
                result.ignored.append(info.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            result.script_files.append(target)

    result.script_files.sort(key=_natural_key)
    return result


def discover_folder(folder: Path) -> ExtractResult:
    """Treat an ordinary folder of ``*.txt`` scripts (searched recursively) like an extracted archive.
    Nothing is copied or modified; junk files and non-script files are skipped/listed as ignored."""
    folder = folder.resolve()
    result = ExtractResult(archive=folder, extract_dir=folder, series_title=folder.name.removesuffix("-script").strip() or "series")
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or _is_junk(rel):
            continue
        if p.suffix.lower() in SCRIPT_SUFFIXES:
            result.script_files.append(p)
        else:
            result.ignored.append(rel)
    result.script_files.sort(key=_natural_key)
    return result


def _natural_key(path: Path) -> tuple[float, str]:
    number, _ = episode_info_from_filename(path.name)
    return (float(number) if number is not None else float("inf"), path.name.lower())


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "series"
