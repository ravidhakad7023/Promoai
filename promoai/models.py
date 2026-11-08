"""Data structures for parsed screenplays.

Hierarchy:   Episode  ->  Scene  ->  Line

ID scheme (all strings, zero padded, unique across the whole dataset):
    episode  E001
    scene    E001-S01
    line     E001-S01-L007      (L-number counts up across the *episode*, so
                                 line numbers never restart inside a scene)

Every Line keeps its original timestamp strings AND the exact source location
(file + 1-based line number), so any ID can be traced back to the raw script.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LineKind(str, Enum):
    DIALOGUE = "dialogue"
    ACTION = "action"
    CAMERA = "camera"
    SOUND_FX = "sound_fx"


class TimestampStatus(str, Enum):
    OK = "ok"                # both start & end parsed, end >= start
    MISSING = "missing"      # no timestamp line followed the label
    INVALID = "invalid"      # timestamp line present but unreadable
    INCONSISTENT = "inconsistent"  # parsed fine but end < start


class Severity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class _Model(BaseModel):
    # Fail loudly on typos in field names rather than silently ignoring them.
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class ParseWarning(_Model):
    code: str                      # machine-friendly, e.g. "missing_timestamp"
    message: str
    severity: Severity = Severity.WARNING
    source_file: str
    source_line: Optional[int] = None
    episode_id: Optional[str] = None


class UnparsedEntry(_Model):
    """A non-blank source line that did not fit any known structure.

    These are kept verbatim so nothing is silently lost."""

    source_file: str
    source_line: int
    text: str
    reason: str
    episode_id: Optional[str] = None


class Line(_Model):
    line_id: str
    scene_id: str
    episode_id: str
    seq: int = Field(description="1-based position in the episode (numeric form of the L-number)")
    index_in_scene: int = Field(description="1-based position within its scene")
    kind: LineKind
    speaker: Optional[str] = Field(None, description="Normalised character name, dialogue only")
    speaker_raw: str = Field(description="Label exactly as written, without the ** markers")
    delivery: list[str] = Field(default_factory=list, description="Parenthetical notes, e.g. ['Cold', 'O.S.']")
    text: str = Field(description="Body text; surrounding quotes removed for dialogue")
    text_raw: str = Field(description="Body text exactly as written")

    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    timestamp_status: TimestampStatus

    source_file: str
    source_line: int = Field(description="1-based line number of the **LABEL:** line")
    source_end_line: int = Field(description="1-based line number of the last body line")


class Scene(_Model):
    scene_id: str
    episode_id: str
    scene_number: int
    heading: Optional[str] = None
    location_type: Optional[str] = Field(None, description="INT / EXT / INT./EXT. when detectable")
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    times_derived: bool = Field(False, description="True when scene start/end were missing and inferred from lines")
    source_file: str
    source_line: int
    lines: list[Line] = Field(default_factory=list)


class Coverage(_Model):
    """Proof that nothing was dropped: every non-blank source line is accounted for."""

    total_lines: int
    non_blank_lines: int
    parsed_lines: int       # classified into a known structure
    unparsed_lines: int     # kept verbatim in Episode.unparsed
    fully_accounted: bool   # parsed + unparsed == non_blank


class Episode(_Model):
    episode_id: str
    episode_number: int
    episode_number_source: str = Field(description="'header', 'filename' or 'sequence'")
    platform_id: Optional[str] = Field(None, description="Numeric id embedded in the filename, if any")
    source_file: str
    source_sha256: str
    series_title: Optional[str] = None
    logline: Optional[str] = None
    continuity_notes: Optional[str] = None
    duration_ms: Optional[int] = Field(None, description="End of the last scene")
    scenes: list[Scene] = Field(default_factory=list)
    warnings: list[ParseWarning] = Field(default_factory=list)
    unparsed: list[UnparsedEntry] = Field(default_factory=list)
    coverage: Coverage

    @property
    def lines(self) -> list[Line]:
        return [ln for sc in self.scenes for ln in sc.lines]


class FailedFile(_Model):
    source_file: str
    reason: str


class DatasetSummary(_Model):
    series_title: Optional[str]
    source_archives: list[str]
    files_discovered: int
    episodes_parsed: int
    scenes: int
    lines: int
    lines_by_kind: dict[str, int]
    lines_with_valid_timestamps: int
    lines_without_valid_timestamps: int
    timestamp_status_counts: dict[str, int]
    warnings: int
    warnings_by_code: dict[str, int]
    unparsed_entries: int
    total_non_blank_source_lines: int
    total_accounted_source_lines: int
    all_files_fully_accounted: bool
    total_duration_ms: int
    failed_files: list[FailedFile]
    ignored_files: list[str] = Field(default_factory=list, description="In the archive but not episode scripts")
