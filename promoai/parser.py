"""Tolerant, line-oriented parser for microdrama episode scripts.

Observed format (see README / Phase 1 notes)::

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...

    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. SOME PLACE - DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    Actions and dialogue:

    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    <body text>

    **ANN (Calm, Quiet):**
    Start: (00:13.50) | End: (00:14.30)
    "Hello there."
    --- SCRIPT END ---

Engineering rules
-----------------
1. *Never drop content.*  Every non-blank source line receives a disposition:
   ``parsed`` (became part of a structure) or ``unparsed`` (copied verbatim into
   ``Episode.unparsed`` with a reason).  ``Coverage`` proves the two add up.
2. *Be tolerant, not silent.*  Case differences, missing timestamps, missing
   markers, multi-line bodies, etc. are accepted but produce a ``ParseWarning``.
3. *Keep provenance.*  Every Line records source file + line numbers.
4. *Deterministic.*  Same input bytes -> same IDs and output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Coverage,
    Episode,
    Line,
    LineKind,
    ParseWarning,
    Scene,
    Severity,
    TimestampStatus,
    UnparsedEntry,
)
from .timecode import TIMESTAMP_PATTERN, TimecodeError, parse_timecode

# --------------------------------------------------------------------------- #
# Regular expressions (all case-insensitive, whitespace tolerant)
# --------------------------------------------------------------------------- #
_TS = TIMESTAMP_PATTERN

HEADER_RE = re.compile(
    r"^(?P<key>episode\s+number|episode\s+logline|continuity\s+notes)\s*:\s*(?P<val>.*)$", re.I
)
SCRIPT_START_RE = re.compile(r"^-{2,}\s*script\s+start\s*-{2,}$", re.I)
SCRIPT_END_RE = re.compile(r"^-{2,}\s*script\s+end\s*-{2,}$", re.I)
SCENE_RE = re.compile(r"^scene\s+(?P<n>\d+)\s*[:.]?\s*$", re.I)
SCENE_HEADING_RE = re.compile(r"^scene\s+heading\s*:\s*(?P<v>.*)$", re.I)
SCENE_START_RE = re.compile(r"^scene\s+start\s*:\s*(?P<v>.*)$", re.I)
SCENE_END_RE = re.compile(r"^scene\s+end\s*:\s*(?P<v>.*)$", re.I)
ACTIONS_MARKER_RE = re.compile(r"^actions?\s+(and|&)\s+dialogue\s*:?\s*$", re.I)
# **LABEL:**  or  **LABEL**:   (optionally followed by inline text)
LABEL_RE = re.compile(r"^\*\*\s*(?P<label>[^*]+?)\s*\*\*\s*:?\s*(?P<rest>.*)$")

TS_LINE_RE = re.compile(
    rf"^start\s*[:=]?\s*(?P<s>{_TS})\s*[|,;/\-–—]*\s*end\s*[:=]?\s*(?P<e>{_TS})\s*$", re.I
)
TS_PREFIX_RE = re.compile(r"^(start|end)\s*[:=]", re.I)
TS_HALVES_RE = re.compile(r"^start\s*[:=]\s*(?P<s>.*?)\s*[|,;]\s*end\s*[:=]\s*(?P<e>.*)$", re.I)

HEADING_RE = re.compile(
    r"^(?P<type>INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|I\s*/\s*E|INT\.?|EXT\.?)(?![A-Za-z])\s*[.\-–—]?\s*(?P<rest>.*)$",
    re.I,
)
HEADING_SPLIT_RE = re.compile(r"\s+[–—-]+\s+")
PAREN_RE = re.compile(r"\(([^()]*)\)")
# Straight/curly double quotes only: a bare apostrophe at both ends could be real text.
QUOTE_CHARS = "\"“”„«»"

# Labels that are *not* characters.  Anything else is treated as a speaker.
NON_DIALOGUE_LABELS: dict[str, LineKind] = {
    "CAMERA": LineKind.CAMERA,
    "CAMERA ANGLE": LineKind.CAMERA,
    "SHOT": LineKind.CAMERA,
    "ACTION": LineKind.ACTION,
    "ACTIONS": LineKind.ACTION,
    "ACTION LINE": LineKind.ACTION,
    "DIRECTION": LineKind.ACTION,
    "STAGE DIRECTION": LineKind.ACTION,
    "SOUND FX": LineKind.SOUND_FX,
    "SOUND": LineKind.SOUND_FX,
    "SFX": LineKind.SOUND_FX,
    "SOUND EFFECT": LineKind.SOUND_FX,
    "SOUND EFFECTS": LineKind.SOUND_FX,
    "MUSIC": LineKind.SOUND_FX,
}

EPISODE_FILENAME_RE = re.compile(r"episode[-_\s]*(\d+)", re.I)


def episode_info_from_filename(filename: str) -> tuple[Optional[int], Optional[str]]:
    """``episode-001-Episode 1671703.txt`` -> (1, '1671703').  Either may be None."""
    found = EPISODE_FILENAME_RE.findall(filename)
    number = int(found[0]) if found else None
    platform_id = found[1] if len(found) > 1 else None
    return number, platform_id


def split_source_lines(text: str) -> list[str]:
    """Split on \\r\\n, \\r or \\n only.

    ``str.splitlines()`` also splits on \\x0b, \\x1c, \\u2028 ... which would make
    our line numbers disagree with what an editor shows."""
    return re.split(r"\r\n|\r|\n", text)


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# Internal builders
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    label_raw: str
    label_line: int
    inline_text: str = ""
    ts_done: bool = False
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    status: TimestampStatus = TimestampStatus.MISSING
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    body: list[str] = field(default_factory=list)
    last_line: int = 0
    closed: bool = False  # a blank line after the body ended it


@dataclass
class _SceneDraft:
    scene: Scene
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_raw_line: Optional[int] = None


class _EpisodeParser:
    def __init__(
        self,
        text: str,
        source_file: str,
        *,
        fallback_number: Optional[int],
        number_override: Optional[int],
        platform_id: Optional[str],
        series_title: Optional[str],
        source_sha256: str,
    ) -> None:
        self.source_file = source_file
        self.src = split_source_lines(text)
        self.fallback_number = fallback_number
        self.number_override = number_override
        self.platform_id = platform_id
        self.series_title = series_title
        self.sha = source_sha256

        self.disp: list[Optional[str]] = [None] * len(self.src)
        self.warnings: list[ParseWarning] = []
        self.unparsed: list[UnparsedEntry] = []

        self.header: dict[str, str] = {}
        self.header_field: Optional[str] = None
        self.saw_script_start = False
        self.saw_script_end = False
        self._warned_after_end = False

        self.drafts: list[_SceneDraft] = []
        self.cur: Optional[_SceneDraft] = None
        self.block: Optional[_Block] = None
        self.seq = 0

        self.episode_number: int = 0
        self.number_source: str = ""
        self.episode_id: str = ""

    # ------------------------------------------------------------------ utils
    def _warn(self, code: str, message: str, line: Optional[int] = None, severity: Severity = Severity.WARNING) -> None:
        self.warnings.append(
            ParseWarning(
                code=code,
                message=message,
                severity=severity,
                source_file=self.source_file,
                source_line=line,
                episode_id=self.episode_id or None,
            )
        )

    def _mark_parsed(self, idx: int) -> None:
        self.disp[idx] = "parsed"

    def _mark_unparsed(self, idx: int, reason: str) -> None:
        self.disp[idx] = "unparsed"
        self.unparsed.append(
            UnparsedEntry(
                source_file=self.source_file,
                source_line=idx + 1,
                text=self.src[idx],
                reason=reason,
                episode_id=self.episode_id or None,
            )
        )
        self._warn("unparsed_line", f"{reason}: {self.src[idx].strip()[:80]!r}", idx + 1)

    # ------------------------------------------------------------ episode id
    def _resolve_episode_number(self) -> None:
        header_n: Optional[int] = None
        raw = self.header.get("episode number")
        if raw:
            m = re.search(r"(\d+)", raw)
            if m:
                header_n = int(m.group(1))
        if self.number_override is not None:
            self.episode_number, self.number_source = self.number_override, "sequence"
        elif header_n is not None:
            self.episode_number, self.number_source = header_n, "header"
        elif self.fallback_number is not None:
            self.episode_number, self.number_source = self.fallback_number, "filename"
        else:
            self.episode_number, self.number_source = 0, "sequence"
        self.episode_id = f"E{self.episode_number:03d}"
        # Retro-fit the id onto warnings/unparsed entries raised before it was known.
        for w in self.warnings:
            w.episode_id = self.episode_id
        for u in self.unparsed:
            u.episode_id = self.episode_id
        if header_n is not None and self.fallback_number is not None and header_n != self.fallback_number:
            self._warn(
                "episode_number_mismatch",
                f"header says episode {header_n} but filename says {self.fallback_number}; "
                f"using {self.episode_number} ({self.number_source})",
            )
        if header_n is None and raw is not None:
            self._warn("unreadable_episode_number", f"cannot read a number from {raw!r}")

    # ---------------------------------------------------------------- scenes
    def _open_scene(self, number: int, line_idx: int, *, implicit: bool = False) -> _SceneDraft:
        self._close_block()
        used = {d.scene.scene_number for d in self.drafts}
        if number in used:
            new_number = max(used) + 1
            self._warn(
                "scene_number_conflict",
                f"scene number {number} already used in this episode; renumbered to {new_number}",
                line_idx + 1,
            )
            number = new_number
        scene = Scene(
            scene_id=f"__tmp__{number}",  # replaced once the episode id is known
            episode_id="",
            scene_number=number,
            source_file=self.source_file,
            source_line=line_idx + 1,
        )
        draft = _SceneDraft(scene=scene)
        self.drafts.append(draft)
        self.cur = draft
        if implicit:
            self._warn("implicit_scene", "content found before any 'SCENE NN' line; created an implicit scene", line_idx + 1)
        return draft

    def _ensure_scene(self, line_idx: int) -> _SceneDraft:
        if self.cur is None:
            n = (max((d.scene.scene_number for d in self.drafts), default=0)) + 1
            self._open_scene(n, line_idx, implicit=True)
        assert self.cur is not None
        return self.cur

    # ---------------------------------------------------------------- blocks
    def _close_block(self) -> None:
        """Turn the pending block (if any) into a Line on the current scene."""
        b = self.block
        if b is None:
            return
        self.block = None
        draft = self._ensure_scene(b.label_line - 1)
        scene = draft.scene

        if not b.ts_done:
            b.status = TimestampStatus.MISSING
            self._warn("missing_timestamp", f"no timestamp line for {b.label_raw!r}", b.label_line)

        # parenthetical notes + normalised label
        parens = [p.strip() for p in PAREN_RE.findall(b.label_raw) if p.strip()]
        core = _norm_ws(PAREN_RE.sub("", b.label_raw)).strip(" :")
        kind = NON_DIALOGUE_LABELS.get(core.upper(), LineKind.DIALOGUE)

        body_lines = ([b.inline_text] if b.inline_text else []) + b.body
        text_raw = "\n".join(body_lines)
        text = _norm_ws(text_raw)
        if not text:
            self._warn("empty_body", f"{b.label_raw!r} has no text", b.label_line)

        speaker: Optional[str] = None
        if kind is LineKind.DIALOGUE:
            speaker = core or b.label_raw.strip(" :")
            if not core:
                self._warn("empty_speaker", f"label {b.label_raw!r} has no character name", b.label_line)
            if text and text[0] in QUOTE_CHARS and text[-1] in QUOTE_CHARS and len(text) >= 2:
                text = text[1:-1].strip()
            elif text:
                self._warn("unquoted_dialogue", f"dialogue for {speaker!r} is not wrapped in quotes", b.label_line)

        self.seq += 1
        scene.lines.append(
            Line(
                line_id=f"__tmp__{self.seq}",  # replaced in _assign_ids
                scene_id="",
                episode_id="",
                seq=self.seq,
                index_in_scene=len(scene.lines) + 1,
                kind=kind,
                speaker=speaker,
                speaker_raw=b.label_raw.strip(" :"),
                delivery=parens,
                text=text,
                text_raw=text_raw,
                start_raw=b.start_raw,
                end_raw=b.end_raw,
                start_ms=b.start_ms,
                end_ms=b.end_ms,
                timestamp_status=b.status,
                source_file=self.source_file,
                source_line=b.label_line,
                source_end_line=max(b.last_line, b.label_line),
            )
        )

    def _read_timestamp_line(self, s: str, idx: int) -> None:
        b = self.block
        assert b is not None
        b.ts_done = True
        m = TS_LINE_RE.match(s)
        if m:
            b.start_raw, b.end_raw = m.group("s").strip("() "), m.group("e").strip("() ")
            try:
                b.start_ms, b.end_ms = parse_timecode(b.start_raw), parse_timecode(b.end_raw)
            except TimecodeError as exc:
                b.status = TimestampStatus.INVALID
                self._warn("invalid_timestamp", f"{exc} (line {idx + 1})", idx + 1)
            else:
                if b.end_ms < b.start_ms:
                    b.status = TimestampStatus.INCONSISTENT
                    self._warn("end_before_start", f"end {b.end_raw} is before start {b.start_raw}", idx + 1)
                else:
                    b.status = TimestampStatus.OK
        else:
            b.status = TimestampStatus.INVALID
            h = TS_HALVES_RE.match(s)
            b.start_raw, b.end_raw = (h.group("s"), h.group("e")) if h else (s, None)
            self._warn("invalid_timestamp", f"unreadable timestamp line: {s[:80]!r}", idx + 1)
        b.last_line = idx + 1
        self._mark_parsed(idx)

    # ------------------------------------------------------------------ main
    def run(self) -> Episode:
        for idx, raw in enumerate(self.src):
            s = raw.strip()
            if not s:
                self.disp[idx] = "blank"
                if self.block is not None and self.block.body:
                    self.block.closed = True  # blank after body = block finished
                self.header_field = None
                continue
            self._handle(idx, s)
        self._close_block()
        return self._finish()

    def _handle(self, idx: int, s: str) -> None:
        # ---- document markers ------------------------------------------------
        if SCRIPT_START_RE.match(s):
            self._close_block()
            self.saw_script_start = True
            self.header_field = None
            self._mark_parsed(idx)
            return
        if SCRIPT_END_RE.match(s):
            self._close_block()
            self.saw_script_end = True
            self.header_field = None
            self._mark_parsed(idx)
            return
        if self.saw_script_end and not self._warned_after_end:
            self._warned_after_end = True
            self._warn("content_after_script_end", "text found after the SCRIPT END marker", idx + 1)

        # ---- episode header fields ---------------------------------------------
        m = HEADER_RE.match(s)
        if m and not self.drafts:
            key = re.sub(r"\s+", " ", m.group("key").lower())
            if key in self.header:
                self._warn("duplicate_header_field", f"{key!r} appears more than once", idx + 1)
            self.header[key] = m.group("val").strip()
            self.header_field = key
            self._mark_parsed(idx)
            return

        # ---- scene structure ---------------------------------------------------
        m = SCENE_RE.match(s)
        if m:
            self._open_scene(int(m.group("n")), idx)
            self.header_field = None
            self._mark_parsed(idx)
            return
        m = SCENE_HEADING_RE.match(s)
        if m:
            self._close_block()
            d = self._ensure_scene(idx)
            if d.scene.heading is not None:
                self._warn("duplicate_scene_field", "second scene heading in the same scene", idx + 1)
            d.scene.heading = m.group("v").strip() or None
            self._mark_parsed(idx)
            return
        m = SCENE_START_RE.match(s)
        if m:
            self._close_block()
            d = self._ensure_scene(idx)
            d.start_raw = m.group("v").strip().strip("() ")
            self._mark_parsed(idx)
            return
        m = SCENE_END_RE.match(s)
        if m:
            self._close_block()
            d = self._ensure_scene(idx)
            d.end_raw = m.group("v").strip().strip("() ")
            self._mark_parsed(idx)
            return
        if ACTIONS_MARKER_RE.match(s):
            self._close_block()
            self._mark_parsed(idx)
            return

        # ---- block label -------------------------------------------------------
        m = LABEL_RE.match(s)
        if m:
            self._close_block()
            self._ensure_scene(idx)
            self.block = _Block(label_raw=m.group("label").strip(), label_line=idx + 1, last_line=idx + 1)
            rest = m.group("rest").strip()
            if rest:
                self.block.inline_text = rest
                self._warn("inline_body", "text on the same line as the label", idx + 1)
            self._mark_parsed(idx)
            return

        # ---- timestamp line ----------------------------------------------------
        if TS_PREFIX_RE.match(s):
            b = self.block
            if b is not None and not b.ts_done and not b.body:
                self._read_timestamp_line(s, idx)
            else:
                self._mark_unparsed(idx, "timestamp line without a preceding label")
            return

        # ---- body text / everything else --------------------------------------
        b = self.block
        if b is not None and not b.closed:
            if not b.ts_done:
                # Label followed by text but no timestamp: keep the text, flag the line.
                b.ts_done = True
                b.status = TimestampStatus.MISSING
                self._warn("missing_timestamp", f"no timestamp line for {b.label_raw!r}", b.label_line)
            b.body.append(s)
            b.last_line = idx + 1
            self._mark_parsed(idx)
            return
        if self.header_field and not self.drafts:
            key = self.header_field
            self.header[key] = (self.header[key] + " " + s).strip()
            self._warn("wrapped_header_field", f"{key!r} continues on a following line", idx + 1)
            self._mark_parsed(idx)
            return
        self._mark_unparsed(idx, "text outside any recognised block")

    # ---------------------------------------------------------------- finish
    def _finish(self) -> Episode:
        self._resolve_episode_number()
        eid = self.episode_id

        for key, label in (
            ("episode number", "EPISODE NUMBER"),
            ("episode logline", "EPISODE LOGLINE"),
            ("continuity notes", "CONTINUITY NOTES"),
        ):
            if not self.header.get(key):
                self._warn("missing_header_field", f"{label} is missing or empty")
        if not self.saw_script_start:
            self._warn("missing_script_start", "no '--- SCRIPT START ---' marker")
        if not self.saw_script_end:
            self._warn("missing_script_end", "no '--- SCRIPT END ---' marker")
        if not self.drafts:
            self._warn("no_scenes", "no scenes found in file", severity=Severity.ERROR)

        prev_end: Optional[int] = None
        scenes: list[Scene] = []
        for draft in self.drafts:
            sc = draft.scene
            sc.scene_id = f"{eid}-S{sc.scene_number:02d}"
            sc.episode_id = eid
            for ln in sc.lines:
                ln.line_id = f"{sc.scene_id}-L{ln.seq:03d}"
                ln.scene_id = sc.scene_id
                ln.episode_id = eid

            self._finish_scene_times(draft)
            self._parse_heading(sc)
            if not sc.lines:
                self._warn("empty_scene", f"{sc.scene_id} contains no lines", sc.source_line)
            for ln in sc.lines:
                if (
                    ln.timestamp_status is TimestampStatus.OK
                    and sc.start_ms is not None
                    and sc.end_ms is not None
                    and (ln.start_ms < sc.start_ms or ln.end_ms > sc.end_ms)
                ):
                    self._warn(
                        "line_outside_scene",
                        f"{ln.line_id} ({ln.start_raw}-{ln.end_raw}) falls outside "
                        f"{sc.scene_id} ({sc.start_raw}-{sc.end_raw})",
                        ln.source_line,
                    )
            if prev_end is not None and sc.start_ms is not None and abs(sc.start_ms - prev_end) > 100:
                self._warn(
                    "scene_time_discontinuity",
                    f"{sc.scene_id} starts at {sc.start_raw} but the previous scene ended at {prev_end} ms",
                    sc.source_line,
                )
            if sc.end_ms is not None:
                prev_end = sc.end_ms
            scenes.append(sc)

        ends = [s.end_ms for s in scenes if s.end_ms is not None]
        non_blank = sum(1 for d in self.disp if d != "blank")
        parsed = sum(1 for d in self.disp if d == "parsed")
        unparsed = sum(1 for d in self.disp if d == "unparsed")
        coverage = Coverage(
            total_lines=len(self.src),
            non_blank_lines=non_blank,
            parsed_lines=parsed,
            unparsed_lines=unparsed,
            fully_accounted=(parsed + unparsed == non_blank and all(d is not None for d in self.disp)),
        )
        return Episode(
            episode_id=eid,
            episode_number=self.episode_number,
            episode_number_source=self.number_source,
            platform_id=self.platform_id,
            source_file=self.source_file,
            source_sha256=self.sha,
            series_title=self.series_title,
            logline=self.header.get("episode logline") or None,
            continuity_notes=self.header.get("continuity notes") or None,
            duration_ms=max(ends) if ends else None,
            scenes=scenes,
            warnings=self.warnings,
            unparsed=self.unparsed,
            coverage=coverage,
        )

    def _finish_scene_times(self, draft: _SceneDraft) -> None:
        sc = draft.scene
        sc.start_raw, sc.end_raw = draft.start_raw, draft.end_raw
        for attr, raw in (("start_ms", draft.start_raw), ("end_ms", draft.end_raw)):
            if raw:
                try:
                    setattr(sc, attr, parse_timecode(raw))
                except TimecodeError:
                    self._warn("invalid_scene_timestamp", f"{sc.scene_id}: cannot read {raw!r}", sc.source_line)
        if sc.start_ms is None or sc.end_ms is None:
            ok = [l for l in sc.lines if l.timestamp_status is TimestampStatus.OK]
            if ok:
                if sc.start_ms is None:
                    sc.start_ms = min(l.start_ms for l in ok)  # type: ignore[type-var]
                if sc.end_ms is None:
                    sc.end_ms = max(l.end_ms for l in ok)  # type: ignore[type-var]
                sc.times_derived = True
                self._warn(
                    "scene_times_derived",
                    f"{sc.scene_id} had missing/invalid scene start/end; derived from its lines",
                    sc.source_line,
                )
            else:
                self._warn("scene_times_unknown", f"{sc.scene_id} has no usable scene or line timestamps", sc.source_line)
        if sc.start_ms is not None and sc.end_ms is not None and sc.end_ms < sc.start_ms:
            self._warn("scene_end_before_start", f"{sc.scene_id}: end before start", sc.source_line)

    def _parse_heading(self, sc: Scene) -> None:
        if not sc.heading:
            self._warn("missing_scene_heading", f"{sc.scene_id} has no heading", sc.source_line)
            return
        m = HEADING_RE.match(sc.heading.strip())
        if not m:
            self._warn("unrecognised_heading", f"cannot split heading {sc.heading!r} into INT/EXT, place, time", sc.source_line)
            return
        sc.location_type = re.sub(r"\s+", "", m.group("type")).upper().replace(".", "")
        parts = HEADING_SPLIT_RE.split(m.group("rest").strip())
        if len(parts) > 1:
            sc.location = " - ".join(p.strip() for p in parts[:-1]).strip(" .-") or None
            sc.time_of_day = parts[-1].strip().upper() or None
        else:
            sc.location = parts[0].strip(" .-") or None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_episode_text(
    text: str,
    source_file: str,
    *,
    fallback_number: Optional[int] = None,
    number_override: Optional[int] = None,
    platform_id: Optional[str] = None,
    series_title: Optional[str] = None,
    source_sha256: str = "",
) -> Episode:
    """Parse one episode script into an :class:`Episode`.

    ``fallback_number`` (usually from the filename) is only used when the header
    has no readable episode number.  ``number_override`` forces the number (used
    by the batch runner to resolve duplicate episode numbers)."""
    return _EpisodeParser(
        text,
        source_file,
        fallback_number=fallback_number,
        number_override=number_override,
        platform_id=platform_id,
        series_title=series_title,
        source_sha256=source_sha256,
    ).run()
