"""Timecode helpers.

Design decision: timestamps are stored as **integer milliseconds** (plus the
original string).  Floats such as 4.8 or 66.7 cannot be represented exactly in
binary, so comparing / summing them later (clip duration checks like
"is this 20-60 s?") would be subtly wrong.  Integers are exact.

Accepted formats (the real data only uses ``MM:SS.ff``, the rest is tolerance):
    00:04.80      MM:SS.ff
    1:02.5        single-digit minutes, one fractional digit
    00:04         no fraction
    00:00:04.80   HH:MM:SS.ff
    00:04,80      comma as decimal separator
"""

from __future__ import annotations

import re

# Optional hours, minutes, seconds, optional fraction.
_TS_BODY = r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,3}):(?P<s>\d{1,2})(?:[.,](?P<f>\d{1,3}))?"
TIMESTAMP_RE = re.compile(rf"^\s*\(?\s*{_TS_BODY}\s*\)?\s*$")

# Same pattern without anchors / named groups, for embedding in larger regexes.
TIMESTAMP_PATTERN = r"\(?\s*(?:\d{1,2}:)?\d{1,3}:\d{1,2}(?:[.,]\d{1,3})?\s*\)?"


class TimecodeError(ValueError):
    """Raised when a string cannot be read as a timestamp."""


def parse_timecode(text: str) -> int:
    """Convert a timestamp string to integer milliseconds.

    >>> parse_timecode("00:04.80")
    4800
    >>> parse_timecode("01:02.5")
    62500
    """
    m = TIMESTAMP_RE.match(text)
    if not m:
        raise TimecodeError(f"not a timestamp: {text!r}")
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    if seconds >= 60:
        raise TimecodeError(f"seconds out of range in {text!r}")
    frac = m.group("f") or ""
    # "8" -> 800 ms, "80" -> 800 ms, "800" -> 800 ms
    millis = int(frac.ljust(3, "0")) if frac else 0
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_timecode(ms: int) -> str:
    """Milliseconds -> ``MM:SS.ff`` (hundredths), used for human-readable output."""
    if ms < 0:
        raise ValueError("negative milliseconds")
    total_cs = round(ms / 10)
    minutes, cs = divmod(total_cs, 6000)
    seconds, cs = divmod(cs, 100)
    return f"{minutes:02d}:{seconds:02d}.{cs:02d}"
