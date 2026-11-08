"""Builders for timeline tests: synthetic episodes with exact, hand-chosen timestamps."""

from __future__ import annotations

from conftest import cand
from promoai.detect import validate_candidate
from promoai.index import ScreenplayIndex
from promoai.parser import parse_episode_text
from promoai.schemas import PromoCandidate, StoredCandidate
from promoai.timecode import format_timecode

NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX"}


def ms(seconds: float) -> int:
    return round(seconds * 1000)


def episode_text(n: int, blocks: list[tuple]) -> str:
    """blocks: (label, start_s, end_s[, text]) in FILE order (not time order).
    A label that is not CAMERA/ACTION/SOUND FX is a speaker.  start_s=None omits the timestamp."""
    starts = [b[1] for b in blocks if b[1] is not None]
    ends = [b[2] for b in blocks if b[2] is not None]
    out = [f"EPISODE NUMBER: Episode {n}", "EPISODE LOGLINE: synthetic.", "CONTINUITY NOTES: none.", "",
           "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. TEST ROOM – DAY",
           f"Scene start: {format_timecode(ms(min(starts)))}", f"Scene end: {format_timecode(ms(max(ends)))}", "",
           "Actions and dialogue:", ""]
    for i, b in enumerate(blocks, 1):
        label, s, e = b[0], b[1], b[2]
        text = b[3] if len(b) > 3 else f"block {i}"
        out.append(f"**{label}:**")
        if s is not None:
            out.append(f"Start: ({format_timecode(ms(s))}) | End: ({format_timecode(ms(e))})")
        out.append(text if label in NON_DIALOGUE else f'"{text}"')
        out.append("")
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def build_index(episodes: dict[int, list[tuple]]) -> ScreenplayIndex:
    return ScreenplayIndex([
        parse_episode_text(episode_text(n, blocks), f"episode-{n:03d}-Episode {1000 + n}.txt")
        for n, blocks in sorted(episodes.items())])


def lid(ep: int, n: int) -> str:
    return f"E{ep:03d}-S01-L{n:03d}"


def stored(index: ScreenplayIndex, segs, hook: str, cliff: str, cid: str = "C1", **extra) -> StoredCandidate:
    """segs: [((ep, first_line_no), (ep, last_line_no)), ...]  -> a Phase-2-valid StoredCandidate."""
    pairs = tuple((lid(*a), lid(*b)) for a, b in segs)
    c = cand(cid, segs=pairs, hook=hook, cliff=cliff, **extra)
    window = {a[0] and f"E{a[0]:03d}" for a, _ in segs} | {f"E{b[0]:03d}" for _, b in segs}
    r = validate_candidate(PromoCandidate.model_validate(c), 1, "W01", window, index)
    assert isinstance(r, StoredCandidate), getattr(r, "reasons", r)
    return r


def one_seg_episode(duration_s: float, hook_at: float = 0.5, tail_cliff: bool = True) -> list[tuple]:
    """A single scene whose selected window is exactly [0, duration_s]:
    camera [0, duration], hook line at hook_at, and a closing line ending exactly at duration_s."""
    blocks = [("CAMERA", 0.0, duration_s, "wide"), ("ANN", hook_at, hook_at + 1.0, "hook line")]
    if tail_cliff:
        blocks.append(("BOB", duration_s - 1.5, duration_s, "closing line"))
    return blocks
