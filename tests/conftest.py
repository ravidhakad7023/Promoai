"""Shared test helpers.

All synthetic scripts below are invented text - safe to commit.  Tests that touch the
confidential screenplay pack live in test_real_data.py and skip when it is absent."""

from __future__ import annotations

import textwrap

import pytest

from promoai.parser import parse_episode_text

HEADER = """\
EPISODE NUMBER: Episode {n}
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Test notes.

--- SCRIPT START ---
"""

SCENE_1 = """\
SCENE 01
Scene heading: INT. TEST HALL – DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
Wide shot of the hall.

**ACTION:**
Start: (00:01.00) | End: (00:04.80)
Ann enters.

**ANN (Calm, Quiet):**
Start: (00:05.00) | End: (00:06.50)
"Hello there."

**SOUND FX:**
Start: (00:06.50) | End: (00:08.00)
A door slams.

**BOB (O.S.) (Shouting):**
Start: (00:08.00) | End: (00:09.25)
"Who goes there?"
"""

SCENE_2 = """\

SCENE 02
Scene heading: EXT. GARDEN — NIGHT
Scene start: 00:20.00
Scene end: 00:30.00

Actions and dialogue:

**ANN:**
Start: (00:20.50) | End: (00:22.00)
"It is cold."
"""

FOOTER = "\n--- SCRIPT END ---\n"


def make_script(n: int = 1, *scenes: str, footer: str = FOOTER) -> str:
    if not scenes:
        scenes = (SCENE_1, SCENE_2)
    return HEADER.format(n=n) + "\n" + "\n".join(s.strip("\n") + "\n" for s in scenes) + footer


@pytest.fixture
def parse():
    """parse(text, name='episode-001-Episode 111.txt') -> Episode"""

    def _parse(text: str, name: str = "episode-001-Episode 111.txt", **kw):
        return parse_episode_text(text, name, **kw)

    return _parse


@pytest.fixture
def good_script() -> str:
    return make_script()


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def codes(ep) -> list[str]:
    return [w.code for w in ep.warnings]


# ---------------------------------------------------------------- Phase 2 helpers
import json as _json

from promoai.index import ScreenplayIndex


def make_index(n_episodes: int = 4) -> ScreenplayIndex:
    """Synthetic series: each episode has 6 lines (L001-L005 in S01, L006 in S02)."""
    eps = [parse_episode_text(make_script(i), f"episode-{i:03d}-Episode {1000 + i}.txt") for i in range(1, n_episodes + 1)]
    return ScreenplayIndex(eps)


GATES = {
    "hook_in_first_3s": True, "hook_note": "n",
    "unresolved_cliffhanger": True, "cliffhanger_note": "n",
    "standalone_clear": True, "standalone_note": "n",
    "duration_plausible_20_60s": True, "duration_note": "n",
}


def cand(cid="C1", segs=(("E001-S01-L001", "E001-S01-L004"),), hook=None, cliff=None, **extra) -> dict:
    return {
        "candidate_id": cid,
        "segments": [{"start_line_id": a, "end_line_id": b} for a, b in segs],
        "hook_line_id": hook or segs[0][0],
        "hook_description": "hook", "cliffhanger_line_id": cliff or segs[-1][1],
        "cliffhanger_description": "cliff", "open_question": "Will Ann find out who sent the letter?",
        "why_it_works": "because", "tags": ["suspense"],
        "gates": dict(GATES), **extra,
    }


def llm_reply(cands=(), characters=(("Ann", ()),), conflicts=("c1",), revelations=("r1",), questions=("q1",)) -> str:
    return _json.dumps({
        "story_context": {
            "characters": [{"name": n, "aliases": list(a), "role": "lead", "relationships": []} for n, a in characters],
            "conflicts": list(conflicts), "revelations": list(revelations), "open_questions": list(questions),
        },
        "candidates": list(cands),
    })
