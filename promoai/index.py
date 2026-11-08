"""Load parsed episodes and look things up by ID.

This is the read side of Phase 1's contract: *an ID is a handle to exactly one
source entry*.  Later phases will ask the LLM for line IDs; this index is how
Python turns those IDs back into timestamps and source locations.
(Range resolution - "start ID .. end ID -> clip window" - belongs to a later phase.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import Episode, Line, Scene


class ScreenplayIndex:
    def __init__(self, episodes: Iterable[Episode]) -> None:
        self.episodes: dict[str, Episode] = {}
        self.scenes: dict[str, Scene] = {}
        self.lines: dict[str, Line] = {}
        for ep in episodes:
            self._add(self.episodes, ep.episode_id, ep, "episode")
            for sc in ep.scenes:
                self._add(self.scenes, sc.scene_id, sc, "scene")
                for ln in sc.lines:
                    self._add(self.lines, ln.line_id, ln, "line")

    @staticmethod
    def _add(table: dict, key: str, value, what: str) -> None:
        if key in table:
            raise ValueError(f"duplicate {what} id: {key}")
        table[key] = value

    @classmethod
    def from_output_dir(cls, series_dir: Path) -> "ScreenplayIndex":
        files = sorted((series_dir / "episodes").glob("E*.json"))
        return cls(Episode.model_validate_json(f.read_text(encoding="utf-8")) for f in files)

    def get_line(self, line_id: str) -> Line:
        try:
            return self.lines[line_id]
        except KeyError:
            raise KeyError(f"unknown line id: {line_id!r}") from None

    def get_scene(self, scene_id: str) -> Scene:
        try:
            return self.scenes[scene_id]
        except KeyError:
            raise KeyError(f"unknown scene id: {scene_id!r}") from None

    def get_episode(self, episode_id: str) -> Episode:
        try:
            return self.episodes[episode_id]
        except KeyError:
            raise KeyError(f"unknown episode id: {episode_id!r}") from None
