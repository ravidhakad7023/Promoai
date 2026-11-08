"""Pydantic schemas for everything the LLM returns (and what Phase 2 stores).

The LLM returns **IDs only** - never timestamps.  Any timestamp-looking field that
sneaks into a reply is dropped on validation and recorded in ``dropped_fields`` so
the guarantee is checkable.

Segment semantics (important, see Phase 1 verification):
    A segment is an inclusive range of line IDs ``start_line_id .. end_line_id`` inside
    ONE episode, in *file order*.  File order is NOT a timeline - CAMERA / ACTION /
    SOUND FX / dialogue blocks overlap in time.  Phase 3 turns a segment into a time
    window (earliest start .. latest end of the lines in the range).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TIME_KEY_RE = re.compile(r"(time|timestamp|seconds|(^|_)(sec|ms)($|_)|duration)", re.I)
LINE_ID_RE = re.compile(r"^E\d{3,}-S\d{2,}-L\d{3,}$")


class _LLMModel(BaseModel):
    # Extra keys are tolerated (a harmless extra field must not cost a retry) but any
    # time-like key is recorded so we can prove no timestamps were consumed.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    dropped_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _flag_time_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields)
            dropped = sorted(k for k in data if k not in known and _TIME_KEY_RE.search(str(k)))
            if dropped:
                data = {**data, "dropped_fields": dropped}
        return data


# ------------------------------------------------------------------ story context
class CharacterInfo(_LLMModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = ""
    relationships: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("character name is empty")
        return v


class StoryContext(_LLMModel):
    """Compact running summary carried from window to window."""

    characters: list[CharacterInfo] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    revelations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Voices heard but not attributed to anyone (e.g. "Off-Screen Voice").  Kept apart on purpose:
    # merging them into a named character would invent a relationship the script never states.
    unattributed_voices: list[str] = Field(default_factory=list)
    through_episode_id: Optional[str] = None   # set by Python, not trusted from the LLM


# ------------------------------------------------------------------ promo candidates
def normalize_line_id(v: str) -> str:
    """Tidy case/whitespace only; the ID must otherwise already be well-formed."""
    v = v.strip().upper()
    if not LINE_ID_RE.match(v):
        raise ValueError(f"not a line id like E001-S01-L007: {v!r}")
    return v


class SegmentRef(_LLMModel):
    start_line_id: str
    end_line_id: str

    @field_validator("start_line_id", "end_line_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        return normalize_line_id(v)


class GateAssessment(_LLMModel):
    """The LLM's *self-assessment*.  Preliminary and unverified - Phase 3 must check
    each gate deterministically; these booleans are never evidence."""

    hook_in_first_3s: bool
    hook_note: str = ""
    unresolved_cliffhanger: bool
    cliffhanger_note: str = ""
    standalone_clear: bool
    standalone_note: str = ""
    duration_plausible_20_60s: bool
    duration_note: str = ""


class PromoCandidate(_LLMModel):
    candidate_id: str
    segments: list[SegmentRef] = Field(min_length=1, max_length=4)
    hook_line_id: str
    hook_description: str
    cliffhanger_line_id: str
    cliffhanger_description: str
    open_question: str = Field(description="The specific unresolved question/threat the ending creates")
    why_it_works: str
    tags: list[str] = Field(default_factory=list)
    gates: GateAssessment

    @field_validator("hook_line_id", "cliffhanger_line_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        return normalize_line_id(v)


class DetectionResponse(_LLMModel):
    """Exactly what one LLM call must return."""

    story_context: StoryContext
    candidates: list[PromoCandidate] = Field(default_factory=list)


# ------------------------------------------------------------------ stored (Python-side) records
class ResolvedSegment(BaseModel):
    """A validated segment with the IDs Python derived from the line IDs."""

    start_line_id: str
    end_line_id: str
    episode_id: str
    start_scene_id: str
    end_scene_id: str
    n_lines: int = Field(description="Number of lines in the ID range (file order). NOT a duration.")


class StoredCandidate(BaseModel):
    candidate_id: str                       # globally unique: <window_id>-<llm id>
    llm_candidate_id: str
    window_id: str
    status: str = "llm_proposed"            # never "approved": Phase 3 decides
    segments: list[ResolvedSegment]
    cross_episode: bool
    hook_line_id: str
    hook_description: str
    cliffhanger_line_id: str
    cliffhanger_description: str
    open_question: str                      # the model's claim about what the ending leaves unresolved
    why_it_works: str
    tags: list[str]
    # The model's own assessment of the four gates.  These are CLAIMS, never validation:
    # live testing showed them to be wrong (a 76 s clip was claimed to fit 20-60 s).
    # Phase 3 must verify duration / hook timing from the real timeline.
    llm_gate_claims: GateAssessment
    gates_verified: bool = False            # flipped only by deterministic Python checks (Phase 3)
    dropped_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    window_id: str
    llm_candidate_id: str
    reasons: list[str]
