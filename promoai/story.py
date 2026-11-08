"""Running story summary: deterministic merge + trimming.

The LLM proposes an updated :class:`StoryContext` after each window.  Python then
applies a few safety rules so a sloppy reply can never make the memory shrink or
grow without bound:

* Known characters are never lost (an update that omits one keeps the old entry).
* Revelations are cumulative facts, so old ones are kept (most recent N survive).
* Conflicts / open questions follow the LLM (it drops resolved ones) - but an empty
  list while the old one was non-empty is treated as a degenerate reply and ignored.
* Every list and string is capped, keeping prompt size bounded.
* Ambiguous voice labels ("Off-Screen Voice", "Narrator", "V.O." ...) are NEVER used to
  match or merge characters and never become anyone's alias.  They are kept separately in
  ``unattributed_voices`` so the uncertainty is preserved instead of inventing a link.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import CharacterInfo, StoryContext

MAX_CHARACTERS = 25
MAX_ALIASES = 6
MAX_RELATIONSHIPS = 5
MAX_CONFLICTS = 8
MAX_REVELATIONS = 15
MAX_QUESTIONS = 8
MAX_VOICES = 6
MAX_CHARS = 300

# "off-screen voice", "unknown voice", "O.S.", "V.O.", "narrator", "the crowd" ...
_VOICE_RE = re.compile(
    r"^(?:the\s+)?(?:(?:off[\s\-]?screen|o\.?s\.?|v\.?o\.?|unseen|unknown|unidentified|mysterious|distant|"
    r"disembodied|anonymous|hidden|female|male)\s+)*(?:voices?|speaker|narrator|announcer|crowd|chorus)$"
    r"|^(?:o\.?s\.?|v\.?o\.?|off[\s\-]?screen)$",
    re.I,
)


def is_ambiguous_voice(name: str) -> bool:
    """True for labels that name a *kind of voice* rather than a person."""
    return bool(_VOICE_RE.match(" ".join(name.split())))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()


def _clip(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= MAX_CHARS else s[: MAX_CHARS - 1].rstrip() + "…"


def _uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        k = _norm(it)
        if k and k not in seen:
            seen.add(k)
            out.append(_clip(it))
    return out


def _keys(c: CharacterInfo) -> set[str]:
    """Names usable for matching two entries.  Ambiguous voice labels never count."""
    names = [c.name, *c.aliases]
    return {_norm(n) for n in names if _norm(n) and not is_ambiguous_voice(n)}


def _split_voices(aliases: list[str]) -> tuple[list[str], list[str]]:
    return [a for a in aliases if not is_ambiguous_voice(a)], [a for a in aliases if is_ambiguous_voice(a)]


def _merge_character(old: CharacterInfo, new: CharacterInfo, voices: list[str]) -> CharacterInfo:
    aliases = list(old.aliases) + list(new.aliases)
    if _norm(new.name) != _norm(old.name):
        aliases.append(new.name)   # same person under another name: keep both
    keep, moved = _split_voices(aliases)
    voices.extend(moved)
    return CharacterInfo(
        name=old.name,
        aliases=_uniq([a for a in keep if _norm(a) != _norm(old.name)])[:MAX_ALIASES],
        role=_clip(new.role or old.role),
        relationships=_uniq(new.relationships or old.relationships)[:MAX_RELATIONSHIPS],
    )


def merge_context(prev: Optional[StoryContext], new: StoryContext, through_episode_id: Optional[str]) -> tuple[StoryContext, list[str]]:
    """Return ``(merged_context, notes)``.  ``notes`` lists corrective actions taken."""
    notes: list[str] = []
    prev = prev or StoryContext()
    voices: list[str] = list(prev.unattributed_voices) + list(new.unattributed_voices)

    chars: list[CharacterInfo] = []
    for oc in prev.characters:           # re-clean old entries too (older contexts may hold voice aliases)
        if is_ambiguous_voice(oc.name):
            voices.append(oc.name)
            continue
        keep, moved = _split_voices(oc.aliases)
        voices.extend(moved)
        chars.append(oc.model_copy(update={"aliases": keep}, deep=True))

    for nc in new.characters:
        if is_ambiguous_voice(nc.name):  # a "character" that is really just a kind of voice
            voices.append(nc.name)
            continue
        hit = next((i for i, oc in enumerate(chars) if _keys(oc) & _keys(nc)), None)
        if hit is None:
            keep, moved = _split_voices(nc.aliases)
            voices.extend(moved)
            chars.append(CharacterInfo(
                name=nc.name, aliases=_uniq(keep)[:MAX_ALIASES],
                role=_clip(nc.role), relationships=_uniq(nc.relationships)[:MAX_RELATIONSHIPS]))
        else:
            chars[hit] = _merge_character(chars[hit], nc, voices)

    voices = _uniq(voices)[:MAX_VOICES]
    prev_voice_keys = {_norm(v) for v in (*prev.unattributed_voices, *new.unattributed_voices)}   # already known
    moved_now = [v for v in voices if _norm(v) not in prev_voice_keys]
    if moved_now:
        notes.append(f"ambiguous voice label(s) kept separate, not merged into any character: {', '.join(moved_now)}")

    kept_old = sum(1 for oc in prev.characters
                   if not is_ambiguous_voice(oc.name) and not any(_keys(oc) & _keys(nc) for nc in new.characters))
    if kept_old:
        notes.append(f"{kept_old} previously known character(s) were absent from the update and were kept")
    if len(chars) > MAX_CHARACTERS:
        notes.append(f"characters trimmed from {len(chars)} to {MAX_CHARACTERS}")
        chars = chars[:MAX_CHARACTERS]

    def follow_llm(old: list[str], upd: list[str], cap: int, label: str) -> list[str]:
        if not upd and old:
            notes.append(f"{label}: empty update ignored, previous list kept")
            return _uniq(old)[:cap]
        out = _uniq(upd)
        if len(out) > cap:
            notes.append(f"{label} trimmed from {len(out)} to {cap}")
        return out[:cap]

    revelations = _uniq(list(prev.revelations) + list(new.revelations))
    if len(revelations) > MAX_REVELATIONS:
        notes.append(f"revelations trimmed from {len(revelations)} to the most recent {MAX_REVELATIONS}")
        revelations = revelations[-MAX_REVELATIONS:]

    merged = StoryContext(
        characters=chars,
        conflicts=follow_llm(prev.conflicts, new.conflicts, MAX_CONFLICTS, "conflicts"),
        revelations=revelations,
        open_questions=follow_llm(prev.open_questions, new.open_questions, MAX_QUESTIONS, "open_questions"),
        unattributed_voices=voices,
        through_episode_id=through_episode_id,
    )
    return merged, notes


def sanitize_context(ctx: StoryContext) -> tuple[StoryContext, list[str]]:
    """Re-apply the merge rules to a context loaded from disk (e.g. one produced before the
    ambiguous-voice rule existed).  Idempotent."""
    return merge_context(None, ctx, ctx.through_episode_id)
