"""Ambiguous voices (e.g. "Off-Screen Voice") must never be merged into a named character."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from promoai.prompts import build_prompt, render_context
from conftest import make_index
from promoai.schemas import CharacterInfo, StoryContext
from promoai.story import is_ambiguous_voice, merge_context, sanitize_context


def ctx(chars=(), voices=()):
    return StoryContext(characters=[CharacterInfo(name=n, aliases=list(a), role=r) for n, a, r in chars],
                        unattributed_voices=list(voices))


@pytest.mark.parametrize("label", ["Off-Screen Voice", "OFF-SCREEN VOICE", "off screen voice", "Voice", "The Voice", "Narrator",
                                   "Unknown Voice", "Mysterious Voice", "V.O.", "O.S.", "Announcer", "Crowd", "Distant Voices"])
def test_voice_labels_are_recognised(label):
    assert is_ambiguous_voice(label)


@pytest.mark.parametrize("name", ["Mysterious Tall Stranger", "Voice Coach Ann", "Dowager Quill", "Mabel", "The Priest",
                                  "Lady in Red Gown", "Off Duty Guard", "Unknown Soldier"])
def test_real_characters_are_not_mistaken_for_voices(name):
    assert not is_ambiguous_voice(name)


def test_voice_alias_is_removed_and_kept_as_unattributed():
    new = ctx([("Dowager Quill", ["Dowager", "Off-Screen Voice", "Lady Quill"], "stepmother")])
    m, notes = merge_context(None, new, "E003")
    c = m.characters[0]
    assert "Off-Screen Voice" not in c.aliases and set(c.aliases) == {"Dowager", "Lady Quill"}
    assert m.unattributed_voices == ["Off-Screen Voice"]
    assert any("Off-Screen Voice" in n and "not merged" in n for n in notes)


def test_a_character_entry_that_is_just_a_voice_is_not_a_character():
    m, _ = merge_context(None, ctx([("Off-Screen Voice", [], "unknown speaker"), ("Ann", [], "lead")]), "E001")
    assert [c.name for c in m.characters] == ["Ann"] and m.unattributed_voices == ["Off-Screen Voice"]


def test_two_people_sharing_a_voice_alias_are_not_merged():
    m, _ = merge_context(None, ctx([("Ann", ["Voice"], "a"), ("Bob", ["Voice"], "b")]), "E001")
    assert [c.name for c in m.characters] == ["Ann", "Bob"]          # the shared 'Voice' alias must not link them


def test_voice_never_matches_an_existing_character_in_a_later_update():
    prev = ctx([("Dowager Quill", ["Dowager"], "stepmother")])
    m, _ = merge_context(prev, ctx([("Unknown Voice", ["Dowager"], "heard in the dark")]), "E004")
    assert [c.name for c in m.characters] == ["Dowager Quill"] and m.characters[0].role == "stepmother"
    assert "Unknown Voice" in m.unattributed_voices


def test_uncertainty_is_preserved_no_relationship_is_invented():
    m, _ = merge_context(None, ctx([("Hugo", [], "groom")], voices=["Off-Screen Voice"]), "E001")
    assert m.unattributed_voices == ["Off-Screen Voice"]
    assert all("Off-Screen Voice" not in (c.name, *c.aliases, *c.relationships) for c in m.characters)


def test_sanitize_is_idempotent_and_fixes_old_contexts():
    old = ctx([("Dowager Quill", ["Dowager", "Off-Screen Voice"], "stepmother"), ("Ann", [], "lead")])
    once, _ = sanitize_context(old)
    twice, notes2 = sanitize_context(once)
    assert once.model_dump() == twice.model_dump() and notes2 == []
    assert once.unattributed_voices == ["Off-Screen Voice"]


def test_unattributed_voices_are_shown_to_the_model_with_the_notes():
    c = ctx([("Ann", [], "lead")], voices=["Off-Screen Voice"])
    assert "Off-Screen Voice" in render_context(c) and "unattributed_voices" in render_context(c)
    assert "unattributed" not in render_context(ctx([("Ann", [], "lead")]))
    assert "Off-Screen Voice" in build_prompt(c, [make_index(1).episodes["E001"]])


def test_voice_list_is_bounded():
    m, _ = merge_context(None, ctx(voices=[f"Unknown Voice {i}" for i in range(20)]), "E001")
    assert len(m.unattributed_voices) <= 6


def test_real_v2_context_is_corrected_when_present():
    paths = glob.glob(str(Path(__file__).resolve().parent.parent / "data" / "output" / "*" / "phase2" / "runs" / "v2-sample-e001-e005" / "story_context.json"))
    if not paths:
        pytest.skip("v2 sample not present")
    old = StoryContext.model_validate_json(Path(paths[0]).read_text(encoding="utf-8"))
    assert any(is_ambiguous_voice(a) for c in old.characters for a in c.aliases)     # the known defect is really in the data
    fixed, _ = sanitize_context(old)
    assert not any(is_ambiguous_voice(a) for c in fixed.characters for a in c.aliases)
    assert fixed.unattributed_voices and {c.name for c in fixed.characters} == {c.name for c in old.characters}


def test_matching_keys_never_include_voice_labels():
    from promoai.story import _keys

    c = CharacterInfo(name="Ann", aliases=["Off-Screen Voice", "Annie", "Narrator"])
    assert _keys(c) == {"ann", "annie"}
    assert _keys(CharacterInfo(name="Off-Screen Voice")) == set()
