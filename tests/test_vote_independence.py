from __future__ import annotations

from promoai.llm import CachedProvider, FakeProvider
from promoai.validate import vote_providers


def test_each_vote_index_gets_its_own_cache_entry_and_cannot_replay_another_answer(tmp_path):
    # An older answer for this exact prompt already sits in the normal (unsalted) cache, as after an earlier run.
    inner = FakeProvider(["OLD-unsalted-answer", "vote-1", "vote-2", "vote-3"])
    base = CachedProvider(inner, tmp_path)
    assert base.generate("PROMPT") == "OLD-unsalted-answer"

    votes = vote_providers(base, 3, "run-A/")
    keys = [v._key("PROMPT") for v in votes]
    assert len({*keys, base._key("PROMPT")}) == 4                          # three votes + the unsalted key: all different, vote 1 included

    # three genuinely independent calls: none of them is served the older cached answer, none replays another vote
    assert [v.generate("PROMPT") for v in votes] == ["vote-1", "vote-2", "vote-3"]
    assert len(inner.prompts) == 4 and base.stats.cache_hits == 0

    # a DIFFERENT run (different salt prefix) must make fresh calls too, not replay run A's votes
    other_inner = FakeProvider(["B1", "B2", "B3"])
    other = vote_providers(CachedProvider(other_inner, tmp_path), 3, "run-B/")
    assert [v.generate("PROMPT") for v in other] == ["B1", "B2", "B3"] and len(other_inner.prompts) == 3

    # RESUME of run A: new provider objects, same prefix -> the same three entries are found again, zero calls
    resumed_inner = FakeProvider(["MUST NOT BE CALLED"])
    resumed = vote_providers(CachedProvider(resumed_inner, tmp_path), 3, "run-A/")
    assert [v.generate("PROMPT") for v in resumed] == ["vote-1", "vote-2", "vote-3"] and resumed_inner.prompts == []
    assert len(list(tmp_path.rglob("*.json"))) == 1 + 3 + 3                  # unsalted + run A votes + run B votes
