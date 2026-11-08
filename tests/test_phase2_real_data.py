"""Phase 2 against the real parsed series - still fully offline (FakeProvider, no API).

Skipped when the confidential pack is absent."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import cand, llm_reply
from promoai.detect import plan_windows, run_detection
from promoai.index import ScreenplayIndex
from promoai.ingest import find_archives
from promoai.llm import FakeProvider
from promoai.pipeline import run
from promoai.prompts import estimate_tokens

INPUT = Path(__file__).resolve().parent.parent / "data" / "input"
if not (INPUT.is_dir() and find_archives(INPUT)):
    pytest.skip("no screenplay ZIP in data/input", allow_module_level=True)

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2")
    [(_, _, out_dir)] = run(INPUT, root / "x", root / "o")
    return ScreenplayIndex.from_output_dir(out_dir)


def ordered_ids(index):
    return [e.episode_id for e in sorted(index.episodes.values(), key=lambda e: e.episode_number)]


def echo_script(prompt: str) -> str:
    """A fake 'LLM' that returns a valid candidate built from IDs actually present in the prompt."""
    ids = re.findall(r"^(E\d{3}-S\d{2}-L\d{3}) \[", prompt, re.M)
    return llm_reply([cand("C1", segs=((ids[0], ids[min(5, len(ids) - 1)]),))])


def test_every_prompt_for_the_full_series_references_only_real_ids(index):
    prov = FakeProvider(echo_script)
    ids = ordered_ids(index)
    run_ = run_detection(index, prov, ids, window_size=6, overlap=1)
    assert len(run_.windows) == len(plan_windows(ids, 6, 1)) > 1
    assert all(w.status == "ok" for w in run_.windows)
    seen_lines = set()
    for prompt, w in zip(prov.prompts, run_.windows):
        in_prompt = re.findall(r"^(E\d{3}-S\d{2}-L\d{3}) \[", prompt, re.M)
        assert in_prompt and all(i in index.lines for i in in_prompt)
        expected = [l.line_id for e in w.episode_ids for l in index.episodes[e].lines]
        assert in_prompt == expected                 # every line of the window, once, in file order
        seen_lines.update(in_prompt)
    assert seen_lines == set(index.lines)            # across windows, every line was shown to the model
    assert len(run_.accepted) == len(run_.windows) and run_.rejected == []


def test_prompt_budget_is_in_the_expected_range(index):
    prov = FakeProvider(echo_script)
    run_detection(index, prov, ordered_ids(index), window_size=6, overlap=1)
    sizes = [estimate_tokens(p) for p in prov.prompts]
    assert max(sizes) < 30_000                       # far below any model's context limit
    assert 120_000 < sum(sizes) < 200_000            # ~146k measured for the 50-episode pack


def test_story_context_stays_bounded_across_all_windows(index):
    def script(prompt, _n=[0]):
        _n[0] += 1
        return llm_reply([], characters=tuple((f"P{_n[0]}-{i}", ()) for i in range(10)), revelations=tuple(f"r{_n[0]}-{i}" for i in range(5)))
    run_ = run_detection(index, FakeProvider(script), ordered_ids(index), window_size=6, overlap=1)
    ctx = run_.final_context
    assert len(ctx.characters) <= 25 and len(ctx.revelations) <= 15
    assert ctx.through_episode_id == ordered_ids(index)[-1]


def test_cross_episode_ids_from_real_data_validate(index):
    first, second = index.episodes["E001"].lines, index.episodes["E002"].lines
    c = cand("X", segs=((first[-4].line_id, first[-1].line_id), (second[0].line_id, second[3].line_id)),
             hook=first[-4].line_id, cliff=second[3].line_id)
    run_ = run_detection(index, FakeProvider([llm_reply([c])]), ["E001", "E002"], window_size=2, overlap=0)
    assert len(run_.accepted) == 1 and run_.accepted[0].cross_episode
