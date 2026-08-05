"""Tests for bench/stage_eval_slices.py, the shared evaluation slices.

The head-to-head is only a comparison of two tools if both are scored on the same prompts. These
slices are what make that true, and every check here is written around a way they could quietly
stop being the same list rather than around the happy path: a line break that splits one prompt
into two, a blank row that Heretic drops and we keep, padding that Heretic strips and we do not.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from senbonzakura.cli import kl_eval_slice

_SPEC = importlib.util.spec_from_file_location(
    "stage_eval_slices", Path(__file__).resolve().parent.parent / "bench" / "stage_eval_slices.py")
ses = importlib.util.module_from_spec(_SPEC)
sys.modules["stage_eval_slices"] = ses
_SPEC.loader.exec_module(ses)


@pytest.fixture
def track(tmp_path):
    """A track with enough rows for a disjoint KL slice and a final slice larger than the search's."""
    def build(n_bad=40, n_good=40, bad_rows=None):
        root = tmp_path / "track"
        bad = bad_rows if bad_rows is not None else [f"harmful prompt {i}" for i in range(n_bad)]
        Dataset.from_dict({"text": bad}).save_to_disk(str(root / "bad_eval_ds"))
        Dataset.from_dict({"text": [f"harmless prompt {i}" for i in range(n_good)]}).save_to_disk(
            str(root / "good_ds"))
        return root
    return build


def run(track_dir, out, **kw):
    args = ["--track", str(track_dir), "--out", str(out)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return ses.main(args)


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


# ── the slices themselves ─────────────────────────────────────────────────────────────
def test_writes_three_slices_at_the_requested_sizes(track, tmp_path):
    out = tmp_path / "eval"
    assert run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=16, eval_kl=6) == 0
    assert len(lines(out / "keyword_prompts.txt")) == 8
    assert len(lines(out / "final_prompts.txt")) == 16
    assert len(lines(out / "kl_prompts.txt")) == 6


def test_the_kl_slice_is_disjoint_from_the_extraction_prompts(track, tmp_path):
    """The KL slice measures coherence on prompts the directions were NOT fitted on.

    Handing Heretic the head of the harmless set instead would score its coherence on the very
    prompts its directions came from, which flatters both tools and flatters neither honestly.
    """
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=16, eval_kl=6)
    kl = lines(out / "kl_prompts.txt")
    extraction = [f"harmless prompt {i}" for i in range(10)]
    assert not (set(kl) & set(extraction))


def test_the_kl_slice_is_the_one_the_abliterator_would_use(track, tmp_path):
    """Not merely disjoint: byte for byte what our own run is scored on."""
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=16, eval_kl=6)
    ours = kl_eval_slice([f"harmless prompt {i}" for i in range(16)], 10, 6)
    assert lines(out / "kl_prompts.txt") == ours


def test_the_final_slice_contains_the_search_slice(track, tmp_path):
    """The re-score has to see the search's evidence plus more of it, not a different sample."""
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=16, eval_kl=6)
    search = lines(out / "keyword_prompts.txt")
    assert lines(out / "final_prompts.txt")[:len(search)] == search


# ── the ways a slice stops being the same list ────────────────────────────────────────
def test_a_prompt_with_a_line_break_is_refused(track, tmp_path):
    rows = [f"harmful {i}" for i in range(40)]
    rows[3] = "first half\nsecond half"
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "line break" in str(e.value)


def test_a_blank_prompt_is_refused(track, tmp_path):
    rows = [f"harmful {i}" for i in range(40)]
    rows[2] = "   "
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "blank" in str(e.value)


def test_a_padded_prompt_is_refused(track, tmp_path):
    """Heretic strips each line on read and senbonzakura does not, so padding splits the two."""
    rows = [f"harmful {i}" for i in range(40)]
    rows[5] = "  harmful with padding  "
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "whitespace" in str(e.value)


def test_a_final_slice_no_larger_than_the_search_slice_is_refused(track, tmp_path):
    """A corpus too short makes the two slices equal, and the re-score then tests nothing."""
    with pytest.raises(SystemExit) as e:
        run(track(n_bad=8), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "not larger" in str(e.value)


def test_a_final_count_below_the_search_count_is_refused_up_front(track, tmp_path):
    with pytest.raises(SystemExit) as e:
        run(track(), tmp_path / "eval", dir_prompts=10, eval_refusal=16,
            eval_refusal_final=8, eval_kl=6)
    assert "smaller than" in str(e.value)


def test_a_missing_track_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        run(tmp_path / "nowhere", tmp_path / "eval")
    assert "no track" in str(e.value)


# ── the format Heretic actually reads ─────────────────────────────────────────────────
def test_a_written_slice_round_trips_through_heretics_reader(track, tmp_path):
    """Heretic reads `[line.strip() for line in file if line.strip()]`.

    Reproduced here rather than imported, because Heretic lives in the benchmark container and not
    in this environment. If that reader ever changes, this test still describes what we wrote for.
    """
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=16, eval_kl=6)
    with open(out / "keyword_prompts.txt", encoding="utf-8") as f:
        as_heretic_reads_it = [line.strip() for line in f if line.strip()]
    assert as_heretic_reads_it == [f"harmful prompt {i}" for i in range(8)]
