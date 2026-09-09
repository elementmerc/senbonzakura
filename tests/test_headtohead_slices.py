# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for `senbonzakura head-to-head stage`, the shared evaluation slices.

The head-to-head is only a comparison of two tools if both are scored on the same prompts. These
slices are what make that true, and every check here is written around a way they could quietly
stop being the same list rather than around the happy path: a line break that splits one prompt
into two, a blank row that Heretic drops and we keep, padding that Heretic strips and we do not.
"""
import pytest
from datasets import Dataset

from senbonzakura import headtohead_stage as ses
from senbonzakura.cli import kl_eval_slice, rescore_eval_slice


@pytest.fixture
def track(tmp_path):
    """A track with enough rows for a disjoint KL slice and a final slice larger than the search's."""
    def build(n_bad=80, n_good=40, bad_rows=None):
        root = tmp_path / "track"
        bad = bad_rows if bad_rows is not None else [f"harmful prompt {i}" for i in range(n_bad)]
        Dataset.from_dict({"text": bad}).save_to_disk(str(root / "bad_eval_ds"))
        Dataset.from_dict({"text": [f"harmless prompt {i}" for i in range(n_good)]}).save_to_disk(
            str(root / "good_ds"))
        # The fitting partition. A real track has always had it; the fixture did not, because
        # nothing read it until the staging step began writing the prompts Heretic extracts its
        # directions from. A fixture narrower than the thing it stands in for hides exactly this.
        Dataset.from_dict({"text": [f"fit harmful {i}" for i in range(n_bad)]}).save_to_disk(
            str(root / "bad_ds"))
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
    assert run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6) == 0
    assert len(lines(out / "keyword_prompts.txt")) == 8
    # 48 requested minus the 8 the search already saw: the re-score keeps only the fresh tail.
    assert len(lines(out / "final_prompts.txt")) == 40
    assert len(lines(out / "kl_prompts.txt")) == 6


def test_the_fitting_prompts_are_staged_too(track, tmp_path):
    """Heretic takes these as files; we read them from the track. Same rows, or the two tools
    fitted their directions on different evidence and nothing in either tool's own reporting
    would ever say so.
    """
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    assert len(lines(out / "bad.txt")) == 10
    assert len(lines(out / "good.txt")) == 10


def test_the_slices_record_the_corpus_they_were_cut_from(track, tmp_path):
    """Otherwise slices from corpus A beside a run pointed at corpus B is silent."""
    import json

    from senbonzakura import headtohead
    t = track()
    out = tmp_path / "eval"
    run(t, out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    recorded = json.loads((out / headtohead.SLICE_PROVENANCE).read_text(encoding="utf-8"))
    assert recorded["track"] == str(t.resolve())
    assert headtohead.slices_match_track(out, t) == []


def test_every_file_the_benchmark_expects_is_written(track, tmp_path):
    """The staging step and the runner must agree about the filenames, or preflight refuses a
    correctly staged directory and nobody can tell which half is wrong.
    """
    from senbonzakura import headtohead
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    for name in headtohead.SLICE_FILES:
        assert (out / name).is_file(), f"{name} is expected by the runner and never written"


def test_the_kl_slice_is_disjoint_from_the_extraction_prompts(track, tmp_path):
    """The KL slice measures coherence on prompts the directions were NOT fitted on.

    Handing Heretic the head of the harmless set instead would score its coherence on the very
    prompts its directions came from, which flatters both tools and flatters neither honestly.
    """
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    kl = lines(out / "kl_prompts.txt")
    extraction = [f"harmless prompt {i}" for i in range(10)]
    assert not (set(kl) & set(extraction))


def test_the_kl_slice_is_the_one_the_abliterator_would_use(track, tmp_path):
    """Not merely disjoint: byte for byte what our own run is scored on."""
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    ours = kl_eval_slice([f"harmless prompt {i}" for i in range(16)], 10, 6)
    assert lines(out / "kl_prompts.txt") == ours


def test_the_final_slice_is_held_out_from_the_search_slice(track, tmp_path):
    """The re-score sees prompts the search did not, which REVERSES what this test used to say.

    It used to assert containment, on the reasoning that "the re-score has to see the search's
    evidence plus more of it, not a different sample". That is a real argument, and it is an
    argument about comparability: a candidate's re-score number could be read against its search
    number because the second set contained the first.

    It is the wrong trade here, and two panel personas reached that independently. This pass is a
    SELECTION over the top candidates, not a re-measurement to be compared against the search's
    own figure, and its stated purpose is that the winner should not be the best of N draws over
    the small evaluation set the search optimised against. Under containment most of the evidence
    it selects on is that same set: 50% of it for a model under 5B, and 67% in the middle size
    tier, where six extra generation rounds bought thirty-two rows the search had not seen.

    So the sets are disjoint now, and the cost is nothing: `bad_eval_ds` holds 4636 rows in the
    bundled track and the largest disjoint requirement is 192.
    """
    out = tmp_path / "eval"
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    search = lines(out / "keyword_prompts.txt")
    final = lines(out / "final_prompts.txt")
    assert len(final) == 40, "the fresh tail of the window, not the whole window"
    assert not set(search) & set(final), (
        f"{len(set(search) & set(final))} prompts appear in both the search slice and the set "
        f"chosen to be held out from it")


# ── the ways a slice stops being the same list ────────────────────────────────────────
def test_a_prompt_with_a_line_break_is_refused(track, tmp_path):
    rows = [f"harmful {i}" for i in range(80)]
    rows[3] = "first half\nsecond half"
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "line break" in str(e.value)


def test_a_blank_prompt_is_refused(track, tmp_path):
    rows = [f"harmful {i}" for i in range(80)]
    rows[2] = "   "
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "blank" in str(e.value)


def test_a_padded_prompt_is_refused(track, tmp_path):
    """Heretic strips each line on read and senbonzakura does not, so padding splits the two."""
    rows = [f"harmful {i}" for i in range(80)]
    rows[5] = "  harmful with padding  "
    with pytest.raises(SystemExit) as e:
        run(track(bad_rows=rows), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "whitespace" in str(e.value)


def test_a_rescore_slice_too_small_to_state_a_rate_over_is_refused(track, tmp_path):
    """A corpus too short leaves no fresh rows at all, and the re-score then tests nothing.

    THIS CHECK'S PREMISE INVERTED WITH THE SLICES. It used to demand that the final slice be
    strictly LARGER than the search slice, because both were the head of the same file and equal
    sizes meant the same rows twice. They share no rows now, so size stopped being a proxy for
    freshness, and what has to hold instead is that there is enough evidence to be worth the
    generations: the floor this project states a rate over.
    """
    with pytest.raises(SystemExit) as e:
        run(track(n_bad=8), tmp_path / "eval", dir_prompts=10, eval_refusal=8,
            eval_refusal_final=16, eval_kl=6)
    assert "below the 30" in str(e.value)
    assert "--search" in str(e.value), "it has to say what the operator can actually change"


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
    run(track(), out, dir_prompts=10, eval_refusal=8, eval_refusal_final=48, eval_kl=6)
    with open(out / "keyword_prompts.txt", encoding="utf-8") as f:
        as_heretic_reads_it = [line.strip() for line in f if line.strip()]
    assert as_heretic_reads_it == [f"harmful prompt {i}" for i in range(8)]


# ── the re-score slice, on its own ────────────────────────────────────────────────────
def test_the_rescore_slice_is_the_tail_of_the_window_not_past_it():
    """THE BOUNDARY THE FIRST VERSION OF THIS FIX WALKED STRAIGHT PAST.

    `bad_eval_ds` is not an undifferentiated pile of rows. It is the track's SEARCH partition
    followed immediately by its MEASURE partition, and the measure half is where every published
    number comes from. The bundled track holds 132 search rows and 4504 measure rows.

    The first attempt at this fix read `rows[eval_refusal : eval_refusal + eval_refusal_final]`,
    reasoning that 4636 rows meant 192 was free. It put sixty rows of the published measurement
    inside the selection: the original defect leaked the selection into the search's own rows, and
    that leaked it into the rows the result is reported on, which is worse and is exactly what
    `track.flag_violations` refuses.

    So the slice stays inside the first `--eval-refusal-final` rows, which is the window that
    guard already checks, and takes only the part of it the search has not seen.
    """
    rows = [f"prompt {i}" for i in range(4636)]
    got = rescore_eval_slice(rows, 64, 128)
    assert got == rows[64:128]
    assert len(got) == 64
    assert rows.index(got[-1]) < 128, "nothing may be read past --eval-refusal-final"


def test_the_rescore_slice_shares_nothing_with_the_search_slice():
    rows = [f"prompt {i}" for i in range(200)]
    assert not set(rows[:64]) & set(rescore_eval_slice(rows, 64, 128))


def test_a_gap_too_narrow_to_be_reportable_says_so():
    """32 fresh prompts is what the middle size tier buys, and 8 is not worth the generations."""
    rows = [f"prompt {i}" for i in range(200)]
    said = []
    got = rescore_eval_slice(rows, 64, 72, said.append)
    assert len(got) == 8
    assert said and "below the reporting floor" in said[0]
    assert "--eval-refusal-final" in said[0], "the message has to name the flag to move"


def test_a_comfortable_gap_does_not_warn():
    rows = [f"prompt {i}" for i in range(200)]
    said = []
    assert len(rescore_eval_slice(rows, 64, 128, said.append)) == 64
    assert not said
