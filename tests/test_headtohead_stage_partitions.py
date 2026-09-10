# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The stage command builds the slices BOTH tools are scored on, and never checked the boundaries.

WHAT WAS WRONG

`track.flag_violations` exists to refuse counts that read past the partitions a track records, and
until 2026-09-10 its only caller was the abliterate path in `cli.py`. `headtohead stage` takes the
same five counts as free integers, reads each dataset by head-count, and consulted no manifest.

On a track whose harmful search partition is 96 rows, the default `--eval-refusal-final 128` cuts
32 rows out of the MEASURE partition. Both tools are then selected by best-of-N on rows they are
later scored on. Every artefact is present, every arm completes, nothing warns, and the published
refusal comparison describes the rows the winner was chosen on.

That is the C4 defect from the September panel reappearing through the one path that builds the
slices both tools share. The bundled track happens to survive it; nothing enforced that, which is
the whole point of a guard.
"""
import json
import shutil
from pathlib import Path

import pytest

from senbonzakura import headtohead_stage

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy-track"


@pytest.fixture
def toy(tmp_path):
    """A real, readable track. Its harmful search partition is 4 rows."""
    dst = tmp_path / "track"
    shutil.copytree(TOY, dst)
    return dst


def test_the_stage_refuses_counts_that_reach_into_the_measure_partition(toy, tmp_path):
    """THE DEFECT. 128 rows asked of a track holding 4 for selection."""
    with pytest.raises(SystemExit) as e:
        headtohead_stage.main(["--track", str(toy), "--out", str(tmp_path / "out"),
                               "--eval-refusal", "2", "--eval-refusal-final", "128",
                               "--dir-prompts", "8", "--eval-kl", "4"])
    said = str(e.value)
    assert "read past the boundaries" in said
    assert "4 harmful rows this track holds for selection" in said
    assert not (tmp_path / "out" / "final_prompts.txt").exists(), (
        "nothing may be written before the check: a slice on disk is a slice something picks up, "
        "and a half-staged run is how a partial comparison gets scored as a whole one")


def test_the_harmless_side_is_checked_too(toy, tmp_path):
    """--dir-prompts plus --eval-kl reaching past fit+search puts the KL reference on rows the
    compass reports on.
    """
    with pytest.raises(SystemExit) as e:
        headtohead_stage.main(["--track", str(toy), "--out", str(tmp_path / "out"),
                               "--eval-refusal", "2", "--eval-refusal-final", "4",
                               "--dir-prompts", "64", "--eval-kl", "64"])
    assert "harmless rows" in str(e.value)


def test_counts_inside_the_partition_get_past_this_check(toy, tmp_path):
    """The gate must not be so blunt that nothing can ever be staged.

    The toy track is 12 harmful rows, so it still trips the separate reporting-floor refusal
    further down; what matters here is that it is NOT stopped by the partition check, and that the
    two refusals stay distinguishable. A gate that says the wrong thing is worse than no gate.
    """
    with pytest.raises(SystemExit) as e:
        headtohead_stage.main(["--track", str(toy), "--out", str(tmp_path / "out"),
                               "--eval-refusal", "2", "--eval-refusal-final", "4",
                               "--dir-prompts", "8", "--eval-kl", "4"])
    said = str(e.value)
    assert "read past the boundaries" not in said, (
        "these counts are inside every partition the track records")
    assert "below the 30 this project will state a rate over" in said


def test_a_track_with_no_manifest_says_the_boundaries_are_unchecked(toy, tmp_path, capsys):
    """An unanswerable question must not read as a reassuring answer.

    Without a manifest the counts genuinely cannot be checked, and staying silent about that is
    the same failure as the disk gate returning "no complaint" when it could not measure anything.
    """
    (toy / "track.json").unlink()
    with pytest.raises(SystemExit):
        headtohead_stage.main(["--track", str(toy), "--out", str(tmp_path / "out"),
                               "--eval-refusal", "2", "--eval-refusal-final", "4",
                               "--dir-prompts", "8", "--eval-kl", "4"])
    err = capsys.readouterr().err
    assert "carries no track.json" in err
    assert "NOTHING here can tell you" in err


def test_the_guard_is_the_same_one_the_abliterator_uses(toy):
    """Two copies of a boundary rule drift, and this rule is the one holding the partition."""
    from senbonzakura import track as track_mod
    manifest = json.loads((toy / "track.json").read_text(encoding="utf-8"))
    assert track_mod.flag_violations(manifest, eval_refusal_final=128), (
        "the stage must be asking the same function, not a second implementation of it")


def test_the_selection_gets_a_coherence_slice_the_rival_did_not_tune_against(toy, tmp_path):
    """`kl_prompts.txt` is handed to Heretic during its search as the set its own KL is computed
    on. Ranking six candidates by a KL measured there asks each tool how it did on prompts one of
    them optimised against, which is the confound `drift_prompt_slice` already refuses for the
    published figure and which was never carried across to the selection.
    """
    out = tmp_path / "out"
    with pytest.raises(SystemExit):          # the toy track trips the reporting floor
        headtohead_stage.main(["--track", str(toy), "--out", str(out),
                               "--eval-refusal", "2", "--eval-refusal-final", "4",
                               "--dir-prompts", "4", "--eval-kl", "4"])
    search = (out / "kl_prompts.txt").read_text(encoding="utf-8").split("\n")
    select = (out / "bestofn_kl_prompts.txt").read_text(encoding="utf-8").split("\n")
    assert search and select
    overlap = {r for r in search if r.strip()} & {r for r in select if r.strip()}
    assert not overlap, (
        f"the selection's coherence slice shares {len(overlap)} row(s) with the one Heretic's "
        f"search optimises against")


def test_the_new_slice_is_one_the_harness_actually_asks_for():
    """A file staged under a name nothing reads is not a fix."""
    from senbonzakura import headtohead
    assert "bestofn_kl_prompts.txt" in headtohead.SLICE_FILES
