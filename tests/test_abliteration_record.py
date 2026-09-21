# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What `abliteration.json` says about a run, asserted without baking a model.

WHY THIS FILE EXISTS

`abliteration.json` is the only part of a run that outlives the session. The model card reads
it, the head-to-head report reads it, every resume guard in the run specs reads it, and the
checker reads it. It is this project's primary output in the sense that matters: the weights
are the thing, and this file is the only claim about how the thing was made.

Until 2026-09-21 it was built by a `json.dump` of a 170 line dict literal inside
`_bake_and_save`, and `_bake_and_save` needs a GPU, a real checkpoint and a completed search to
reach. So nothing in the suite had ever looked at the document. Every field added to it over
two months of work was added on the strength of somebody reading a diff, which is the review
standard this project has already watched fail: a check nobody runs is a check that is wrong
whenever it matters.

`build_abliteration_record` is that dict as a pure function, so these tests hand it a stand-in
and read what it produces. Nothing about the contents changed when it was extracted; the point
was to make them visible.

WHAT IS AND IS NOT ASSERTED HERE. These tests check the claims the artefact makes, not that a
particular key exists for its own sake: that the figures carry the provenance which makes them
readable, that the statistic travels with its null and its threshold, that a partial ablation
cannot pass for a whole one, and that the budget caveat travels with the number rather than
only being printed. The full key list is pinned once, separately and deliberately, because a
field silently disappearing from this document is a regression nothing else would catch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from senbonzakura import cli, lengthsweep

#: The eight-tuple the search returns: the o profile then the d profile, each as
#: (max_weight_position, max_weight, min_weight, min_weight_distance).
BPR = (12, 1.0, 0.1, 4, 9, 0.8, 0.2, 3)


def _run(**over):
    """A stand-in for the abliterator, carrying only what the record reads off it.

    Every attribute here is one the builder reaches for. Anything absent is reached through
    `getattr(run, name, None)`, which is the behaviour the last test in this file pins: the
    record says "not measured" rather than inventing a value.
    """
    base = dict(
        KMAX=8,
        ablate_conv=True,
        partial_layers=set(),
        dev="cpu",
        model=SimpleNamespace(config=SimpleNamespace(
            _name_or_path="Qwen/Qwen3-1.7B", _commit_hash="deadbeef")),
        tok=SimpleNamespace(senbon_chat_template="qwen3"),
        gov=SimpleNamespace(report=lambda: {"throttled": True}),
        capture_gov=SimpleNamespace(report=lambda: {"throttled": False}),
        eval_provenance=lambda: {"partition": "selection", "n": 64},
        dirs_per_layer=[3, 2, 1],
        dirs_per_position=[0, 3, 2, 1],
        axis_separations=[0.4, 1.9, 2.2],
        separation_statistic="cohens-d",
        separation_null=0.0,
        max_axis_separation=2.2,
        best_rejected_separation=0.9,
        null_separation_floor=0.35,
        trials_ran=200,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _args(**over):
    base = dict(
        out="/nowhere", per_component=True, model="Qwen/Qwen3-1.7B", track=None,
        seed=7, search="tpe", trials=200, warm_start=False, no_good_orth=False,
        gen_tokens=512, dir_prompts=128, sparsity=0.9, ablation_rounds=4,
        no_norm_restore=False, method="searched",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _build(run=None, args=None, **over):
    post = dict(refusals=0.02, heretic=0.01, broken=0.0, kl=0.13)
    post.update(over)
    return cli.build_abliteration_record(
        run or _run(), args or _args(), BPR, 1, "single", 0, base_ref=0.39, post=post)


def test_the_record_can_be_built_without_a_model_or_a_gpu():
    """The extraction's whole purpose, asserted first so a regression here is unmistakable."""
    rec = _build()
    assert isinstance(rec, dict) and rec["model_id"] == "Qwen/Qwen3-1.7B"


def test_the_refusal_figures_say_which_rows_they_were_taken_on():
    """A rate with no partition beside it is the defect that cost this project a published 0.0%.

    On 2026-08-16 a run's refusal rate travelled as a measured figure when it had been scored
    on the SELECTION partition, which is the set the search ran two hundred trials against and
    therefore the one set it cannot be quoted from. `refusal_eval` is what stops the number
    leaving the file without its caveat.
    """
    rec = _build()
    assert rec["baseline_refusals"] == 0.39
    assert rec["post_bake_refusals"] == 0.02
    assert rec["refusal_eval"] == {"partition": "selection", "n": 64}


def test_the_statistic_travels_with_its_null_and_its_threshold():
    """2.2 means nothing until something says what a direction carrying nothing scores.

    All three are written because any one of them alone is unreadable: a threshold with no
    statistic named cannot be interpreted once there is more than one statistic, and a score
    with no null is a number rather than evidence.
    """
    from senbonzakura import separation

    rec = _build()
    assert rec["separation_statistic"] == "cohens-d"
    assert rec["separation_null"] == 0.0
    assert rec["axis_separation_threshold"] == separation.get("cohens-d").threshold
    assert rec["null_separation_floor"] == 0.35


def test_the_threshold_follows_the_statistic_that_was_actually_used():
    """The pairing, not just the presence. A threshold belonging to another statistic would
    read as a sound record and describe a comparison nobody made.
    """
    from senbonzakura import separation

    other = next(s for s in separation.STATISTICS if s != "cohens-d")
    rec = _build(run=_run(separation_statistic=other))
    assert rec["separation_statistic"] == other
    assert rec["axis_separation_threshold"] == separation.get(other).threshold


def test_a_partial_ablation_cannot_pass_for_a_whole_one():
    """On a hybrid architecture the convolution blocks write the residual stream too, so a run
    told to leave them alone produces a model whose refusal behaviour is only partly removed.
    That is a legitimate arm of an experiment and an indefensible thing to publish unlabelled,
    so the flag and the layers it skipped are both recorded rather than inferred from a log.
    """
    rec = _build(run=_run(ablate_conv=False, partial_layers={4, 1, 9}))
    assert rec["ablate_conv"] is False
    assert rec["partially_ablated_layers"] == [1, 4, 9]


def test_the_three_flags_that_describe_the_surgery_are_recorded_together():
    """`sparsity` alone could not tell one arm from another. Before 2026-09-12 the refinement
    rounds re-projected every row with no mask, so `sparsity 0.9, rounds 4` edited the whole
    weight while the artefact said 0.9, and a peer asked which of their arms had set both.
    """
    rec = _build(args=_args(sparsity=0.9, ablation_rounds=4, no_norm_restore=True))
    assert rec["sparsity"] == 0.9
    assert rec["ablation_rounds"] == 4
    assert rec["norm_restore"] is False


def test_the_budget_caveat_travels_with_the_number():
    """A short budget scrolled one warning past an operator at the start of a run and then
    wrote an artefact that looked like any other, so the caveat was lost exactly where the
    figure got quoted from. None when the budget is sound, so its presence is the signal.
    """
    short = _build(args=_args(gen_tokens=48))["generation"]["budget_warning"]
    sound = _build(args=_args(gen_tokens=lengthsweep.VISIBILITY_FLOOR))["generation"]["budget_warning"]
    assert short and "48 tokens" in short
    assert sound is None


def test_both_direction_views_are_written_with_the_note_that_disambiguates_them():
    """`directions_per_layer` is the count EXTRACTED, not the count ablated, and a reader
    meeting [3, 2, 1] beside num_directions: 1 will otherwise read the list as the edit.
    """
    rec = _build()
    assert rec["directions_per_layer"] == [3, 2, 1]
    assert rec["directions_per_position"] == [0, 3, 2, 1]
    assert rec["num_directions"] == 1
    assert "NOT THE NUMBER ABLATED" in rec["directions_index_note"]


def test_the_record_names_the_checkpoint_and_the_revision_it_resolved_to():
    """A Hub id resolves to whatever the Hub serves on the day, so a third party redoing this
    next year gets a different checkpoint under the same string. `model` is the path as given,
    which inside a sealed container is a mount point rather than an identity.
    """
    rec = _build()
    assert rec["model"] == "Qwen/Qwen3-1.7B"
    assert rec["model_id"] == "Qwen/Qwen3-1.7B"
    assert rec["model_revision"] == "deadbeef"


def test_what_was_asked_for_and_what_ran_are_both_recorded():
    """They came apart on 2026-08-06, when a resumed arm ran its full budget a second time and
    every artefact it wrote still reported the budget. An equal-budget claim that cannot be
    checked against the artefact is not a claim.
    """
    rec = _build(run=_run(trials_ran=413), args=_args(trials=200))
    assert rec["trials"] == 200
    assert rec["trials_ran"] == 413


@pytest.mark.parametrize("field", [
    "max_axis_separation", "best_rejected_separation", "null_separation_floor",
    "axes_measured_total", "axes_rejected_total", "matching_quality", "trials_ran",
    "hedge_applied_layers", "directions_per_layer", "axis_separations",
])
def test_a_figure_that_was_never_measured_is_null_rather_than_a_default(field):
    """THE SILENT-DEGRADATION CASE. Every one of these is read through `getattr(run, name,
    None)`, and None is the only honest answer when a search did not produce it: a zero would
    read as "measured, and it was zero", which is a different and much stronger claim.

    The stand-in here carries none of these attributes at all, which is the shape of a run that
    failed early or took a path that skips the measurement.
    """
    bare = SimpleNamespace(
        KMAX=8, ablate_conv=True, partial_layers=set(), dev="cpu",
        model=SimpleNamespace(config=SimpleNamespace(_name_or_path=None, _commit_hash=None)),
        tok=SimpleNamespace(senbon_chat_template=None),
        gov=SimpleNamespace(report=dict), capture_gov=SimpleNamespace(report=dict),
        eval_provenance=dict)
    assert _build(run=bare)[field] is None


#: Every key the document has carried since it was extracted, pinned so that one going missing
#: is a failure here rather than a field a downstream reader silently stops finding. Adding a
#: key is expected and the test says so; removing one is a decision that should be made on
#: purpose, because something out there reads it.
EXPECTED_KEYS = {
    "per_component", "o_profile", "d_profile", "num_directions", "dir_mode", "direction_index",
    "max_directions", "ablate_conv", "partially_ablated_layers", "baseline_refusals",
    "post_bake_refusals", "post_bake_heretic", "post_bake_broken", "post_bake_kl",
    "refusal_eval", "generation", "direction_capture", "sparsity", "ablation_rounds",
    "norm_restore", "model", "model_id", "model_revision", "track_digest", "seed", "search",
    "trials", "trials_ran", "warm_start", "good_orth", "chat_template", "directions_per_layer",
    "directions_per_position", "directions_index_note", "axis_separations",
    "separation_statistic", "separation_null", "matched_scoring", "matched_source", "method",
    "matching_quality", "match_closeness", "axis_separation_threshold", "axes_measured_total",
    "max_axis_separation", "best_rejected_separation", "axes_rejected_total",
    "null_separation_floor", "null_separation_floor_per_layer", "axes_rejected_by_null",
    "separation_held_out", "hedge_applied_layers", "filter_is_unsatisfiable", "provenance",
}


def test_no_field_disappears_from_the_record_without_somebody_noticing():
    got = set(_build())
    assert not EXPECTED_KEYS - got, (
        f"these fields have gone from abliteration.json: {sorted(EXPECTED_KEYS - got)}. "
        f"Something downstream reads each of them. Removing one is a decision; make it on "
        f"purpose and update EXPECTED_KEYS in the same commit.")
    assert not got - EXPECTED_KEYS, (
        f"new fields in abliteration.json: {sorted(got - EXPECTED_KEYS)}. Add them to "
        f"EXPECTED_KEYS here, and check the model card and the head-to-head report still parse.")


# ── the label that described a run it was not having ─────────────────────────────
#
# Separate from the record above, and here rather than in its own file because it is the same
# defect: a statement about how a run was made that nothing checked against how it was made.
class TestTheRunLogSaysHowTheDirectionsWereBuilt:
    """`--no-good-orth` is the toggle that isolates the projection, and the log ignored it.

    Recovered evidence, 2026-09-21: a good-orth A/B run on 2026-07-21 was salvaged from a build
    box before deletion. Both arms' logs read "good-orthogonalized" and both `abliteration.json`
    files carried eleven fields with no flags in them at all, so nothing in the run's own output
    could say which arm was the control. The numbers survived; the experiment did not.
    """

    def test_the_label_follows_the_flag(self):
        assert cli.orthogonalisation_label(False) == "good-orthogonalized"
        assert "no-good-orth" in cli.orthogonalisation_label(True)

    def test_the_two_labels_are_not_the_same_string(self):
        """THE MUTATION THIS FILE EXISTS TO KILL. The original was a constant, so the arms were
        indistinguishable in the log. Any change that collapses them back fails here.
        """
        assert cli.orthogonalisation_label(True) != cli.orthogonalisation_label(False)

    def test_the_raw_arm_does_not_claim_the_property_it_did_not_have(self):
        """Not just different: the raw label must not contain the word that names the projection,
        or a reader grepping the log for it finds both arms again.
        """
        assert "good-orthogonaliz" not in cli.orthogonalisation_label(True).replace("no-good-orth", "")

    def test_the_record_still_carries_the_flag_beside_the_log_line(self):
        """The log is for a human reading a run; the artefact is what survives it. The July
        records had neither, and one without the other is how this was lost.
        """
        assert _build(args=_args(no_good_orth=True))["good_orth"] is False
        assert _build(args=_args(no_good_orth=False))["good_orth"] is True
