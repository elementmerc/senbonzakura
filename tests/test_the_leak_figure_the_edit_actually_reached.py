# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""D1: the leak figure over the positions the edit reached, beside the one over all of them.

THE DEFECT THIS CLOSES. The metric's headline was a mean over every residual-stream position, and
an abliteration does not touch every position: the strength is a taper that peaks at one layer and
falls to zero at a distance, so untouched positions keep the whole component and pull the average
up. Measured on 2026-10-09 (`private/research/leak-mechanism-2026-10-09/`): an edit that had
removed about two thirds of the direction where it acted read 88 to 96 per cent whole-stack,
because fourteen of twenty-five positions had never been edited.

WHY THE OLD FIGURE KEEPS ITS NAME. `mean` is read by `tools/research/matched_leak_strength.py` and
is stamped into published artefacts. Changing what it means would restate every leak figure this
project has published, for a presentation improvement, which is the habit this project has been
right to avoid. The new figure arrives beside it, and the prose says which to quote.
"""
from types import SimpleNamespace

import pytest

from senbonzakura import cli, residualleak

#: The five pinned fields a report carries. Their VALUES do not matter to anything under test
#: here, because `leak_block` never reads them; they are present because the dataclass requires
#: them and a report without them is not a report this project would ever write.
PINNED = {"model": "tiny", "metric": "residual_leak", "input_digest": "0" * 16,
          "prompt_format": "raw", "precision": "float32"}


def _report(per_position, **kw):
    """A LeakReport with a chosen profile, built through the real dataclass."""
    return residualleak.LeakReport(
        leak_per_position=tuple(per_position),
        positions=len(per_position),
        probe_prompts=kw.pop("probe_prompts", 8),
        pinned=kw.pop("pinned", dict(PINNED)),
        output=kw.pop("output", None),
        warnings=kw.pop("warnings", ()),
        **kw)


# ── the arithmetic, and the refusals ──────────────────────────────────────────────────

def test_the_mean_over_chosen_positions_is_the_mean_of_those_positions():
    r = _report([0.0, 1.0, 2.0, 3.0, 4.0])
    assert r.mean_over([1, 3]) == pytest.approx(2.0)
    assert r.mean_over([0]) == pytest.approx(0.0)


def test_duplicates_and_order_do_not_change_the_answer():
    """The caller derives these from two tapers' supports, so a position can arrive twice and the
    set can arrive unsorted. Neither may weight a position double.
    """
    r = _report([0.0, 1.0, 2.0, 3.0])
    assert r.mean_over([3, 1, 1, 3]) == r.mean_over([1, 3]) == pytest.approx(2.0)


def test_an_empty_set_is_refused_rather_than_averaged():
    """A mean over no positions is not a low figure, it is no figure, and the bare arithmetic would
    be a ZeroDivisionError several frames from whoever passed the wrong thing.
    """
    r = _report([0.1, 0.2])
    with pytest.raises(ValueError) as e:
        r.mean_over([])
    msg = str(e.value)
    assert "not a low figure" in msg
    assert "no figure" in msg


def test_a_position_outside_the_profile_is_refused_and_named():
    """The caller and the measurement disagreeing about the model's depth is a bug in one of them,
    so it has to be loud and has to say which positions were wrong.
    """
    r = _report([0.1, 0.2, 0.3])
    with pytest.raises(ValueError) as e:
        r.mean_over([1, 7, -2])
    msg = str(e.value)
    assert "7" in msg and "-2" in msg
    assert "disagree about the model's depth" in msg


# ── the dilution, which is the whole reason the field exists ──────────────────────────

def test_the_two_figures_differ_when_the_edit_reached_only_part_of_the_stack():
    """The 2026-10-09 finding, as a test. The edit removed the component where it acted and the
    untouched positions kept theirs, so the whole-stack mean reads high and the edited mean reads
    low. If these two were ever equal the second figure would be decoration.
    """
    # Positions 0 to 3 untouched and still carrying the component; 4 to 7 edited and mostly clear.
    profile = [0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05]
    r = _report(profile)
    edited = [4, 5, 6, 7]
    assert r.mean == pytest.approx(0.5)
    assert r.mean_over(edited) == pytest.approx(0.05)
    assert r.mean_over(edited) < r.mean / 5, (
        "the edited figure has to be able to differ sharply from the whole-stack one, or reporting "
        "both says nothing a reader could not get from either")


def test_the_whole_stack_mean_is_unchanged_by_the_new_figure_existing():
    """The no-restatement guarantee, asserted rather than asserted in prose: `mean` is what it
    always was, whether or not a caller supplies the edited set.
    """
    r = _report([0.9, 0.1, 0.2, 0.4])
    without = residualleak.leak_block(r)
    with_edited = residualleak.leak_block(r, edited_positions=[1, 2])
    assert without["mean"] == with_edited["mean"] == r.mean


# ── the block, in both states ─────────────────────────────────────────────────────────

def test_a_block_told_the_edited_positions_carries_the_figure_and_the_positions():
    r = _report([0.9, 0.9, 0.1, 0.1])
    block = residualleak.leak_block(r, edited_positions=[2, 3])
    assert block["mean_over_edited_positions"] == pytest.approx(0.1)
    assert block["edited_positions"] == [2, 3]
    assert block["edited_positions_absent_because"] is None
    assert "WHERE IT ACTED" in block["mean_over_edited_positions_answers"]
    assert "never edited" in block["edited_positions_are"]


def test_a_block_with_no_edited_positions_states_the_reason_rather_than_omitting_the_key():
    """An absent key reads as whoever wrote the file forgetting. The standalone command genuinely
    cannot know the edit, and that is a statement rather than a gap.
    """
    r = _report([0.9, 0.1])
    block = residualleak.leak_block(r)
    assert block["mean_over_edited_positions"] is None
    assert block["edited_positions"] is None
    why = block["edited_positions_absent_because"]
    assert "did not perform the edit" in why
    assert "standalone" in why


def test_the_block_says_what_each_of_the_two_figures_answers():
    """Both fields carry their own question, because a record with two means and no labels is
    worse than one mean: a reader quotes whichever they met first.
    """
    r = _report([0.9, 0.1, 0.1])
    block = residualleak.leak_block(r, edited_positions=[1, 2])
    assert "ANYWHERE in the residual stream" in block["mean_answers"]
    assert "every position including the ones the edit never reached" in block["mean_answers"]
    assert "Quote this one about the edit" in block["mean_over_edited_positions_answers"]


def test_an_out_of_range_edited_position_refuses_the_block_rather_than_writing_a_figure():
    """The refusal has to survive being called through `leak_block`, because that is the only path
    a real run takes and a guard that only fires on the direct call is decoration.
    """
    r = _report([0.9, 0.1])
    with pytest.raises(ValueError):
        residualleak.leak_block(r, edited_positions=[0, 5])


def test_the_two_blocks_that_are_not_measurements_are_untouched_by_this():
    """The three states this block keeps apart predate D1 and must not have gained leak fields."""
    not_asked = residualleak.leak_block(requested=False)
    assert not_asked["state"] == residualleak.LEAK_NOT_REQUESTED
    assert "mean_over_edited_positions" not in not_asked

    asked_and_failed = residualleak.leak_block(refused_because="no single direction", prompts=8)
    assert asked_and_failed["state"] == residualleak.LEAK_REQUESTED_BUT_NOT_MEASURED
    assert "mean_over_edited_positions" not in asked_and_failed


def test_the_banned_key_gate_is_still_satisfied_by_the_new_fields():
    """`tools/ci/check_prompt_artefacts.py` refuses an artefact carrying `prompts` at any depth,
    and it is the one control between a harmful prompt set and a public push. A field added here
    must not reintroduce a banned name, and the count stays `probe_prompts`.
    """
    r = _report([0.9, 0.1])
    block = residualleak.leak_block(r, edited_positions=[1])
    banned = {"prompts", "prompt", "text", "texts", "response", "responses",
              "input", "inputs", "output", "outputs", "completion", "completions"}
    assert not (set(block) & banned), sorted(set(block) & banned)
    assert "probe_prompts" in block


def test_a_block_cannot_carry_both_the_positions_and_an_excuse_for_their_absence():
    r = _report([0.9, 0.1])
    with pytest.raises(ValueError, match="both"):
        residualleak.leak_block(r, edited_positions=[1],
                                edited_unavailable_because="and also no")


# ── which positions the taper reached, derived from the bake's own test ───────────────

def test_the_taper_support_is_the_edited_set_and_not_the_search_bounds():
    """The defect's root. `[lo, hi]` bounds where the PEAK may sit; the strength runs out to
    distance D either side, so a peak at layer 2 with D = 2 reaches layers 0 to 4 even though
    `lo` was 1. Verified against a real bake on 2026-10-09: four of six reachable profiles put
    layer 0 in the edit.
    """
    # o-profile peaks at 2 with D = 2; d-profile is all-zero, so the union is the o-support.
    bpr = (2, 1.0, 0.5, 2, 0, 0.0, 0.0, 0)
    assert cli.edited_positions_from_taper(bpr, 8) == [1, 2, 3, 4, 5]


def test_the_union_of_the_two_profiles_is_taken_and_not_either_one():
    """`attn.o_proj` and `mlp.down_proj` follow decoupled profiles, and a layer with either one
    non-zero was edited. Taking one profile alone would under-report the set.
    """
    o_only = (1, 1.0, 1.0, 1, 0, 0.0, 0.0, 0)        # layers 0 to 2
    d_only = (6, 1.0, 1.0, 1, 6, 1.0, 1.0, 1)        # o and d both at 5 to 7
    both = (1, 1.0, 1.0, 1, 6, 1.0, 1.0, 1)
    assert cli.edited_positions_from_taper(o_only, 8) == [1, 2, 3]
    assert cli.edited_positions_from_taper(d_only, 8) == [6, 7, 8]
    assert cli.edited_positions_from_taper(both, 8) == [1, 2, 3, 6, 7, 8]


def test_a_profile_with_no_taper_edits_every_layer():
    """D is None for the "every layer, no taper" profile, which a fixed recipe cannot express as a
    number because it does not know how deep the model is. Every layer is then in the edit, and the
    edited figure correctly equals the whole-stack one bar position 0.
    """
    bpr = (0, 1.0, 1.0, None, 0, 0.0, 0.0, 0)
    assert cli.edited_positions_from_taper(bpr, 5) == [1, 2, 3, 4, 5]


def test_an_all_zero_pair_of_profiles_reaches_no_position():
    """A legal config that edits nothing. The empty set is returned rather than guessed at, and the
    caller is what decides this is a statement about the config rather than a failed measurement.
    """
    assert cli.edited_positions_from_taper((3, 0.0, 0.0, 2, 3, 0.0, 0.0, 2), 8) == []


def test_position_zero_is_never_in_the_edited_set():
    """Position 0 is the embedding output, `embed_tokens` is deliberately left alone, and layer
    `idx` writes into position `idx + 1`. An off-by-one here would average in the one position the
    bake is documented never to touch.
    """
    for n in (1, 4, 32):
        assert 0 not in cli.edited_positions_from_taper((0, 1.0, 1.0, None, 0, 0.0, 0.0, 0), n)


# ── and through the abliterator, which is the only path a real run takes ──────────────

class _Run:
    """The parts of the abliterator `leak_report` reads, with a chosen leak profile."""

    def __init__(self, profile, n_layers):
        self.args = SimpleNamespace(leak_prompts=4)
        self.bad_eval = [f"probe {i}" for i in range(8)]
        self.model, self.tok = object(), object()
        self.layers = [object()] * n_layers
        self.said = []
        self.log = self.said.append
        self.report = _report(profile)

    def active_dirs(self, idx, k):
        return [[1.0, 0.0]]

    leak_report = cli.Abliterator.leak_report


def _drive(run, monkeypatch, bpr):
    monkeypatch.setattr(residualleak, "measure_leak",
                        lambda *a, **k: run.report)
    return run.leak_report(1, "single", bpr)


def test_a_real_run_records_the_figure_over_the_positions_its_taper_reached(monkeypatch):
    # Four layers, so five positions. The o-profile peaks at layer 2 with D = 1, reaching layers
    # 1 to 3 and therefore positions 2, 3 and 4, where the component is mostly gone.
    run = _Run([0.9, 0.9, 0.05, 0.05, 0.05], n_layers=4)
    block = _drive(run, monkeypatch, (2, 1.0, 1.0, 1, 0, 0.0, 0.0, 0))
    assert block["edited_positions"] == [2, 3, 4]
    assert block["mean_over_edited_positions"] == pytest.approx(0.05)
    assert block["mean"] == pytest.approx(0.39), "the whole-stack figure, diluted eightfold"
    assert block["edited_positions_absent_because"] is None


def test_the_terminal_names_the_edited_figure_before_the_whole_stack_one(monkeypatch):
    """A terminal is read top down, and the whole-stack mean was the only figure a watcher saw for
    as long as the defect lasted. The honest one goes first or the fix only reaches the JSON.
    """
    run = _Run([0.9, 0.9, 0.05, 0.05, 0.05], n_layers=4)
    _drive(run, monkeypatch, (2, 1.0, 1.0, 1, 0, 0.0, 0.0, 0))
    said = "\n".join(run.said)
    first = next(i for i, line in enumerate(run.said) if "residual leak over" in line)
    assert "the 3 positions the edit reached" in run.said[first]
    assert "including the 2 the edit never reached" in run.said[first + 1]
    assert said.index("the edit reached") < said.index("the edit never reached")


def test_a_run_given_no_profiles_keeps_the_old_line_and_states_why_there_is_no_second(monkeypatch):
    """`--bake-config` recovery and the forward-only paths may have no profiles to hand. The block
    says so rather than dropping the field, and the printed line is the one it always was.
    """
    run = _Run([0.9, 0.1, 0.1], n_layers=2)
    monkeypatch.setattr(residualleak, "measure_leak", lambda *a, **k: run.report)
    block = run.leak_report(1, "single")
    assert block["mean_over_edited_positions"] is None
    assert "did not perform the edit" in block["edited_positions_absent_because"]
    assert any("across 3 positions" in line for line in run.said)


def test_an_all_zero_config_states_that_rather_than_borrowing_the_depth_reason(monkeypatch):
    run = _Run([0.9, 0.1, 0.1], n_layers=2)
    block = _drive(run, monkeypatch, (0, 0.0, 0.0, 1, 0, 0.0, 0.0, 1))
    why = block["edited_positions_absent_because"]
    assert "edited no position at all" in why
    assert "disagree" not in why, "an empty config is not a depth defect"
    assert block["mean"] == pytest.approx(run.report.mean)


def test_a_depth_disagreement_degrades_loudly_and_never_costs_the_run_its_record(monkeypatch):
    """By the time this is called the weights are on disk and `abliteration.json` is not, so a
    taper that names a position the profile does not have must not raise. It is a defect in one of
    the two, so it is said out loud as well as recorded.
    """
    # Two positions measured, but the run claims eight layers: position 8 does not exist.
    run = _Run([0.9, 0.1], n_layers=8)
    block = _drive(run, monkeypatch, (7, 1.0, 1.0, 0, 0, 0.0, 0.0, 0))
    assert block["state"] == residualleak.LEAK_MEASURED
    assert block["mean"] == pytest.approx(0.5)
    assert block["mean_over_edited_positions"] is None
    why = block["edited_positions_absent_because"]
    assert "disagree about the model's depth" in why
    assert any("no edited-window figure" in line for line in run.said), (
        "a degradation that reaches only a field is invisible to somebody watching a terminal")
