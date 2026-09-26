# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The caption, the number's width, and the two flag combinations that refuse.

WHY THIS FILE EXISTS. `measure.py` carried the worst of the 2026-09-25 panel's three criticals:
the command scored the prompts a configuration had been chosen on and captioned the figure as
held out. The repair put the caption's wording under `partition_note`, so the claim comes out of
the same stamp the number does. Then CI measured the module at 85.36% with `partition_note`'s
branches almost entirely unread, which is the repair for a caption that lied sitting untested.

That is not a coverage arithmetic problem, it is the same defect shape one layer along: the fix
for a claim nobody checked was itself a claim nobody checked. Every branch below is a different
sentence a reader acts on, and getting the wrong one is how the critical happened in the first
place.

THE THREE GROUPS, and what each is for.

  * `partition_note` decides which of five sentences a row carries. Four of them are hedges and
    one is the strong claim, and only the strong claim may appear over a verified boundary.
  * `_figure` renders one number. Its rule is a property rather than a format: no rendering may
    reach a boundary the value did not, because a KL of 3e-05 printed as `0` says an edit changed
    nothing, and an AUC of 0.99996 printed as `1` says a classifier is perfect.
  * `wanted_stages` refuses two flag combinations that used to be accepted silently, which is the
    shape the panel kept finding: a flag that changes nothing, taken without comment.
"""
import types

import pytest

from senbonzakura import measure, stamps


def _stamped(metric, **block):
    return {"metrics": {metric: block}}


# ── the caption, which is the critical's own repair ──────────────────────────────────

class TestPartitionNote:
    def test_a_verified_boundary_earns_the_strong_claim(self):
        note = measure.partition_note(_stamped("refusal_rate", partition=stamps.MEASURE),
                                      "refusal_rate")
        assert note == "Measured on this track's held-out rows and on no others."

    def test_every_row_says_in_sample_and_says_why_that_matters(self):
        """The exact case the critical produced, and the caption has to give it away."""
        note = measure.partition_note(_stamped("refusal_rate", partition=stamps.ALL_ROWS),
                                      "refusal_rate")
        assert "EVERY row" in note
        assert "in-sample" in note
        assert "not comparable" in note

    def test_an_unconfirmed_boundary_names_the_number_and_refuses_the_claim(self):
        """A skip nothing confirmed is the honest middle case, and it was the missing one.

        The run skipped some rows and no manifest agreed that the skip is where the held-out
        partition begins. The note must carry the count, so a reader can check it, and must not
        borrow the verified wording.
        """
        note = measure.partition_note(
            _stamped("refusal_rate", partition=f"{stamps.UNVERIFIED_PREFIX}132"), "refusal_rate")
        assert "132" in note
        assert "no manifest confirmed" in note
        assert "not established" in note
        assert "and on no others" not in note

    def test_a_partition_nobody_here_knows_is_reported_rather_than_interpreted(self):
        note = measure.partition_note(_stamped("refusal_rate", partition="fit"), "refusal_rate")
        assert "'fit'" in note

    def test_no_partition_at_all_says_the_question_cannot_be_answered(self):
        """ABSENCE AND A VERDICT MUST NOT SHARE A REPRESENTATION. An artefact that records no
        partition is not an artefact recording a good one.
        """
        note = measure.partition_note(_stamped("refusal_rate", value=0.1), "refusal_rate")
        assert "records no partition" in note

    @pytest.mark.parametrize("result", [None, "text", 42, [], {"metrics": None},
                                        {"metrics": {"refusal_rate": "not a block"}}])
    def test_nothing_readable_produces_no_caption_rather_than_a_wrong_one(self, result):
        """An empty string, never a sentence. A caption invented over an unreadable artefact is
        the original defect with the stamp removed.
        """
        assert measure.partition_note(result, "refusal_rate") == ""


# ── the figure, whose rule is a property ─────────────────────────────────────────────

class TestFigure:
    def test_exact_zero_is_a_measurement_and_prints_as_zero(self):
        assert measure._figure(0.0) == "0"

    def test_a_reading_too_small_for_four_places_keeps_its_digits(self):
        """3e-05 printed as `0` is the claim this project has already had to withdraw once."""
        out = measure._figure(3e-05)
        assert float(out) == pytest.approx(3e-05)
        assert out not in ("0", "-0")

    def test_a_negative_reading_too_small_for_four_places_does_not_print_as_minus_zero(self):
        out = measure._figure(-2e-05)
        assert float(out) == pytest.approx(-2e-05)

    def test_a_value_just_under_one_does_not_print_as_one(self):
        """The same defect at the other end, which is why `.4g` was rejected."""
        out = measure._figure(0.99996)
        assert out != "1"
        assert float(out) == pytest.approx(0.99996)

    def test_a_value_at_one_may_print_as_one(self):
        """The ceiling rule is about values BELOW it. A rate that really is 1.0 says so."""
        assert float(measure._figure(1.0)) == 1.0

    def test_an_ordinary_reading_keeps_four_places_and_no_more(self):
        assert measure._figure(13.613728595914115) == "13.6137"

    def test_a_reading_past_twelve_places_falls_back_to_the_exponent_form(self):
        """Fixed point stops fitting a column long before it stops being possible."""
        out = measure._figure(1e-20)
        assert "e-" in out
        assert float(out) == pytest.approx(1e-20)

    def test_an_integer_and_a_bool_are_printed_as_themselves(self):
        """A bool is an int in Python and a count is not a rate. Neither goes through the rule."""
        assert measure._figure(200) == "200"
        assert measure._figure(True) == "True"

    def test_anything_that_is_not_a_number_is_printed_rather_than_formatted(self):
        assert measure._figure("n/a") == "n/a"
        assert measure._figure(None) == "None"


# ── the interval column ──────────────────────────────────────────────────────────────

class TestIntervalOf:
    def test_a_stamped_interval_is_rendered_through_the_same_formatter(self):
        got = measure._interval_of(_stamped("kl", interval=[3e-05, 0.002]), "kl")
        assert got.startswith("[") and got.endswith("]")
        lo, hi = (s.strip() for s in got[1:-1].split(","))
        assert float(lo) == pytest.approx(3e-05), "the interval's low end collapsed to zero"
        assert float(hi) == pytest.approx(0.002)

    def test_a_deterministic_figure_says_so_rather_than_leaving_a_blank(self):
        """A blank reads as an oversight; one forward pass over a fixed passage is a claim."""
        assert measure._interval_of(_stamped("nll", deterministic=True), "nll") == "deterministic"

    @pytest.mark.parametrize("interval", [None, [], [0.1], [0.1, 0.2, 0.3], "0.1 to 0.2"])
    def test_an_interval_that_is_not_a_pair_reads_as_absent(self, interval):
        assert measure._interval_of(_stamped("kl", interval=interval), "kl") == "no interval"

    @pytest.mark.parametrize("result", [None, "text", {"metrics": {"kl": 0.2}}])
    def test_nothing_readable_produces_an_empty_cell(self, result):
        assert measure._interval_of(result, "kl") == ""


# ── the figure a stage wrote, and why it might be missing ────────────────────────────

class TestReadFigure:
    def test_a_result_that_is_not_an_object_says_that_and_not_that_the_figure_is_absent(self):
        _value, why, _units = measure.read_figure("nonsense", "refusal_rate")
        assert "did not parse" in why

    def test_a_metric_stamped_without_a_value_is_a_different_fault_from_an_absent_metric(self):
        """Two faults, two sentences. Only one of them is the user's problem."""
        _v, why_absent, _u = measure.read_figure({"metrics": {}}, "refusal_rate")
        _v, why_empty, _u = measure.read_figure(_stamped("refusal_rate", units="rate"),
                                                "refusal_rate")
        assert "no stamped" in why_absent
        assert "without a value" in why_empty
        assert why_absent != why_empty

    def test_the_units_come_from_the_stamp(self):
        """The first version named them in this module, over a log likelihood it called a
        perplexity.
        """
        value, why, units = measure.read_figure(
            _stamped("nll", value=2.61, units="nats-per-token"), "nll")
        assert (value, why, units) == (2.61, None, "nats-per-token")


# ── the two flag combinations that must refuse ───────────────────────────────────────

def _args(only=(), baseline=None):
    return types.SimpleNamespace(only=list(only), baseline=baseline)


class TestWantedStages:
    def test_drift_alone_refuses_because_there_is_nothing_to_compare_with(self):
        with pytest.raises(SystemExit, match="needs --baseline"):
            measure.wanted_stages(_args(only=["drift"]))

    def test_a_baseline_no_stage_will_read_is_refused_rather_than_ignored(self):
        """A FLAG THAT CHANGES NOTHING IS REFUSED. This combination used to run one stage, never
        mention the baseline, and exit 0, so the operator got a table that looked like the one
        they asked for with the comparison they named a model for missing.
        """
        with pytest.raises(SystemExit, match="only read by the drift stage"):
            measure.wanted_stages(_args(only=["score"], baseline="m"))

    def test_drift_is_dropped_without_complaint_when_no_stage_was_named(self):
        """The default run has no baseline and must not refuse: the user asked for everything
        measurable, not for a comparison.
        """
        assert "drift" not in measure.wanted_stages(_args())

    def test_a_baseline_adds_drift_to_a_default_run(self):
        assert "drift" in measure.wanted_stages(_args(baseline="m"))

    def test_the_order_is_the_declared_order_whatever_order_was_asked_for(self):
        got = measure.wanted_stages(_args(only=["capability", "score"]))
        assert got == ["score", "capability"]


def test_an_empty_table_says_nothing_was_measured():
    """Zero rows is not an empty table, it is a run that measured nothing, and a reader needs the
    difference: a blank table reads as a formatting fault.
    """
    assert measure.format_table([]) == ["nothing was measured"]
