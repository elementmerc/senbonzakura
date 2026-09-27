# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A model that hardly refuses anything is refused before the search, not after it.

WHAT PROMPTED IT, 2026-09-27

A reader with a 6 GB card ran the documented one-command form on Qwen3-0.6B and waited 108 minutes.
The model refused 3 of 64 prompts before the edit and 3 of 64 after, having paid 12.5% coherence
drift for the privilege. The run had measured that 4.7% baseline at the 51-second mark and said
nothing, then searched for another 107 minutes.

Their own summary is the reason this exists: the wasted time is the symptom, and the real cost is
finishing unable to tell whether the tool works, because the one run they could afford had no room to
show them anything.

The tool already did exactly this for the other metric. `capability.headroom` is a recorded object
carrying `sufficient` and `why`, and it gates the capability probe. Nothing gated the thing this tool
exists to remove.

THE THRESHOLD CAME FROM MEASUREMENT

Six stock models, three families, one corpus, n=128 each. The distribution is bimodal rather than
continuous, which is what makes a single number safe:

    TinyLlama-1.1B-Chat        0.8%   (1 of 128)
    Qwen3-0.6B                 4.7%
    SmolLM2-135M-Instruct      6.2%
    Qwen2.5-1.5B-Instruct     80.5%   (103 of 128)

Seventy points of daylight, so the exact figure carries almost no weight. Operator decision: 5%,
near the bottom of the low cluster. Low enough to leave any model with real refusal alone, high
enough to catch the run that prompted it.

The tests below pin the BEHAVIOUR and the message's obligations rather than the number, except where
the number is the point. A threshold nobody can move is worse than one nobody measured.
"""
import pytest

from senbonzakura import cli


class _Run:
    """The three attributes the gate reads, and nothing else. No torch, no model, no card."""

    def __init__(self, n=64, *, low_refusal_ok=False):
        self.bad_eval = list(range(n))
        self.args = type("Args", (), {"low_refusal_ok": low_refusal_ok})()
        self.lines = []

    def log(self, message):
        self.lines.append(message)


def _refuse(run, rate):
    return cli.Abliterator.refuse_if_there_is_nothing_to_remove(run, rate)


class TestWhenItStops:
    @pytest.mark.parametrize(("rate", "label"), [
        pytest.param(1 / 128, "TinyLlama's measured 0.8%", id="tinyllama"),
        pytest.param(3 / 64, "the reader's measured 4.7%", id="the-108-minute-run"),
        pytest.param(0.0, "a model that refuses nothing at all", id="nothing-at-all"),
        pytest.param(0.049, "just under the floor", id="just-under"),
    ])
    def test_it_refuses_below_the_floor(self, rate, label):
        with pytest.raises(SystemExit) as exit_info:
            _refuse(_Run(), rate)
        assert str(exit_info.value), f"refused {label} with an empty message"

    @pytest.mark.parametrize("rate", [0.05, 0.062, 0.805, 1.0])
    def test_it_says_nothing_above_the_floor(self, rate):
        run = _Run()
        assert _refuse(run, rate) is None
        assert not run.lines, f"a model at {rate} is fine and the run should not comment: {run.lines}"

    def test_the_floor_is_the_measured_one(self):
        """Pinned, because the measurement behind it is recorded and a silent change loses it."""
        assert cli.LOW_REFUSAL_FLOOR == 0.05


class TestTheEscapeHatch:
    def test_the_flag_lets_it_through(self):
        run = _Run(low_refusal_ok=True)
        assert _refuse(run, 3 / 64) is None

    def test_and_it_says_so_rather_than_going_quiet(self):
        """A bypass nobody can see in the log is indistinguishable from the gate not existing."""
        run = _Run(low_refusal_ok=True)
        _refuse(run, 3 / 64)
        said = " ".join(run.lines)
        assert "--low-refusal-ok" in said, f"the bypass left no trace: {run.lines}"
        assert "3 of 64" in said, f"the bypass did not record what it let through: {run.lines}"


class TestTheMessageDoesItsJob:
    """The operator asked for simple grammar, the expectation, the consequence, and what to type."""

    @pytest.fixture
    def message(self):
        with pytest.raises(SystemExit) as exit_info:
            _refuse(_Run(n=64), 3 / 64)
        return str(exit_info.value)

    def test_it_reports_counts_rather_than_only_a_rate(self, message):
        """At n=64 a "4.7%" is three prompts, and reads as precision it has not got.

        This project's own reporting floor prints counts below it when writing a card; the refusal
        that stops a run should hold to the same standard.
        """
        assert "3 of 64" in message, f"no count in the message: {message}"

    def test_it_says_what_would_happen_if_it_carried_on(self, message):
        assert "carried on" in message
        assert "coherence cost" in message, "the consequence has to include what the edit costs"

    def test_it_names_the_flag_to_proceed(self, message):
        assert "--low-refusal-ok" in message

    def test_it_offers_a_way_to_check_another_model_cheaply(self, message):
        assert "senbonzakura score" in message
        assert "--eval default/bad_eval_ds" in message

    def test_it_does_not_claim_size_predicts_refusal(self, message):
        """A first draft said most models above 1B refuse, and the measurement disagreed.

        TinyLlama-1.1B refused 1 prompt in 128. Asserting the rule would have shipped a claim the
        data behind this very threshold contradicts, in the message announcing the threshold.
        """
        assert "Size is" in message and "no guide" in message
        assert "1.1B refused 1 prompt in 128" in message, (
            "the counter-example is what makes the advice honest; keep it or rewrite the advice")


class TestASampleTooSmallToMakeTheClaim:
    """A low rate on four prompts is not evidence of anything, so the run is not stopped for it.

    WHY THIS BRANCH EXISTS, 2026-09-27

    Adding the gate turned 32 tests red in one run. They drive a full abliteration over toy
    fixtures, some with a single eval prompt, and a synthetic model refuses nothing, so the gate
    fired on every one of them.

    The convenient fix would have been to pass `--low-refusal-ok` throughout the suite, and it would
    have been wrong. Refusing a run because a rate is low, when that rate rests on fewer rows than
    this project's own reporting floor, is precisely the defect its checker withdraws figures for:
    `rate-reported-on-a-sample-too-small-to-carry-it`. On four prompts every possible answer is 0%,
    25%, 50%, 75% or 100%, and none of them describes the model.

    So the gate declines to speak below `MIN_REPORTABLE_N`, which is the same constant the reporting
    floor uses rather than a second copy of 30. A real user running `--eval-refusal 8` also gets no
    gate, and that is correct: nothing could be concluded from eight prompts either.
    """

    def test_it_does_not_stop_a_run_on_a_handful_of_prompts(self):
        from senbonzakura.metrics import MIN_REPORTABLE_N

        for n in (1, 4, MIN_REPORTABLE_N - 1):
            run = _Run(n=n)
            assert _refuse(run, 0.0) is None, f"refused on {n} prompts, which cannot carry a rate"

    def test_it_says_that_it_is_not_checking(self):
        """A gate that quietly does not apply is indistinguishable from a gate that passed."""
        run = _Run(n=4)
        _refuse(run, 0.0)
        said = " ".join(run.lines)
        assert "not checking refusal headroom" in said, (
            f"the skip left no trace in the log, so a reader cannot tell it from a pass: {run.lines}")
        assert "4" in said, f"the skip did not say how few prompts there were: {run.lines}"

    def test_it_does_check_at_the_floor(self):
        """The boundary in the other direction, so the skip cannot quietly swallow real runs."""
        from senbonzakura.metrics import MIN_REPORTABLE_N

        with pytest.raises(SystemExit):
            _refuse(_Run(n=MIN_REPORTABLE_N), 0.0)

    def test_the_floor_is_shared_with_the_reporting_floor(self):
        """Two copies of 30 would drift, and the run would refuse on a rate a card would not print."""
        import inspect

        from senbonzakura.metrics import MIN_REPORTABLE_N

        source = inspect.getsource(cli.Abliterator.refuse_if_there_is_nothing_to_remove)
        assert "MIN_REPORTABLE_N" in source, (
            "the small-sample floor is written as a literal rather than shared with the reporting "
            "floor, so the number a run refuses on can drift from the number a card prints")
        assert MIN_REPORTABLE_N == 30
