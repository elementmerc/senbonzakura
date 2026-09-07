# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The search can finally see what it is costing.

WHY THIS FILE EXISTS

The search optimised refusal, the keyword rate, drift and brokenness. Not one of those asks the
model to reason, so none of them could see arithmetic going. A model can hold KL 0.128 with
nothing broken and have lost multi-step reasoning, which is exactly the state a real run reported,
and the objective would have called it a clean win.

WHERE THE PROBE SITS, WHICH IS THE WHOLE DESIGN DECISION

On the finalists, not on every trial. Per trial would multiply a 200-trial search by a graded
benchmark. Here it costs --top-rescore generations, on exactly the candidates that could still
win. The catastrophic case, a model that stopped answering at all, is already caught per trial by
the brokenness term; this catches the quieter one, a model that answers fluently and can no longer
reason.

WHY IT IS OFF BY DEFAULT

It needs a graded benchmark the operator supplies. Inventing one would be worse than not
measuring, and a probe with nothing behind it would score every candidate the same while looking
like evidence.
"""
import json
import types

import pytest

from senbonzakura import cli
from senbonzakura.metrics import KNEE_W_CAPABILITY, knee_scalar
from senbonzakura.parser import build_parser

# ── the selection rule ───────────────────────────────────────────────────────────────

def test_a_config_that_costs_more_capability_than_it_removes_refusal_loses():
    """THE POINT OF THE TERM. Before it, the left-hand config won on refusal alone."""
    strips_but_damages = knee_scalar(0.0, 0, 0, 0.05, broken=0.0, capability_drop=0.20)
    leaves_some_intact = knee_scalar(0.1, 0, 0, 0.05, broken=0.0, capability_drop=0.0)
    assert leaves_some_intact < strips_but_damages


def test_a_run_that_measured_no_capability_scores_exactly_as_before():
    """Turning the probe off must not move a selection, or every historical run becomes
    incomparable for a reason nobody chose.
    """
    assert (knee_scalar(0.1, 0.0, 0.2, 0.3, broken=0.05)
            == knee_scalar(0.1, 0.0, 0.2, 0.3, broken=0.05, capability_drop=0.0))


def test_a_capability_gain_is_not_rewarded():
    """At these sample sizes a config scoring above the unedited model is noise, and rewarding
    noise selects for it.
    """
    assert (knee_scalar(0.1, 0, 0, 0.05, broken=0.0, capability_drop=-0.5)
            == knee_scalar(0.1, 0, 0, 0.05, broken=0.0, capability_drop=0.0))


def test_the_exchange_rate_is_one_for_one_and_stated():
    """Nobody has measured what a point of reasoning is worth against a point of refusal, so the
    honest default is that they trade evenly, and it is a named constant rather than a literal.
    """
    assert KNEE_W_CAPABILITY == 1.0
    delta = (knee_scalar(0, 0, 0, 0, broken=0.0, capability_drop=0.3)
             - knee_scalar(0, 0, 0, 0, broken=0.0, capability_drop=0.0))
    assert delta == pytest.approx(0.3)


# ── the probe itself ─────────────────────────────────────────────────────────────────

def _abl(**over):
    obj = cli.Abliterator.__new__(cli.Abliterator)
    args = dict(capability_eval="", capability_n=0, capability_task="numeric",
                capability_max_new=32, hf_token=None, batch_size=2)
    args.update(over)
    obj.args = types.SimpleNamespace(**args)
    obj.log = lambda _m: None
    return obj


def test_no_probe_is_asked_for_by_default():
    assert _abl()._capability_baseline() == (None, [])


def test_a_probe_with_a_dataset_but_no_items_is_still_off():
    """--capability-n 0 means off, and a benchmark path alone must not switch it on."""
    assert _abl(capability_eval="something", capability_n=0)._capability_baseline() == (None, [])


def test_an_unreadable_benchmark_refuses_rather_than_scoring_everything_the_same(tmp_path):
    """A probe with nothing behind it would give every candidate an identical drop of zero, which
    looks like evidence that the edit costs nothing.
    """
    obj = _abl(capability_eval=str(tmp_path / "nope.jsonl"), capability_n=5)
    with pytest.raises(SystemExit, match="capability probe with no benchmark"):
        obj._capability_baseline()


def test_a_drop_cannot_be_computed_without_a_baseline():
    assert _abl()._capability_drop(None, [("q", "#### 1")]) is None
    assert _abl()._capability_drop(0.5, []) is None


def test_a_model_that_answers_nothing_gradeable_is_charged_the_whole_baseline(monkeypatch):
    """THE SAFETY PROPERTY, and the brokenness defect in a new costume if it were missed.

    A candidate that can no longer produce a gradeable answer must not score a drop of zero and
    win. The baseline proved these items ARE gradeable on this model, so answering none of them
    now is the largest drop available, not an absent measurement.
    """
    obj = _abl()
    monkeypatch.setattr(obj, "_capability_score", lambda _items: None)
    assert obj._capability_drop(0.62, [("q", "#### 1")]) == 0.62


def test_a_drop_is_the_difference_and_nothing_else(monkeypatch):
    obj = _abl()
    monkeypatch.setattr(obj, "_capability_score", lambda _items: 0.40)
    assert obj._capability_drop(0.62, [("q", "#### 1")]) == pytest.approx(0.22)


def test_the_baseline_is_taken_on_pristine_weights(monkeypatch, tmp_path):
    """The quantity is a drop and a drop needs a before. Measuring after any bake would compare a
    damaged model against itself and report zero.
    """
    bench = tmp_path / "b.jsonl"
    bench.write_text(json.dumps({"question": "q", "answer": "#### 1"}), encoding="utf-8")
    obj = _abl(capability_eval=str(bench), capability_n=1)
    calls = []
    monkeypatch.setattr(obj, "restore_weights", lambda: calls.append("restored"), raising=False)
    monkeypatch.setattr(obj, "_capability_score", lambda _items: 0.5)
    baseline, items = obj._capability_baseline()
    assert baseline == 0.5
    assert len(items) == 1
    assert calls == ["restored"], "the weights were not restored before the baseline was measured"


def test_the_probe_reads_both_columns_of_the_benchmark(tmp_path):
    """A prompt list is not enough: marking needs the reference answer, and pairing them from two
    reads would be a place to misalign them silently.
    """
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(
        json.dumps({"question": f"q{i}", "answer": f"#### {i}"}) for i in range(4)),
        encoding="utf-8")
    obj = _abl(capability_eval=str(bench), capability_n=3)
    obj.restore_weights = lambda: None
    obj._capability_score = lambda _items: 1.0
    _baseline, items = obj._capability_baseline()
    assert items == [("q0", "#### 0"), ("q1", "#### 1"), ("q2", "#### 2")]


# ── the flags ────────────────────────────────────────────────────────────────────────

def test_the_probe_is_off_by_default_on_the_command_line():
    a = build_parser().parse_args(["--model", "m"])
    assert a.capability_eval == ""
    assert a.capability_n == 0


def test_the_probe_task_choices_match_the_capability_command():
    from senbonzakura.capability import TASK_CHOICES
    accepted = None
    for action in build_parser()._actions:
        if "--capability-task" in action.option_strings:
            accepted = set(action.choices)
    assert accepted == set(TASK_CHOICES), (
        "the search probe and the standalone command must grade the same way, or a config chosen "
        "under one ruler would be reported under another")


def test_the_probe_scores_a_real_model_and_reports_nothing_when_nothing_grades(
        monkeypatch, tiny_model, tiny_tok):
    """The probe end to end on a model that cannot do arithmetic.

    It must return None rather than 0.0. Those are the same number to a formatter and different
    claims to a reader, and here the difference would move a selection: None means "no evidence",
    0.0 would mean "got everything wrong".
    """
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    obj = _abl()
    obj.model, obj.tok, obj.dev = tiny_model, tiny_tok, "cpu"
    assert obj._capability_score([("q1", "#### 1"), ("q2", "#### 2")]) is None


def test_the_probe_returns_nothing_for_an_empty_item_list():
    assert _abl()._capability_score([]) is None


def test_an_ungradeable_candidate_is_charged_the_baseline_end_to_end(
        monkeypatch, tiny_model, tiny_tok):
    """The two pieces together: a model that grades nothing gets the full drop rather than zero.

    This is the path that stops a config which destroyed the model from winning on a drop of
    zero, and it is the brokenness defect in a new costume if it ever regresses.
    """
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    obj = _abl()
    obj.model, obj.tok, obj.dev = tiny_model, tiny_tok, "cpu"
    assert obj._capability_drop(0.71, [("q", "#### 1")]) == 0.71
