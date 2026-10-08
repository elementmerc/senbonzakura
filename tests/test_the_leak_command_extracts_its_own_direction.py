# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The standalone `leak` command, and the two things it must not get wrong.

WHY THE COMMAND EXISTS

`abliterate --leak-report` reports the residual leak for a model this tool just edited, which is
the cheap case because the direction is already in hand. This is the other case, and it is the one
that turns the metric into a ruler rather than a self-assessment: point it at somebody else's
published checkpoint and it says whether their edit landed. It works even on models this tool
cannot edit, because reading a model is just running it.

THE TWO THINGS

**One: the contrast.** A direction is only about refusal if the two prompt sets differ in refusal
and not in their words. This project has withdrawn a figure for getting that wrong, a depth probe
reporting held-out AUC 0.99 at layer 1 against `advbench` and `xstest-safe`, which was reading
vocabulary. So the default is the matched pair and an unmatched one is announced.

**Two: the position.** The direction is taken where the mean difference is largest, and the first
version measured that difference in raw length. A transformer's residual norm grows through depth,
so the raw figures on SmolLM2-135M run 0.195 at position 1 to 153.2 at position 30 and the rule
meant "always the last position", on every model. Scaling by the residual length there makes the
positions comparable. It is still a magnitude rather than a separation and every run says so.
"""
import json
from typing import ClassVar

import pytest

from senbonzakura import leak, residualleak

# ── the contrast, checked before a model loads ───────────────────────────────────────

def test_the_default_pair_is_the_matched_one():
    """Not a convenience: the unmatched alternative produced a withdrawn figure."""
    assert leak.DEFAULT_HARMFUL == "xstest-unsafe"
    assert leak.DEFAULT_HARMLESS == "xstest-safe"
    assert leak.contrast_warning(leak.DEFAULT_HARMFUL, leak.DEFAULT_HARMLESS) is None


def test_the_known_confounded_pair_is_named_with_its_history():
    why = leak.contrast_warning("advbench", "xstest-safe")
    assert why is not None
    assert "withdrawn" in why
    assert "vocabulary" in why


def test_the_same_corpus_in_both_arms_is_refused_rather_than_warned():
    """A difference of means between a set and itself is zero by construction, so there is nothing
    to take a direction from and no run worth starting.
    """
    why = leak.contrast_warning("advbench", "advbench")
    assert why is not None
    assert "zero by construction" in why


def test_an_unlisted_pair_is_not_assumed_to_be_wrong():
    """The warning list names pairs we have evidence about. Silence is not approval, and the
    caveat printed on every run says the choice is a rule rather than a test.
    """
    assert leak.contrast_warning("xstest-unsafe", "harmless-alpaca") is None


# ── the position, which is where the first version was degenerate ────────────────────

class _FakeTensor:
    """Enough of a tensor for `contrast_direction`: indexing, subtraction and a norm."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, i):
        return _Row(self.rows[i])

    def __sub__(self, other):
        return _FakeTensor([[a - b for a, b in zip(x, y, strict=True)]
                            for x, y in zip(self.rows, other.rows, strict=True)])


class _Row:
    def __init__(self, values):
        self.values = list(values)

    def norm(self):
        return sum(v * v for v in self.values) ** 0.5


def _patch_means(monkeypatch, harmful_rows, harmless_rows):
    calls = []

    def _mean(model, tok, prompts, *, log=print):
        calls.append(list(prompts))
        return _FakeTensor(harmful_rows if len(calls) == 1 else harmless_rows)

    monkeypatch.setattr(residualleak, "mean_residual", _mean)
    monkeypatch.setattr(residualleak, "_unit", lambda v, _what: v)
    return calls


def test_the_position_is_chosen_on_the_scaled_difference_and_not_the_raw_one(monkeypatch):
    """THE DEGENERATE CASE THE FIRST VERSION SHIPPED.

    Position 0 separates completely: the two means point in opposite directions and the difference
    is the whole of the residual. Position 1 is a thousand times longer and the two means are
    almost identical, so its raw difference is bigger and its relative difference is tiny.

    A raw rule picks position 1. The right answer is 0.
    """
    # Position 0: means point opposite ways, raw difference 2 over a residual of 1, relative 2.0.
    # Position 1: means nearly identical, raw difference 10 over a residual of 995, relative 0.01.
    # So the raw rule prefers position 1 and the right answer is position 0.
    harmful = [[1.0, 0.0], [1000.0, 0.0]]
    harmless = [[-1.0, 0.0], [990.0, 0.0]]
    _patch_means(monkeypatch, harmful, harmless)
    _d, position, norms, _caveats = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], log=lambda _m: None)

    assert norms["raw"][1] > norms["raw"][0], "the fixture must have a longer raw difference late"
    assert position == 0, "chose the longest raw difference rather than the best separated position"
    assert norms["relative"][0] > norms["relative"][1]


def test_both_readings_are_returned_because_a_reader_expects_the_raw_one(monkeypatch):
    _patch_means(monkeypatch, [[2.0, 0.0], [4.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])
    _d, _p, norms, _c = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], log=lambda _m: None)
    assert set(norms) == {"raw", "relative"}
    assert len(norms["raw"]) == len(norms["relative"]) == 2


def test_a_named_position_overrides_the_rule_and_says_so(monkeypatch):
    _patch_means(monkeypatch, [[1.0, 0.0], [5.0, 0.0]], [[-1.0, 0.0], [4.0, 0.0]])
    _d, position, _n, caveats = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], at=1, log=lambda _m: None)
    assert position == 1
    assert any("named by the caller" in c for c in caveats)


def test_a_position_that_does_not_exist_is_refused(monkeypatch):
    _patch_means(monkeypatch, [[1.0, 0.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="does not exist"):
        residualleak.contrast_direction(object(), object(), ["a"], ["b"], at=9,
                                        log=lambda _m: None)


def test_a_negative_position_is_not_a_position():
    with pytest.raises(ValueError, match="not a position"):
        residualleak.contrast_direction(object(), object(), ["a"], ["b"], at=-1,
                                        log=lambda _m: None)


def test_every_run_carries_the_caveat_that_this_is_a_rule_and_not_a_measurement(monkeypatch):
    """The whole defence against the figure being over-read. A difference of means can be large
    where the two classes still overlap, and the rigorous instrument is a held-out AUC.
    """
    _patch_means(monkeypatch, [[3.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    _d, _p, _n, caveats = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], log=lambda _m: None)
    joined = " ".join(caveats)
    assert "not a measured separation" in joined
    assert "withdrawn" in joined


def test_flat_positions_are_called_close_to_arbitrary(monkeypatch):
    """When no position stands out, the choice is nearly a coin toss and the run says that rather
    than presenting a winner.
    """
    rows = [[1.0, 0.0]] * 5
    _patch_means(monkeypatch, rows, [[0.0, 0.0]] * 5)
    _d, _p, _n, caveats = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], log=lambda _m: None)
    assert any("close to arbitrary" in c for c in caveats)


def test_mismatched_shapes_cannot_come_from_one_model(monkeypatch):
    def _mean(model, tok, prompts, *, log=print):
        return _FakeTensor([[1.0, 0.0]] if prompts == ["a"] else [[1.0, 0.0], [2.0, 0.0]])

    monkeypatch.setattr(residualleak, "mean_residual", _mean)
    with pytest.raises(ValueError, match="not measured on the same one"):
        residualleak.contrast_direction(object(), object(), ["a"], ["b"], log=lambda _m: None)


def test_no_prompts_is_refused_by_the_mean_itself():
    with pytest.raises(ValueError, match="no mean residual"):
        residualleak.mean_residual(object(), object(), [], log=lambda _m: None)


# ── the command's own surface ────────────────────────────────────────────────────────

def test_the_command_is_registered_under_a_name_a_person_would_type():
    from senbonzakura.entry import DELEGATED
    assert DELEGATED["leak"] == ("leak", "main")


def test_the_parser_declares_what_the_help_describes():
    dests = {a.dest for a in leak.build_parser()._actions}
    for flag in ("harmful", "harmless", "n", "probe_n", "at", "out", "label", "model"):
        assert flag in dests, f"--{flag.replace('_', '-')} is described and not declared"


def test_the_defaults_do_not_need_a_flag_to_be_right():
    a = leak.build_parser().parse_args(["some/model"])
    assert a.harmful == leak.DEFAULT_HARMFUL
    assert a.harmless == leak.DEFAULT_HARMLESS
    assert a.at is None, "a position must not be guessed by default"


def test_identical_arms_stop_the_run_before_a_model_loads(tmp_path):
    """The refusal is reachable from the command line and happens before the weights."""
    with pytest.raises(SystemExit, match="zero by construction"):
        leak.main(["--model", "m", "--harmful", "advbench", "--harmless", "advbench",
                   "--out", str(tmp_path / "leak.json")])


def test_the_artefact_records_the_evidence_for_its_own_choice(tmp_path, monkeypatch):
    """Both norm readings and the caveats reach the file, so a reader can see whether the position
    was a peak or a coin toss rather than taking the number on trust.
    """
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(residualleak, "contrast_direction",
                        lambda *a, **k: ("dir", 7, {"raw": [1.0], "relative": [0.5]},
                                         ("a stated rule",)))

    class _Report:
        positions, probe_prompts = 2, 4
        leak_per_position = (1e-8, 2e-8)
        pinned: ClassVar[dict] = {
            "input_digest": "d" * 16, "partition": "measure", "prompt_format": "chat",
            "precision": "float32", "tool_version": "0.4.1"}
        output = None
        warnings = ()
        basis = "the pre-norm residual stream"
        mean = 1.5e-8

    monkeypatch.setattr(residualleak, "measure_leak", lambda *a, **k: _Report())
    monkeypatch.setattr(residualleak, "stamp_report", lambda doc, _r: doc)
    from senbonzakura import corpora
    monkeypatch.setattr(corpora, "load", lambda key, **k: [f"{key} prompt {i}" for i in range(4)])

    out = tmp_path / "leak.json"
    assert leak.main(["--model", "m", "--device", "cpu", "--n", "4", "--probe-n", "2",
                      "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["direction_position"] == 7
    assert doc["contrast_matched"] is True
    assert doc["difference_norms"] == {"raw": [1.0], "relative": [0.5]}
    assert doc["caveats"] == ["a stated rule"]
    assert doc["residual_leak"]["measured"] is True


# ── the decoy case, which is the one input that inverts the metric's meaning ─────────

def test_every_run_says_a_low_figure_can_mean_the_edit_landed_on_a_decoy(monkeypatch):
    """THE FALSE PASS, and it is a published defence rather than a hypothetical.

    Decoy Direction Optimization (arXiv 2609.16204) and AMRA (arXiv 2608.18093) both target the
    extractor instead of the refusal circuit: offer a contrastive estimator a high-magnitude feature
    orthogonal to refusal, and a run ablates the decoy, reports a low leak, and leaves refusal
    untouched. Nothing about variance, position or sample size touches that, because the metric
    never measured refusal. It measures whether the direction it extracted left.

    So the output has to name the other figure a reader needs, and it has to do it on every run
    rather than on a condition, because there is no condition detectable from inside this metric.
    """
    _patch_means(monkeypatch, [[3.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    _d, _p, _n, caveats = residualleak.contrast_direction(
        object(), object(), ["a"], ["b"], log=lambda _m: None)
    joined = " ".join(caveats)
    assert "decoy" in joined, "no run warns that a low figure is also a defended model's signature"
    assert "refusal rate" in joined, "the reader is not told which other figure to read beside it"
    assert "validate" in joined, (
        "the command that answers 'does this direction carry refusal' is not named")
