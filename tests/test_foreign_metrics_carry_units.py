# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The two checks that catch an impossible number have to be able to fire on a foreign artefact.

WHAT HAPPENED, 2026-09-27

Two checks are gated on `units == "proportion"`: `impossible-proportion-reported`, which fires on a
rate outside 0 to 1, and `rate-reported-on-a-sample-too-small-to-carry-it`, which fires on a rate
resting on fewer than 30 rows. They are the two most likely to catch a number that is simply wrong.

Both third-party adapters wrote `"units": None` for every metric, so **both checks were
structurally unreachable for every foreign artefact.** A reader fed the checker an
lm-evaluation-harness file reporting an accuracy of 1.4 and an `acc_norm` of -0.2 over a sample of
3, and got `0 finding(s)` and exit 0.

Three things make this worth a test file rather than a commit message.

`REPRODUCING.md` invites people to point this tool at somebody else's numbers, and for those people
the answer was a clean bill of health produced by two checks that never ran.

Our own adapter set `units` correctly throughout, so nothing we tested it on could reveal the gap.
Every artefact in this repository has the field.

And `impossible-proportion-reported` said so itself. Its own "when this check is wrong" note reads:
"a harness that does not record units escapes it entirely, which is a gap in coverage rather than a
reason to widen the rule." The gap was written down, by us, and left open. A documented gap is still
a gap, and prose cannot fail.

THE READER ONLY TESTED LM-EVAL. The Inspect adapter had the identical dead line, found by going and
looking for the other spelling rather than waiting for somebody to meet it.
"""
import json

import pytest
from senbonzakura_check.adapters import normalise
from senbonzakura_check.adapters._units import (
    NOT_PROPORTIONS,
    PROPORTION_METRICS,
    units_for,
)
from senbonzakura_check.registry import load_checks, run_checks


def _lm_eval(metrics, *, n=3):
    """A minimally valid lm-evaluation-harness document.

    `results`, `configs` and `versions` together are what the adapter detects on. Worth knowing
    because the reader's first attempt carried `results` and `config` and was rejected as
    unrecognised, with nothing saying which keys were wanted.
    """
    return {
        "results": {"gsm8k": {"alias": "gsm8k", **metrics}},
        "configs": {"gsm8k": {"task": "gsm8k"}},
        "versions": {"gsm8k": 3},
        "config": {"model": "hf", "model_args": "pretrained=x"},
        "n-samples": {"gsm8k": {"effective": n, "original": n}},
        "higher_is_better": {"gsm8k": {"acc": True}},
        "git_hash": "abc1234", "date": "2026-09-27",
    }


def _inspect(metric_name, value, *, n=3):
    # `version` as an integer plus an `eval` block naming a task is what the adapter detects on.
    return {
        "version": 2,
        "eval": {"task": "arc", "model": "openai/gpt-x", "model_args": {}},
        "results": {
            "scores": [{
                "name": "choice",
                "scorer": "choice",
                "scored_samples": n,
                "metrics": {metric_name: {"name": metric_name, "value": value}},
            }],
            "total_samples": n,
            "completed_samples": n,
        },
    }


def _findings(doc):
    """Check ids that FIRED. `run_checks` returns (findings, skipped_ids) and skipped is not a pass."""
    findings, _skipped = run_checks(normalise(doc), load_checks())
    return [finding.check_id for finding in findings]


class TestTheUnitsInference:
    @pytest.mark.parametrize("name", sorted(PROPORTION_METRICS))
    def test_a_known_rate_is_a_proportion(self, name):
        assert units_for(name) == "proportion"

    @pytest.mark.parametrize("name", sorted(NOT_PROPORTIONS))
    def test_a_metric_that_is_not_a_rate_stays_silent(self, name):
        """Silence is the safe answer: calling perplexity a proportion would accuse it falsely."""
        assert units_for(name) is None

    @pytest.mark.parametrize("name", ["", "something_nobody_has_heard_of", None, 7, ["acc"]])
    def test_anything_unrecognised_stays_silent(self, name):
        assert units_for(name) is None

    def test_the_two_lists_do_not_overlap(self):
        """An overlap would make the answer depend on which membership test ran first."""
        assert not (PROPORTION_METRICS & NOT_PROPORTIONS)


class TestTheReadersCaseNowFires:
    """The exact artefact from 2026-09-27, which returned zero findings and exit 0."""

    def test_an_impossible_accuracy_is_caught(self):
        doc = _lm_eval({"acc,none": 1.4, "acc_norm,none": -0.2})
        assert "impossible-proportion-reported" in _findings(doc)

    def test_a_rate_on_three_rows_is_caught(self):
        doc = _lm_eval({"acc,none": 0.5}, n=3)
        assert "rate-reported-on-a-sample-too-small-to-carry-it" in _findings(doc)

    def test_a_sane_lm_eval_file_is_not_accused(self):
        """The fix must not have bought coverage by flagging correct work."""
        doc = _lm_eval({"acc,none": 0.64}, n=200)
        found = _findings(doc)
        assert "impossible-proportion-reported" not in found
        assert "rate-reported-on-a-sample-too-small-to-carry-it" not in found

    def test_a_perplexity_out_of_that_range_is_not_accused(self):
        """Perplexity is unbounded above; 20.4 is an ordinary value, not an impossible rate."""
        doc = _lm_eval({"word_perplexity,none": 20.4}, n=200)
        assert "impossible-proportion-reported" not in _findings(doc)


class TestTheOtherSpelling:
    """Inspect carried the identical dead line. Found by looking, not by being told."""

    def test_an_impossible_inspect_accuracy_is_caught(self):
        assert "impossible-proportion-reported" in _findings(_inspect("accuracy", 1.4))

    def test_a_small_inspect_sample_is_caught(self):
        assert "rate-reported-on-a-sample-too-small-to-carry-it" in _findings(
            _inspect("accuracy", 0.5, n=3))

    def test_a_sane_inspect_file_is_not_accused(self):
        found = _findings(_inspect("accuracy", 0.77, n=400))
        assert "impossible-proportion-reported" not in found
        assert "rate-reported-on-a-sample-too-small-to-carry-it" not in found


def test_neither_foreign_adapter_writes_a_blanket_none(tmp_path):
    """The regression in the form it will come back in: a new adapter with the same dead line.

    Asserting on the normalised output rather than on the source, so it holds for whatever an
    adapter is rewritten to look like.
    """
    for doc, label in ((_lm_eval({"acc,none": 0.5}, n=200), "lm-eval"),
                       (_inspect("accuracy", 0.5, n=200), "inspect")):
        metrics = normalise(doc)["metrics"]
        assert metrics, f"{label} produced no metrics at all"
        units = {entry.get("units") for entry in metrics.values()}
        assert units != {None}, (
            f"every {label} metric carries units None, so the two value checks cannot fire on this "
            f"format. That was the defect: {json.dumps(metrics)[:300]}")
