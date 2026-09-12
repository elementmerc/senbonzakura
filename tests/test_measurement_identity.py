# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every number carries its name, its estimator and its units, and the registry refuses the rest.

Item D of the non-GPU shortlist. The incident is 2026-08-05: four published claims withdrawn
because arms doing different amounts of work were compared as one measurement, and no field in
any artefact said otherwise. Nothing was mislabelled. There were simply no labels.

So the tests here are mostly about REFUSALS, because the value of this module is what it will not
let a writer do. A vocabulary that accepts anything is a free-text field with extra steps.
"""
import json

import pytest
from senbonzakura_check import measurement
from senbonzakura_check.measurement import (
    METRICS,
    METRICS_KEY,
    MeasurementError,
    estimator_description,
    identity,
    instrument_sentence,
    stamp,
)

# ── the registry itself ──────────────────────────────────────────────────────────────────────

def test_there_are_metrics_declared():
    assert METRICS, "an empty registry would make every refusal below vacuous"


@pytest.mark.parametrize("name", sorted(METRICS))
def test_every_metric_declares_units_and_at_least_one_estimator(name):
    """A metric with no declared estimator cannot be stamped at all, so it is dead weight that
    reads as coverage of a quantity nothing can emit.
    """
    m = METRICS[name]
    assert m.units, f"{name} has no units"
    assert m.measures and len(m.measures) > 20, f"{name} does not say what it measures"
    assert m.estimators, f"{name} declares no estimator"


@pytest.mark.parametrize("name", sorted(METRICS))
def test_every_estimator_describes_itself_distinguishably(name):
    """A reader holding only the artefact has to be able to tell two estimators apart.

    That is the whole failure: `kl` from a full first-token distribution and `kl` from a
    continuation negative log-likelihood are different numbers with the same name and a similar
    range, so the descriptions have to differ by more than a word.
    """
    descriptions = list(METRICS[name].estimators.values())
    for d in descriptions:
        assert len(d) > 30, f"{name}: an estimator description too short to distinguish anything"
    assert len(set(descriptions)) == len(descriptions), (
        f"{name}: two estimators share a description, so the artefact cannot tell them apart")


def test_the_kl_registry_keeps_the_dangerous_pair_and_names_the_danger():
    """The one that cost four claims is declared, deliberately, with the warning attached.

    Removing `continuation-nll-difference` would be the wrong fix: other tools legitimately
    report it, the checker has to be able to describe it, and pretending it does not exist is how
    it gets reported as a KL by somebody who has never heard the argument.
    """
    kl = METRICS["kl"]
    assert set(kl.estimators) >= {"first-token-full-distribution", "continuation-nll-difference"}
    assert "NOT a KL" in kl.estimators["continuation-nll-difference"]


# ── the refusals ─────────────────────────────────────────────────────────────────────────────

def test_an_undeclared_metric_is_refused_and_lists_what_is_declared():
    with pytest.raises(MeasurementError, match="not a metric this project declares") as e:
        estimator_description("vibes", "eyeballed")
    assert "kl" in str(e.value), "the refusal has to say what the alternatives are"


def test_an_undeclared_estimator_under_a_known_metric_is_refused():
    """THE DANGEROUS ONE. The name looks right and the procedure behind it is undeclared."""
    with pytest.raises(MeasurementError, match="no declared estimator") as e:
        estimator_description("kl", "however the other tool did it")
    assert "2026-08-05" in str(e.value), "the refusal cites the incident, so it can be judged"
    assert "first-token-full-distribution" in str(e.value)


def test_stamping_the_same_metric_twice_is_refused():
    """Two values for one metric in one document is exactly the ambiguity being prevented."""
    doc = {}
    stamp(doc, "kl", 0.1, "first-token-full-distribution")
    with pytest.raises(MeasurementError, match="already stamped"):
        stamp(doc, "kl", 0.2, "continuation-nll-difference")


# ── what a stamp produces ────────────────────────────────────────────────────────────────────

def test_a_stamp_carries_name_estimator_units_and_the_value():
    doc = stamp({}, "kl", 0.0313, "first-token-full-distribution", n=200)
    block = doc[METRICS_KEY]["kl"]
    assert block["metric"] == "kl"
    assert block["estimator"] == "first-token-full-distribution"
    assert block["units"] == "nats"
    assert block["value"] == 0.0313
    assert block["n"] == 200
    assert "log-probabilities" in block["estimator_description"]


def test_a_stamp_says_which_direction_is_better():
    """A reader comparing two arms needs it, and half of these metrics go each way.

    `compass_auc` higher is better, `kl` lower is better, and a report that got that backwards is
    how this project once described a result in its own favour.
    """
    assert identity("kl", "first-token-full-distribution")["higher_is_better"] is False
    assert identity("compass_auc", "margin-past-preamble")["higher_is_better"] is True


def test_units_may_be_overridden_for_a_legitimate_alternative():
    """A refusal rate reported as a percentage is not wrong, and the alternative to allowing it
    is a writer silently disagreeing with this registry while nothing notices.
    """
    block = identity("refusal_rate", "senbonzakura-ruler", units="percent")
    assert block["units"] == "percent"


def test_extra_fields_are_carried_through():
    block = identity("kl", "first-token-full-distribution", interval=[0.01, 0.05], computed_in="bf16")
    assert block["interval"] == [0.01, 0.05]
    assert block["computed_in"] == "bf16"


def test_a_stamp_is_additive_and_leaves_existing_fields_alone():
    """The published arms are committed and a test recomputes their figures from them, so a
    writer that renamed a field would rewrite a published measurement rather than annotate it.
    """
    doc = {"kl": 0.0313, "label": "arm-1", "instrument": "an older sentence"}
    before = dict(doc)
    stamp(doc, "kl", 0.0313, "first-token-full-distribution")
    for k, v in before.items():
        assert doc[k] == v, f"{k} was modified by stamping"


def test_a_stamped_document_is_json_serialisable():
    """It is going straight into an artefact, so anything unserialisable fails at write time on
    somebody's long run rather than here.
    """
    doc = stamp({}, "capability", 0.62, "code-graded", n=5)
    assert json.loads(json.dumps(doc))[METRICS_KEY]["capability"]["value"] == 0.62


# ── the sentence older artefacts carry ───────────────────────────────────────────────────────

def test_the_instrument_sentence_is_generated_from_the_registry():
    """`drift` wrote this sentence by hand and it was right. Generating it means the prose a
    reader sees and the structured block beside it cannot come to say different things.
    """
    got = instrument_sentence("kl", "first-token-full-distribution")
    assert got.startswith("senbonzakura.kl")
    assert "first-token" in got
    with pytest.raises(MeasurementError):
        instrument_sentence("kl", "not-declared")


# ── the module stays importable with nothing installed ───────────────────────────────────────

def test_measurement_imports_nothing(monkeypatch):
    """The checker and the abliterator both need this vocabulary, and only one of them has torch.

    A shared vocabulary that only the heavy side can import is not shared, so this asserts the
    import graph rather than trusting the docstring.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(measurement.__file__).read_text(encoding="utf-8"))
    imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = []
    for node in imported:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        else:
            names.append(node.module or "")
    assert set(names) <= {"__future__"}, f"measurement.py imports {names}"


# ── the drift writer uses it ─────────────────────────────────────────────────────────────────

def test_drift_declares_its_estimator_as_a_named_constant():
    """`kl` accepts a second estimator that is NOT a KL divergence, so which one `drift` reports
    has to be a named decision rather than a literal at a call site.
    """
    from senbonzakura import drift

    assert drift.KL_ESTIMATOR in METRICS["kl"].estimators
    assert drift.KL_ESTIMATOR == "first-token-full-distribution", (
        "drift always has the true log-probabilities, so it always reports the real KL")


def test_drifts_instrument_sentence_comes_from_the_registry():
    """If the registry's wording changes, the sentence in the artefact changes with it, which is
    the property that stops the two drifting apart.
    """
    from senbonzakura import drift

    assert instrument_sentence("kl", drift.KL_ESTIMATOR).startswith("senbonzakura.kl")


# ── two estimators of one metric ─────────────────────────────────────────────────────────────

def test_two_estimators_of_one_metric_are_keyed_by_estimator():
    """A refusal artefact carries this project's ruler AND Heretic's keyword metric, and a
    compass artefact carries the real AUC and the length-only control. Those pairs are the
    comparison, not duplicates, and the senbonzakura adapter has always normalised the older
    artefacts into `metric.estimator` keys. Stamping writes the same shape.
    """
    doc = {}
    measurement.stamp(doc, "refusal_rate", 0.12, "senbonzakura-ruler", n=200, by_estimator=True)
    measurement.stamp(doc, "refusal_rate", 0.09, "heretic-keyword", n=200, by_estimator=True)
    assert sorted(doc[measurement.METRICS_KEY]) == [
        "refusal_rate.heretic-keyword", "refusal_rate.senbonzakura-ruler"]
    assert doc[measurement.METRICS_KEY]["refusal_rate.heretic-keyword"]["metric"] == "refusal_rate"


def test_the_same_estimator_twice_is_still_refused():
    """What is relaxed is two INSTRUMENTS, not two values from one."""
    doc = {}
    measurement.stamp(doc, "refusal_rate", 0.12, "senbonzakura-ruler", by_estimator=True)
    with pytest.raises(measurement.MeasurementError):
        measurement.stamp(doc, "refusal_rate", 0.13, "senbonzakura-ruler", by_estimator=True)


def test_mixing_a_bare_stamp_with_a_per_estimator_one_is_refused():
    """`refusal_rate` beside `refusal_rate.heretic-keyword` leaves a reader unable to say which
    instrument the bare one came from, which is the ambiguity the module exists to prevent.
    """
    doc = {}
    measurement.stamp(doc, "refusal_rate", 0.12, "senbonzakura-ruler")
    with pytest.raises(measurement.MeasurementError) as e:
        measurement.stamp(doc, "refusal_rate", 0.09, "heretic-keyword", by_estimator=True)
    assert "already stamped bare" in str(e.value)

    other = {}
    measurement.stamp(other, "refusal_rate", 0.09, "heretic-keyword", by_estimator=True)
    with pytest.raises(measurement.MeasurementError) as e:
        measurement.stamp(other, "refusal_rate", 0.12, "senbonzakura-ruler")
    assert "already stamped per estimator" in str(e.value)
