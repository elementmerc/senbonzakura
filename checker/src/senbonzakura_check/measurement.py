# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every number this project emits says what it is, how it was computed, and in what units.

THE INCIDENT THIS EXISTS BECAUSE OF

On 2026-08-05 four published claims were withdrawn. Arms that had done different amounts of work
were compared as though they were one measurement, and **no field in any artefact said
otherwise**. Nothing was mislabelled; there were simply no labels. A reader, including a later
version of the person who ran it, had no way to tell two `kl` fields apart.

The practice was read out of abliterix, which carries name, estimator and units on each
measurement and refuses to call a continuation negative log-likelihood a KL divergence when the
true log-probabilities were not available. That refusal is the whole idea: a metric is not its
name, it is its name plus the procedure that produced it, and two procedures under one name are
two different numbers.

WHY A REGISTRY RATHER THAN A FREE-TEXT FIELD

`drift.py` already stamped `instrument` with a hand-written sentence, and it was correct. The
problem with a hand-written sentence is that the next writer invents a different one: today
`drift` says `instrument`, `headtohead_report` says `kl_source`, and `score`, `capability` and
`margin` say nothing at all. A reader cannot look in one place, and neither can the checker in
`senbonzakura/check/`, which is the tool whose entire job is noticing when provenance is absent.

So the vocabulary is declared once, here, and a metric cannot be emitted under a name this module
does not know, with an estimator that name does not accept. That is deliberately strict: the
failure this prevents is a number acquiring a plausible label rather than a true one.

TORCH-FREE ON PURPOSE

This module imports nothing. The writers that use it are heavy; the checker that reads its output
is not, and per decision Q-29 the checker ships as its own distribution. A shared vocabulary that
only one side can import is not shared.
"""
from __future__ import annotations

#: The canonical field a stamped artefact carries. Named rather than spelled out at each call
#: site, because the point of this module is that there is one place to look.
METRICS_KEY = "metrics"


class MeasurementError(Exception):
    """A number was about to be emitted under a name or an estimator that is not declared.

    Raised rather than warned. A warning at the point of writing an artefact is read by nobody:
    the artefact outlives the terminal it was printed in, and the whole purpose here is that the
    artefact carries the truth on its own.
    """


class _Metric:
    __slots__ = ("estimators", "higher_is_better", "measures", "name", "units")

    def __init__(self, name, measures, units, estimators, higher_is_better):
        self.name = name
        self.measures = measures
        self.units = units
        self.estimators = dict(estimators)
        self.higher_is_better = higher_is_better


def _m(name, measures, units, estimators, higher_is_better=False):
    return _Metric(name, measures, units, estimators, higher_is_better)


#: Every metric this project publishes, and every estimator allowed to produce it.
#:
#: ADDING AN ESTIMATOR IS A DECISION, not a convenience. Two estimators under one metric name is
#: exactly the situation that cost four claims, so each one here has to be distinguishable from
#: its siblings by a reader who has only the artefact.
METRICS = {
    m.name: m
    for m in (
        _m("kl", "how far the output distribution moved from the base model",
           "nats",
           {
               "first-token-full-distribution":
                   "KL(base||candidate) over the full first-token distribution, computed from "
                   "log-probabilities on both sides",
               "continuation-nll-difference":
                   "a difference of continuation negative log-likelihoods, which is NOT a KL "
                   "divergence and must never be reported as one when the true "
                   "log-probabilities were available",
           }),
        _m("refusal_rate", "how often the model refuses, on rows nothing was fitted on",
           "proportion",
           {
               "senbonzakura-ruler": "this project's refusal ruler on the re-score slice",
               "heretic-keyword": "Heretic's keyword metric, copied verbatim and kept "
                                  "byte-identical so the number is comparable with theirs",
           }),
        _m("compass_auc", "whether the model still recognises harm, as opposed to refusing it",
           "auc",
           {
               "margin-past-preamble":
                   "area under the ROC of the harmful/harmless margin, read past the preamble",
               "length-only-control":
                   "the same statistic computed from prompt length alone, which is the control "
                   "that exposes an instrument measuring sentence length",
           },
           higher_is_better=True),
        _m("capability", "what the edit cost on tasks the model either gets right or does not",
           "proportion",
           {"code-graded": "graded by code, with no model in the loop"},
           higher_is_better=True),
        _m("separation", "whether a direction set carries refusal or carries topic",
           "statistic",
           {
               "variance-ratio": "the variance-ratio separation statistic",
               "difference-of-means": "a difference of class means along the candidate axis",
           },
           higher_is_better=True),
    )
}


def estimator_description(metric: str, estimator: str) -> str:
    """The declared account of how this number was produced, or a refusal.

    Both halves are validated. An unknown METRIC means somebody is publishing a quantity this
    project has never described; an unknown ESTIMATOR under a known metric is the dangerous one,
    because the name looks right and the procedure behind it is undeclared.
    """
    known = METRICS.get(metric)
    if known is None:
        raise MeasurementError(
            f"{metric!r} is not a metric this project declares. Add it to METRICS with its "
            f"units and the estimators allowed to produce it, or use one of: "
            f"{', '.join(sorted(METRICS))}.")
    described = known.estimators.get(estimator)
    if described is None:
        raise MeasurementError(
            f"{metric!r} has no declared estimator {estimator!r}. Two procedures under one "
            f"metric name is how four published claims came to be withdrawn on 2026-08-05. "
            f"Declared for {metric!r}: {', '.join(sorted(known.estimators))}.")
    return described


def identity(metric: str, estimator: str, *, n=None, units=None, **extra) -> dict:
    """The identity block for one measurement.

    `units` may be overridden for a metric legitimately reported in another unit (a refusal rate
    as a percentage rather than a proportion), because the alternative is the writer silently
    disagreeing with this registry and nothing noticing.
    """
    # Validated FIRST, so the lookup below cannot fail: `estimator_description` raises on an
    # unknown metric, which makes the presence of `metric` in METRICS a guarantee by the next
    # line rather than something to re-test.
    described = estimator_description(metric, estimator)
    known = METRICS[metric]
    out = {
        "metric": metric,
        "measures": known.measures,
        "estimator": estimator,
        "estimator_description": described,
        "units": units if units is not None else known.units,
        "higher_is_better": known.higher_is_better,
    }
    if n is not None:
        out["n"] = int(n)
    out.update(extra)
    return out


def stamp(doc: dict, metric: str, value, estimator: str, *, n=None, units=None,
          by_estimator: bool = False, **extra) -> dict:
    """Record `value` in `doc` under the canonical metrics block, with its identity.

    Additive by design: the caller's existing top-level fields are untouched, so an artefact
    gains provenance without changing shape for anything already reading it. The published
    head-to-head arms are committed to this repository and are recomputed by a test, so a writer
    that renamed a field would break a published number rather than annotate it.

    TWO ESTIMATORS OF ONE METRIC IS A REAL CASE, NOT AN ERROR, and `by_estimator=True` is how it
    is written. A refusal artefact carries this project's ruler and Heretic's keyword metric side
    by side; a compass artefact carries the real AUC and the length-only control that exists to
    expose an instrument measuring sentence length. Those are the comparison, not duplicates to
    be collapsed. They are keyed `metric.estimator`, which is exactly what the senbonzakura
    adapter already normalises older artefacts into, so the stamped and unstamped shapes read
    identically.

    What stays refused is two values under ONE key, which is the ambiguity that made two
    different `kl` fields indistinguishable in the first place.
    """
    block = doc.setdefault(METRICS_KEY, {})
    slot = f"{metric}.{estimator}" if by_estimator else metric
    if slot in block:
        raise MeasurementError(
            f"{slot!r} is already stamped on this artefact. Two values for one metric in one "
            f"document is the ambiguity this module exists to prevent; pass "
            f"`by_estimator=True` if these are genuinely two instruments measuring the same "
            f"quantity, or give the second one its own artefact.")
    if by_estimator and metric in block:
        raise MeasurementError(
            f"{metric!r} is already stamped bare on this artefact, so adding "
            f"{slot!r} beside it would leave a reader unable to tell which is which. Stamp "
            f"every estimator of a metric with `by_estimator=True`, or none of them.")
    if not by_estimator and any(k.startswith(f"{metric}.") for k in block):
        raise MeasurementError(
            f"{metric!r} is already stamped per estimator on this artefact, so a bare "
            f"{metric!r} beside it would be a third number of unclear provenance. Stamp this "
            f"one with `by_estimator=True` too.")
    block[slot] = {**identity(metric, estimator, n=n, units=units, **extra), "value": value}
    return doc


def instrument_sentence(metric: str, estimator: str) -> str:
    """The one-line description older artefacts carry in their `instrument` field.

    `drift.py` has written such a sentence by hand since it was added, and it was right. It is
    generated from the registry now so the sentence and the structured block cannot drift apart,
    which is the failure that made two `kl` fields indistinguishable in the first place.
    """
    return f"senbonzakura.{metric}, {estimator_description(metric, estimator)}"
