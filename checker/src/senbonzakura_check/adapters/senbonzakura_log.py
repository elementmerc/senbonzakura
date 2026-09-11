# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Senbonzakura's own artefacts, read through the same door as everybody else's.

WHY OURS NEEDS AN ADAPTER AT ALL

It would be easy to let our own files bypass normalisation, since the checks were written against
our vocabulary in the first place. That is the wrong shape, for one reason worth stating: a
checker that treats its author's output as a special case is one whose author never finds out
when the general path breaks. Ours goes through the same door, and the suite runs the whole check
set over the thirty-one published head-to-head arms on every commit.

It is also the LAST adapter tried. A foreign artefact that happens to carry a field we also use
must be read as what it is, and the two foreign detectors are far more specific than this one.

FOUR ARTEFACT SHAPES, NOT ONE, and the first version of this file knew about one of them.

    drift-*     `kl`, `instrument`, `kl_ci`, `logits_dtype`, `precision_ok`
    refusal-*   `refusal` AND `heretic`, `soft_refusal`, `noncompliant`, `n`, `eval`
    scored-*    `auc`, `auc_ci`, `controls.length_only_auc`, `mode`, `n_harmful`, `n_harmless`
    summary     the assembled head-to-head

TWO OF THEM CARRY TWO ESTIMATORS OF ONE METRIC, WHICH IS THE INTERESTING PART. A refusal artefact
holds our ruler's figure and Heretic's keyword figure side by side; a scored artefact holds the
real compass AUC and the length-only control that exists to expose an instrument measuring
sentence length. Those are not duplicates to be collapsed: they are the comparison. So they
normalise to separate entries keyed `metric.estimator`, and `measurement.stamp` refusing to put
two values under one metric name is the same rule seen from the other side.
"""
from __future__ import annotations

from senbonzakura_check.measurement import METRICS_KEY

#: A metric key we publish. Used for detection only, so a foreign file carrying one of these
#: names alone is not claimed: it has to look like ours in shape as well.
_OUR_METRICS = ("kl", "refusal", "auc", "capability")


class SenbonzakuraAdapter:
    name = "senbonzakura"

    @staticmethod
    def detects(doc) -> bool:
        """A stamped metrics block, or one of the older per-command shapes.

        Both are accepted because artefacts already published carry the older shape, and a
        checker that only understands what this version writes cannot read this project's own
        evidence from three weeks ago.
        """
        if isinstance(doc.get(METRICS_KEY), dict) and doc[METRICS_KEY]:
            return True
        has_metric = any(isinstance(doc.get(k), (int, float)) for k in _OUR_METRICS)
        if not has_metric:
            return False
        # `provenance` and `instrument` are ours; `label` plus `model` is the pairing every one
        # of our per-arm artefacts carries. Any of the three, alongside a numeric metric, is
        # specific enough without claiming a foreign file that merely has an `auc` in it.
        return bool(doc.get("instrument") or doc.get("provenance")
                    or (doc.get("label") and doc.get("model")))

    @staticmethod
    def normalise(doc) -> dict:
        metrics = {}

        for name, block in (doc.get(METRICS_KEY) or {}).items():
            if isinstance(block, dict):
                metrics[name] = {
                    "metric": block.get("metric", name),
                    "value": block.get("value"),
                    "estimator": block.get("estimator"),
                    "units": block.get("units"),
                    "n": block.get("n"),
                    "higher_is_better": block.get("higher_is_better"),
                    "interval": block.get("interval"),
                }

        if not metrics:
            metrics.update(_older_shapes(doc))

        template = doc.get("chat_template")
        return {
            "model": doc.get("model"),
            "tasks": [doc["label"]] if doc.get("label") else [],
            "metrics": metrics,
            "base": doc.get("base"),
            "eval_split": doc.get("eval"),
            # A dict here means a template WAS resolved and applied, and it records where from.
            # Lifted so the chat-template check can read it without knowing this format.
            "chat_template": (template or {}).get("source") if isinstance(template, dict)
            else template,
            "chat_template_applied": bool(template) or None,
            # Ours alone, and worth keeping: a KL below the floor of the arithmetic that produced
            # it is a number about bfloat16 rather than about the model.
            "computed_in": doc.get("logits_dtype"),
            "supported_by_precision": doc.get("precision_ok"),
            "provenance": doc.get("provenance"),
        }


def _older_shapes(doc) -> dict:
    """The per-command artefacts written before `measurement.stamp` existed."""
    out = {}

    if isinstance(doc.get("kl"), (int, float)):
        out["kl"] = {
            "metric": "kl",
            "value": doc["kl"],
            # Prose rather than a named estimator, which is weaker and is carried across as what
            # it is rather than parsed into a false precision.
            "estimator": doc.get("kl_source") or doc.get("instrument"),
            "units": "nats",
            "n": doc.get("n_prompts"),
            "higher_is_better": False,
            "interval": doc.get("kl_ci"),
        }

    n = doc.get("n")
    if isinstance(doc.get("refusal"), (int, float)):
        out["refusal_rate.senbonzakura-ruler"] = {
            "metric": "refusal_rate", "value": doc["refusal"],
            "estimator": "senbonzakura-ruler", "units": "proportion",
            "n": n, "higher_is_better": False, "interval": None,
        }
    if isinstance(doc.get("heretic"), (int, float)):
        # THE COMPARISON, not a duplicate. Heretic's keyword metric is kept byte-identical so
        # this one number is a like-for-like figure with theirs; everything else is ours.
        out["refusal_rate.heretic-keyword"] = {
            "metric": "refusal_rate", "value": doc["heretic"],
            "estimator": "heretic-keyword", "units": "proportion",
            "n": n, "higher_is_better": False, "interval": None,
        }

    if isinstance(doc.get("auc"), (int, float)):
        n_total = None
        if isinstance(doc.get("n_harmful"), int) and isinstance(doc.get("n_harmless"), int):
            n_total = doc["n_harmful"] + doc["n_harmless"]
        out["compass_auc.margin-past-preamble"] = {
            "metric": "compass_auc", "value": doc["auc"],
            "estimator": "margin-past-preamble", "units": "auc",
            "n": n_total, "higher_is_better": True, "interval": doc.get("auc_ci"),
        }
        control = (doc.get("controls") or {}).get("length_only_auc")
        if isinstance(control, (int, float)):
            # THE CONTROL THAT EXPOSES THE INSTRUMENT. If a ruler reading nothing but prompt
            # length separates the arms as well as the real one does, the real one is measuring
            # sentence length. Carried across so a reader of a finding sees both numbers.
            out["compass_auc.length-only-control"] = {
                "metric": "compass_auc", "value": control,
                "estimator": "length-only-control", "units": "auc",
                "n": n_total, "higher_is_better": True, "interval": None,
            }

    return out
