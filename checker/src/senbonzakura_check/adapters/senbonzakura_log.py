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
    abliteration.json  `num_directions`, `dir_mode`, `post_bake_refusals`, `refusal_eval`,
                `directions_per_layer`, `max_directions`, and the settings block, which is
                `generation_settings` since 2026-09-21 and `generation` in every record
                written before it

FIVE, AND THE FIFTH WAS MISSING UNTIL 2026-09-21, which is the uncomfortable one. `abliteration.json`
is this project's PRIMARY artefact: it is what the model card, the head-to-head report, every
resume guard and this checker are supposed to read, and it names none of its figures `refusal`,
`kl` or `auc`. They are `post_bake_refusals`, `post_bake_kl` and `post_bake_heretic`, so the
detector above answered false and `senbonzakura check abliteration.json` reported it UNCHECKED.
Not clean, which is the one thing worth saying for the design: the refusal was loud and correct,
and it was still the case that the tool could not read the file its own runs write. Found by
building the seeded incident corpus and pointing the checker at it, which is the whole reason
that corpus exists: a check can pass every one of its own controls and never fire in the field
because the adapter never carried the field it reads.

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
#: The top-level numeric fields that mark a document as one of ours, for the artefacts written
#: before `measurement.stamp` existed.
#:
#: `nll` was added on 2026-09-21 and is the weakest of the five, which is why it is worth a note:
#: `kl`, `auc` and `capability` are specific enough to be near-conclusive on their own, while a
#: field called `nll` appears in plenty of other people's files. It is safe here only because
#: `detects` requires the `instrument`, `provenance` or `label`-plus-`model` pairing alongside it.
#: A coherence artefact carries `label` and `model`; a stray perplexity dump from another tool
#: almost never carries both, and if one did, the checks it would then meet are the ones that
#: report SKIPPED rather than passed when a record cannot answer them.
_OUR_METRICS = ("kl", "refusal", "auc", "capability", "nll")


#: The two names one block has worn. `build_abliteration_record` wrote these settings under
#: `generation` until 2026-09-21 and under `generation_settings` after it, because this
#: repository's pre-commit leak gate refuses any committed JSON carrying a key named `generation`
#: at any depth, and was therefore refusing this project's own primary artefact. The gate was not
#: weakened; the field moved. Newest first.
_GENERATION_BLOCKS = ("generation_settings", "generation")


def _generation_field(doc, field):
    """One field from whichever spelling of the settings block actually carries it.

    THIS EXISTS BECAUSE THERE WERE TWO COPIES OF IT. The adapter resolved the block once at the
    top and `_arm_settings` walked the same two names again, and the second copy was untested:
    narrowing it to one name left all 91 adapter tests passing. It is load-bearing, because
    without the fallback an arm recorded before the rename compared against one recorded after it
    silently drops the budget from the comparison, and two arms differing in budget then look
    like two arms that match.

    The two copies had also drifted. The old top-level resolution took the first block that was a
    dict and read the field from it; this takes the first block that CARRIES the field. A record
    holding an empty `generation_settings` beside a populated `generation` reported a budget in
    the settings and None at the top level, which is one record giving two answers.
    """
    for name in _GENERATION_BLOCKS:
        block = doc.get(name)
        if isinstance(block, dict) and block.get(field) is not None:
            return block[field]
    return None


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
        if _is_abliteration_record(doc) or _is_conversion_record(doc):
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
        if _is_conversion_record(doc):
            return _conversion(doc)
        metrics = {}

        for name, block in (doc.get(METRICS_KEY) or {}).items():
            if isinstance(block, dict):
                normalised = {
                    "metric": block.get("metric", name),
                    "value": block.get("value"),
                    "estimator": block.get("estimator"),
                    "units": block.get("units"),
                    "n": block.get("n"),
                    "higher_is_better": block.get("higher_is_better"),
                    "interval": block.get("interval"),
                    # WHICH BUILD PRODUCED THIS NUMBER, carried across rather than dropped.
                    # `measurement.stamp` writes it as an extra, and until a pair check needed it
                    # this adapter silently discarded it: the seven fields above were a fixed
                    # list. A figure whose estimator changed between two versions is not
                    # comparable with the same figure from the other version, and without this
                    # field nothing downstream can tell those two cases apart.
                    "tool_version": block.get("tool_version"),
                }
                # EVERYTHING ELSE THE STAMP WROTE, carried rather than dropped. The fixed list
                # above was the whole contract until 2026-09-21, which made `input_digest`,
                # `prompt_format`, `partition` and `n_tokens` write-only: the writers stamped
                # them, this adapter discarded them, and no check could ever read one. Two
                # commits' worth of identity work was decorative. The list is not extended field
                # by field, because the next field added to a stamp would be dropped in exactly
                # the same silence; anything a writer chose to record is carried, and the
                # canonical names above win so an extra can never redefine the number itself.
                for key, value in block.items():
                    if key not in normalised:
                        normalised[key] = value
                metrics[name] = normalised

        if not metrics:
            metrics.update(_older_shapes(doc))

        template = doc.get("chat_template")
        # The abliteration record names the same question differently at every turn: the rows are
        # `refusal_eval` rather than `eval`, and its own warning about its own generation budget
        # is nested one level down rather than sitting at the top. Lifted here, into the names the
        # checks already read, rather than teaching five checks a second vocabulary.
        # TWO NAMES FOR ONE BLOCK, NEWEST FIRST, and the fallback is load-bearing rather than
        # polite. `build_abliteration_record` wrote these settings under `generation` until
        # 2026-09-21 and writes `generation_settings` after it, so every record already on disk
        # carries the old name. This adapter's whole job is reading artefacts somebody else
        # produced, and most of those were produced before today: an adapter that read only the
        # new name would stop finding the budget on every existing file, which would leave
        # `quoted-at-a-budget-below-the-visibility-floor` unable to fire on anything real while
        # passing all of its own controls.
        return {
            "model": doc.get("model"),
            "tasks": [doc["label"]] if doc.get("label") else [],
            "metrics": metrics,
            "base": doc.get("base"),
            "eval_split": doc.get("eval") or doc.get("refusal_eval"),
            # A dict here means a template WAS resolved and applied, and it records where from.
            # Lifted so the chat-template check can read it without knowing this format.
            "chat_template": (template or {}).get("source") if isinstance(template, dict)
            else template,
            "chat_template_applied": bool(template) or None,
            # HOW THE PROMPTS WERE ACTUALLY RENDERED, lifted to where the check looks for it.
            # `chat-template-never-applied` reads `prompt_format` at the top level of the
            # normalised record, and every writer here stamps it INSIDE a metrics block. So the
            # one clause that catches "a template was named and the prompts went raw anyway"
            # could not see the field that says so, on any artefact this project produces. Lifted
            # here rather than written twice by every writer, because the adapter is the layer
            # whose job is putting a producer's fields into the canonical vocabulary, and a
            # second copy in the artefact is the duplication that drifts.
            "prompt_format": doc.get("prompt_format") or _agreed(metrics, "prompt_format"),
            # Ours alone, and worth keeping: a KL below the floor of the arithmetic that produced
            # it is a number about bfloat16 rather than about the model.
            "computed_in": doc.get("logits_dtype"),
            "supported_by_precision": doc.get("precision_ok"),
            # THE ARTEFACT'S OWN WARNING ABOUT ITSELF. The scorer writes this when the generation
            # budget was below the visibility floor, which is the condition under which a refusal
            # rate describes the budget rather than the model. It was written into the file and
            # read by nothing: a hostile reviewer on 2026-09-17 fed this checker an artefact
            # carrying its own short-budget warning and got "nothing found". A checker that
            # ignores the one sentence the producer left about why its number might be wrong is
            # not reading the artefact, it is reading past it.
            "budget_warning": doc.get("budget_warning") or _generation_field(doc, "budget_warning"),
            # THE BUDGET ITSELF, beside the warning about it, and lifted to a flat name for two
            # reasons. A check reads the normalised vocabulary rather than one producer's nesting,
            # which is the whole point of an adapter. And the record nests it: under
            # `generation_settings` since 2026-09-21, and under `generation` before that, because
            # this repository's pre-commit leak gate refuses any committed JSON carrying a key
            # named `generation` at any depth and was therefore refusing this project's own
            # primary artefact. The gate was not weakened; the field moved. The fallback above is
            # what keeps every record written before the move readable.
            "generation_budget": _generation_field(doc, "max_new_tokens"),
            "provenance": doc.get("provenance"),
            "settings": _arm_settings(doc),
        }


#: The fields of an abliteration record that DEFINE AN ARM, as (the record's name, the canonical
#: name a check reads). Everything else the record carries is a result, a profile or a provenance
#: stamp, and a difference in one of those is not a difference in the experiment.
#:
#: `seed` IS DELIBERATELY ABSENT, and it is the one worth explaining. Two arms of a five-seed
#: comparison differ in their seed by construction, and that is the comparison working rather than
#: a confound: the whole reason to run five is that one is a sample. Counting the seed as a
#: differing setting would make `arms-that-differ-in-more-than-the-named-variable` fire on every
#: correctly-run multi-seed experiment, and a check that fires on the right answer is uninstalled
#: within the week.
#:
#: `model` IS PRESENT, because two arms on different checkpoints are not arms of one experiment
#: however carefully everything else was matched.
_ARM_SETTINGS = (
    ("num_directions", "num_directions"),
    ("dir_mode", "dir_mode"),
    ("max_directions", "max_directions"),
    ("direction_index", "direction_index"),
    ("per_component", "per_component"),
    ("sparsity", "sparsity"),
    ("ablation_rounds", "ablation_rounds"),
    ("norm_restore", "norm_restore"),
    ("ablate_conv", "ablate_conv"),
    ("good_orth", "good_orth"),
    ("warm_start", "warm_start"),
    ("search", "search"),
    ("trials", "trials"),
    ("method", "method"),
    ("matched_scoring", "matched_scoring"),
    ("separation_statistic", "separation_statistic"),
    ("model", "model"),
    ("model_id", "model_id"),
    ("model_revision", "model_revision"),
    ("track_digest", "track_digest"),
)


def _arm_settings(doc) -> dict:
    """What this run was configured to do, for a check that compares two runs.

    A SETTING THE RECORD DOES NOT CARRY IS ABSENT, not null. `differs_in_more_than` compares only
    keys present on both sides, so an absent setting drops out of the comparison rather than
    counting as a difference against an arm that recorded it. That is the honest reading: a field
    one producer writes and the other does not is a fact about the two producers, and the check
    that owns that question asks it directly.
    """
    out = {}
    for field, name in _ARM_SETTINGS:
        value = doc.get(field)
        if value is not None:
            out[name] = value
    template = doc.get("chat_template")
    if template is not None:
        out["chat_template"] = (template.get("source") if isinstance(template, dict)
                                else template)
    budget = _generation_field(doc, "max_new_tokens")
    if budget is not None:
        out["generation_budget"] = budget
    return out


#: What `senbonzakura convert` stamps into its receipt. Matched exactly rather than sniffed at,
#: because this artefact carries no measurement at all and the usual detectors have nothing to
#: catch: a file that reports no number cannot be recognised by the numbers it reports.
_CONVERSION_KIND = "gguf-conversion"


def _is_conversion_record(doc) -> bool:
    return doc.get("record") == _CONVERSION_KIND and isinstance(doc.get("target"), dict)


def _conversion(doc) -> dict:
    """A conversion receipt in the canonical vocabulary.

    NO `metrics` BLOCK, AND THAT IS THE POINT RATHER THAN A GAP. A conversion measures nothing:
    it reports what went in, what came out, and whether the prompt format survived the trip. The
    metric checks all skip it on an empty mapping, which is the correct outcome and is visible as
    a skip rather than as a clean pass.

    The two template fields are kept SEPARATE from `chat_template` and `chat_template_applied`,
    which the run artefacts use. Those two ask whether the prompts a run scored were rendered in
    the format the model expects. These two ask whether a file carries the template at all, which
    is a different question with a different remedy, and collapsing them would give one check
    two meanings.
    """
    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    target = doc.get("target") if isinstance(doc.get("target"), dict) else {}
    return {
        "model": source.get("path") or source.get("name"),
        "tasks": [],
        "metrics": {},
        "artefact_kind": "conversion",
        "architecture": target.get("architecture") or source.get("architecture"),
        # True, False, or None for "nothing in the checkpoint had an opinion". The three are kept
        # apart deliberately: a confident False on a checkpoint nobody could read is
        # indistinguishable from a checked, clean result, which is the failure that made
        # `source_chat_template` return None in the first place.
        "source_declares_chat_template": source.get("declares_chat_template"),
        "target_carries_chat_template": target.get("carries_chat_template"),
        # The producer's own sentence about what it lost, carried across for the same reason
        # `budget_warning` is: the one line saying why a file might be wrong is worth more than
        # anything the checker can infer, and reading past it is how it got lost the first time.
        "conversion_warning": doc.get("chat_template_warning"),
        "provenance": doc.get("provenance"),
    }


#: The fields that identify an abliteration record and nothing else this project writes. Two are
#: required together rather than one alone, because `model` and `label` are on every artefact
#: here and a single edit field could plausibly appear in somebody else's log.
_ABLITERATION_MARKERS = ("num_directions", "dir_mode", "max_directions",
                         "directions_per_layer", "post_bake_refusals", "refusal_eval")


def _is_abliteration_record(doc) -> bool:
    """Is this `abliteration.json`, the record a bake writes?

    Deliberately not "does it have `num_directions`". A foreign artefact with one field in common
    must not be claimed, and ours is the LAST adapter tried precisely so that a near miss goes to
    whoever is more specific. Two of the markers together is a shape nothing else writes.
    """
    return sum(1 for k in _ABLITERATION_MARKERS if doc.get(k) is not None) >= 2


#: The abliteration record's post-bake figures, as (its field, metric, estimator, units).
#:
#: NO SAMPLE SIZE IS INVENTED FOR ANY OF THEM, and that is deliberate rather than an omission.
#: The record does not carry one, so `n` stays absent, and the checks that read a denominator
#: skip these rather than being handed a number nobody measured. An adapter that filled the gap
#: with a plausible default would be manufacturing the exact field whose absence is the finding.
_POST_BAKE = (
    ("post_bake_refusals", "refusal_rate", "senbonzakura-ruler", "proportion"),
    ("post_bake_heretic", "refusal_rate", "heretic-keyword", "proportion"),
    ("post_bake_kl", "kl", "continuation-nll-difference", "nats"),
)


def _abliteration_metrics(doc) -> dict:
    """The figures an `abliteration.json` reports, in the canonical vocabulary.

    THE KL ESTIMATOR IS THE WEAK ONE, NAMED AS THE WEAK ONE. The post-bake KL here is not the
    first-token full-distribution figure `drift` computes; it is whatever the bake measured on its
    way past, and calling it `first-token-full-distribution` would be this project doing to itself
    the precise thing the registry exists to prevent, which is a number acquiring a plausible
    label rather than a true one. Recorded as the continuation-NLL estimator, which the registry
    declares and describes as NOT a KL divergence, so a reader meets the caveat with the number.
    """
    out = {}
    for field, metric, estimator, units in _POST_BAKE:
        value = doc.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[f"{metric}.{estimator}"] = {
            "metric": metric, "value": value, "estimator": estimator, "units": units,
            "n": None, "higher_is_better": False, "interval": None,
        }
    return out


def _agreed(metrics, field):
    """One value of `field` across every stamped metric, or None if they do not all agree.

    DISAGREEMENT IS NOT A TIE TO BREAK. A document whose two metrics were rendered in different
    prompt formats has no single prompt format, and picking either one would hand a check a fact
    about half the file while looking like a fact about the file. None means "this record does not
    answer that question", which is what makes a check SKIP rather than pass, and skipping is the
    honest outcome: a check that examined it and said nothing would report it clean on a question
    it never asked. That is the failure shape this project keeps finding, so it is refused here
    rather than rediscovered downstream.
    """
    seen = {block.get(field) for block in metrics.values()
            if isinstance(block, dict) and block.get(field) is not None}
    return seen.pop() if len(seen) == 1 else None


def _older_shapes(doc) -> dict:
    """The per-command artefacts written before `measurement.stamp` existed."""
    out = _abliteration_metrics(doc)

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

    # THE COHERENCE PROBE, WRITTEN BEFORE THE STAMP EXISTED. Three of these are the only coherence
    # evidence this project has ever produced: the base, Heretic and senbonzakura arms of the
    # 2026-07-14 head-to-head, sitting on the ROG with `nll`, `ppl`, `n_tokens`, `label` and
    # `model` and nothing else. Without this the checker declines them outright, which is honest
    # but leaves a published comparison permanently unreadable by the tool whose job is reading it.
    #
    # `n` IS ONE, matching what `coherence._stamp_coherence` writes and for the same reason: the
    # probe reads one passage, and `n_tokens` in the denominator would hand the sample-size check
    # 268 independent observations that are really the words of one paragraph.
    #
    # NO `input_digest`, NO `partition`, NO `prompt_format` HERE, and that is the point rather than
    # an omission. The file does not record which passage was read, so this adapter cannot say, and
    # inventing the constant's digest would assert that the default passage was used on evidence
    # that does not exist. The number becomes checkable; it stays uncomparable, which is true.
    #
    # THE ESTIMATOR IS NAMED, THOUGH, AND THE ASYMMETRY IS DELIBERATE. That looks inconsistent
    # beside the paragraph above, so here is the difference: `senbonzakura coherence` has no flag
    # that can vary the passage and its `main` calls `coherence(model, tok)` with no text
    # argument, so every artefact it has ever written is a neutral-passage NLL. Naming the
    # estimator decodes the format; naming the digest would guess at the content. The moment the
    # command CAN score another passage that stops being true for new files, so
    # `test_coherence.py` pins the absence of such a flag to this inference rather than leaving
    # the two to drift.
    if isinstance(doc.get("nll"), (int, float)):
        out["coherence"] = {
            "metric": "coherence", "value": doc["nll"],
            "estimator": "neutral-passage-nll", "units": "nats-per-token",
            "n": 1, "higher_is_better": False, "interval": None,
            "n_tokens": doc.get("n_tokens"),
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
