# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Adapters for other harnesses' result files.

WHAT THESE FIXTURES ARE, AND WHAT THEY ARE NOT

The shapes below were read out of each project's own source on 2026-09-11:

- lm-evaluation-harness: `lm_eval/evaluator_utils.py::_to_eval_results`, which assembles
  `results`, `group_subtasks`, `configs`, `versions`, `n-shot`, `higher_is_better`, `n-samples`;
  plus `lm_eval/evaluator.py`, which adds the `config`, `git_hash` and `date` block afterwards.
  The metric key format is `f"{metric},{filter_key}"`, read off line 203 of that file.
- Inspect: `src/inspect_ai/log/_log.py`, the `EvalLog`, `EvalResults`, `EvalScore` and
  `EvalMetric` models.

**They are CONSTRUCTED FROM THOSE DEFINITIONS, not captured from a real run of either tool.**
That is a weaker basis than it looks and it is said plainly rather than glossed, because this
project has published a p-value that came from a fixture built to exercise a report. The claim
these fixtures support is "the schema says this is the shape". They do NOT support "a real file
looks like this". Validating against genuine output of both tools is owed and is in `DEFERRED.md`.

The practical consequence: a test here failing means the adapter disagrees with the schema. It
does not follow that the adapter works on a real file, and nothing below should be read as
saying so.
"""
import json

import pytest
from senbonzakura_check import adapters
from senbonzakura_check.adapters import UnknownArtefactError, detect, normalise

# ── fixtures, each traceable to the source that defines it ───────────────────────────────────

#: lm-evaluation-harness. Two tasks, one with two filters over the same metric, which is the
#: case the comma in the key exists for.
LM_EVAL = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.4131,
            "exact_match_stderr,strict-match": 0.0136,
            "exact_match,flexible-extract": 0.4238,
            "exact_match_stderr,flexible-extract": 0.0136,
        },
        "hellaswag": {"alias": "hellaswag", "acc,none": 0.5712, "acc_stderr,none": 0.0049},
    },
    "group_subtasks": {},
    "configs": {"gsm8k": {"task": "gsm8k"}, "hellaswag": {"task": "hellaswag"}},
    "versions": {"gsm8k": 3.0, "hellaswag": 1.0},
    "n-shot": {"gsm8k": 5, "hellaswag": 0},
    "higher_is_better": {"gsm8k": {"exact_match": True}, "hellaswag": {"acc": True}},
    "n-samples": {"gsm8k": {"original": 1319, "effective": 1319},
                  "hellaswag": {"original": 10042, "effective": 10042}},
    "config": {
        "model": "hf",
        "model_args": "pretrained=Qwen/Qwen3-1.7B",
        "batch_size": 8,
        "device": "cuda:0",
        "use_cache": None,
        "limit": None,
        "bootstrap_iters": 100000,
        "random_seed": 0,
    },
    "git_hash": "abc1234",
    "date": 1757600000.0,
}

#: Inspect. One score with two metrics, which is the ordinary shape.
INSPECT = {
    "version": 2,
    "status": "success",
    "eval": {
        "run_id": "run-1",
        "created": "2026-09-11T10:00:00Z",
        "task": "arc_easy",
        "task_version": 0,
        "dataset": {"name": "arc", "samples": 500},
        "model": "openai/gpt-4o-mini",
        "model_args": {},
        "config": {"limit": None, "epochs": 1},
    },
    "plan": {},
    "results": {
        "total_samples": 500,
        "completed_samples": 500,
        "scores": [{
            "name": "choice",
            "scorer": "choice",
            "reducer": None,
            "scored_samples": 500,
            "unscored_samples": 0,
            "params": {},
            "metrics": {
                "accuracy": {"name": "accuracy", "group": None, "value": 0.812, "params": {}},
                "stderr": {"name": "stderr", "group": None, "value": 0.017, "params": {}},
            },
        }],
    },
    "stats": {},
    "error": None,
}


# ── detection ────────────────────────────────────────────────────────────────────────────────

def test_lm_eval_is_detected():
    assert detect(LM_EVAL) is adapters.LmEvalAdapter


def test_inspect_is_detected():
    assert detect(INSPECT) is adapters.InspectAdapter


def test_our_own_artefact_is_detected():
    doc = {"kl": 0.03, "instrument": "senbonzakura.drift, KL over first-token distributions"}
    assert detect(doc) is adapters.SenbonzakuraAdapter


@pytest.mark.parametrize("doc", [
    {},
    {"results": "not a mapping", "configs": {}, "versions": {}},
    {"results": {"a": 1}, "configs": {}, "versions": {}},     # task metrics must be mappings
    {"version": 2},                                            # no eval block
    {"eval": {"task": "x"}},                                   # no version
    {"eval": {"task": "x"}, "version": "two"},                 # version is not an int
    {"some": "unrelated json"},
    [1, 2, 3],
    "a string",
])
def test_nothing_else_is_claimed(doc):
    """A DETECTOR THAT GUESSES IS HOW A FILE GETS READ AS SOMETHING IT IS NOT, and the checks
    then answer the wrong question confidently. Each of these is a near miss on purpose.
    """
    assert detect(doc) is None


def test_an_unrecognised_file_is_refused_rather_than_reported_clean():
    """The worst possible output, per the v0.8 plan, is a silent pass on a document nobody
    parsed: it is indistinguishable from a clean bill of health.
    """
    with pytest.raises(UnknownArtefactError, match="not a result artefact"):
        normalise({"something": "else"})


def test_the_refusal_names_what_is_supported():
    """A reader who has just been refused needs to know what would have worked."""
    with pytest.raises(UnknownArtefactError) as e:
        normalise({})
    for name in ("lm-evaluation-harness", "inspect", "senbonzakura"):
        assert name in str(e.value)


def test_ours_is_tried_last():
    """A foreign artefact carrying a field we also use must be read as what it is."""
    assert adapters.ADAPTERS[-1] is adapters.SenbonzakuraAdapter


# ── lm-evaluation-harness ────────────────────────────────────────────────────────────────────

def test_lm_eval_metrics_keep_the_filter_that_produced_them():
    """THE COMMA IS PROVENANCE. `exact_match,strict-match` and `exact_match,flexible-extract` are
    the same metric under different extraction rules and are not interchangeable figures.

    Splitting it away would throw out the only estimator-shaped information this format carries.
    """
    got = normalise(LM_EVAL)["metrics"]
    assert "gsm8k.exact_match,strict-match" in got
    assert "gsm8k.exact_match,flexible-extract" in got
    strict = got["gsm8k.exact_match,strict-match"]
    assert strict["metric"] == "exact_match"
    assert strict["value"] == 0.4131
    assert strict["estimator"] == "lm-eval filter strict-match"
    assert got["gsm8k.exact_match,flexible-extract"]["value"] == 0.4238


def test_lm_eval_stderr_columns_are_not_read_as_metrics():
    """`exact_match_stderr,strict-match` is the interval, not a second result. Reading it as a
    metric would double every task's findings and put an interval in a results table.
    """
    got = normalise(LM_EVAL)["metrics"]
    assert not any("_stderr" in k for k in got)
    assert got["gsm8k.exact_match,strict-match"]["stderr"] == 0.0136


def test_lm_eval_carries_direction_and_sample_count():
    got = normalise(LM_EVAL)["metrics"]["hellaswag.acc,none"]
    assert got["higher_is_better"] is True
    assert got["n"] == 10042


def test_lm_eval_lifts_the_limit_where_a_check_can_see_it():
    """A RUN WITH `limit: 10` LOOKS EXACTLY LIKE A FULL RUN in every other field.

    It is the single field that decides whether a headline number means anything, so it is lifted
    to the top rather than left in a config block a check would have to know about.
    """
    assert normalise(LM_EVAL)["limit"] is None
    truncated = {**LM_EVAL, "config": {**LM_EVAL["config"], "limit": 10}}
    assert normalise(truncated)["limit"] == 10


def test_lm_eval_carries_the_model_and_the_tasks():
    got = normalise(LM_EVAL)
    assert got["model"] == "hf"
    assert got["model_args"] == "pretrained=Qwen/Qwen3-1.7B"
    assert got["tasks"] == ["gsm8k", "hellaswag"]
    assert got["harness"] == "lm-evaluation-harness"


def test_lm_eval_survives_a_task_with_no_numeric_metrics():
    """An alias-only entry is legal and must not produce an empty metric or an exception."""
    doc = {**LM_EVAL, "results": {"empty": {"alias": "empty"}}}
    assert normalise(doc)["metrics"] == {}


# ── Inspect ──────────────────────────────────────────────────────────────────────────────────

def test_inspect_carries_the_scorer_as_the_estimator():
    """THE FIELD THAT MAKES THIS FORMAT WORTH READING.

    `EvalScore.scorer` names the function that produced the number, which is exactly the
    provenance a bare `kl` lacks and the thing the 2026-08-05 withdrawal was about. An Inspect log
    arrives with it, so the adapter carries it across rather than inventing one.
    """
    got = normalise(INSPECT)["metrics"]["choice.accuracy"]
    assert got["value"] == 0.812
    assert got["estimator"] == "choice"
    assert got["n"] == 500
    assert got["task"] == "arc_easy"


def test_inspect_reports_a_truncated_run_as_truncated():
    """A LOG CAN BE `success` WITH FEWER COMPLETED SAMPLES THAN TOTAL.

    `--fail-on-error` and early stopping both truncate, and a headline metric over a truncated
    run looks exactly like one over a full run. The comparison is done here so a check does not
    have to know this format to notice.
    """
    assert normalise(INSPECT)["truncated"] is False

    short = json.loads(json.dumps(INSPECT))
    short["results"]["completed_samples"] = 431
    got = normalise(short)
    assert got["truncated"] is True
    assert (got["completed_samples"], got["total_samples"]) == (431, 500)


def test_inspect_truncation_is_unknown_rather_than_false_when_the_counts_are_absent():
    """An unanswerable question must not read as a reassuring answer, which is the same rule
    `host_free_bytes` follows. None here means "this log does not say".
    """
    doc = json.loads(json.dumps(INSPECT))
    doc["results"] = {"scores": []}
    assert normalise(doc)["truncated"] is None


def test_inspect_carries_status_and_error():
    doc = json.loads(json.dumps(INSPECT))
    doc["status"] = "error"
    doc["error"] = {"message": "the provider hung up"}
    got = normalise(doc)
    assert got["status"] == "error"
    assert got["error"]["message"] == "the provider hung up"


def test_inspect_survives_a_log_with_no_results_at_all():
    """`results` is `EvalResults | None`, so a started-but-unfinished log has none. That is a
    real state on disk and must normalise rather than raise.
    """
    doc = {"version": 2, "status": "started", "eval": {"task": "x", "model": "m"}}
    got = normalise(doc)
    assert got["metrics"] == {}
    assert got["status"] == "started"
    assert got["truncated"] is None


def test_inspect_skips_a_malformed_score_without_losing_the_rest():
    """Graceful degradation per baseline section 2.1: one unreadable score must not cost the
    other scores in the same file.
    """
    doc = json.loads(json.dumps(INSPECT))
    doc["results"]["scores"].insert(0, "not a score object")
    doc["results"]["scores"].append({
        "name": "extra", "scorer": "model_graded",
        "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}})
    got = normalise(doc)["metrics"]
    assert "choice.accuracy" in got
    assert got["extra.accuracy"]["estimator"] == "model_graded"


# ── our own artefacts, through the same door ─────────────────────────────────────────────────

def test_our_older_artefacts_are_readable_through_the_prose_instrument():
    """A checker that only understands what this version writes cannot read this project's own
    evidence from three weeks ago.
    """
    doc = {"kl": 0.0313, "instrument": "senbonzakura.drift, KL over first-token distributions",
           "n_prompts": 200, "kl_ci": [0.02, 0.04], "logits_dtype": "bfloat16",
           "precision_ok": True}
    got = normalise(doc)
    assert got["harness"] == "senbonzakura"
    assert got["metrics"]["kl"]["value"] == 0.0313
    assert got["metrics"]["kl"]["estimator"].startswith("senbonzakura.drift")
    assert got["metrics"]["kl"]["n"] == 200
    assert got["computed_in"] == "bfloat16"


def test_a_stamped_artefact_is_read_from_the_canonical_block():
    from senbonzakura_check.measurement import stamp

    doc = stamp({}, "kl", 0.02, "first-token-full-distribution", n=150)
    got = normalise(doc)
    assert got["metrics"]["kl"]["estimator"] == "first-token-full-distribution"
    assert got["metrics"]["kl"]["units"] == "nats"
    assert got["metrics"]["kl"]["higher_is_better"] is False


def _published():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "head-to-head" / "results"
    return sorted(root.glob("*/*.json"))


#: The run index, which records which arms ran and which failed. It is NOT a measurement, and
#: excluding it is a claim about what it is rather than a convenience: its `refusal`, `drift` and
#: `scored` keys hold lists of arms, not numbers.
SUMMARY = "headtohead-summary.json"


@pytest.mark.parametrize("path", [p for p in _published() if p.name != SUMMARY],
                         ids=lambda p: p.name)
def test_every_published_arm_normalises(path):
    """Our own published evidence, read through the general path rather than a special case.

    A checker that treats its author's output specially is one whose author never finds out when
    the general path breaks. Covers all three per-arm shapes: drift, refusal and scored.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    got = normalise(doc)
    assert got["harness"] == "senbonzakura"
    assert got["metrics"], f"{path.name} normalised to no metrics at all"
    assert got["raw"] is doc, "the original must be carried through, not discarded"


def test_the_run_summary_is_refused_rather_than_read_as_a_measurement():
    """IT IS AN INDEX, NOT A RESULT, and calling it one would be the checker inventing a subject.

    Its `refusal`, `drift` and `scored` keys hold lists of arm labels. A detector loose enough to
    claim it would report findings about a document that contains no measurement, which is worse
    than reporting it unchecked.
    """
    path = next(p for p in _published() if p.name == SUMMARY)
    doc = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(UnknownArtefactError):
        normalise(doc)


def test_a_refusal_arm_carries_both_rulers_as_separate_estimators():
    """THE COMPARISON, NOT A DUPLICATE.

    Our ruler and Heretic's keyword metric sit side by side in one file. Collapsing them would
    destroy the only like-for-like figure this project has with another tool, which is exactly
    why the keyword metric is kept byte-identical.
    """
    path = next(p for p in _published() if p.name.startswith("refusal-"))
    got = normalise(json.loads(path.read_text(encoding="utf-8")))["metrics"]
    assert "refusal_rate.senbonzakura-ruler" in got
    assert "refusal_rate.heretic-keyword" in got
    assert got["refusal_rate.heretic-keyword"]["metric"] == "refusal_rate"
    assert got["refusal_rate.senbonzakura-ruler"]["estimator"] != \
        got["refusal_rate.heretic-keyword"]["estimator"]


def test_a_scored_arm_carries_its_length_only_control():
    """THE CONTROL THAT EXPOSES THE INSTRUMENT travels with the figure it controls.

    If a ruler reading nothing but prompt length separates the arms as well as the real one does,
    the real one is measuring sentence length. A reader of a finding needs both numbers, so the
    adapter carries both rather than the headline alone.
    """
    path = next(p for p in _published() if p.name.startswith("scored-"))
    got = normalise(json.loads(path.read_text(encoding="utf-8")))["metrics"]
    assert "compass_auc.margin-past-preamble" in got
    assert "compass_auc.length-only-control" in got
    assert got["compass_auc.length-only-control"]["value"] < \
        got["compass_auc.margin-past-preamble"]["value"], (
        "on this arm the real instrument beats the length-only ruler; if that ever inverts the "
        "published compass figure is measuring sentence length")


# ── the architectural rule ───────────────────────────────────────────────────────────────────

def test_no_adapter_imports_the_harness_it_reads():
    """LOOPHOLE 1 OF THE v0.8 PLAN, and it is the reason this package exists in this shape.

    Adapting to another project's API makes their release schedule our breakage: a minor version
    bump would break the adapter with no warning and no test of ours failing first. A file format
    changes far more slowly than an API, and reading one needs no dependency on the tool at all.
    """
    import ast
    from pathlib import Path

    forbidden = {"lm_eval", "inspect_ai", "torch", "transformers", "datasets"}
    pkg = Path(adapters.__file__).resolve().parent
    offenders = []
    for py in sorted(pkg.glob("*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [f"{py.name}:{node.lineno} imports {n}"
                          for n in names if n in forbidden]
    assert not offenders, "\n  ".join(["an adapter imports what it is meant to read at arm's length:", *offenders])


def test_normalise_never_mutates_what_it_was_given():
    """It is pointed at other people's evidence, and the one unforgivable behaviour is changing it."""
    before = json.dumps(LM_EVAL, sort_keys=True)
    normalise(LM_EVAL)
    assert json.dumps(LM_EVAL, sort_keys=True) == before

    before = json.dumps(INSPECT, sort_keys=True)
    normalise(INSPECT)
    assert json.dumps(INSPECT, sort_keys=True) == before


# ── the point of the whole exercise: one check set, every harness ────────────────────────────

def test_the_same_checks_run_over_an_lm_eval_artefact():
    """PROPERTY 2, BREADTH INHERITED RATHER THAN AUTHORED.

    lm-evaluation-harness encodes its extraction filter in the metric key after the comma, which
    the adapter carries across as the estimator. So a well-formed lm-eval file passes the
    provenance check without anybody writing an lm-eval-specific check.
    """
    from senbonzakura_check import check_document

    findings, skipped = check_document(LM_EVAL)
    assert [f.check_id for f in findings] == []
    assert "chat-template-never-applied" in skipped, (
        "lm-eval records the template request inside model_args, so this check cannot see it "
        "and must SKIP rather than pass: silence here carries no information")


def test_an_lm_eval_artefact_with_an_unfiltered_metric_is_caught():
    """The negative control for the above. A metric key with no comma has no filter recorded,
    so there is nothing saying how the number was extracted.
    """
    from senbonzakura_check import check_document

    doc = json.loads(json.dumps(LM_EVAL))
    doc["results"]["gsm8k"] = {"alias": "gsm8k", "exact_match": 0.41}
    findings, _ = check_document(doc)
    assert [f.check_id for f in findings] == ["metric-reported-without-its-estimator"]


def test_the_same_checks_run_over_an_inspect_log():
    """Inspect records `scorer` beside every metric, which is the provenance the 2026-08-05
    withdrawal was about. A well-formed log passes without a bespoke check.
    """
    from senbonzakura_check import check_document

    findings, _ = check_document(INSPECT)
    assert [f.check_id for f in findings] == []


def test_an_inspect_log_with_no_scorer_is_caught():
    """`scorer` is required by the model, but a hand-edited or truncated log can lack it, and
    that is exactly the artefact whose number cannot be compared with anybody's.
    """
    from senbonzakura_check import check_document

    doc = json.loads(json.dumps(INSPECT))
    doc["results"]["scores"][0]["scorer"] = None
    findings, _ = check_document(doc)
    assert [f.check_id for f in findings] == ["metric-reported-without-its-estimator"]


def test_a_finding_from_a_foreign_artefact_still_carries_its_caveat():
    """The `false_positive` sentence travels into the finding regardless of which harness wrote
    the file, because the person judging it is reading the report and not the source.
    """
    from senbonzakura_check import check_document

    doc = json.loads(json.dumps(INSPECT))
    doc["results"]["scores"][0]["scorer"] = None
    (finding,), _ = check_document(doc)
    assert "teach the adapter" in finding.false_positive
    assert "2026-08-05" in finding.incident


def test_an_unknown_file_raises_rather_than_reporting_clean():
    """`check_document` inherits the adapter's refusal, which is the behaviour that matters:
    a clean report on a document nobody parsed is worse than no report.
    """
    from senbonzakura_check import UnknownArtefactError, check_document

    with pytest.raises(UnknownArtefactError):
        check_document({"unrelated": "json"})


# ── a stamped artefact and an older one must read the same ───────────────────────────────────

def test_a_stamped_refusal_artefact_normalises_exactly_like_an_unstamped_one():
    """THE PROPERTY THAT MAKES STAMPING ADDITIVE rather than a format change.

    `score` gained a canonical metrics block on 2026-09-12 and kept every field it has always
    written. Thirty-one published head-to-head arms carry the OLD shape and a test recomputes
    their figures, so if the two shapes normalised differently, a check would say one thing
    about last month's evidence and another about tomorrow's.
    """
    from senbonzakura.score import _stamp_refusal

    old = {"label": "arm", "model": "m", "eval": "e", "n": 200,
           "refusal": 0.12, "soft_refusal": 0.03, "noncompliant": 0.01,
           "broken": 0.0, "heretic": 0.09, "provenance": {}}
    stamped = dict(old)
    _stamp_refusal(stamped)

    a, b = normalise(old)["metrics"], normalise(stamped)["metrics"]
    assert sorted(a) == sorted(b) == [
        "refusal_rate.heretic-keyword", "refusal_rate.senbonzakura-ruler"]
    for key in a:
        assert a[key]["value"] == b[key]["value"], key
        assert a[key]["estimator"] == b[key]["estimator"], key
        assert a[key]["n"] == b[key]["n"], key


def test_a_stamped_compass_artefact_normalises_exactly_like_an_unstamped_one():
    from senbonzakura.margin import _stamp_compass

    old = {"label": "arm", "model": "m", "auc": 0.91, "auc_ci": [0.87, 0.95],
           "n_harmful": 100, "n_harmless": 100, "mode": "harm_recognition",
           "controls": {"length_only_auc": 0.52}, "provenance": {}}
    stamped = dict(old)
    _stamp_compass(stamped)

    a, b = normalise(old)["metrics"], normalise(stamped)["metrics"]
    assert sorted(a) == sorted(b) == [
        "compass_auc.length-only-control", "compass_auc.margin-past-preamble"]
    for key in a:
        assert a[key]["value"] == b[key]["value"], key
        assert a[key]["estimator"] == b[key]["estimator"], key
        assert a[key]["n"] == b[key]["n"], key


def test_the_length_only_control_travels_with_the_headline_auc():
    """A reader given the compass AUC without the control that exposes a length-reading ruler
    has half the measurement. Stamping one and not the other would be worse than stamping
    neither, because the block would look complete.
    """
    from senbonzakura.margin import _stamp_compass

    doc = {"auc": 0.91, "n_harmful": 10, "n_harmless": 10,
           "controls": {"length_only_auc": 0.88}}
    _stamp_compass(doc)
    assert "compass_auc.length-only-control" in doc["metrics"]


# ── abliteration.json, the primary artefact the checker could not read ───────────────────────

#: The record a bake writes, in its real shape, built in Python rather than as a fixture file.
#:
#: NOT A FIXTURE ON PURPOSE, and the reason is now history rather than a live collision. This
#: repository's pre-commit leak gate refuses any committed JSON carrying a key named `generation`
#: at any depth, because that name is what a retained model output is called, and
#: `build_abliteration_record` nested the budget under exactly that name until 2026-09-21. The
#: gate was therefore refusing this project's own primary artefact. The gate was not weakened;
#: the field moved to `generation_settings`, which is what the default below writes. The old
#: spelling is still exercised, in `test_the_budget_is_read_from_the_pre_rename_spelling_too`,
#: because every record already on disk carries it and reading those is what an adapter is for.
def _abliteration_record(**over):
    doc = {
        "model": "Qwen/Qwen3-1.7B",
        "label": "arm-a",
        "num_directions": 1,
        "dir_mode": "per_layer",
        "max_directions": 1,
        "directions_per_layer": [1, 1, 1, 1],
        "baseline_refusals": 0.391,
        "post_bake_refusals": 0.012,
        "post_bake_heretic": 0.031,
        "post_bake_broken": 0.0,
        "post_bake_kl": 0.041,
        "refusal_eval": "track selection partition, rows 0-131",
        "generation_settings": {"greedy": True, "max_new_tokens": 48,
                                "budget_warning": None},
        "seed": 42,
    }
    doc.update(over)
    return doc


def test_an_abliteration_record_is_recognised_at_all():
    """IT WAS NOT, UNTIL 2026-09-21, and that is the uncomfortable finding behind this block.

    `abliteration.json` is what every run writes and what the model card, the head-to-head
    report, every resume guard and this checker are supposed to read. It names none of its
    figures `refusal`, `kl` or `auc`: they are `post_bake_refusals`, `post_bake_kl` and
    `post_bake_heretic`. So the detector answered false, and `senbonzakura check
    abliteration.json` reported the file UNCHECKED.

    Not clean, which is the one thing worth saying for the design: the refusal was loud and it
    was correct. It was still the case that the tool could not read the file its own runs write.
    """
    from senbonzakura_check.adapters import detect

    adapter = detect(_abliteration_record())
    assert adapter is not None, "the abliteration record is not recognised by any adapter"
    assert adapter.name == "senbonzakura"


def test_the_post_bake_figures_arrive_with_their_estimators():
    got = normalise(_abliteration_record())["metrics"]
    assert sorted(got) == ["kl.continuation-nll-difference",
                           "refusal_rate.heretic-keyword",
                           "refusal_rate.senbonzakura-ruler"]
    assert got["refusal_rate.senbonzakura-ruler"]["value"] == 0.012
    assert got["refusal_rate.heretic-keyword"]["value"] == 0.031


def test_the_post_bake_kl_is_not_called_the_real_one():
    """The registry declares a KL estimator that is NOT a KL divergence, and this is it.

    The bake's post-bake KL is not `drift`'s first-token full-distribution figure. Labelling it
    as though it were would be this project doing to itself the exact thing the measurement
    registry exists to prevent: a number acquiring a plausible label rather than a true one. The
    declared description of the estimator it does get carries the warning with it.
    """
    from senbonzakura_check.measurement import estimator_description

    got = normalise(_abliteration_record())["metrics"]["kl.continuation-nll-difference"]
    assert got["estimator"] == "continuation-nll-difference"
    assert "NOT a KL" in estimator_description("kl", got["estimator"])


def test_no_sample_size_is_invented_for_a_record_that_records_none():
    """The abliteration record carries no denominator for its post-bake figures, so the
    normalised document carries none either. Filling the gap with a plausible default would
    manufacture the exact field whose absence is what the sample-size check is looking for.
    """
    got = normalise(_abliteration_record())["metrics"]
    assert all(block["n"] is None for block in got.values())


def test_the_rows_the_figures_were_scored_on_arrive_under_the_name_the_checks_read():
    """`refusal_eval` is the abliteration record's name for what every other artefact calls
    `eval`. Lifted by the adapter rather than taught to five checks as a second vocabulary.
    """
    got = normalise(_abliteration_record())
    assert got["eval_split"] == "track selection partition, rows 0-131"
    assert normalise(_abliteration_record(refusal_eval=None))["eval_split"] is None


def test_the_generation_budget_and_its_warning_are_both_lifted_out_of_the_nesting():
    """THE FIELD THAT MADE THE CHECK POSSIBLE, and the one the seeded corpus cannot carry.

    The budget and the tool's warning about it are nested one level down in the real record. The
    2026-09-17 finding was that a warning written into an artefact and not carried into the
    normalised document is a warning no check can read however many exist, and the budget is the
    same field one step earlier.
    """
    got = normalise(_abliteration_record())
    assert got["generation_budget"] == 48
    assert got["budget_warning"] is None

    warned = normalise(_abliteration_record(
        generation_settings={"greedy": True, "max_new_tokens": 48,
                             "budget_warning": "gen-tokens 48 is below the 96 visibility floor"}))
    assert warned["generation_budget"] == 48
    assert "visibility floor" in warned["budget_warning"]


def test_the_budget_is_read_from_the_post_rename_spelling():
    """`generation_settings` is where `build_abliteration_record` has nested the budget since
    2026-09-21, so this is the spelling every record written from now on carries.

    Deliberately a separate test from the one below rather than one test over a document
    carrying both names, which would pass if either path worked and prove nothing about which.
    """
    doc = {"num_directions": 1, "dir_mode": "per_layer",
           "generation_settings": {"greedy": True, "max_new_tokens": 48,
                                   "budget_warning": "below the 96 visibility floor"}}
    got = normalise(doc)
    assert got["generation_budget"] == 48
    assert "visibility floor" in got["budget_warning"]


def test_the_budget_is_read_from_the_pre_rename_spelling_too():
    """THE FALLBACK IS LOAD-BEARING, and this is the half that would rot silently.

    Every `abliteration.json` written before 2026-09-21 nests the budget under `generation`,
    which is most of the ones that exist. An adapter that read only the new name would stop
    finding the budget on every one of them, so
    `quoted-at-a-budget-below-the-visibility-floor` would pass all four of its own controls and
    never fire on a real file again. That is exactly the failure this project keeps meeting: a
    check that works against documents somebody wrote for it and not against the ones a
    producer writes.
    """
    doc = {"num_directions": 1, "dir_mode": "per_layer",
           "generation": {"greedy": True, "max_new_tokens": 48,
                          "budget_warning": "below the 96 visibility floor"}}
    got = normalise(doc)
    assert got["generation_budget"] == 48
    assert "visibility floor" in got["budget_warning"]


def test_an_artefact_that_records_no_budget_is_skipped_rather_than_passed():
    """SKIPPED IS NOT PASSED, and the adapter is what makes that hard to get right here.

    The normalised document always carries a `generation_budget` key, because a dictionary
    literal writes every key whether or not the producer recorded anything under it. So a check
    whose `applies_to` asked only whether the key was PRESENT examined every artefact this
    project has ever written and reported each one as clean on a question it had not asked.
    Measured on 2026-09-21 against the real records in the 2026-09-21 salvage directory, which
    carry no budget block at all: the check applied to both and found nothing.
    """
    from senbonzakura_check import check_document
    from senbonzakura_check.registry import load_checks

    doc = {"num_directions": 1, "dir_mode": "per_layer", "post_bake_refusals": 0.0}
    assert normalise(doc)["generation_budget"] is None
    findings, skipped = check_document(doc, load_checks())
    assert "quoted-at-a-budget-below-the-visibility-floor" in skipped, (
        "an artefact recording no budget was examined by the budget check and passed, which "
        "reads in the report exactly like a budget that was checked and found sound")
    assert "quoted-at-a-budget-below-the-visibility-floor" not in {f.check_id for f in findings}


def test_a_short_budget_in_the_real_nesting_reaches_the_check_that_looks_for_it():
    """End to end on the real shape: the artefact a bake writes, through detection and
    normalisation, into the check registry, and out as a named finding.

    This is the evidence for `quoted-at-a-budget-below-the-visibility-floor` that the seeded
    incident corpus cannot hold, because the real nesting uses a key the leak gate refuses in
    committed JSON.
    """
    from senbonzakura_check import check_document
    from senbonzakura_check.registry import load_checks

    checks = load_checks()
    findings, _ = check_document(_abliteration_record(), checks)
    assert "quoted-at-a-budget-below-the-visibility-floor" in {f.check_id for f in findings}

    at_the_floor = _abliteration_record(
        generation_settings={"greedy": True, "max_new_tokens": 96, "budget_warning": None})
    findings, _ = check_document(at_the_floor, checks)
    assert "quoted-at-a-budget-below-the-visibility-floor" not in {f.check_id for f in findings}, (
        "a check that fires at the floor as well as below it is measuring nothing")
