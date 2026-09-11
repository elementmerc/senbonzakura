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
