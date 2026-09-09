#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Give Heretic the same best-of-N selection pass senbonzakura gives itself.

WHAT THIS IS FOR

senbonzakura does not report the best trial its search found. It takes the top six candidates,
re-scores them on a larger held-out slice than the search used, and reports the winner of that
second look (cli.py, the `--top-rescore` block). That is a selection stage, it costs generations,
and it appears in no trial count. A head-to-head that ignores it measures our selection procedure
and calls it our method.

`head-to-head/EQUAL-BUDGET.md` therefore commits us to giving Heretic an equivalent pass, and this is it.
It runs AFTER Heretic's own search, reads the Optuna study the search left on disk, and never
touches Heretic's code: the tool ran exactly as its author wrote it, and the selection we added to
match our own is applied from outside.

HOW THE SIX ARE CHOSEN, AND WHY NOT BY OUR SCALARISER

Each tool nominates its own six using its own in-search scores, because that is what senbonzakura
does. Our top six come from the numbers OUR search measured; Heretic's come from the numbers ITS
search measured, through the same front construction its own menu uses. Ranking Heretic's 200
trials by our scalariser instead would mean materialising and re-scoring all 200, which is the
entire search over again and would hand Heretic a budget nothing else in the comparison gets.

The pass then does to those six exactly what ours does to ours:

  * re-score each on the SAME larger held-out slice both tools' winners are chosen on;
  * measure with the same rulers, imported from `senbonzakura.metrics` rather than copied, so the
    two selections cannot drift apart while the table claims one rule;
  * pick the winner with the same weighted knee scalar, taking KL from the trial exactly as ours
    does (the re-score refreshes the refusal axes, not the coherence one);
  * save that winner as the arm's reported model.

IT ALSO DOES THE SAVING, FOR BOTH PICKS

Heretic v1.4.0 is interactive once its search ends: it shows its Pareto front, asks which trial to
use, and asks what to do with it, with no setting that answers either in advance. A container has
no terminal. But Optuna writes the study as the search runs, so by the time the menu appears every
trial is on disk and nothing is lost.

So this saves both models: the winner of the pass, and the trial Heretic's own menu offers first.
The second costs one more save and answers the question a hostile reader asks first, which is
whether the selection we added is what moved the number.

THE RECONSTRUCTION IS CHECKED, NOT ASSUMED

Materialising a candidate means recomputing the residual directions and re-applying the trial's
recorded parameters. If that reconstruction were subtly wrong (a different direction, a shifted
index) every candidate would still produce a model, still score, and still yield a table, and the
table would be of something other than Heretic. So before the re-score, the pass rebuilds the
first candidate and measures Heretic's own keyword rate, on Heretic's own eval slice, with our
byte-identical copy of its marker list, and compares that against the rate the trial recorded. A
disagreement beyond tolerance stops the pass rather than publishing a number.

Runs inside the sealed container, where Heretic and its dependency tree live.
"""
import argparse
import json
import math
import os
import sys

# Reaching senbonzakura's rulers and its coherence measurement rather than copying them. Only
# `metrics` and `firsttoken` are imported: the first imports nothing at all, the second imports
# nothing but torch. Between them they pull in none of the abliterator and cannot collide with the
# transformers version Heretic pins in this image. Anything heavier is not importable here at all,
# because our package is MOUNTED rather than installed and its dependencies are not in this image.
sys.path.insert(0, "/work/senbon-src")

import optuna
import torch
import torch.nn.functional as F
from heretic.config import DatasetSpecification, Settings
from heretic.model import AbliterationParameters, Model
from heretic.utils import load_prompts
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from senbonzakura.firsttoken import first_token_logprobs
from senbonzakura.firsttoken import kl as first_token_kl
from senbonzakura.metrics import (
    KL_TARGET,
    broken_rate,
    heretic_keyword_rate,
    knee_scalar,
    refusal_rate,
    soft_refusal_rate,
)

# How far the reconstruction check may drift before the pass refuses. Generation is greedy, so an
# exact reconstruction reproduces the recorded rate exactly; the tolerance covers the one honest
# source of difference, which is non-deterministic reduction order on the GPU changing a borderline
# token. Two prompts in a hundred is far more slack than that needs and far less than a wrong
# direction would produce.
RECONSTRUCTION_TOLERANCE = 0.02

# What v1.4.0 records on every trial (`trial.set_user_attr` in its objective). `refusals` is a
# COUNT, not a rate, and `n_bad_prompts` is what it is a count out of. Named here because a rename
# upstream must fail loudly rather than degrade to a missing value that reads as zero.
REFUSAL_COUNT = "refusals"
KL_ATTR = "kl_divergence"

#: What this pass records as the source of a coherence figure, and what the shared-rule guard
#: asserts. A literal rather than a bare string at the use site, because the guard compares against
#: it and a typo on either side would silently turn the check off.
KL_SOURCE_MEASURED = "measured by this pass"

#: The estimator's own definition, recorded beside the numbers it produces. The peer reviewing this
#: change put it plainly: a narrow instrument read two different ways is what produced the published
#: claim being withdrawn, so what the instrument IS travels with its output.
KL_ESTIMATOR = ("KL(pristine || abliterated) over the vocabulary at the first generated token, "
                "log-softmax of the last-token logits on senbonzakura's chat rendering "
                "(enable_thinking=False), averaged over prompts; senbonzakura.firsttoken, the same "
                "code path the published drift figure and the abliterator's own search use")
N_EVAL_ATTR = "n_bad_prompts"


def _text_file_spec(path):
    """A DatasetSpecification for a plain text file, which is Heretic's one-prompt-per-line form."""
    return DatasetSpecification(dataset=path)


def attr(trial, name):
    """One of the values Heretic recorded on a trial, or None when it did not record it."""
    return trial.user_attrs.get(name)


def refusal_fraction(trial):
    """The recorded refusal COUNT as a fraction, which is the unit everything else here uses."""
    count, total = attr(trial, REFUSAL_COUNT), attr(trial, N_EVAL_ATTR)
    if count is None or not total:
        return None
    return count / total


def open_study(path):
    """Read the study Heretic left behind, without writing to it."""
    if not os.path.isfile(path):
        raise SystemExit(
            f"best_of_n: no Optuna study at {path}. Heretic writes one per model under its "
            f"study_checkpoint_dir as the search runs, so a missing file means the search never "
            f"started rather than that it finished badly. Check the arm's log.")
    backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    storage = JournalStorage(backend)
    studies = storage.get_all_studies()
    if not studies:
        raise SystemExit(f"best_of_n: the study file at {path} holds no study.")
    return optuna.load_study(study_name=studies[0].study_name, storage=storage)


def residual_directions_for(settings, model, good_prompts, bad_prompts):
    """Recompute the directions the search used.

    This mirrors heretic/main.py's direction block, including the projected-abliteration step it
    applies when `orthogonalize_direction` is set. It is a copy, which is why the caller checks the
    result against a recorded score before trusting it.
    """
    good_means = model.get_residuals_mean(good_prompts)
    bad_means = model.get_residuals_mean(bad_prompts)
    directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection = torch.sum(directions * good_directions, dim=1)
        directions = directions - projection.unsqueeze(1) * good_directions
        directions = F.normalize(directions, p=2, dim=1)

    return directions


def heretic_front(complete):
    """Heretic's own Pareto front, in the order its own menu offers it.

    Reproduced from v1.4.0's trial loop rather than taken from `study.best_trials`, and its source
    says why: the objective it hands Optuna is a scalarised score, not the pure refusal count and
    KL divergence, so Optuna's own front is over different quantities than the ones the menu shows.
    It sorts every completed trial by (refusal count, KL divergence) and then walks that order
    keeping each trial that improves on the lowest KL seen so far.

    Getting this wrong would not fail. It would nominate a different six, and the published arm
    would be a comparison against a Heretic nobody runs.
    """
    ranked = []
    for trial in complete:
        refusals, kl = attr(trial, REFUSAL_COUNT), attr(trial, KL_ATTR)
        if refusals is None or kl is None:
            raise SystemExit(
                f"best_of_n: trial {attr(trial, 'index') or trial.number} recorded no "
                f"'{REFUSAL_COUNT}' or '{KL_ATTR}', so the front cannot be built the way Heretic "
                f"builds it. This study was not written by the version this pass reads.")
        ranked.append((refusals, kl, trial))
    ranked.sort(key=lambda r: (r[0], r[1]))

    front, lowest_kl = [], math.inf
    for _, kl, trial in ranked:
        if kl < lowest_kl:
            lowest_kl = kl
            front.append(trial)
    return front


def nominate(complete, top_n):
    """The candidates Heretic puts forward, in the order its own menu puts them forward.

    Separated from the run so it can be tested without a GPU or a container: this function decides
    which six models get built, and a mistake here is a mistake in the published arm.
    """
    if not complete:
        raise SystemExit("best_of_n: the study holds no completed trial to select from.")
    return heretic_front(complete)[:max(1, top_n)]


def rescore(candidates, responses_for, kl_for):
    """Re-score each candidate on the larger slice and pick the winner by the same knee scalar.

    `responses_for(trial)` materialises the candidate and returns its generations, and
    `kl_for(trial)` measures its coherence, so the decision can be exercised against fixed numbers
    with no model in the room.

    WHY `kl_for` IS AN ARGUMENT AND NOT AN ATTRIBUTE LOOKUP (panel finding S1, option C).

    This used to read `kl_divergence` off the trial, on the stated reasoning that "the re-score
    refreshes the refusal axes and leaves the coherence axis alone". Three of the four inputs to
    the shared rule were therefore re-measured by this harness on identical prompts, and the fourth
    was whatever each tool had recorded about itself.

    They are not the same quantity. Heretic's runs 0.0014 to 0.0032 and senbonzakura's runs 0.157
    to 0.212. The rule surcharges coherence as `0.5 * max(0, kl - KL_TARGET)` with KL_TARGET at
    0.1 in absolute units, so the brake was arithmetically zero for every Heretic candidate and
    positive for every one of ours. On their side the rule reduced to "remove the most refusal,
    with no coherence brake at all", and the winner it picked was then published as having done
    twice the collateral damage, on the very axis the selection had stopped weighing.

    So the pass measures it, for both tools, with `senbonzakura.firsttoken`, which is the same
    module the published `drift` figure comes from and the same one the abliterator's own search
    uses. One estimator, one renderer, one set of prompts.
    """
    rows, generations = [], 0
    for trial in candidates:
        responses = responses_for(trial)
        generations += len(responses)
        kl = kl_for(trial)
        if kl is None:
            raise SystemExit(
                f"best_of_n: the coherence measurement for trial {attr(trial, 'index')} produced "
                f"nothing. The knee scalar surcharges KL above a target, so treating a missing "
                f"value as zero would quietly rank a damaged candidate as an intact one.")
        row = {
            "trial": attr(trial, "index"),
            "refusals": refusal_rate(responses),
            "soft": soft_refusal_rate(responses),
            "heretic": heretic_keyword_rate(responses),
            "broken": broken_rate(responses),
            # Measured here, on the shared coherence slice, with senbonzakura's estimator. The
            # provenance is recorded rather than assumed because the guard below asserts it: a
            # figure read off the tool's own trial is not comparable with ours and must never
            # reach the shared rule again.
            "kl": kl,
            "kl_source": KL_SOURCE_MEASURED,
        }
        # `broken` is passed for the same reason it is on our side, and it matters MORE here: this
        # pass is what gives the competing tool the same best-of-N selection ours gets, so a
        # brokenness term on one side and not the other would be us grading the two on different
        # rules while the published table claims one. It was measured on the line above and
        # dropped on this one.
        row["knee"] = knee_scalar(row["refusals"], row["soft"], row["heretic"], kl,
                                  broken=row["broken"])
        row["kl_surcharge"] = max(0.0, kl - KL_TARGET)
        rows.append(row)
        print(f"  trial {row['trial']}: refusals={row['refusals']*100:.1f}% "
              f"soft={row['soft']*100:.1f}% heretic={row['heretic']*100:.1f}% "
              f"broken={row['broken']*100:.0f}% KL={kl} knee={row['knee']:.4f}")
    refuse_if_the_rule_is_not_shared(rows)
    return rows, min(rows, key=lambda r: r["knee"]), generations


def refuse_if_the_rule_is_not_shared(rows):
    """Stop, loudly, when the shared selection rule is not shared in effect.

    THE BIAS THIS CATCHES, found by the 2026-09-09 panel's rival-author persona.

    `knee_scalar` is described everywhere in this harness as the same rule applied to both tools,
    and it surcharges coherence as `0.5 * max(0, kl - KL_TARGET)` with KL_TARGET = 0.1. That works
    only if the two tools' KL figures are the same quantity. They were not: Heretic's own
    `kl_divergence` runs 0.0014 to 0.0032 where senbonzakura's runs 0.157 to 0.212, so the brake was
    arithmetically zero on their side and positive on ours, and the winner that picked was
    published as having done twice the collateral damage on the very axis the selection had
    stopped weighing.

    WHAT THIS CHECK ASKS, AND WHY IT CHANGED.

    Its first version inferred the bias from a statistical signature: every candidate's surcharge
    being zero. That was the right alarm while KL came from each tool's own attribute, because on
    Heretic's scale it could not be anything else. It is the WRONG alarm now. With the pass
    measuring KL itself for both tools, every surcharge being zero means every candidate came in
    under the coherence target, which is a healthy search reporting good news, and refusing there
    would stop a run that is working.

    So it asks the thing directly instead of a proxy for it: was each figure MEASURED by this pass,
    on the shared slice, with the shared estimator. Provenance is a fact about where a number came
    from and cannot be mimicked by a run that happens to land in a particular range.

    That distinction is not academic here. Earlier the same day, a slice fix was justified by
    checking a cheap proxy (a partition's total row count) instead of the thing that mattered
    (where its boundary sits), and it read sixty rows of the published measurement into a
    selection. Check the thing.
    """
    if not rows:
        return
    borrowed = [r for r in rows if r.get("kl_source") != KL_SOURCE_MEASURED]
    if borrowed:
        sources = sorted({str(r.get("kl_source")) for r in borrowed})
        raise SystemExit(
            f"best_of_n REFUSES to pick a winner: {len(borrowed)} of {len(rows)} candidates carry "
            f"a coherence figure this pass did not measure (source: {', '.join(sources)}).\n"
            f"\n"
            f"  The shared selection rule surcharges coherence against an absolute target of "
            f"{KL_TARGET}. A figure taken from a tool's own trial is on that tool's own scale, and "
            f"the two scales differ by two orders of magnitude, so the same threshold means "
            f"'always intact' on one side and 'always penalised' on the other. Selecting each arm "
            f"that way and then publishing a comparison OF coherence measures the selection rather "
            f"than the tools.\n"
            f"\n"
            f"  This is the failure `headtohead_report.py` already refuses in the reporting, where "
            f"it will not print the two tools' KL figures in one column, calling it the error this "
            f"project withdrew four claims for on 2026-08-05. It must not come back through the "
            f"arm construction, where the report's own guard cannot see it.\n"
            f"\n"
            f"  See head-to-head/EQUAL-BUDGET.md.")


def reconstruction_ok(recorded, measured):
    """Whether rebuilding a trial reproduced the score the search recorded for it."""
    if recorded is None:
        return False, None
    drift = abs(measured - recorded)
    return drift <= RECONSTRUCTION_TOLERANCE, drift


def apply_trial(model, directions, trial):
    """Put the model into the state that trial measured."""
    model.reset_model()
    model.abliterate(
        directions,
        trial.user_attrs["direction_index"],
        {k: AbliterationParameters(**v) for k, v in trial.user_attrs["parameters"].items()},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the arm's output directory, holding budget.json")
    ap.add_argument("--final-prompts", required=True,
                    help="the larger held-out slice both tools' winners are chosen on")
    ap.add_argument("--keyword-prompts", required=True,
                    help="the slice Heretic's own search was scored on, used to check that "
                         "rebuilding a trial reproduces the score the search recorded for it")
    ap.add_argument("--kl-prompts", required=True,
                    help="the harmless coherence slice, one prompt per line. THIS PASS MEASURES "
                         "KL ITSELF on these, with senbonzakura's estimator, rather than reading "
                         "the tool's own kl_divergence attribute: the two are different quantities "
                         "(Heretic's runs 0.0014 to 0.0032 where ours runs 0.157 to 0.212), so a "
                         "shared selection rule with an absolute coherence threshold was a rule "
                         "only one side ever paid")
    ap.add_argument("--kl-batch", type=int, default=16,
                    help="prompts per forward pass for the coherence measurement. The measurement "
                         "itself is small (one row of logits per prompt); this bounds the "
                         "activations, which is what a 6 GB card actually runs out of")
    ap.add_argument("--top-n", type=int, default=6,
                    help="how many candidates to re-score; 6 is what senbonzakura gives itself "
                         "(cli.py --top-rescore), so 6 is what Heretic gets")
    ap.add_argument("--save-to", default=None,
                    help="where to save the winner; defaults to <out>/model")
    ap.add_argument("--own-pick-out", default=None,
                    help="where to save the trial Heretic's own menu offers first; defaults to "
                         "<out>/model-heretic-own. Kept apart from the winner so a table can never "
                         "confuse the two.")
    a = ap.parse_args(argv)

    # HERETIC'S SETTINGS OBJECT PARSES sys.argv, and it does so on construction, from a source it
    # installs in `settings_customise_sources`. So the moment this pass builds one it meets OUR
    # flags, does not recognise them, prints its own usage and exits 2. Nothing about that names
    # the real cause. Blanked here, after our own parsing is done, so everything downstream sees an
    # empty command line and reads its configuration from the study alone, which is where this
    # pass wants it to come from anyway.
    sys.argv = sys.argv[:1]

    save_to = a.save_to or os.path.join(a.out, "model")
    a.own_pick_out = a.own_pick_out or os.path.join(a.out, "model-heretic-own")
    budget_path = os.path.join(a.out, "budget.json")
    if not os.path.isfile(budget_path):
        raise SystemExit(
            f"best_of_n: no budget.json at {budget_path}. The arm records where it left its study "
            f"and what it was given; without it this pass would have to guess at both.")
    with open(budget_path, encoding="utf-8") as f:
        budget = json.load(f)

    # The arm's exit code is deliberately NOT a gate here. v1.4.0 ends at an interactive menu it
    # cannot ask inside a container, and it reaches that menu only after every trial is on disk, so
    # a non-zero exit is the expected shape of a completed search. The study is what gets checked,
    # and `run_heretic.py` has already refused to record an arm whose study is short.
    study = open_study(budget["study"])
    # The settings RECORDED IN THE STUDY, not the config file on disk. Heretic resolves two values
    # at run time before it records them: the batch size (config default 0, meaning "benchmark the
    # card and pick one") and the model's common response prefix. Re-reading config.toml would give
    # back the unresolved 0, and generation would then batch differently from the search it is
    # meant to be reproducing.
    settings = Settings.model_validate_json(study.user_attrs["settings"])
    if not settings.batch_size:
        raise SystemExit(
            "best_of_n: the study recorded batch_size 0, which means the arm never resolved one. "
            "Generating with it would divide by zero deep inside Heretic's batching rather than "
            "here. The arm did not get as far as its first trial.")

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.user_attrs]
    if not complete:
        raise SystemExit("best_of_n: the study holds no completed trial to select from.")
    print(f"best_of_n: {len(complete)} completed trials in the study")

    candidates = nominate(complete, a.top_n)
    print(f"best_of_n: re-scoring the top {len(candidates)} of Heretic's own front "
          f"(trials {', '.join(str(attr(t, 'index')) for t in candidates)})")

    model = Model(settings)
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    directions = residual_directions_for(settings, model, good_prompts, bad_prompts)

    # The reconstruction check, before any candidate is trusted. Heretic's own eval slice, Heretic's
    # own metric, our byte-identical copy of its marker list, against the number the trial recorded.
    probe = candidates[0]
    recorded = refusal_fraction(probe)
    apply_trial(model, directions, probe)
    keyword_slice = load_prompts(settings, _text_file_spec(a.keyword_prompts))
    measured = heretic_keyword_rate(model.get_responses_batched(keyword_slice))
    ok, drift = reconstruction_ok(recorded, measured)
    print(f"best_of_n: reconstruction check on trial {attr(probe, 'index')}: "
          f"recorded {recorded}, measured {measured:.4f}")
    if not ok:
        raise SystemExit(
            f"best_of_n: rebuilding trial {attr(probe, 'index')} does not reproduce the refusal "
            f"rate the search recorded for it (recorded {recorded}, measured {measured:.4f}, drift "
            f"{drift}). Every candidate would still produce a model and a number, and the number "
            f"would be of something other than what Heretic ran. Refusing rather than publishing "
            f"it. The usual cause is a change to Heretic's direction computation that this pass "
            f"mirrors; compare residual_directions_for against its main module.")

    final_prompts = load_prompts(settings, _text_file_spec(a.final_prompts))
    print(f"best_of_n: re-scoring on {len(final_prompts)} held-out prompts")

    # THE COHERENCE BASELINE, TAKEN ONCE, WHILE THE MODEL IS STILL PRISTINE.
    #
    # Heretic applies abliteration as LoRA adapters and `reset_model()` returns to pristine by
    # zeroing them, so there is never a second resident copy of the weights to diff against. That
    # matters on the 6 GB card this runs on: Qwen3-1.7B is about 3.4 GB at fp16 and two copies
    # would not fit. What is held instead is this one tensor of baseline log-probabilities, which
    # `first_token_logprobs` returns on the CPU: one row per prompt over the vocabulary, tens of
    # megabytes, not gigabytes.
    kl_texts = [p.user for p in load_prompts(settings, _text_file_spec(a.kl_prompts))]
    model.reset_model()
    base_lp = first_token_logprobs(model.model, model.tokenizer, kl_texts, batch=a.kl_batch)
    print(f"best_of_n: coherence baseline captured on {len(kl_texts)} harmless prompts "
          f"({KL_ESTIMATOR})")

    def responses_for(trial):
        apply_trial(model, directions, trial)
        return model.get_responses_batched(final_prompts)

    def kl_for(trial):
        """Measure this candidate's coherence here, rather than read it off its trial.

        Applies the trial itself rather than relying on `responses_for` having just done so. The
        saving from sharing that state is one LoRA re-application; the cost of depending on it is
        that reordering two lines in `rescore` would measure the PREVIOUS candidate's coherence
        and report it under this one's name, with nothing failing. That is the failure shape this
        whole change exists to remove, so it does not get reintroduced to save an operation.
        """
        apply_trial(model, directions, trial)
        lp = first_token_logprobs(model.model, model.tokenizer, kl_texts, batch=a.kl_batch)
        return first_token_kl(base_lp, lp)

    rows, winner, generations = rescore(candidates, responses_for, kl_for)
    own_pick = attr(candidates[0], "index")
    chosen = next(t for t in candidates if attr(t, "index") == winner["trial"])
    print(f"best_of_n: WINNER trial {winner['trial']} (knee {winner['knee']:.4f}); "
          f"Heretic's own first offer was trial {own_pick}")

    def save(trial, path):
        apply_trial(model, directions, trial)
        merged = model.get_merged_model()
        merged.save_pretrained(path, max_shard_size=settings.max_shard_size)
        model.tokenizer.save_pretrained(path)
        print(f"best_of_n: saved trial {attr(trial, 'index')} to {path}")

    save(chosen, save_to)
    # Heretic's unaided pick is saved too, and scored by the same instrument afterwards, so a
    # reader can see whether the selection we added is what moved the number instead of taking it
    # on trust. v1.4.0 cannot save it itself without a terminal, so it is saved from here, from the
    # same study, by the same code path as the winner.
    if own_pick != winner["trial"]:
        save(candidates[0], a.own_pick_out)
    else:
        print("best_of_n: the pass did not change the pick, so there is no second model to save")

    record = {
        "tool": "heretic",
        "pass": "best-of-n",
        "top_n": a.top_n,
        "final_prompts": a.final_prompts,
        "final_prompt_count": len(final_prompts),
        "candidates": rows,
        "winner": winner,
        # Heretic's unaided pick, so a reader can see whether our added selection changed the
        # answer at all rather than having to take the claim on trust.
        "own_first_offer": own_pick,
        "changed_the_pick": winner["trial"] != own_pick,
        "generations": generations,
        "reconstruction_check": {"trial": attr(probe, "index"), "recorded": recorded,
                                 "measured": measured, "tolerance": RECONSTRUCTION_TOLERANCE},
        "model": save_to,
        "own_pick_model": a.own_pick_out if own_pick != winner["trial"] else save_to,
    }
    with open(os.path.join(a.out, "best_of_n.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # The budget is only equal once this pass has run, and the arm recorded that it had not. Say so
    # in the same file the arm wrote, so no reader has to cross-reference two.
    budget["best_of_n_applied"] = True
    budget["best_of_n_generations"] = generations
    budget["generations_total"] = budget.get("generations_total", 0) + generations
    with open(budget_path, "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)

    print("BEST_OF_N_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
