#!/usr/bin/env python3
"""Give Heretic the same best-of-N selection pass senbonzakura gives itself.

WHAT THIS IS FOR

senbonzakura does not report the best trial its search found. It takes the top six candidates,
re-scores them on a larger held-out slice than the search used, and reports the winner of that
second look (cli.py, the `--top-rescore` block). That is a selection stage, it costs generations,
and it appears in no trial count. A head-to-head that ignores it measures our selection procedure
and calls it our method.

`bench/EQUAL-BUDGET.md` therefore commits us to giving Heretic an equivalent pass, and this is it.
It runs AFTER Heretic's own search, reads the Optuna study the search left on disk, and never
touches Heretic's code: the tool ran exactly as its author wrote it, and the selection we added to
match our own is applied from outside.

HOW THE SIX ARE CHOSEN, AND WHY NOT BY OUR SCALARISER

Each tool nominates its own six using its own in-search scores, because that is what senbonzakura
does. Our top six come from the numbers OUR search measured; Heretic's come from the numbers ITS
search measured. Ranking Heretic's 200 trials by our scalariser instead would mean materialising
and re-scoring all 200, which is the entire search over again and would hand Heretic a budget
nothing else in the comparison gets.

The pass then does to those six exactly what ours does to ours:

  * re-score each on the SAME larger held-out slice both tools' winners are chosen on;
  * measure with the same rulers, imported from `senbonzakura.metrics` rather than copied, so the
    two selections cannot drift apart while the table claims one rule;
  * pick the winner with the same weighted knee scalar, taking KL from the trial exactly as ours
    does (the re-score refreshes the refusal axes, not the coherence one);
  * save that winner as the arm's reported model.

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
import os
import sys

# Reaching senbonzakura's rulers rather than copying them. Only `metrics` is imported and that
# module imports nothing at all, so this pulls in none of the abliterator and cannot collide with
# the transformers version Heretic pins in this image.
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

from senbonzakura.metrics import (
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

# The names Heretic's two default scorers report under (`score_name` in each scorer module). Named
# constants because a rename upstream must fail here loudly rather than degrade to a missing value.
KEYWORD_SCORE_NAME = "Keywords"
KL_SCORE_NAME = "KL divergence"


def _text_file_spec(path):
    """A DatasetSpecification for a plain text file, which is Heretic's one-prompt-per-line form."""
    return DatasetSpecification(dataset=path)


def score_name(trial, name):
    """A named score out of a Heretic trial's recorded score records."""
    for score in trial.user_attrs.get("scores", []):
        if score["name"] == name:
            return score["score"]["value"]
    return None


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


def nominate(front, complete, objective_names, top_n):
    """The candidates Heretic puts forward, ranked the way Heretic ranks its own front.

    Separated from the run so it can be tested without a GPU or a container: this function decides
    which six models get built, and a mistake here is a mistake in the published arm.
    """
    pool = [t for t in front if t.user_attrs] or complete
    if not pool:
        raise SystemExit("best_of_n: the study holds no completed trial to select from.")

    def rank_key(trial):
        # A trial missing one of the recorded score names would otherwise sort None against a
        # float and raise from inside sorted(), several minutes into a GPU job, with a traceback
        # that names neither the trial nor the score.
        values = [score_name(trial, n) for n in objective_names]
        missing = [n for n, v in zip(objective_names, values, strict=True) if v is None]
        if missing:
            raise SystemExit(
                f"best_of_n: trial {trial.user_attrs.get('index', getattr(trial, 'number', '?'))} "
                f"has no recorded value for {', '.join(missing)}, so the front cannot be ranked the "
                f"way Heretic ranks it. The study was written by a Heretic whose scorers differ "
                f"from the ones this arm configured.")
        return tuple(values)

    return sorted(pool, key=rank_key)[:max(1, top_n)]


def rescore(candidates, responses_for):
    """Re-score each candidate on the larger slice and pick the winner by the same knee scalar.

    `responses_for(trial)` materialises the candidate and returns its generations, so the decision
    can be exercised against fixed text with no model in the room.
    """
    rows, generations = [], 0
    for trial in candidates:
        responses = responses_for(trial)
        generations += len(responses)
        kl = score_name(trial, KL_SCORE_NAME)
        if kl is None:
            raise SystemExit(
                f"best_of_n: trial {trial.user_attrs['index']} recorded no '{KL_SCORE_NAME}'. The "
                f"knee scalar surcharges KL above a target, so treating a missing value as zero "
                f"would quietly rank a damaged candidate as an intact one.")
        row = {
            "trial": trial.user_attrs["index"],
            "refusals": refusal_rate(responses),
            "soft": soft_refusal_rate(responses),
            "heretic": heretic_keyword_rate(responses),
            "broken": broken_rate(responses),
            # KL comes from the trial, not from this pass, exactly as it does in ours: the re-score
            # refreshes the refusal axes on more evidence and leaves the coherence axis alone.
            "kl": kl,
            "kl_source": "trial",
        }
        row["knee"] = knee_scalar(row["refusals"], row["soft"], row["heretic"], kl)
        rows.append(row)
        print(f"  trial {row['trial']}: refusals={row['refusals']*100:.1f}% "
              f"soft={row['soft']*100:.1f}% heretic={row['heretic']*100:.1f}% "
              f"broken={row['broken']*100:.0f}% KL={kl} knee={row['knee']:.4f}")
    return rows, min(rows, key=lambda r: r["knee"]), generations


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
                    help="the slice Heretic's own KeywordRate scorer was given, used to check that "
                         "rebuilding a trial reproduces the score the search recorded for it")
    ap.add_argument("--top-n", type=int, default=6,
                    help="how many candidates to re-score; 6 is what senbonzakura gives itself "
                         "(cli.py --top-rescore), so 6 is what Heretic gets")
    ap.add_argument("--save-to", default=None,
                    help="where to save the winner; defaults to <out>/model")
    a = ap.parse_args(argv)

    save_to = a.save_to or os.path.join(a.out, "model")
    budget_path = os.path.join(a.out, "budget.json")
    if not os.path.isfile(budget_path):
        raise SystemExit(
            f"best_of_n: no budget.json at {budget_path}. The arm records where it left its study "
            f"and what it was given; without it this pass would have to guess at both.")
    with open(budget_path, encoding="utf-8") as f:
        budget = json.load(f)

    if budget.get("exit_code") not in (0, None):
        raise SystemExit(
            f"best_of_n: the Heretic arm exited {budget['exit_code']}, so its study is incomplete. "
            f"Selecting from a truncated search would report a budget the run never spent.")

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

    # Heretic nominates its six the way it ranks its own front: the Pareto-optimal trials, sorted
    # by their recorded objective values. Falling back to every completed trial when the front is
    # somehow empty keeps the pass from silently selecting nothing.
    objective_names = [s["name"] for s in complete[0].user_attrs["scores"]]
    candidates = nominate(study.best_trials, complete, objective_names, a.top_n)
    print(f"best_of_n: re-scoring the top {len(candidates)} of Heretic's front")

    model = Model(settings)
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    directions = residual_directions_for(settings, model, good_prompts, bad_prompts)

    # The reconstruction check, before any candidate is trusted. Heretic's own eval slice, Heretic's
    # own metric, our byte-identical copy of its marker list, against the number the trial recorded.
    probe = candidates[0]
    recorded = score_name(probe, KEYWORD_SCORE_NAME)
    apply_trial(model, directions, probe)
    keyword_slice = load_prompts(settings, _text_file_spec(a.keyword_prompts))
    measured = heretic_keyword_rate(model.get_responses_batched(keyword_slice))
    ok, drift = reconstruction_ok(recorded, measured)
    print(f"best_of_n: reconstruction check on trial {probe.user_attrs['index']}: "
          f"recorded {recorded}, measured {measured:.4f}")
    if not ok:
        raise SystemExit(
            f"best_of_n: rebuilding trial {probe.user_attrs['index']} does not reproduce the score "
            f"the search recorded for it (recorded {recorded}, measured {measured:.4f}, drift "
            f"{drift}). Every candidate would still produce a model and a number, and the number "
            f"would be of something other than what Heretic ran. Refusing rather than publishing "
            f"it. The usual cause is a change to Heretic's direction computation that this pass "
            f"mirrors; compare residual_directions_for against heretic/main.py.")

    final_prompts = load_prompts(settings, _text_file_spec(a.final_prompts))
    print(f"best_of_n: re-scoring on {len(final_prompts)} held-out prompts")

    def responses_for(trial):
        apply_trial(model, directions, trial)
        return model.get_responses_batched(final_prompts)

    rows, winner, generations = rescore(candidates, responses_for)
    chosen = next(t for t in candidates if t.user_attrs["index"] == winner["trial"])
    print(f"best_of_n: WINNER trial {winner['trial']} (knee {winner['knee']:.4f}); "
          f"Heretic's own first offer was trial {candidates[0].user_attrs['index']}")

    apply_trial(model, directions, chosen)
    merged = model.get_merged_model()
    merged.save_pretrained(save_to, max_shard_size=settings.max_shard_size)
    model.tokenizer.save_pretrained(save_to)
    print(f"best_of_n: saved to {save_to}")

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
        "own_first_offer": candidates[0].user_attrs["index"],
        "changed_the_pick": winner["trial"] != candidates[0].user_attrs["index"],
        "generations": generations,
        "reconstruction_check": {"trial": probe.user_attrs["index"], "recorded": recorded,
                                 "measured": measured, "tolerance": RECONSTRUCTION_TOLERANCE},
        "model": save_to,
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
