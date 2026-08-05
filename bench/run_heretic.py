#!/usr/bin/env python3
"""Drive Heretic against OUR corpus, inside the sealed box, and record what it was given.

This is an adapter, not a wrapper that changes behaviour. Heretic runs as its author wrote it. All
this does is:

  * point it at the same model and the same prompts senbonzakura gets, because a comparison across
    two corpora is a comparison of corpora;
  * pin the seed and the trial count to the numbers in `bench/EQUAL-BUDGET.md`;
  * record the budget actually consumed in all three units, since trials alone hide a selection
    stage;
  * leave the Optuna study on disk so the best-of-N pass required by the equal-budget definition
    can be applied to it afterwards, from outside Heretic's code.

WHY A CONFIG FILE AND NOT FLAGS

Heretic reads, in precedence order, CLI arguments, then `HERETIC_` environment variables, then a
`config.toml` in the working directory. The prompt sets are nested tables (`[good_prompts]`,
`[bad_prompts]`), which are awkward to express as flags or env vars and easy to get subtly wrong.
A TOML file is what the author documents, so a TOML file is what this writes, and the file is left
in the output directory so anyone disputing a row can read exactly what Heretic was told.

The working directory has to be writable and `/work` is not: the container's root filesystem is
read-only by design and only `/work/out` is mounted read-write. So this runs from there.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def write_config(args, workdir):
    """The exact configuration Heretic is given, left on disk beside the result."""
    cfg = f"""# Written by bench/run_heretic.py. Heretic is not modified; this is what it was told.
model = "{args.model}"
seed = {args.seed}
n_trials = {args.trials}

# HERETIC IS INTERACTIVE BY DEFAULT and the container has no terminal to answer it with. Once the
# search finishes it asks which trial to use and what to do with the model, and an unanswered
# question inside a batch job is a hang that looks like a slow run. Every one of these settings is
# Heretic's own documented way to answer in advance; none of them changes what it computes.
#
#   trial_index = 0     the first entry of ITS OWN sorted Pareto front, which is what its menu
#                       offers first. This is Heretic's unaided pick, saved separately, so the
#                       published table can show whether our best-of-N pass changed the answer.
#   model_action        save to disk rather than upload; the box has no network in any case.
#   export_strategy     merge the adapter into the weights, so the saved model is a plain model
#                       that our scorer can read the same way it reads senbonzakura's.
#   checkpoint_action   resume an interrupted study rather than ask. A GPU arm that dies at trial
#                       180 must not silently start again from zero on the retry.
trial_index = 0
model_action = "save"
export_strategy = "merge"
checkpoint_action = "continue"
save_directory = "{args.own_pick_out}"

# OUR corpus, mounted read-only. Heretic's defaults would fetch prompt sets from the Hub, which
# the sealed box cannot reach, so pointing it here is what makes the two tools comparable rather
# than merely co-located. `column` is the field name our track writer uses.
[good_prompts]
dataset = "{args.good}"
split = "train"
column = "text"

[bad_prompts]
dataset = "{args.bad}"
split = "train"
column = "text"

# THE SCORERS OWN THEIR OWN PROMPT SETS, and those are separate from the two tables above.
# Left at their defaults they fetch `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`
# from the Hub, so inside a box with no network the run dies at scorer initialisation before it
# runs a single trial. Beyond that: a tool's search is steered by whatever its scorers measure,
# so two tools scored on different prompts have not been given the same problem. These files are
# the slices senbonzakura is scored on, written by bench/stage_eval_slices.py from the same code
# that builds them for our own arm. Heretic reads a plain text file as one prompt per line, and
# `split`/`column` are not needed for that form.
[scorer.KeywordRate.prompts]
dataset = "{args.keyword_prompts}"

[scorer.KLDivergence.prompts]
dataset = "{args.kl_prompts}"
"""
    path = os.path.join(workdir, "config.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg)
    return path


def study_path(model, workdir):
    """Where Heretic leaves its Optuna study, derived the way Heretic derives it.

    `study_checkpoint_dir` defaults to `checkpoints` relative to the working directory, and the
    file within it is the model identifier with every character that is not alphanumeric, an
    underscore or a hyphen replaced by a double hyphen (heretic/main.py, the study checkpoint
    block). Recording the path here means the best-of-N pass finds the study by being told, rather
    than by guessing at a naming rule that could change under it.
    """
    stem = "".join(c if (c.isalnum() or c in ("_", "-")) else "--" for c in model)
    return os.path.join(workdir, "checkpoints", stem + ".jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to the staged weights, mounted read-only")
    ap.add_argument("--good", required=True, help="harmless prompts, on disk")
    ap.add_argument("--bad", required=True, help="harmful prompts, on disk")
    ap.add_argument("--keyword-prompts", required=True,
                    help="the refusal eval slice, one prompt per line; the same slice senbonzakura's "
                         "search is scored on")
    ap.add_argument("--kl-prompts", required=True,
                    help="the coherence eval slice, one prompt per line; disjoint from the prompts "
                         "the directions are fitted on")
    ap.add_argument("--out", required=True, help="writable output directory")
    ap.add_argument("--own-pick-out", default=None,
                    help="where Heretic saves the trial IT would have offered first; defaults to "
                         "<out>/model-heretic-own. Kept apart from the best-of-N winner so the two "
                         "are never confused for one another in a table.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--trials", type=int, default=200,
                    help="matched to Heretic's own default rather than to our lower one, because "
                         "capping it at ours would buy a result by starving the comparison")
    a = ap.parse_args()
    a.own_pick_out = a.own_pick_out or os.path.join(a.out, "model-heretic-own")

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.own_pick_out, exist_ok=True)
    # Heretic reads config.toml from the CURRENT directory, and only this path is writable.
    os.chdir(a.out)

    for label, path in (("model", a.model), ("good prompts", a.good), ("bad prompts", a.bad),
                        ("keyword eval slice", a.keyword_prompts),
                        ("KL eval slice", a.kl_prompts)):
        if not os.path.exists(path):
            print(f"run_heretic: {label} not found at {path}. Inputs are staged before the run "
                  f"because this container has no network to fetch them with.", file=sys.stderr)
            return 2

    cfg_path = write_config(a, a.out)
    print(f"run_heretic: config written to {cfg_path}")

    started = time.time()
    # Heretic's own entry point, unmodified, running as its author wrote it.
    proc = subprocess.run([sys.executable, "-m", "heretic.main"], check=False)
    elapsed = time.time() - started

    # The budget, in the three units the equal-budget definition requires. Trials and wall clock
    # are recorded here; generations are counted from Heretic's own study by the best-of-N pass,
    # which reads the study this run leaves behind.
    budget = {
        "tool": "heretic",
        "tool_ref": os.environ.get("TOOL_REF") or os.environ.get("BENCH_TOOL_REF", "unknown"),
        "seed": a.seed,
        "trials_configured": a.trials,
        "wall_clock_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
        "model": a.model,
        "good_prompts": a.good,
        "bad_prompts": a.bad,
        "keyword_prompts": a.keyword_prompts,
        "kl_prompts": a.kl_prompts,
        "config_written": cfg_path,
        "study": study_path(a.model, a.out),
        "own_pick_model": a.own_pick_out,
        "own_pick_trial_index": 0,
        # Stated rather than implied: this arm has not yet had the best-of-N selection pass that
        # `bench/EQUAL-BUDGET.md` promises it, and a row read before that pass is applied is not
        # the matched comparison the gate asks for.
        "best_of_n_applied": False,
    }
    with open(os.path.join(a.out, "budget.json"), "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)

    print(f"run_heretic: exit {proc.returncode} after {elapsed:.0f}s")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
