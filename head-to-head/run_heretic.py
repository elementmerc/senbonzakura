#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Drive Heretic against OUR corpus, inside the sealed box, and record what it was given.

This is an adapter, not a wrapper that changes behaviour. Heretic runs as its author wrote it. All
this does is:

  * point it at the same model and the same prompts senbonzakura gets, because a comparison across
    two corpora is a comparison of corpora;
  * pin the seed and the trial count to the numbers in `head-to-head/EQUAL-BUDGET.md`;
  * record the budget actually consumed in all three units, since trials alone hide a selection
    stage;
  * leave the Optuna study on disk so the best-of-N pass required by the equal-budget definition
    can be applied to it afterwards, from outside Heretic's code.

WHAT THIS DOES NOT DO: SAVE A MODEL

v1.4.0 is interactive once its search finishes. It presents its Pareto front, asks which trial to
use, then asks what to do with it, and it has no setting that answers either question in advance
(the release after this one adds them; v1.4.0 is the current release and is what people install).
A batch job has no terminal to answer with.

That is not a problem, because the search is the tool and the save is only serialisation. Optuna
writes the study to disk as the search runs, so by the time the menu appears every trial is
already on disk. `head-to-head/best_of_n_heretic.py` reads it, rebuilds the candidates, and saves both
Heretic's own first offer and the winner of the best-of-N pass. Heretic's search runs exactly as
released; nothing about it is modified.

So the exit code here is not the measure of success. **The study is.** Heretic will exit non-zero,
or exit zero having done nothing, once it reaches a question it cannot ask. What matters is
whether the study on disk holds the trials that were asked for, and that is what is checked.

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
    cfg = f"""# Written by head-to-head/run_heretic.py. Heretic is not modified; this is what it was told.
model = "{args.model}"
seed = {args.seed}
n_trials = {args.trials}

# OUR corpus, mounted read-only. Heretic's defaults would fetch prompt sets from the Hub, which
# the sealed box cannot reach, so pointing it here is what makes the two tools comparable rather
# than merely co-located. `column` is the field name our track writer uses.
#
# THE SLICE IS AS LOAD-BEARING AS THE PATH. Handed the whole harmless partition, Heretic fits its
# directions on 4982 prompts where senbonzakura fits on {args.dir_prompts}, which is a different
# experiment wearing the same corpus. Both tools now read the same first {args.dir_prompts} rows of
# each side, which is what senbonzakura's --dir-prompts selects.
[good_prompts]
dataset = "{args.good}"
split = "train[:{args.dir_prompts}]"
column = "text"

[bad_prompts]
dataset = "{args.bad}"
split = "train[:{args.dir_prompts}]"
column = "text"

# THE EVALUATION PROMPTS ARE A SEPARATE PAIR OF TABLES from the two above, and left at their
# defaults they fetch from the Hub, which a box with no network cannot reach. Beyond that: a
# tool's search is steered by whatever it is scored on, so two tools scored on different prompts
# have not been given the same problem, and a table built from that would compare evaluation sets
# while claiming to compare tools. These files are the slices senbonzakura is scored on, written
# by head-to-head/stage_eval_slices.py from the same code that builds them for our own arm. Heretic reads
# a plain text file as one prompt per line, so `split` and `column` are not needed for that form.
[bad_evaluation_prompts]
dataset = "{args.keyword_prompts}"

[good_evaluation_prompts]
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


def count_trials(study_file):
    """How many trials the study on disk actually holds.

    Imported here rather than at module scope so this file stays readable, and testable, outside
    the container where Optuna lives.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    storage = JournalStorage(JournalFileBackend(study_file,
                                                lock_obj=JournalFileOpenLock(study_file)))
    studies = storage.get_all_studies()
    if not studies:
        return 0
    return len(optuna.load_study(study_name=studies[0].study_name, storage=storage).trials)


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
    ap.add_argument("--dir-prompts", type=int, default=256,
                    help="how many prompts per side the directions are fitted on, matched to "
                         "senbonzakura's --dir-prompts. Left unmatched, the two tools fit their "
                         "directions on differently sized corpora and the comparison is of that.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--trials", type=int, default=200,
                    help="matched to Heretic's own default rather than to our lower one, because "
                         "capping it at ours would buy a result by starving the comparison")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
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
    #
    # NOT `-m heretic.main`. That module has no `if __name__ == "__main__"` block, because Heretic
    # ships as a console script declared in its pyproject (`heretic = "heretic.main:main"`). Run
    # with `-m` it imports cleanly, runs nothing, and EXITS 0 after fourteen seconds. Nothing about
    # that reads as a failure: no traceback, no message, a successful exit code and a budget file
    # recording a completed arm. It was caught by a dry run, and the check below is what catches it
    # if it ever happens for a different reason.
    proc = subprocess.run(
        [sys.executable, "-c", "from heretic.main import main; main()"], check=False)
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
        # Stated rather than implied: this arm has not yet had the best-of-N selection pass that
        # `head-to-head/EQUAL-BUDGET.md` promises it, and a row read before that pass is applied is not
        # the matched comparison the gate asks for.
        "best_of_n_applied": False,
    }
    with open(os.path.join(a.out, "budget.json"), "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)

    print(f"run_heretic: exit {proc.returncode} after {elapsed:.0f}s")

    # THE STUDY IS THE MEASURE OF SUCCESS, NOT THE EXIT CODE, and both directions of that matter.
    #
    # A non-zero exit is expected here: v1.4.0 ends with an interactive menu it cannot ask in a
    # container, and it reaches that menu only after every trial is already on disk. Treating that
    # as a failed arm would throw away a completed search.
    #
    # A zero exit is equally not evidence that a search happened. `-m heretic.main` exits 0 having
    # run nothing at all, and a configuration error exits 0 too. So the file Heretic writes as it
    # goes is what gets checked, in both cases.
    study = budget["study"]
    if not os.path.isfile(study):
        print(f"run_heretic: FAILED. No study was written to {study}, so no search ran. Heretic "
              f"writes one as it goes, so this means it never started rather than that it "
              f"finished badly. Read the output above for the reason.", file=sys.stderr)
        return 3

    # Counted through Optuna rather than by pattern-matching the journal, because the journal is
    # an internal format and a guess at its shape would fail silently the day it changes.
    trials = count_trials(study)
    print(f"run_heretic: study at {study} ({trials} of {a.trials} trials)")
    if trials < a.trials:
        print(f"run_heretic: FAILED. The search was configured for {a.trials} trials and the study "
              f"holds {trials}. A short arm is not an equal-budget arm, so it is reported as a "
              f"failure rather than recorded as a row.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
