# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every flag the harness passes exists on the script it passes it to, and vice versa.

WHY THIS FILE EXISTS

The harness builds a command line in `headtohead.py` and a separate script in `head-to-head/`
parses it. Nothing compared the two, and the failure that gap produces is the worst-shaped one this
project has: it does not appear until the arm has already run. `_heretic_finalise` fires AFTER
Heretic's search finishes, so a flag mismatch there costs the entire search before anything says a
word, and the operator sees a dead arm with no model and a container exit code.

That is not hypothetical. Two senbonzakura containers on the ROG from 2026-09-01 exited non-zero
after 47 seconds and 5 minutes, on the day a rename left two constants naming a module that has
never existed on either side of the container boundary. And this exact check, for the other
experiment driver, is `test_every_flag_it_passes_exists_on_the_real_parser` in `test_e2_arms.py`.

WHY THE SCRIPTS ARE READ RATHER THAN IMPORTED

`best_of_n_heretic.py` and `run_heretic.py` import `heretic`, which is installed only inside the
sealed container built for it. Importing them here is impossible on any developer machine, which is
precisely why nothing checked them. Their argument parsers are plain `ap.add_argument` calls, so
the flag names are readable from the source tree with `ast`, on any machine, in milliseconds.

BOTH DIRECTIONS ARE CHECKED, and the second is the one that caught a real defect. A flag the
harness passes and the script does not know is an immediate `error: unrecognized arguments`. A flag
the script REQUIRES and the harness never passes is the same death by a different message, and that
is exactly what `--kl-prompts` was: staged, cut, written to disk, and never handed to the pass that
needed it.
"""
import ast
from pathlib import Path

import pytest

from senbonzakura import headtohead

ROOT = Path(__file__).resolve().parent.parent


def parser_flags(script):
    """({every flag}, {required flags}) declared by `ap.add_argument` calls in a script.

    Read from the syntax tree rather than with a regular expression: a pattern over the text would
    be our own reimplementation of Python's parser agreeing with itself, which is the failure this
    project wrote a rule about.
    """
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    every, required = set(), set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and str(a.value).startswith("-")]
        if not names:
            continue
        every.update(names)
        for kw in node.keywords:
            if kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value:
                required.update(names)
    return every, required


def flags_in(argv):
    return {a for a in argv if isinstance(a, str) and a.startswith("--")}


#: The guest paths every builder is exercised with. Values do not matter here; the flag names do.
GUEST = dict(model=headtohead.GUEST_MODEL, track=headtohead.GUEST_CORPUS,
             out=headtohead.GUEST_OUT, slices=headtohead.GUEST_EVAL)


def finalise_argv():
    return headtohead._heretic_finalise(out=GUEST["out"], slices=GUEST["slices"])


def heretic_argv():
    return headtohead._heretic_argv(seed=42, trials=10, extra=[], **GUEST)


CASES = [
    ("head-to-head/best_of_n_heretic.py", finalise_argv, "the best-of-N selection pass"),
    ("head-to-head/run_heretic.py", heretic_argv, "the Heretic runner"),
]


@pytest.mark.parametrize(("script", "build", "what"), CASES)
def test_every_flag_the_harness_passes_is_one_the_script_accepts(script, build, what):
    every, _ = parser_flags(script)
    passed = flags_in(build())
    unknown = sorted(passed - every)
    assert not unknown, (
        f"{what} would exit with 'unrecognized arguments' on {unknown}. The harness builds this "
        f"command line in headtohead.py and {script} parses it; nothing else compares them.")


@pytest.mark.parametrize(("script", "build", "what"), CASES)
def test_every_flag_the_script_requires_is_one_the_harness_passes(script, build, what):
    """THE DIRECTION THAT CAUGHT A REAL ONE.

    `--kl-prompts` was staged, cut from the corpus, written to disk, and never passed to the pass
    that needed it, because the selection pass had no reason to want it until it started measuring
    coherence itself. A required flag nobody passes kills the arm after the search has run.
    """
    _, required = parser_flags(script)
    passed = flags_in(build())
    # A parser declares both spellings of an option in one call; satisfying either satisfies it.
    missing = sorted(f for f in required if f not in passed)
    assert not missing, (
        f"{what} requires {missing} and the harness never passes it, so every arm dies at "
        f"argument parsing after its search has already run.")


def test_the_coherence_slice_reaches_the_pass_that_measures_it():
    """Named on its own because it is the wiring S1 turned on, and it is invisible until it is not.

    The pass measures KL for both tools now. It cannot do that without the prompts, and the prompts
    were already being staged for Heretic's own scorer, so the only thing that was ever missing was
    the six characters that hand them over.
    """
    argv = finalise_argv()
    assert "--kl-prompts" in argv
    # `bestofn_kl_prompts.txt`, NOT `kl_prompts.txt`. The latter is what Heretic's search
    # optimises its own KL against, so ranking six candidates by a KL measured there rewards
    # overfitting it. `drift_prompt_slice` already refuses that confound for the published
    # coherence figure; the selection pass got the same treatment on 2026-09-10.
    given = argv[argv.index("--kl-prompts") + 1]
    assert given == f"{headtohead.GUEST_EVAL}/bestofn_kl_prompts.txt"
    assert given != f"{headtohead.GUEST_EVAL}/kl_prompts.txt"


def test_the_slice_the_pass_is_given_is_one_the_staging_step_writes():
    """The file has to be on disk under that name, or the pass opens nothing.

    `SLICE_FILES` is what the staging step writes and what the runner's preflight checks for, so
    asking it here ties the three together rather than trusting that two of them agree.
    """
    argv = finalise_argv()
    name = argv[argv.index("--kl-prompts") + 1].rsplit("/", 1)[-1]
    assert name in headtohead.SLICE_FILES, (
        f"the pass is handed {name}, which the staging step does not write "
        f"(it writes {sorted(headtohead.SLICE_FILES)})")


# ── the artefact has to carry the reasoning, not just the code ──────────────────────
#
# Every successful Heretic arm of the 2026-09-10 comparison wrote `exit_code: 1` into its
# budget.json, because v1.4.0 ends with an interactive menu it cannot present in a container and
# raises EOFError AFTER all 200 trials are on disk. `run_heretic.py` reads that correctly and
# accepts the arm. The FILE said only "1", and the file is what outlives the run.

def test_a_non_zero_exit_that_is_accepted_says_why_in_the_artefact():
    """Read from the source, because the script imports `heretic` and cannot be imported here.

    A reviewer opening ten arms and finding a failure code on all of them, with nothing recording
    why it was fine, is the position this project was in on the morning of 2026-09-10.
    """
    src = (ROOT / "head-to-head" / "run_heretic.py").read_text(encoding="utf-8")
    assert "exit_code_accepted" in src, (
        "a non-zero exit is expected from this tool and is accepted on the strength of the trial "
        "count; the artefact has to record that, or the number reads as a failure forever")
    assert "trials_completed" in src, (
        "the evidence that made the exit acceptable is the trial count, so it travels with it")
    # Written AFTER the trial count is known, or it cannot carry it.
    assert src.index("trials = count_trials(study)") < src.index("exit_code_accepted")


def test_the_short_arm_guard_still_refuses_before_any_of_that():
    """Accepting a non-zero exit must not become accepting a short search.

    The guard returns 3 on a short study, and it has to keep doing so before the acceptance is
    ever recorded, or "the search completed" gets written about a search that did not.
    """
    src = (ROOT / "head-to-head" / "run_heretic.py").read_text(encoding="utf-8")
    assert src.index("A short arm is not an equal-budget arm") < src.index("exit_code_accepted")
