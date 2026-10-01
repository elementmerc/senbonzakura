#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Break each guard in turn and confirm something notices, as a file rather than a temp directory.

WHAT THIS MEASURES, AND WHY IT IS NOT COVERAGE

A coverage figure says a line ran. It does not say a test would have NOTICED if the line were
wrong, and those are different questions. This answers the second one the only way it can be
answered: change the line, run the test that claims to guard it, and see whether the test goes
red.

    coverage asks:  did this line execute?
    this asks:      if I break this line, does anything complain?

WHERE IT CAME FROM, because the provenance is the point

Written on 2026-09-27 as two throwaway scripts, `/tmp/mutate.py` and `/tmp/mutate2.py`, on the
build box, to check twenty-one fixes made that day. The result is the reason this file exists:

    21 guards tested.  6 MISSED.  A 29% miss rate, on a tree with a 95% coverage floor.

Five of the six were CALL SITES: the helper was correct and tested, and nothing checked that the
caller actually used it, so deleting the call changed nothing any test could see. The sixth was a
print in the compass whose removal broke nothing at all. That split is why there are two rounds
below and why the second one exists: round one tests the helpers, round two tests whether each
helper is wired in.

The scripts then sat in `/tmp` on one machine for four days, which is where the 29% figure that
every argument about this project's test quality rests on was living. One reboot and the evidence
was gone. That is the whole reason for moving them here.

WHAT CHANGED ON THE WAY IN, stated so this file can still be called the thing that produced the
number. The twenty-one cases are VERBATIM, in their original order, split into the same two
rounds. Nothing about what is mutated or which test is run has been touched, including the one
case that no longer matches the tree (see DRIFT below). What changed is the runner, in five
places, and five of the six are defects the originals had:

1. `ROOT` was hardcoded to one machine's checkout path. It is now derived from this file.
2. **Neither original exited non-zero.** Both printed their findings and fell off the end, so
   `python mutate.py; echo $?` said success whatever happened. An instrument built to find
   guards that cannot fail could not itself fail, which is the defect it exists to find, in
   itself. It now exits 1 on any MISSED or DRIFT.
3. Nothing refused a dirty tree. The runner edits files in place and restores them in a
   `finally`, so a SIGKILL, an OOM or a closed lid between the write and the restore leaves the
   working tree silently mutated, and the only way to notice is `git status`. It now refuses to
   start unless the tree is clean, handles SIGINT and SIGTERM, and verifies after every case that
   the file came back byte-identical.
4. The two timeouts disagreed (300s and 600s) for no recorded reason. One constant now.
5. A missing literal was reported but not separated from a real miss. There are now three
   outcomes and never two: NOTICED, MISSED, DRIFT. A gate that cannot say "this did not run" is a
   gate that reports a silent pass, which this project has already paid for once when a single
   unresolvable pin made a vulnerability scanner report NOT SCANNED for a whole file and a real
   advisory looked exactly like a scan that never happened.
6. **`noticed = r.returncode != 0` counted five pytest exit codes as the guard working when only
   one of them means a test failed.** Found by the test written for point 2, on its first run. See
   `_read_pytest_status` for which code means what, and for why this could only have made the
   2026-09-27 figure look better than it was rather than worse.

THE ONE CASE THAT HAS DRIFTED, and it drifted for a good reason

`R2 measure points at the stage log` no longer matches `src/senbonzakura/measure.py`, checked
2026-10-01. The case is kept unchanged rather than repaired, because repairing it would make this
file a reconstruction of the measurement instead of the measurement.

It drifted because the line it mutates was found to be WRONG and was fixed. The old text claimed
a refused stage's reason was "in this run's log above"; an int exit code comes from argparse,
which writes to stderr, and the log is stdout, so `senbonzakura measure ... > run.log` produced a
log saying the reason was above it with no reason anywhere in it. The code now branches, and the
old sentence survives only on the non-int path. So the guard's subject improved and the guard's
literal went stale, which is exactly the drift a file in the tree with a test over it notices and
a script in `/tmp` does not.

Fixing it means choosing a new literal and saying so here. Do not do it silently.

HOW TO RUN IT

    python tools/research/mutate.py --list          # what it would do, changing nothing
    python tools/research/mutate.py                 # both rounds
    python tools/research/mutate.py --round 2       # call sites only

It needs a clean working tree and it needs the suite's dependencies, so it runs where the suite
runs. It is deliberately NOT wired into CI: see `private/plans/2026-10-01-supply-chain-ci.md` §8
for the arithmetic, which says a standing full mutation gate over this tree is not expensive but
impossible, and for what is proposed instead.
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

# The checkout this file lives in, rather than one machine's path. tools/research/mutate.py, so up
# three.
ROOT = pathlib.Path(__file__).resolve().parents[2]

# One value, where the originals had 300 and 600 with no reason given for the difference. The
# slowest case in either round is a `tests/test_convert.py` run; 600s is the larger of the two
# originals and nothing here has ever approached it.
TEST_TIMEOUT_S = 600

# ROUND ONE: the helpers. Verbatim from /tmp/mutate.py, 2026-09-27.
# Original docstring: "Break each of today's fixes in turn and confirm its guard notices."
ROUND_ONE = [
    ("abbreviation off on convert", "src/senbonzakura/convert.py",
     "        allow_abbrev=False,\n", "",
     "tests/test_a_flag_that_is_a_prefix_of_another_flag.py"),
    ("the wheel's build stamp is read", "src/senbonzakura/crashsafe.py",
     "        from ._build import COMMIT\n", "        COMMIT = None\n",
     "tests/test_a_wheel_can_say_which_commit_built_it.py"),
    ("a cut value is marked", "src/senbonzakura/say.py",
     "    return text[:limit - len(CUT)] + CUT", "    return text[:limit]",
     "tests/test_say_wraps_prose_and_nothing_else.py"),
    ("a cut list is counted", "src/senbonzakura/say.py",
     '    return joiner.join(shown) + (f", and {rest} more" if rest > 0 else "")',
     "    return joiner.join(shown)",
     "tests/test_say_wraps_prose_and_nothing_else.py"),
    ("a provenance absence is refused", "tools/ci/artefact_ok.py",
     "        if (value.strip().lower() in AN_ABSENCE\n                and key.startswith(WHERE_AN_ABSENCE_ROTS)):",
     "        if False:",
     "tests/test_a_wheel_can_say_which_commit_built_it.py"),
    ("the typo is named too", "src/senbonzakura/argresolve.py",
     '        if "required" in message and not self._has_subcommands():',
     "        if False:",
     "tests/test_a_typo_is_not_hidden_by_a_missing_flag.py"),
    ("a tokeniser counts as occupying", "src/senbonzakura/runrecord.py",
     '    "tokenizer.json", "tokenizer_config.json",', '    "tokenizer_config.json",',
     "tests/test_occupied_output.py"),
    ("a warning survives a quiet run", "src/senbonzakura/vendored.py",
     "        if any(pattern.search(line) for pattern in always):",
     "        if False:",
     "tests/test_a_quiet_tool_still_explains_its_failure.py"),
    ("measure links rather than paths", "src/senbonzakura/measure.py",
     "f\"{bundled.doc_url('guide/what-we-know')}\"",
     '"docs/guide/what-we-know"',
     "tests/test_output_never_names_a_file_the_wheel_lacks.py"),
    ("baseline names a directory", "src/senbonzakura/baseline.py",
     "        if where.is_dir():", "        if False:",
     "tests/test_a_refusal_names_a_next_step.py"),
    ("compass says it in words", "src/senbonzakura/margin.py",
     "    for line in in_words(res, score):\n        print(line)\n",
     "",
     "tests/test_margin.py::test_the_compass_prints_its_reading_in_words_as_well_as_markers"),
]

# ROUND TWO: the call sites. Verbatim from /tmp/mutate2.py, 2026-09-27.
# Original docstring: "Round two: the CALL SITES, not the helpers. Is each fix actually wired in?"
# FIVE OF THE SIX MISSES WERE IN HERE, which is the finding this whole file carries.
ROUND_TWO = [
    ("doctor uses some_of", "src/senbonzakura/doctor.py",
     "say.some_of(broken, 6)", "', '.join(broken[:6])",
     "tests/test_a_counted_list_names_what_it_left_out.py"),
    ("convert uses some_of", "src/senbonzakura/convert.py",
     "say.some_of(broken, 5)", "', '.join(broken[:5])",
     "tests/test_a_counted_list_names_what_it_left_out.py"),
    ("dataset uses some_of", "src/senbonzakura/dataset.py",
     "say.some_of(sorted(doc), 8)", "sorted(doc)[:8]",
     "tests/test_a_counted_list_names_what_it_left_out.py"),
    # The literal moved on 2026-10-01: the fixed sixty became `say.width() - _CARD_INDENT` when
    # cards went onto a line each. The drift guard in `test_the_mutation_harness_is_honest.py`
    # caught it on the first run after the edit, which is the whole reason that guard exists, and
    # the case is repaired rather than exempted because the line it mutates is still there.
    ("envsetup marks a cut card", "src/senbonzakura/envsetup.py",
     "say.shorten(g, say.width() - _CARD_INDENT)", "g[:60]",
     "tests/test_a_cut_gpu_line_says_it_was_cut.py"),
    ("the checker names --fail-on-empty", "checker/src/senbonzakura_check/cli.py",
     "            if args.fail_on_empty:", "            if False:",
     "tests/test_check_cli.py"),
    ("quantise hashes its files", "src/senbonzakura/quantise.py",
     'digest_for_the_record(out, what="output", log=log)', "None",
     "tests/test_quantise.py"),
    ("convert hashes its output", "src/senbonzakura/convert.py",
     'digest_for_the_record(out, what="GGUF") if Path(out).exists() else None', "None",
     "tests/test_convert.py"),
    ("gate says how to make a baseline", "src/senbonzakura/baseline.py",
     '            f"  A baseline is a file this tool writes, from a measurement you have already taken:\\n"',
     '            f""', "tests/test_a_refusal_names_a_next_step.py"),
    ("track gives a worked command", "src/senbonzakura/track.py",
     '            f"    senbonzakura track --harmful harmful.txt --harmless harmless.txt \\\\\\n"',
     '            f""', "tests/test_a_refusal_names_a_next_step.py"),
    ("measure points at the stage log", "src/senbonzakura/measure.py",
     'f"refused, exit status {code}. Its own reason is in this run\'s log above, "',
     'f"exited {code}. "', "tests/test_measure.py"),
]

ROUNDS = {1: ROUND_ONE, 2: ROUND_TWO}

NOTICED, MISSED, DRIFT = "NOTICED", "MISSED", "DRIFT"


def tree_is_clean(root: pathlib.Path) -> tuple[bool, str]:
    """Whether `git status` reports nothing, and what it reported if not.

    This runner edits tracked files in place. If the tree already has changes, a restore that goes
    wrong is indistinguishable from work in progress, and the operator loses the ability to tell
    which. Refusing is cheaper than explaining.
    """
    try:
        done = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                              capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"git status could not be run ({e}), so the tree's state is unknown"
    if done.returncode != 0:
        return False, f"git status exited {done.returncode}: {done.stderr.strip()}"
    dirty = done.stdout.strip()
    return (not dirty), dirty


def pytest_argv(test: str) -> list[str]:
    """How one guard is run. A function so it can be stated once and replaced in a test.

    `--no-cov` is load-bearing in this repository and is also the reason this is not inlined. The
    suite carries a coverage gate (`fail_under`), so without it a mutant that happens to drop a
    branch fails on COVERAGE rather than on the assertion, and the case is scored NOTICED for the
    wrong reason. It is also a pytest-cov option, so in an environment without that plugin it is a
    usage error, which `_read_pytest_status` now reports as DRIFT rather than as a guard working.

    `-x` stops at the first failure, since one failure is the whole answer. `-p no:cacheprovider`
    keeps a mutated run from writing to `.pytest_cache` and changing what the next run sees.

    `-B` is the same idea for the other cache, and it was added after the trap below bit.
    """
    # NO BYTECODE, AND THE REASON IS NOT TIDINESS.
    #
    # The restore at the end of `_one_case` checks the SOURCE and is right to, but a mutated run
    # leaves a `__pycache__/*.pyc` compiled from the mutant, and CPython decides a `.pyc` is still
    # valid from the source's size and its mtime in WHOLE SECONDS. A mutation that does not change
    # the file's length, such as swapping two blocks or substituting an equal-length literal, and
    # which is restored inside the same second it was written, leaves a stale `.pyc` the next
    # process imports in preference to the file on disk.
    #
    # Observed, not theorised: a block swap in `interactive.py` scored SURVIVED, and the test that
    # was then written to catch it failed against the CLEAN tree while `inspect.getsource` showed
    # the correct code, because the interpreter was running the mutant's bytecode. Both answers
    # were wrong and both looked ordinary.
    #
    # `-B` means no `.pyc` is written from a mutated source at all, so there is nothing to go
    # stale. It costs a recompile per case and buys the harness's own honesty, which is the one
    # thing it cannot trade.
    return [sys.executable, "-B", "-m", "pytest", test, "-q", "-p", "no:cacheprovider",
            "--no-cov", "-x"]


def _read_pytest_status(code: int, test: str) -> tuple[str, str]:
    """Which pytest exit codes mean the guard noticed, and which mean nothing ran.

    THE DEFECT THIS REPLACES, AND IT IS WORSE THAN THE EXIT-STATUS ONE

    Both originals scored a case with `noticed = r.returncode != 0`. Only ONE of pytest's non-zero
    codes means a test failed:

        0  all collected tests passed          -> the guard did not notice. MISSED.
        1  some test failed                    -> the guard noticed. The only NOTICED code.
        2  interrupted (Ctrl+C, a plugin)      -> nothing was concluded.
        3  internal error                      -> nothing was concluded.
        4  usage error (a bad flag, no plugin) -> pytest never ran a test.
        5  no tests were collected             -> pytest never ran a test.

    So a misspelled test path, a missing plugin or an import error in the test module all exited
    non-zero and were scored as the guard working. **Exit 5 is the one that would have bitten**: a
    path matching no file collects nothing, exits 5, and under the old rule reported NOTICED. This
    project already has that scar written down from another direction, that "no tests ran in 0.00s"
    has verified nothing, and the harness built to find unverified guards was itself reading one.

    Which way the error ran matters for the figure. A case whose test never ran was scored NOTICED,
    so the bug could only ever have made the 2026-09-27 miss rate look BETTER than it was. 6 of 21
    is a floor, not a ceiling. All twenty-one test paths are asserted to exist by
    `tests/test_the_mutation_harness_is_honest.py`, which closes the most likely route to a 5.
    """
    if code == 1:
        return NOTICED, "pytest exited 1, so the guard failed on the mutant"
    if code == 0:
        return MISSED, f"pytest exited 0 with the line broken, so {test} does not guard it"
    reasons = {2: "pytest was interrupted", 3: "pytest hit an internal error",
               4: "pytest refused its own arguments", 5: f"pytest collected no tests from {test}"}
    why = reasons.get(code, f"pytest exited {code}")
    return DRIFT, f"{why}, so this case did not run and proves nothing either way"


def run_case(case: tuple[str, str, str, str, str]) -> tuple[str, str]:
    """Mutate one line, run its guard, put the line back. Returns an outcome and a detail."""
    _name, path, old, new, test = case
    target = ROOT / path
    if not target.exists():
        return DRIFT, f"{path} does not exist"
    original = target.read_text()
    if old not in target.read_text():
        return DRIFT, f"the text to mutate is no longer in {path}"

    # The restore has to happen on every path out of here, including a signal. `finally` covers
    # an exception and a normal return; the handler installed by main() covers a signal.
    _IN_FLIGHT.append((target, original))
    try:
        target.write_text(original.replace(old, new, 1))
        try:
            done = subprocess.run(
                pytest_argv(test),
                cwd=ROOT, capture_output=True, text=True, timeout=TEST_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            # A guard that hangs on a mutant has noticed in the loudest possible way, and this
            # project has a scar from exactly that: a test written to prove one wedge was fixed
            # hung the suite and found a second wedge six lines away.
            return NOTICED, f"the guard hung, so the mutant was noticed (timeout {TEST_TIMEOUT_S}s)"
        return _read_pytest_status(done.returncode, test)
    finally:
        target.write_text(original)
        _IN_FLIGHT.pop()
        # VERIFY THE RESTORE RATHER THAN ASSUME IT. The whole risk of this script is leaving the
        # tree edited, so the one thing it must not do is trust its own write.
        if target.read_text() != original:
            sys.exit(f"FATAL: {path} was not restored. Recover it with `git checkout -- {path}`.")


_IN_FLIGHT: list[tuple[pathlib.Path, str]] = []


def _restore_and_die(signum: int, _frame: types.FrameType | None) -> None:
    for target, original in reversed(_IN_FLIGHT):
        # The per-file try is deliberate and PERF203 is suppressed for it. The restore is
        # best-effort per file: one file that cannot be written back must not stop the others
        # being written back, so this cannot be a single try around the loop. Performance is not
        # a consideration on the way out of a signal.
        try:
            target.write_text(original)
        except OSError as e:  # noqa: PERF203  # pragma: no cover - a signal during a write
            print(f"could not restore {target}: {e}", file=sys.stderr)
    print(f"\ninterrupted by signal {signum}; the tree was put back", file=sys.stderr)
    sys.exit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Break each guard in turn and confirm its test notices.",
        allow_abbrev=False)
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), action="append",
                        help="run one round only; repeatable. Default is both.")
    parser.add_argument("--list", action="store_true",
                        help="print what would be done and change nothing")
    parser.add_argument("--force", action="store_true",
                        help="run against a dirty working tree, accepting that a crash mid-run "
                             "leaves edits that look like your own")
    args = parser.parse_args(argv)

    chosen = sorted(set(args.round)) if args.round else sorted(ROUNDS)
    cases = [(n, c) for n in chosen for c in ROUNDS[n]]

    if args.list:
        for n, (name, path, _old, _new, test) in cases:
            print(f"round {n}  {name}\n          mutates {path}\n          runs    {test}")
        print(f"\n{len(cases)} case(s). Nothing was changed.")
        return 0

    clean, dirty = tree_is_clean(ROOT)
    if not clean and not args.force:
        print("REFUSING TO RUN: the working tree is not clean.\n", file=sys.stderr)
        print("This edits tracked files in place and puts them back. If it dies between those\n"
              "two steps, the edits it leaves are indistinguishable from yours. Commit, stash,\n"
              "or pass --force and accept that.\n", file=sys.stderr)
        print(dirty or "(git status gave no detail)", file=sys.stderr)
        return 2

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _restore_and_die)

    tally = {NOTICED: 0, MISSED: 0, DRIFT: 0}
    problems = []
    for n, case in cases:
        outcome, detail = run_case(case)
        tally[outcome] += 1
        print(f"{outcome:8} round {n}  {case[0]}  ({detail})", flush=True)
        if outcome != NOTICED:
            problems.append(f"{outcome}: {case[0]} ({detail})")

    print(f"\n{len(cases)} case(s): {tally[NOTICED]} noticed, "
          f"{tally[MISSED]} missed, {tally[DRIFT]} drifted")
    for line in problems:
        print(f"  {line}")

    # THE DEFECT THE ORIGINALS HAD. Both printed their findings and returned nothing, so the exit
    # status was 0 whatever they found. A missed mutant and a drifted literal are both failures of
    # this instrument's own job, and neither is allowed to be silent.
    if problems:
        print("\nA MISSED line is a guard that does not guard. A DRIFTED line is a case that did "
              "not run,\nwhich is not the same as a case that passed.")
        return 1
    print("\nevery guard noticed its mutant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
