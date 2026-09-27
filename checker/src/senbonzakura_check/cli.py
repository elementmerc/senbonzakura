# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura check`: point it at a result file and find out how the number could be wrong.

ITEM G OF THE SHORTLIST, AND PROPERTY 1 OF THE SIX.

`strategy-2026-08-02.md` is specific about why the 10x axis scored "partly" and what would fix
it: *"the axis is trust, and trust does not demo. Time to find a real bug does: 'your chat
template was never applied', in ten seconds, on a config the user already has."*

So this command is not a feature so much as a demonstration that has to be true. A stranger
points it at an artefact they already have and gets back a named defect, the incident it comes
from, what to do about it, and what would make the finding wrong. No model, no corpus, no card,
no account, and nothing downloaded.

THREE OUTPUT RULES, EACH FROM A LOOPHOLE IN THE v0.8 PLAN

**Unreadable is not clean.** A file no adapter recognises is reported as unchecked and exits
non-zero for it. Loophole 6: a silent pass on a file nobody parsed is indistinguishable from a
clean bill of health, and is the worst output this command could produce.

**Skipped is not passed.** A check whose `applies_to` is false did not examine the artefact.
The summary line says how many checks did not apply, because "no findings" over a set of checks
that mostly could not run is a different statement from "no findings".

**A clean report is not a certificate.** Loophole 7: it checks for known failure modes and
cannot certify that a number is right. The output says so in its own words rather than leaving
it to the README.

NAMING A FILE IS A CLAIM ABOUT IT; SWEEPING A DIRECTORY IS NOT. An unrecognised file that was
named explicitly is UNCHECKED and exits 2, because the user asserted it was a result. One swept
out of a directory is reported as "not a result artefact" and costs nothing, because they did
not. Found by pointing this at our own `head-to-head/results/`, where the run summary is
correctly not a measurement and made the whole directory exit 2.

SOME DEFECTS ARE NOT VISIBLE IN ONE FILE, AND `--pair` IS FOR THOSE. Whether two arms of a
comparison differ in more than the variable it names, and whether two figures were produced by
the same instrument, are facts about two artefacts. Those checks declare `arity: pair` and are
SKIPPED on a single document rather than passed, because a pair check reported as passing on one
file would be claiming a comparison is sound when no comparison was read. `--pair` takes exactly
two files, named: which two artefacts are arms of one experiment is a claim only the caller can
make, and a directory sweep that paired everything would report findings about comparisons
nobody ran.

EXIT CODES, and they distinguish the three outcomes on purpose, because a CI gate that treats
"could not check" the same as "checked, nothing found" is worse than no gate:

    0  every file was read and nothing fired
    1  at least one finding
    2  at least one file NAMED EXPLICITLY could not be read at all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .adapters import UnknownArtefactError, normalise
from .registry import (
    ArtefactTooLargeError,
    load_checks,
    read_json_bounded,
    run_checks,
    run_pair_checks,
)

#: What each severity means, in the words a reader meets rather than as a bare label. The label
#: alone ("severity: notes") tells somebody who has not read the documentation nothing, and the
#: person reading a finding is exactly the person who has not read the documentation.
_SEVERITY_SENTENCE = {
    "withdraws": "THE FIGURE CANNOT BE QUOTED AS WHAT IT CLAIMS TO BE",
    "qualifies": "the figure stands only with a caveat attached",
    "notes": "two numbers here could be read for each other",
}


def _invoked_as():
    """The name this was actually reached by, because there are two and only one of them exists.

    `senbonzakura check` and `senbonzakura-check` are the same command reached two ways, and the
    usage line said the first unconditionally. Somebody who installed only this distribution, which
    the documentation recommends for exactly the sceptic who wants no PyTorch, got a usage block
    naming a command their machine does not have; copying it fails. Found 2026-09-26 by a reader who
    did copy it.

    Deliberately not a hardcoded swap in the other direction either: when both are installed both
    are true, and the honest answer is whichever one the reader typed.
    """
    name = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
    return "senbonzakura-check" if name.startswith("senbonzakura-check") else "senbonzakura check"


def build_parser():
    ap = argparse.ArgumentParser(
        prog=_invoked_as(),
        description="Read an evaluation result file and report how the number could be wrong.")
    # A BUILD THAT CANNOT SAY WHICH BUILD IT IS CANNOT GATE ANYTHING.
    #
    # This is documented for continuous integration, where the point is to record what gated a
    # build. `--version` raised argparse's "unrecognised arguments" and exited 2, so a pipeline
    # asking which checker ran got a usage block and a failure. Found 2026-09-26; the abliterator
    # has carried `--version` throughout, which is how the gap survived in the smaller package.
    ap.add_argument("--version", action="version", version=f"senbonzakura-check {__version__}")
    ap.add_argument("paths", nargs="+",
                    help="result files, or directories to search for .json files")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output, one object per file")
    # `--quiet` MEANT ALMOST NOTHING. It dropped the two closing lines, 134 bytes of 5,435, and left
    # the whole finding in place, while `gate --quiet` in the same tool meant "the verdict line only".
    # One flag name, two behaviours, and the one a reader expects was the other command's.
    ap.add_argument("--quiet", action="store_true",
                    help="one line per finding, and nothing else. For a build log or a grep")
    ap.add_argument("--fail-on-empty", action="store_true",
                    help="exit non-zero when NOTHING was checked. A directory that exists and "
                         "holds no result artefacts otherwise exits 0, which in CI is a green "
                         "that means 'I found no files' and is indistinguishable from 'I found "
                         "files and they were fine'. Worth setting wherever a path could drift")
    ap.add_argument("--min-applied", type=int, default=0, metavar="N",
                    help="exit non-zero when any artefact had FEWER THAN N checks apply to it. "
                         "--fail-on-empty catches a run that examined nothing; this catches a "
                         "run that examined almost nothing, which looks identical in the exit "
                         "code and is the likelier drift. An artefact counts as checked when a "
                         "single check applied to it, so a rename that quietly stops fourteen "
                         "of fifteen checks from recognising a file still exits 0")
    ap.add_argument("--pair", action="store_true",
                    help="the paths name two arms of ONE comparison. Runs the checks that read "
                         "two artefacts, which are skipped otherwise because they cannot be "
                         "answered from a single file. Exactly two files, named explicitly: "
                         "which two artefacts are arms of one experiment is a claim only you can "
                         "make, so nothing here infers it from a directory")
    ap.add_argument("--skip-unknown", action="store_true",
                    help="treat a named file that is not a result artefact the way a swept one "
                         "is treated: report it and carry on, rather than exiting 2")
    return ap


def _files(paths):
    """Every file to check, as (path, named_explicitly), in a stable order.

    THE FLAG IS THE USER'S CLAIM ABOUT THE FILE, and it decides how an unrecognised one is
    reported. Naming a file is an assertion that it is a result; sweeping a directory is not.

    Found by pointing the checker at this project's own `head-to-head/results/`, which holds
    thirty per-arm artefacts and one run summary. The summary is an index of which arms ran and
    is correctly not a measurement, so the whole directory exited 2 and would have failed CI for
    a file that is exactly what it should be. Any real results directory has a config or a
    manifest sitting beside the results, and a checker that exits 2 on all of them is a checker
    nobody can point at a directory.

    Sorted because two runs over the same tree must produce the same report, per baseline
    section 2.1: a report whose order depends on the filesystem cannot be diffed between runs.
    """
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend((q, False) for q in sorted(p.rglob("*.json")) if q.is_file())
        else:
            out.append((p, True))
    return out


def read_artefact(path):
    """One file in the canonical vocabulary, or (None, why not).

    Split out of `inspect_file` because `--pair` needs the same three refusals (unreadable,
    unparseable, unrecognised) on both arms before it can compare anything, and a second copy of
    them would be a second set of error sentences to keep in step.
    """
    try:
        doc = read_json_bounded(path)
    except OSError as e:
        return None, f"could not read it: {e}"
    except json.JSONDecodeError as e:
        return None, f"not valid JSON: {e}"
    except ArtefactTooLargeError as e:
        # A third refusal beside the other two, in the same shape, because this path reads files
        # nobody here wrote. SECURITY.md puts a crafted result file in scope in those words.
        return None, f"this tool declines to read it: {e}"

    try:
        return normalise(doc), None
    except UnknownArtefactError as e:
        return None, str(e)


def inspect_file(path, checks):
    """Check one file. Returns (findings, skipped, problem).

    `problem` is a sentence when the file could not be checked at all, and is the outcome that
    must never be confused with a clean one.
    """
    normalised, problem = read_artefact(path)
    if problem is not None:
        return [], [], problem
    findings, skipped = run_checks(normalised, checks, artefact=str(path))
    return findings, skipped, None


def inspect_pair(path_a, path_b, checks):
    """Compare two artefacts. Returns (findings, skipped, problem).

    A pair with one unreadable arm is a PROBLEM rather than an empty result, for the same reason
    a single unreadable file is: half a comparison examined is not a comparison examined, and
    reporting it as no findings would be indistinguishable from the arms agreeing.
    """
    arms = []
    for path in (path_a, path_b):
        doc, problem = read_artefact(path)
        if problem is not None:
            return [], [], f"{path}: {problem}"
        arms.append(doc)
    findings, skipped = run_pair_checks(*arms, checks, artefact=f"{path_a} vs {path_b}")
    return findings, skipped, None


#: Where the text wraps. 76 leaves room for the 3-space indent inside an 80-column terminal.
_WIDTH = 76

#: The indent every wrapped section body sits at.
_BODY = "     "


def _section(label, text, out):
    """A labelled section: the label on its own line, the body wrapped under it, a blank line after.

    WHY THIS EXISTS, 2026-09-27

    Each section used to be printed as ONE `print` of the whole string on a single line. An output
    review measured a single finding at 5,435 characters across lines of 879, 1646, 1022 and 1458,
    with not one blank line in the output, which is 64 unbroken rows on an 80-column terminal. The
    hanging indent after `what it is:` was lost the moment the terminal soft-wrapped it, so the
    labels stopped marking anything.

    Nothing here is about the words. This is the whitespace, and on its own it turns four walls into
    four paragraphs.
    """
    import textwrap

    print(f"   {label}", file=out)
    for line in textwrap.wrap(" ".join(text.split()), width=_WIDTH,
                              initial_indent=_BODY, subsequent_indent=_BODY,
                              break_long_words=False, break_on_hyphens=False):
        print(line, file=out)
    print(file=out)


def _render(path, findings, skipped, problem, out, *, named=True, quiet=False, total=None):
    if problem:
        if named:
            print(f"?  {path}\n   UNCHECKED: {problem}", file=out)
        else:
            # Swept out of a directory rather than named, so the user never claimed it was a
            # result. Reported, because silence about a file that was read would be its own
            # small dishonesty, but not counted against the run.
            print(f"-  {path}\n   not a result artefact, skipped", file=out)
        return
    for f in findings:
        if quiet:
            # ONE LINE PER FINDING, which is what `--quiet` was advertised as and was not. It used
            # to drop only the two closing lines, 134 bytes of 5,435, leaving the whole wall in
            # place, while `gate --quiet` in the same tool meant what a reader expects.
            print(f"!  {path}  [{f.check_id}]  "
                  f"{_SEVERITY_SENTENCE.get(f.severity, f.severity)}", file=out)
            continue
        print(f"\n!  {path}", file=out)
        # The check id on its own line, and the title wrapped. Titles run past 100 characters and
        # used to trail the id off the right-hand edge, so the one string a reader needs in order to
        # look the check up was the one most likely to be wrapped away from the eye.
        print(f"   [{f.check_id}]", file=out)
        for line in __import__("textwrap").wrap(" ".join(f.title.split()), width=_WIDTH,
                                                initial_indent="   ", subsequent_indent="   "):
            print(line, file=out)
        # BOTH AXES, NAMED, because they are routinely read as one. Severity is what this costs
        # if the finding is right; confidence is how likely it is to be right. A high-confidence
        # note and a medium-confidence withdrawal are not the same news.
        # Two short lines rather than one 95-column line. Both axes still named, because they are
        # routinely read as one: severity is what this costs if the finding is right, confidence is
        # how likely it is to be right, and a high-confidence note is not a medium-confidence
        # withdrawal.
        print(f"   {_SEVERITY_SENTENCE.get(f.severity, f.severity)}", file=out)
        print(f"   {f.confidence} confidence this finding is right.", file=out)
        print(file=out)
        _section("What it is", f.detects, out)
        _section("What to do", f.remedy, out)
        # ALL FOUR SECTIONS, ALWAYS. An output review called the incident the least useful thing on
        # the screen, and a first attempt at this put it behind a flag on that advice.
        # `test_a_finding_carries_everything_needed_to_judge_it` refused, and its docstring says why
        # in terms: "the incident is the citation without which a finding is an opinion". It asserts
        # the date is QUOTED rather than referenced. That is a commitment from the v0.8 plan, not
        # decoration, and hiding it would have turned every finding back into an assertion the reader
        # has to take on faith.
        #
        # So the fix was never deletion. It was the WRAPPING above, which turns a 1,646-character
        # line into a paragraph, plus shortening the incident text itself where it had grown to
        # recount three separate fixes.
        _section("Seen before", f.incident, out)
        # A linter with a bad false-positive rate is uninstalled once and never again, so the reader
        # has to be able to judge this one without going and reading the source.
        _section("When this check is wrong", f.false_positive, out)
    if quiet:
        return
    if not findings and skipped:
        # COUNTED THE WAY A READER READS IT. This said "(13 checks did not apply)" beside a summary
        # line saying "16 checks available", and `--min-applied` then printed "3 of 13 applied",
        # reusing the count of checks that did NOT apply as the denominator of the ones that did.
        # Three numbers on one screen, one of them used two ways.
        applied = (total - len(skipped)) if total is not None else None
        detail = (f"{applied} of {total} checks applied" if applied is not None
                  else f"{len(skipped)} checks did not apply")
        print(f"✓  {path}\n   nothing found ({detail})", file=out)
    elif not findings:
        print(f"✓  {path}\n   nothing found", file=out)


def main(argv=None, out=None):
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    checks = load_checks()

    # A PATTERN CHOSE THESE PATHS, NOT A PERSON. pre-commit hands the hook whatever matched its
    # `files` regex, so the paths arrive named on the command line while carrying none of the
    # assertion that naming one usually carries. Without this the hook would block a commit over
    # a config file that happened to live in `results/`, and a hook that blocks wrongly is a
    # hook removed within the week.
    claimed = not args.skip_unknown

    files = _files(args.paths)
    n_pair_checks = sum(1 for c in checks if c.arity == "pair")

    results = []
    for path, was_named in files:
        named = was_named and claimed
        findings, skipped, problem = inspect_file(path, checks)
        results.append((path, findings, skipped, problem, named,
                        0 if problem else len(checks) - len(skipped)))

    # THE PAIR RUNS AFTER THE SINGLES, OVER THE SAME TWO FILES. Each arm is still checked on its
    # own, because a defect that is visible in one artefact is visible whether or not it is being
    # compared with another, and `--pair` adds the questions that need both rather than replacing
    # the ones that do not.
    if args.pair:
        if len(files) != 2 or not all(named for _, named in files):
            print("--pair needs exactly two result files, named on the command line. Which two "
                  "artefacts are arms of one comparison is a claim only you can make: sweeping a "
                  "directory and pairing everything in it would report findings about "
                  "comparisons nobody ran.", file=out)
            return 2
        pair_findings, pair_skipped, pair_problem = inspect_pair(
            files[0][0], files[1][0], checks)
        results.append((f"{files[0][0]} vs {files[1][0]}",
                        pair_findings, pair_skipped, pair_problem, True,
                        0 if pair_problem else n_pair_checks - len(pair_skipped)))

    if args.json:
        json.dump([
            {
                "artefact": str(p),
                "unchecked": problem if named else None,
                "not_a_result": problem if not named else None,
                "skipped": sk,
                "findings": [{
                    "check": f.check_id, "title": f.title, "confidence": f.confidence,
                    "severity": f.severity,
                    "detects": f.detects, "incident": f.incident, "remedy": f.remedy,
                    "false_positive": f.false_positive,
                } for f in fs],
            }
            for p, fs, sk, problem, named, applied in results
        ], out, indent=2)
        print(file=out)
    else:
        for path, findings, skipped, problem, named, _applied in results:
            if args.quiet and not findings and not (problem and named):
                continue
            _render(path, findings, skipped, problem, out, named=named,
                    quiet=args.quiet, total=len(checks))

    n_findings = sum(len(fs) for _, fs, _, _, _, _ in results)
    n_unchecked = sum(1 for _, _, _, problem, named, _ in results if problem and named)
    n_not_result = sum(1 for _, _, _, problem, named, _ in results if problem and not named)
    # WHAT WAS ACTUALLY EXAMINED, which is not the same as what was listed, and the difference
    # is the whole of this number. An entry is checked when it was read, recognised, and at least
    # one check applied to it. A file that was unreadable, unrecognised, or that every check
    # skipped has been listed and not checked, and `--fail-on-empty` used to read the listing:
    # with `--skip-unknown` beside it, a named non-result file made `results` non-empty and the
    # flag whose entire job is to catch "nothing happened" returned success having checked
    # nothing. That is the defect this command exists to find in other people's pipelines.
    n_checked = sum(1 for _, _, _, problem, _, applied in results if not problem and applied)

    if not args.json and not args.quiet:
        tail = f", {n_not_result} not a result" if n_not_result else ""
        # The pair is an entry in `results` and is NOT a file, so it is counted separately. A
        # summary reading "3 file(s)" after two were named would be a small lie of exactly the
        # kind this command exists to catch in other people's output.
        pair_tail = ", 1 pair" if args.pair else ""
        print(f"\n{len(files)} file(s){pair_tail}, {n_findings} finding(s), "
              f"{n_unchecked} unchecked{tail}, {len(checks)} checks available.", file=out)
        # SAID IN ITS OWN SENTENCE, because "0 file(s)" sits inside a line that otherwise reads
        # like a clean report, and a reader skims it as one. Found by pointing the checker at a
        # directory that existed and held nothing: it exited 0, which in CI is a green that
        # checked nothing.
        if not n_checked:
            print("NOTHING WAS CHECKED: no result artefacts were found at the path(s) given. "
                  "That is not the same as a clean result.", file=out)
        # LOOPHOLE 7, IN THE OUTPUT RATHER THAN THE README. Somebody will otherwise quote a
        # clean report as a claim of correctness, and it is not one.
        print("This looks for known failure modes. It cannot tell you a number is right.",
              file=out)

    if n_unchecked:
        return 2
    if n_findings:
        return 1
    # Nothing checked is only a failure when the caller says so: a sweep over a tree that
    # legitimately holds no artefacts yet is a normal thing to do, and breaking it would make the
    # default unusable. The flag is for the case where a path could drift, which is CI.
    #
    # IT READS `n_checked` AND NOT `results`, and that changed on 2026-09-21. Reading the listing
    # meant `--fail-on-empty --skip-unknown` exited 0 on a named non-result file: the file was
    # listed, so the flag never fired, while the help text promised to exit non-zero when nothing
    # was checked. A flag whose entire job is to catch "nothing happened" going green when
    # nothing happened is the defect class this project keeps finding in its own gates.
    # THE OTHER HALF OF --fail-on-empty, and the half the panel found missing. `n_checked`
    # counts an artefact where ANY check applied, so a change that stops fourteen of fifteen
    # checks recognising a file leaves this command reporting the same numbers and the same exit
    # code as a healthy run. The step in our own CI that runs the checker over our own published
    # evidence could not have failed for that reason, which makes it a gate that proves the
    # checker still runs rather than that it still checks.
    if args.min_applied:
        # THE DENOMINATOR IS WHAT COULD APPLY HERE, NOT EVERY CHECK THAT EXISTS.
        #
        # `arity: pair` checks read two arms and are skipped without evaluation on a single
        # document. Counting them made the ceiling unreachable: 4 of the 15 checks are pair
        # checks, so a single artefact can never exceed 11 applied, while this compared against
        # 15 and printed "of 15". `--min-applied 12` and above could not be satisfied by a
        # healthy run, which is the opposite of what a floor is for.
        reachable = [c for c in checks if getattr(c, "arity", "document") != "pair"]
        thin = sorted((path, applied) for path, _, _, problem, _, applied in results
                      if not problem and applied < min(args.min_applied, len(reachable)))
        if thin:
            if not args.json and not args.quiet:
                print(f"\nFEWER THAN {args.min_applied} CHECKS APPLIED to "
                      f"{len(thin)} artefact(s):", file=out)
                for path, applied in thin:
                    print(f"  {path}: {applied} of {len(reachable)} applied", file=out)
                print("Either these artefacts stopped carrying what the checks read, or the "
                      "checks stopped recognising them. Both look like a clean run.", file=out)
            return 1
    return 1 if (args.fail_on_empty and not n_checked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
