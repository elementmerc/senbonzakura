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

from . import adapters
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


class Unrecognised(str):
    """A refusal meaning "I read this and no adapter knows the shape", as opposed to "I could not
    read it at all".

    A `str` subclass so every caller that treats a problem as a sentence keeps working untouched;
    the type is carried alongside the words rather than in a second return value nobody would
    thread through `--pair`.

    WHY THE DIFFERENCE IS LOAD-BEARING, found 2026-09-28 by running the tool rather than reading
    it. Sweeping a directory is not a claim that everything in it is a result, so an UNRECOGNISED
    file swept from one costs nothing and the run stays green. That rule was applied to every
    refusal alike, including "not valid JSON", so a corrupt artefact exited 2 when named and 0 when
    it sat in a directory: the same bytes, two opposite verdicts. A CI step pointed at `results/`
    went green on a directory where every file was broken.

    Being unable to READ a file is a fact about the file. Nobody has to claim anything for it to be
    true, so it is reported as unchecked either way.
    """

    __slots__ = ()


def read_artefact(path):
    """One file in the canonical vocabulary, or (None, why not).

    Split out of `inspect_file` because `--pair` needs the same three refusals (unreadable,
    unparseable, unrecognised) on both arms before it can compare anything, and a second copy of
    them would be a second set of error sentences to keep in step.

    The unrecognised one is returned as `Unrecognised`, which reads as its own sentence everywhere
    and lets the sweep tell "not a result" from "not readable". See that class.
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
        return None, Unrecognised(str(e))


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


def _wrapped(text, out, *, indent="  "):
    """One paragraph of prose, wrapped to the same width the findings use.

    `_section` wraps a labelled body; this wraps a bare paragraph, which is what the summary block
    at the end of a run is made of. A URL or a long path is left whole rather than broken, because
    a wrapped path cannot be pasted.
    """
    import textwrap

    for line in textwrap.wrap(" ".join(text.split()), width=_WIDTH,
                              initial_indent=indent, subsequent_indent=indent,
                              break_long_words=False, break_on_hyphens=False):
        print(line, file=out)


#: The schema field this project stamps on an evidence file that is a COLLECTION of figures rather
#: than one measurement. Read rather than guessed: the value is what
#: `evidence/k-sweep-2026-08-13/drift-per-seed.json` actually carries.
OUR_EVIDENCE_SCHEMA = "senbonzakura-evidence/"


def _why_it_was_skipped(path):
    """What to say about a swept file no adapter claimed, which is not always the same sentence.

    Our own evidence collections get their own wording, because calling them "not a result
    artefact" is false and reads as a verdict on the file rather than on the question.
    """
    import json

    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        schema = doc.get("schema") if isinstance(doc, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        # Unreadable here is not a finding: the caller already read this file and decided no
        # adapter claimed it, so an error on a second read is about this sentence and nothing
        # else. The general wording is the honest fallback.
        schema = None
    if isinstance(schema, str) and schema.startswith(OUR_EVIDENCE_SCHEMA):
        return ("a senbonzakura evidence collection, not a single measurement, so no check can "
                "read it: skipped")
    return "not a result artefact, skipped"


def _render(path, findings, skipped, problem, out, *, named=True, quiet=False, total=None):
    if problem:
        if named:
            print(f"?  {path}\n   UNCHECKED: {problem}", file=out)
        else:
            # Swept out of a directory rather than named, so the user never claimed it was a
            # result. Reported, because silence about a file that was read would be its own
            # small dishonesty, but not counted against the run.
            #
            # OUR OWN EVIDENCE IS NOT "NOT A RESULT ARTEFACT", 2026-10-01. Pointed at this
            # project's own `evidence/`, the checker said that of
            # `k-sweep-2026-08-13/drift-per-seed.json`, which declares `schema:
            # senbonzakura-evidence/1` in its first field. Skipping it is right: it holds
            # per-seed lists for two arms (`drift_kl.k1`, `drift_kl.k2`) rather than one
            # measurement with an estimator, so there is no single figure a check can read. The
            # WORDS were wrong, and wrong in the direction that matters, because they say a file
            # carrying our schema is foreign to us. A reader checking our own published evidence
            # met that sentence and had no way to tell a collection from a stranger's file.
            print(f"-  {path}\n   {_why_it_was_skipped(path)}", file=out)
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


#: How often a sweep says it is still alive, in seconds. Baseline section 2.1 asks for one every
#: 30 to 60 in any long-running loop, and this is the bottom of that range because the loop's unit
#: of work is one file: a sweep that has gone quiet for half a minute has either stalled on one
#: enormous artefact or wedged, and both are worth hearing about promptly.
HEARTBEAT_SECONDS = 30


def _entry(path, findings, skipped, problem, named, applied):
    """One file's object in the machine-readable report.

    Pulled out of the `--json` branch when the report started streaming: the shape has to be
    identical whether it is written during the loop or after it, and the only way to be sure of
    that is for there to be one copy of it.
    """
    return {
        "artefact": str(path),
        "unchecked": problem if named else None,
        "not_a_result": problem if not named else None,
        # HOW MANY CHECKS ACTUALLY RAN ON THIS FILE, which is the difference between a file that
        # was examined and one that was merely opened. It was computed from the start and dropped
        # on the floor, so every consumer had to reconstruct "was this checked" from `unchecked`
        # and `not_a_result` alone, and a file that parsed with every check skipped came out
        # looking checked. The Action did exactly that and its `fail-on-empty` could not fire.
        "applied": applied,
        "skipped": skipped,
        "findings": [{
            "check": f.check_id, "title": f.title, "confidence": f.confidence,
            "severity": f.severity,
            "detects": f.detects, "incident": f.incident, "remedy": f.remedy,
            "false_positive": f.false_positive,
        } for f in findings],
    }


class _Heartbeat:
    """Say that a long sweep is still alive, on stderr, at a stated interval.

    ON STDERR, AND THIS IS THE WHOLE DESIGN. `--json` on stdout is read by `action.yml` and by the
    pre-commit entry point, and a progress line in that stream turns a clean run into a parse
    error on somebody else's CI. The project has the opposite scar too: a quiet mode once
    suppressed warnings because a recorded decision's premise about which stream carried what was
    false. So the separation is asserted by a test rather than trusted to this comment.

    THE CLOCK IS INJECTABLE because a test that waits thirty seconds to find out whether a
    heartbeat fires is a test nobody runs, and a heartbeat nobody has watched fire is the same
    kind of nothing as a gate nobody has watched fail.
    """

    def __init__(self, total, *, stream=None, clock=None, every=HEARTBEAT_SECONDS):
        import time
        self._clock = clock or time.monotonic
        # STDERR IS RESOLVED AT WRITE TIME, not here, so a caller who redirects the stream after
        # the sweep has started still gets the heartbeat on the stream in effect when it writes.
        #
        # THE REASON FIRST RECORDED HERE WAS WRONG and a surviving mutant found it, 2026-10-02. It
        # said construction-time resolution breaks under "every test harness", which is false:
        # pytest's capture replaces `sys.stderr` before a test body runs, so resolving at
        # construction picks up the already-replaced stream and lands in the right place. That is
        # why reverting this line left the checker's own tests green. What it really protects is a
        # redirect entered BETWEEN construction and the write, which no caller in this tree does
        # today, which is exactly why the guard has to construct that sequence deliberately. See
        # `test_a_redirect_after_construction_still_reaches_the_new_stream`.
        self._stream = stream
        self._every = every
        self._total = total
        self._last = self._clock()
        self._done = 0

    def tick(self, path):
        """One unit of work finished. Emits only when the interval has elapsed."""
        self._done += 1
        now = self._clock()
        if now - self._last < self._every:
            return False
        self._last = now
        print(f"... {self._done} of {self._total} checked, now at {path}",
              file=self._stream if self._stream is not None else sys.stderr, flush=True)
        return True


def main(argv=None, out=None, clock=None):
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

    # THE PAIR'S ARITY IS CHECKED BEFORE ANYTHING IS READ, and it used to be checked after the
    # whole sweep. Moved up when the report started streaming, because a refusal that arrives
    # after half the output has been written is a refusal nobody can act on, and reading every
    # file before saying "this needed exactly two" was work spent to reach a conclusion that was
    # available from the arguments alone. Pre-flight, per baseline section 2.1.
    if args.pair and (len(files) != 2 or not all(named for _, named in files)):
        print("--pair needs exactly two result files, named on the command line. Which two "
              "artefacts are arms of one comparison is a claim only you can make: sweeping a "
              "directory and pairing everything in it would report findings about "
              "comparisons nobody ran.", file=out)
        return 2

    # ── the report is written as the sweep goes, not accumulated and rendered afterwards ────────
    #
    # WHY, per baseline section 12.1 and this project's own scale note: one order of magnitude
    # past today is a repository of a few thousand result artefacts, or a CI run over a monorepo
    # of them, and a run over those that dies at four thousand should still have told you about
    # the first four thousand. It used to build a list of every file's result and render after the
    # loop, so a killed run reported nothing at all.
    #
    # WHAT IS STILL ACCUMULATED, AND WHY IT IS BOUNDED. Four integer counters, and the artefacts
    # that fell below `--min-applied`, which in a healthy run is an empty list. Nothing here grows
    # with the number of files that were fine.
    #
    # THE EXIT STATUS IS STILL DECIDED AT THE END. Loophole 10: `--pair` runs after the singles,
    # and one unreadable file in the last artefact changes the status of everything before it, so
    # streaming the OUTPUT must not mean deciding the STATUS early. A partial report carries no
    # footer and no status line, which is what tells a reader it is partial.
    n_findings = n_unchecked = n_not_result = n_checked = 0
    thin = []
    reachable = [c for c in checks if getattr(c, "arity", "document") != "pair"]
    heartbeat = _Heartbeat(len(files) + (1 if args.pair else 0), clock=clock)
    first_json_entry = True

    def emit(path, findings, skipped, problem, named, applied):
        """Write one entry, in whichever of the two formats was asked for."""
        nonlocal first_json_entry
        if args.json:
            # STREAMED AS A JSON ARRAY, one element at a time, so the format a consumer parses is
            # unchanged. A run killed partway leaves the array unterminated, which `json.load`
            # refuses, and a report that cannot be parsed is reported by `action.yml` as "nothing
            # about this run is known" rather than as a clean sweep. That is the correct reading
            # of a partial file and the reason this is an array rather than one object per line.
            print("" if first_json_entry else ",", file=out)
            json.dump(_entry(path, findings, skipped, problem, named, applied), out, indent=2)
            first_json_entry = False
        elif not (args.quiet and not findings and not (problem and named)):
            _render(path, findings, skipped, problem, out, named=named,
                    quiet=args.quiet, total=len(checks))

    def tally(findings, problem, named, applied):
        nonlocal n_findings, n_unchecked, n_not_result, n_checked
        n_findings += len(findings)
        if problem and named:
            n_unchecked += 1
        elif problem:
            n_not_result += 1
        elif applied:
            # WHAT WAS ACTUALLY EXAMINED, which is not the same as what was listed. An entry is
            # checked when it was read, recognised, and at least one check applied to it. A file
            # that was unreadable, unrecognised, or that every check skipped has been listed and
            # not checked, and `--fail-on-empty` used to read the listing.
            n_checked += 1

    if args.json:
        print("[", end="", file=out)

    for path, was_named in files:
        named = was_named and claimed
        findings, skipped, problem = inspect_file(path, checks)
        if not was_named and problem is not None and not isinstance(problem, Unrecognised):
            # UNREADABLE IS NOT A MATTER OF WHO CLAIMED WHAT. The sweep discount exists for a file
            # that is simply not a result; a file that could not be parsed at all is broken whether
            # or not anybody named it, and discounting it let a directory of corrupt artefacts
            # report a clean sweep and exit 0.
            #
            # SCOPED TO THE SWEEP, and `--skip-unknown` is deliberately left alone. That flag is a
            # different statement: pre-commit picked these paths with a regex, so NONE of them is a
            # person's claim, and its own help says it treats a named file "the way a swept one" is
            # treated. Widening it here would change what blocks somebody's commit, which is a
            # decision rather than a bug fix. The open question it leaves, whether a staged file
            # that is corrupt should block, is in DEFERRED.md.
            named = True
        applied = 0 if problem else len(checks) - len(skipped)
        emit(path, findings, skipped, problem, named, applied)
        tally(findings, problem, named, applied)
        if not problem and applied < min(args.min_applied or 0, len(reachable)):
            thin.append((path, applied))
        heartbeat.tick(path)

    # THE PAIR RUNS AFTER THE SINGLES, OVER THE SAME TWO FILES. Each arm is still checked on its
    # own, because a defect that is visible in one artefact is visible whether or not it is being
    # compared with another, and `--pair` adds the questions that need both rather than replacing
    # the ones that do not.
    if args.pair:
        pair_findings, pair_skipped, pair_problem = inspect_pair(
            files[0][0], files[1][0], checks)
        pair_applied = 0 if pair_problem else n_pair_checks - len(pair_skipped)
        pair_label = f"{files[0][0]} vs {files[1][0]}"
        emit(pair_label, pair_findings, pair_skipped, pair_problem, True, pair_applied)
        tally(pair_findings, pair_problem, True, pair_applied)
        heartbeat.tick(pair_label)

    if args.json:
        # THE ARRAY IS CLOSED HERE AND NOWHERE ELSE, which is what makes a killed run detectable:
        # the bracket is the last thing written, so a report that has one is a report that
        # finished.
        print("\n]" if not first_json_entry else "]", file=out)

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
            # WRAPPED, 2026-10-01. Driven at COLUMNS=80 this block printed lines of 113 and 123
            # columns, so the paragraph a reader meets when nothing was checked was the one
            # paragraph the terminal reflowed into a wall. `_section` above already owns the width
            # for findings; this block had been written with bare `print` calls and never passed
            # through it.
            _wrapped("NOTHING WAS CHECKED: no result artefacts were found at the path(s) given. "
                     "That is not the same as a clean result.", out, indent="")
            # AND WHICH OF THE TWO OUTCOMES THIS IS, added 2026-09-27. The sentence above was
            # printed identically whether the command was about to exit 0 or non-zero, so a reader
            # could not tell from the output whether their CI step had just failed, and the reader
            # who exits 0 is not told that the behaviour they almost certainly want is one flag
            # away. An empty sweep is legitimately fine from a shell and almost never fine in CI.
            # AND WHAT IT WOULD HAVE ACCEPTED. A file named on the command line is told this by
            # `UnknownArtefactError`; a SWEPT file gets "not a result artefact, skipped" and the
            # summary never says it either, so the one run where the reader has nothing else to go
            # on was the run that named no kinds. Read off the registry rather than typed here, so
            # a fourth adapter cannot leave this sentence stale.
            _wrapped("Recognised kinds: "
                     + ", ".join(a.name for a in adapters.ADAPTERS)
                     + ". Point it at one of those, or at a directory holding them.", out)
            if args.fail_on_empty:
                _wrapped("Exiting non-zero because --fail-on-empty was given.", out)
            elif not n_unchecked:
                _wrapped("This run exits 0. Pass --fail-on-empty to make an empty sweep a "
                         "failure, which is what a CI step usually wants.", out)
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
        #
        # SORTED AT THE BOUNDARY, not in the loop that filled it, so two runs over the same tree
        # print the same list whatever order the filesystem handed the files over in.
        if thin:
            if not args.json and not args.quiet:
                print(f"\nFEWER THAN {args.min_applied} CHECKS APPLIED to "
                      f"{len(thin)} artefact(s):", file=out)
                for path, applied in sorted(thin):
                    print(f"  {path}: {applied} of {len(reachable)} applied", file=out)
                print("Either these artefacts stopped carrying what the checks read, or the "
                      "checks stopped recognising them. Both look like a clean run.", file=out)
            return 1
    return 1 if (args.fail_on_empty and not n_checked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
