# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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

from .adapters import UnknownArtefactError, normalise
from .registry import load_checks, run_checks


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura check",
        description="Read an evaluation result file and report how the number could be wrong.")
    ap.add_argument("paths", nargs="+",
                    help="result files, or directories to search for .json files")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output, one object per file")
    ap.add_argument("--quiet", action="store_true",
                    help="print findings only, no summary and no reassurance")
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


def inspect_file(path, checks):
    """Check one file. Returns (findings, skipped, problem).

    `problem` is a sentence when the file could not be checked at all, and is the outcome that
    must never be confused with a clean one.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        return [], [], f"could not read it: {e}"
    except json.JSONDecodeError as e:
        return [], [], f"not valid JSON: {e}"

    try:
        normalised = normalise(doc)
    except UnknownArtefactError as e:
        return [], [], str(e)

    findings, skipped = run_checks(normalised, checks, artefact=str(path))
    return findings, skipped, None


def _render(path, findings, skipped, problem, out, *, named=True):
    if problem:
        if named:
            print(f"?  {path}\n   UNCHECKED: {problem}", file=out)
        else:
            # Swept out of a directory rather than named, so the user never claimed it was a
            # result. Reported, because silence about a file that was read would be its own
            # small dishonesty, but not counted against the run.
            print(f"-  {path}: not a result artefact, skipped", file=out)
        return
    for f in findings:
        print(f"\n!  {path}", file=out)
        print(f"   {f.title}  [{f.check_id}, {f.confidence} confidence]", file=out)
        print(f"   what it is:   {f.detects}", file=out)
        print(f"   seen before:  {f.incident}", file=out)
        print(f"   what to do:   {f.remedy}", file=out)
        # PRINTED WITH THE FINDING, not kept in the documentation. A linter with a bad
        # false-positive rate is uninstalled once and never again, so the reader has to be able
        # to judge this one without going and reading the source.
        print(f"   when this check is wrong: {f.false_positive}", file=out)
    if not findings and skipped:
        print(f"   {path}: nothing found ({len(skipped)} checks did not apply)", file=out)
    elif not findings:
        print(f"   {path}: nothing found", file=out)


def main(argv=None, out=None):
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    checks = load_checks()

    results = []
    for path, named in _files(args.paths):
        findings, skipped, problem = inspect_file(path, checks)
        results.append((path, findings, skipped, problem, named))

    if args.json:
        json.dump([
            {
                "artefact": str(p),
                "unchecked": problem if named else None,
                "not_a_result": problem if not named else None,
                "skipped": sk,
                "findings": [{
                    "check": f.check_id, "title": f.title, "confidence": f.confidence,
                    "detects": f.detects, "incident": f.incident, "remedy": f.remedy,
                    "false_positive": f.false_positive,
                } for f in fs],
            }
            for p, fs, sk, problem, named in results
        ], out, indent=2)
        print(file=out)
    else:
        for path, findings, skipped, problem, named in results:
            if args.quiet and not findings and not (problem and named):
                continue
            _render(path, findings, skipped, problem, out, named=named)

    n_findings = sum(len(fs) for _, fs, _, _, _ in results)
    n_unchecked = sum(1 for _, _, _, problem, named in results if problem and named)
    n_not_result = sum(1 for _, _, _, problem, named in results if problem and not named)

    if not args.json and not args.quiet:
        tail = f", {n_not_result} not a result" if n_not_result else ""
        print(f"\n{len(results)} file(s), {n_findings} finding(s), "
              f"{n_unchecked} unchecked{tail}, {len(checks)} checks available.", file=out)
        # LOOPHOLE 7, IN THE OUTPUT RATHER THAN THE README. Somebody will otherwise quote a
        # clean report as a claim of correctness, and it is not one.
        print("This looks for known failure modes. It cannot tell you a number is right.",
              file=out)

    if n_unchecked:
        return 2
    return 1 if n_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
