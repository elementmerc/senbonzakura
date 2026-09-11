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

EXIT CODES, and they distinguish the three outcomes on purpose, because a CI gate that treats
"could not check" the same as "checked, nothing found" is worse than no gate:

    0  every file was read and nothing fired
    1  at least one finding
    2  at least one file could not be read at all
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
    """Every file named, with directories expanded, in a stable order.

    Sorted because two runs over the same tree must produce the same report, per baseline
    section 2.1. A report whose order depends on the filesystem cannot be diffed between runs.
    """
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.json") if q.is_file()))
        else:
            out.append(p)
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


def _render(path, findings, skipped, problem, out):
    if problem:
        print(f"?  {path}\n   UNCHECKED: {problem}", file=out)
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
    for path in _files(args.paths):
        findings, skipped, problem = inspect_file(path, checks)
        results.append((path, findings, skipped, problem))

    if args.json:
        json.dump([
            {
                "artefact": str(p),
                "unchecked": problem,
                "skipped": sk,
                "findings": [{
                    "check": f.check_id, "title": f.title, "confidence": f.confidence,
                    "detects": f.detects, "incident": f.incident, "remedy": f.remedy,
                    "false_positive": f.false_positive,
                } for f in fs],
            }
            for p, fs, sk, problem in results
        ], out, indent=2)
        print(file=out)
    else:
        for path, findings, skipped, problem in results:
            if args.quiet and not findings and not problem:
                continue
            _render(path, findings, skipped, problem, out)

    n_findings = sum(len(fs) for _, fs, _, _ in results)
    n_unchecked = sum(1 for _, _, _, problem in results if problem)

    if not args.json and not args.quiet:
        print(f"\n{len(results)} file(s), {n_findings} finding(s), "
              f"{n_unchecked} unchecked, {len(checks)} checks available.", file=out)
        # LOOPHOLE 7, IN THE OUTPUT RATHER THAN THE README. Somebody will otherwise quote a
        # clean report as a claim of correctness, and it is not one.
        print("This looks for known failure modes. It cannot tell you a number is right.",
              file=out)

    if n_unchecked:
        return 2
    return 1 if n_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
