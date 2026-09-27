#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Is this result file the one the caller is about to produce, or merely A result file?

Every resume guard in this project's run specs asked the same question, and asked it wrong:

    if python -c "import json; json.load(open('out.json'))"; then echo ALREADY_COMPLETE; fi

That tests whether a file PARSES. Yesterday's files parse perfectly. On 2026-08-04 a run whose
entire purpose was to measure three code changes reported "7 done, 0 failed" while every job
printed ALREADY_COMPLETE, because the outputs from the night before were valid JSON produced by a
different configuration. A guard that cannot tell those apart does not protect a resume; it
silently converts one into a no-op that reports success.

So the guard states what it expects, and the file has to agree:

    python tools/ci/artefact_ok.py out.json model=Qwen/Qwen3-1.7B seed=42 init=mean-diff

Exit 0 means "this file was produced by that configuration, skip the work". Any mismatch, missing
key or unreadable file exits non-zero and says which key disagreed, so the job re-runs and the log
records why. Keys are looked up through nested dicts by dotted path, and compared as strings so a
spec never has to care whether a value was written as 42 or "42".

The point is not the hashing. It is that the expectation is written down next to the command it
guards, where a reader can check it against the flags on the line below.

NEVER ASSERT THAT A PROVENANCE FIELD IS ABSENT OR NULL. Written down 2026-09-27, before it cost
anything, after a blast-radius pass found the trap while a related fix was being held for it.

Until that day every artefact produced by an installed wheel carried
`provenance.senbonzakura.git = null`, because the commit stamp the release gate insists on was
generated, shipped, and read by nothing. A spec saying `provenance.senbonzakura.git.commit=null`
would have passed every time and looked like a working guard. The moment the stamp started being
read, that spec became unsatisfiable for ever: `scripts/runpod/seed-sweep-bootstrap.sh` decides with
one of these whether an arm is finished, so a completed arm would have read as stale and spent its
GPU hours again, on a pod, silently, for as long as nobody looked.

The general form is that an absence records what a machine could not tell us. That is a fact about
the environment rather than about the result, and it is exactly the sort of thing that legitimately
improves later. Assert on what a result CONTAINS, never on what it lacks.
"""
import argparse
import json
import sys

MISSING = object()


def lookup(doc, path):
    """Fetch a dotted path out of nested dicts, or MISSING.

    Strictly one container: the root. An earlier version also retried the whole path under a
    top-level `meta` or `config` key, on the theory that this repo's writers disagree about where
    configuration lives. They do not: `tools/research/rdo.py` and `src/senbonzakura/validate.py` both write
    config at the root, and no writer in the repo emits either container. So the fallback bought
    nothing and cost a false-match surface, because a path that died PART WAY through the real
    container would silently retry elsewhere and could be satisfied by a value the authoritative
    container never held. A guard that can answer from the wrong place is worse than no guard.
    """
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def matches(got, want):
    """Does a recorded value satisfy an expectation string?

    JSON null is spelled `null` and matches ONLY an actual null. Comparing `str(got)` alone made
    a recorded null and the recorded STRING "None" indistinguishable, which matters because
    `degenerate_reason=null` ("the grid was readable") is one of the guards most worth writing,
    and a file recording the literal text "None" would have satisfied it.
    """
    if want == "null":
        return got is None
    if got is None:
        return False
    if isinstance(got, bool):
        # JSON spells these true/false; Python str() spells them True/False. Accept both rather
        # than fail a spec author for writing the JSON they are looking at.
        return want.lower() == str(got).lower()
    return str(got) == want


def mismatches(doc, expectations):
    """Every expectation the document fails, as human-readable lines. Empty means it matches."""
    bad = []
    for key, want in expectations:
        got = lookup(doc, key)
        if got is MISSING:
            bad.append(f"{key}: the file does not record it, so it cannot be shown to match")
        elif not matches(got, want):
            bad.append(f"{key}: file says {got!r}, this run wants {want!r}")
    return bad


#: Spellings of "this field has no value". See the module docstring for what the trap cost nobody,
#: and would have cost a pod full of GPU hours.
AN_ABSENCE = frozenset({"null", "none", "nil", ""})

#: Where an absence means "nothing here could tell us", which is the dangerous kind: it is a fact
#: about the machine rather than about the run, and it is what changes when a gap gets filled in.
#:
#: SCOPED, AND THE FIRST VERSION OF THIS WAS NOT. It refused every absence assertion, and the live
#: spec in `scripts/runpod/seed-sweep-bootstrap.sh` includes `directions_from=None`, which is
#: legitimate and permanent: the baseline arm is DEFINED by having no directions file, so that field
#: will never acquire a value and the assertion cannot rot. Refusing it would have made a finished
#: arm re-run on every check, which is precisely the cost this guard exists to prevent, arriving by
#: the guard itself. A check answering a broader question than the one being asked.
WHERE_AN_ABSENCE_ROTS = ("provenance.",)


def parse_expectations(pairs):
    out = []
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"expected key=value, got {p!r}")
        key, _, value = p.partition("=")
        if not key:
            raise ValueError(f"empty key in {p!r}")
        # A SPEC THAT ASSERTS AN ABSENCE IS REFUSED, not merely discouraged in prose above. It
        # passes for as long as the tool cannot fill the field in and becomes unsatisfiable for ever
        # the day it can, which turns a finished arm into a stale one and re-runs the work. The
        # failure is silent and expensive and arrives long after the spec was written, so the guard
        # is here rather than in a comment the author has already scrolled past.
        if (value.strip().lower() in AN_ABSENCE
                and key.startswith(WHERE_AN_ABSENCE_ROTS)):
            raise ValueError(
                f"{p!r} expects {key} to be absent, and that is a provenance field. An absence "
                f"there records what the machine could not tell us, which is a fact about the "
                f"environment rather than about the run, and it is exactly what changes when a gap "
                f"gets filled in. The spec would pass until then and be unsatisfiable afterwards, "
                f"so a finished run would read as stale and be done again, on a pod, silently. "
                f"Assert on what the artefact contains.")
        out.append((key, value))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("expect", nargs="*", metavar="key=value")
    a = ap.parse_args(argv)

    try:
        expectations = parse_expectations(a.expect)
    except ValueError as e:
        print(f"artefact_ok: {e}", file=sys.stderr)
        return 2

    try:
        with open(a.file, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        print(f"artefact_ok: {a.file} does not exist, so there is nothing to reuse")
        return 1
    except (OSError, ValueError) as e:
        print(f"artefact_ok: {a.file} is unreadable ({e}), so it will be rebuilt")
        return 1

    # An expectation-free call is the old parses-therefore-done guard, and it is refused rather
    # than quietly allowed: that is precisely the check this file exists to replace.
    if not expectations:
        print("artefact_ok: no expectations given. A file that merely parses proves nothing "
              "about which configuration produced it; name the keys that must match.",
              file=sys.stderr)
        return 2

    if not isinstance(doc, dict):
        # A bare JSON array (cli.py writes trials.json that way) records no configuration at all,
        # so no expectation can ever be shown to hold. Say that, rather than reporting every key
        # as individually missing and leaving the reader to work out why.
        print(f"artefact_ok: {a.file} is a {type(doc).__name__}, not an object, so it records no "
              f"configuration to check expectations against; it will be rebuilt")
        return 1

    bad = mismatches(doc, expectations)
    if bad:
        print(f"artefact_ok: {a.file} exists but was produced by a different configuration, "
              f"so it will be rebuilt rather than reused:")
        for line in bad:
            print(f"  {line}")
        return 1

    print(f"artefact_ok: {a.file} matches all {len(expectations)} expectations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
