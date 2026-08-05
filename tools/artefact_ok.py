#!/usr/bin/env python3
"""Is this result file the one the caller is about to produce, or merely A result file?

Every resume guard in this project's run specs asked the same question, and asked it wrong:

    if python -c "import json; json.load(open('out.json'))"; then echo ALREADY_COMPLETE; fi

That tests whether a file PARSES. Yesterday's files parse perfectly. On 2026-08-04 a run whose
entire purpose was to measure three code changes reported "7 done, 0 failed" while every job
printed ALREADY_COMPLETE, because the outputs from the night before were valid JSON produced by a
different configuration. A guard that cannot tell those apart does not protect a resume; it
silently converts one into a no-op that reports success.

So the guard states what it expects, and the file has to agree:

    python tools/artefact_ok.py out.json model=Qwen/Qwen3-1.7B seed=42 init=mean-diff

Exit 0 means "this file was produced by that configuration, skip the work". Any mismatch, missing
key or unreadable file exits non-zero and says which key disagreed, so the job re-runs and the log
records why. Keys are looked up through nested dicts by dotted path, and compared as strings so a
spec never has to care whether a value was written as 42 or "42".

The point is not the hashing. It is that the expectation is written down next to the command it
guards, where a reader can check it against the flags on the line below.
"""
import argparse
import json
import sys

MISSING = object()


def lookup(doc, path):
    """Fetch a dotted path out of nested dicts, or MISSING.

    Also searches one level into a top-level `meta`/`config` container, because the two writers in
    this repo disagree about whether configuration lives at the root or under a key, and a guard
    that only worked for one of them would be a guard nobody used for the other.
    """
    for prefix in ([], ["meta"], ["config"]):
        cur = doc
        for part in prefix + path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = MISSING
                break
            cur = cur[part]
        if cur is not MISSING:
            return cur
    return MISSING


def mismatches(doc, expectations):
    """Every expectation the document fails, as human-readable lines. Empty means it matches."""
    bad = []
    for key, want in expectations:
        got = lookup(doc, key)
        if got is MISSING:
            bad.append(f"{key}: the file does not record it, so it cannot be shown to match")
        elif str(got) != str(want):
            bad.append(f"{key}: file says {got!r}, this run wants {want!r}")
    return bad


def parse_expectations(pairs):
    out = []
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"expected key=value, got {p!r}")
        key, _, value = p.partition("=")
        if not key:
            raise ValueError(f"empty key in {p!r}")
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
