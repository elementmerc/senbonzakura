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

    Strictly one container: the root. An earlier version also retried the whole path under a
    top-level `meta` or `config` key, on the theory that this repo's writers disagree about where
    configuration lives. They do not: `tools/rdo.py` and `src/senbonzakura/validate.py` both write
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
