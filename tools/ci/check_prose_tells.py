#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Scan prose files for the writing tells, including files no commit hook can ever see.

WHY THIS EXISTS

The `pre-commit` hook catches these patterns on staged additions to public docs, which covers
everything in the repository and nothing outside it. The copy that goes out under the operator's
name lives in `private/`, which is excluded through `.git/info/exclude` because this repository has
a public remote. So it is never staged, no hook can ever read it, and **the one body of writing
where register matters most was the one body with nothing watching it.**

A checker for it was written during a session on 2026-10-06, found nothing after a voice pass, and
lived in a scratchpad where it was lost. This is that checker, in a place it survives from.

WHAT IT IS AND IS NOT

It is the hook's own pattern list, applied to whole files instead of to staged additions. It is not
a second opinion and it is not a fork: **the patterns are read out of the hook at run time.** Two
copies of this list would drift, and the hook's own history is a record of patterns being added one
at a time as real examples slipped through it, so a stale copy here would be a checker that passes
text the hook would stop.

That history is also why the list is worth reusing rather than rewriting. One pattern exists only
because a negative parallelism survived a hand review AND a purpose-written scanner, because both
looked for the literal "not just" and the sentence said "isn't just".

WHAT IT DELIBERATELY DOES NOT KNOW

Nothing about outreach, prospects, or which directories hold private material. It takes paths and
reports findings. A public file naming the private tree it inspects is a signpost to that tree, and
the whole reason the private tree is excluded rather than gitignored is that a name in a committed
file tells every visitor it exists. The caller supplies the paths; see `private/` for the wrapper
that does.

USAGE

    python3 tools/ci/check_prose_tells.py docs/guide/*.md
    python3 tools/ci/check_prose_tells.py --quiet somefile.md && echo clean

Exit 0 when clean, 1 when something matched, 2 when it could not do its job at all. The third is
the one that matters: a checker that cannot read its pattern list must fail loudly rather than
report a clean scan.
"""
import argparse
import pathlib
import re
import sys

#: The hook that owns the patterns. Found relative to this file so a clone works anywhere, and in
#: the project copy rather than the canonical home, because the project copy is the one whose
#: behaviour a contributor here is subject to.
HOOK = pathlib.Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"

#: The bash array to lift them out of.
ARRAY = "LLM_TELL_PATTERNS=("

#: Fewer than this many patterns means the parse went wrong rather than that the hook got shorter.
#: It is a floor and not an equality on purpose: a new pattern should not break this tool, and a
#: parse that silently returns two should.
MIN_PATTERNS = 8

#: POSIX character classes bash understands and Python's `re` does not.
POSIX_CLASSES = {
    "[[:space:]]": r"\s",
    "[[:alpha:]]": r"[A-Za-z]",
    "[[:alnum:]]": r"[A-Za-z0-9]",
    "[[:digit:]]": r"\d",
    "[[:upper:]]": r"[A-Z]",
    "[[:lower:]]": r"[a-z]",
    "[[:punct:]]": r"[^\w\s]",
}


class PatternsUnreadableError(RuntimeError):
    """The hook's list could not be read, so no scan can be trusted.

    UNREADABLE IS NOT CLEAN. A checker that cannot find its patterns and reports no findings has
    told the caller their prose is fine, which is the most expensive possible way to be wrong about
    a gate. This project has shipped that shape before, in a test run that collected nothing and
    read as a pass.
    """


def translate(pattern):
    """A bash ERE from the hook, as a Python regex.

    Only the POSIX classes need translating. The rest of extended regular expression syntax is
    shared, which is why this is a substitution table rather than a parser.
    """
    for posix, py in POSIX_CLASSES.items():
        pattern = pattern.replace(posix, py)
    return pattern


def read_patterns(hook=HOOK):
    """The tell patterns, lifted out of the hook's bash array.

    Parsed rather than duplicated, and the parse is strict: if the array cannot be found, or yields
    implausibly few entries, or any entry fails to compile, this raises. A hook whose formatting
    changes should break this tool noisily at the next run, where somebody is looking, rather than
    quietly reduce it to a checker that finds nothing.
    """
    try:
        text = hook.read_text(encoding="utf-8")
    except OSError as e:
        raise PatternsUnreadableError(
            f"cannot read the pattern list from {hook}: {e}. No scan was run, because a scan "
            f"without the patterns would report every file as clean.") from e

    start = text.find(ARRAY)
    if start < 0:
        raise PatternsUnreadableError(
            f"{hook} has no `{ARRAY}` array. The hook's format has changed and this tool reads "
            f"its list rather than keeping a copy, so it cannot run until it is updated. It "
            f"refuses rather than falling back to a built-in list, because a stale built-in list "
            f"is a checker that passes text the hook would stop.")
    body = text[start + len(ARRAY):]
    end = body.find("\n    )")
    if end < 0:
        raise PatternsUnreadableError(
            f"the `{ARRAY}` array in {hook} is not closed where this expects. Refusing rather "
            f"than guessing where it ends.")
    body = body[:end]

    raw = re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*$', body, re.MULTILINE)
    if len(raw) < MIN_PATTERNS:
        raise PatternsUnreadableError(
            f"only {len(raw)} pattern(s) parsed out of {hook}, below the floor of "
            f"{MIN_PATTERNS}. That is a parse failure rather than a short list, and reporting a "
            f"clean scan from it would be a lie.")

    out = []
    for pattern in raw:
        translated = translate(pattern)
        try:
            out.append((pattern, re.compile(translated, re.IGNORECASE)))
        except re.error as e:
            raise PatternsUnreadableError(
                f"pattern {pattern!r} from {hook} does not compile as a Python regex after "
                f"translation ({e}). Refusing rather than skipping it, because a silently skipped "
                f"pattern is a tell this tool would stop catching without saying so.") from e
    return out


def read_allowlist(path):
    """Lines a project has decided are legitimate, from `.baseline-hook-allow`.

    Same file the hook consults, so an override recorded once holds in both places. Comments and
    blank lines are skipped; everything else is matched as a substring, because the file records
    phrases rather than patterns.
    """
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def scan_text(text, patterns, allow=()):
    """Every (line number, line, the pattern's own source) a file matches."""
    found = []
    for n, line in enumerate(text.splitlines(), start=1):
        if any(a in line for a in allow):
            continue
        for source, rx in patterns:
            m = rx.search(line)
            if m:
                found.append((n, line.strip(), source, m.group(0)))
                # One finding per line. A line matching three patterns is one sentence to rewrite,
                # and three entries for it would make a short list look like a long one.
                break
    return found


def scan_paths(paths, patterns, allow=()):
    """(findings, unreadable) over every path, where findings are (path, line, text, pattern, hit)."""
    findings, unreadable = [], []
    for p in paths:
        path = pathlib.Path(p)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # NAMED, NOT SKIPPED. A file this could not open is a file it cannot vouch for, and a
            # silent skip turns an unreadable file into a clean one.
            unreadable.append((str(path), str(e)))
            continue
        findings.extend((str(path), *row) for row in scan_text(text, patterns, allow))
    return findings, unreadable


def build_parser():
    ap = argparse.ArgumentParser(
        prog="check_prose_tells.py",
        description="Scan prose files for the writing tells, using the pre-commit hook's own "
                    "pattern list rather than a copy of it.")
    ap.add_argument("paths", nargs="+", help="files to scan")
    ap.add_argument("--hook", default=str(HOOK),
                    help="where to read the pattern list from (default: the project's pre-commit)")
    ap.add_argument("--allow", default=".baseline-hook-allow",
                    help="phrases a project has decided are legitimate, one per line")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing on a clean scan, for use in a pipeline")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        patterns = read_patterns(pathlib.Path(a.hook))
    except PatternsUnreadableError as e:
        print(f"prose tells: {e}", file=sys.stderr)
        return 2

    findings, unreadable = scan_paths(a.paths, patterns, read_allowlist(a.allow))

    for path, reason in unreadable:
        print(f"prose tells: could not read {path}: {reason}", file=sys.stderr)

    if not findings:
        if not a.quiet:
            print(f"prose tells: {len(a.paths)} file(s) scanned against "
                  f"{len(patterns)} patterns, nothing found."
                  + (f" {len(unreadable)} file(s) could not be read." if unreadable else ""))
        return 2 if unreadable else 0

    print(f"prose tells: {len(findings)} finding(s) across {len(a.paths)} file(s)")
    for path, n, line, _source, hit in findings:
        print(f"  {path}:{n}  {hit}")
        print(f"      {line[:140]}")
    print("\nEach of these reads as machine-written even when the content is right. Rewrite the "
          "sentence; do not reach for the allowlist unless the phrase is genuinely the right one.")
    return 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
