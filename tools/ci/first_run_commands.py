#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Pick the commands the documented first run executes, and refuse to guess which ones.

WHY MARKERS AND NOT CONTENT

The documented-first-run job runs the install and quickstart commands the documentation shows, as
written. It used to find them by content: the one ```sh block containing `pip install senbonzakura`.
The README now holds two such blocks, because the virtual environment route contains
`bin/pip install senbonzakura` as a substring. The job stopped, loudly, which was correct.

The obvious repair is to take the first match, and that one fails silently. Reorder the README and
the job runs the other route without saying so, and nothing turns red.

So each block the job runs is named by an HTML comment directly above it. The comment is invisible
on the rendered page. A block is selected by the marker that names it, not by where it sits or
what it happens to say. Every refusal here stops the job with a reason: a marker missing, a marker
repeated, a marker that is not directly followed by a ```sh fence, or an install command the
substitution cannot reach.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

README = "README.md"
QUICKSTART = "docs/guide/quickstart.md"
INSTALL_MARKER = "documented-first-run: install"
QUICKSTART_MARKER = "documented-first-run: quickstart"

#: The documented install points at the package on the index. This job tests the checkout instead,
#: so the substitution is the one place the documentation is not run as written, and it is named.
OLD_INSTALL = "pip install senbonzakura"
NEW_INSTALL = "pip install ./checker ."

_FENCE_AFTER_MARKER = re.compile(r"\s*```sh\n(?P<body>.*?)```", re.DOTALL)


class FirstRunError(Exception):
    """A refusal a person can act on. The message is the whole reason, so the job log is enough."""


def marked_block(text, marker, path):
    """The body of the ```sh block that the marker names, and nothing else."""
    token = f"<!-- {marker} -->"
    count = text.count(token)
    if count != 1:
        raise FirstRunError(
            f"{path} has {count} `{token}` markers and needs exactly one, because the job cannot "
            f"tell which block the documentation means otherwise. Put the marker directly above "
            f"the ```sh block it names.")
    match = _FENCE_AFTER_MARKER.match(text.split(token, 1)[1])
    if match is None:
        raise FirstRunError(
            f"the `{token}` marker in {path} is not directly followed by a ```sh block. Move the "
            f"marker so that it sits directly above that block, with only whitespace between.")
    return match.group("body")


def _read(root, rel):
    try:
        return (root / rel).read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise FirstRunError(f"cannot read {rel}: {exc}") from exc


def build(root, out):
    """Write install.sh and quickstart.sh into `out`, and return what was written."""
    install = marked_block(_read(root, README), INSTALL_MARKER, README)
    quickstart = marked_block(_read(root, QUICKSTART), QUICKSTART_MARKER, QUICKSTART)
    if OLD_INSTALL not in install:
        raise FirstRunError(
            f"the install block marked in {README} no longer contains {OLD_INSTALL!r}, so it "
            f"cannot be pointed at this checkout. Update this job together with the README.")
    install = install.replace(OLD_INSTALL, NEW_INSTALL, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "install.sh").write_text(install, encoding="utf-8")
    (out / "quickstart.sh").write_text(quickstart, encoding="utf-8")
    return install, quickstart


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="directory to write install.sh and quickstart.sh into")
    parser.add_argument("--root", default=ROOT, type=pathlib.Path,
                        help="repository root to read the documentation from")
    args = parser.parse_args(argv)
    try:
        install, quickstart = build(args.root, args.out)
    except FirstRunError as exc:
        print(f"first-run: {exc}", file=sys.stderr)
        return 1
    print("--- install ---")
    print(install)
    print("--- quickstart ---")
    print(quickstart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
