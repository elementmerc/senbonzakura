#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Put the licence and copyright on every source file, once, idempotently.

WHY IT MATTERS MORE THAN IT LOOKS

AGPL section 5 asks a modified work to carry appropriate legal notices, and a project whose only
copyright line is in `LICENSE` has asserted nothing about the files themselves. Files travel: they
get vendored, pasted into issues, and lifted into other projects, and a file with no header is a
file whose licence a downstream reader has to guess. The guess is usually "permissive".

WHAT IT WRITES, AND WHERE

Two lines at the very top, above the module docstring, matching the twelve modules that already
carry the SPDX line:

    # SPDX-License-Identifier: AGPL-3.0-or-later
    # Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>

A shebang stays first, because a `#!` on line two is not a shebang. `from __future__` imports are
untouched: they must precede code, and comments are not code.

WHAT IT WILL NOT TOUCH

Vendored third-party source, which carries somebody else's copyright and where adding ours would
be a false claim. That exclusion is the reason this is a script rather than a `sed`.

    python tools/add_license_headers.py --check    # exit 1 if any file is missing a header
    python tools/add_license_headers.py            # add them
"""
from __future__ import annotations

import argparse
import pathlib
import sys

SPDX = "# SPDX-License-Identifier: AGPL-3.0-or-later"
COPYRIGHT = "# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>"

#: Ours to licence. Everything under `vendor/` is somebody else's and is deliberately absent.
ROOTS = ("src/senbonzakura", "tools", "tests", "head-to-head")
SKIP = ("/vendor/", "egg-info", "__pycache__", "/node_modules/")


def targets(root):
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if not any(s in str(path) for s in SKIP):
            yield path


def needs_header(text):
    head = text.splitlines()[:5]
    return not any(line.startswith("# SPDX-License-Identifier") for line in head), \
        not any(line.startswith("# Copyright") for line in head)


def apply(path, *, check=False):
    """Return True when the file was (or would be) changed."""
    text = path.read_text(encoding="utf-8")
    want_spdx, want_copyright = needs_header(text)
    if not (want_spdx or want_copyright):
        return False
    if check:
        return True

    lines = text.splitlines(keepends=True)
    # A shebang must stay on line 1 or it stops being a shebang.
    at = 1 if lines and lines[0].startswith("#!") else 0

    # Insert each line at the right place rather than blindly at the top. The first version put
    # the copyright ABOVE an existing SPDX line, inverting the conventional order in the twelve
    # files that already had one: SPDX first, then copyright.
    if want_spdx:
        lines.insert(at, SPDX + "\n")
        at += 1
    else:
        # Land the copyright immediately after the SPDX line that is already there.
        for i, line in enumerate(lines[:5]):
            if line.startswith("# SPDX-License-Identifier"):
                at = i + 1
                break
    if want_copyright:
        lines.insert(at, COPYRIGHT + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report what is missing and change nothing; exit 1 if any file is")
    a = ap.parse_args(argv)

    found = [p for root in ROOTS if pathlib.Path(root).is_dir() for p in targets(root)]
    if not found:
        print("no source files found; run this from the repository root", file=sys.stderr)
        return 1

    changed = [p for p in found if apply(p, check=a.check)]
    if a.check:
        if changed:
            print(f"{len(changed)} of {len(found)} file(s) have no licence header:",
                  file=sys.stderr)
            for p in changed:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"licence headers: {len(found)} file(s), all present")
        return 0
    print(f"licence headers: added to {len(changed)} of {len(found)} file(s)")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
