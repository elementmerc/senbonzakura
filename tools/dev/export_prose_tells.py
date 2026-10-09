#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Copy the hook's tell patterns into a tracked file, so CI has something to read.

`check_prose_tells.py` deliberately parses the pre-commit hook's own pattern array rather than
keeping a second copy, and refuses to run when it cannot find it. That is the right design on a
developer's machine. It cannot work in CI: the hooks are kept out of this repository on purpose,
because tracking them into a public repo was itself the disclosure, so the checker's default
source is a file a public checkout will never have.

The way out that keeps the single source of truth is to generate the copy rather than write it.
This script lifts the array out of the hook verbatim, comments and all, and writes it with a
provenance header. CI then points the checker at the generated file explicitly, so the source it
read is named in the command rather than guessed at by a fallback.

Drift is the obvious risk, and it is checked rather than hoped about: a test compares the two
pattern lists whenever the hook is present, which is every machine where a change to the hook can
actually be made. Regenerate with:

    python tools/dev/export_prose_tells.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOOK = ROOT / ".githooks" / "pre-commit"
OUT = ROOT / "tools" / "ci" / "prose-tells.sh"
ARRAY = "LLM_TELL_PATTERNS=("
CLOSE = "\n    )"

HEADER = """# GENERATED FILE. Do not edit by hand.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King
#
# The tell patterns from the project's pre-commit hook, exported so the CI checker has a pattern
# list to read: the hooks are kept out of this repository deliberately, so a public checkout has
# no hook to parse. The hook remains the source of truth, this is a copy of it, and a test fails
# if the two drift.
#
# Regenerate with:  python tools/dev/export_prose_tells.py
#
# The syntax below is the hook's bash array, kept verbatim so one parser reads both.
"""


def extract(hook_text):
    """The array block out of the hook, verbatim, or a reason it could not be found."""
    start = hook_text.find(ARRAY)
    if start < 0:
        raise SystemExit(
            f"the hook has no `{ARRAY}` array, so there is nothing to export. Its format has "
            f"changed and this script needs updating before CI can have a pattern list.")
    body = hook_text[start:]
    end = body.find(CLOSE)
    if end < 0:
        raise SystemExit(
            "the pattern array is not closed where this expects. Refusing to guess where it ends, "
            "because a truncated export is a checker that silently stops catching things.")
    return body[:end + len(CLOSE)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in argv

    try:
        hook_text = HOOK.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(
            f"cannot read the hook at {HOOK}: {e}. This script runs where the fleet hooks are "
            f"installed; it is not meant to run in CI, which consumes the file it writes.") from e

    wanted = HEADER + "\n" + extract(hook_text) + "\n"

    if check_only:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != wanted:
            print(f"{OUT} is stale against {HOOK}. Regenerate it with:\n"
                  f"    python tools/dev/export_prose_tells.py", file=sys.stderr)
            return 1
        print(f"{OUT.name} matches the hook.")
        return 0

    OUT.write_text(wanted, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} from {HOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
