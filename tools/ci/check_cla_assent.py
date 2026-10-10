#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every author of a commit under review has agreed to the CLA, in writing, in the repository.

WHY A GATE AND NOT A CONVENTION. `CLA.md` keeps one option open: offering this code to somebody
who cannot take AGPL terms. That needs permission from everyone who wrote part of it, and asking
later is the part that fails, because people change jobs and addresses and some cannot be found at
all. The agreement exists and `CONTRIBUTING.md` asks for a line in `CONTRIBUTORS.md`, which is an
honour system: nothing refuses a merge from an author who never added one. One unagreed commit is
enough to close the option for the file it touched, and nobody notices until a lawyer reads the
history years later.

So the check is mechanical: for the commits in this range, every distinct author either appears in
`CONTRIBUTORS.md` or the run fails naming them.

WHAT THIS IS NOT. It is not legal advice and it is not a signature. It proves an assent line exists
in a file in a commit, which is evidence and not a contract, and `CLA.md` says on its own first
line that it has not been reviewed by a solicitor. A verifiable record is what makes the review
worth paying for later; it does not replace it.

MATCHING IS BY EMAIL, NOT NAME. A name is written four ways by the same person across a history and
a mailmap only fixes the ones somebody thought of. The local part and domain are compared in lower
case, because git records whatever case the author's client sent.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: Authors that are this project's own automation rather than people. A bot cannot agree to a
#: licence, and its commits carry no authorship a contributor could later object to.
MACHINE_AUTHORS = frozenset({
    "41898282+github-actions[bot]@users.noreply.github.com",
    "ops@themalwarefiles.com",
})

EMAIL = re.compile(r"<([^<>@\s]+@[^<>@\s]+)>")


def agreed_emails(contributors: str) -> set[str]:
    """Every email on a line that actually states agreement, or is the maintainer's own.

    The instruction in `CONTRIBUTORS.md` is to write the agreement sentence on the line, so a line
    carrying an address and nothing else is not assent and is not counted. The maintainer's line
    predates the agreement and says what it is instead; he holds the copyright, so there is nobody
    for him to grant a licence to.
    """
    found = set()
    for line in contributors.splitlines():
        if line.lstrip().startswith(("#", ">", "```", "Name <email>")):
            continue
        match = EMAIL.search(line)
        if not match:
            continue
        said_yes = "agree to the cla" in line.lower()
        is_holder = "copyright holder" in line.lower()
        if said_yes or is_holder:
            found.add(match.group(1).lower())
    return found


def authors_in(rev_range: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%an|%ae", rev_range],
        cwd=ROOT, capture_output=True, text=True, check=True)
    seen: dict[str, str] = {}
    for line in out.stdout.splitlines():
        name, _, email = line.partition("|")
        if email:
            seen.setdefault(email.lower(), name)
    return sorted((email, name) for email, name in seen.items())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--range", dest="rev_range", default="HEAD~1..HEAD",
                    help="the git revision range to check (default: the last commit)")
    ap.add_argument("--contributors", type=Path, default=ROOT / "CONTRIBUTORS.md")
    args = ap.parse_args(argv)

    if not args.contributors.is_file():
        print(f"FAIL: no contributors file at {args.contributors}. This check cannot pass by "
              f"finding nothing to check: without that file nobody has recorded agreement.",
              file=sys.stderr)
        return 2

    agreed = agreed_emails(args.contributors.read_text(encoding="utf-8"))
    if not agreed:
        print(f"FAIL: {args.contributors.name} records no agreement at all. Either the file lost "
              f"its content or the agreement sentence changed and this check is now reading for "
              f"the wrong words. Both need a human.", file=sys.stderr)
        return 2

    try:
        authors = authors_in(args.rev_range)
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: could not read the authors of {args.rev_range}: "
              f"{(exc.stderr or '').strip() or exc}", file=sys.stderr)
        return 2

    if not authors:
        print(f"FAIL: {args.rev_range} contains no commits, so nothing was checked. A range that "
              f"matches nothing and a range whose authors all agreed look identical in a log, "
              f"which is why this is a failure and not a pass.", file=sys.stderr)
        return 2

    missing = [(e, n) for e, n in authors if e not in agreed and e not in MACHINE_AUTHORS]
    if missing:
        print(f"FAIL: {len(missing)} of {len(authors)} author(s) in {args.rev_range} have not "
              f"agreed to the CLA:", file=sys.stderr)
        for email, name in missing:
            print(f"  {name} <{email}>", file=sys.stderr)
        print("\nAdd a line to CONTRIBUTORS.md, as CONTRIBUTING.md describes:\n"
              "  Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.\n"
              "\nThis is not a formality. CLA.md clause 2 is what allows this code to be offered "
              "under terms other than the AGPL later, and that permission can only be given by "
              "the person who wrote the contribution. One commit without it closes the option for "
              "the code it touched, permanently, because a contributor who cannot be found cannot "
              "be asked.", file=sys.stderr)
        return 1

    print(f"CLA: {len(authors)} author(s) in {args.rev_range}, all recorded as agreed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
