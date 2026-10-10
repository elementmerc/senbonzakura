# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The CLA gate, tested on the ways it could pass without checking anything.

`CLA.md` clause 2 is the only reason this code could ever be offered under terms other than the
AGPL, and that permission comes from the person who wrote each contribution. A commit merged from
an author who never agreed closes the option for the code it touched, permanently, because a
contributor who has moved on cannot be asked.

So the failure worth catching is not a wrong verdict. It is a check that reports success while
examining nothing: an empty range, a contributors file that lost its content, an assent sentence
that changed wording so the matcher reads for words no longer there. Each of those is a test below,
and each returns a failure rather than a pass, which is the opposite of what a careless
implementation does.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "check_cla_assent", ROOT / "tools" / "ci" / "check_cla_assent.py")
cla = importlib.util.module_from_spec(_spec)
sys.modules["check_cla_assent"] = cla
_spec.loader.exec_module(cla)

AGREED = "Someone Else <someone@example.com>  2026-10-10  I agree to the CLA in CLA.md.\n"


def test_the_real_history_has_no_unagreed_author():
    """THE ONE THAT MATTERS. Every author of every commit in this repository is recorded.

    If this fails, somebody's contribution is in the tree without the permission that keeps
    relicensing possible, and the fix is a line in CONTRIBUTORS.md rather than a change here.
    """
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    first = root.stdout.split()[0]
    assert cla.main([f"--range={first}..HEAD"]) == 0


def test_an_author_who_never_agreed_is_named(tmp_path):
    contributors = tmp_path / "CONTRIBUTORS.md"
    contributors.write_text(AGREED, encoding="utf-8")
    # The real history's authors are not in that file, so every one of them is missing.
    assert cla.main(["--range=HEAD~1..HEAD", f"--contributors={contributors}"]) == 1


def test_an_empty_range_fails_rather_than_passing(tmp_path):
    """A range matching no commits and a range whose authors all agreed look identical in a log.

    This project has twice reported a run that measured nothing as a clean one, so the only safe
    verdict for an empty range is a failure.
    """
    contributors = tmp_path / "CONTRIBUTORS.md"
    contributors.write_text(AGREED, encoding="utf-8")
    assert cla.main(["--range=HEAD..HEAD", f"--contributors={contributors}"]) == 2


def test_a_missing_contributors_file_fails_rather_than_finding_nothing_to_check(tmp_path):
    assert cla.main(["--range=HEAD~1..HEAD",
                     f"--contributors={tmp_path / 'absent.md'}"]) == 2


def test_a_contributors_file_with_no_agreement_in_it_fails(tmp_path):
    """Covers the clobber this project has already had, and a reworded assent sentence.

    Either the file lost its content or the matcher is reading for words that are no longer the
    ones used. Both need a person, and neither may report success.
    """
    contributors = tmp_path / "CONTRIBUTORS.md"
    contributors.write_text("# Contributors\n\nNobody yet.\n", encoding="utf-8")
    assert cla.main(["--range=HEAD~1..HEAD", f"--contributors={contributors}"]) == 2


def test_an_address_alone_is_not_agreement():
    """The instruction is to write the sentence, so a bare address on a line is not assent.

    Otherwise the template's own example line, and any address in a credit list, would count as
    somebody agreeing to a licence they never read.
    """
    assert cla.agreed_emails("Someone <someone@example.com>\n") == set()
    assert cla.agreed_emails(AGREED) == {"someone@example.com"}


def test_the_files_own_instruction_line_is_not_read_as_a_signature():
    # `CONTRIBUTORS.md` carries `Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.` as the
    # instruction. It contains the sentence, so a matcher that only looks for the sentence counts
    # the instruction itself and the file is never empty of agreements.
    assert cla.agreed_emails(
        "Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.\n") == set()


def test_the_maintainers_own_line_counts_without_the_sentence():
    # He holds the copyright, so there is nobody for him to grant a licence to, and his line says
    # what it is instead of agreeing with itself.
    assert cla.agreed_emails(
        "Daniel Iwugo <d@example.com>  2026-08-03  Project maintainer and copyright holder.\n"
    ) == {"d@example.com"}


def test_matching_is_case_insensitive_on_the_address():
    # git records whatever case the author's client sent, and the same person's address arrives
    # both ways across a long history.
    assert cla.agreed_emails(
        "P <Someone@Example.COM>  2026-10-10  I agree to the CLA in CLA.md.\n"
    ) == {"someone@example.com"}


@pytest.mark.parametrize("line", [
    "# Contributors",
    "> a blockquote",
    "```",
    "Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.",
])
def test_structural_lines_are_never_signatures(line):
    assert cla.agreed_emails(line + "\n") == set()


def test_the_real_contributors_file_records_the_maintainer():
    # Guards the clobber directly: a templated CONTRIBUTORS.md would make the gate above fail
    # loudly, and this says what the file is supposed to contain so the reason is obvious.
    agreed = cla.agreed_emails((ROOT / "CONTRIBUTORS.md").read_text(encoding="utf-8"))
    assert "daniel@themalwarefiles.com" in agreed
    assert len(agreed) >= 1


def test_the_cla_still_grants_the_relicensing_right():
    """The clause the whole mechanism exists to protect.

    If clause 2 is ever softened to AGPL-only, this gate goes on passing while protecting nothing,
    because every author would be agreeing to something that no longer keeps the option open.
    """
    text = (ROOT / "CLA.md").read_text(encoding="utf-8").lower()
    assert "sublicense" in text
    assert "any other licence terms the maintainer chooses" in text, (
        "CLA.md clause 2 no longer grants the right to offer this code under other terms, so "
        "collecting assent to it no longer keeps relicensing possible. If that change was "
        "deliberate, this gate and its tests should go with it.")
