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


@pytest.fixture
def a_repo_with_one_commit(tmp_path, monkeypatch):
    """A throwaway repository with a known author, so these tests do not read the real history.

    THREE OF THESE TESTS USED `HEAD~1..HEAD` AGAINST THIS REPOSITORY AND WENT RED IN CI, which is
    the whole reason this exists. `actions/checkout` clones at depth 1 unless a job asks for more,
    so `HEAD~1` does not resolve, the checker correctly reports a range it cannot read, and the
    tests asserted 1 against the 2 they got. The assertions were right and the fixture was the
    ambient repository, which is a different depth on every machine that runs it.

    A test about what the checker does with a range should supply the range. `cla.ROOT` is
    monkeypatched rather than a `--repo` flag being added, because the tool's job is to examine
    the repository it ships in and widening its surface to make a test hermetic is the wrong way
    round.
    """
    def run(*argv):
        subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.name", "Nobody Inparticular")
    run("git", "config", "user.email", "nobody@example.invalid")
    run("git", "config", "commit.gpgsign", "false")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    run("git", "add", "a.txt")
    run("git", "commit", "-q", "-m", "one")
    (tmp_path / "a.txt").write_text("b\n", encoding="utf-8")
    run("git", "commit", "-q", "-a", "-m", "two")
    monkeypatch.setattr(cla, "ROOT", tmp_path)
    return tmp_path


def test_the_real_history_has_no_unagreed_author():
    """THE ONE THAT MATTERS. Every author of every commit in this repository is recorded.

    If this fails, somebody's contribution is in the tree without the permission that keeps
    relicensing possible, and the fix is a line in CONTRIBUTORS.md rather than a change here.
    """
    # ASKED DIRECTLY, BECAUSE A SHALLOW CLONE LIES ABOUT HAVING A ROOT COMMIT.
    #
    # The first fix here tested `rev-list --max-parents=0 HEAD` for failure. In a depth-1 clone
    # that command SUCCEEDS: the only commit present has no parents, so git reports it as a root.
    # The range then resolved to `HEAD..HEAD`, the checker correctly refused an empty range, and
    # the test failed in exactly the place the skip was meant to cover. Caught by cloning this
    # repository at depth 1 and running the file against it, not by reading the code.
    #
    # `--is-shallow-repository` is the question actually being asked, and it answers `true` in a
    # depth-1 clone and `false` here. Two CI jobs run without `fetch-depth: 0`.
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                             cwd=ROOT, capture_output=True, text=True, check=False)
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    if shallow.stdout.strip() == "true" or root.returncode != 0 or not root.stdout.split():
        pytest.skip("this is a shallow clone or not a checkout, so the full history is not here. "
                    "The real history was NOT checked by this run; the job that checks it needs "
                    "fetch-depth: 0.")
    first = root.stdout.split()[0]
    assert cla.main([f"--range={first}..HEAD"]) == 0


def test_an_author_who_never_agreed_is_named(tmp_path, a_repo_with_one_commit):
    contributors = tmp_path / "CONTRIBUTORS.md"
    contributors.write_text(AGREED, encoding="utf-8")
    # The real history's authors are not in that file, so every one of them is missing.
    assert cla.main(["--range=HEAD~1..HEAD", f"--contributors={contributors}"]) == 1


def test_an_empty_range_fails_rather_than_passing(tmp_path, a_repo_with_one_commit):
    """A range matching no commits and a range whose authors all agreed look identical in a log.

    This project has twice reported a run that measured nothing as a clean one, so the only safe
    verdict for an empty range is a failure.
    """
    contributors = tmp_path / "CONTRIBUTORS.md"
    contributors.write_text(AGREED, encoding="utf-8")
    assert cla.main(["--range=HEAD..HEAD", f"--contributors={contributors}"]) == 2


def test_a_missing_contributors_file_fails_rather_than_finding_nothing_to_check(
        tmp_path, a_repo_with_one_commit):
    assert cla.main(["--range=HEAD~1..HEAD",
                     f"--contributors={tmp_path / 'absent.md'}"]) == 2


def test_a_contributors_file_with_no_agreement_in_it_fails(tmp_path, a_repo_with_one_commit):
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
