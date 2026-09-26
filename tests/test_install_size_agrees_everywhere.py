# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""How big the install is, said in more than one place, must be said the same way.

WHAT PROMPTED IT, 2026-09-26

A stranger installed the tool, counted what arrived, and found three pages giving three answers:
the README said 69 packages and 5.9 GB, the quickstart said 68 and 5.8 GB, and the install guide
said 68 and 5.8 GB with nineteen CUDA wheels. They measured 69, 5.9 GB and sixteen. One of the
three had been updated and the other two had not.

The figure itself is small. The shape is not, and this repository already carries a commit called
"three defects where a second copy of a fact disagreed with the first". A reader who finds two
numbers for one thing learns something true about the project and stops trusting the rest of the
page, which costs far more than the number was worth.

WHY A TEST RATHER THAN CARE

Nothing about a stale figure fails. It does not break a build, it does not raise, and it does not
appear in a diff of the file that went stale, because the staleness is in the file nobody touched.
That is precisely the class of defect this project answers with arithmetic instead of attention.

The canonical measurement lives in `docs/guide/install.md`, with the date and the interpreter
beside it, because any count of this kind decays: a different Python ships different defaults.
This file does not check the figure is RIGHT, which needs a 5.9 GB install and belongs in the
clean-room job. It checks that every page telling a reader the same fact tells them the same thing.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every page that quotes the size of a full install. Adding one here is the point: a new page that
#: repeats the figure has to agree with the others or this fails.
PAGES = [
    ROOT / "README.md",
    ROOT / "docs" / "guide" / "install.md",
    ROOT / "docs" / "guide" / "quickstart.md",
]

#: "69 packages", "68 packages". Not "packages" alone, which appears in plenty of other sentences.
_COUNT = re.compile(r"\b(\d{2,3})\s+packages\b")

#: "5.9 GB", "5.8 GB". Restricted to one decimal place so a table of quantisation sizes elsewhere
#: cannot be mistaken for this claim.
_SIZE = re.compile(r"\b(\d\.\d)\s*GB\b")


def _figures(pattern):
    found = {}
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        values = set(pattern.findall(text))
        if values:
            found[page.relative_to(ROOT).as_posix()] = values
    return found


def test_the_pages_that_state_a_package_count_exist():
    """A moved or renamed page would make every assertion below pass by checking nothing."""
    missing = [p.relative_to(ROOT).as_posix() for p in PAGES if not p.is_file()]
    assert not missing, f"these pages are named here and are not in the tree: {missing}"


def test_every_page_gives_the_same_package_count():
    found = _figures(_COUNT)
    assert found, (
        "no page states a package count any more. If the claim has been dropped deliberately, "
        "drop this test with it; if a page was reworded, the count is now unguarded.")
    values = set().union(*found.values())
    assert len(values) == 1, (
        f"the pages disagree about how many packages a full install brings: {found}. "
        f"A reader who counts finds one of them wrong and stops believing the others. The measured "
        f"figure and the date it was taken live in docs/guide/install.md.")


def test_every_page_gives_the_same_install_size():
    found = _figures(_SIZE)
    assert found, "no page states an install size any more; see the note in the count test above"
    values = set().union(*found.values())
    assert len(values) == 1, (
        f"the pages disagree about how big a full install is: {found}. "
        f"The measured figure and its date live in docs/guide/install.md.")


def test_the_canonical_page_dates_its_measurement():
    """A count without a date cannot be known to have gone stale, so it never gets corrected.

    Different interpreters ship different default packages, so this figure decays by itself. The
    date is what lets the next reader tell "wrong" from "measured on something else".
    """
    text = (ROOT / "docs" / "guide" / "install.md").read_text(encoding="utf-8")
    # Anchored on the COUNT rather than on the word "packages", which appears in several unrelated
    # sentences earlier on the page; the first of those was what this matched when it was written.
    match = _COUNT.search(text)
    assert match, "the install guide no longer states a package count at all"
    window = text[max(0, match.start() - 700):match.end() + 700]
    assert re.search(r"20\d\d-\d\d-\d\d", window), (
        "the install guide states a package count with no date near it. Give the date and the "
        "Python version it was measured on, because this figure goes out of date without anybody "
        "editing the file it is in.")
