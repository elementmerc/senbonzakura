# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A figure in an `evidence/` README has to come from a file committed beside it.

WHY THIS EXISTS

`evidence/compass-2026-10-11/README.md` printed a comparison table whose first row read 0.9851 and
0.8339. No artefact in this repository produces those two numbers, and they could not be
recomputed from the record beside the README either: it holds summary statistics rather than the
per-item margins, and the `margins_path` it names is not committed.

**This is the exact class the v0.4.1 release gate said could not happen again, in the directory
whose whole rule it is.** `evidence/README.md` opens by saying the difference between this tree and
`results/` is that everything here is version-controlled, and the reason given is that a number
without its artefact is not evidence. A reader who checks our figures starts here, and one
unbacked row costs more than the table was worth.

WHAT IS ASSERTED

Every result-shaped number in an `evidence/` README appears in a JSON artefact in the same
directory, at full precision or at any rounding of it.

"Result-shaped" is deliberately narrow: a decimal with three or more places. Sample sizes, dates,
version numbers and row counts are not figures a reader would try to check, and sweeping them in
would make this a test somebody disables rather than one somebody heeds.

Prose may still DISCUSS a number it cannot show, which is what the withdrawn row became, as long as
it does not present it as a measurement. That is why the check reads numbers rather than sentences:
a withdrawn figure is deleted, not reformatted.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

#: A number a reader would try to reproduce. Three decimal places or more.
FIGURE = re.compile(r"(?<![\w.])(\d+\.\d{3,})(?![\w])")

#: Numbers that are not claims about a measurement even though they have the shape of one.
#: Kept tiny and justified; a growing list here is the signal that the rule has stopped fitting.
NOT_A_FIGURE = frozenset({
    "0.0001",      # a threshold quoted in prose, not a result
})

#: A section that exists to record a figure this project WITHDREW is the opposite of the defect
#: and must be allowed to print it.
#:
#: This guard found one on its first run: `compass-2026-07-30/README.md` has a section headed "An
#: earlier version of this file had different numbers", which prints 0.9636 and 0.7263 and
#: explains that they were measured at the wrong position. Its own sentence gives the reason it
#: belongs there: "recorded here rather than quietly replaced, because the artefact is meant to be
#: a record". Flagging that would push this project toward deleting its corrections, which is the
#: behaviour the whole directory exists to argue against.
#:
#: Scoped to the section rather than the sentence, because a correction table's rows carry the
#: figures and the heading carries the reason, and a per-sentence rule would flag every row.
WITHDRAWN = ("earlier version", "first published with", "withdrawn", "removed from this table",
             "were measured at", "was measured at", "wrong position", "no longer", "used to read",
             "replaced", "corrected")


def _sections(text):
    """`(heading plus its body, the body's figures)` for each markdown section.

    A section is a `##` heading and everything up to the next one. Text before the first heading
    is its own section with an empty heading, so a README with no headings still gets checked.
    """
    parts, current = [], []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _readmes():
    return sorted(EVIDENCE.rglob("README.md"))


def _numbers_in(obj):
    """Every number anywhere in a parsed artefact, as strings at full precision."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(repr(float(obj)))
    return out


def _artefact_numbers(directory):
    """Every number in every JSON artefact in one evidence directory."""
    found = set()
    for path in sorted(directory.glob("*.json")) + sorted(directory.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        for line in (text.splitlines() if path.suffix == ".jsonl" else [text]):
            if line.strip():
                found |= _numbers_in(json.loads(line))
    return found


def _backs(printed, artefact_numbers):
    """Does any artefact number round to the printed one?

    Rounding, not equality, because a README quotes 0.6853 for a stored 0.6852777777777778 and
    asking for the full tail would make the rule unusable in prose.
    """
    places = len(printed.split(".")[1])
    target = float(printed)
    return any(round(float(n), places) == target for n in artefact_numbers)


@pytest.mark.parametrize("readme", _readmes(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_figure_in_this_readme_comes_from_an_artefact_beside_it(readme):
    directory = readme.parent
    artefact_numbers = _artefact_numbers(directory)
    unbacked = []
    for section in _sections(readme.read_text(encoding="utf-8")):
        if any(marker in section.lower() for marker in WITHDRAWN):
            continue
        for printed in FIGURE.findall(section):
            if printed in NOT_A_FIGURE:
                continue
            if not _backs(printed, artefact_numbers):
                unbacked.append(printed)
    assert not unbacked, (
        f"{readme.relative_to(ROOT)} prints {sorted(set(unbacked))} and no artefact in "
        f"{directory.relative_to(ROOT)} produces them.\n"
        f"Commit the artefact, or remove the figure. A number without its artefact is not "
        f"evidence, which is what evidence/README.md says this directory is for.")


def test_the_directories_that_hold_a_readme_also_hold_an_artefact():
    """A README alone would pass the test above vacuously, having no figures to check."""
    for readme in _readmes():
        directory = readme.parent
        if directory == EVIDENCE:
            continue        # the top-level README describes the tree rather than a result
        assert _artefact_numbers(directory), \
            f"{directory.relative_to(ROOT)} holds a README and no artefact with numbers in it"


def test_the_withdrawn_row_is_gone_and_said_to_be_gone():
    """The specific defect, and that its removal is explained rather than silent.

    Deleting a published row without a word is its own small dishonesty: a reader who saw the old
    table would find it simply absent and have no way to tell whether it was wrong or inconvenient.
    """
    readme = (EVIDENCE / "compass-2026-10-11" / "README.md").read_text(encoding="utf-8")
    assert "0.9851" not in readme
    assert "0.8339" not in readme
    assert "removed from this table" in readme, "the withdrawal is not explained in place"


def test_the_guard_would_catch_the_figure_it_was_written_for():
    """Driven against the real artefacts, so it cannot pass by reading a fixture."""
    numbers = _artefact_numbers(EVIDENCE / "compass-2026-10-11")
    assert _backs("0.9872", numbers), "the figure that IS backed no longer reads as backed"
    assert _backs("0.6853", numbers), "rounding is not being allowed"
    assert not _backs("0.9851", numbers), "the withdrawn figure now appears to be backed"
    assert not _backs("0.8339", numbers)


def test_a_recorded_correction_may_print_the_figure_it_withdrew():
    """Found by this guard on its first run, and it would have been the wrong finding to act on.

    `compass-2026-07-30` keeps the two figures it published and then withdrew, under a heading
    saying so, because "the artefact is meant to be a record". A rule that flagged those would
    push this project toward deleting its corrections, which is what the directory argues against.
    """
    readme = (EVIDENCE / "compass-2026-07-30" / "README.md").read_text(encoding="utf-8")
    assert "0.9636" in readme and "0.7263" in readme, \
        "the withdrawn figures are gone; if that was deliberate, re-read this test"
    marked = [s for s in _sections(readme)
              if "0.9636" in s and any(m in s.lower() for m in WITHDRAWN)]
    assert marked, "the correction section no longer declares itself one, so the guard will flag it"


def test_an_unmarked_section_printing_an_unbacked_figure_is_still_caught():
    """The exemption must not be a hole big enough to drive the original defect through."""
    numbers = _artefact_numbers(EVIDENCE / "compass-2026-10-11")
    section = "## Results\n\n| | AUC |\n|---|---|\n| Scoring from row 0 | 0.9851 |\n"
    assert not any(marker in section.lower() for marker in WITHDRAWN)
    found = [f for f in FIGURE.findall(section) if not _backs(f, numbers)]
    assert found == ["0.9851"], found
