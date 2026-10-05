# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura setup` marks the card line it shortened, and never cuts a UUID silently.

WHAT PROMPTED IT, 2026-09-27

Audit finding 15: values truncated to a column budget with no ellipsis. The occurrence in
`envsetup._describe` could only be seen on a machine with an NVIDIA card, which is why it survived
several passes on hardware that has none, and it was confirmed on 2026-09-27 on a 6 GB laptop
card. What `nvidia_gpus` returns is the whole `nvidia-smi -L` line, which on that machine is 90
characters:

    GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-0e1d2c3b-4a59-5867-7564-8a9b0c1d2e3f)

Cut to sixty with `g[:60]`, the reader was shown `... (UUID: GPU-0e1d2c3b-4a` and no sign that
anything had been removed, so a truncated identifier read as a complete one. That is the worst case
of the whole finding: two cards whose names agree for sixty characters differ only in the UUID, so
cutting it makes them the same card on screen.

THE RULE

A card line longer than the budget ends in the cut marker. A card line that fits is printed whole,
with no marker, because a marker that always appears carries no information. The test drives the
same function `senbonzakura setup` prints through, with the real string off the real card.

THE SECOND HALF, 2026-10-01: ONE CARD PER LINE

The marker was the half that got fixed in September. The half left over was the width. Joined with
`"; "` onto the label's own line, the two cards below measured **149 columns at COLUMNS=80**, so
the terminal reflowed the second card's text into the middle of the first's and the per-card cut
markers stopped marking anything a reader could point at. The sixty was a fixed budget besides,
chosen before this module shared `say.width()`, so a wide terminal cut a line that would have fit.

Cards now get a line each at `say.width()`, which also means the UUID survives 20 characters
rather than 8 on an ordinary terminal. The helper below reads the whole driver block rather than
one line, and that change is why the three tests written in September needed rewiring: they were
asserting on a single line because there used to be only one.
"""
from __future__ import annotations

import pytest

from senbonzakura import envsetup, say

#: The shape `nvidia-smi -L` really emitted on a 6 GB laptop card, 2026-09-27, with the card's own
#: UUID swapped for a synthetic one of identical length. The length is what this test measures, so
#: the substitution costs the test nothing, and a hardware identifier is nobody's business in a
#: public repository.
CARD_LINE = "GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-0e1d2c3b-4a59-5867-7564-8a9b0c1d2e3f)"

#: Two cards of the same model. Their names agree far past any sensible budget, and only the UUID
#: tells them apart, which is the whole argument for marking the cut.
TWINS = [CARD_LINE, CARD_LINE.replace("GPU 0", "GPU 1").replace("0e1d2c3b", "911a2b3c")]


def _block(gpus):
    """The driver block as `senbonzakura setup` prints it: the label, then a line per card.

    The whole block rather than one line, because the cards moved off the label's line. Found by
    shape (the label, then every line indented under it) rather than by index, since `_describe`
    prints a variable number of rows above and below.
    """
    out = []
    envsetup._describe("Linux", "x86_64", gpus, (13, 3), "2.13.0+cu126", "cu126", out.append,
                       compute=(8, 6))
    start = next(i for i, line in enumerate(out) if "NVIDIA driver" in line)
    block = [out[start]]
    for line in out[start + 1:]:
        if not line.startswith("    "):
            break
        block.append(line)
    return block


def _cards(gpus):
    """Just the card rows, which is what the shortening applies to."""
    return _block(gpus)[1:]


def test_a_card_line_longer_than_the_budget_says_it_was_cut(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    line, = _cards([CARD_LINE])
    assert say.CUT in line, (
        f"a card's identifier was cut with nothing to show it, so a partial UUID reads as a whole "
        f"one: {line!r}")


def test_a_card_line_that_fits_is_printed_whole_and_unmarked():
    """The other half of the property. A marker on everything says nothing about anything."""
    line, = _cards(["GPU 0: NVIDIA GeForce RTX 3060"])
    assert "GPU 0: NVIDIA GeForce RTX 3060" in line
    assert say.CUT not in line


def test_every_card_is_still_counted_and_listed(monkeypatch):
    """Shortening must not lose a card. The count is what tells a reader the driver saw both."""
    monkeypatch.setenv("COLUMNS", "80")
    block = _block(TWINS)
    assert "2 GPU(s)" in block[0]
    assert len(block) - 1 == 2, f"one of two cards went missing: {block!r}"
    assert sum(say.CUT in line for line in block[1:]) == 2


def test_no_card_is_described_when_the_driver_reports_none():
    """A machine with a driver and no card is not the same answer as a machine with no driver, and
    neither is allowed to become a cut string.
    """
    assert "reports no GPU" in _block([])[0]
    assert "not found" in _block(None)[0]


# ── the width, which is the half that was left ──────────────────────────────────
@pytest.mark.parametrize("columns", [80, 120, 40])
@pytest.mark.parametrize("count", [1, 2, 4])
def test_no_card_line_exceeds_the_ceiling(columns, count, monkeypatch):
    """Measured at 149 columns before this, with two cards at COLUMNS=80.

    120 is in the list deliberately: `say.width()` caps at `say.CEILING`, so a wide terminal must
    not widen these lines. A fix reading the terminal without the cap would pass at 80 and 40 and
    fail here. 40 is the floor `say.width()` clamps to.
    """
    monkeypatch.setenv("COLUMNS", str(columns))
    gpus = [CARD_LINE.replace("GPU 0", f"GPU {i}") for i in range(count)]
    for line in _block(gpus):
        assert len(line) <= say.CEILING, (
            f"{len(line)} columns with {count} card(s) at COLUMNS={columns}: {line!r}")


def test_each_card_is_on_its_own_line(monkeypatch):
    """The property, rather than the symptom.

    A fix that only shortened harder would pass the ceiling test above and still put two cards on
    one row, where neither index nor cut marker is where a reader can find it.
    """
    monkeypatch.setenv("COLUMNS", "80")
    cards = _cards(TWINS)
    assert len(cards) == 2, cards
    assert "GPU 0" in cards[0] and "GPU 1" not in cards[0], cards
    assert "GPU 1" in cards[1] and "GPU 0" not in cards[1], cards


def test_a_card_gets_more_of_a_wide_terminal_than_sixty_columns(monkeypatch):
    """The fixed sixty was the other half of the defect, and it cut what would have fitted.

    At COLUMNS=80 the budget is `say.CEILING` minus the four-space indent, so 20 characters of
    UUID survive where 8 did. Asserted as "more than the old budget" rather than as an exact
    number, so moving `CEILING` or the indent does not make this test wrong.
    """
    monkeypatch.setenv("COLUMNS", "80")
    line, = _cards([CARD_LINE])
    assert len(line.strip()) > 60, (
        f"the card line is still inside the old sixty-column budget: {line!r}")
    assert len(line) <= say.CEILING
