# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura setup` marks the card line it shortened, and never cuts a UUID silently.

WHAT PROMPTED IT, 2026-09-27

Audit finding 15: values truncated to a column budget with no ellipsis. The occurrence in
`envsetup._describe` could only be seen on a machine with an NVIDIA card, which is why it survived
several passes on hardware that has none, and it was confirmed on the ROG on 2026-09-27. What
`nvidia_gpus` returns is the whole `nvidia-smi -L` line, which on that machine is 90 characters:

    GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-633f1990-0faf-53a0-cfcb-f1a382526659)

Cut to sixty with `g[:60]`, the reader was shown `... (UUID: GPU-633f1990-0f` and no sign that
anything had been removed, so a truncated identifier read as a complete one. That is the worst case
of the whole finding: two cards whose names agree for sixty characters differ only in the UUID, so
cutting it makes them the same card on screen.

THE RULE

A card line longer than the budget ends in the cut marker. A card line that fits is printed whole,
with no marker, because a marker that always appears carries no information. The test drives the
same function `senbonzakura setup` prints through, with the real string off the real card.
"""
from __future__ import annotations

from senbonzakura import envsetup, say

#: Verbatim from `nvidia-smi -L` on the ROG, 2026-09-27. 90 characters.
ROG = "GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-633f1990-0faf-53a0-cfcb-f1a382526659)"

#: Two cards of the same model. Their names agree far past any sensible budget, and only the UUID
#: tells them apart, which is the whole argument for marking the cut.
TWINS = [ROG, ROG.replace("GPU 0", "GPU 1").replace("633f1990", "911a2b3c")]


def _said(gpus):
    out = []
    envsetup._describe("Linux", "x86_64", gpus, (13, 3), "2.13.0+cu126", "cu126", out.append,
                       compute=(8, 6))
    return [line for line in out if "NVIDIA driver" in line]


def test_a_card_line_longer_than_the_budget_says_it_was_cut():
    line, = _said([ROG])
    assert say.CUT in line, (
        f"a card's identifier was cut with nothing to show it, so a partial UUID reads as a whole "
        f"one: {line!r}")


def test_a_card_line_that_fits_is_printed_whole_and_unmarked():
    """The other half of the property. A marker on everything says nothing about anything."""
    line, = _said(["GPU 0: NVIDIA GeForce RTX 3060"])
    assert "GPU 0: NVIDIA GeForce RTX 3060" in line
    assert say.CUT not in line


def test_every_card_is_still_counted_and_listed():
    """Shortening must not lose a card. The count is what tells a reader the driver saw both."""
    line, = _said(TWINS)
    assert "2 GPU(s)" in line
    assert line.count(say.CUT) == 2, f"one of two cards went missing: {line!r}"


def test_no_card_is_described_when_the_driver_reports_none():
    """A machine with a driver and no card is not the same answer as a machine with no driver, and
    neither is allowed to become a cut string.
    """
    assert "reports no GPU" in _said([])[0]
    assert "not found" in _said(None)[0]
