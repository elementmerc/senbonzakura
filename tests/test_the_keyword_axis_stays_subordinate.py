# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The keyword term must stay smaller than the metric it is meant to support.

WHY THIS FILE EXISTS. `KNEE_W_KEYWORD` sat at 1.0 from 2026-09-06 to 2026-10-05, and at 1.0 it
decided the selection: measured across the ten real arms of the September head-to-head it ran at
357% to 933% of every other term combined, and removing it reordered nine of them. The note beside
it said it had been raised so the axis would carry "real weight" rather than only break exact ties,
so the intent was a contributor and what shipped was a dictator.

**NO TEST COULD HAVE CAUGHT THAT, AND THAT IS THE REAL DEFECT.** `test_knee_scalar_weights_keyword_axis`
asserts a lower keyword rate scores better, which is true at any positive weight including 100. Every
other test here passes heretic=0 or two equal values. So the suite pinned the term's DIRECTION and
never its SIZE, and a weight is a size.

These tests pin the size. They are written against the measured ranges rather than against invented
numbers, so they fail when the real relationship breaks rather than when a toy one does.
"""
import pytest

from senbonzakura.metrics import KNEE_W_KEYWORD, KNEE_W_NONCOMPLIANCE, knee_scalar

#: The ten real arms, from `private/results/h2h-2026-09-10/refusal-*.json`, as
#: (name, refusal, soft_refusal, heretic_keyword_rate). `broken` was 0.0 on every arm.
#:
#: COPIED IN RATHER THAN READ FROM DISK, deliberately: those artefacts live under `private/`, which
#: is excluded from both remotes, so a test that loaded them would pass here and skip in CI, which
#: is the shape of guard this project keeps finding. These five numbers per arm are aggregate rates
#: and carry no prompt or generation text, so they are safe to commit where the artefacts are not.
MEASURED_ARMS = [
    ("heretic-seed42", 0.0000, 0.0150, 0.0850),
    ("heretic-seed43", 0.0000, 0.0150, 0.1400),
    ("heretic-seed44", 0.0000, 0.0200, 0.1100),
    ("heretic-seed45", 0.0000, 0.0350, 0.1850),
    ("heretic-seed46", 0.0000, 0.0350, 0.1250),
    ("senbon-seed42", 0.0000, 0.0500, 0.2400),
    ("senbon-seed43", 0.0000, 0.0250, 0.1700),
    ("senbon-seed44", 0.0000, 0.0200, 0.1200),
    ("senbon-seed45", 0.0000, 0.0300, 0.1950),
    ("senbon-seed46", 0.0050, 0.0250, 0.2800),
]

#: The ceiling the 2026-10-05 decision chose. At 0.1 the term reaches 93.3% of the rest on the
#: worst measured arm, so this is not "the keyword term is always small": it is "the keyword term
#: cannot EXCEED the metric it supports". That is the property 1.0 violated by up to ninefold.
MAX_SHARE_OF_THE_REST = 1.0


def _rest(refusal, soft, broken=0.0):
    return KNEE_W_NONCOMPLIANCE * (refusal + soft) + 2.0 * broken


@pytest.mark.parametrize(("name", "refusal", "soft", "heretic"), MEASURED_ARMS)
def test_the_keyword_term_does_not_exceed_every_other_term_combined(name, refusal, soft, heretic):
    """The property 1.0 broke, checked on the arms it broke it on.

    At the shipped 1.0 this assertion failed on all ten, by between 3.6 and 9.3 times.
    """
    rest = _rest(refusal, soft)
    assert rest > 0, f"{name} has no non-keyword signal at all, so the share is undefined"
    share = KNEE_W_KEYWORD * heretic / rest
    assert share <= MAX_SHARE_OF_THE_REST, (
        f"on {name} the keyword term is {share:.1%} of every other term combined. A term meant to "
        f"support our own refusal metric is outweighing it, which means the selection is being "
        f"made by lexical markers rather than by the measurement. KNEE_W_KEYWORD is "
        f"{KNEE_W_KEYWORD}; the 2026-10-05 measurement is in the comment beside it.")


def test_the_keyword_axis_still_has_a_direction():
    """Subordinate is not zero. Lower keyword rate must still score better, all else equal."""
    worse = knee_scalar(0.01, 0.01, 0.50, 0.05, broken=0.0)
    better = knee_scalar(0.01, 0.01, 0.10, 0.05, broken=0.0)
    assert better < worse, (
        "the keyword axis has lost its direction entirely. If that is intended, remove the term "
        "and its weight rather than leaving a weight of zero, so a reader is not told the "
        "selection considers something it ignores.")


def test_the_keyword_term_cannot_reverse_a_wide_separation():
    """A tiebreaker breaks ties. It must not overturn a gap our own metric calls decisive.

    Built from the measured extremes: the best arm's non-compliance (0.0150) against the worst
    (0.0500), paired with the keyword rates that would most favour reversing them. At the shipped
    1.0 this assertion failed, which is how a term described as a tiebreaker was deciding arms.
    """
    clearly_better = knee_scalar(0.0, 0.0150, 0.2800, 0.05, broken=0.0)  # worst keyword rate
    clearly_worse = knee_scalar(0.0, 0.0500, 0.0850, 0.05, broken=0.0)   # best keyword rate
    assert clearly_better < clearly_worse, (
        "the keyword term reversed a threefold gap in our own refusal metric. At that strength it "
        "is the selection rule and our metric is the tiebreaker, which is the opposite of what "
        "the weights are documented to mean.")
