# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The future-version gate must refuse a version and allow a measurement.

WHAT PROMPTED IT, 2026-09-27

The commit adding "it refuses 58.6% of the bundled evaluation set" to the first-run guide was
blocked by the future-version gate, which read `58.6` as a reference to version 58.6. The project
had set no pattern, so the hook fell back to a generic one that treats any bare decimal as a
version number.

That fallback would equally have refused the 5.9 GB install size, the 80.5% refusal rate, the 94.5%
coverage floor and the 0.8% figure that is half of this project's argument that model size predicts
nothing about refusal. A gate that fires on every measured percentage, in a repository whose whole
claim is its measured percentages, gets bypassed within a day; and the bypass is the real damage,
because the next person reaches for it without reading why.

So the pattern is set explicitly, narrowed to require a version SHAPE rather than any decimal, and
this test is the reason it can be trusted: the gate's own examples live here, executed, rather than
in a comment nobody runs.

THE PART THAT WILL ROT IF NOTHING WATCHES IT

The pattern encodes that 0.4 is the current release. The day 0.5.0 ships, the gate would refuse the
release that is actually shipping, which is the same class of defect as a check pinned to a number
that moved. `test_the_pattern_has_not_outlived_the_release_it_was_written_for` reads the packaged
version and fails when that day arrives, with the instruction attached.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".baseline-hook-config"

#: Strings that ARE references to a version after 0.4, and must be refused in public docs.
MUST_REFUSE = [
    "v0.5", "v0.5.0", "0.5.0", "0.6.0", "v0.9", "v1.0", "1.0.0", "v2.3", "12.0.1",
    "planned for v0.7.0", "landing in 1.2.0",
]

#: Strings that are MEASUREMENTS or the current version, and must pass. Every one of these is a real
#: figure from this project's own documentation, which is the point: they are not hypothetical.
MUST_ALLOW = [
    "58.6% refusal",            # Qwen2.5-0.5B-Instruct on the bundled evaluation set, n=128
    "0.8% refusal",             # TinyLlama-1.1B, the low end of the bimodal distribution
    "80.5%",                    # Qwen2.5-1.5B, the high end
    "4.7%", "6.2%", "25.0%", "28.1%",
    "5.9 GB",                   # the measured install size
    "94.5",                     # the coverage floor
    "AUC 0.9887", "0.6616",     # compass figures, one of them a withdrawn one
    "0.2066 KL", "0.05 floor", "0.73 GB",
    "0.4.0", "v0.4.0",          # THE CURRENT RELEASE, which must never be refused
    "108 minutes", "857 seconds", "1.1B", "1.7B", "16 GB", "6 GB",
]


def _pattern():
    """The live pattern, read out of the config the hook reads, not a copy of it.

    `.baseline-hook-config` IS GITIGNORED, so it exists on a working machine and not in CI or a
    fresh clone. Skipping there is correct rather than a hole: the gate it configures is a
    pre-commit hook, which only ever runs where the file is. What must not happen is this file
    erroring out on the runners and being deleted for being flaky, so the skip says why.
    """
    if not CONFIG.is_file():
        pytest.skip(f"{CONFIG.name} is gitignored and absent here, so no hook reads it either")
    text = CONFIG.read_text(encoding="utf-8")
    hits = [ln for ln in text.splitlines()
            if ln.startswith("BLOCK_FUTURE_VERSIONS_REGEX=")]
    assert len(hits) == 1, (
        f"expected exactly one active BLOCK_FUTURE_VERSIONS_REGEX in {CONFIG.name}, found "
        f"{len(hits)}. A commented example plus an active line is fine; two active lines means the "
        f"hook uses one of them and this test checks the other.")
    return hits[0].split("=", 1)[1].strip().strip("'\"")


def _blocked(pattern, text):
    """Ask grep, not `re`. The hook uses `grep -E`, and the two dialects are not the same.

    A test that validated the pattern with Python's `re` would be checking a different engine from
    the one that decides whether a commit lands. This project has the phrase for it: run the gate's
    own command, not a command like it.
    """
    done = subprocess.run(["grep", "-qE", "-e", pattern], input=text, text=True, check=False)
    return done.returncode == 0


def test_the_pattern_is_set_at_all():
    """Unset means the hook falls back to the generic pattern this test exists to replace."""
    assert _pattern(), "no active pattern, so the hook uses its own fallback"


@pytest.mark.parametrize("text", MUST_REFUSE)
def test_a_real_future_version_is_still_refused(text):
    """Narrowing the pattern must not have opened the hole it exists to close."""
    assert _blocked(_pattern(), text), (
        f"{text!r} is a reference to a version after 0.4 and the gate would let it into public "
        f"docs. Baseline section 16 is the rule this breaks.")


@pytest.mark.parametrize("text", MUST_ALLOW)
def test_a_measured_figure_is_not_mistaken_for_a_version(text):
    """Every string here is a real figure from this project's docs."""
    assert not _blocked(_pattern(), text), (
        f"{text!r} is a measurement, not a version, and the gate refuses it. A gate that blocks "
        f"published measurements in a project built on published measurements gets bypassed, and "
        f"the bypass is worse than the false positive.")


def test_the_pattern_has_not_outlived_the_release_it_was_written_for():
    """The pattern hardcodes "0.4 is current". This fails the day that stops being true.

    A check pinned to a number that has since moved reports clean while being wrong, which is the
    failure this whole file is about. So the expiry is asserted rather than left in a comment.
    """
    try:
        from senbonzakura import __version__
    except ImportError:
        pytest.skip("the package is not importable here")

    series = ".".join(__version__.split(".")[:2])
    assert not _blocked(_pattern(), f"{series}.0"), (
        f"the shipping version is {__version__} and the future-version gate refuses {series}.0, so "
        f"it would block the release that is actually shipping.\n"
        f"  WIDEN THE PATTERN in .baseline-hook-config: the `0\\.([5-9]|...)` branches are written "
        f"for a 0.4 release line and have to move with it. Then add the new series' successors to "
        f"MUST_REFUSE in this file.")
