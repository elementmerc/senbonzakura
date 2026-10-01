# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura setup`'s verdict paragraph obeys the same column ceiling as everything else.

WHAT PROMPTED IT, 2026-10-01

A CLI surface sweep item: "`envsetup.py` wraps at `width=94`, above `say.CEILING` (79)." Measured
on an 80-column terminal before the fix, the blocked-no-torch verdict printed at **90 and 95
columns**:

      BLOCKED: torch is installed as a CPU build and this machine has an NVIDIA card, so every
        measurement here runs on the processor while a GPU sits idle beside it, which is the single

So the one message in the tool whose entire job is explaining a problem was the message that
wrapped twice: once here to 94, and again in the terminal at 80. A reader meeting a blocked
verdict is already stuck, and handing them a ragged paragraph is the worst moment to do it.

TWO DEFECTS, NOT ONE

The width was wrong, and the arithmetic was wrong in the same direction. The old code wrapped to a
budget and *then* prefixed two or four spaces onto each line, so the printed line was always wider
than the number it had been wrapped to. Both are closed by handing the indents to `textwrap`,
which makes the width in the call the width on the screen.

WHY THIS ASSERTS OVER EVERY PLANNER OUTCOME RATHER THAN ONE

The longest reason is 252 characters and belongs to the `blocked` verdict, which only appears when
torch is absent, which is the configuration a developer with a working environment never sees.
Picking one case would have tested whichever case the author happened to hold. So the test walks
every verdict `plan` can reach and holds all of them to the ceiling, which is also what stops a
future reason being written long enough to break the width again.

The narrow terminal case is here for the same reason in the other direction: `say.width()` floors
at 40, and a test that only ever runs at 80 would not notice a change that ignored the floor.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from senbonzakura import envsetup, say

#: Every configuration `plan` branches on, so the ceiling is asserted over all of them rather than
#: over whichever one this machine happens to be in. The names are the situation, not the verdict,
#: because the verdict is what is under test.
CONFIGURATIONS = {
    "torch is not installed at all": dict(
        system="Linux", machine="x86_64", gpus=None, driver=None,
        torch_version=None, variant=None, compute=None),
    "a CPU build beside an NVIDIA card": dict(
        system="Linux", machine="x86_64", gpus=["GPU 0: NVIDIA GeForce RTX 3060"],
        driver=(13, 3), torch_version="2.13.0+cpu", variant="cpu", compute=(8, 6)),
    "a CUDA build and a card that suits it": dict(
        system="Linux", machine="x86_64", gpus=["GPU 0: NVIDIA GeForce RTX 3060"],
        driver=(13, 3), torch_version="2.13.0+cu126", variant="cu126", compute=(8, 6)),
    "a CPU build and no card": dict(
        system="Linux", machine="x86_64", gpus=[], driver=None,
        torch_version="2.13.0+cpu", variant="cpu", compute=None),
    "apple silicon": dict(
        system="Darwin", machine="arm64", gpus=None, driver=None,
        torch_version="2.13.0", variant=None, compute=None),
    "a card older than the build expects": dict(
        system="Linux", machine="x86_64", gpus=["GPU 0: NVIDIA GeForce GTX 1060"],
        driver=(11, 8), torch_version="2.13.0+cu126", variant="cu126", compute=(6, 1)),
}


def _verdict_lines(configuration, monkeypatch):
    """The verdict paragraph as `senbonzakura setup` prints it, out of `main` itself.

    THE FIRST VERSION OF THIS HELPER DID NOT DO THAT, AND THE TEST WAS WORTHLESS BECAUSE OF IT.

    It called `plan` for the text and then re-wrapped it here with the corrected arguments, and the
    docstring claimed it drove `main`. Mutation-checked on 2026-10-01 by putting the `width=94`
    defect back: **25 of the 26 tests still passed**, because the helper was asserting on this
    file's arithmetic rather than on the module's. Only the source-text test noticed.

    That is the same defect as the thing the sweep is looking for, written into the sweep's own
    test: a verification that does not call the shipped code path is not a verification. It is also
    exactly the shape recorded against a rival's `TestNormPreservation`, which re-implements the
    projection locally and asserts on its own four lines of maths while the production path does
    something else.

    So this drives `main` with the probes monkeypatched, and the verdict block is picked out of
    real stdout.
    """
    monkeypatch.setattr(envsetup, "nvidia_gpus", lambda: configuration["gpus"])
    monkeypatch.setattr(envsetup, "driver_cuda_version", lambda: configuration["driver"])
    monkeypatch.setattr(envsetup, "compute_capability", lambda: configuration["compute"])
    monkeypatch.setattr(envsetup, "installed_torch",
                        lambda: (configuration["torch_version"], configuration["variant"]))
    monkeypatch.setattr(envsetup.platform, "system", lambda: configuration["system"])
    monkeypatch.setattr(envsetup.platform, "machine", lambda: configuration["machine"])

    out = io.StringIO()
    with redirect_stdout(out):
        envsetup.main([])
    printed = out.getvalue().splitlines()

    # The verdict paragraph is the run of lines starting at the one whose first non-space token is
    # the verdict, shouted, followed by its hanging-indented continuations. Found by shape rather
    # than by index, because `_describe` above it prints a variable number of lines.
    verdicts = tuple(f"  {v.upper()}: " for v in ("ok", "fix", "blocked"))
    start = next((i for i, line in enumerate(printed) if line.startswith(verdicts)), None)
    assert start is not None, "no verdict line in:\n" + "\n".join(printed)
    block = [printed[start]]
    for line in printed[start + 1:]:
        if not line.startswith("    "):
            break
        block.append(line)
    return block


@pytest.mark.parametrize("situation", sorted(CONFIGURATIONS))
@pytest.mark.parametrize("columns", [80, 120, 40])
def test_no_verdict_line_exceeds_the_ceiling(situation, columns, monkeypatch):
    """The rule the rest of the tool already follows, applied to the one paragraph that did not.

    120 is in the list deliberately: `say.width()` caps at `say.CEILING`, so a wide terminal must
    not widen this paragraph. A fix that read the terminal without the cap would pass at 80 and 40
    and fail here.
    """
    monkeypatch.setenv("COLUMNS", str(columns))
    lines = _verdict_lines(CONFIGURATIONS[situation], monkeypatch)
    assert lines, f"{situation}: the verdict printed nothing"
    for line in lines:
        assert len(line) <= say.CEILING, (
            f"{situation} at COLUMNS={columns}: {len(line)} columns against a ceiling of "
            f"{say.CEILING}: {line!r}")


@pytest.mark.parametrize("situation", sorted(CONFIGURATIONS))
def test_the_indent_is_two_then_four(situation, monkeypatch):
    """The shape is load-bearing: the hanging indent is what makes a wrapped verdict read as one
    paragraph rather than as several findings. Asserted because the fix moved responsibility for
    the indent from the print statement to textwrap, and a silent change of shape there would look
    like nothing.
    """
    monkeypatch.setenv("COLUMNS", "80")
    first, *rest = _verdict_lines(CONFIGURATIONS[situation], monkeypatch)
    assert first.startswith("  ") and not first.startswith("   "), repr(first)
    for line in rest:
        assert line.startswith("    "), f"a continuation line lost its hanging indent: {line!r}"


def test_the_longest_reason_is_the_one_a_working_machine_never_sees():
    """Guards the reason this test enumerates configurations instead of picking one.

    If the longest reason ever stops being the `blocked` one, the argument above is stale and
    somebody should re-read which case is worst rather than trusting this file's docstring.
    """
    lengths = {name: len("{}: {}".format(*envsetup.plan(**kw)[:2]))
               for name, kw in CONFIGURATIONS.items()}
    worst = max(lengths, key=lengths.get)
    assert worst == "torch is not installed at all", (
        f"the longest verdict is now {worst!r} at {lengths[worst]} characters; this file's "
        f"docstring explains the test's shape in terms of the blocked case and needs re-reading")
    assert lengths[worst] > say.CEILING, (
        "the longest reason now fits on one line, so this test can no longer fail for the reason "
        "it was written; check whether the reasons were shortened deliberately")


def test_the_source_wraps_to_the_shared_width_rather_than_a_number_of_its_own():
    """The 94 is gone and nothing put a bare number back.

    A line-length test passes if somebody hardcodes 79, and then drifts the day `CEILING` moves.
    This asserts the coupling rather than the consequence, which is the half a column count cannot
    see.
    """
    source = envsetup.main.__globals__["__file__"]
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "width=94" not in text, "the old 94-column budget is back"
    assert "width=say.width()" in text, (
        "the verdict no longer wraps to the shared width; a number of its own will drift from "
        "say.CEILING the next time that moves")
