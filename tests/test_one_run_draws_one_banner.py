# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One run draws one banner, and a guided run does not answer "yes" with a second identity.

WHAT PROMPTED IT, 2026-09-28. A `senbonzakura interactive` session drew the petal banner at
startup, asked its questions, and then answered "Run it? [y/N]: y" with the block-capitals banner:
two designs and two version lines in one run. Three call sites emit one (`entry.main`, `cli.main`,
and `interactive.run` re-entering `cli.main`), and `banner.choose` picks at RANDOM per call, so the
second was usually a different design from the first.

`cli.main` already had `emit_banner=False` for exactly this and nothing in `src/` or `tests/` ever
passed it. A switch nobody flips is not a control.
"""
from __future__ import annotations

import io

from senbonzakura import banner


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_the_second_call_in_a_process_draws_nothing():
    banner.reset_for_tests()
    first, second = _Terminal(), _Terminal()
    banner.emit("0.4.0", first, env={"SENBON_BANNER": "gokei"})
    banner.emit("0.4.0", second, env={"SENBON_BANNER": "block"})
    assert first.getvalue(), "the first call drew no banner at all"
    assert not second.getvalue(), (
        "a second banner was drawn in one process. entry.main, cli.main and interactive.run all "
        "ask for one, and choose() randomises, so this is two identities in a single run.")


def test_the_two_designs_would_otherwise_differ():
    """The guard matters BECAUSE the designs differ; if they were identical this would be cosmetic.

    Asserted rather than assumed so nobody later reads the guard as redundant.
    """
    banner.reset_for_tests()
    a = banner.render("gokei", "0.4.0")
    b = banner.render("block", "0.4.0")
    assert a != b
    assert len(b.splitlines()) != len(a.splitlines()), (
        "the two designs are the same height, so the visible jump this guards against is gone and "
        "this test should be re-read rather than trusted")


def test_the_flag_that_suppresses_it_still_suppresses_it():
    banner.reset_for_tests()
    out = _Terminal()
    banner.emit("0.4.0", out, env={"SENBON_BANNER": "off"})
    assert not out.getvalue()


def test_a_suppressed_banner_does_not_consume_the_one_draw():
    """`off` means none at all, not "the first one is spent"."""
    banner.reset_for_tests()
    banner.emit("0.4.0", _Terminal(), env={"SENBON_BANNER": "off"})
    after = _Terminal()
    banner.emit("0.4.0", after, env={"SENBON_BANNER": "gokei"})
    assert after.getvalue(), (
        "a suppressed banner marked the process as having drawn one, so turning suppression off "
        "mid-process would then draw nothing")
