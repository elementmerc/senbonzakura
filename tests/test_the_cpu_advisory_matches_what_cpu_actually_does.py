# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`doctor` promised a slow success on CPU where the tool delivers a refusal.

FOUND BY A READER WITH NO KNOWLEDGE OF THE PROJECT, 2026-09-27. On a machine with no card,
`senbonzakura doctor` printed:

    !  torch  2.14.0+cu130, no cuda device
       -> editing a model on CPU works and is slow; scoring is fine

An edit on CPU does not work on the default flags. `capability.refuse_slow_cpu_probe` prices the
default 200-item, 512-token capability probe at roughly four hours of host generation and exits
rather than spending it. The refusal itself is good: it names three ways out and explains why it
stopped. The advisory describing it was the wrong shape, and the failure it causes is the expensive
one, a reader who was told this would work hunting for a fault in their install.

The two tests here are a pair on purpose. The first checks the advisory says what happens; the
second checks that it is still TRUE, by asking the same question the guard asks. If the defaults
ever move far enough that a CPU edit stops refusing, the second one fails and the advisory gets
revisited instead of quietly going stale in the other direction.
"""
import sys
import types

from senbonzakura import capability, doctor


def _torch_without_a_card(monkeypatch):
    fake = types.SimpleNamespace(
        __version__="fake",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_the_no_card_advisory_says_an_edit_refuses_and_names_both_ways_past_it(monkeypatch):
    _torch_without_a_card(monkeypatch)
    c = doctor.check_torch()
    assert c.status == "warn"
    assert "no cuda device" in c.detail
    fix = c.fix
    assert "refuses" in fix, (
        "the advisory has to say the run stops. It used to say editing 'works and is slow', which "
        "is the one outcome the default flags do not produce.")
    for flag in ("--capability-n 0", "--slow-probe-ok"):
        assert flag in fix, (
            f"the advisory does not name {flag}. The refusal it is describing names three ways "
            f"out; an advisory that names none of them leaves the reader with a dead end.")


def test_a_cpu_edit_on_the_defaults_really_does_get_refused():
    """The claim above, asked of the guard rather than of the prose."""
    from senbonzakura.parser import build_parser

    defaults = {}
    for action in build_parser()._actions:
        for opt in action.option_strings:
            if opt in ("--capability-n", "--capability-max-new"):
                defaults[opt] = action.default

    assert set(defaults) == {"--capability-n", "--capability-max-new"}, (
        f"the capability probe's default flags have been renamed: found {sorted(defaults)}. The "
        f"advisory in doctor.check_torch names them, so it is now wrong too.")

    seconds = capability.cpu_probe_estimate(
        defaults["--capability-n"], defaults["--capability-max-new"])
    assert seconds > capability.CPU_REFUSE_AFTER_SECONDS, (
        f"at the shipped defaults the CPU capability probe is estimated at {seconds / 3600:.2f} "
        f"hours, which is under the {capability.CPU_REFUSE_AFTER_SECONDS / 3600:.2f} hour refusal "
        f"threshold, so a CPU edit no longer refuses. Good news, and it means the doctor advisory "
        f"that says it does is now the stale one.")
