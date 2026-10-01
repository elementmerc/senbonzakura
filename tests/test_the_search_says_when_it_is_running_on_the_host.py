# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A run whose weights did not fit the card has to say so before the long part, not after.

WHAT PROMPTED IT, 2026-09-27

`offloaded_share` had exactly one caller: the capability probe, which refuses when placement would
make it slow. The search is the same fact applied to a job one to two orders of magnitude longer,
and it said nothing at all. So somebody on an undersized card got a careful warning about the
four-minute probe and silence about the four-hour search it was about to begin.

That is this project's recurring shape, stated in its own words: a guard that covers one spelling of
a defect reports clean on the others. `--device cuda` is a request, not a placement; accelerate
decides, and it puts whatever will not fit into host RAM or onto disk.

WHY IT REPORTS RATHER THAN REFUSES

The operator's call, and the right one for this path. An offloaded probe spends hours buying a
number the edit does not need, so refusing it is a kindness. An offloaded search is simply slow, and
refusing it would leave somebody with a 6 GB card unable to run the tool this project is aimed at.
"""
from __future__ import annotations

import pytest

from senbonzakura import capability


class _Model:
    """Only the attribute the reporter reads. A real model would need a GPU to place."""

    def __init__(self, device_map):
        self.hf_device_map = device_map


ALL_ON_GPU = _Model({f"layer.{i}": 0 for i in range(10)})
HALF_ON_HOST = _Model({**{f"layer.{i}": 0 for i in range(5)},
                       **{f"layer.{i}": "cpu" for i in range(5, 10)}})
SOME_ON_DISK = _Model({**{f"layer.{i}": 0 for i in range(8)},
                       "layer.8": "disk", "layer.9": "cpu"})
NO_MAP = _Model(None)


def _lines(model, **kw):
    out = []
    kw.setdefault("trials", 40)
    kw.setdefault("prompts_per_trial", 160)
    kw.setdefault("gen_tokens", 64)
    capability.report_offload_cost_for_a_search(model, log=out.append, **kw)
    return out


def test_a_model_wholly_on_the_card_says_nothing():
    """The common case must stay silent. A notice on every run is a notice nobody reads."""
    assert _lines(ALL_ON_GPU) == []


def test_a_model_with_no_device_map_says_nothing():
    """A single-device load carries no map, and absence of information is not a finding."""
    assert _lines(NO_MAP) == []


@pytest.mark.parametrize("model", [HALF_ON_HOST, SOME_ON_DISK])
def test_an_offloaded_model_is_reported(model):
    assert _lines(model), "a partly offloaded search said nothing about its placement"


def test_the_notice_says_the_run_still_works():
    """It is a slow run, not a broken one, and a reader who thinks it is broken kills it.

    HOST RAM ONLY. The disk case is the test below, and it is the opposite claim.
    """
    text = " ".join(_lines(HALF_ON_HOST))
    assert "works" in text, (
        "the notice has to say the run is fine, or somebody reaches for Ctrl+C on working work")
    assert "DISK" not in text, "nothing is on disk in this placement, so nothing should say so"


# ── disk is a different outcome, not a slower one ────────────────────────────────

def test_a_disk_placement_is_never_told_the_run_works():
    """The 2026-10-01 finding, and the whole reason the disk reading is taken separately.

    `offloaded_share` groups `cpu`, `disk` and `meta` because for pricing generation they are all
    "not the accelerator". This notice was built on that one number, so it could only ever say
    "slower", and it said "The run works" for a placement under which the edit does not run at
    all: `cli._real_tensor` raises on the first disk-offloaded writer it is asked to edit, because
    the offload map hands back a fresh tensor on every read and the bake would write into a copy
    that is discarded before the next forward pass.

    That is the same defect as `doctor`'s CPU advisory, which said "editing a model on CPU works
    and is slow" while the default flags refuse: **a reassurance built on a check that never
    tested the thing it was reassuring about.** Two instances of one pattern in one tool, so this
    one gets an assertion on the words rather than on the presence of words.

    `SOME_ON_DISK` was already a fixture here, parametrised into
    `test_an_offloaded_model_is_reported`, which asserts the notice says *something*. Nothing
    asserted WHAT, which is how the wrong sentence survived.
    """
    text = " ".join(_lines(SOME_ON_DISK))
    assert "works" not in text, (
        "a disk placement was told the run works. The edit is refused when it reaches a "
        "disk-offloaded weight, so this promises an outcome the tool does not deliver.")
    assert "DISK" in text, "the notice has to name disk, because it is the thing that decides"
    assert "not run" in text or "refused" in text, (
        "the notice has to say the edit will not run, which is the fact a reader needs before "
        "spending an hour finding out")


def test_a_disk_placement_is_told_what_to_do_instead():
    """A refusal that names no way forward sends somebody to the issue tracker."""
    text = " ".join(_lines(SOME_ON_DISK))
    assert "host RAM" in text or "RAM" in text, "adding host RAM is the fix that makes it editable"
    assert "--load-in-4bit" in text, "the flag that makes it fit has to be named"


def test_a_disk_placement_is_reported_at_any_token_budget():
    """The disk notice must not sit behind the rate arithmetic, and it did in the first draft.

    `gen_tokens` arrives as 0 or None on the paths that inject a model, and the function returns
    early on that because there is no rate to project. A disk placement refuses the edit at any
    budget, so a notice that returned first would be silent on exactly the paths most likely to
    be driven from a script. Caught 2026-10-01 by running the zero-budget case against the new
    branch rather than by reading it.
    """
    for kw in ({"gen_tokens": 0}, {"gen_tokens": None},
               {"trials": 0}, {"prompts_per_trial": None}):
        text = " ".join(_lines(SOME_ON_DISK, **kw))
        assert "DISK" in text, (
            f"a disk placement said nothing with {kw}; the placement fact does not depend on the "
            f"budget and must be reported before the rate is worked out")


def test_the_disk_reading_counts_only_disk():
    """`cpu` and `meta` are not disk, and a notice that lumped them would fire on every offload."""
    on_disk, total = capability.disk_offloaded_entries(SOME_ON_DISK)
    assert (on_disk, total) == (1, 10), f"expected exactly one disk entry of ten, got {on_disk}/{total}"
    assert capability.disk_offloaded_entries(HALF_ON_HOST)[0] == 0, "host RAM is not disk"
    assert capability.disk_offloaded_entries(ALL_ON_GPU)[0] == 0
    assert capability.disk_offloaded_entries(NO_MAP) == (0, 0), "no map is not a disk placement"


def test_the_notice_carries_a_rate_and_owns_that_it_is_a_projection():
    """A number with no provenance is the thing this project keeps withdrawing.

    The operator asked for a projected rate rather than a bare warning, so the rate has to be there
    AND has to say it is projected, from what, and which way it errs.
    """
    text = " ".join(_lines(HALF_ON_HOST))
    assert "hours" in text, "no rate was given, which was the point of reporting at all"
    assert "projection" in text, "a figure that does not say it is projected reads as measured"
    assert "1.7B" in text, "the projection's provenance is the model its constant was measured on"
    assert "under-states" in text, "which direction it errs in is what makes it usable"


def test_the_notice_says_what_to_do_about_it():
    """Every refusal and warning in this project names a next step; this one is no different."""
    text = " ".join(_lines(HALF_ON_HOST))
    assert "--load-in-4bit" in text
    assert "--device cuda" in text, (
        "the whole trap is that --device cuda looks like it settled the question, so the notice "
        "has to say outright that it does not")


def test_more_offload_projects_a_longer_run():
    """The share is a multiplier, so the figure has to move with it rather than be decorative."""
    def hours(share_of_ten):
        model = _Model({**{f"layer.{i}": 0 for i in range(10 - share_of_ten)},
                        **{f"layer.{i}": "cpu" for i in range(10 - share_of_ten, 10)}})
        text = " ".join(_lines(model))
        return float(text.split("roughly ")[1].split(" hours")[0])

    assert hours(2) < hours(5) < hours(9), (
        "the projected time did not grow with the offloaded fraction, so it is not reading it")


def test_a_bigger_search_projects_a_longer_run():
    """Trials and prompts both multiply the generation count, so both must reach the figure."""
    def hours(**kw):
        text = " ".join(_lines(HALF_ON_HOST, **kw))
        return float(text.split("roughly ")[1].split(" hours")[0])

    base = hours(trials=10, prompts_per_trial=100)
    assert hours(trials=100, prompts_per_trial=100) > base
    assert hours(trials=10, prompts_per_trial=1000) > base


def test_a_zero_budget_does_not_divide_by_anything():
    """Defaults can be 0 or absent on paths that inject a model, and a crash here stops a real run.

    This reports on the way into the longest job the tool has. It must never be the reason a run
    fails to start.
    """
    for kw in ({"trials": 0}, {"prompts_per_trial": 0}, {"gen_tokens": 0},
               {"trials": None}, {"prompts_per_trial": None}, {"gen_tokens": None}):
        _lines(HALF_ON_HOST, **kw)
