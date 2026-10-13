# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`ResourceGovernor.run` declared a contract in its docstring and enforced nothing.

THE DEFECT, found 2026-10-01 during a ledger sweep. `run` says `fn` "takes a list of items and
returns a list of results of the same length". Both call sites did `out.extend(fn(chunk))` and then
advanced the cursor by the chunk size, with no comparison between the two. A worker returning the
wrong count therefore shifted every LATER result against its item, by an offset that drifts as the
batch size floats with free VRAM.

WHY IT IS THE WORST SHAPE AVAILABLE HERE. Every batched measurement in this project runs through
this method, and callers pair prompts with results by position. A misalignment does not surface as
an error; it surfaces as a refusal rate, with one prompt's generation scored against another
prompt's label for the remainder of the run. The project already has this scar once:
`senbon-track-35axis-clean` had its partition manifest read loosely and sliced every published
number off the wrong rows, unnoticed for months.

The batch size is the aggravating factor rather than an incidental detail. On the governed path it
shrinks on OOM and grows on headroom, so the offset is not a constant anybody would spot in a
diff; it changes with how busy the card was.
"""
import re
import time

import pytest

from senbonzakura.resources import ResourceGovernor


def _governor(**kw):
    """A governor on the disabled path, which is the one that runs without a card."""
    return ResourceGovernor("cpu", lambda *_a, **_k: None,
                            enabled=False, max_batch=4, **kw)


def test_a_well_behaved_worker_is_untouched():
    got = _governor().run(lambda chunk: [x * 2 for x in chunk], list(range(10)))
    assert got == [x * 2 for x in range(10)]


def test_results_stay_paired_with_their_items_across_several_batches():
    """The property that matters, stated as the pairing rather than as the count."""
    items = [f"item-{i}" for i in range(11)]
    got = _governor().run(lambda chunk: [f"ans({x})" for x in chunk], items)
    assert got == [f"ans({x})" for x in items]


@pytest.mark.parametrize(("shape", "expected_in_message"), [
    ("short", "3 result(s) for a chunk of 4"),
    ("long", "5 result(s) for a chunk of 4"),
    ("empty", "0 result(s) for a chunk of 4"),
])
def test_a_wrong_length_return_is_refused_loudly_on_the_first_batch(shape, expected_in_message):
    returned = {"short": lambda c: list(c)[:-1],
                "long": lambda c: [*list(c), "extra"],
                "empty": lambda c: []}[shape]
    with pytest.raises(RuntimeError, match="paired with items by position"):
        _governor().run(returned, list(range(8)))
    with pytest.raises(RuntimeError, match=re.escape(expected_in_message)):
        _governor().run(returned, list(range(8)))


def test_the_refusal_names_where_it_happened_not_just_that_it_did():
    """A later batch failing must say which item, because that is the diagnostic.

    The first batch is fine and the second is short, so a reader needs "at item 4" to know the
    output before it was sound. Without the offset the message cannot distinguish a worker that was
    always wrong from one that broke partway through a long run.
    """
    calls = {"n": 0}

    def flaky(chunk):
        calls["n"] += 1
        return list(chunk) if calls["n"] == 1 else list(chunk)[:-1]

    with pytest.raises(RuntimeError, match="at item 4"):
        _governor().run(flaky, list(range(8)))


def test_a_generator_is_accepted_rather_than_counted_as_zero():
    """`len()` on a generator raises, and a worker yielding its results is a reasonable worker.

    Materialising before comparing is what makes the check safe to add to a path it did not
    previously constrain. Getting this wrong would turn every streaming worker into a hard failure,
    which is a worse defect than the one being fixed.
    """
    got = _governor().run(lambda chunk: (x + 1 for x in chunk), list(range(6)))
    assert got == [1, 2, 3, 4, 5, 6]


def test_nothing_is_returned_for_no_items_and_the_worker_is_not_called():
    called = []
    def worker(chunk):
        called.append(chunk)
        return list(chunk)

    assert _governor().run(worker, []) == []
    assert called == []


def test_both_call_sites_are_guarded_not_only_the_governed_one():
    """The disabled path had the same hole and is the path every CPU run takes.

    Checked in the source because the governed branch needs a card to reach. A fix applied to one
    branch and not the other is this project's most repeated defect: a guard that covers one
    spelling of a defect reports clean on the others.
    """
    import inspect
    src = inspect.getsource(ResourceGovernor.run)
    assert "out.extend(fn(" not in src, (
        "a call site still extends the output with an unchecked worker return")
    assert src.count("self._checked(") == 2, (
        f"expected both call sites to go through _checked, found {src.count('self._checked(')}")


# ── The heartbeat, added 2026-10-10 ─────────────────────────────────────────
#
# A journey run that day recorded 250 seconds of complete silence from this governor during
# direction extraction on a CPU, which is the gap that makes somebody kill a healthy run. Baseline
# 2.1 asks for a heartbeat every 30 to 60 seconds on every long-running loop and this loop had
# none. These drive it rather than reading it: the rate limit is set low so a test can see several
# beats without waiting half a minute.

def test_a_long_run_emits_a_heartbeat_rather_than_going_silent():
    seen = []
    gov = ResourceGovernor("cpu", seen.append, max_batch=2, enabled=False, heartbeat_s=0.05)

    def slow(chunk):
        time.sleep(0.06)
        return list(chunk)

    out = gov.run(slow, list(range(10)))
    assert out == list(range(10)), "the heartbeat must not disturb what the governor returns"
    beats = [m for m in seen if "captured" in m]
    assert beats, "a loop that took half a second with a 0.05s heartbeat emitted nothing"
    assert any("/10 captured" in m for m in beats), f"the beat does not name the total: {beats[:2]}"


def test_the_heartbeat_is_rate_limited_rather_than_per_batch():
    """A line per batch is how a heartbeat gets switched off.

    A fast run does hundreds of batches a minute, and one line each is noise rather than
    reassurance. The contract is a beat at most every `heartbeat_s`, not a beat per chunk.
    """
    seen = []
    gov = ResourceGovernor("cpu", seen.append, max_batch=1, enabled=False, heartbeat_s=3600.0)
    gov.run(lambda chunk: list(chunk), list(range(50)))
    beats = [m for m in seen if "captured" in m]
    assert not beats, (
        f"50 batches under an hour-long heartbeat should emit no beat, emitted {len(beats)}")


def test_an_empty_item_list_emits_no_heartbeat_and_does_not_raise():
    seen = []
    gov = ResourceGovernor("cpu", seen.append, max_batch=4, enabled=False, heartbeat_s=0.0)
    assert gov.run(lambda chunk: list(chunk), []) == []
    assert not [m for m in seen if "captured" in m]
