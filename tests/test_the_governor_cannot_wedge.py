# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A batch=1 out-of-memory must always hand control back to the counter that can raise.

THE DEFECT, found by the 2026-09-25 panel and reproduced through this module's own injectable
hooks. `_forced_pause` waited for `grow_free_frac` (0.20), the threshold for growing the batch back,
and escaped only `if self.max_pause_s is not None`, which defaults to None at every construction
site. On a card the model itself fills, which is the only condition under which a batch=1 OOM
happens at all, the availability fraction is permanently below 0.20. So the function never returned,
`batch1_ooms` never reached 2, and the counter whose entire purpose is to say what to lower could
not fire. The run printed "VRAM OOM at batch=1 (1 of 20)" and then nothing, for as long as the
operator left it, having promised nineteen more attempts that would never come.

WHAT THIS FILE OWNS is not "the pause is bounded". It is the property that makes the bound matter:
**control returns to `run`'s counter, so a card that cannot hold one prompt produces a sentence
rather than a silence.** A future change that bounds the pause but leaves the counter unreachable
would pass a narrower test and reproduce the outage.
"""
import pytest

from senbonzakura.resources import (
    EXTERNAL_HOLD_FLOOR_BYTES,
    FORCED_PAUSE_DEADLINE_S,
    HEARTBEAT_S,
    ResourceGovernor,
)

TOTAL = 6.0e9          # the 6 GB card this project is sized around
OWN = 5.0e9            # the model itself, which is not somebody else's fault


def _governor(free_bytes, external=0.0, **kw):
    """A governor over a simulated card, with the clock and the sleeps injected.

    `external` is what something OTHER than senbonzakura holds, and it defaults to nothing. That
    default is load-bearing rather than convenient: `own` is then derived so that every byte not
    free belongs to us, which is the case this file is about, a card its own model fills.

    The first version of this helper hardcoded `own` at 5 GB while the fixture said 0.2 GB free of
    6 GB, which quietly asserted that 0.8 GB was held by somebody else. That is a different
    scenario from the one every test here claims to be testing, and it is why the suite hung
    rather than failed: the governor correctly waited for an external process that the fixture had
    invented and that would never release anything.
    """
    clock = {"t": 0.0}
    logs = []
    own = max(0.0, TOTAL - free_bytes - external)
    gov = ResourceGovernor(
        "cuda:0", log=logs.append,
        mem_fn=lambda: (free_bytes, TOTAL),
        reclaim_fn=lambda: 0.0,          # nothing to reclaim: the caller just emptied the cache
        own_fn=lambda: own,
        empty_cache_fn=lambda: None,
        sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
        clock=lambda: clock["t"],
        **kw)
    return gov, logs, clock


def test_the_pause_returns_when_the_model_itself_fills_the_card():
    """The exact scenario: 1 GB free of 6, nothing reclaimable. Availability sits at 16.7%, which is
    below the grow threshold of 20% and above the min of 6%, and that gap is where it hung.
    """
    gov, _logs, clock = _governor(1.0e9)
    assert gov.min_free_frac < gov._avail_frac() < gov.grow_free_frac, (
        "this test is only meaningful in the band between the two thresholds; if the defaults "
        "moved, pick a free-memory figure that lands back in it")
    waited = gov._forced_pause()
    assert waited == pytest.approx(gov.poll_s), (
        "the card has enough room to try one item again, so the pause is the single deliberate "
        "sleep and no more")
    assert clock["t"] < FORCED_PAUSE_DEADLINE_S


def test_the_pause_is_bounded_even_when_the_card_never_frees():
    gov, _logs, _clock = _governor(0.2e9)      # 3.3% available: genuinely starved
    assert gov._avail_frac() < gov.min_free_frac
    assert gov.max_pause_s is None, "the default is what the defect depended on"
    assert gov._forced_pause() == pytest.approx(FORCED_PAUSE_DEADLINE_S)


def test_an_operator_cap_still_wins_over_the_default_deadline():
    gov, _logs, _clock = _governor(0.2e9, max_pause_s=120.0)
    assert gov._forced_pause() == pytest.approx(120.0)


def test_the_counter_can_reach_its_refusal(monkeypatch):
    """The property the outage actually violated, driven through the public entry point."""
    gov, logs, _clock = _governor(0.2e9)

    class Boom(RuntimeError):
        pass

    gov._oom_types = (Boom,)
    monkeypatch.setattr("senbonzakura.resources._is_oom", lambda exc, types: isinstance(exc, Boom))

    def always_oom(_chunk):
        raise Boom("CUDA out of memory")

    with pytest.raises(RuntimeError, match="cannot hold one prompt"):
        gov.run(always_oom, ["one item"])
    assert any("of 20" in m for m in logs), "the run should count its attempts out loud"


def test_the_pause_says_something_when_it_genuinely_waits():
    gov, logs, _clock = _governor(0.2e9)
    gov._forced_pause()
    assert any("still too full" in m for m in logs), (
        "a wait the operator cannot see is indistinguishable from a wedge, which is what made "
        "this defect cost a night rather than a minute")


def test_the_pause_stays_quiet_when_it_does_not_wait():
    """One deliberate sleep is not an event. A line here would train the reader to skip them."""
    gov, logs, _clock = _governor(1.0e9)
    gov._forced_pause()
    assert logs == []


def test_the_unbounded_wait_keeps_speaking():
    """`wait_for_headroom` blocks on another process releasing the card and is unbounded by design,
    so the heartbeat is the only thing distinguishing it from a hang.

    `external` is passed because this is the one scenario in the file where waiting is the RIGHT
    behaviour: somebody else holds the card and may give it back. Without it the governor now
    declines to wait at all, which is the fix this file also covers, and the heartbeat would have
    nothing to report.
    """
    gov, logs, _clock = _governor(0.2e9, external=2.0e9, max_pause_s=200.0)
    gov.wait_for_headroom()
    beats = [m for m in logs if "still paused" in m]
    assert len(beats) >= 2, f"expected a heartbeat every {HEARTBEAT_S}s over 200s, got {len(beats)}"
    assert "Close whatever else is on the card" in beats[0], (
        "the heartbeat should say what the operator can do, since the card is often something "
        "they can free")


def test_every_heartbeat_interval_is_reachable_within_its_own_deadline():
    """The guard on the first version of this repair, which shipped a heartbeat that could not fire.

    A periodic log inside a loop bounded at the same interval is unreachable code. Any future
    deadline shorter than the heartbeat puts it back, silently.
    """
    assert HEARTBEAT_S >= FORCED_PAUSE_DEADLINE_S, (
        "if a forced pause may outlast the heartbeat interval, it needs the periodic heartbeat "
        "back; if it may not, the periodic form is unreachable there and must not be reintroduced")


def test_a_card_our_own_model_fills_does_not_wait_for_headroom():
    """The second wedge, and the one this file found by hanging the suite twice.

    `_avail_frac` counts driver-free VRAM plus senbonzakura's own RECLAIMABLE cache, so memory the
    model has ALLOCATED reads as unavailable. On a card the model fills, availability sits under
    `min_free_frac` with nothing external running, and `wait_for_headroom` is unbounded by design.
    So `run` waited forever before the first batch, for headroom that only unloading the model
    could provide, while the docstring said this could only happen when another process held the
    card.

    Waiting cannot change our own residency. Shrinking the batch can, and the OOM counter can say
    what to lower, so the run has to reach them.
    """
    gov, _logs, clock = _governor(0.2e9)          # every non-free byte is ours
    assert gov._avail_frac() < gov.min_free_frac, "the fixture must look starved"
    assert gov._external_used() <= EXTERNAL_HOLD_FLOOR_BYTES, "and nothing external holds it"
    assert gov.wait_for_headroom() == 0.0, (
        "a card filled by our own model must not pause: no amount of waiting makes us smaller")
    assert clock["t"] == 0.0


def test_a_card_someone_else_filled_still_waits():
    """The property the fix must not break. Patience is the right answer to another application."""
    gov, _logs, _clock = _governor(0.2e9, external=2.0e9, max_pause_s=20.0)
    assert gov.wait_for_headroom() > 0.0, (
        "when something else holds the card, waiting is what stops a run crashing on it")


def test_run_reaches_its_refusal_on_a_card_it_filled_itself(monkeypatch):
    """End to end: the wedge was that `run` never got past its first `wait_for_headroom`."""
    gov, logs, _clock = _governor(0.2e9)

    class Boom(RuntimeError):
        pass

    gov._oom_types = (Boom,)
    monkeypatch.setattr("senbonzakura.resources._is_oom", lambda exc, types: isinstance(exc, Boom))

    def always_oom(_chunk):
        raise Boom("CUDA out of memory")

    with pytest.raises(RuntimeError, match="cannot hold one prompt"):
        gov.run(always_oom, ["one item"])
    assert any("of 20" in m for m in logs)
