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
    STARVATION_DEADLINE_S,
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


# ── the half the fixture above cannot see ─────────────────────────────────────────────────────────

def _governor_that_under_reports_its_own(free_bytes, reported_own, *, polls_allowed=10_000, **kw):
    """A card where `own_fn` reports LESS than we are actually using.

    WHY THIS HELPER EXISTS ALONGSIDE `_governor`, 2026-09-28.

    `_governor` derives `own` as `TOTAL - free - external`, so every byte not free belongs to us or
    to a named external holder. That is the right shape for the scenarios it was written for and it
    defines away the thing a reviewer found: the real `own_fn` is `torch.cuda.memory_reserved`, which
    counts the caching allocator and nothing else, so the CUDA context, the cuBLAS and cuDNN
    workspaces and any non torch allocation are all missing from it. `(total - free) - own` therefore
    attributes this process's own context to somebody else.

    On a 6 GB card with a 5 GB model and nothing else running, that was enough to clear
    `EXTERNAL_HOLD_FLOOR_BYTES`, so `_should_pause` stayed true and an unbounded wait waited for a
    process that does not exist to release memory that is ours.

    `polls_allowed` makes a broken deadline FAIL rather than hang. This file already records a suite
    that hung at the same test twice; a test for an unbounded wait must not be able to become one.
    """
    clock = {"t": 0.0, "polls": 0}

    def sleep(seconds):
        clock["polls"] += 1
        if clock["polls"] > polls_allowed:
            raise AssertionError(
                f"the governor polled {clock['polls']} times without returning, which is the wedge "
                f"this test exists to catch")
        clock["t"] += seconds

    logs = []
    gov = ResourceGovernor(
        "cuda:0", log=logs.append,
        mem_fn=lambda: (free_bytes, TOTAL),
        reclaim_fn=lambda: 0.0,
        own_fn=lambda: reported_own,
        empty_cache_fn=lambda: None,
        sleep_fn=sleep,
        clock=lambda: clock["t"],
        **kw)
    return gov, logs, clock


def test_the_external_figure_over_reports_and_the_test_suite_says_so_out_loud():
    """The mis-attribution is REAL and is not corrected here, which is worth pinning either way.

    `own_fn` is `torch.cuda.memory_reserved`, which counts the caching allocator and nothing else, so
    our own CUDA context and workspaces are attributed to another process. 0.2 GB free of 6 with the
    allocator reporting 5.0 GB leaves 0.8 GB unaccounted, twelve times the floor, and every byte of
    it is ours.

    A measured baseline of our own overhead was tried on 2026-09-28 and backed out: it has to be
    measured at some moment, and whatever sits on the card at that moment is absorbed, so an app that
    was already running became invisible and `test_external_pressure_triggers_pause` went green by
    losing the behaviour it is named after. Telling our context from their allocation needs
    per-process accounting, which this class avoids on purpose so that it works in WSL2, where the
    Windows-side app is invisible to nvidia-smi.

    So this asserts the over-report rather than its absence, and the wedge is closed by bounding the
    wait instead. If somebody makes the accounting exact, this test should fail and be deleted.
    """
    gov, _logs, _clock = _governor_that_under_reports_its_own(0.2e9, 5.0e9)
    assert gov._external_used() > EXTERNAL_HOLD_FLOOR_BYTES, (
        "the external figure no longer over-reports our own memory. If that is deliberate, the "
        "deadline below may no longer be the only thing standing between a full card and a wedge, "
        "and the comment in _should_pause needs rewriting.")
    assert gov._should_pause() is True, (
        "the starvation pause no longer fires on a card its own model fills, so either the "
        "accounting was fixed or the pause was turned off; find out which")


def test_a_starvation_wait_is_bounded_even_if_the_accounting_is_still_wrong():
    """The belt to the accounting's braces, and the reason both are here.

    The accounting fix narrows the mis-attribution; it cannot prove it gone on every driver and
    every card. An unbounded wait whose predicate can be permanently true is a wedge whatever the
    predicate is, so the starvation pause now has a deadline, and past it the run pushes on and says
    why. Worst case is an out of memory, which halves the batch; a run that never returns has no
    recovery at all.
    """
    # Nothing is ever released: free stays low and the reported own stays low, so with the overhead
    # baseline never taken (no calibration) the predicate holds for ever.
    gov, logs, clock = _governor_that_under_reports_its_own(0.2e9, 1.0e9)
    assert gov._should_pause() is True, "the precondition failed: this governor would not pause"

    waited = gov.wait_for_headroom()

    assert waited >= STARVATION_DEADLINE_S, (
        f"returned after {waited}s, before the deadline, so something else ended the wait")
    assert waited < STARVATION_DEADLINE_S + 60, f"overshot the deadline by a long way: {waited}s"
    printed = "\n".join(logs)
    assert "WARNING" in printed and "pushing on" in printed, (
        f"it gave up silently, so an operator cannot tell this from a normal resume: {printed!r}")
    assert "own" in printed, (
        "the message does not raise the possibility that the memory being waited for is this run's, "
        "which is the one explanation an operator cannot reach on their own")


def test_a_foreground_yield_is_still_patient():
    """Bounding that one would break a behaviour the operator opted into.

    Good-gaming-citizen mode exists to keep somebody's game smooth, and a game is not a wedge: the
    predicate is about another process, and it becomes false when they close it. So the deadline
    above is deliberately not applied here, and this pins that difference.
    """
    gov, logs, clock = _governor_that_under_reports_its_own(
        0.2e9, 1.0e9, background_mode=True, polls_allowed=400)
    assert gov._foreground_pressure() is True

    with pytest.raises(AssertionError, match="without returning"):
        gov.wait_for_headroom()
    assert any("yielding" in line for line in logs)
