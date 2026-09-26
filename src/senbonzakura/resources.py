# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Adaptive GPU resource governor for senbonzakura.

Abliterating a model is heavy, sustained GPU work. If you open a game or a browser mid-run, the
card can be starved of VRAM and either the run crashes or the game stutters. The governor makes
senbonzakura a good neighbour: it watches the card's free VRAM and

  * shrinks the generation batch when free VRAM falls (and grows it back when the space returns),
  * catches an out-of-memory error and retries on a smaller batch instead of crashing, and
  * pauses entirely when even a batch of one will not fit, then resumes the instant space frees,
    any number of times during a single run.

It is a no-op on CPU (there is no VRAM to police), and every external dependency (the memory
reading, the cache flush, the sleep, the clock, the out-of-memory exception type) is injectable so
the whole policy is unit-testable on a machine with no GPU.
"""

from __future__ import annotations

import time


def _torch():
    # Imported lazily so importing this module never forces torch, and tests can run without it.
    import torch
    return torch


def _cuda_index(device):
    torch = _torch()
    return int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()


def cuda_free_total(device="cuda:0"):
    # (free, total) VRAM in bytes for the given cuda device, or None when torch/cuda is unavailable
    # or the device is not a cuda device. mem_get_info reports the WHOLE card's free memory, so it
    # sees VRAM a game or another process has taken, which is exactly what the governor must react to.
    try:
        torch = _torch()
        if not (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()):
            return None
        free, total = torch.cuda.mem_get_info(_cuda_index(device))
        return int(free), int(total)
    except Exception:
        return None


def cuda_reclaimable(device="cuda:0"):
    # Bytes senbonzakura already holds in torch's caching allocator that it could reclaim on demand
    # (reserved minus actually-allocated). This matters because raw "free VRAM" counts that cache as
    # used, so a model that has run a few batches looks starved when it is not: the memory is its own
    # and empty_cache() would hand it straight back. Returns 0 when it cannot be read.
    try:
        torch = _torch()
        if not (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()):
            return 0
        idx = _cuda_index(device)
        return int(torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx))
    except Exception:
        return 0


def cuda_own_reserved(device="cuda:0"):
    # Bytes senbonzakura itself has reserved on the card (torch's caching allocator total). Subtracting
    # this from the card's total-used leaves what OTHER processes hold, which is how the governor spots
    # a foreground game on WSL2, where the Windows-side game process is invisible to nvidia-smi but its
    # VRAM still shows up in the card's total-used. Returns 0 when it cannot be read.
    try:
        torch = _torch()
        if not (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()):
            return 0
        return int(torch.cuda.memory_reserved(_cuda_index(device)))
    except Exception:
        return 0


def _is_oom(exc, extra_types=()):
    # True for a CUDA out-of-memory error across torch versions: the dedicated OutOfMemoryError on
    # newer torch, or a RuntimeError whose message says so on older ones.
    if extra_types and isinstance(exc, tuple(extra_types)):
        return True
    try:
        from torch.cuda import OutOfMemoryError
        if isinstance(exc, OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


#: How long a single forced pause may run before control returns to the caller, when the operator
#: has set no `--max-pause`.
#:
#: NOT A TIMEOUT ON THE PROBLEM. It is a deadline on `_forced_pause`, and it exists because the
#: thing that can turn a full card into a sentence a user can act on is the batch=1 OOM counter,
#: which only advances when the pause returns. Thirty seconds times twenty permitted OOMs is ten
#: minutes of patience before the run names what to lower, which is generous for a transient (a
#: game closing, another notebook exiting) and finite for the case that is not transient.
FORCED_PAUSE_DEADLINE_S = 30.0

#: How much VRAM something other than senbonzakura must hold before a starvation pause is worth
#: waiting out.
#:
#: NOT A TUNING KNOB, A QUESTION OF ATTRIBUTION. The pause exists so that another application
#: taking the card does not crash a run; it cannot help when the run's own model is what fills the
#: card, because no amount of waiting will make us smaller. Below this floor, nothing external is
#: meaningfully holding memory and a shortfall is ours to handle by shrinking the batch or by
#: refusing, not by waiting. 64 MB is comfortably under a desktop compositor and comfortably over
#: measurement noise.
EXTERNAL_HOLD_FLOOR_BYTES = 64 * 1024 * 1024

#: How often a wait says it is still waiting.
#:
#: Every long wait in this class was announced once and then went quiet, which is the property that
#: made a wedge indistinguishable from patience on 2026-09-25. The baseline asks for a heartbeat
#: every 30 to 60 seconds on any long-running loop; this is the low end of it, because the thing
#: being waited on is a card the user may be able to free by closing something.
HEARTBEAT_S = 30.0


def fmt_duration(seconds):
    # Compact human duration: "45s", "12m 30s", "2h 05m". Negative/NaN guarded to "0s".
    try:
        s = int(max(0, round(seconds)))
    except (ValueError, OverflowError):
        return "0s"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


class ResourceGovernor:
    """Run batched GPU work under live free-VRAM pressure without crashing.

    The one entry point callers use is :meth:`run`, which takes a worker ``fn(list) -> list`` and the
    full item list, and drives the chunking itself: it picks a batch size from the current headroom,
    shrinks on out-of-memory, and pauses when the card is too full even for a single item. On CPU (or
    when disabled) it just calls ``fn(items)`` once and returns, so the caller path is identical.
    """

    def __init__(self, device, log=None, *, min_free_frac=0.06, grow_free_frac=0.20,
                 poll_s=2.0, max_pause_s=None, max_batch=16, enabled=True, max_batch1_ooms=20,
                 background_mode=False, external_pressure_mb=500,
                 mem_fn=None, reclaim_fn=None, own_fn=None, empty_cache_fn=None, sleep_fn=None,
                 clock=None, oom_types=()):
        self.device = str(device)
        self.log = log or (lambda _m: None)
        # Thresholds are on AVAILABILITY = (free VRAM + senbon's own reclaimable cache) / total, not
        # raw free, so a model that simply fills the card does not read as starved: only memory taken
        # by ANOTHER process (a game, a browser) pulls availability down and trips a pause.
        self.min_free_frac = min_free_frac        # pause below this availability (external starvation)
        self.grow_free_frac = grow_free_frac      # grow the batch back only above this availability
        self.poll_s = poll_s
        self.max_pause_s = max_pause_s             # None = wait indefinitely for headroom
        # HOW MANY TIMES A SINGLE ITEM MAY FAIL BEFORE THE RUN SAYS SO. There was no cap: an OOM
        # at batch 1 cannot shrink, so the driver paused and retried forever. On the 6 GB card
        # this project is built around, a run that cannot fit one prompt wedged overnight instead
        # of naming what to lower. Twenty is generous for a transient (another process closing a
        # window) and finite for the case that is not transient.
        self.max_batch1_ooms = max(1, int(max_batch1_ooms))
        self.max_batch = max(1, int(max_batch))
        self.cur_batch = self.max_batch
        # Good-gaming-citizen mode: yield COMPUTE (pause generation), not just react to a VRAM crash.
        # A foreground game shares the card's compute even when VRAM is fine, so under background_mode
        # the governor pauses generation whenever an external process is holding more than
        # `external_pressure_mb` of VRAM, freeing the GPU for the game, and resumes when it closes.
        self.background_mode = bool(background_mode)
        self.external_pressure_bytes = max(0, int(external_pressure_mb)) * 1024 * 1024
        self._mem_fn = mem_fn or (lambda: cuda_free_total(self.device))
        self._reclaim_fn = reclaim_fn or (lambda: cuda_reclaimable(self.device))
        self._own_fn = own_fn or (lambda: cuda_own_reserved(self.device))
        self._sleep = sleep_fn or time.sleep
        self._clock = clock or time.monotonic
        self._oom_types = tuple(oom_types)
        self._empty_cache = empty_cache_fn or self._default_empty_cache
        # Active only when asked AND there is a real cuda card to police.
        self.enabled = bool(enabled) and self._mem_fn() is not None
        self.paused_s = 0.0                        # cumulative time spent paused this run (for ETA)
        # What actually happened, for the run's artefact. A number produced under a floating batch
        # is not reproducible on that axis, and the artefact has to be able to say so.
        self.batch_sizes = {}                      # realised chunk size -> how many chunks
        self.oom_shrinks = 0                       # times an OOM halved the working batch
        self.pauses = 0                            # times generation yielded the card
        self._ok_streak = 0                        # consecutive full-size successes, gates growth
        self._calibrated = False                   # has the startup VRAM baseline been announced

    def _default_empty_cache(self):
        try:
            _torch().cuda.empty_cache()
        except Exception:
            pass

    # ── VRAM sensing ────────────────────────────────────────────────────────────────
    def _avail_frac(self):
        # Availability = (driver-free VRAM + senbon's own reclaimable cache) / total. Its own cache
        # is memory it can hand back instantly, so counting it keeps a full-but-healthy run from
        # reading as starved; only another process taking the card pulls this fraction down.
        ft = self._mem_fn()
        if ft is None:
            return 1.0
        free, total = ft
        if not total:
            return 1.0
        reclaim = max(0, self._reclaim_fn() or 0)
        return (free + reclaim) / total

    def _external_used(self):
        # VRAM held by processes OTHER than senbon: the card's total-used minus senbon's own
        # reservation. WSL2-safe by construction: it needs no process list (the Windows-side game is
        # invisible to nvidia-smi in WSL2), only the card total-used and torch's own reserved bytes.
        ft = self._mem_fn()
        if ft is None:
            return 0
        free, total = ft
        own = max(0, self._own_fn() or 0)
        return max(0, (total - free) - own)

    def _foreground_pressure(self):
        # True when a foreground GPU app (a game) is on the card: external VRAM exceeds the threshold.
        # Absolute, not baseline-relative, so it fires even for a game that was already running when
        # senbon started. Armed only in background (good-gaming-citizen) mode.
        if not self.background_mode:
            return False
        return self._external_used() > self.external_pressure_bytes

    def _should_pause(self):
        # Pause for either reason: a VRAM crash risk (another app dropped usable VRAM too low) or,
        # in background mode, a foreground app on the card that wants the compute.
        #
        # THE STARVATION PAUSE NOW ASKS WHO TOOK THE MEMORY, and that is the second half of the
        # 2026-09-25 wedge. `_avail_frac` counts driver-free VRAM plus senbon's own RECLAIMABLE
        # cache, and its comment claimed "only another process taking the card pulls this fraction
        # down". That is false for memory the model has ALLOCATED rather than cached: a 5 GB model
        # on a 6 GB card with nothing else running reads as starved. `wait_for_headroom` is
        # unbounded by design, so `run` waited forever, before the first batch, for headroom that
        # only unloading the model could ever provide. Found by the test written to prove the
        # OTHER wedge was fixed, which hung the suite at the same test twice.
        #
        # Waiting is the right answer to somebody else holding the card and the wrong answer to
        # ourselves holding it: patience cannot change our own residency, and the batch sizer and
        # the OOM counter are what handle a card we have filled. So low availability only pauses
        # when something external is actually holding memory.
        if self._foreground_pressure():
            return True
        if self._avail_frac() >= self.min_free_frac:
            return False
        return self._external_used() > EXTERNAL_HOLD_FLOOR_BYTES

    def _calibrate_once(self):
        # On the first real batch, announce the operating baseline: how much of the card senbon has
        # to work with. This is the "check VRAM first, run a throttled baseline" step, so the run
        # sizes itself to what is actually free instead of assuming it owns the whole card.
        if self._calibrated or not self.enabled:
            return
        self._calibrated = True
        ft = self._mem_fn()
        if ft:
            avail = self._avail_frac() * ft[1]
            self.log(f"  VRAM baseline: ~{avail / 1e9:.1f} GB usable of {ft[1] / 1e9:.1f} GB; sizing "
                     f"batches to fit and pausing only if another app drops usable VRAM below "
                     f"{self.min_free_frac * 100:.0f}%")
            if self.background_mode:
                ext_gb = self._external_used() / 1e9
                thr_gb = self.external_pressure_bytes / 1e9
                self.log(f"  good-gaming-citizen mode on: ~{ext_gb:.1f} GB external now; will yield the "
                         f"GPU (pause generation) whenever a foreground app holds more than "
                         f"{thr_gb:.1f} GB, and resume when it closes")

    # ── pause / resume ───────────────────────────────────────────────────────────────
    def wait_for_headroom(self):
        """Block until usable VRAM recovers above ``min_free_frac``, polling and flushing the cache.

        "Usable" is availability (free + senbon's own reclaimable cache), so this only ever blocks
        when ANOTHER process is holding the card, not because the model fills it. Returns the seconds
        spent waiting (0 when there was headroom).

        ``max_pause_s`` caps it WHEN THE OPERATOR SETS ONE, and its default is None. This docstring
        used to say the cap meant a mismeasuring driver "can never wedge a run forever", which was
        not true of any default invocation: nothing in the CLI passes `--max-pause`. The wait is
        deliberately unbounded here, because the condition it waits on is another process releasing
        the card and pushing on regardless would crash a run that only needed to be patient. What
        it now does instead is SAY SO, every ``HEARTBEAT_S``, so a person watching can tell waiting
        from wedged. That distinction is the whole of the 2026-09-25 finding.
        """
        if not self.enabled:
            return 0.0
        waited = 0.0
        announced = False
        announced_at = 0.0
        while self._should_pause():
            if not announced:
                if self._foreground_pressure():
                    ext_gb = self._external_used() / 1e9
                    self.log(f"  yielding: a foreground app is on the card (~{ext_gb:.1f} GB external); "
                             "pausing generation so it stays smooth, will resume when it closes")
                else:
                    usable_gb = self._avail_frac() * (self._mem_fn()[1] if self._mem_fn() else 0) / 1e9
                    self.log(f"  paused: another app is using the card, only ~{usable_gb:.1f} GB usable; "
                             "waiting for it to free up")
                announced = True
                self.pauses += 1
            # empty_cache here also hands senbon's own idle cache back to the driver, so a foreground
            # game gets a little VRAM too while generation is paused (compute yield, partial VRAM relief).
            self._empty_cache()
            self._sleep(self.poll_s)
            waited += self.poll_s
            self.paused_s += self.poll_s
            # Announced once and then silent, until 2026-09-25. An operator cannot tell a wait
            # from a wedge without this, and the card is often something they could free.
            if waited - announced_at >= HEARTBEAT_S:
                announced_at = waited
                self.log(f"  still paused for VRAM: {fmt_duration(waited)} so far, "
                         f"{self._avail_frac():.0%} usable, need {self.min_free_frac:.0%}. "
                         f"Close whatever else is on the card, or pass --max-pause to push on")
            if self.max_pause_s is not None and waited >= self.max_pause_s:
                self.log(f"  resuming after {fmt_duration(waited)} paused (max-pause reached)")
                return waited
        if announced:
            self.log(f"  resumed: ~{self._avail_frac() * (self._mem_fn()[1] if self._mem_fn() else 0) / 1e9:.1f} "
                     "GB usable again, generation continuing")
        return waited

    # ── batch sizing ─────────────────────────────────────────────────────────────────
    def _grow_maybe(self, used_bs):
        # Ramp the batch back toward the ceiling once the card is comfortably free again, so the run
        # speeds up when a game is closed. Requires a couple of clean full-size batches first, to
        # avoid oscillating on a card that is right at the edge.
        if used_bs >= self.cur_batch and self._avail_frac() >= self.grow_free_frac:
            self._ok_streak += 1
            if self._ok_streak >= 2 and self.cur_batch < self.max_batch:
                self.cur_batch = min(self.max_batch, self.cur_batch * 2)
                self._ok_streak = 0
        else:
            self._ok_streak = 0

    def _shrink(self):
        # Halve the working batch after an out-of-memory; returns True if there was room to shrink.
        if self.cur_batch > 1:
            self.cur_batch = max(1, self.cur_batch // 2)
            self.oom_shrinks += 1
            self.log(f"  VRAM tight: batch -> {self.cur_batch}")
            return True
        return False

    # ── the driver ───────────────────────────────────────────────────────────────────
    def run(self, fn, items):
        """Process ``items`` through ``fn`` in adaptive, OOM-safe, pause-aware chunks.

        ``fn`` takes a list of items and returns a list of results of the same length. On CPU/disabled
        the whole list goes through in one call. On GPU the batch size floats with free VRAM.
        """
        items = list(items)
        if not items:
            return []
        if not self.enabled:
            # No VRAM to police (CPU/disabled): still chunk at the ceiling so batches stay bounded,
            # but skip the pause/shrink/grow machinery entirely.
            out = []
            for i in range(0, len(items), self.max_batch):
                bs = len(items[i:i + self.max_batch])
                self.batch_sizes[bs] = self.batch_sizes.get(bs, 0) + 1
                out.extend(fn(items[i:i + self.max_batch]))
            return out
        self._calibrate_once()
        out = []
        i = 0
        n = len(items)
        batch1_ooms = 0
        while i < n:
            self.wait_for_headroom()
            bs = min(self.cur_batch, n - i)
            chunk = items[i:i + bs]
            try:
                res = fn(chunk)
            except Exception as exc:
                if not _is_oom(exc, self._oom_types):
                    raise
                self._empty_cache()
                if not self._shrink():
                    batch1_ooms += 1
                    if batch1_ooms >= self.max_batch1_ooms:
                        raise RuntimeError(
                            f"out of VRAM on a single item {batch1_ooms} times in a row on "
                            f"{self.device}, so the card cannot hold one prompt of this run and "
                            f"waiting will not change that. Lower --max-new-tokens, lower "
                            f"--eval-refusal / --eval-kl, use a smaller model, or free the card. "
                            f"There is nothing left to shrink: the batch is already 1."
                        ) from exc
                    self.log(f"  VRAM OOM at batch=1 ({batch1_ooms} of "
                             f"{self.max_batch1_ooms}): pausing until the card frees up")
                    # Force a pause even if the fraction check would pass: the card just proved it is
                    # too full for one item, so wait for a clear margin before trying again.
                    self._forced_pause()
                continue
            out.extend(res)
            # Progress, so the streak of hopeless retries is over. Reset rather than decay: what
            # the cap is counting is consecutive failures on an item nothing can make smaller.
            batch1_ooms = 0
            self.batch_sizes[bs] = self.batch_sizes.get(bs, 0) + 1
            i += bs
            self._grow_maybe(bs)
        return out

    def report(self):
        """What this governor actually did, for the run's artefact.

        Generation is greedy, so there is no sampling noise; but left-padding means the batch a
        prompt travelled in changes its numerics, and this governor picks that batch from live
        free VRAM. Two runs of the same command on the same machine can therefore differ because
        something else was on the card. That is a real property of the measurement and belongs
        beside it: a reader comparing two numbers needs to know whether the machinery was pinned.

        `throttled=False` means the batch was fixed and the run is reproducible on this axis.
        """
        return {
            "throttled": bool(self.enabled),
            "max_batch": self.max_batch,
            "batch_sizes_used": dict(sorted(self.batch_sizes.items())),
            "oom_shrinks": self.oom_shrinks,
            "pauses": self.pauses,
        }

    def _forced_pause(self):
        # Wait after a batch=1 OOM, then hand control back to the counter that can raise.
        #
        # THE FIRST SLEEP IS UNCONDITIONAL, and without it this function did nothing at all.
        # `_avail_frac` counts senbon's OWN reclaimable cache as available, and the caller has
        # just called `_empty_cache()`, so the fraction it reads is almost always above the
        # grow threshold and the loop below exits before sleeping once. The card had just
        # refused a single item, and the "pause" returned in microseconds: that is what turned
        # a retry into a spin. A card that cannot fit one prompt is not helped by asking it
        # again immediately.
        #
        # AND THEN IT WENT THE OTHER WAY, found by the 2026-09-25 panel and reproduced through the
        # injectable hooks below. Two defects, both of which only bite on the hardware this module
        # exists for:
        #
        #   1. The loop waited for `grow_free_frac` (0.20), which is the threshold for GROWING the
        #      batch back. After an OOM at the irreducible batch, that asks the card to become
        #      emptier than it was when the model loaded. On a card the model itself fills, the
        #      availability fraction is permanently below it, so the condition is never satisfied.
        #   2. The escape was `if self.max_pause_s is not None`, and `max_pause_s` defaults to None
        #      at every construction site. So there was no escape.
        #
        # Together: two log lines, then silence for as long as the operator left it. `batch1_ooms`
        # never reached 2, so the counter whose whole purpose is to name what to lower could not
        # fire, and the log's own "1 of 20" promised nineteen retries that would never come.
        #
        # The pause is now bounded ALWAYS. The bound is not a timeout on the problem, it is a
        # deadline on this function: control has to return to the counter, because the counter is
        # the thing that can turn a wedge into a sentence.
        deadline = self.max_pause_s if self.max_pause_s is not None else FORCED_PAUSE_DEADLINE_S
        waited = self.poll_s
        self._sleep(self.poll_s)
        self.paused_s += self.poll_s
        # `min_free_frac`, not `grow_free_frac`: enough to try one item again, which is all this
        # pause is for. Growing the batch back is `_maybe_grow`'s decision and it has its own
        # threshold.
        #
        # ANNOUNCED ON ENTRY RATHER THAN PERIODICALLY, and that is a correction to this fix rather
        # than the original design. The first version put a HEARTBEAT_S (30s) periodic log inside a
        # loop bounded at FORCED_PAUSE_DEADLINE_S (30s), so the interval could never elapse and the
        # heartbeat was unreachable code: a guard that cannot fire, which is the exact defect class
        # this whole repair came out of. This pause is short and bounded, so one line when it turns
        # out to be a real wait is the honest amount of noise. The periodic heartbeat belongs in
        # `wait_for_headroom`, which is unbounded by design.
        announced = False
        while self._avail_frac() < self.min_free_frac:
            if not announced:
                announced = True
                self.log(f"  the card is still too full after {fmt_duration(waited)}: "
                         f"{self._avail_frac():.0%} usable, need {self.min_free_frac:.0%}. "
                         f"Waiting up to {fmt_duration(deadline)} before trying again")
            self._empty_cache()
            self._sleep(self.poll_s)
            waited += self.poll_s
            self.paused_s += self.poll_s
            if waited >= deadline:
                return waited
        return waited


class SearchProgress:
    """Trial-by-trial progress with an ETA that excludes time spent paused for VRAM.

    Wall-clock would over-estimate when the run keeps pausing for a game; measuring against ACTIVE
    time (elapsed minus paused) gives an honest estimate that a "paused" note explains.
    """

    def __init__(self, total, log, governor=None, clock=None):
        self.total = max(1, int(total))
        self.log = log or (lambda _m: None)
        self.gov = governor
        self._clock = clock or time.monotonic
        self.start = self._clock()
        self.done = 0

    def _paused(self):
        return self.gov.paused_s if self.gov is not None else 0.0

    def tick(self):
        self.done += 1
        now = self._clock()
        paused = self._paused()
        active = max(1e-9, (now - self.start) - paused)
        rate = self.done / active                       # trials per active second
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        note = f" (+{fmt_duration(paused)} paused)" if paused > 0 else ""
        self.log(f"  progress {self.done}/{self.total} | {fmt_duration(active)} active{note} | "
                 f"ETA {fmt_duration(remaining)}")
