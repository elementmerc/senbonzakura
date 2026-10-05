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


def cuda_used_bytes_external(index=0):
    """Device-wide VRAM in use, read WITHOUT creating a CUDA context in this process.

    WHY NOT `cuda_free_total`, WHICH ALREADY READS THE WHOLE CARD. Because the caller is measuring
    somebody ELSE: a competing tool's arm, running as a subprocess, often in a container. Touching
    `torch.cuda` here would initialise a CUDA context in the measuring process, and a context costs
    a few hundred MB. On the 6 GB card this project targets that is not noise, it is a measurable
    slice of the budget the arm is being judged on. **An instrument that changes the quantity it
    measures is worse than no instrument**, because the number it hands back looks fine.

    So this shells out, which `ResourceGovernor` deliberately does not. The reason the two differ is
    the sampling rate, not taste: the governor was rejected for shelling out because it refreshes at
    4 Hz, and this samples at well under 1 Hz over an arm that runs for tens of minutes.

    Device-wide is also the only HONEST cross-tool number. A rival's process is not ours to
    introspect, and asking each tool to report its own peak would compare four different
    definitions. The cost is that anything else on the card is counted too, which is why the caller
    records the baseline as well as the peak and the pair is published together.

    Returns bytes, or None when it cannot be read at all.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={int(index)}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    try:
        return int(float(line.strip())) * 1024 * 1024       # nvidia-smi reports MiB
    except ValueError:
        return None


class PeakVram:
    """Samples device-wide VRAM on a thread while something else runs, and reports the peak.

    A context manager, so the sampling cannot outlive the thing it is measuring:

        with PeakVram() as peak:
            run_the_arm()
        peak.peak_bytes, peak.baseline_bytes, peak.samples

    `baseline_bytes` is read once before the body starts, and it is reported alongside the peak
    rather than subtracted from it. Subtracting would invent a number: the baseline is what was
    resident a moment earlier, not what that other process held throughout, and a difference
    presented as "what this tool used" would be a guess wearing a measurement's clothes.

    **A card this cannot read yields None rather than zero.** Zero reads as "used no memory", which
    is the shape of claim this project keeps withdrawing: `samples` is published too, so a reader
    can tell a measurement from a card that was never polled.
    """

    def __init__(self, index=0, interval_s=5.0, reader=None):
        self._index = int(index)
        self._interval = float(interval_s)
        self._read = reader or (lambda: cuda_used_bytes_external(self._index))
        self.baseline_bytes = None
        self.peak_bytes = None
        self.samples = 0
        self._stop = None
        self._thread = None

    def _sample(self):
        value = self._read()
        if value is None:
            return
        self.samples += 1
        if self.peak_bytes is None or value > self.peak_bytes:
            self.peak_bytes = value

    def __enter__(self):
        import threading
        self.baseline_bytes = self._read()
        self._sample()
        self._stop = threading.Event()

        def loop():
            # `wait` rather than `sleep`, so a finished arm is not followed by up to one interval
            # of pointless polling, and so the thread cannot outlive the body on an exception.
            while not self._stop.wait(self._interval):
                self._sample()

        self._thread = threading.Thread(target=loop, name="peak-vram", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 10)
        self._sample()
        return False

    def as_record(self):
        """The pair, in MiB, shaped for a run record. None stays None."""
        mib = lambda b: None if b is None else round(b / (1024 * 1024))  # noqa: E731
        return {"peak_vram_mib": mib(self.peak_bytes),
                "baseline_vram_mib": mib(self.baseline_bytes),
                "vram_samples": self.samples}


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

#: How long the STARVATION pause may run before control goes back to the caller. Not a tolerance and
#: not a tuning knob: it is the answer to "what if the thing we are waiting for is us".
#:
#: `_external_used` subtracts `torch.cuda.memory_reserved`, which counts the caching allocator and
#: nothing else, so this process's own CUDA context, its cuBLAS and cuDNN workspaces and any non
#: torch allocation all read as somebody else holding the card. On a 6 GB card with a 5 GB model and
#: nothing else running, that is enough to clear the floor above, so `_should_pause` stayed true and
#: an unbounded wait waited for a process that does not exist to release memory that is ours. That is
#: the 2026-09-25 wedge, in the code written to fix it, and the accounting fix below narrows it
#: without being able to prove it gone on every driver.
#:
#: So the wait is bounded as well. Pushing on risks an out of memory, which this class already
#: handles by halving the batch, and an out of memory that shrinks a batch is a far better outcome
#: than a run that never returns. The foreground yield is deliberately NOT bounded by this: that
#: patience is a good citizen behaviour the operator asked for, and its predicate is about somebody
#: else's app rather than about our own residency.
STARVATION_DEADLINE_S = 300.0

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
        # THIS FIGURE IS KNOWN TO OVER-REPORT, and the deadline in `wait_for_headroom` is the answer
        # rather than a correction here. `_external_used` subtracts `torch.cuda.memory_reserved`,
        # which counts the caching allocator and nothing else, so our own CUDA context and workspaces
        # are attributed to another process; on a 6 GB card with a 5 GB model that alone clears the
        # floor below.
        #
        # SUBTRACTING A MEASURED BASELINE OF OUR OWN OVERHEAD WAS TRIED ON 2026-09-28 AND BACKED OUT.
        # It has to be measured at some moment, and whatever is on the card at that moment is
        # absorbed into it, so an app that was already running became invisible and the starvation
        # pause stopped firing for it. `test_external_pressure_triggers_pause` is that scenario and it
        # went green by losing the behaviour it names. Telling our context from their allocation needs
        # per-process accounting, which is what this class deliberately avoids so it works in WSL2
        # where the Windows-side app is invisible to nvidia-smi.
        #
        # So the over-report stands and the WAIT is bounded. An unbounded wait on a predicate that can
        # be permanently true is the wedge, whatever the predicate is.
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
            # THE STARVATION WAIT IS BOUNDED AND THE FOREGROUND YIELD IS NOT. Patience is the right
            # answer to another app holding the card, and it cannot change our own residency, so a
            # pause caused by mis-attributing our own memory would otherwise never end. Pushing on
            # risks an out of memory, which halves the batch; a run that never returns has no
            # recovery at all.
            if not self._foreground_pressure() and waited >= STARVATION_DEADLINE_S:
                self.log(
                    f"  WARNING: {fmt_duration(waited)} paused waiting for VRAM that has not come "
                    f"back, so pushing on. If the card is genuinely full this will show up as an "
                    f"out of memory and the batch will shrink; if nothing else is on the card, the "
                    f"memory being waited for is this run's own and the pause was the fault.")
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
    @staticmethod
    def _checked(res, expected, offset):
        """Refuse a batch whose result count does not match the chunk it came from.

        THE CONTRACT WAS DECLARED AND NOT ENFORCED. `run`'s own docstring says `fn` returns a list
        of the same length, and both call sites used to extend the output and advance the cursor by
        the chunk size regardless. A short or long return therefore shifted every LATER result
        against its item by a drifting offset, silently, because the cursor moved by the chunk size
        while the output grew by something else.

        That is the worst failure this file can have. Callers pair prompts with results by index, so
        a misalignment does not look like an error, it looks like a refusal rate: one prompt's
        generation scored against another prompt's label, for the rest of the run. This project has
        the scar already, in a manifest whose partition boundaries were read loosely and sliced
        every published number off the wrong rows.

        A check rather than trust, although §6 says trust internal code, because `fn` is supplied by
        the caller and crosses out of this module, the failure is silent rather than loud, and the
        length is the one property the contract names.
        """
        res = list(res)
        if len(res) != expected:
            raise RuntimeError(
                f"a batched worker returned {len(res)} result(s) for a chunk of {expected} "
                f"item(s), at item {offset}. Results are paired with items by position, so "
                f"continuing would score every later item against the wrong one and report a "
                f"number rather than an error. The worker passed to ResourceGovernor.run must "
                f"return exactly one result per item, in order.")
        return res

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
                chunk = items[i:i + self.max_batch]
                bs = len(chunk)
                self.batch_sizes[bs] = self.batch_sizes.get(bs, 0) + 1
                out.extend(self._checked(fn(chunk), bs, i))
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
            out.extend(self._checked(res, bs, i))
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


# ── the streaming run's memory budget ────────────────────────────────────────────────────────────
#
# WHAT THIS SECTION IS FOR. A streaming abliteration reads a checkpoint off disk one layer at a
# time and may run for days. Everything that can stop it is knowable in about a second from the
# checkpoint's headers and four readings off the machine, and every one of them is a thing that
# otherwise announces itself hours in: the card is too small for the widest layer, the host store
# crosses the page-locked ceiling and the copies stop overlapping, the disk fills while a shard is
# half rewritten, the laptop is on battery and clocked to a third.
#
# WHY THE ARITHMETIC LIVES HERE AND THE READING DOES NOT. Baseline Section 12 asks for a boundary
# between logic and storage. `Checkpoint` below is the whole contract: anything that can fill in
# those numbers can drive the budget, and `streaming.describe` is today's one filler. So the
# budget is testable with no checkpoint, no card and no torch, which is the machine most of this
# project's tests run on.
#
# WHAT IT DELIBERATELY DOES NOT DO. It allocates nothing proportional to the model. Every figure
# is a sum over header metadata, and the one probe that allocates (the page-locked ceiling) is
# bounded by its own argument and frees what it took. A preflight whose own footprint grew with
# the input would be answering its question by becoming it.


#: Sustained cold read rate assumed for the time estimate, in bytes per second.
#:
#: MEASURED, 2026-09-23, `tools/research/layer_read_spike.py`: a mixture-of-experts layer read by
#: `pread` off the ROG's ext4 NVMe, cache evicted and the eviction confirmed, reached 1.95 GB/s.
#: That is the slowest of the three layouts measured and the one the streaming path exists for, so
#: it is the honest constant for an estimate rather than the best of them.
#:
#: ONE MACHINE, ONE DISK. The report says so wherever it uses this number, and `Run.read_bytes_s`
#: overrides it, because an estimate carrying somebody else's disk as if it were yours is worse
#: than no estimate.
STREAMING_READ_BYTES_S = 1.95e9

#: Rated write endurance assumed for the wear note, in bytes. 600 TB is the typical figure for a
#: 1 TB consumer TLC drive. Reported as a fraction of a run rather than as a verdict: it is the
#: user's drive and the user's decision, and the plan's loophole 13 asks only that the number be
#: visible before they spend it.
WRITE_ENDURANCE_BYTES = 600e12

#: Disk a resumable run spends per layer on its completion marker and the digest of what it wrote.
#: Small, and counted anyway, because a budget that silently omits a term is a budget nobody can
#: check against what the run actually used.
CHECKPOINT_BYTES_PER_LAYER = 4096

#: How many float32 tensors of the block's shape `orthogonalize_np_3d_` holds live at its peak,
#: without and with `--sparsity`. From reading the routine: the float32 cast, the row norms, the
#: two einsum results and the accumulating output, plus the boolean row mask and the magnitudes it
#: ranks when sparsity is on. An estimate of a peak, named as one, and the number the exit gate's
#: "peak memory is asserted by a test" item is the thing that will eventually pin.
FP32_LIVE_TENSORS = 5
FP32_LIVE_TENSORS_SPARSE = 6

#: The fraction of a measured resource the budget will plan into, leaving the rest for everything
#: that is not the run. The same 0.9 the resident snapshot guard has always used, kept the same on
#: purpose: two preflights over the same machine disagreeing about what "full" means is how an
#: operator learns to ignore both.
BUDGET_HEADROOM = 0.9


def fmt_span(seconds):
    """A duration that may be measured in days, which `fmt_duration` is not.

    `fmt_duration` is the governor's ETA format and tops out at hours, which is right for a pause
    and wrong for a five-day search: "110h 00m" is a number a reader has to do arithmetic on. This
    adds the days tier and delegates everything below a day, so there is one implementation of the
    hours and minutes and no chance of the two drifting.
    """
    try:
        s = int(max(0, round(seconds)))
    except (ValueError, OverflowError):
        return "0s"
    if s < 86400:
        return fmt_duration(s)
    return f"{s // 86400}d {(s % 86400) // 3600:02d}h"


def fmt_bytes(n):
    """Bytes as a figure with a unit, because a bare count of bytes is not a quantity anyone reads."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "unknown"
    # PB and EB are here because a search's read total genuinely reaches them: a deliberately
    # absurd prompt count printed "6017.5 TB", which is a number a reader has to count the digits
    # of to understand. A unit table that stops one tier below what the arithmetic produces is a
    # table that stops being readable exactly where the figure starts being alarming.
    for unit, scale in (("EB", 1e18), ("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6),
                        ("kB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} B"


# ── reading the machine ──────────────────────────────────────────────────────────────────────────


def host_ram_available():
    """Available host RAM in bytes, or None on a platform that will not say.

    None means UNMEASURED, never zero and never "plenty". Both probes are POSIX: `/proc/meminfo`
    is Linux only and `SC_AVPHYS_PAGES` is absent on Windows and unreliable on macOS, so a caller
    that treats None as a pass has to say out loud that it skipped the check.

    UNDER WSL2 THIS IS THE VIRTUAL MACHINE, NOT THE HOST. `/proc/meminfo` inside WSL2 reports the
    ballooned VM's memory, which is a fraction of the Windows machine's and moves while the run
    is going. It is still the right number, because it is the memory this process can actually
    get, but a report that prints it beside "your laptop has 32 GB" needs to say which it means.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def own_footprint():
    """The calling process's resident set in bytes, or None where the platform will not say.

    WHY A RUN SHOULD REPORT ITS OWN SIZE. On 2026-10-05 a tamper run on a 135M model grew to
    1.8 GB and kept climbing, took nearly four hours for about two minutes of arithmetic, and was
    found by somebody else measuring the machine rather than by anything in its own log. The cause
    was a `free(model)` that deleted its own parameter and nothing else, so every checkpoint the
    run had loaded stayed resident; by the time it mattered the working set was in swap and the
    box was thrashing rather than failing, which is the mode that looks survivable and is not.

    A number in the log every few steps is what turns that into a line somebody reads at minute
    two. `MemAvailable` beside it answers the other half: a footprint that is fine on a 29 GB box
    and fatal on a 7 GB one is the same number.

    RESIDENT RATHER THAN PEAK, deliberately. `ru_maxrss` is a high-water mark and never comes
    back down, so it cannot show a leak being fixed or a model being released. The question here
    is "how much is held right now", which is the one that grows.

    None means UNMEASURED, never zero. `/proc/self/statm` is Linux only, which includes WSL2, and
    a caller that treats None as "small" has to say out loud that it did not measure.
    """
    try:
        import os
        with open("/proc/self/statm", encoding="utf-8") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def free_disk(path):
    """Free bytes on the filesystem that would hold `path`, or None.

    WALKS UP TO AN EXISTING ANCESTOR, because the output directory of a run that has not started
    does not exist yet, and `disk_usage` on a missing path raises. Asking about the parent answers
    the same question: a preflight that refused to estimate because the destination was not there
    would decline exactly when it is wanted.
    """
    import pathlib
    import shutil
    try:
        here = pathlib.Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for candidate in (here, *here.parents):
        if candidate.is_dir():
            break
    else:
        return None
    try:
        return int(shutil.disk_usage(candidate).free)
    except (OSError, ValueError):
        return None


def on_mains_power(root="/sys/class/power_supply"):
    """True on mains, False on battery, None when the machine will not say.

    WHY A MEMORY BUDGET ASKS ABOUT POWER. A laptop discrete GPU on battery clocks to roughly a
    third, so a five-day estimate becomes a fortnight and nothing in any log says why. The plan's
    loophole 9 asks for it to be preflighted rather than discovered.

    Reads `/sys/class/power_supply`, which is the kernel's own account: a supply of type `Mains`
    reporting `online` is the answer, and a machine with no battery at all has no supply of type
    `Battery`, which is itself a reliable mains. Desktops, containers and non-Linux return None,
    and None is reported as unknown rather than assumed to be fine.

    `root` is a parameter for the same reason every other dependency in this module is injectable:
    a test for "this laptop is on battery" that has to patch `pathlib.Path` globally is a test
    that breaks whatever else the interpreter was doing with paths.
    """
    import pathlib
    root = pathlib.Path(root)
    try:
        supplies = sorted(root.iterdir())
    except OSError:
        return None
    mains, battery = None, False
    for supply in supplies:
        try:
            kind = (supply / "type").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if kind == "Battery":
            battery = True
        elif kind == "Mains":
            try:
                online = (supply / "online").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            # Any mains supply that is online is enough. A dock and a charger both present means
            # two supplies and one of them offline, and "one of them is plugged in" is the answer.
            mains = bool(mains) or online == "1"
    if mains is not None:
        return mains
    if supplies and not battery:
        return True                 # supplies exist and none of them is a battery: not a laptop
    return None


def probe_pinned_ceiling(max_bytes=None):
    """`(bytes, hit_limit)` for the page-locked ceiling, or None when it cannot be measured.

    ONE IMPLEMENTATION, CALLED FROM TWO PLACES. `doctor.check_pinned_memory` reports this number
    as a diagnostic and this reports it as a design constraint, and the measurement is the same
    measurement. A second copy of a probe that allocates gigabytes is not a thing to keep.

    `hit_limit` False means the probe stopped because it ran out of budget, so the figure is a
    floor and the report must say "at least" rather than naming a ceiling that was never found.
    """
    try:
        from .doctor import PINNED_SHALLOW_STEPS, PINNED_STEP_BYTES, _measure_pinned_ceiling
    except ImportError:
        return None
    budget = PINNED_STEP_BYTES * PINNED_SHALLOW_STEPS if max_bytes is None else int(max_bytes)
    try:
        reached, hit_limit, _note = _measure_pinned_ceiling(budget)
    except Exception:
        # A diagnostic probe must never be the reason a run does not start. Unmeasured is a
        # reportable state and the report carries it as one.
        return None
    return int(reached), bool(hit_limit)


# ── the contract the budget is computed over ─────────────────────────────────────────────────────


class Checkpoint:
    """What the budget needs to know about a model on disk, and nothing about how it was read.

    Every field is a sum or a maximum over safetensors header metadata, so filling this in costs
    one pass over the headers and reads no weights. `streaming.describe` is the filler; a test
    builds one by hand, which is the point of the boundary.

    `widest_layer_bytes` rather than a mean, because the memory budget is set by the WORST layer:
    an architecture that puts a dense mixture-of-experts block on some layers and not others
    exists, and budgeting the average fits every layer but one and fails hours in.
    """

    __slots__ = ("experts", "head_dim", "hidden", "kv_heads", "largest_shard_bytes", "layers",
                 "shards", "total_bytes", "widest_layer_bytes", "widest_writer_bytes",
                 "writer_bytes", "writer_width")

    def __init__(self, *, layers, total_bytes, widest_layer_bytes, widest_writer_bytes,
                 writer_bytes, largest_shard_bytes, shards, writer_width,
                 experts=0, kv_heads=None, head_dim=None, hidden=None):
        self.layers = int(layers)
        self.total_bytes = int(total_bytes)
        self.widest_layer_bytes = int(widest_layer_bytes)
        self.widest_writer_bytes = int(widest_writer_bytes)
        self.writer_bytes = int(writer_bytes)
        self.largest_shard_bytes = int(largest_shard_bytes)
        self.shards = int(shards)
        self.writer_width = int(writer_width)
        self.experts = int(experts or 0)
        self.kv_heads = None if kv_heads is None else int(kv_heads)
        self.head_dim = None if head_dim is None else int(head_dim)
        self.hidden = None if hidden is None else int(hidden)


class Run:
    """The knobs that change what a streaming run costs.

    Defaults are deliberately absent for the counts: a budget computed against numbers the caller
    did not choose is a budget for a different run, and this project has shipped a figure measured
    on a slice nobody picked before.
    """

    __slots__ = ("expert_block", "passes_per_trial", "prompts", "read_bytes_s", "sparsity",
                 "tokens", "trials")

    def __init__(self, *, prompts, tokens, trials, passes_per_trial, expert_block=8,
                 sparsity=False, read_bytes_s=None):
        self.prompts = int(prompts)
        self.tokens = int(tokens)
        self.trials = int(trials)
        self.passes_per_trial = int(passes_per_trial)
        self.expert_block = max(1, int(expert_block))
        self.sparsity = bool(sparsity)
        self.read_bytes_s = float(read_bytes_s or STREAMING_READ_BYTES_S)


class Machine:
    """Four readings and a power state. Every one of them may be None, meaning unmeasured.

    None is NOT zero and NOT fine. A pool whose availability is None is reported as skipped rather
    than passed, which is the discipline the resident snapshot preflight already holds: an
    operator who believes a guard ran when none did is worse off than one told it could not.
    """

    __slots__ = ("disk_free", "on_mains", "pinned_ceiling", "pinned_is_floor", "ram_available",
                 "vram_free", "vram_total")

    def __init__(self, *, vram_free=None, vram_total=None, ram_available=None,
                 disk_free=None, pinned_ceiling=None, pinned_is_floor=False, on_mains=None):
        self.vram_free = vram_free
        self.vram_total = vram_total
        self.ram_available = ram_available
        self.disk_free = disk_free
        self.pinned_ceiling = pinned_ceiling
        self.pinned_is_floor = bool(pinned_is_floor)
        self.on_mains = on_mains

    @classmethod
    def measure(cls, *, device="cuda:0", path=".", pinned=True, pinned_max_bytes=None):
        """Read this machine. Nothing here raises: every probe returns None when it cannot answer.

        `pinned=False` skips the one probe that allocates, which is what a caller wants when it is
        asking the budget a hypothetical rather than preparing to launch.
        """
        vram = cuda_free_total(device)
        ceiling, is_floor = (None, False)
        if pinned:
            probed = probe_pinned_ceiling(pinned_max_bytes)
            if probed is not None:
                ceiling, hit_limit = probed
                is_floor = not hit_limit
        return cls(vram_free=None if vram is None else vram[0],
                   vram_total=None if vram is None else vram[1],
                   ram_available=host_ram_available(),
                   disk_free=free_disk(path),
                   pinned_ceiling=ceiling, pinned_is_floor=is_floor,
                   on_mains=on_mains_power())


# ── the arithmetic ───────────────────────────────────────────────────────────────────────────────


def kv_cache_bytes(*, layers, kv_heads, head_dim, width, prompts, tokens):
    """Bytes the key/value cache holds at the end of a generation of `tokens` over `prompts`.

    `2 *` because there is a key and a value. Per layer, per attention head that owns its own
    key/value (grouped-query attention shares them, which is why this takes `kv_heads` and not
    the query head count), per token, in the cache's dtype.

    WHY IT IS ITS OWN TERM. The cache is the one structure in a streaming run that cannot be
    evicted with its layer: every layer's keys and values have to survive until the generation
    finishes, so it sits on the card for the whole of a decode while layers come and go around it.
    The plan's memory model omitted it entirely. For Qwen3-30B-A3B (48 layers, 4 key/value heads,
    head dimension 128, bfloat16) this is 98 kB a token, which is 402 MB at 16 prompts of 256
    tokens: not the dominant term and far too big to leave out of a 6 GB budget.
    """
    return 2 * int(layers) * int(kv_heads) * int(head_dim) * int(width) * int(prompts) * int(tokens)


def activation_cache_bytes(*, layers, hidden, prompts, width=4):
    """Bytes the captured residual cloud holds: one vector per layer boundary, per prompt.

    NOT STREAMED, DELIBERATELY. At the largest preset's 192 prompts on a 48-layer model with a
    hidden size of 2048 this is 77 MB a side, so there is nothing worth streaming, and an
    incremental covariance rewrite would break the axis-separation score, which needs both
    complete clouds to judge a candidate. It is budgeted rather than bounded for the same reason:
    it is small, and a term left out of a budget is a term nobody can check.

    float32 by default because the capture promotes: the clouds are what an SVD runs over.
    """
    return (int(layers) + 1) * int(prompts) * int(hidden) * int(width)


def passes_per_trial(*, eval_refusal, eval_kl, gen_batch, gen_tokens, coherence_prompts=16):
    """Full-model forward passes one search trial costs, which is what sets the time estimate.

    FROM THE OBJECTIVE AS WRITTEN, not from the plan's prose. `Abliterator.objective` does three
    things that touch the model: generate over the harmful evaluation slice, generate over a small
    harmless slice for coherence, and take first-token log probabilities over the KL slice. Both
    generations are chunked by the governor at `--gen-batch`, and each chunk costs one prefill
    plus one pass per new token, because every decode step needs every layer again. The log
    probabilities are prefill only, one pass a chunk.

    This is the quantity streaming is bad at and the reason the spike exists: amortising a layer
    load across a batch suits a single forward pass and is pathological for decode.

    WORTH A NOTE WHERE THE TWO ACCOUNTS DIFFER. The plan recounted this as 198 passes for a
    48-prompt evaluation at batch 16 and 48 tokens; the formula here gives 199, and the extra one
    is the KL slice's third chunk, which the plan's count appears to have taken as a single pass.
    The code is the authority and the gap is one pass in two hundred, so the plan's five-day
    headline stands.
    """
    def chunks(n):
        n = max(0, int(n))
        return -(-n // max(1, int(gen_batch)))          # ceiling division, no float anywhere

    decode = 1 + max(0, int(gen_tokens))                # the prefill, then one pass per new token
    return (chunks(eval_refusal) * decode
            + chunks(min(int(coherence_prompts), int(eval_kl))) * decode
            + chunks(eval_kl))


def rewrite_working_bytes(checkpoint, run):
    """Peak float32 working set of the on-disk rewrite, for one block of experts.

    `orthogonalize_np_3d_` works in float32 and holds several tensors of the block's shape live at
    once, so the peak is set by the widest residual-writing tensor, scaled down by the expert
    block and up by the width ratio between float32 and whatever the checkpoint stores. On a dense
    checkpoint there is no expert axis and the block does not divide anything.

    THIS PASS SETS THE FLOOR, NOT THE CAPTURE. A fused expert stack is the largest thing the run
    ever holds, and it is held at four bytes an element rather than two.
    """
    live = FP32_LIVE_TENSORS_SPARSE if run.sparsity else FP32_LIVE_TENSORS
    block = checkpoint.widest_writer_bytes
    if checkpoint.experts > 0:
        block = block * min(run.expert_block, checkpoint.experts) / checkpoint.experts
    return int(live * block * (4.0 / max(1, checkpoint.writer_width)))


def streaming_read_bytes(checkpoint, run):
    """Bytes read off disk across a whole run, which is the time estimate's numerator.

    Four phases, and the search dominates by three orders of magnitude:

      capture   one pass over the model
      search    per trial, one restore from the source shards plus `passes_per_trial` forwards
      bake      one pass to read what is being rewritten
      rescore   one final pass over the model

    THE RESTORE IS A READ NOW, AND IT USED NOT TO BE. Dropping the 20 GB resident snapshot means
    the pristine weights come back off the source shards, which recovers 20 GB of host RAM and
    costs one model read a trial. For a 61 GB checkpoint over 64 trials that is 3.9 TB of extra
    reads. It is the right trade against what 20 GB of resident snapshot does to the page cache,
    and it is a trade rather than a free recovery, so it is a named term here.
    """
    per_trial = run.passes_per_trial + 1                # the forwards, plus the restore
    return int(checkpoint.total_bytes * (2 + run.trials * per_trial))


def streaming_write_bytes(checkpoint, run):
    """Bytes written to disk across a whole run, which is the wear note's numerator.

    Per trial the bake rewrites the residual-writing tensors and nothing else, and at the end the
    output checkpoint is written once in full. The plan's loophole 13 put this at roughly 10.5 TB
    for a 30B search, which assumed a full model round trip every trial; this counts the tensors
    the rewrite actually touches, so it is the smaller and more defensible figure.
    """
    return int(checkpoint.writer_bytes * run.trials + checkpoint.total_bytes)


# ── the verdict ──────────────────────────────────────────────────────────────────────────────────


class Pool:
    """One resource, what each part of the run wants from it, and whether that fits.

    `needs` is a list of `(label, bytes)` rather than one total, because a refusal that says
    "6.4 GB needed, 5.8 GB free" tells an operator nothing they can act on, and one that names the
    widest layer, the rewrite's working set and the key/value cache tells them which flag to move.
    """

    __slots__ = ("available", "headroom", "name", "needs", "unit")

    def __init__(self, name, needs, available, headroom=BUDGET_HEADROOM):
        self.name = name
        self.needs = list(needs)
        self.available = available
        self.headroom = float(headroom)

    @property
    def total(self):
        return sum(int(n) for _label, n in self.needs)

    @property
    def budget(self):
        """What the pool will plan into, or None when availability was never measured."""
        return None if self.available is None else int(self.available * self.headroom)

    @property
    def measured(self):
        return self.available is not None

    @property
    def fits(self):
        """True, False, or None for unmeasured. None is never True: a skipped check is not a pass."""
        return None if not self.measured else self.total <= self.budget

    @property
    def shortfall(self):
        return 0 if not self.measured else max(0, self.total - self.budget)


class Budget:
    """Everything the preflight worked out, as data, so the report and the refusal share one source.

    Deliberately not a printer. `report` renders it and `preflight` decides on it; keeping the
    verdict as data is what lets a test assert the numbers without reading prose, and lets the
    guided mode draw the same figures its own way.
    """

    __slots__ = ("checkpoint", "machine", "pools", "read_bytes", "run", "seconds", "write_bytes")

    def __init__(self, *, checkpoint, run, machine, pools, read_bytes, write_bytes, seconds):
        self.checkpoint = checkpoint
        self.run = run
        self.machine = machine
        self.pools = list(pools)
        self.read_bytes = int(read_bytes)
        self.write_bytes = int(write_bytes)
        self.seconds = float(seconds)

    @property
    def fits(self):
        """False if any MEASURED pool does not fit. Unmeasured pools cannot make it fit or not."""
        return not any(pool.fits is False for pool in self.pools)

    @property
    def unmeasured(self):
        return [pool.name for pool in self.pools if not pool.measured]

    @property
    def verdict(self):
        """The headline sentence, and it may not say "fits" when a pool was never measured.

        FOUND BY RUNNING IT. On a machine with no card, a run needing 84 GB of key/value cache
        printed "this run fits on this machine", because the video memory pool was unmeasured and
        an unmeasured pool cannot refuse. Every individual line was right and the headline was
        false, which is this project's most recurring defect shape: a check that answered a
        narrower question than the one asked, reporting clean.

        So the headline carries its own scope. The pool lines already say SKIPPED; this stops the
        last line of the report from quietly overruling them.
        """
        short = [pool.name for pool in self.pools if pool.fits is False]
        if short:
            return f"this machine is short of {', '.join(short)}, so the run would not finish"
        if self.unmeasured:
            return (f"this run fits everything that could be measured here, and "
                    f"{', '.join(self.unmeasured)} could not be read on this machine, so that "
                    f"much is unchecked rather than passed")
        return "this run fits on this machine"

    @property
    def host_store_bytes(self):
        """The host-side layer store, which is the quantity the page-locked ceiling bounds.

        Two layers, because the point of a host store is that the next layer is already in host
        memory when the current one finishes, and a single buffer cannot be filled and read at the
        same time.
        """
        return 2 * self.checkpoint.widest_layer_bytes

    @property
    def pinned_verdict(self):
        """`(side, ceiling, store)` where side is "under", "over" or None for unmeasured.

        WHY THIS IS A VERDICT AND NOT A NOTE. Measured on the ROG on 2026-09-22, a store above the
        page-locked ceiling does not degrade: pinned transfers hid 99.9% of their cost behind
        compute and pageable ones hid 1.8%, at the balance the loader actually runs at. So
        crossing the line forfeits essentially the whole benefit of streaming, quietly, and the
        run then looks slow for a reason that has nothing to do with streaming.
        """
        ceiling = self.machine.pinned_ceiling
        if ceiling is None:
            return None, None, self.host_store_bytes
        if self.machine.pinned_is_floor and self.host_store_bytes <= ceiling:
            # The probe stopped at its own budget, so the real ceiling is at least this and the
            # store is under it either way. Reported as "under" because that much is known.
            return "under", ceiling, self.host_store_bytes
        return ("under" if self.host_store_bytes <= ceiling else "over",
                ceiling, self.host_store_bytes)

    @property
    def wear_fraction(self):
        return self.write_bytes / WRITE_ENDURANCE_BYTES


def plan(checkpoint, run, machine):
    """Work out what a streaming run needs and whether this machine has it.

    Reads nothing and allocates nothing: three records in, one verdict out. The probing happened
    in `Machine.measure`, which is a separate call precisely so that a test, the guided mode and a
    hypothetical ("would a 30B fit if I had 32 GB?") all drive the same arithmetic.
    """
    kv = 0
    if checkpoint.kv_heads and checkpoint.head_dim:
        kv = kv_cache_bytes(layers=checkpoint.layers, kv_heads=checkpoint.kv_heads,
                            head_dim=checkpoint.head_dim, width=checkpoint.writer_width,
                            prompts=run.prompts, tokens=run.tokens)
    acts = 0
    if checkpoint.hidden:
        acts = activation_cache_bytes(layers=checkpoint.layers, hidden=checkpoint.hidden,
                                      prompts=run.prompts)

    card = [("the widest layer, resident", checkpoint.widest_layer_bytes),
            ("the rewrite's float32 working set", rewrite_working_bytes(checkpoint, run))]
    if kv:
        card.append(("the key and value cache", kv))
    if acts:
        card.append(("the captured activations", acts))

    host = [("the host side layer store, double buffered",
             2 * checkpoint.widest_layer_bytes)]

    disk = [("a writable working copy of the checkpoint", checkpoint.total_bytes),
            ("the output checkpoint", checkpoint.total_bytes),
            ("one shard, while it is being rewritten", checkpoint.largest_shard_bytes),
            ("per layer markers and digests",
             checkpoint.layers * CHECKPOINT_BYTES_PER_LAYER)]

    read = streaming_read_bytes(checkpoint, run)
    write = streaming_write_bytes(checkpoint, run)
    return Budget(checkpoint=checkpoint, run=run, machine=machine,
                  pools=[Pool("video memory", card, machine.vram_free),
                         Pool("host memory", host, machine.ram_available),
                         Pool("free disk", disk, machine.disk_free)],
                  read_bytes=read, write_bytes=write,
                  seconds=read / max(1.0, run.read_bytes_s))


class BudgetRefusedError(MemoryError):
    """A run this machine cannot finish, with the reason and what to change.

    A MemoryError subclass because that is what the resident snapshot guard raises for the same
    class of problem, and a caller catching one should catch both.
    """


def report(budget, log=print):
    """Print the budget as a person reads it: what it needs, what is there, and the verdict.

    Every unmeasured pool says it was SKIPPED rather than passed, because an operator who believes
    a check ran when none did is worse off than one who knows it could not run.

    WRAPPED THROUGH `say`, because several of these lines run past 100 characters and the first
    terminal this will be read on is a laptop beside the card. The need lines keep their own
    six-space indent, which `say` leaves verbatim, so the column stays a column.
    """
    from . import say
    ck, run = budget.checkpoint, budget.run

    def tell(text):
        say.say(text, indent="  ", log=log)

    tell(f"{ck.layers} layers, {ck.shards} shard(s), {fmt_bytes(ck.total_bytes)} on disk; "
         f"widest layer {fmt_bytes(ck.widest_layer_bytes)}")
    for pool in budget.pools:
        if not pool.measured:
            tell(f"{pool.name}: needs {fmt_bytes(pool.total)}, and this machine will not report "
                 f"what it has, so the check was SKIPPED rather than passed")
        else:
            outcome = "fits" if pool.fits else f"SHORT by {fmt_bytes(pool.shortfall)}"
            tell(f"{pool.name}: needs {fmt_bytes(pool.total)}, "
                 f"{fmt_bytes(pool.available)} free ({fmt_bytes(pool.budget)} planned into), "
                 f"{outcome}")
        for label, nbytes in pool.needs:
            log(f"      {fmt_bytes(nbytes):>10}  {label}")

    side, ceiling, store = budget.pinned_verdict
    if side is None:
        tell(f"page-locked ceiling: not measured, so whether the {fmt_bytes(store)} host store "
             f"keeps its copy and compute overlap is unknown")
    elif side == "under":
        at_least = "at least " if budget.machine.pinned_is_floor else ""
        tell(f"page-locked ceiling: {at_least}{fmt_bytes(ceiling)}, and the host store is "
             f"{fmt_bytes(store)}, so the copies overlap with compute")
    else:
        tell(f"page-locked ceiling: {fmt_bytes(ceiling)}, and the host store is "
             f"{fmt_bytes(store)}, which is OVER it. Above the ceiling the copies stop overlapping "
             f"and the run looks slow for a reason that is not streaming")

    tell(f"reads {fmt_bytes(budget.read_bytes)} over {run.trials} trial(s) at "
         f"{run.passes_per_trial} forward passes each, so about {fmt_span(budget.seconds)} at "
         f"{run.read_bytes_s / 1e9:.2f} GB/s (measured on one machine and one disk, 2026-09-23)")
    tell(f"writes {fmt_bytes(budget.write_bytes)}, about "
         f"{budget.wear_fraction * 100:.2f}% of a typical consumer drive's rated lifetime")

    if budget.machine.on_mains is False:
        tell("power: on battery. A laptop card on battery clocks to roughly a third, so this "
             "estimate is optimistic by about three times until it is plugged in")
    elif budget.machine.on_mains is None:
        tell("power: unknown on this machine, so nothing checked that a multi day run is not "
             "about to be done on battery")


def preflight(checkpoint, run, machine=None, *, log=print):
    """Plan the run, report it, and refuse one that cannot finish.

    REFUSES BEFORE THE RUN, which is the whole point: everything above is readable from the
    checkpoint's headers in about a second, and the alternative is learning it hours into a job
    that is holding a card.

    An unmeasured pool never refuses. A machine that will not say how much memory it has is not a
    small machine, and a preflight that hard-failed on a platform it could not read would be worse
    than the problem it solves. The report says which pools were skipped.
    """
    from . import say
    machine = machine if machine is not None else Machine.measure()
    budget = plan(checkpoint, run, machine)
    report(budget, log=log)
    say.say(budget.verdict, indent="  ", log=log)
    short = [pool for pool in budget.pools if pool.fits is False]
    if short:
        lines = "\n".join(
            f"  {pool.name}: needs {fmt_bytes(pool.total)} and has {fmt_bytes(pool.available)}, "
            f"short by {fmt_bytes(pool.shortfall)}" for pool in short)
        raise BudgetRefusedError(
            "this machine cannot finish this run, and here is what it is short of:\n"
            f"{lines}\n"
            "  Nothing has been loaded and nothing has been written, so stopping here costs you "
            "only this message.\n"
            "  Lower the prompt count or the generated token count to shrink the cache, pick a "
            "smaller model, or free the resource named above.")
    return budget
