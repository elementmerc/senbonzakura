# SPDX-License-Identifier: AGPL-3.0-or-later
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
                 poll_s=2.0, max_pause_s=None, max_batch=16, enabled=True,
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
        return self._avail_frac() < self.min_free_frac or self._foreground_pressure()

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
        spent waiting (0 when there was headroom). Honours ``max_pause_s`` as a safety cap so a
        mismeasuring driver can never wedge a run forever.
        """
        if not self.enabled:
            return 0.0
        waited = 0.0
        announced = False
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
            # empty_cache here also hands senbon's own idle cache back to the driver, so a foreground
            # game gets a little VRAM too while generation is paused (compute yield, partial VRAM relief).
            self._empty_cache()
            self._sleep(self.poll_s)
            waited += self.poll_s
            self.paused_s += self.poll_s
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
                out.extend(fn(items[i:i + self.max_batch]))
            return out
        self._calibrate_once()
        out = []
        i = 0
        n = len(items)
        while i < n:
            self.wait_for_headroom()
            bs = min(self.cur_batch, n - i)
            chunk = items[i:i + bs]
            try:
                res = fn(chunk)
            except Exception as exc:  # noqa: BLE001 - re-raised below unless it is an OOM we handle
                if not _is_oom(exc, self._oom_types):
                    raise
                self._empty_cache()
                if not self._shrink():
                    self.log("  VRAM OOM at batch=1: pausing until the card frees up")
                    # Force a pause even if the fraction check would pass: the card just proved it is
                    # too full for one item, so wait for a clear margin before trying again.
                    self._forced_pause()
                continue
            out.extend(res)
            i += bs
            self._grow_maybe(bs)
        return out

    def _forced_pause(self):
        # Wait for a comfortable margin (grow-fraction, not just the min) after a batch=1 OOM.
        waited = 0.0
        while self._avail_frac() < self.grow_free_frac:
            self._empty_cache()
            self._sleep(self.poll_s)
            waited += self.poll_s
            self.paused_s += self.poll_s
            if self.max_pause_s is not None and waited >= self.max_pause_s:
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
