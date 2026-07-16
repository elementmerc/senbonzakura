"""Unit tests for the adaptive VRAM governor (resources.py).

Every external dependency the governor touches (VRAM reading, reclaimable-cache reading, cache flush,
sleep, clock, the out-of-memory exception type) is injectable, so the whole calibrate/pause/shrink/
grow/resume policy is exercised deterministically here with no GPU. `reclaim_fn=lambda: 0` is passed
wherever we want availability to equal raw free VRAM.
"""
import pytest

from senbonzakura.resources import (
    ResourceGovernor,
    SearchProgress,
    cuda_free_total,
    cuda_reclaimable,
    fmt_duration,
    _is_oom,
)


class _FakeOOM(Exception):
    pass


def _mem(seq):
    """A mem_fn that walks a list of (free, total) tuples, holding the last value forever."""
    box = {"i": 0}

    def fn():
        i = min(box["i"], len(seq) - 1)
        box["i"] += 1
        return seq[i]

    return fn


def _gov(**kw):
    # Governor with availability == raw free (reclaim 0) unless a test overrides it.
    kw.setdefault("reclaim_fn", lambda: 0)
    kw.setdefault("sleep_fn", lambda s: None)
    return ResourceGovernor("cuda:0", **kw)


# ── helpers ──────────────────────────────────────────────────────────────────────────
def test_fmt_duration_ranges():
    assert fmt_duration(0) == "0s"
    assert fmt_duration(45) == "45s"
    assert fmt_duration(90) == "1m 30s"
    assert fmt_duration(3725) == "1h 02m"
    assert fmt_duration(-5) == "0s"
    assert fmt_duration(float("nan")) == "0s"


def test_is_oom_detection():
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) is True
    assert _is_oom(RuntimeError("some other runtime failure")) is False
    assert _is_oom(ValueError("nope")) is False
    assert _is_oom(_FakeOOM("boom"), extra_types=(_FakeOOM,)) is True


def test_is_oom_real_torch_type():
    import torch
    try:
        exc = torch.cuda.OutOfMemoryError("simulated")
    except Exception:
        pytest.skip("OutOfMemoryError not constructible on this torch")
    assert _is_oom(exc) is True


def test_cuda_free_total_none_on_cpu():
    assert cuda_free_total("cpu") is None


def test_cuda_reclaimable_zero_on_cpu():
    assert cuda_reclaimable("cpu") == 0


def test_cuda_free_total_live_when_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    ft = cuda_free_total("cuda:0")
    assert ft is not None and ft[0] >= 0 and ft[1] > 0
    assert cuda_reclaimable("cuda:0") >= 0


# ── disabled / CPU path ────────────────────────────────────────────────────────────────
def test_disabled_governor_chunks_by_max_batch():
    seen = []

    def fn(chunk):
        seen.append(list(chunk))
        return [x * 10 for x in chunk]

    g = ResourceGovernor("cpu", max_batch=3, mem_fn=lambda: None)
    assert g.enabled is False
    out = g.run(fn, list(range(7)))
    assert out == [x * 10 for x in range(7)]
    assert seen == [[0, 1, 2], [3, 4, 5], [6]]


def test_empty_items_returns_empty():
    g = ResourceGovernor("cpu", mem_fn=lambda: None)
    assert g.run(lambda c: [1], []) == []


# ── availability semantics (the key design point) ──────────────────────────────────────
def test_own_cache_not_counted_as_starvation():
    # Raw free is tiny (the model fills the card), but senbon's OWN reclaimable cache is large, so
    # availability is healthy and the run must NOT pause. This is the whole point of the redesign.
    total = 8 << 30
    tiny_free = int(total * 0.01)
    big_cache = int(total * 0.40)
    slept = []
    g = ResourceGovernor("cuda:0", max_batch=2, min_free_frac=0.06,
                         mem_fn=lambda: (tiny_free, total), reclaim_fn=lambda: big_cache,
                         sleep_fn=slept.append)
    out = g.run(lambda c: list(c), [1, 2, 3])
    assert out == [1, 2, 3]
    assert slept == []                 # never paused, because the low free is its own cache
    assert g.paused_s == 0.0


def test_external_pressure_triggers_pause():
    # Raw free low AND reclaimable cache ~0 -> a real external process holds the card -> pause until
    # it frees. The card frees after a couple of sleeps (time passing), not after N reads, matching
    # how a real driver reports current state on every call.
    total = 8 << 30
    low, high = int(total * 0.01), int(total * 0.5)
    state = {"slept": 0}

    def sleep(_s):
        state["slept"] += 1

    def mem():
        return (high, total) if state["slept"] >= 2 else (low, total)

    g = ResourceGovernor("cuda:0", max_batch=2, min_free_frac=0.06,
                         mem_fn=mem, reclaim_fn=lambda: 0, sleep_fn=sleep)
    out = g.run(lambda c: list(c), [1, 2])
    assert out == [1, 2]
    assert g.paused_s > 0 and state["slept"] >= 2


# ── good-gaming-citizen mode: yield COMPUTE to a foreground app ────────────────────────

def test_background_off_never_yields_to_foreground():
    # A foreground app holds ~1G of VRAM but background_mode is off, so the governor does not yield on
    # compute pressure; usable VRAM is healthy, so nothing pauses.
    total = 8 << 30
    g = ResourceGovernor("cuda:0", max_batch=2, min_free_frac=0.06, background_mode=False,
                         mem_fn=lambda: (2 << 30, total), reclaim_fn=lambda: 4 << 30,
                         own_fn=lambda: 5 << 30, sleep_fn=lambda _s: None)
    assert not g._foreground_pressure()
    assert not g._should_pause()
    out = g.run(lambda c: list(c), [1, 2])
    assert out == [1, 2] and g.paused_s == 0


def test_foreground_pressure_yields_even_when_vram_is_fine():
    # The good-gaming-citizen case: usable VRAM is healthy (senbon's own cache is reclaimable), so the
    # VRAM guard alone would NOT pause, but a foreground game is on the card and background mode yields
    # to it, then resumes when the game closes.
    total = 8 << 30
    state = {"slept": 0}

    def sleep(_s):
        state["slept"] += 1

    def mem():
        # Game closes after a couple of sleeps: free 1G -> 3G, used 7G -> 5G, external 2G -> 0.
        return (3 << 30, total) if state["slept"] >= 2 else (1 << 30, total)

    g = ResourceGovernor("cuda:0", max_batch=2, min_free_frac=0.06, background_mode=True,
                         external_pressure_mb=500, mem_fn=mem, reclaim_fn=lambda: 4 << 30,
                         own_fn=lambda: 5 << 30, sleep_fn=sleep)
    assert g._avail_frac() > g.min_free_frac        # not a VRAM-starvation pause
    assert g._foreground_pressure() and g._should_pause()
    out = g.run(lambda c: list(c), [1, 2])
    assert out == [1, 2]
    assert g.paused_s > 0 and state["slept"] >= 2    # yielded, then resumed once the game closed


def test_absolute_detection_catches_already_running_game():
    # A game already running when senbon starts must still be detected: detection is absolute
    # (external VRAM over the threshold), not a rise from a clean baseline.
    total = 8 << 30
    g = ResourceGovernor("cuda:0", max_batch=2, min_free_frac=0.0, background_mode=True,
                         external_pressure_mb=500, mem_fn=lambda: (1 << 30, total),
                         reclaim_fn=lambda: 4 << 30, own_fn=lambda: 5 << 30, sleep_fn=lambda _s: None)
    assert g._external_used() == (2 << 30)           # used 7G minus senbon's own 5G
    assert g._foreground_pressure()


# ── normal, no pressure ────────────────────────────────────────────────────────────────
def test_enabled_processes_all_in_order_no_pressure():
    g = _gov(max_batch=4, mem_fn=lambda: (8 << 30, 8 << 30))
    assert g.enabled is True
    out = g.run(lambda c: [x + 1 for x in c], list(range(10)))
    assert out == [x + 1 for x in range(10)]


# ── OOM shrink ─────────────────────────────────────────────────────────────────────────
def test_oom_shrinks_batch_and_completes():
    def fn(chunk):
        if len(chunk) > 2:
            raise RuntimeError("CUDA out of memory")
        return list(chunk)

    g = _gov(max_batch=8, grow_free_frac=1.1, mem_fn=lambda: (8 << 30, 8 << 30))
    out = g.run(fn, list(range(6)))
    assert out == list(range(6))
    assert g.cur_batch <= 2


def test_oom_custom_exception_type():
    def fn(chunk):
        if len(chunk) > 1:
            raise _FakeOOM("simulated")
        return list(chunk)

    g = _gov(max_batch=4, grow_free_frac=1.1, mem_fn=lambda: (8 << 30, 8 << 30), oom_types=(_FakeOOM,))
    assert g.run(fn, [1, 2, 3]) == [1, 2, 3]


def test_non_oom_exception_propagates():
    def fn(chunk):
        raise ValueError("a real bug, not OOM")

    g = _gov(max_batch=2, mem_fn=lambda: (8 << 30, 8 << 30))
    with pytest.raises(ValueError):
        g.run(fn, [1, 2])


# ── pause / resume ─────────────────────────────────────────────────────────────────────
def test_max_pause_cap_forces_progress():
    total = 8 << 30
    low = int(total * 0.01)
    slept = []
    g = _gov(max_batch=1, min_free_frac=0.5, poll_s=1.0, max_pause_s=3.0,
             mem_fn=lambda: (low, total), sleep_fn=slept.append)
    out = g.run(lambda c: list(c), [7])
    assert out == [7]
    assert sum(slept) <= 3.0 + 1e-6


def test_wait_for_headroom_noop_when_disabled():
    g = ResourceGovernor("cpu", mem_fn=lambda: None)
    assert g.wait_for_headroom() == 0.0


def test_avail_frac_none_transient_returns_full():
    total = 8 << 30
    g = _gov(mem_fn=_mem([(total, total), None]))
    assert g.enabled is True
    assert g._avail_frac() == 1.0
    assert g._avail_frac() == 1.0        # None reading -> treated as fully available, no crash


def test_forced_pause_body_and_cap():
    total = 8 << 30
    low = int(total * 0.05)
    slept = []
    tries = {"n": 0}

    def fn(chunk):
        tries["n"] += 1
        if tries["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return list(chunk)

    g = _gov(max_batch=1, min_free_frac=0.01, grow_free_frac=0.3, poll_s=1.0, max_pause_s=2.0,
             mem_fn=lambda: (low, total), sleep_fn=slept.append)
    out = g.run(fn, [9])
    assert out == [9]
    assert sum(slept) <= 2.0 + 1e-6
    assert g.paused_s > 0


def test_batch1_oom_triggers_forced_pause_then_succeeds():
    total = 8 << 30
    state = {"freed": False}

    def fn(chunk):
        if not state["freed"]:
            state["freed"] = True
            raise RuntimeError("CUDA out of memory")
        return list(chunk)

    g = _gov(max_batch=1, min_free_frac=0.06, grow_free_frac=0.3, poll_s=0.5,
             mem_fn=lambda: (int(total * 0.5), total))
    assert g.run(fn, [42]) == [42]


# ── grow-back ──────────────────────────────────────────────────────────────────────────
def test_batch_grows_back_when_free():
    g = _gov(max_batch=8, grow_free_frac=0.3, mem_fn=lambda: (8 << 30, 8 << 30))
    g.cur_batch = 2
    g.run(lambda c: list(c), list(range(20)))
    assert g.cur_batch > 2


# ── calibration ────────────────────────────────────────────────────────────────────────
def test_calibrate_logs_baseline_once():
    logs = []
    total = 8 << 30
    g = _gov(max_batch=2, mem_fn=lambda: (int(total * 0.3), total), log=logs.append)
    g.run(lambda c: list(c), [1, 2])
    g.run(lambda c: list(c), [3, 4])
    baseline_lines = [m for m in logs if "VRAM baseline" in m]
    assert len(baseline_lines) == 1       # announced once, not per run


# ── ETA ──────────────────────────────────────────────────────────────────────────────
def test_search_progress_eta_excludes_paused():
    logs = []
    clock = {"t": 0.0}

    class _Gov:
        paused_s = 10.0

    sp = SearchProgress(total=4, log=logs.append, governor=_Gov(), clock=lambda: clock["t"])
    clock["t"] = 30.0
    sp.tick()
    assert sp.done == 1
    assert "1/4" in logs[-1]
    assert "paused" in logs[-1]
    assert "ETA 1m 00s" in logs[-1]


def test_search_progress_without_governor():
    logs = []
    clock = {"t": 0.0}
    sp = SearchProgress(total=2, log=logs.append, clock=lambda: clock["t"])
    clock["t"] = 5.0
    sp.tick()
    assert "1/2" in logs[-1]
    assert "paused" not in logs[-1]
