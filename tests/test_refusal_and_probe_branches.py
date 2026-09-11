# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The refusal messages and hardware probes that nothing had ever driven.

Third file of shortlist item A, and the branches here are the ones that matter most per line.
Every function below is code that runs when something has gone WRONG: a track that is not
installed, a dependency that will not import, a card that cannot be read. Those are precisely the
paths a developer never takes, because a developer's machine has all of it, and precisely the
paths a stranger meets first.

`bundled.ensure` is the clearest case. It distinguishes a source checkout (build the track
yourself) from an installed wheel that was packaged without one (a packaging fault you cannot
repair), and the two messages say opposite things. Before this file, neither had been asserted.
The single sentence they replaced told an installed user to run a tool their install does not
carry, to build a corpus of harmful prompts they have no way to obtain.

CPU only. No card, no corpus, no network. The CUDA probes are driven through a fake torch, which
is the point: they are the functions that have to behave on a machine without one.
"""
import builtins

import pytest

from senbonzakura import bundled, doctor, entry, modelcard, resources

# ── modelcard.method_section: every optional field, present and absent ───────────────────────


def test_no_abliteration_record_says_so_rather_than_guessing():
    """NOT MEASURED is a real answer. Omitting the section would read as "nothing to report"."""
    assert modelcard.method_section(None) == [modelcard.NOT_MEASURED]
    assert modelcard.method_section({}) == [modelcard.NOT_MEASURED]


def test_a_minimal_record_defaults_the_method_and_says_matching_is_off():
    """The shortest real record. `matched scoring: no` is stated rather than left out, because a
    reader cannot tell an absent field from a false one.

    The record has no `method` key at all, which is the case the `"searched"` default is for. A
    key present and set to null would render `**None**`, but nothing can produce that: the
    argparse default is `methods.DEFAULT_METHOD`, so the field is always written with a value.
    Checked rather than assumed, because the first version of this test asserted the wrong thing.
    """
    got = modelcard.method_section({"matched_scoring": False})
    assert got == ["- method: **searched**", "- matched scoring: no"]


def test_every_optional_field_appears_when_it_is_present():
    """Four branches that had only ever been skipped, including the two alarms.

    The matching-quality alarm near 1.0 means the matching achieved nothing, which is the
    condition a reader of the card most needs to see and the one most easily left out.
    """
    got = modelcard.method_section({
        "method": "kageyoshi",
        "separation_statistic": "variance_ratio",
        "matched_scoring": True,
        "matched_source": "good_matched_ds",
        "matching_quality": 0.42,
        "match_closeness": 0.91,
    })
    joined = "\n".join(got)
    assert "- method: **kageyoshi**" in joined
    assert "`variance_ratio`" in joined
    assert "- matched scoring: yes" in joined
    assert "`good_matched_ds`" in joined
    assert "0.42" in joined and "means the matching achieved nothing" in joined
    assert "0.91" in joined and "as near as the space allows" in joined


def test_a_matching_quality_of_zero_is_still_reported():
    """`is not None` rather than truthiness, and 0.0 is the value that tells them apart.

    A falsy-but-present alarm silently vanishing from a model card is the kind of omission that
    makes a card say something better than the run did.
    """
    got = "\n".join(modelcard.method_section({"matching_quality": 0.0, "match_closeness": 0.0}))
    assert "matching quality alarm: 0.0" in got
    assert "match closeness: 0.0" in got


# ── doctor.check_torch: the three outcomes, one of which needs torch to be absent ────────────

def test_doctor_reports_a_missing_torch_as_a_failure(monkeypatch):
    """The branch a developer checkout can never reach, and the one an install most often hits."""
    real_import = builtins.__import__

    def _no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no module named torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    got = doctor.check_torch()
    assert got.status == "fail"
    assert "not installed" in got.detail


class _FakeCuda:
    def __init__(self, available, free=None, name="Fake 3060", boom=False):
        self._available, self._free, self._name, self._boom = available, free, name, boom

    def is_available(self):
        return self._available

    def mem_get_info(self):
        if self._boom:
            raise RuntimeError("the driver went away")
        return (self._free, self._free * 2)

    def get_device_name(self, _i):
        return self._name


class _FakeTorch:
    __version__ = "2.9.0"

    def __init__(self, cuda):
        self.cuda = cuda


def _with_torch(monkeypatch, fake):
    import sys
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_doctor_reports_a_working_card_with_its_free_memory(monkeypatch):
    _with_torch(monkeypatch, _FakeTorch(_FakeCuda(True, free=6_000_000_000)))
    got = doctor.check_torch()
    assert got.status == "pass"
    assert "cuda" in got.detail and "Fake 3060" in got.detail and "6.0 GB free" in got.detail


def test_a_card_that_will_not_report_its_memory_still_passes(monkeypatch):
    """THE EXCEPT BRANCH, never taken before. A driver that refuses `mem_get_info` is not a
    reason to tell the user their torch is broken: the card is there and the edit will run.
    """
    _with_torch(monkeypatch, _FakeTorch(_FakeCuda(True, free=1, boom=True)))
    got = doctor.check_torch()
    assert got.status == "pass"
    assert got.detail == "2.9.0, cuda"


def test_no_card_is_a_warning_and_not_a_failure(monkeypatch):
    """Scoring is fine on CPU, so this must not read as "you cannot use this tool"."""
    _with_torch(monkeypatch, _FakeTorch(_FakeCuda(False)))
    got = doctor.check_torch()
    assert got.status == "warn"
    assert "no cuda device" in got.detail
    assert "scoring is fine" in got.fix


# ── resources: the two CUDA accounting probes, on a machine with no CUDA ─────────────────────

class _MemTorch:
    def __init__(self, available=True, reserved=0, allocated=0, boom=False):
        self._boom = boom
        outer = self

        class _C:
            @staticmethod
            def is_available():
                return available

            @staticmethod
            def memory_reserved(_i):
                if outer._boom:
                    raise RuntimeError("no such device")
                return reserved

            @staticmethod
            def memory_allocated(_i):
                return allocated

        self.cuda = _C


@pytest.mark.parametrize("fn", [resources.cuda_reclaimable, resources.cuda_own_reserved])
def test_a_cpu_device_reports_nothing_reclaimable(fn, monkeypatch):
    """`device="cpu"` must short circuit before any CUDA call. Zero, not an exception."""
    monkeypatch.setattr(resources, "_torch", lambda: _MemTorch())
    assert fn("cpu") == 0


@pytest.mark.parametrize("fn", [resources.cuda_reclaimable, resources.cuda_own_reserved])
def test_a_non_string_device_reports_nothing_reclaimable(fn, monkeypatch):
    """A `torch.device` object rather than a string reaches the `isinstance` half of the guard."""
    monkeypatch.setattr(resources, "_torch", lambda: _MemTorch())
    assert fn(object()) == 0


@pytest.mark.parametrize("fn", [resources.cuda_reclaimable, resources.cuda_own_reserved])
def test_no_cuda_at_all_reports_nothing_reclaimable(fn, monkeypatch):
    monkeypatch.setattr(resources, "_torch", lambda: _MemTorch(available=False))
    assert fn("cuda:0") == 0


def test_reclaimable_is_reserved_minus_allocated(monkeypatch):
    """THE NUMBER THE GOVERNOR ACTS ON. Raw "free VRAM" counts this cache as used, so a model
    that has run a few batches looks starved when the memory is its own and empty_cache() would
    hand it straight back.
    """
    monkeypatch.setattr(resources, "_torch",
                        lambda: _MemTorch(reserved=900, allocated=400))
    assert resources.cuda_reclaimable("cuda:0") == 500


def test_own_reserved_is_the_allocator_total(monkeypatch):
    """Subtracting this from the card's total-used leaves what OTHER processes hold, which is how
    a foreground game on WSL2 gets spotted at all: the Windows-side process is invisible to
    nvidia-smi but its VRAM is not.
    """
    monkeypatch.setattr(resources, "_torch", lambda: _MemTorch(reserved=900, allocated=400))
    assert resources.cuda_own_reserved("cuda:0") == 900


@pytest.mark.parametrize("fn", [resources.cuda_reclaimable, resources.cuda_own_reserved])
def test_a_probe_that_raises_reports_zero_rather_than_killing_the_run(fn, monkeypatch):
    """The except branch. An accounting probe must never be the thing that ends an eight-hour job."""
    monkeypatch.setattr(resources, "_torch", lambda: _MemTorch(boom=True))
    assert fn("cuda:0") == 0


# ── bundled.ensure: the two refusals, which say opposite things on purpose ───────────────────

def test_a_checkout_with_no_packed_track_is_told_to_build_one(monkeypatch, tmp_path):
    """A source checkout does not carry the blob, deliberately: it holds harmful prompts and is
    kept out of git. So the remedy is real and the user can act on it.
    """
    monkeypatch.setattr(bundled, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(bundled, "is_available", lambda: False)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda _t: False)
    monkeypatch.setattr(bundled, "running_from_a_checkout", lambda: True)

    with pytest.raises(bundled.BundledTrackError) as e:
        bundled.ensure(log=lambda *_a, **_k: None)
    msg = str(e.value)
    assert "tools/pack_track.py" in msg
    assert "kept out of git on purpose" in msg


def test_an_install_with_no_packed_track_is_told_it_is_a_packaging_fault(monkeypatch, tmp_path):
    """THE MESSAGE THAT REPLACED A HARMFUL ONE, and it had never been asserted.

    An install with no track is a packaging fault, not something the user did. The sentence this
    replaced told them to run a tool their install does not carry, and to build a corpus of
    harmful prompts they have no way to obtain. This one says it cannot be repaired from their
    side, points at the issue tracker, and gives them a way to carry on now.

    It is the wheel that shipped as 0.3.0, so this is not hypothetical.
    """
    monkeypatch.setattr(bundled, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(bundled, "is_available", lambda: False)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda _t: False)
    monkeypatch.setattr(bundled, "running_from_a_checkout", lambda: False)

    with pytest.raises(bundled.BundledTrackError) as e:
        bundled.ensure(log=lambda *_a, **_k: None)
    msg = str(e.value)
    assert "fault in the package rather than anything you did" in msg
    assert bundled.ISSUES in msg
    assert "tools/pack_track.py" not in msg, (
        "an installed user cannot run a tool their install does not carry; that is the exact "
        "advice this message exists to stop giving")


def test_a_current_cache_is_returned_without_unpacking_anything(monkeypatch, tmp_path):
    """The warm path, and it must print the attribution notice rather than skipping it."""
    target = tmp_path / "cache"
    target.mkdir()
    monkeypatch.setattr(bundled, "cache_dir", lambda: target)
    monkeypatch.setattr(bundled, "is_available", lambda: True)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda _t: True)

    def _boom(*_a, **_k):
        raise AssertionError("a current cache must not be extracted again")
    monkeypatch.setattr(bundled, "extract", _boom)

    said = []
    monkeypatch.setattr(bundled, "notice", lambda log=print: said.append(True))
    assert bundled.ensure(log=lambda *_a, **_k: None) == target
    assert said, "the attribution notice is required by the corpus licences and must not be skipped"


def test_the_stamp_is_written_last_so_a_broken_extraction_is_redone(monkeypatch, tmp_path):
    """ORDERING IS THE WHOLE POINT of the stamp, so it is asserted rather than described.

    Stamped last, after everything else is in place, an interrupted extraction leaves a cache
    that FAILS the check and gets redone. Stamped first, it would leave one that passes the check
    and is half a track, which is the worse failure by a distance: every later run would quietly
    use it.
    """
    target = tmp_path / "cache"
    monkeypatch.setattr(bundled, "cache_dir", lambda: target)
    monkeypatch.setattr(bundled, "is_available", lambda: True)

    current = {"value": False}
    monkeypatch.setattr(bundled, "_cache_is_current", lambda _t: current["value"])
    monkeypatch.setattr(bundled, "packed_digest", lambda: "deadbeef")

    order = []

    def _extract(destination, log=print):
        order.append("extract")
        (destination / "track").mkdir(parents=True)
        (destination / "track" / "rows.arrow").write_bytes(b"x")

    monkeypatch.setattr(bundled, "extract", _extract)
    got = bundled.ensure(log=lambda *_a, **_k: None)

    assert got == target
    assert (target / "rows.arrow").is_file(), "the inner track/ directory is what gets promoted"
    stamp = target / bundled.STAMP_NAME
    assert stamp.read_text(encoding="utf-8").strip() == "deadbeef"
    assert order == ["extract"]


def test_a_left_over_unpacking_directory_is_cleared_first(monkeypatch, tmp_path):
    """An interrupted run leaves `.unpacking` behind, and the next run must not build on it."""
    target = tmp_path / "cache"
    tmp = target.with_name(target.name + ".unpacking")
    tmp.mkdir(parents=True)
    (tmp / "stale.txt").write_text("from a dead run", encoding="utf-8")
    target.mkdir()
    (target / "old.txt").write_text("a previous track", encoding="utf-8")

    monkeypatch.setattr(bundled, "cache_dir", lambda: target)
    monkeypatch.setattr(bundled, "is_available", lambda: True)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda _t: False)
    monkeypatch.setattr(bundled, "packed_digest", lambda: "cafe")

    def _extract(destination, log=print):
        assert not (destination / "stale.txt").exists(), "the stale directory was not cleared"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "rows.arrow").write_bytes(b"x")

    monkeypatch.setattr(bundled, "extract", _extract)
    got = bundled.ensure(log=lambda *_a, **_k: None)

    assert (got / "rows.arrow").is_file()
    assert not (got / "old.txt").exists(), "the previous cache must be replaced, not merged into"
    assert not tmp.exists(), "the scratch directory must not be left behind"


# ── entry.main: the two refusals that only exist in an install ───────────────────────────────

def test_a_delegated_command_whose_module_will_not_import_is_explained(monkeypatch):
    """An eleven-frame importlib traceback ending inside our files describes OUR code to somebody
    whose actual problem is their install. This branch turns it into a sentence they can act on.

    It is unreachable in a developer checkout, where the stack is always there, which is why it
    needs a test rather than a run.
    """
    name = next(iter(entry.DELEGATED))

    def _dispatch(_n):
        def _run(_argv):
            raise ImportError("no module named transformers")
        return _run

    monkeypatch.setattr(entry, "dispatch", _dispatch)
    with pytest.raises(SystemExit) as e:
        entry.main([name])
    assert "transformers" in str(e.value) or name in str(e.value)


def test_a_delegated_command_that_runs_returns_its_status(monkeypatch):
    """The ordinary side of the same branch, so the refusal above is not the only behaviour."""
    name = next(iter(entry.DELEGATED))
    monkeypatch.setattr(entry, "dispatch", lambda _n: (lambda argv: 0))
    assert entry.main([name]) == 0
