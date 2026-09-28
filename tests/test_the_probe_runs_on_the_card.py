# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A twenty-fold slowdown this project introduced, measured on hardware, and the guard that missed it.

WHAT HAPPENED, in order, because the order is the whole finding.

The capability probe is the hungriest generation in a run: `--capability-n 200` at
`--capability-max-new 512`. It used to sit between the bake and the save, where an out-of-memory
threw away a baked model after hours of search, so the 2026-09-25 panel moved it to AFTER the save.
That was right.

`free_before_save` moves a resident model into host RAM so `save_pretrained` has the VRAM it needs.
That was also right, and its docstring said it "runs after the post-bake measurement", which was
true when it was written.

Putting the two together left the probe on the far side of the line that takes the weights off the
card. MEASURED ON THE ROG, 2026-09-26, from the run's own log: sixteen probe items took **604
seconds** with the weights on the host against **25 to 37 seconds** with them on the card, while
`nvidia-smi` read **187 MiB**. At the default 200 items that turns a six minute measurement into
two hours, per arm, and a ten arm comparison from an evening's work into a day's.

WHY NONE OF THE THREE EXISTING GUARDS SAW IT. There are three doors into the slow probe and a
comment in `cli.py` names them. The command-line one reads `--device`, and the arm was given
`--device cuda`; the flag was correct the whole way through and only the weights moved. The
after-load one inspects the real model and would have caught it, except that it fires ONCE, on the
first probe of a run, which is the baseline, taken while the model is still resident. So the check
that could see the problem had already run before the problem existed.

That is this project's most-repeated shape written once more: a check covering one spelling of a
defect reports clean on the others. Here the spelling was WHEN rather than WHERE.

THE REPAIR IS TWO PARTS, and both are tested below. The cause: the weights go back on the card after
the write, before the probe, and the save still gets its headroom because the write is finished. The
backstop: the probe re-checks on PLACEMENT rather than on a flag, so it notices whenever the weights
move instead of once per run. The backstop warns rather than refuses, because by then the search is
over and the weights are on disk, so refusing would cost the user the figure rather than the wait.
"""
import types
import typing

import pytest

from senbonzakura import cli


class _FakeParam:
    def __init__(self, kind):
        self.device = types.SimpleNamespace(type=kind)


class _FakeModel:
    """Only what the placement question needs: parameters, and a `.to` that records."""

    def __init__(self, kind="cuda", dispatched=False):
        self._kind = kind
        self.moves = []
        if dispatched:
            self.hf_device_map = {"model.layers.0": 0}

    def parameters(self):
        return [_FakeParam(self._kind), _FakeParam(self._kind)]

    def to(self, dev):
        self.moves.append(str(dev))
        self._kind = str(dev).split(":")[0]
        return self


def _abl(dev="cuda", model=None):
    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.dev = dev
    obj.model = model if model is not None else _FakeModel()
    obj.log = lambda _m: None
    obj._pristine = {}
    obj._dirty = set()
    obj._weights_parked_on_host = False
    obj._last_probe_device = None
    return obj


# ── reading where the weights actually are ───────────────────────────────────────────

class TestWhereTheWeightsLive:
    def test_it_answers_the_kind_and_not_the_index(self):
        """`cuda:0` against `cuda:1` does not change whether this is card speed or host speed."""
        assert cli._weights_live_on(_FakeModel("cuda")) == "cuda"
        assert cli._weights_live_on(_FakeModel("cpu")) == "cpu"

    def test_a_split_model_is_reported_as_mixed_rather_than_as_one_of_its_halves(self):
        class Split(_FakeModel):
            def parameters(self):
                return [_FakeParam("cuda"), _FakeParam("cpu")]

        assert cli._weights_live_on(Split()) == "mixed"

    def test_something_that_cannot_be_asked_returns_none_rather_than_a_guess(self):
        """ABSENCE AND A VERDICT MUST NOT SHARE A REPRESENTATION. `None` means the question could
        not be answered, and the caller treats that as "do not warn" rather than as "on the host".
        """
        class Hopeless:
            def parameters(self):
                raise RuntimeError("no")

        assert cli._weights_live_on(Hopeless()) is None
        assert cli._weights_live_on(object()) is None

    def test_a_model_with_no_parameters_is_not_reported_as_being_somewhere(self):
        class Empty(_FakeModel):
            def parameters(self):
                return []

        assert cli._weights_live_on(Empty()) is None


# ── the cause: the weights come back ─────────────────────────────────────────────────

class TestTheWeightsComeBack:
    def test_a_resident_model_is_parked_for_the_write_and_put_back_after_it(self):
        obj = _abl()
        obj.free_before_save()
        assert obj.model.moves == ["cpu"], "the write no longer gets its headroom"
        assert obj._weights_parked_on_host is True
        obj.restore_device_after_save()
        assert obj.model.moves == ["cpu", "cuda"], (
            "the weights were not put back, so the probe after the save generates on the host: "
            "604 seconds for sixteen items against 25 to 37, measured on the ROG")
        assert cli._weights_live_on(obj.model) == "cuda"

    def test_putting_them_back_twice_moves_them_once(self):
        """Idempotent, because this runs on a path that can be re-entered by a resume."""
        obj = _abl()
        obj.free_before_save()
        obj.restore_device_after_save()
        obj.restore_device_after_save()
        assert obj.model.moves == ["cpu", "cuda"]

    def test_a_model_dispatched_across_devices_is_left_alone_in_both_directions(self):
        """THE HALF THAT MUST NOT MOVE. A model accelerate dispatched cannot be moved with `.to()`,
        so `free_before_save` leaves it alone, and the way back has to make the same choice. A repair
        that remembered nothing would have moved a dispatched model on the return leg, which is the
        crash the original condition exists to avoid.
        """
        obj = _abl(model=_FakeModel(dispatched=True))
        obj.free_before_save()
        assert obj.model.moves == []
        obj.restore_device_after_save()
        assert obj.model.moves == []

    def test_a_cpu_run_is_never_moved_anywhere(self):
        obj = _abl(dev="cpu", model=_FakeModel("cpu"))
        obj.free_before_save()
        obj.restore_device_after_save()
        assert obj.model.moves == []

    def test_nothing_is_restored_when_nothing_was_parked(self):
        obj = _abl()
        obj.restore_device_after_save()
        assert obj.model.moves == []


# ── the backstop: a check that can see a change ──────────────────────────────────────

class TestTheBackstopNoticesAMove:
    """These drive `_capability_score` far enough to reach the placement check and no further.

    ONE ITEM, NOT ZERO, and the first version of this file got that wrong. `_capability_score`
    returns early on an empty list, before the check, which is the right place for it: a probe that
    generates nothing has no placement worth reporting, and warning there would be noise. So the
    cases below hand it one item and let the generation fail immediately for want of a real
    tokeniser, by which point the check has already run and recorded what it saw.
    """

    ONE_ITEM: typing.ClassVar = [("what is 1 + 1", "#### 2")]

    def _probe(self, obj):
        try:
            obj._capability_score(self.ONE_ITEM)
        except Exception:                                       # the generation, not the check
            pass
        return obj._last_probe_device

    def test_the_first_probe_records_where_it_ran_and_says_nothing(self):
        obj = _abl()
        obj.args = types.SimpleNamespace(capability_max_new=32, capability_eval="bundled",
                                         slow_probe_ok=True, capability_task="numeric")
        obj._slow_probe_checked = True
        said = []
        obj.log = said.append
        self._probe(obj)
        assert not [s for s in said if "WARNING" in s], (
            "the first probe warned, and there is nothing to compare it against yet")

    def test_a_probe_that_moved_to_the_host_between_measurements_says_so(self):
        """THE CASE THAT ACTUALLY HAPPENED, and the one a once-per-run check cannot reach."""
        obj = _abl()
        obj.args = types.SimpleNamespace(capability_max_new=32, capability_eval="bundled",
                                         slow_probe_ok=True, capability_task="numeric")
        obj._slow_probe_checked = True
        obj._last_probe_device = "cuda"
        obj.model = _FakeModel("cpu")
        said = []
        obj.log = said.append
        self._probe(obj)
        warned = [s for s in said if "WARNING" in s]
        assert warned, (
            "the weights moved from the card to the host between two capability measurements and "
            "nothing said so. That is the twenty-fold slowdown, and the three existing guards "
            "cannot see it: two read the declared device, and the third fires once.")
        assert "twenty times" in warned[0], "the warning does not say what it costs"

    def test_it_does_not_warn_when_the_placement_did_not_change(self):
        obj = _abl()
        obj.args = types.SimpleNamespace(capability_max_new=32, capability_eval="bundled",
                                         slow_probe_ok=True, capability_task="numeric")
        obj._slow_probe_checked = True
        obj._last_probe_device = "cuda"
        said = []
        obj.log = said.append
        self._probe(obj)
        assert not [s for s in said if "WARNING" in s]

    def test_a_placement_that_cannot_be_read_does_not_cry_wolf(self):
        """`None` is "could not be asked", and warning on it would train people to ignore this."""
        obj = _abl()
        obj.args = types.SimpleNamespace(capability_max_new=32, capability_eval="bundled",
                                         slow_probe_ok=True, capability_task="numeric")
        obj._slow_probe_checked = True
        obj._last_probe_device = "cuda"

        class Hopeless(_FakeModel):
            def parameters(self):
                raise RuntimeError("no")

        obj.model = Hopeless()
        said = []
        obj.log = said.append
        self._probe(obj)
        assert not [s for s in said if "WARNING" in s]


def test_the_probe_is_reached_after_the_weights_are_restored_and_not_before():
    """THE ORDER, asserted on the source's call sequence rather than on a comment.

    Both calls are in `_bake_and_save`, and the whole defect was their relative position: the probe
    moved after the save and the save parks the weights. A test that checked only that both exist
    would pass on the broken arrangement.
    """
    import inspect
    body = inspect.getsource(cli.Abliterator._bake_and_save)
    park = body.index("self.free_before_save()")
    back = body.index("self.restore_device_after_save()")
    probe = body.index("self._capability_score(cap_items)")
    assert park < back < probe, (
        "the order is park, restore, probe. Anything else puts the hungriest generation in the run "
        "on whichever device the save happened to leave the weights on.")


# ── an out of memory putting them back must not cost the record ───────────────────────────────────

class _RefusesToMoveBack(_FakeModel):
    """A card with no room left. `free_before_save` succeeded; something else took the VRAM since."""

    def __init__(self, fail_times=99):
        super().__init__("cpu")
        self.fail_times = fail_times
        self.attempts = 0

    def to(self, dev):
        self.attempts += 1
        if str(dev).startswith("cuda") and self.attempts <= self.fail_times:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().to(dev)


def _parked(model):
    obj = _abl(model=model)
    obj._weights_parked_on_host = True
    return obj


def test_an_out_of_memory_putting_the_weights_back_does_not_end_the_run():
    """WHAT PROMPTED IT, 2026-09-28. Two reviewers reached this independently.

    By the time this runs the weights are on disk and `abliteration.json`, the model card and the
    completion marker are not yet written. A bare `self.model.to(self.dev)` therefore ended a
    finished run with a traceback and left a checkpoint with no receipt, which is the state the
    whole provenance layer exists to prevent.

    It is six lines before the probe whose own out of memory this ordering was introduced to
    survive, so the run was protected against the expensive failure and not against the cheap one
    immediately before it.
    """
    model = _RefusesToMoveBack()
    obj = _parked(model)
    said = []
    obj.log = said.append

    obj.restore_device_after_save()   # must not raise

    assert model.attempts >= 2, "it gave up without clearing the cache and trying once more"
    printed = "\n".join(said)
    assert "WARNING" in printed, f"the degradation was silent: {printed!r}"
    assert "twenty times slower" in printed, (
        "the warning does not say what it costs, so a reader cannot tell whether to re-run")
    assert "saved weights" in printed, (
        "nothing tells the reader their model is fine, which is the one thing they will want to "
        f"know: {printed!r}")


def test_a_retry_that_succeeds_is_not_reported_as_a_degradation():
    """The cache clear is worth having only if the second attempt is really taken."""
    model = _RefusesToMoveBack(fail_times=1)
    obj = _parked(model)
    said = []
    obj.log = said.append

    obj.restore_device_after_save()

    assert model.attempts == 2
    assert obj._weights_parked_on_host is False
    assert "WARNING" not in "\n".join(said)
    assert model.moves and model.moves[-1].startswith("cuda")


def test_the_weights_are_still_recorded_as_parked_when_the_move_failed():
    """Otherwise the next thing to ask is told they are on the card, and they are not."""
    obj = _parked(_RefusesToMoveBack())
    obj.log = lambda _m: None
    obj.restore_device_after_save()
    assert obj._weights_parked_on_host is True, (
        "a failed move cleared the flag, so every later check believes the weights are on the card")


def test_an_error_that_is_not_an_out_of_memory_is_still_raised():
    """The handler is for a full card, not for hiding faults."""
    class Broken(_FakeModel):
        def to(self, _dev):
            raise RuntimeError("device-side assert triggered")

    obj = _parked(Broken())
    obj.log = lambda _m: None
    with pytest.raises(RuntimeError, match="device-side assert"):
        obj.restore_device_after_save()


def test_the_weights_are_not_moved_back_when_nothing_that_follows_needs_the_card():
    """`cap_items` is empty under `--capability-n 0`, which the low-memory advice and the Colab
    probe's reduced budget both set. Those runs moved a whole model onto the card for nothing.
    """
    import inspect
    body = inspect.getsource(cli.Abliterator._bake_and_save)
    back = body.index("self.restore_device_after_save()")
    guard = body.rindex("if cap_items:", 0, back)
    between = body[guard:back]
    assert between.count("\n") < 12, (
        "the restore is no longer inside the branch that checks there is a probe to run:\n"
        f"{between[:400]}")
