# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The wavefront executor: the refusal map with one block on the card at a time.

THE PICTURE. The map runs the same prompts through the model once per condition, each time with some
blocks nudged. The reference order walks the whole model for every condition, so every block's weights
are read again for every one of them. The wavefront turns the loops inside out: it walks the model once
per group of micro batches, and at each block it pushes every condition that has reached that block
through it before moving on. A block's weights are copied once per sweep, not once per condition.

    reference                                wavefront
    for each condition:                      for each group of micro batches:
        for each block 0..n:                     for each block j in 0..n:
            copy block, run it                       copy block j once
                                                     for each condition that has reached j: run it

THE NUMBERS DO NOT CHANGE, AND THAT IS CHECKED. Every block is the model's own block, called with the
keyword arguments the model itself gave it (captured once per micro batch), on a micro batch of exactly
the reference's size and padding, because the card's arithmetic gives different bits for a different
batch shape. The embedding, the final norm and the output head are the model's own code too: a pass with
every block stood in for by what the walk wrote. The first micro batch of the first run is compared
bitwise (`torch.equal`) against the reference path, and a difference stops the run. It never falls back.

WHAT IT REFUSES, BY NAME. A block that is handed a cache, or anything else that carries state outside
its own arguments, hands on more than its stream, changes a buffer while it runs, or shares weights with
anything else cannot be walked this way. The run stops and says which block and why; that model keeps the
reference executor.

The captured arguments are taken in a pass whose blocks are stand-ins that hand their input straight on,
so the capture needs no block weights on the card. That is safe only while the arguments do not depend on
the stream's values; the self check is what would show it if they did.
"""
from __future__ import annotations

import contextlib
import time

import torch

from . import attribution as A
from .firsttoken import last_token_logits

#: How much state the live conditions may take, in bytes, for the micro batches in one sweep. The sweep
#: is split into groups of micro batches to stay under it, which costs one more walk of the blocks per
#: group and changes no number.
STATE_BYTES = 4 * 1024 ** 3
#: The self check compares at most this many conditions of the first micro batch with the reference path,
#: spread over the ones asked for. Each costs one full pass of that micro batch.
CHECK_CONDITIONS = 6
#: Arguments that must be None: any of them set means the block reads or writes a cache.
CACHE_ARGUMENTS = ("past_key_values", "past_key_value", "cache_params", "cache", "layer_cache")
#: How deep a tuple, list or dict argument is searched for something that is not plain data.
MAX_ARGUMENT_DEPTH = 4
#: Seconds between progress lines inside a sweep.
HEARTBEAT_S = 30.0


def block_name(j, layer):
    return f"block {j} ({type(layer).__name__})"


def describe(blocks):
    """A condition in words, for messages."""
    blocks = tuple(blocks)
    if not blocks:
        return "no blocks"
    if len(blocks) > 6:
        return f"{len(blocks)} blocks from {blocks[0]} to {blocks[-1]}"
    word = "block" if len(blocks) == 1 else "blocks"
    return f"{word} " + ", ".join(map(str, blocks))


def overlap_seconds(first, second):
    """How long two `(start, end)` intervals overlap, never below zero."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def pick_check_conditions(conditions, limit=CHECK_CONDITIONS):
    """Which conditions the self check runs: the unedited one if asked for, then spread over the rest.

    Ordered by where the condition first edits and how many blocks it edits, so the choice is the same
    on every run and covers early, middle and late edits.
    """
    unique = sorted(set(conditions), key=lambda c: (min(c) if c else -1, len(c), c))
    if len(unique) <= limit:
        return unique
    spread = sorted({round(i * (len(unique) - 1) / (limit - 1)) for i in range(limit)})
    return [unique[i] for i in spread]


def _vet(where, label, value, depth):
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor, torch.Size,
                                           torch.dtype, torch.device)):
        return
    if isinstance(value, (tuple, list)) and depth < MAX_ARGUMENT_DEPTH:
        for item in value:
            _vet(where, label, item, depth + 1)
        return
    if isinstance(value, dict) and depth < MAX_ARGUMENT_DEPTH:
        for item in value.values():
            _vet(where, label, item, depth + 1)
        return
    raise A.AttributionError(
        f"{where} is handed {label}, which is a {type(value).__name__} and not plain data or tensors, "
        f"so it may carry state from outside the block. The wavefront can't run this model; use "
        f"--executor reference.")


def vet_arguments(where, args, kwargs):
    """Refuse a block whose arguments carry state outside the block (a cache, an object of its own)."""
    for key, value in kwargs.items():
        if key in CACHE_ARGUMENTS and value is not None:
            raise A.AttributionError(
                f"{where} is handed a cache in `{key}`, so what it computes depends on state outside its "
                f"arguments. The wavefront can't run this model; use --executor reference.")
        if key == "use_cache" and value:
            raise A.AttributionError(
                f"{where} is asked to keep a cache (`use_cache` is on), so it writes state outside its "
                f"arguments. The wavefront can't run this model; use --executor reference.")
    for position, value in enumerate(args):
        _vet(where, f"argument {position}", value, 0)
    for key, value in kwargs.items():
        _vet(where, f"`{key}`", value, 0)


def stream_only(out):
    """True when a block hands on its stream and nothing else, the only thing a walk can carry."""
    if isinstance(out, tuple) and any(isinstance(t, torch.Tensor) for t in out[1:]):
        return False
    return isinstance(A.unwrap(out), torch.Tensor)


class _Stand(torch.nn.Module):
    """A stand-in for a block. Attributes it doesn't have come from the block it replaces, because the
    model's own loop reads some of them (the layer type, for one) to choose what to hand each block.
    """

    def __init__(self, original):
        super().__init__()
        object.__setattr__(self, "_original", original)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            original = self.__dict__.get("_original")
            if original is None:
                raise
            return getattr(original, name)


class _Capture(_Stand):
    """Records what the model hands a block, and hands the stream straight on."""

    def __init__(self, original, record, j):
        super().__init__(original)
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_j", j)

    def forward(self, *args, **kwargs):
        if args:
            first, rest, by_name, kw = args[0], tuple(args[1:]), False, dict(kwargs)
        elif "hidden_states" in kwargs:
            first, rest, by_name = kwargs["hidden_states"], (), True
            kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        else:
            raise A.AttributionError(
                f"block {self._j} was called with no hidden states that could be found, by position or by "
                f"name, so the wavefront can't tell what it is handed.")
        self._record[self._j] = (rest, kw, by_name)
        if self._j == 0:
            self._record["x0"] = first.detach()
        return first


class _Hold(_Stand):
    """Hands back what the walk wrote, so the model's own norm and head can finish the job."""

    def __init__(self, original, box):
        super().__init__(original)
        object.__setattr__(self, "_box", box)

    def forward(self, *_args, **_kwargs):
        return self._box[0]


def _slots(layer):
    """Every tensor a block owns, as `(owner module, name, is a parameter)`, each tensor once."""
    seen, out = set(), []
    for module in layer.modules():
        for name, tensor in module._parameters.items():
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                out.append((module, name, True))
        for name, tensor in module._buffers.items():
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                out.append((module, name, False))
    return out


def _get(owner, name, is_param):
    return owner._parameters[name] if is_param else owner._buffers[name]


def _put(owner, name, is_param, tensor):
    if is_param:
        owner._parameters[name].data = tensor
    else:
        owner._buffers[name] = tensor


class _Clock:
    """Marks on a timeline, in seconds from the start of the sweep, from graphics card events when the
    work is on a card (so overlapped work is timed where it ran) and from the wall clock otherwise.
    """

    def __init__(self, cuda):
        self.cuda = cuda
        self.marks = {}
        if cuda:                                                    # pragma: no cover - needs a card
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()
        else:
            self.origin = time.perf_counter()

    def mark(self, key, stream=None):
        if self.cuda:                                               # pragma: no cover - needs a card
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream)
            self.marks[key] = event
        else:
            self.marks[key] = time.perf_counter() - self.origin

    def resolve(self):
        if self.cuda:                                               # pragma: no cover - needs a card
            torch.cuda.synchronize()
            return {key: self.origin.elapsed_time(event) / 1000.0 for key, event in self.marks.items()}
        return dict(self.marks)


class _Window:
    """The blocks on their way to the card: parked on the host, copied one ahead of the one that runs.

    With `streaming` off the blocks stay where they are and every call here does nothing. With it on,
    `park` takes each block's tensors off the card (into pinned host memory when there is a card), and
    the sweep asks for block j + 1 to be copied while block j runs, so at most two blocks are resident.
    The module objects never change, only the tensors inside them, so hooks stay attached.
    """

    def __init__(self, layers, device, streaming, log):
        self.layers, self.device, self.streaming, self.log = layers, device, streaming, log
        self.cuda = device.type == "cuda"
        self.slots = [_slots(layer) for layer in layers]
        self.masters = None
        self.resident = {}
        self.ready_events = {}
        self.peak_resident = 0
        self.clock = None
        self.pinned = True
        if self.streaming and self.cuda:                            # pragma: no cover - needs a card
            self.copy_stream = torch.cuda.Stream()

    def park(self):
        if not self.streaming or self.masters is not None:
            return
        self.masters = [[] for _ in self.layers]
        try:
            for j, slots in enumerate(self.slots):
                for owner, name, is_param in slots:
                    tensor = _get(owner, name, is_param)
                    master = tensor.detach().to("cpu")
                    if self.cuda:                                   # pragma: no cover - needs a card
                        master = self._pin(master)
                    self.masters[j].append((owner, name, is_param, master))
                    _put(owner, name, is_param, master)
        except BaseException:
            self.unpark()
            raise

    def _pin(self, tensor):                                         # pragma: no cover - needs a card
        if self.pinned:
            try:
                return tensor.pin_memory()
            except RuntimeError as error:
                self.pinned = False
                self.log(f"  wavefront: pinned memory was refused ({str(error).splitlines()[0]}); copies "
                         f"from ordinary host memory can't overlap with compute")
        return tensor

    def unpark(self):
        if self.masters is None:
            return
        masters, self.masters = self.masters, None
        self.resident.clear()
        for entries in masters:
            for owner, name, is_param, master in entries:
                _put(owner, name, is_param, master.to(self.device))

    def begin(self, clock):
        self.clock = clock
        self.resident.clear()
        self.ready_events.clear()

    def prefetch(self, j):
        if not self.streaming or j >= len(self.layers) or j in self.resident:
            return
        keep = []
        if self.cuda:                                               # pragma: no cover - needs a card
            with torch.cuda.stream(self.copy_stream):
                self.clock.mark(("copy", j, "start"), self.copy_stream)
                for owner, name, is_param, master in self.masters[j]:
                    on_card = master.to(self.device, non_blocking=True)
                    _put(owner, name, is_param, on_card)
                    keep.append(on_card)
                self.clock.mark(("copy", j, "end"), self.copy_stream)
                done = torch.cuda.Event()
                done.record(self.copy_stream)
            current = torch.cuda.current_stream()
            for tensor in keep:
                tensor.record_stream(current)
            self.ready_events[j] = done
        else:
            self.clock.mark(("copy", j, "start"))
            for owner, name, is_param, master in self.masters[j]:
                _put(owner, name, is_param, master.clone())
            self.clock.mark(("copy", j, "end"))
        self.resident[j] = True
        self.peak_resident = max(self.peak_resident, len(self.resident))

    def ready(self, j):
        if self.streaming and self.cuda and j in self.ready_events:  # pragma: no cover - needs a card
            torch.cuda.current_stream().wait_event(self.ready_events.pop(j))

    def release(self, j):
        if not self.streaming or j not in self.resident:
            return
        for owner, name, is_param, master in self.masters[j]:
            _put(owner, name, is_param, master)
        del self.resident[j]


class WavefrontExecutor:
    """Runs a list of conditions over a list of micro batches, block by block.

    `run` is the whole contract: encoded micro batches in, `{condition: [per micro batch logits at the
    asked token ids]}` out, with exactly the numbers the reference path gives for each condition. The
    subject owns what the micro batches are and how they come back into prompt order.
    """

    def __init__(self, model, layers, *, standing, reference, log=print, streaming=True,
                 state_bytes=STATE_BYTES, host_free=None):
        if state_bytes <= 0:
            raise A.AttributionError(f"the wavefront state budget is {state_bytes} bytes, which leaves "
                                     f"no room for a single condition.")
        self.model, self.layers, self.log = model, layers, log
        self.standing, self.reference = standing, reference
        self.state_bytes, self.host_free = int(state_bytes), host_free
        self.device = getattr(model, "device", None) or next(model.parameters()).device
        self.streaming = bool(streaming)
        self.window = _Window(layers, self.device, self.streaming, log)
        #: States wait on the host while blocks stream, and on the card when the blocks stay resident.
        self.home = torch.device("cpu") if self.streaming else self.device
        self.vetted = False
        self.check = None
        self.sweeps = 0
        self.blocks = [{"copy_s": 0.0, "compute_s": 0.0, "overlap_s": 0.0} for _ in layers]

    # ----------------------------------------------------------------------------- what it refuses
    def _refuse_unwalkable(self):
        """Blocks the walk can't move: offloaded by the loader, or sharing weights with anything else."""
        if self.vetted:
            return
        for j, layer in enumerate(self.layers):
            where = block_name(j, layer)
            if any(hasattr(m, "_hf_hook") for m in layer.modules()):
                raise A.AttributionError(
                    f"{where} is offloaded by the loader's own hooks, which move its weights on every call "
                    f"and so can't be walked this way. The wavefront can't run this model yet; use "
                    f"--executor reference.")
            if any(p.is_meta for p in layer.parameters()):
                raise A.AttributionError(
                    f"{where} has weights that aren't loaded (they are on the meta device), so it can't be "
                    f"copied to the card. The wavefront can't run this model; use --executor reference.")
        owners = {}
        for j, layer in enumerate(self.layers):
            for p in layer.parameters():
                if owners.setdefault(id(p), j) != j:
                    raise A.AttributionError(
                        f"{block_name(j, layer)} shares a weight with block {owners[id(p)]}, so streaming one "
                        f"block would move the other's. The wavefront can't run this model; use "
                        f"--executor reference.")
        for name, p in self.model.named_parameters(remove_duplicate=False):
            if id(p) in owners and not any(name.startswith(f"{prefix}.") for prefix in self._block_prefixes()):
                raise A.AttributionError(
                    f"the weight `{name}` outside the blocks is shared with block {owners[id(p)]}, so "
                    f"streaming that block would move it too. The wavefront can't run this model; use "
                    f"--executor reference.")
        self.vetted = True

    def _block_prefixes(self):
        ids = {id(layer) for layer in self.layers}
        return [name for name, module in self.model.named_modules() if id(module) in ids]

    def _normalise(self, conditions, direction):
        nb = len(self.layers)
        out = []
        for given in conditions:
            blocks = tuple(int(b) for b in given)
            bad = [k for k in blocks if not 0 <= k < nb]
            if bad:
                raise A.AttributionError(f"block(s) {bad} do not exist: this model has blocks 0 to {nb - 1}.")
            if blocks and direction is None:
                raise A.AttributionError(
                    "a condition removes a direction from some blocks, but no direction was given.")
            if blocks not in out:
                out.append(blocks)
        if any(out) and direction is not None:
            with A.ablating(self.layers, direction, (0,)):
                pass
        return out

    # ------------------------------------------------------------------------------------- capture
    @torch.no_grad()
    def _capture(self, enc):
        """What each block is called with for this micro batch, from a pass whose blocks hand the stream on."""
        record = {}
        originals = list(self.layers)
        try:
            for j, layer in enumerate(originals):
                self.layers[j] = _Capture(layer, record, j)
            last_token_logits(self.model, enc, log=self.log)
        finally:
            for j, layer in enumerate(originals):
                self.layers[j] = layer
        for j, layer in enumerate(originals):
            if j not in record:
                raise A.AttributionError(f"{block_name(j, layer)} never ran in the capture pass, so the "
                                         f"wavefront can't tell what it is handed.")
            rest, kw, _by_name = record[j]
            vet_arguments(block_name(j, layer), rest, kw)
        record["x0"] = record["x0"].to(self.home)
        return record

    def _state_bytes(self, cap):
        x0 = cap["x0"]
        return x0.numel() * x0.element_size()

    def _budget(self):
        budget = self.state_bytes
        if self.device.type == "cuda" and not self.streaming:       # pragma: no cover - needs a card
            budget = min(budget, int(0.4 * torch.cuda.mem_get_info(self.device)[0]))
        if self.streaming and self.host_free is not None:
            free = self.host_free()
            if free is not None:
                budget = min(budget, free // 2)
        return budget

    def _groups(self, caps, live):
        """Micro batches in the order given, split so each group's states stay under the budget."""
        budget = self._budget()
        cost = [self._state_bytes(cap) * (live + 1) for cap in caps]
        if cost and max(cost) > budget:
            worst = max(range(len(cost)), key=cost.__getitem__)
            raise A.AttributionError(
                f"one micro batch of {live} conditions needs {cost[worst] / 2 ** 30:.2f} GiB of state, over "
                f"the {budget / 2 ** 30:.2f} GiB the wavefront may use here. Lower --batch, or use "
                f"--executor reference.")
        groups, current, used = [], [], 0
        for m, c in enumerate(cost):
            if current and used + c > budget:
                groups.append(current)
                current, used = [], 0
            current.append(m)
            used += c
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------------------------- the walk
    def _call(self, j, cap, x, watch):
        layer = self.layers[j]
        rest, kw, by_name = cap
        x = x.to(self.device)
        before = None
        if watch:
            tensors = [*layer.parameters(), *layer.buffers()]
            before = [t._version for t in tensors]
        out = layer(**{**kw, "hidden_states": x}) if by_name else layer(x, *rest, **kw)
        where = block_name(j, layer)
        if not stream_only(out):
            raise A.AttributionError(
                f"{where} hands on more than its stream, which a block by block walk can't carry. The "
                f"wavefront can't run this model; use --executor reference.")
        if A.unwrap(out).shape != x.shape:
            raise A.AttributionError(
                f"{where} returned a stream of shape {tuple(A.unwrap(out).shape)} for an input of "
                f"{tuple(x.shape)}, so the shapes aren't frozen. The wavefront can't run this model; use "
                f"--executor reference.")
        if watch and [t._version for t in tensors] != before:
            raise A.AttributionError(
                f"{where} changed one of its own weights or buffers while it ran, so what it computes "
                f"depends on how many times it has run. The wavefront can't run this model; use "
                f"--executor reference.")
        return out

    @torch.no_grad()
    def _sweep(self, encs, caps, group, direction, conds, ids, keep_full, watch):
        """One walk of the blocks for the micro batches in `group`. Returns the readouts and kept logits."""
        nb = len(self.layers)
        none = ()
        states = {m: {none: caps[m]["x0"]} for m in group}
        finals = {}
        results = {c: {} for c in conds}
        kept = {}
        clock = _Clock(self.device.type == "cuda")
        self.window.begin(clock)
        self.window.prefetch(0)
        beat = time.monotonic()
        started = {j: [c for c in conds if c and min(c) == j] for j in range(nb)}
        with self.standing():
            for j in range(nb):
                self.window.ready(j)
                self.window.prefetch(j + 1)
                clock.mark(("compute", j, "start"))
                for c in started[j]:
                    for m in group:
                        states[m][c] = states[m][none]
                live = [none] + [c for c in conds if c and min(c) <= j]
                by_reps = {}
                for c in live:
                    by_reps.setdefault(c.count(j), []).append(c)
                for reps in sorted(by_reps):
                    with (A.ablating(self.layers, direction, (j,) * reps) if reps else contextlib.nullcontext()):
                        for m in group:
                            for c in by_reps[reps]:
                                out = self._call(j, caps[m][j], states[m][c], watch)
                                if j == nb - 1:
                                    finals[m, c] = out
                                    del states[m][c]
                                else:
                                    states[m][c] = A.unwrap(out).to(self.home)
                clock.mark(("compute", j, "end"))
                self.window.release(j)
                if time.monotonic() - beat > HEARTBEAT_S:           # pragma: no cover - timing
                    beat = time.monotonic()
                    self.log(f"  wavefront: block {j + 1} of {nb}, micro batches {group[0] + 1} to "
                             f"{group[-1] + 1} of {len(encs)}")
        self._finish(encs, finals, ids, results, kept, keep_full)
        self._account(clock.resolve(), nb)
        return results, kept

    @torch.no_grad()
    def _finish(self, encs, finals, ids, results, kept, keep_full):
        """The model's own embedding, final norm and head, applied to what the walk wrote."""
        box = [None]
        originals = list(self.layers)
        try:
            for j, layer in enumerate(originals):
                self.layers[j] = _Hold(layer, box)
            for (m, c), out in list(finals.items()):
                del finals[m, c]
                if c not in results:
                    continue
                box[0] = out
                logits = last_token_logits(self.model, encs[m], log=self.log)
                results[c][m] = logits[:, ids.to(logits.device)].cpu()
                if (c, m) in keep_full:
                    kept[c, m] = logits.cpu()
        finally:
            for j, layer in enumerate(originals):
                self.layers[j] = layer

    def _account(self, marks, nb):
        self.sweeps += 1
        previous = None
        for j in range(nb):
            compute = (marks[("compute", j, "start")], marks[("compute", j, "end")])
            copy_key = ("copy", j, "start")
            copy = (marks[copy_key], marks[("copy", j, "end")]) if copy_key in marks else None
            copy_s = 0.0 if copy is None else copy[1] - copy[0]
            hidden = 0.0 if copy is None or previous is None else overlap_seconds(copy, previous)
            self.blocks[j]["copy_s"] += copy_s
            self.blocks[j]["compute_s"] += compute[1] - compute[0]
            self.blocks[j]["overlap_s"] += hidden
            if self.streaming:
                share = f" ({hidden / copy_s:.0%} of the copy hidden)" if copy_s > 0 else ""
                self.log(f"  wavefront block {j}: copy {copy_s:.3f} s, compute {compute[1] - compute[0]:.3f} s, "
                         f"overlap {hidden:.3f} s{share}")
            previous = compute

    def summary(self):
        """What the walk cost, per block and in total, for the log and the artefacts beside the result."""
        return {"streaming": self.streaming, "sweeps": self.sweeps, "peak_blocks_resident": self.window.peak_resident,
                "copy_s": sum(b["copy_s"] for b in self.blocks), "compute_s": sum(b["compute_s"] for b in self.blocks),
                "overlap_s": sum(b["overlap_s"] for b in self.blocks), "blocks": [dict(b) for b in self.blocks],
                "self_check": self.check}

    # ------------------------------------------------------------------------------------- the run
    @torch.no_grad()
    def run(self, encs, direction, conditions, ids):
        """Every condition's logits at `ids`, one tensor per micro batch, in the micro batches' order."""
        if self.check is not None and not self.check["passed"]:
            raise A.AttributionError("the wavefront executor already failed its self check on this model, so it "
                                     "won't run again; use --executor reference.")
        self._refuse_unwalkable()
        conds = self._normalise(conditions, direction)
        if not encs:
            return {c: [] for c in conds}
        caps = [self._capture(enc) for enc in encs]
        groups = self._groups(caps, len([c for c in conds if c]))
        want = self._reference_logits(encs[0], direction, conds) if self.check is None else None
        if want is not None:
            groups = self._with_first_alone(groups)
        out = {c: [None] * len(encs) for c in conds}
        self.window.park()
        try:
            for group in groups:
                checking = want is not None and 0 in group
                keep_full = {(c, 0) for c in want} if checking else set()
                results, kept = self._sweep(encs, caps, group, direction, conds, ids, keep_full, watch=checking)
                if checking:
                    self._verify(want, kept)
                    want = None
                for c in conds:
                    for m in group:
                        out[c][m] = results[c][m]
        finally:
            self.window.unpark()
        return out

    @staticmethod
    def _with_first_alone(groups):
        """Micro batch 0 gets a walk to itself so a disagreement stops the run before the long work."""
        rest = [[m for m in group if m != 0] for group in groups]
        return [[0], *[g for g in rest if g]]

    def _reference_logits(self, enc, direction, conds):
        chosen = pick_check_conditions(conds)
        return {c: self.reference(enc, direction, c).cpu() for c in chosen}

    def _verify(self, want, kept):
        worst = 0.0
        for c, expected in want.items():
            got = kept[c, 0]
            if not torch.equal(expected, got):
                differ = int((expected != got).sum())
                worst = float((expected - got).abs().max())
                self.check = {"passed": False, "condition": list(c), "logits_that_differ": differ,
                              "max_abs_diff": worst}
                raise A.AttributionError(
                    f"the wavefront executor gave different numbers from the reference path on the first "
                    f"micro batch (the condition that removes the direction from {describe(c)}: {differ} of "
                    f"{expected.numel()} logits differ, by up to {worst:.3g}). The run stops here rather than "
                    f"carry on with numbers that could differ; rerun with --executor reference.")
        self.check = {"passed": True, "conditions": [list(c) for c in want], "max_abs_diff": worst,
                      "micro_batches": 1}
        self.log(f"  wavefront self check passed: {len(want)} condition(s) of the first micro batch are bitwise "
                 f"equal to the reference path")
