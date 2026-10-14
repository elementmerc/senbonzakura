# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Tiny models whose answer is planted, and the answer each one must produce.

WRITTEN BEFORE THE INSTRUMENT EXISTS, ON PURPOSE. These fixtures are the definition of "correct"
for per-block attribution, so the expected status of each is a literal in `EXPECTED` below, fixed
before any code that could disagree with it was written. A fixture written after the code is a
test written by somebody who already knows what the code does, and this project has a documented
case of exactly that: fixtures whose `expect` lines named the checks that had to fire, which made a
gate read as met on evidence that could not support it.

A SECOND DRAFT, RECORDED. The first draft let detectors read the harmful feature from the residual
stream. The instrument then reported `localised` at block 3 on the `redundant` plant and an early
block biased exceeding set on `spread`, and the arithmetic showed why: the extracted direction is a
mean difference, so it also spans the harmful input feature, and removing a block's write along it
also subtracted that feature from the stream that later detectors were reading. Those two plants
did not match their own descriptions ("each block suffices alone"), because block to block
dependence had leaked in. The detectors now read a side channel the stream cannot disturb. The
expected statuses were not touched. Real models DO have block to block dependence through
class informative features, and the instrument measures the total effect including that; these
plants deliberately switch it off so that what they plant is what their names say.

THE PLANTED MODELS. Each is a stack of 12 residual blocks over a 32 wide stream. A block is one of:

    mixer     writes small noise into coordinates nothing reads (so it carries no behaviour)
    detector  reads the "request is harmful" feature from the prompt's own input and writes a
              behaviour signal
    backup    reads the residual, and writes only what is MISSING from the behaviour channel, so
              it silences itself while another block is covering for it
    decoy     writes a huge feature that separates the classes and carries no behaviour at all

and the behaviour readout is the final stream projected onto a fixed pair of axes (the behaviour
channel plus a per-prompt noise channel), so a block carries the behaviour exactly when its writes
reach the behaviour channel.

    FIXTURE        WHAT IS PLANTED                                   WHAT THE INSTRUMENT MUST SAY
    one_block      block 5 writes the behaviour, nobody else does    localised, at {5} and only {5}
    spread         all 12 blocks write one twelfth each              diffuse (every block matters,
                                                                     none matters more)
    redundant      blocks 3 and 11 each suffice alone; 11 only       redundant (single ablations see
                   writes what 3 has not already written             nothing, the pair is everything)
    absent         nothing writes the behaviour                      absent (nothing to attribute)
    orthogonal     the behaviour runs along an axis the extracted    direction_inert (the direction
                   direction is almost perfectly orthogonal to; a    cannot see it; this is NOT
                   decoy feature of 200x the magnitude dominates     "spread out")
                   the class contrast
    bad_readout    one_block, with a readout that does not predict   readout_invalid
                   the generated behaviour
    underpowered   one_block, with 24 measurement prompts per arm    unresolved (below the floor of
                                                                     30 a rate is not reported)

THE DECISION CONVENTIONS THESE EXPECTATIONS ASSUME, stated here so they are fixed before the engine:

  * `absent`: the baseline readout does not separate harmful from harmless prompts, meaning the
    lower end of its AUC interval is not above 0.6.
  * `readout_invalid`: the readout's agreement with the generated behaviour, measured on the search
    partition, is below 0.75.
  * `unresolved`: fewer than 30 measurement prompts in an arm, or the blocks that clear the null are
    not reproduced between random halves of the measurement prompts (median Jaccard below 0.5).
  * `direction_inert`: ablating the direction from EVERY block does not move the readout by more
    than the shuffled label and random direction nulls do.
  * `redundant`: the total effect is real and the single block effects sum to under 0.6 of it, at
    the upper end of their interval.
  * `localised`: at least one block clears the family wise null (the 95th percentile of the
    LARGEST block effect across shuffled label replicates) and the effective number of blocks
    carrying the effect is at most half of them.
  * `diffuse`: the total effect is real and none of the above applies.

These are declared conventions, not measured thresholds, and the artefact records them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

#: Width of the residual stream and number of blocks. Small enough that a whole fixture runs in
#: a fraction of a second on the CPU, which is what lets every test run all of them.
WIDTH = 32
N_BLOCKS = 12
POSITIONS = 2

#: Coordinates of the residual stream. The behaviour readout reads BEH and NOISE_Q only.
BEH, NOISE_Q, IN, DECOY, BEH_B = 0, 1, 2, 3, 4
FIRST_FREE = 5

#: How strongly harmful prompts carry the input feature, against unit noise.
HARM_MEAN = 2.0
#: Per-prompt noise on the readout channel, fixed per prompt so it cancels in a paired comparison.
READOUT_NOISE = 0.15
#: The magnitude of the decoy feature, 200 times a unit.
DECOY_AMPLITUDE = 200.0


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    #: block index -> (kind, amplitude, axis)
    blocks: dict
    read_axis: int
    #: What the readout's agreement with generation is claimed to be.
    agreement: float
    n_measure: int
    why: str
    expected_status: str
    expected_blocks: frozenset = field(default_factory=frozenset)


def _spread(n_blocks, total):
    return dict.fromkeys(range(n_blocks), ("detector", total / n_blocks, BEH))


SPECS = {
    "one_block": FixtureSpec(
        "one_block", {5: ("detector", 3.0, BEH)}, BEH, 0.95, 300,
        "block 5 alone writes the behaviour", "localised", frozenset({5})),
    "spread": FixtureSpec(
        "spread", _spread(N_BLOCKS, 6.0), BEH, 0.95, 300,
        "every block writes an equal share", "diffuse"),
    "redundant": FixtureSpec(
        "redundant", {3: ("detector", 3.0, BEH), 11: ("backup", 3.0, BEH)}, BEH, 0.95, 300,
        "blocks 3 and 11 each suffice; 11 only writes what 3 has not", "redundant"),
    "absent": FixtureSpec(
        "absent", {}, BEH, 0.95, 300, "nothing writes the behaviour", "absent"),
    "orthogonal": FixtureSpec(
        "orthogonal",
        {**dict.fromkeys(range(4, 9), ("decoy", DECOY_AMPLITUDE / 5, DECOY)),
         N_BLOCKS - 1: ("detector", 3.0, BEH_B)},
        BEH_B, 0.95, 300,
        "the behaviour runs along an axis the extracted direction is orthogonal to",
        "direction_inert"),
    "bad_readout": FixtureSpec(
        "bad_readout", {5: ("detector", 3.0, BEH)}, BEH, 0.40, 300,
        "a localised behaviour read through a readout that does not predict generation",
        "readout_invalid"),
    "underpowered": FixtureSpec(
        "underpowered", {5: ("detector", 3.0, BEH)}, BEH, 0.95, 24,
        "a localised behaviour with too few measurement prompts to say so", "unresolved"),
}

#: The expected statuses, as bare literals and not derived from SPECS, so a test can assert that
#: the two agree and a later edit to one cannot silently move the other.
EXPECTED = {
    "one_block": "localised",
    "spread": "diffuse",
    "redundant": "redundant",
    "absent": "absent",
    "orthogonal": "direction_inert",
    "bad_readout": "readout_invalid",
    "underpowered": "unresolved",
}


class ToyBlock(nn.Module):
    """One residual block. Returns a tuple, as a Hugging Face decoder layer does."""

    def __init__(self, kind, amplitude, axis, seed):
        super().__init__()
        self.kind, self.amplitude, self.axis = kind, float(amplitude), int(axis)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("mix", torch.randn(WIDTH - FIRST_FREE, WIDTH - FIRST_FREE,
                                                generator=gen) / math.sqrt(WIDTH))

    def forward(self, h, side):
        write = torch.zeros_like(h)
        signal = side.clamp(0.0, 3.0) / 2.0
        if self.kind == "mixer":
            write[..., FIRST_FREE:] = 0.1 * torch.tanh(h[..., FIRST_FREE:] @ self.mix)
        elif self.kind in ("detector", "decoy"):
            write[..., self.axis] = self.amplitude * signal
        elif self.kind == "backup":
            write[..., self.axis] = (self.amplitude * signal - h[..., self.axis]).clamp(min=0.0)
        else:
            raise ValueError(f"unknown block kind {self.kind!r}")
        return (h + write,)


class ToyStack(nn.Module):
    def __init__(self, spec, seed=0):
        super().__init__()
        layers = []
        for k in range(N_BLOCKS):
            kind, amp, axis = spec.blocks.get(k, ("mixer", 0.0, BEH))
            layers.append(ToyBlock(kind, amp, axis, seed * 1000 + k))
        self.layers = nn.ModuleList(layers)
        read = torch.zeros(WIDTH)
        read[spec.read_axis] = 1.0
        read[NOISE_Q] = 1.0
        self.register_buffer("read", read)

    def forward(self, x, keep_all=False):
        h, seen = x, [x]
        side = x[..., IN]
        for layer in self.layers:
            h = layer(h, side)[0]
            seen.append(h)
        return (h, seen) if keep_all else h


def make_prompts(n_harmful, n_harmless, seed):
    """Prompt tensors `[n, POSITIONS, WIDTH]` for the two arms. The noise is fixed per prompt."""
    gen = torch.Generator().manual_seed(seed)

    def arm(n, harmful):
        x = torch.zeros(n, POSITIONS, WIDTH)
        x[..., FIRST_FREE:] = torch.randn(n, POSITIONS, WIDTH - FIRST_FREE, generator=gen)
        x[..., IN] = torch.randn(n, POSITIONS, generator=gen) + (HARM_MEAN if harmful else 0.0)
        x[..., NOISE_Q] = READOUT_NOISE * torch.randn(n, 1, generator=gen).expand(n, POSITIONS)
        return x
    return arm(n_harmful, True), arm(n_harmless, False)


class ToySubject:
    """A planted model presented to the instrument through the same interface a real one uses."""

    def __init__(self, name, seed=0):
        from . import attribution
        self._attribution = attribution
        self.spec = SPECS[name]
        self.stack = ToyStack(self.spec, seed=seed)
        self.n_blocks = N_BLOCKS
        self._threshold = None
        self._stacked = {}

    def _batch(self, prompts):
        # Stacking 600 small tensors per condition dominated the run; the arms never change.
        key = (id(prompts[0]), len(prompts))
        if key not in self._stacked:
            self._stacked[key] = torch.stack(list(prompts))
        return self._stacked[key]

    def residuals(self, prompts):
        """Last position residual after the input and after every block: `[blocks + 1, n, width]`."""
        with torch.no_grad():
            _, seen = self.stack(self._batch(prompts), keep_all=True)
        return torch.stack([h[:, -1, :] for h in seen])

    def readout(self, prompts, direction=None, blocks=()):
        with torch.no_grad(), self._attribution.ablating(self.stack.layers, direction, blocks):
            return self.stack(self._batch(prompts))[:, -1, :] @ self.stack.read

    def readout_agreement(self, _harmful, _harmless):
        return self.spec.agreement

    def behaviour_rate(self, prompts, direction, blocks):
        """The share of prompts whose readout is past the midpoint of the two unedited arms."""
        if self._threshold is None:
            raise RuntimeError("calibrate the behaviour threshold before asking for a rate")
        return float((self.readout(prompts, direction, blocks) > self._threshold).float().mean())

    def calibrate(self, harmful, harmless):
        self._threshold = float((self.readout(harmful).mean() + self.readout(harmless).mean()) / 2)


def partitions(name, seed=0):
    """Fit, search and measure partitions for a fixture. Disjoint draws from one distribution."""
    from .attribution import arm
    spec = SPECS[name]
    sizes = {"fit": 200, "search": 200, "measure": spec.n_measure}
    out = {}
    for i, (part, n) in enumerate(sizes.items()):
        harmful, harmless = make_prompts(n, n, seed=seed * 100 + 11 + i)
        out[part] = {"harmful": arm(list(harmful)), "harmless": arm(list(harmless))}
    return out
