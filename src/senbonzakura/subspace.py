# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Do two sets of directions span the same subspace?

Built for the v0.5 streaming work, where the question is whether directions extracted by streaming
layers through the card match directions extracted with the model resident. It is a general
instrument and the parity gate is its first caller.

WHY NOT PER-VECTOR COSINES, WHICH IS WHAT EVERYONE REACHES FOR FIRST

Two runs can produce the same subspace and disagree completely vector by vector, for two reasons
that have nothing to do with either run being wrong.

**Signs are arbitrary.** A direction and its negation ablate identically. A cosine of -1 between
`u` and `-u` reads as total disagreement and means nothing happened.

**A near-degenerate subspace is an arbitrary rotation.** When two directions carry nearly equal
weight, which one comes out "first" is decided by floating-point noise, so the two runs return the
same plane with its axes swapped or spun. Per-vector comparison calls that a failure. It is not one:
ablating a subspace removes the subspace, and the basis chosen to describe it is bookkeeping.

So the comparison is between the SUBSPACES, through their projectors, which are unique. Measured
on synthetic bases: rotating a basis moves the number by 1.05e-08 and flipping every sign moves it
by exactly 0, while replacing one direction of eight moves it to 0.353.

THE NUMBER

`subspace_distance` is the Frobenius norm of the difference of the two projectors, normalised so 0
means the same subspace and 1 means completely orthogonal ones. `directions_disagreeing` is the
same quantity before normalisation, in units a person can hold: "these two spans differ by about
0.4 of a direction".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: A direction whose norm is below this is not a direction. `dirs_multi` is padded with exact zeros
#: for the slots a layer did not fill, and those must be dropped before anything is orthonormalised
#: or the padding becomes part of the answer.
MIN_DIRECTION_NORM = 1e-6

#: How far apart two subspaces may be and still count as the same one.
#:
#: DERIVED, NOT CHOSEN. Measured on synthetic 8-dimensional spans in 2048 dimensions:
#:
#:     the same subspace, rotated basis        1.05e-08
#:     the same subspace, every sign flipped   0.00e+00
#:     a round trip through float32            2.58e-08
#:     a round trip through float16            2.06e-04
#:     a round trip through bfloat16           1.64e-03   <- the largest noise we actually incur
#:     one direction of eight replaced         3.53e-01   <- the smallest real difference
#:
#: `dirs_multi` is stored in bfloat16, so 1.64e-03 is the noise floor that matters. This sits about
#: six times above it and about thirty-five times below the smallest genuine disagreement, in a gap
#: spanning more than two orders of magnitude. A single wrong direction stays detectable up to
#: roughly ten thousand dimensions, since one bad direction out of K scores about 1/sqrt(K).
SAME_SUBSPACE_TOL = 1e-2


def orthonormalise(directions):
    """An orthonormal basis for the span of `directions`, given as rows.

    Drops padding rows and anything that is already inside the span of what came before, so the
    result's width is the true rank rather than the number of rows handed in. That distinction is
    load-bearing here: a layer that filled three of eight direction slots must compare as a
    three-dimensional subspace, not as an eight-dimensional one with five zero axes.

    Returns a [H, rank] tensor in float64, or a [H, 0] tensor when nothing survives.
    """
    m = torch.as_tensor(directions).double()
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D [rows, H] tensor of directions, got shape {tuple(m.shape)}")
    keep = m.norm(dim=1) > MIN_DIRECTION_NORM
    m = m[keep]
    if m.shape[0] == 0:
        return torch.zeros(int(torch.as_tensor(directions).shape[1]), 0, dtype=torch.float64)
    # Column space of the transpose, so each direction is a column of the basis.
    q, r = torch.linalg.qr(m.T, mode="reduced")
    # QR does not drop dependent columns on its own; a near-zero diagonal in R marks one.
    rank_mask = r.diagonal().abs() > MIN_DIRECTION_NORM
    return q[:, rank_mask]


def principal_angles(a, b):
    """The angles between two subspaces, smallest first, in radians.

    The cosines are the singular values of `aᵀb` for orthonormal `a` and `b`. Clamped into range
    before the arccos, because a singular value can land at 1 + 1e-16 and produce a NaN angle from
    two subspaces that are identical.
    """
    if a.shape[1] == 0 or b.shape[1] == 0:
        return torch.zeros(0, dtype=torch.float64)
    cosines = torch.linalg.svdvals(a.T @ b).clamp(-1.0, 1.0)
    return torch.arccos(cosines)


def directions_disagreeing(a, b):
    """How much of a direction the two spans differ by, in whole directions.

    The squared Frobenius norm of the projector difference is `2 * sum(sin^2 theta)`, so this is
    that sum: 0 when the spans coincide, 1 when one whole direction is unshared, and at most the
    larger of the two ranks. Reported alongside the normalised distance because a reader can hold
    "these differ by 0.4 of a direction" and cannot hold 0.158.
    """
    ka, kb = a.shape[1], b.shape[1]
    if ka == 0 or kb == 0:
        return float(max(ka, kb))
    overlap = float((torch.linalg.svdvals(a.T @ b).clamp(-1.0, 1.0) ** 2).sum())
    # (ka + kb)/2 - overlap, which reduces to sum(sin^2) when the ranks are equal and stays
    # meaningful when they are not: a rank difference is itself a disagreement.
    return max(0.0, (ka + kb) / 2.0 - overlap)


def subspace_distance(a, b):
    """Frobenius distance between the two projectors, normalised into [0, 1].

    0 is the same subspace, 1 is two subspaces sharing nothing. Basis-independent and
    sign-independent by construction, which is the entire reason it is this and not a cosine.
    """
    ka, kb = a.shape[1], b.shape[1]
    if ka == 0 and kb == 0:
        return 0.0        # two empty spans agree, and agree exactly
    return float((2.0 * directions_disagreeing(a, b) / (ka + kb)) ** 0.5)


@dataclass(frozen=True)
class Comparison:
    """The verdict, and enough of the working to argue with it."""

    distance: float
    disagreeing: float
    rank_a: int
    rank_b: int
    max_angle_degrees: float
    tolerance: float

    @property
    def same(self):
        return self.distance <= self.tolerance and self.rank_a == self.rank_b

    def describe(self):
        if self.rank_a != self.rank_b:
            return (f"DIFFERENT: the two sets span {self.rank_a} and {self.rank_b} dimensions. A "
                    f"rank difference is a disagreement no tolerance covers, because one side "
                    f"carries a direction the other has nowhere to put.")
        if self.same:
            return (f"same subspace: distance {self.distance:.2e} within {self.tolerance:.0e}, "
                    f"differing by {self.disagreeing:.2e} of a direction across "
                    f"{self.rank_a} dimensions")
        return (f"DIFFERENT: distance {self.distance:.4f} exceeds {self.tolerance:.0e}. The spans "
                f"differ by {self.disagreeing:.3f} of a direction, and the worst-aligned pair sits "
                f"{self.max_angle_degrees:.1f} degrees apart.")


def compare(directions_a, directions_b, tolerance=SAME_SUBSPACE_TOL):
    """Do these two sets of directions span the same subspace? Rows are directions."""
    a, b = orthonormalise(directions_a), orthonormalise(directions_b)
    angles = principal_angles(a, b)
    return Comparison(
        distance=subspace_distance(a, b),
        disagreeing=directions_disagreeing(a, b),
        rank_a=int(a.shape[1]), rank_b=int(b.shape[1]),
        max_angle_degrees=float(torch.rad2deg(angles.max())) if angles.numel() else 0.0,
        tolerance=float(tolerance),
    )


# ── the self-check, and the control that makes it fail ───────────────────────────────
def _reference_pair(control, seed=0):
    """Two direction sets that should agree, unless `control` names a way to break one.

    The controls are the point. An instrument whose entire output is "these two agree" has to be
    shown saying no, and shown saying it for each shape of disagreement it claims to catch.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(8, 512, generator=g)
    if control is None:
        # The realistic same-subspace case: a different basis for the same span, round-tripped
        # through the dtype `dirs_multi` is actually stored in.
        rotation = torch.linalg.qr(torch.randn(8, 8, generator=g).double())[0]
        return a, (rotation @ a.double()).to(torch.bfloat16).float()
    b = a.clone()
    if control == "replace":
        b[0] = torch.randn(512, generator=g)
    elif control == "drop":
        b = b[:-1]
    elif control == "noise":
        b = b + torch.randn(8, 512, generator=g) * 0.05
    else:
        raise ValueError(f"unknown control {control!r}; known: replace, drop, noise")
    return a, b


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m senbonzakura.subspace",
        description="Check the subspace-comparison instrument on cases whose answer is known.")
    p.add_argument("--control", choices=["replace", "drop", "noise"], default=None,
                   help="break one side on purpose. The instrument must report a difference and "
                        "this command must exit non-zero. An instrument that cannot be made to "
                        "fail has not been shown to work.")
    p.add_argument("--tolerance", type=float, default=SAME_SUBSPACE_TOL)
    a = p.parse_args(argv)

    left, right = _reference_pair(a.control)
    result = compare(left, right, tolerance=a.tolerance)
    print(f"  distance      {result.distance:.6e}   (tolerance {result.tolerance:.0e})")
    print(f"  disagreeing   {result.disagreeing:.6e} directions")
    print(f"  ranks         {result.rank_a} against {result.rank_b}")
    print(f"  worst angle   {result.max_angle_degrees:.4f} degrees")
    print(f"  {result.describe()}")

    if a.control is None:
        if result.same:
            print("\nOK: a rotated, bfloat16 round-tripped copy of a span compares as the same "
                  "subspace.")
            return 0
        print("\nFAILED: two descriptions of ONE subspace were reported as different. The "
              "instrument is too strict to be useful.")
        return 1
    if result.same:
        print(f"\nFAILED: the '{a.control}' control did not move the number. This instrument "
              f"cannot say no, so its agreements mean nothing.")
        return 1
    print(f"\nOK: the '{a.control}' control was caught, and this command exits non-zero to prove "
          f"the check has teeth.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
