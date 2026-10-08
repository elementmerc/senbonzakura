#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Compare two direction sets as SUBSPACES rather than as lists of vectors.

WHY VECTOR-BY-VECTOR COMPARISON IS THE WRONG INSTRUMENT, which is the whole reason this exists.

Directions come out of a singular value decomposition, and an SVD fixes a subspace rather than a
basis for it. Two consequences, both of which make a per-vector cosine lie:

- **A singular vector's sign is arbitrary.** Flip one and you have the same decomposition. The
  cosine between a vector and its negation is -1, which reads as "completely different" and means
  "identical".
- **A near-degenerate subspace has an arbitrary rotation inside it.** When two singular values are
  close, which basis the solver hands back for that pair depends on arithmetic order, not on the
  model. Two numerically identical implementations can therefore return per-vector cosines near
  zero while spanning exactly the same space.

So the question "did the streaming path find the same directions" has to be asked about the SPAN.
Two measures are reported, because they fail differently:

PRINCIPAL ANGLES. Take the two K-dimensional spans and ask: what is the smallest angle between
any line in the first and the whole of the second? Then the smallest angle not already accounted
for, and so on, K of them. Physical picture: two flat sheets of card held in a room. If they lie
on top of each other every angle is zero. Tilt one and the angle between them is the first
principal angle. The LARGEST of the K angles is the honest headline, because it is the worst
direction, and a mean would hide one badly rotated axis among several good ones.

THE PROJECTOR DISTANCE. Each span has one projector, the matrix that flattens any arrow onto that
span. Two spans are the same exactly when their projectors are the same, and a projector has no
basis freedom at all: it does not care about signs or rotations. The Frobenius norm of the
difference is therefore a single number with no arbitrary content in it. It is bounded by
`sqrt(2K)`, and dividing by that puts it on 0 to 1 so a figure is comparable across K.

THE THRESHOLD IS STATED BEFORE THE RUN, not read off the result. `DEFAULT_TOLERANCE` is the
number this instrument was built with, and a caller that needs a different one passes it and says
why. Reading a threshold off the figure you just measured is how a parity check becomes a
rubber stamp.

WHAT THIS IS NOT. It says two sets span the same space. It does not say either span is the right
one, and it does not say the edits they produce behave alike; two identical subspaces applied at
different strengths give different models. Behaviour is a separate measurement and this does not
substitute for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Largest principal angle, in radians, at which two sets are called the same span.
#:
#: STATED HERE AND NOT DERIVED FROM A RUN. 1e-3 rad is about 0.057 degrees: far looser than the
#: float32 noise two orderings of the same arithmetic produce (which lands near 1e-6), and far
#: tighter than any real disagreement about which space the refusal direction lives in. The gap
#: between those two scales is wide enough that the exact choice inside it does not matter, which
#: is the property a threshold wants.
DEFAULT_TOLERANCE = 1e-3

#: Below this, a row is treated as absent rather than as a direction.
#:
#: The same figure the abliterator uses to count populated directions, because the two have to
#: agree: `dirs_multi` pads unused slots with zeros, and a zero row is not a basis vector. If this
#: were looser than the abliterator's, a padded slot would be read as a direction and the span
#: would gain an axis made of nothing.
ZERO_NORM = 1e-6


class SubspaceError(ValueError):
    """A comparison this instrument refuses to make, with the reason."""


def _orthonormal_basis(block, *, where):
    """An orthonormal basis for the span of `block`'s non-zero rows, as columns.

    Through a QR of the non-zero rows rather than by trusting them to be orthonormal already.
    They are supposed to be, and `_assert_basis_orthonormal` checks it at extraction time, but an
    instrument that assumes its input is well formed cannot tell a real disagreement from a
    malformed file; the whole point of this one is to be believed about the second case.
    """
    rows = np.asarray(block, dtype=np.float64)
    if rows.ndim != 2:
        raise SubspaceError(f"{where}: expected [K, hidden] and got shape {rows.shape}.")
    keep = rows[np.linalg.norm(rows, axis=1) > ZERO_NORM]
    if keep.size == 0:
        return np.zeros((rows.shape[1], 0))
    q, _r = np.linalg.qr(keep.T)
    # QR returns as many columns as the input had, and a rank-deficient input makes some of them
    # arbitrary. Keeping only the rank is what stops a duplicated direction inventing an axis.
    rank = int(np.linalg.matrix_rank(keep.T))
    return q[:, :rank]


def principal_angles(a, b, *, where="positions"):
    """The principal angles in radians between two spans, smallest first.

    Via the singular values of `A.T @ B`, which are the cosines of the angles. Clipped into
    [-1, 1] before `arccos` because a cosine of 1.0000000000000002 is a float32 artefact and not
    a complex angle, and `arccos` of it returns nan, which would propagate silently into a
    comparison that then reads as a pass.
    """
    qa = _orthonormal_basis(a, where=f"{where} (first set)")
    qb = _orthonormal_basis(b, where=f"{where} (second set)")
    if qa.shape[1] == 0 and qb.shape[1] == 0:
        return np.zeros(0)
    if qa.shape[1] != qb.shape[1]:
        raise SubspaceError(
            f"{where}: the two sets span {qa.shape[1]} and {qb.shape[1]} dimensions. Principal "
            f"angles are defined between spans of equal dimension, and a difference in rank is "
            f"already the finding: one side populated a different number of directions.")
    if qa.shape[0] != qb.shape[0]:
        raise SubspaceError(
            f"{where}: hidden sizes differ, {qa.shape[0]} against {qb.shape[0]}. These are not "
            f"two measurements of the same model.")
    cos = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(cos, -1.0, 1.0))[::-1].copy()


def projector_distance(a, b, *, where="positions"):
    """`||Pa - Pb||_F / sqrt(2K)`: zero for the same span, one for fully orthogonal spans.

    Normalised so the figure means the same thing at every K. The raw Frobenius norm of the
    difference of two rank-K projectors is at most `sqrt(2K)`, so an unnormalised number looks
    worse at larger K for no reason at all, which is exactly the kind of figure somebody compares
    across two runs and draws a conclusion from.
    """
    qa = _orthonormal_basis(a, where=f"{where} (first set)")
    qb = _orthonormal_basis(b, where=f"{where} (second set)")
    if qa.shape[1] == 0 and qb.shape[1] == 0:
        return 0.0
    if qa.shape[0] != qb.shape[0]:
        raise SubspaceError(
            f"{where}: hidden sizes differ, {qa.shape[0]} against {qb.shape[0]}.")
    pa, pb = qa @ qa.T, qb @ qb.T
    k = max(qa.shape[1], qb.shape[1])
    return float(np.linalg.norm(pa - pb, "fro") / np.sqrt(2.0 * k))


def compare(first, second, *, tolerance=DEFAULT_TOLERANCE):
    """Per-position subspace agreement between two `[positions, K, hidden]` direction sets.

    Returns a dict carrying the verdict, the worst position and every position's figures, so a
    failure names WHERE rather than only that something disagreed. A parity check that says "no"
    without saying which layer sends somebody back to bisect by hand.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3:
        raise SubspaceError(
            f"a direction set is [positions, K, hidden]; got {a.shape} and {b.shape}.")
    if a.shape[0] != b.shape[0]:
        raise SubspaceError(
            f"the two sets cover {a.shape[0]} and {b.shape[0]} residual-stream positions, so "
            f"they are not two measurements of the same stack.")
    if tolerance <= 0:
        raise SubspaceError(
            f"tolerance={tolerance} admits nothing: exact equality is not reachable in floating "
            f"point and a threshold of zero makes every comparison fail. State a positive one.")

    positions = []
    for i in range(a.shape[0]):
        angles = principal_angles(a[i], b[i], where=f"position {i}")
        worst = float(angles.max()) if angles.size else 0.0
        positions.append({
            "position": i,
            "rank": int(_orthonormal_basis(a[i], where=f"position {i}").shape[1]),
            "largest_angle_rad": worst,
            "largest_angle_deg": float(np.degrees(worst)),
            "projector_distance": projector_distance(a[i], b[i], where=f"position {i}"),
            "agrees": worst <= tolerance,
        })

    disagreeing = [p for p in positions if not p["agrees"]]
    worst = max(positions, key=lambda p: p["largest_angle_rad"]) if positions else None
    return {
        "tolerance_rad": tolerance,
        "positions_compared": len(positions),
        "positions_disagreeing": len(disagreeing),
        "agrees": not disagreeing,
        "worst_position": worst,
        "per_position": positions,
    }


def report(result, *, log=print):
    log(f"subspace parity over {result['positions_compared']} position(s), "
        f"tolerance {result['tolerance_rad']:.1e} rad "
        f"({np.degrees(result['tolerance_rad']):.4f} deg)")
    worst = result["worst_position"]
    if worst is not None:
        log(f"  worst: position {worst['position']}, largest principal angle "
            f"{worst['largest_angle_rad']:.3e} rad ({worst['largest_angle_deg']:.4f} deg), "
            f"projector distance {worst['projector_distance']:.3e}")
    if result["agrees"]:
        log("  SAME SPAN at every position.")
        log("  That is a statement about the space, not about behaviour: two identical subspaces "
            "applied at different strengths give different models.")
    else:
        log(f"  DIFFERENT: {result['positions_disagreeing']} position(s) outside the tolerance.")
        for p in result["per_position"]:
            if not p["agrees"]:
                log(f"    position {p['position']}: {p['largest_angle_deg']:.4f} deg, "
                    f"projector {p['projector_distance']:.3e}")
    return result


def _load(path):
    """A direction set from a file `save_directions` wrote."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from senbonzakura import streambake

    arr, meta = streambake.load_directions(path)
    return np.asarray(arr), meta


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="subspace_parity.py",
        description="Compare two direction sets as subspaces, which is the only fair comparison: "
                    "singular-vector signs are arbitrary and a near-degenerate subspace is an "
                    "arbitrary rotation, so per-vector cosines can read as total disagreement "
                    "between two identical implementations.")
    p.add_argument("first", help="a directions file")
    p.add_argument("second", help="the other directions file")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"largest principal angle in radians at which the spans are called the "
                        f"same (default: {DEFAULT_TOLERANCE:g}, about 0.057 degrees)")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="write the full per-position result to standard output as JSON")
    a = p.parse_args(argv)

    try:
        first, meta_a = _load(a.first)
        second, meta_b = _load(a.second)
        result = compare(first, second, tolerance=a.tolerance)
    except Exception as e:
        raise SystemExit(f"subspace parity: {type(e).__name__}: {e}") from e

    if meta_a.get("model") != meta_b.get("model"):
        print(f"NOTE: the two files name different models ({meta_a.get('model')!r} and "
              f"{meta_b.get('model')!r}), so a disagreement here may be the models rather than "
              f"the implementations.", file=sys.stderr)
    report(result, log=lambda m: print(m, file=sys.stderr))
    if a.as_json:
        json.dump(result, sys.stdout, indent=2)
        print()
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
