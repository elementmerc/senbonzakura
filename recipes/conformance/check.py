#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Check an edited checkpoint against the conformance fixture's expected output.

Any tool that claims to implement the abliteration recipe format can prove it with this script.
It needs no senbonzakura code: it reads two safetensors directories and compares them.

    python check.py ACTUAL_DIR EXPECTED_DIR

ACTUAL_DIR is the output of your tool, applied to `recipes/conformance/` with `recipe.json`.
EXPECTED_DIR is `recipes/conformance/expected/`.

The check passes when every tensor has the same name, dtype and shape, and:

  - float32 tensors agree to within 1e-5 in absolute value (float32 rounding in a different
    summation order is expected; anything larger is a different edit), and
  - float16 and bfloat16 tensors agree to within one unit in the last place of each value.

Exit status is 0 on a pass and 1 on a fail. The report names the first defect, and every tensor
that failed, so a failure is something a person can act on.
"""

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import apply_recipe as ar  # noqa: E402  (the header and decoding code lives beside this file)

FLOAT32_TOL = 1e-5


def read_tensors(directory):
    """`{name: (dtype, float32 array)}` for every tensor in the directory's model.safetensors."""
    path = pathlib.Path(directory) / "model.safetensors"
    header, data_start, _size = ar.read_safetensors_header(path)
    out = {}
    with path.open("rb") as f:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            raw = ar.read_tensor_bytes(f, data_start, entry)
            out[name] = (entry["dtype"], ar.decode(raw, entry["dtype"], entry["shape"]), entry["shape"])
    return out


def compare(actual, expected):
    """A list of human-readable defects. Empty means the two checkpoints agree."""
    problems = []
    if actual.keys() != expected.keys():
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        problems.append(f"tensor names differ: missing {missing}, unexpected {extra}")
        return problems
    for name in sorted(expected):
        a_dtype, a, a_shape = actual[name]
        e_dtype, e, e_shape = expected[name]
        if a_dtype != e_dtype or a_shape != e_shape:
            problems.append(f"{name}: dtype and shape are {a_dtype} {a_shape}, expected {e_dtype} {e_shape}")
            continue
        diff = np.abs(a.astype(np.float64) - e.astype(np.float64))
        if e_dtype == "F32":
            worst = float(diff.max()) if diff.size else 0.0
            if worst > FLOAT32_TOL:
                problems.append(f"{name}: differs by {worst:.3g}, tolerance is {FLOAT32_TOL:g}")
        else:
            # One unit in the last place is 2^-7 of the value for bfloat16 (7 stored mantissa
            # bits) and 2^-10 for float16 (10 stored mantissa bits).
            relative = 2.0**-7 if e_dtype == "BF16" else 2.0**-10
            ulp = np.maximum(np.abs(e.astype(np.float64)), 2.0**-24) * relative
            bad = int((diff > ulp).sum())
            if bad:
                problems.append(f"{name}: {bad} element(s) differ by more than one {e_dtype} unit in the last place")
    return problems


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    actual_dir, expected_dir = argv[1], argv[2]
    try:
        actual = read_tensors(actual_dir)
        expected = read_tensors(expected_dir)
    except (OSError, ar.RecipeError) as e:
        print(f"FAIL: could not read a checkpoint: {e}")
        return 1
    problems = compare(actual, expected)
    if problems:
        print(f"FAIL: {len(problems)} defect(s). First: {problems[0]}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"PASS: all {len(expected)} tensor(s) agree with the expected output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
