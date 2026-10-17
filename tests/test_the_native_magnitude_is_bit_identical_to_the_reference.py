# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The compiled magnitude step must give the same bits as the pure Python one it replaced.

WHY BITS AND NOT A TOLERANCE. A diff report is meant to be re-derivable: two runs over the same two
tensors must produce the same figure, byte for byte, on any machine. A tolerance would hide exactly
the failure that matters, a reordered sum or a fused multiply-add, which moves the last bit and
nothing else. So every returned number is compared through `float.hex`, which distinguishes -0.0
from 0.0 and every finite value from its neighbour.

THE REFERENCE is `modeldiff._magnitude_reference`, the old Python loop with its arithmetic unchanged.
Nothing in production calls it. `modeldiff._magnitude` is the path that does, and it calls the Rust
extension. These tests hold the two together over every dtype, at the chunk boundaries, at the edge
values, over every 16-bit pattern, and over raw bytes that include the refusals.
"""
import math
import random
import struct
import sys
from itertools import product

import pytest

from senbonzakura import _native, modeldiff

DTYPES = ("F64", "F32", "F16", "BF16")
WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def _bits(result):
    """Every returned number as its exact hex form, and the refusal text unchanged."""
    rel, peak, why, sums = result

    def hexed(x):
        return None if x is None else float.hex(x)

    return (hexed(rel), hexed(peak), why, None if sums is None else tuple(hexed(s) for s in sums))


def _assert_same(a, b, dtype):
    native = modeldiff._magnitude(a, b, dtype)
    reference = modeldiff._magnitude_reference(a, b, dtype)
    assert _bits(native) == _bits(reference), (
        f"{dtype}: the extension gave {native!r} and the reference gave {reference!r}")
    return reference


def _bf16_word(value):
    """The top sixteen bits of an IEEE single, which is what BF16 is, truncated not rounded."""
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _pack(dtype, values):
    if dtype == "F64":
        return struct.pack(f"<{len(values)}d", *values)
    if dtype == "F32":
        return struct.pack(f"<{len(values)}f", *values)
    if dtype == "F16":
        return struct.pack(f"<{len(values)}e", *values)
    return struct.pack(f"<{len(values)}H", *(_bf16_word(v) for v in values))


def _finite_values(dtype, rng, n):
    """Ordinary finite values spread over several orders of magnitude, packed in `dtype`.

    Kept to a thousand in magnitude so the F16 pack never overflows: this is the success path, and
    the failure path is what the raw-bytes test is for.
    """
    values = [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-6, 3) for _ in range(n)]
    return _pack(dtype, values)


# Each dtype's edge values, chosen to sit on its boundaries. The F16 and BF16 lists cannot use the
# F32 maximum: struct refuses to pack it into a half, and BF16 has a different largest finite value.
EDGES = {
    "F64": [0.0, -0.0, 1.0, -1.0, 5e-324, -5e-324, sys.float_info.max, -sys.float_info.max,
            sys.float_info.min, 0.1, math.inf, -math.inf, math.nan],
    "F32": [0.0, -0.0, 1.0, -1.0, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126, 3.4028234663852886e38,
            -3.4028234663852886e38, 0.1, math.inf, -math.inf, math.nan],
    "F16": [0.0, -0.0, 1.0, -1.0, 2.0 ** -24, 2.0 ** -14, 65504.0, -65504.0, 0.1, math.inf,
            -math.inf, math.nan],
    "BF16": [0.0, -0.0, 1.0, -1.0, 2.0 ** -133, 2.0 ** -126, 3.3895313892515355e38,
             -3.3895313892515355e38, 0.1, math.inf, -math.inf, math.nan],
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 4_095, 4_096, 4_097, 8_205])
def test_ordinary_values_agree_bit_for_bit_at_every_boundary(dtype, n):
    rng = random.Random(f"ordinary-{dtype}-{n}")
    result = _assert_same(_finite_values(dtype, rng, n), _finite_values(dtype, rng, n), dtype)
    if n:
        assert result[2] is None, f"an ordinary {dtype} pair was refused: {result[2]}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_bit_patterns_agree_including_the_refusals(dtype, seed):
    """Random bytes hold NaNs, infinities and overflowing squares, so this drives the refusals."""
    rng = random.Random(f"raw-{dtype}-{seed}")
    n = rng.choice([1, 2, 5, 4_096, 4_097, 9_000])
    width = WIDTH[dtype]
    _assert_same(rng.randbytes(n * width), rng.randbytes(n * width), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_edge_value_against_every_other_agrees(dtype):
    values = EDGES[dtype]
    for x, y in product(values, repeat=2):
        _assert_same(_pack(dtype, [x]), _pack(dtype, [y]), dtype)
    xs = [x for x in values for _ in values]
    ys = [y for _ in values for y in values]
    _assert_same(_pack(dtype, xs), _pack(dtype, ys), dtype)


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_every_sixteen_bit_pattern_agrees_as_both_candidate_and_base(dtype):
    """All 65,536 patterns, each paired with another, so every value is walked on both sides.

    7919 is prime and odd, so `i * 7919 + 13` permutes the patterns: every pattern appears once as
    a candidate value and once as a base value.
    """
    for i in range(65_536):
        j = (i * 7_919 + 13) % 65_536
        _assert_same(i.to_bytes(2, "little"), j.to_bytes(2, "little"), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunk", [1, 3, 4_096, 10_000])
def test_the_chunk_size_changes_nothing(dtype, chunk, monkeypatch):
    rng = random.Random(f"chunk-{dtype}")
    a = _finite_values(dtype, rng, 5_000)
    b = _finite_values(dtype, rng, 5_000)
    expected = _bits(modeldiff._magnitude_reference(a, b, dtype))
    monkeypatch.setattr(modeldiff, "CHUNK_ELEMENTS", chunk)
    assert _bits(modeldiff._magnitude(a, b, dtype)) == expected, f"chunk {chunk} changed the bits"


def test_the_production_path_is_the_extension(monkeypatch):
    """A test that the reference is compared against is no use if production never reaches Rust."""
    calls = []
    real = _native.diff_accumulate

    def spy(*args):
        calls.append(args[2])
        return real(*args)

    monkeypatch.setattr(modeldiff._native, "diff_accumulate", spy)
    modeldiff._magnitude(_pack("F32", [1.0, 2.0]), _pack("F32", [3.0, 4.0]), "F32")
    assert calls == ["F32"]


@pytest.mark.parametrize("fn", [modeldiff._magnitude, modeldiff._magnitude_reference])
def test_both_paths_refuse_a_length_that_is_not_whole_elements(fn):
    rel, peak, why, sums = fn(b"\0" * 3, b"\0" * 3, "F32")
    assert (rel, peak, sums) == (None, None, None)
    assert "not a whole number of F32 elements" in why, why


@pytest.mark.parametrize("fn", [modeldiff._magnitude, modeldiff._magnitude_reference])
def test_both_paths_refuse_mismatched_lengths_with_the_same_words(fn):
    _rel, _peak, why, _sums = fn(b"\0" * 8, b"\0" * 4, "F32")
    assert "8 and 4 bytes" in why, why


def test_the_extension_refuses_bad_input_with_a_value_error_and_never_a_panic():
    zero = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="not a whole number of 4-byte elements"):
        _native.diff_accumulate(b"\0" * 3, b"\0" * 3, "F32", zero)
    with pytest.raises(ValueError, match="holds 8 bytes and the base holds 4"):
        _native.diff_accumulate(b"\0" * 8, b"\0" * 4, "F32", zero)
    with pytest.raises(ValueError, match="cannot decode I64"):
        _native.diff_accumulate(b"\0" * 8, b"\0" * 8, "I64", zero)
    with pytest.raises(ValueError, match="not a whole number of 2-byte elements"):
        _native.diff_has_non_finite(b"\0" * 5, "BF16")
    with pytest.raises(ValueError, match="cannot decode I8"):
        _native.diff_has_non_finite(b"", "I8")


def test_an_empty_pair_is_the_same_empty_answer_on_both_paths():
    assert _bits(modeldiff._magnitude(b"", b"", "F16")) == \
        _bits(modeldiff._magnitude_reference(b"", b"", "F16"))
    assert _native.diff_has_non_finite(b"", "F64") is False
