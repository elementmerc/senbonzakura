// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

//! The element arithmetic behind the magnitude step of `senbonzakura diff`.
//!
//! WHAT THIS IS FOR. `diff` measures how far an edit moved a tensor: the sum of squared differences,
//! the sum of squared base values, and the largest single move. Over a model that is hundreds of
//! millions of elements, and the Python loop that did it spent its time on interpreter overhead per
//! element rather than on the arithmetic. This is a hot walk over a large file, which is what the
//! crate root says Rust is for.
//!
//! THE CONTRACT IS BIT-FOR-BIT, NOT APPROXIMATE. A report is meant to be re-derivable, so the same
//! two tensors must give the same bits every time and on every machine. The Python loop this
//! replaced is kept in `modeldiff.py` as the reference, and the tests compare the two with
//! `float.hex`. That holds only if three things hold here:
//!
//!   1. Every stored value is widened to f64 exactly, as CPython's `struct` module does it. Each of
//!      the four widths is a lossless conversion, so there is no rounding to reproduce.
//!   2. Elements are visited in order, one at a time, with the same four operations in the same
//!      sequence: `d = x - y`, `num += d * d`, `base += y * y`, `peak = max(peak, |d|)`. There is no
//!      fused multiply-add (Rust does not contract by default), no SIMD lane sums and no parallel
//!      reduction. Each of those changes the rounding, and the change is silent in the output.
//!   3. `max` keeps the earlier value unless the new one is strictly greater, which is Python's rule
//!      too. A NaN therefore never becomes the peak. Non-finite values are caught from the sums by
//!      the caller, and `any_non_finite` answers which side held one.
//!
//! Chunking changes nothing here, because the state is threaded from one call to the next and the
//! order of operations is unchanged. The caller picks the chunk size for memory and heartbeats.
//!
//! WHAT IS REFUSED. An unknown dtype, two buffers of different lengths, and a length that is not a
//! whole number of elements. Each is an error value that the Python boundary turns into a plain
//! message. Nothing here panics: every fallible step returns a `Result`.

/// The dtypes this decodes, named as safetensors names them. Anything else is refused by the
/// caller, which keeps the identity verdict exact and reports the magnitude as unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
}

impl Dtype {
    pub fn parse(name: &str) -> Option<Dtype> {
        match name {
            "F64" => Some(Dtype::F64),
            "F32" => Some(Dtype::F32),
            "F16" => Some(Dtype::F16),
            "BF16" => Some(Dtype::BF16),
            _ => None,
        }
    }

    pub fn width(self) -> usize {
        match self {
            Dtype::F64 => 8,
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

/// Why a pair of buffers cannot be walked. Plain facts, so the caller can say them to a reader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffError {
    LengthMismatch { candidate: usize, base: usize },
    NotWholeElements { bytes: usize, width: usize },
}

impl DiffError {
    pub fn message(&self) -> String {
        match self {
            DiffError::LengthMismatch { candidate, base } => format!(
                "the candidate holds {candidate} bytes and the base holds {base}, so there is no \
                 element-by-element comparison to make"
            ),
            DiffError::NotWholeElements { bytes, width } => format!(
                "a buffer holds {bytes} bytes, which is not a whole number of {width}-byte elements, \
                 so its header disagrees with its data"
            ),
        }
    }
}

/// The three running values a walk carries from one chunk to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sums {
    pub num: f64,
    pub base: f64,
    pub peak: f64,
}

fn check_whole(bytes: usize, width: usize) -> Result<(), DiffError> {
    if bytes % width == 0 {
        Ok(())
    } else {
        Err(DiffError::NotWholeElements { bytes, width })
    }
}

fn element<const W: usize>(chunk: &[u8]) -> Result<[u8; W], DiffError> {
    <[u8; W]>::try_from(chunk).map_err(|_| DiffError::NotWholeElements {
        bytes: chunk.len(),
        width: W,
    })
}

fn accumulate_with<const W: usize, F>(
    candidate: &[u8],
    base: &[u8],
    start: Sums,
    decode: F,
) -> Result<Sums, DiffError>
where
    F: Fn([u8; W]) -> f64,
{
    if candidate.len() != base.len() {
        return Err(DiffError::LengthMismatch {
            candidate: candidate.len(),
            base: base.len(),
        });
    }
    check_whole(candidate.len(), W)?;
    let mut num = start.num;
    let mut sum_base = start.base;
    let mut peak = start.peak;
    for (cx, cy) in candidate.chunks_exact(W).zip(base.chunks_exact(W)) {
        let x = decode(element::<W>(cx)?);
        let y = decode(element::<W>(cy)?);
        let d = x - y;
        num += d * d;
        sum_base += y * y;
        let moved = d.abs();
        if moved > peak {
            peak = moved;
        }
    }
    Ok(Sums {
        num,
        base: sum_base,
        peak,
    })
}

/// Folds one pair of equal-length buffers onto `start`, element by element, in order.
///
/// `candidate` is `x` and `base` is `y`, which is the orientation the Python caller uses, so the
/// sign of every difference matches the reference exactly.
pub fn accumulate(
    candidate: &[u8],
    base: &[u8],
    dtype: Dtype,
    start: Sums,
) -> Result<Sums, DiffError> {
    match dtype {
        Dtype::F64 => accumulate_with::<8, _>(candidate, base, start, f64::from_le_bytes),
        Dtype::F32 => accumulate_with::<4, _>(candidate, base, start, |b| {
            f64::from(f32::from_le_bytes(b))
        }),
        Dtype::F16 => accumulate_with::<2, _>(candidate, base, start, |b| {
            decode_f16(u16::from_le_bytes(b))
        }),
        Dtype::BF16 => accumulate_with::<2, _>(candidate, base, start, |b| {
            decode_bf16(u16::from_le_bytes(b))
        }),
    }
}

fn non_finite_with<const W: usize, F>(raw: &[u8], decode: F) -> Result<bool, DiffError>
where
    F: Fn([u8; W]) -> f64,
{
    check_whole(raw.len(), W)?;
    for chunk in raw.chunks_exact(W) {
        if !decode(element::<W>(chunk)?).is_finite() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// True when any element of `raw` is a NaN or an infinity.
pub fn any_non_finite(raw: &[u8], dtype: Dtype) -> Result<bool, DiffError> {
    match dtype {
        Dtype::F64 => non_finite_with::<8, _>(raw, f64::from_le_bytes),
        Dtype::F32 => non_finite_with::<4, _>(raw, |b| f64::from(f32::from_le_bytes(b))),
        Dtype::F16 => non_finite_with::<2, _>(raw, |b| decode_f16(u16::from_le_bytes(b))),
        Dtype::BF16 => non_finite_with::<2, _>(raw, |b| decode_bf16(u16::from_le_bytes(b))),
    }
}

/// BF16 is the top sixteen bits of an IEEE single, so widening is a shift and then a lossless
/// conversion. Exact, which is why it can be compared bit for bit with the struct-based reference.
pub fn decode_bf16(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// IEEE 754 binary16 widened to f64. Exact for every bit pattern: every half-precision value is
/// representable in double precision, so the only work is placing the fields. Subnormals are
/// `mantissa * 2^-24` and normals are `(1024 + mantissa) * 2^(exponent - 25)`, both exact because
/// every factor is a small power of two. A NaN keeps no payload, which the caller never inspects.
pub fn decode_f16(bits: u16) -> f64 {
    let negative = (bits & 0x8000) != 0;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let magnitude = match exponent {
        0 => f64::from(mantissa) * 2f64.powi(-24),
        0x1f if mantissa == 0 => f64::INFINITY,
        0x1f => f64::NAN,
        e => f64::from(0x0400 | mantissa) * 2f64.powi(i32::from(e) - 25),
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: Sums = Sums {
        num: 0.0,
        base: 0.0,
        peak: 0.0,
    };

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn the_four_dtypes_parse_and_the_rest_do_not() {
        assert_eq!(Dtype::parse("F64"), Some(Dtype::F64));
        assert_eq!(Dtype::parse("F32"), Some(Dtype::F32));
        assert_eq!(Dtype::parse("F16"), Some(Dtype::F16));
        assert_eq!(Dtype::parse("BF16"), Some(Dtype::BF16));
        assert_eq!(Dtype::parse("I64"), None);
        assert_eq!(Dtype::parse("f32"), None, "dtype names are case sensitive, as safetensors is");
    }

    #[test]
    fn widths_match_the_dtype_names() {
        assert_eq!(Dtype::F64.width(), 8);
        assert_eq!(Dtype::F32.width(), 4);
        assert_eq!(Dtype::F16.width(), 2);
        assert_eq!(Dtype::BF16.width(), 2);
    }

    #[test]
    fn bf16_widens_known_values_exactly() {
        assert_eq!(decode_bf16(0x3f80), 1.0);
        assert_eq!(decode_bf16(0xc000), -2.0);
        assert_eq!(decode_bf16(0x0000), 0.0);
        assert!(decode_bf16(0x8000).is_sign_negative(), "-0.0 must keep its sign");
        assert_eq!(decode_bf16(0x7f80), f64::INFINITY);
        assert_eq!(decode_bf16(0xff80), f64::NEG_INFINITY);
        assert!(decode_bf16(0x7fc0).is_nan());
    }

    #[test]
    fn f16_widens_known_values_exactly() {
        assert_eq!(decode_f16(0x3c00), 1.0);
        assert_eq!(decode_f16(0xc000), -2.0);
        assert_eq!(decode_f16(0x7bff), 65504.0, "the largest finite half");
        assert_eq!(decode_f16(0x0400), 2f64.powi(-14), "the smallest normal half");
        assert_eq!(decode_f16(0x0001), 2f64.powi(-24), "the smallest subnormal half");
        assert_eq!(decode_f16(0x03ff), 1023.0 * 2f64.powi(-24), "the largest subnormal half");
        assert!(decode_f16(0x8000).is_sign_negative(), "-0.0 must keep its sign");
        assert_eq!(decode_f16(0x7c00), f64::INFINITY);
        assert_eq!(decode_f16(0xfc00), f64::NEG_INFINITY);
        assert!(decode_f16(0x7e00).is_nan());
    }

    #[test]
    fn every_half_pattern_is_finite_or_declared_non_finite_by_its_exponent() {
        for bits in 0..=u16::MAX {
            let v = decode_f16(bits);
            let exponent = (bits >> 10) & 0x1f;
            assert_eq!(v.is_finite(), exponent != 0x1f, "pattern {bits:#06x}");
            if v.is_finite() {
                assert!(v.abs() <= 65504.0, "pattern {bits:#06x} widened out of range");
                assert_eq!(v.is_sign_negative(), (bits & 0x8000) != 0, "pattern {bits:#06x}");
            }
        }
    }

    #[test]
    fn a_hand_worked_pair_gives_the_hand_worked_sums() {
        // d = [0.5, -2.0]; num = 0.25 + 4.0; base = 0.25 + 16.0; peak = 2.0. All exact in binary.
        let candidate = f32_bytes(&[1.0, 2.0]);
        let base = f32_bytes(&[0.5, 4.0]);
        let s = accumulate(&candidate, &base, Dtype::F32, ZERO).expect("valid input");
        assert_eq!(s, Sums { num: 4.25, base: 16.25, peak: 2.0 });
    }

    #[test]
    fn splitting_the_walk_gives_the_same_bits_as_one_walk() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let xs: Vec<f32> = (0..1000).map(|_| next() * 3.0).collect();
        let ys: Vec<f32> = (0..1000).map(|_| next() * 3.0).collect();
        let (cx, by) = (f32_bytes(&xs), f32_bytes(&ys));

        let whole = accumulate(&cx, &by, Dtype::F32, ZERO).expect("valid input");
        let mut state = ZERO;
        for start in (0..1000).step_by(7) {
            let end = (start + 7).min(1000);
            state = accumulate(
                &cx[start * 4..end * 4],
                &by[start * 4..end * 4],
                Dtype::F32,
                state,
            )
            .expect("valid input");
        }
        assert_eq!(state.num.to_bits(), whole.num.to_bits());
        assert_eq!(state.base.to_bits(), whole.base.to_bits());
        assert_eq!(state.peak.to_bits(), whole.peak.to_bits());
    }

    #[test]
    fn a_nan_difference_never_becomes_the_peak() {
        let candidate = f32_bytes(&[f32::NAN, 1.0]);
        let base = f32_bytes(&[0.0, 0.0]);
        let s = accumulate(&candidate, &base, Dtype::F32, ZERO).expect("valid input");
        assert!(s.num.is_nan(), "the NaN must reach the sums so the caller refuses");
        assert_eq!(s.peak, 1.0, "the NaN must not displace the peak");
    }

    #[test]
    fn a_negative_zero_difference_keeps_a_positive_zero_peak() {
        let candidate = f32_bytes(&[-0.0]);
        let base = f32_bytes(&[0.0]);
        let s = accumulate(&candidate, &base, Dtype::F32, ZERO).expect("valid input");
        assert_eq!(s.peak, 0.0);
        assert!(s.peak.is_sign_positive());
    }

    #[test]
    fn the_peak_is_the_largest_magnitude_whichever_way_the_move_went() {
        let candidate = f32_bytes(&[1.0, -5.0, 2.0]);
        let base = f32_bytes(&[2.0, 1.0, 2.0]);
        let s = accumulate(&candidate, &base, Dtype::F32, ZERO).expect("valid input");
        assert_eq!(s.peak, 6.0, "|-5 - 1| is the largest move, and it is negative before abs");
    }

    #[test]
    fn bf16_and_f32_walks_agree_on_values_both_can_hold() {
        let values = [1.0f32, -0.5, 3.0, 0.0, -0.0];
        let refs = [0.25f32, 1.0, 2.0, -0.0, 0.0];
        let a = accumulate(&bf16_bytes(&values), &bf16_bytes(&refs), Dtype::BF16, ZERO)
            .expect("valid input");
        let b = accumulate(&f32_bytes(&values), &f32_bytes(&refs), Dtype::F32, ZERO)
            .expect("valid input");
        assert_eq!(a, b);
    }

    #[test]
    fn an_empty_pair_is_a_valid_empty_walk() {
        let s = accumulate(&[], &[], Dtype::F16, ZERO).expect("nothing to refuse");
        assert_eq!(s, ZERO);
        assert_eq!(any_non_finite(&[], Dtype::BF16), Ok(false));
    }

    #[test]
    fn mismatched_lengths_are_refused_not_truncated() {
        let err = accumulate(&[0u8; 8], &[0u8; 4], Dtype::F32, ZERO).unwrap_err();
        assert_eq!(err, DiffError::LengthMismatch { candidate: 8, base: 4 });
        assert!(err.message().contains("8 bytes"), "{}", err.message());
    }

    #[test]
    fn a_length_that_is_not_whole_elements_is_refused() {
        let err = accumulate(&[0u8; 3], &[0u8; 3], Dtype::F32, ZERO).unwrap_err();
        assert_eq!(err, DiffError::NotWholeElements { bytes: 3, width: 4 });
        assert!(err.message().contains("not a whole number"), "{}", err.message());
        assert_eq!(
            any_non_finite(&[0u8; 5], Dtype::BF16),
            Err(DiffError::NotWholeElements { bytes: 5, width: 2 })
        );
    }

    #[test]
    fn non_finite_values_are_found_in_every_dtype() {
        assert_eq!(any_non_finite(&f32_bytes(&[1.0, f32::INFINITY]), Dtype::F32), Ok(true));
        assert_eq!(any_non_finite(&f32_bytes(&[1.0, 2.0]), Dtype::F32), Ok(false));
        assert_eq!(any_non_finite(&bf16_bytes(&[f32::NAN]), Dtype::BF16), Ok(true));
        assert_eq!(any_non_finite(&0x7c00u16.to_le_bytes(), Dtype::F16), Ok(true));
        assert_eq!(any_non_finite(&0x3c00u16.to_le_bytes(), Dtype::F16), Ok(false));
        assert_eq!(
            any_non_finite(&f64::NAN.to_le_bytes(), Dtype::F64),
            Ok(true)
        );
    }

    #[test]
    fn the_non_finite_scan_stops_at_the_first_hit() {
        let mut raw = f32_bytes(&[f32::NAN]);
        raw.extend(vec![0u8; 4 * 1000]);
        assert_eq!(any_non_finite(&raw, Dtype::F32), Ok(true));
    }
}
