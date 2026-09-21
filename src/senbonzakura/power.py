# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a comparison can and cannot see, worked out before it is run.

WHY THIS EXISTS

The v0.7 exit gate asks for *"at least five seeds per configuration, with the spread reported, and
a stated effect size that five seeds can actually resolve"*. Nothing in this repository computed
that number, so the requirement was met by writing "five seeds" in a plan and hoping.

The cost of not having it is not theoretical and it arrived from next door. The hephaestus session
compared three KV configurations at n=200 and got what looked like a clean ordering across ECE
0.123 to 0.140. Then it ran the SAME configuration twice: 0.103 and 0.157. Replication moved the
number by 0.054 while the whole spread between arms was 0.017, so **the noise floor was three
times the effect** and all three arms were indistinguishable. Every remaining v0.4 comparison here
is a small-effect comparison of exactly that shape.

THE NUMBER IN THE PLAN IS THE OPTIMISTIC ONE

The rung says five seeds resolve a gap of roughly 1.8 standard deviations. That is the NORMAL
approximation: (1.96 + 0.84) * sqrt(2/5) = 1.77. With five seeds you do not have a normal
distribution, you have a t with eight degrees of freedom, and the same calculation gives
(2.306 + 0.889) * sqrt(2/5) = 2.02.

The difference is about 14%, it runs in the dangerous direction, and it is the direction a
research tool must not err in: believing a comparison can resolve a gap it cannot is how a tie
gets published as a win. So this module uses t by default and says so, and the plan's 1.8 is kept
here only as the thing being corrected.

WHY THERE IS A TABLE RATHER THAN A DISTRIBUTION FUNCTION

Computing t and chi-square quantiles needs scipy, and this module is imported by the reporting
path, which must stay light. The tables below cover the seed counts this project actually runs.
**Outside the table it REFUSES rather than interpolating**, because an interpolated critical value
is a number that looks authoritative and is not, and every wrong number this project has published
looked authoritative too.

WHAT THIS DOES NOT DO

It says nothing about whether the standard deviation you hand it is right. An `sd` from three
seeds is itself barely known: `sd_interval` exists to show that, and at three seeds the 95%
interval on the spread runs from about half to six times the estimate, which is why a tie rule
built on three seeds looks like rigour and behaves like a coin flip.
"""
from __future__ import annotations

import math

#: Two-sided 0.975 and one-sided 0.80 critical values of t, by degrees of freedom. The pair is
#: what a power calculation needs: the first controls false positives, the second controls how
#: often a real gap is missed.
_T = {
    2: (4.302653, 1.060660),   # n=2 per arm
    4: (2.776445, 0.940965),   # n=3 per arm
    6: (2.446912, 0.905703),   # n=4 per arm
    8: (2.306004, 0.888890),   # n=5 per arm
    10: (2.228139, 0.879058),   # n=6 per arm
    12: (2.178813, 0.872609),   # n=7 per arm
    14: (2.144787, 0.868055),   # n=8 per arm
    16: (2.119905, 0.864667),   # n=9 per arm
    18: (2.100922, 0.862049),   # n=10 per arm
    20: (2.085963, 0.859964),   # n=11 per arm
    22: (2.073873, 0.858266),   # n=12 per arm
    24: (2.063899, 0.856855),   # n=13 per arm
    26: (2.055529, 0.855665),   # n=14 per arm
    28: (2.048407, 0.854647),   # n=15 per arm
    30: (2.042272, 0.853767),   # n=16 per arm
    32: (2.036933, 0.852998),   # n=17 per arm
    34: (2.032245, 0.852321),   # n=18 per arm
    36: (2.028094, 0.851720),   # n=19 per arm
    38: (2.024394, 0.851183),   # n=20 per arm
    40: (2.021075, 0.850700),   # n=21 per arm
    48: (2.010635, 0.849174),   # n=25 per arm
    58: (2.001717, 0.847862),   # n=30 per arm
    78: (1.990847, 0.846254),   # n=40 per arm
    98: (1.984467, 0.845304),   # n=50 per arm
    198: (1.972017, 0.843440),   # n=100 per arm
}

#: Chi-square 0.025 and 0.975 quantiles by degrees of freedom, for the interval on a variance.
_CHI2 = {
    1: (0.000982, 5.023886),   # n=2 per arm
    2: (0.050636, 7.377759),   # n=3 per arm
    3: (0.215795, 9.348404),   # n=4 per arm
    4: (0.484419, 11.143287),   # n=5 per arm
    5: (0.831212, 12.832502),   # n=6 per arm
    6: (1.237344, 14.449375),   # n=7 per arm
    7: (1.689869, 16.012764),   # n=8 per arm
    8: (2.179731, 17.534546),   # n=9 per arm
    9: (2.700389, 19.022768),   # n=10 per arm
    10: (3.246973, 20.483177),   # n=11 per arm
    11: (3.815748, 21.920049),   # n=12 per arm
    12: (4.403789, 23.336664),   # n=13 per arm
    13: (5.008751, 24.735605),   # n=14 per arm
    14: (5.628726, 26.118948),   # n=15 per arm
    15: (6.262138, 27.488393),   # n=16 per arm
    16: (6.907664, 28.845351),   # n=17 per arm
    17: (7.564186, 30.191009),   # n=18 per arm
    18: (8.230746, 31.526378),   # n=19 per arm
    19: (8.906516, 32.852327),   # n=20 per arm
    20: (9.590777, 34.169607),   # n=21 per arm
    24: (12.401150, 39.364077),   # n=25 per arm
    29: (16.047072, 45.722286),   # n=30 per arm
    39: (23.654325, 58.120060),   # n=40 per arm
    49: (31.554916, 70.222414),   # n=50 per arm
    99: (73.361080, 128.421989),   # n=100 per arm
}

#: The normal-approximation constant the v0.7 plan quotes, kept so the correction is legible
#: rather than silent: 1.959964 + 0.841621 at alpha 0.05 two-sided and 80% power.
NORMAL_CONSTANT = 2.801585


class PowerError(ValueError):
    """Asked for a number this module will not make up."""


def _critical(df):
    if df not in _T:
        raise PowerError(
            f"no tabulated critical value for {df} degrees of freedom. The table covers "
            f"{sorted(_T)}, which are the seed counts this project runs. Interpolating would "
            f"return a number that looks authoritative and is not; add the exact value to the "
            f"table instead.")
    return _T[df]


def detectable_gap(sd, n_per_arm, *, normal=False):
    """The smallest true difference two arms of `n_per_arm` seeds can resolve, in the same units
    as `sd`. At alpha 0.05 two-sided and 80% power.

    Below this, a real difference of that size is more likely than not to be reported as a tie.
    It is NOT a threshold for believing a result: a gap larger than this is resolvable, not true.

    `normal=True` reproduces the plan's 1.8-standard-deviation figure, which is the z-based
    approximation. It is offered so the two can be compared rather than so it can be used.
    """
    if n_per_arm < 2:
        raise PowerError(
            f"{n_per_arm} seeds per arm cannot produce a spread at all, so there is no "
            f"detectable gap to report. Two is the arithmetic minimum and it is not a "
            f"recommendation.")
    if sd < 0:
        raise PowerError(f"a standard deviation cannot be negative, got {sd}.")
    if sd == 0:
        # REFUSED RATHER THAN TREATED AS PERFECT PRECISION. A zero spread makes every gap
        # resolvable, including a gap of zero, and `report` would print "a gap of 0 is resolvable
        # at 5 seeds per arm". In this project's own experience a standard deviation of exactly
        # zero across seeds is the signature of a scorer that returned a constant, not of an
        # instrument with no noise: three byte-identical re-sampled Optuna points were recovered
        # on 2026-09-21 and that was a property of the resampling, not of the measurement.
        #
        # It matters here more than it looks, because `panel.judge_verdict` passes `sd` straight
        # in: a degenerate judge would come back `resolvable: True` and get a vote, in the module
        # written so that an underpowered judge cannot break a tie.
        raise PowerError(
            "a standard deviation of exactly zero is not infinite precision, it is a measurement "
            "that did not vary. Every gap, including a gap of nothing, would be reportable as "
            "resolvable. Check whether the scorer returned a constant before treating this as a "
            "result.")
    spread = math.sqrt(2.0 / n_per_arm)
    if normal:
        return NORMAL_CONSTANT * sd * spread
    a, b = _critical(2 * (n_per_arm - 1))
    return (a + b) * sd * spread


def sd_interval(sd, n_per_arm):
    """The 95% interval on the standard deviation itself, as (low, high).

    THE NUMBER THAT MAKES A SMALL-N TIE RULE READABLE. Every power calculation above takes `sd`
    as given, and with few seeds it is barely known. At three seeds the interval runs from about
    half to six times the estimate, so a rule built on it *looks* like rigour and behaves like a
    coin flip. The rung says exactly this about three seeds; this is the arithmetic behind it.
    """
    if n_per_arm < 2:
        raise PowerError(f"{n_per_arm} seeds gives no spread to put an interval around.")
    df = n_per_arm - 1
    if df not in _CHI2:
        raise PowerError(
            f"no tabulated chi-square quantiles for {df} degrees of freedom; the table covers "
            f"{sorted(_CHI2)}.")
    lo_q, hi_q = _CHI2[df]
    return (sd * math.sqrt(df / hi_q), sd * math.sqrt(df / lo_q))


def can_resolve(gap, sd, n_per_arm, *, normal=False):
    """Whether a comparison of this size can see a difference of `gap`. A verdict, with its reason.

    Returns a dict rather than a boolean, because "no" is the useful answer and it is only useful
    with the numbers beside it: how many seeds this gap WOULD need is the actionable part, and a
    bare False sends somebody to run the same underpowered comparison again.
    """
    need = detectable_gap(sd, n_per_arm, normal=normal)
    out = {
        "gap": gap,
        "sd": sd,
        "n_per_arm": n_per_arm,
        "detectable_gap": need,
        "resolvable": abs(gap) >= need,
        "basis": "normal approximation" if normal else f"t, {2 * (n_per_arm - 1)} df",
    }
    if not out["resolvable"]:
        out["seeds_needed"] = seeds_for(gap, sd, normal=normal)
    lo, hi = sd_interval(sd, n_per_arm)
    out["sd_interval"] = (lo, hi)
    # The spread is an estimate too, so the detectable gap has a range. Reported because a
    # comparison that is resolvable only at the optimistic end of the spread is not resolvable.
    out["detectable_gap_interval"] = (
        detectable_gap(lo, n_per_arm, normal=normal),
        detectable_gap(hi, n_per_arm, normal=normal))
    return out


#: The seeds-per-arm the table can answer for, smallest first. Derived from `_T` rather than
#: written out, so the two cannot drift.
TABULATED_SEEDS = tuple(sorted(df // 2 + 1 for df in _T))


def seeds_for(gap, sd, *, normal=False):
    """The smallest tabulated seeds-per-arm that resolves `gap`, or None when none does.

    ONLY TABULATED COUNTS ARE CONSIDERED, and the first version of this silently skipped the rest,
    which made it answer the wrong question. Asked for a gap needing about a hundred seeds, it
    walked a contiguous range, found no tabulated entry that fitted, and returned None, which the
    report then rendered as "no practical number of seeds resolves it; the instrument is the
    problem". That is a different and much stronger claim than "my table stops before there".

    A returned None now means genuinely out of reach: beyond the largest count the table holds,
    which is the point at which the honest answer really is that the effect is small relative to
    the noise and the fix is a better instrument rather than a longer night.
    """
    if gap == 0:
        return None
    for n in TABULATED_SEEDS:
        if abs(gap) >= detectable_gap(sd, n, normal=normal):
            return n
    return None


def report(v):
    """The verdict as lines a person reads, saying what was and was not established."""
    lines = []
    basis = v["basis"]
    if v["resolvable"]:
        lines.append(
            f"a gap of {v['gap']:.4g} is resolvable at {v['n_per_arm']} seeds per arm: the "
            f"smallest detectable difference is {v['detectable_gap']:.4g} ({basis}).")
    else:
        lines.append(
            f"a gap of {v['gap']:.4g} is NOT resolvable at {v['n_per_arm']} seeds per arm. The "
            f"smallest detectable difference is {v['detectable_gap']:.4g} ({basis}), so a real "
            f"difference this size would more often than not be reported as a tie.")
        need = v.get("seeds_needed")
        lines.append(
            f"  it would need about {need} seeds per arm." if need else
            f"  more than {max(TABULATED_SEEDS)} seeds per arm, which is past the point where "
            f"the answer is a longer night: the effect is small relative to the noise, and the "
            f"fix is a better instrument.")
    lo, hi = v["sd_interval"]
    lines.append(
        f"  the spread itself is estimated from {v['n_per_arm']} seeds and its own 95% interval "
        f"is {lo:.4g} to {hi:.4g}, so the detectable gap ranges "
        f"{v['detectable_gap_interval'][0]:.4g} to {v['detectable_gap_interval'][1]:.4g}.")
    lines.append(
        "  this says what the comparison can SEE, not what is true. A resolvable gap is not a "
        "real one, and a replication of the control arm is what shows whether the ruler can see "
        "the difference at all.")
    return lines
