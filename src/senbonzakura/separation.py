# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Candidate statistics for "does this axis separate refusal from topic?".

The question this module exists to answer is pre-registered in
`private/plans/pre-registration-2026-09-02-separation-statistic.md`, written before any of this
code. Read that first: it names the primary comparison, the nulls, and the ways the measurement
could turn out worthless. Nothing here may be tuned in the light of a result.

**Why there is more than one statistic.** The incumbent, Cohen's d, has never worked. Its first
form was unsatisfiable by construction and rejected 100% of candidates; the 2026-08-16 held-out
fix cured the cause and swapped the failure mode, so it now rejects 0%. A filter that accepts
everything is not a filter. Q-14 decided to MEASURE two replacements rather than argue one into
place, on the grounds that a project whose distinctive claim is that it measures what others
assert should not pick its central statistic from a list.

Each statistic carries its own null, because that is the number that makes its score readable.
"This axis scored 2.1" means nothing until something says what an axis carrying nothing scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:      # types only, and that is load-bearing
    from collections.abc import Callable

    import torch

# NOT AN OPTIMISATION AND NOT A STYLE CHOICE. `parser.py` reads this module's CHOICES to build the
# flag, and `parser.py` may import nothing heavy: that arrangement is what took `--help` from 2.80s
# to 0.06s and what lets `doctor` diagnose an install with no torch in it. Every function below
# works on tensors through their own methods and calls nothing off the `torch` namespace, so the
# import is needed for the annotations alone, and `from __future__ import annotations` defers
# those. A future `torch.something(...)` in here needs a local import inside the function.
# tests/test_startup_cost.py lists this module as LIGHT and measures the claim.

#: Both halves of a comparison need at least this many rows before the statistic means anything.
#: Below it the caller drops the candidate rather than reading a number computed over three rows.
MIN_GROUP_ROWS = 4


def cohens_d(pb: torch.Tensor, pg: torch.Tensor) -> float:
    """Standardised mean difference between the two projected clouds. The incumbent.

    `pb` and `pg` are 1-D projections of the harmful and harmless rows onto a unit candidate axis.

    Unweighted pooling, matching what this project has always computed, so a run under this
    statistic reproduces the recorded numbers exactly rather than approximately.
    """
    md = (pb.mean() - pg.mean()).abs()
    pooled = ((pb.var(unbiased=False) + pg.var(unbiased=False)) / 2).clamp_min(1e-12).sqrt()
    return float(md / pooled)


def variance_ratio(pb: torch.Tensor, pg: torch.Tensor) -> float:
    """Candidate A: between-group variance over within-group variance, with the df correction.

    The one-way ANOVA F statistic on the projections. Between-group mean square over within-group
    mean square, so a non-separating axis scores ~1.0 whatever the group sizes, and that
    size-invariance is the entire reason this candidate is interesting.

    THE DEGREES-OF-FREEDOM DIVISION IS NOT DECORATION. Without it the ratio is a raw sum-of-squares
    quotient whose null falls towards zero as the groups grow, which is the same disease as the
    incumbent's and would make this candidate a reparametrisation rather than a replacement.

    **What this fixes, measured before the model run** (20,000 null draws per size, harmless group
    held at 64; see the pre-registration addendum). Cohen's d against the fixed `0.5` keeps a
    meaningless axis 21.2% of the time when the cluster has 8 rows and 0.0% of the time when it has
    256, because d's null grows as the group shrinks and the fixed constant never accounted for it.
    The same sweep under this statistic against `4.0` holds 4.6% to 5.1% across the whole range.
    The incumbent is not one threshold; it is a different test at every cluster size, and the
    clusters this code judges differ in size by more than an order of magnitude by construction.

    At a FIXED pair of group sizes this is a strictly increasing function of |d| (exactly
    `d^2 (n-1)/2` for balanced groups), so it does not rank same-sized candidates differently. Its
    whole contribution is the size calibration, and the write-up must say so rather than let a
    reader infer independent discriminating power that is not there.

    **WHERE THE 1.0 NULL STOPS HOLDING, measured before any model run.** The ratio assumes the two
    groups share a variance. It tolerates that being false while they are the same size (4.9% to
    5.3% across an eightfold spread ratio at 32 against 32) and stops tolerating it as the sizes
    diverge, which is the Behrens-Fisher problem and not an implementation defect. A 4-row group
    four times as dispersed as its 32-row comparison clears the threshold 43.5% of the time with
    no separation present at all; a 128-row group half as dispersed clears it 17.8% of the time.

    That case is this extractor's NORMAL case, not a corner: a candidate is one harmful cluster,
    as few as MIN_CLUSTER_ROWS rows, judged against half the harmless set, which defaults to 128.
    The measured null floor the caller computes is what has to carry that weight, and it can only
    do so if it is drawn at the candidate's own group size. `tests/test_separation_statistics.py`
    pins both halves of this so a future change to the pooling has to be argued rather than
    noticed later.
    """
    n1, n2 = int(pb.numel()), int(pg.numel())
    total = n1 + n2
    if total <= 2:
        # Two degrees of freedom go to the group means; with none left the ratio is 0/0. The
        # caller's MIN_GROUP_ROWS check should make this unreachable, and it returns the null
        # rather than raising so a degenerate layer cannot take a run down.
        return 1.0
    grand = (pb.sum() + pg.sum()) / total
    between = n1 * (pb.mean() - grand) ** 2 + n2 * (pg.mean() - grand) ** 2
    within = ((pb - pb.mean()) ** 2).sum() + ((pg - pg.mean()) ** 2).sum()
    within_ms = (within / (total - 2)).clamp_min(1e-12)
    return float(between / within_ms)


def classifier_auc(pb: torch.Tensor, pg: torch.Tensor) -> float:
    """Candidate B: the probability that a harmful projection outranks a harmless one.

    The Mann-Whitney U as an area under the curve, which is the one-dimensional classifier the
    pre-registration asks for: there is no threshold to fit, because the AUC integrates over every
    threshold at once. 0.5 is chance.

    NOT FOLDED AROUND 0.5, and that is deliberate. The candidate axis is the harmful cluster's mean
    minus the harmless mean, so a genuine refusal axis puts the harmful cloud on the high side by
    construction and an axis that separates the other way is not the thing being looked for.
    Folding would also destroy the null: `0.5 + |auc - 0.5|` has an expectation strictly above 0.5,
    so the pre-registered null would be wrong for the statistic actually computed.

    **Where it beats Candidate A, measured before any model run.** Its null mean is 0.5 at every
    group size AND at every spread ratio (0.4980 to 0.4989 across a spread ratio of 0.5 to 4.0 on
    an 8-against-128 comparison, where the variance ratio's keep-rate goes from 0.7% to 29.9%).
    Being rank-based, it never assumes the two clouds share a variance, so the Behrens-Fisher
    failure that limits Candidate A cannot reach it.

    **Where it does not.** Its null SPREAD is not size-invariant: the 95th percentile is 0.736 for
    a 4-row group and 0.559 for a 128-row one, both against 128 harmless rows. So a fixed
    threshold is again a different test at each cluster size, in the same way the incumbent's is
    and for the same reason, though far less severely. The margin below is therefore set at the
    worst case rather than the typical one, and the measured null floor is what actually adapts.
    """
    # The compass's own implementation. It imports torch, so it cannot be imported at module
    # level: `parser.py` reads this module's CHOICES and may pull in nothing heavy.
    from .margin import auc
    return float(auc(pb.tolist(), pg.tolist()))


@dataclass(frozen=True)
class Statistic:
    """A separation statistic together with the two numbers that make it readable.

    `null` is what an axis carrying nothing scores; `margin` is how far above that a candidate has
    to land before the fixed backstop keeps it. Both are stated here, in advance, and neither may
    be moved in the light of a rejection rate. A threshold chosen after seeing the rates is named
    in the pre-registration as one of the things that would make the measurement worthless.
    """

    name: str
    fn: Callable[[torch.Tensor, torch.Tensor], float]
    null: float
    margin: float
    rationale: str

    @property
    def threshold(self) -> float:
        """The fixed backstop. A run also measures a null floor and takes whichever is larger."""
        return self.null + self.margin

    def score(self, pb: torch.Tensor, pg: torch.Tensor) -> float | None:
        """The statistic, or None when either side is too small for it to mean anything."""
        if int(pb.numel()) < MIN_GROUP_ROWS or int(pg.numel()) < MIN_GROUP_ROWS:
            return None
        return self.fn(pb, pg)


STATISTICS: dict[str, Statistic] = {
    "cohens-d": Statistic(
        name="cohens-d",
        fn=cohens_d,
        null=0.0,
        margin=0.5,
        # 0.5 is the conventional "medium effect" line, chosen once in this project and never
        # validated against a measurement. It is kept verbatim as the incumbent so that the
        # comparison is between statistics and not between a statistic and a retuned constant.
        rationale="conventional medium effect size; the incumbent, kept unchanged for comparison",
    ),
    "variance-ratio": Statistic(
        name="variance-ratio",
        fn=variance_ratio,
        null=1.0,
        margin=3.0,
        # 4.0 is the 95th percentile of the null, measured at 3.83 to 4.01 across harmful groups
        # of 8 to 256 rows, so the backstop is a stated one-in-twenty per-axis false-keep rate
        # rather than a round number. It is close to the F(1, inf) critical value of 3.84 because
        # that is what it is.
        rationale="null 95th percentile: a 5% per-axis false-keep rate, flat across group sizes",
    ),
    "auc": Statistic(
        name="auc",
        fn=classifier_auc,
        null=0.5,
        margin=0.25,
        # 0.75 is the null's 95th percentile at the SMALLEST comparison this code permits, a
        # 4-row group against a large harmless half, where the null is at its widest. Set at the
        # worst case rather than the typical one because the alternative is a backstop that only
        # holds for big clusters, and small clusters are where a filter is most easily fooled. It
        # is conservative for a large cluster by design: the measured null floor is what tightens
        # there, and a backstop is meant to be the loose outer bound.
        rationale="null 95th percentile at the smallest permitted group; deliberately conservative",
    ),
}

#: What `--separation-statistic` defaults to. The incumbent, so that adding this module changes no
#: existing result. Q-14 is a measurement, and it does not become the default by being implemented.
DEFAULT_STATISTIC = "cohens-d"

#: The flag's choices, in a stable order so `--help` does not depend on dictionary iteration.
CHOICES = sorted(STATISTICS)


def get(name: str) -> Statistic:
    """Look up a statistic by its flag value, failing with the list rather than a KeyError."""
    try:
        return STATISTICS[name]
    except KeyError:
        known = ", ".join(sorted(STATISTICS))
        raise ValueError(
            f"unknown separation statistic {name!r}; this build knows: {known}") from None
