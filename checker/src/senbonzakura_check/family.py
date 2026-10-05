# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Many single-model results, read as one statement about a family, or a refusal to make one.

WHAT THIS ANSWERS

A publisher with a lineage can say what each of its models scored and nothing about the lineage.
The questions they actually have are "does this family hold up across sizes" and "is my quality
consistent", and no artefact answers either, because every artefact is about one model.

WHY IT LIVES HERE AND NOT IN THE MAIN PACKAGE

Aggregation needs no model, no GPU and no corpus: it reads result files that already exist. That
is this package's exact constraint profile, and the adapters here already express six artefact
kinds in one vocabulary, which is the hard half of aggregating them. Putting it in the producing
package would mean re-reading artefacts it has already written out, and would couple a
zero-dependency job to a torch install.

THE DEFECT THIS MODULE EXISTS TO AVOID

**Averaging the point estimates.** Five models scoring 10%, 12%, 9%, 40% and 11% do not have a
family rate of 16.4%. That number describes no model, hides the one that is different, and moves
with how many small models the publisher happened to release. So this module never averages. It
asks a different question whose answer is always defensible:

    Is there a single value consistent with EVERY member's interval?

If there is, the evidence does not distinguish the members, and one family figure is honest. If
there is not, **the family is not one thing**, and the correct output says so and names the pairs
that are actually separated. That verdict is the product. The pooled number is a by-product and
is only offered in the case where it cannot mislead.

WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL IS LOAD-BEARING

- **Mixed metrics, estimators or units are refused**, not reconciled. `baseline.py` in the main
  package already refuses incomparable pairs, and the reasoning is the same one order of magnitude
  up: a refusal rate from a keyword ruler and one from a model judge are different quantities, and
  aggregating them produces a figure of nothing.
- **A member with no interval is listed and excluded from the verdict, loudly.** Dropping it
  silently would make the family look more consistent the less evidence it had, which inverts the
  whole point.
- **Two artefacts is a comparison, not a family.** Three is the minimum, and the refusal points at
  the pair mode rather than quietly producing a two-row table.
- **A size trend needs sizes, and sizes are supplied by the caller**, because no artefact records a
  parameter count. Asked for one without them, this says so rather than parsing a guess out of a
  model name, where "Qwen3.5-27B-abliterated-v2" would hand a regex two numbers and no way to know
  which is the size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# wilson_interval lives in `cardread` because that is where it was first needed. It is imported
# rather than moved: the move would be a rename with no behaviour change, and a test already pins
# it there and asserts it agrees with the main package's implementation to one part in a trillion.
from .cardread import wilson_interval

#: Below this, the word "family" is not earned. Two artefacts is a head-to-head and the pair mode
#: already reads those, with checks written for exactly that shape.
MIN_MEMBERS = 3

#: A trend claim needs more than a direction between two points. With fewer separated pairs than
#: this, the honest output is the per-member table and no trend sentence at all.
MIN_SEPARATED_PAIRS_FOR_TREND = 2

#: The fields that must agree across every member before anything is aggregated.
COMPARABLE_FIELDS = ("metric", "estimator", "units")


class FamilyError(Exception):
    """An aggregation that cannot honestly be made, phrased for a person."""


@dataclass(frozen=True)
class Member:
    """One model's reading of one metric, with everything needed to reason about it."""

    source: str
    model: str
    value: float
    n: int | None
    low: float | None
    high: float | None
    #: Where the interval came from. `artefact` means the producing tool computed it, `wilson`
    #: means this module did, and `none` means there isn't one. Recorded because an interval the
    #: tool computed may be a bootstrap or a paired interval, which is not the same object as a
    #: Wilson interval on a count, and a reader comparing a mixed column should be able to see it.
    interval_basis: str
    size: float | None = None

    @property
    def has_interval(self) -> bool:
        return self.low is not None and self.high is not None


@dataclass(frozen=True)
class FamilyResult:
    """What can be said about the family, and what cannot."""

    metric: str
    estimator: str
    units: str | None
    members: tuple
    #: Members carrying no interval. Listed rather than dropped.
    without_interval: tuple
    #: True when one value is consistent with every member that has an interval.
    one_figure_defensible: bool | None
    #: The widest range of values consistent with every member, when there is one.
    common_low: float | None
    common_high: float | None
    #: Pairs the evidence actually separates, as (member_a, member_b, gap).
    separated: tuple
    pooled: dict | None
    trend: dict | None
    notes: tuple = field(default=())


def _interval_for(block, value):
    """The interval to reason with, and an honest account of where it came from."""
    raw = block.get("interval")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = raw
        if _is_number(lo) and _is_number(hi) and lo <= hi:
            return float(lo), float(hi), "artefact"

    n = block.get("n")
    # Wilson needs a count, which only exists if the value really is a proportion of n. A metric
    # whose units are not a proportion may be a divergence or a duration, and multiplying it by n
    # to recover a count would invent one.
    if block.get("units") == "proportion" and _is_whole(n) and n > 0 and 0.0 <= value <= 1.0:
        count = round(value * n)
        lo, hi = wilson_interval(count, int(n))
        return lo, hi, "wilson"
    return None, None, "none"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def collect(documents, *, metric, sizes=None):
    """Members for `metric` across normalised documents, or a refusal naming the obstacle.

    `documents` is an iterable of `(source, normalised_doc)`. The source is carried through so
    every refusal and every row can name the file it came from, which is the difference between a
    message a person can act on and one they have to go looking for.
    """
    sizes = dict(sizes or {})
    members, seen = [], {}
    agreed = {}
    missing_metric = []

    for source, doc in documents:
        blocks = doc.get("metrics") or {}
        block = _one_block(blocks, metric)
        if block is None:
            missing_metric.append(source)
            continue

        value = block.get("value")
        if not _is_number(value):
            raise FamilyError(
                f"{source}: {metric} has value {value!r}, which is not a number, so it cannot "
                f"take part in an aggregate. One unusable row is reported rather than skipped, "
                f"because a family figure quietly computed over fewer models than were supplied "
                f"would look stronger than the evidence behind it.")

        for key in COMPARABLE_FIELDS:
            got = block.get(key)
            if key in agreed and agreed[key] != got:
                raise FamilyError(
                    f"{source}: {key} is {got!r} and the artefacts before it agree on "
                    f"{agreed[key]!r}. These are different quantities and aggregating them would "
                    f"produce a figure of nothing. Aggregate each group separately.")
            agreed[key] = got

        model = doc.get("model") or source
        if model in seen:
            raise FamilyError(
                f"two artefacts describe the model {model!r} ({seen[model]} and {source}). A "
                f"family aggregate counts each member once; two readings of one model are a "
                f"reproducibility question and the pair mode is the tool for it.")
        seen[model] = source

        low, high, basis = _interval_for(block, float(value))
        members.append(Member(
            source=source, model=model, value=float(value), n=block.get("n"),
            low=low, high=high, interval_basis=basis, size=_size_for(model, sizes)))

    if missing_metric and not members:
        raise FamilyError(
            f"none of the {len(missing_metric)} artefacts carries a {metric!r} metric. Check the "
            f"name against what the artefacts actually record; a metric that is absent everywhere "
            f"is usually a spelling rather than a gap.")
    if len(members) < MIN_MEMBERS:
        raise FamilyError(
            f"{len(members)} artefact(s) carry {metric!r} and a family statement needs at least "
            f"{MIN_MEMBERS}. Two is a comparison rather than a family: run it with --pair, which "
            f"has checks written for exactly that shape.")

    unknown = sorted(set(sizes) - {m.model for m in members})
    if unknown:
        raise FamilyError(
            f"a size was supplied for {', '.join(repr(u) for u in unknown)}, which no artefact "
            f"describes. A size attached to nothing is usually a typo in the model name, and "
            f"silently ignoring it would mean the trend was computed over fewer points than the "
            f"caller believes.")

    return tuple(members), agreed, tuple(missing_metric)


def _one_block(blocks, metric):
    """The block for `metric`, tolerating the key-prefixing adapters do.

    An adapter may key a metric `claim0.percent` or `refusal` or `arm_a.refusal`, so the match is
    on the block's own declared `metric` field first and the key only as a fallback. Matching the
    key alone would miss every prefixed artefact, and matching loosely would let `soft_refusal`
    answer a request for `refusal`.
    """
    for key, block in sorted(blocks.items()):
        if not isinstance(block, dict):
            continue
        if block.get("metric") == metric or key == metric:
            return block
    return None


def _size_for(model, sizes):
    value = sizes.get(model)
    return float(value) if _is_number(value) else None


def _common_range(members):
    """The widest range of values consistent with every member's interval, or None."""
    withs = [m for m in members if m.has_interval]
    if not withs:
        return None, None
    low = max(m.low for m in withs)
    high = min(m.high for m in withs)
    if low > high:
        return None, None
    return low, high


def _separated(members):
    """Every pair whose intervals do not touch, with the gap between them.

    These are the only differences inside the family that the evidence supports. A pair whose
    intervals overlap is reported nowhere, which is the point: the absence of a row is the finding.
    """
    out = []
    withs = [m for m in members if m.has_interval]
    for i, a in enumerate(withs):
        for b in withs[i + 1:]:
            lower, upper = (a, b) if a.value <= b.value else (b, a)
            if lower.high < upper.low:
                out.append((lower, upper, upper.low - lower.high))
    out.sort(key=lambda row: -row[2])
    return tuple(out)


def _pooled(members, units):
    """One figure for the family, offered only where it cannot mislead.

    THREE CONDITIONS, and each one removes a way for this number to be read as something it is
    not. The units must be `proportion`, so there are counts to add. Every member must carry an
    `n`, because a sum over a subset weights the family by which artefacts happened to record a
    sample size. And the members must not be separated, because pooling things the evidence says
    are different is the averaging defect wearing a Wilson interval.
    """
    if units != "proportion":
        return None
    if any(not (_is_whole(m.n) and m.n > 0) for m in members):
        return None
    total = sum(m.n for m in members)
    count = sum(round(m.value * m.n) for m in members)
    lo, hi = wilson_interval(count, total)
    return {"count": count, "n": total, "value": count / total, "low": lo, "high": hi,
            "means": "the rate over every member's prompts pooled, which is a statement about a "
                     "reply drawn from this family's runs and not about any one model"}


def _trend(members):
    """Whether the separated pairs agree about direction as size rises, or a refusal to say.

    Computed over SEPARATED pairs only. A trend fitted to point estimates would find a direction
    in noise every time, which is the same defect as averaging and is harder to see.
    """
    sized = [m for m in members if m.size is not None and m.has_interval]
    if len(sized) < MIN_MEMBERS:
        return {"verdict": "no trend can be read",
                "why": f"{len(sized)} member(s) have both a size and an interval, and a trend "
                       f"needs at least {MIN_MEMBERS}. Sizes are supplied by the caller because "
                       f"no artefact records a parameter count."}

    rising = falling = 0
    for i, a in enumerate(sized):
        for b in sized[i + 1:]:
            if a.size == b.size:
                continue
            small, large = (a, b) if a.size < b.size else (b, a)
            if small.high < large.low:
                rising += 1
            elif large.high < small.low:
                falling += 1

    pairs = rising + falling
    if pairs < MIN_SEPARATED_PAIRS_FOR_TREND:
        return {"verdict": "no trend can be read", "rising": rising, "falling": falling,
                "why": f"{pairs} pair(s) of different sizes are actually separated, and a "
                       f"direction needs at least {MIN_SEPARATED_PAIRS_FOR_TREND}. Every other "
                       f"pair's intervals overlap, so the evidence does not order them."}
    if rising and falling:
        return {"verdict": "the direction is inconsistent", "rising": rising, "falling": falling,
                "why": f"{rising} separated pair(s) rise with size and {falling} fall, so the "
                       f"family does not move one way. A single trend line through this would "
                       f"describe the sample and not the lineage."}
    return {"verdict": "rises with size" if rising else "falls with size",
            "rising": rising, "falling": falling,
            "why": f"every one of the {pairs} separated pair(s) runs the same way. This orders "
                   f"the members that are distinguishable; it is not a fitted slope and no rate "
                   f"of change is claimed."}


def aggregate(documents, *, metric, sizes=None, want_trend=False):
    """The family statement for `metric`, or a FamilyError saying why there isn't one."""
    members, agreed, missing = collect(documents, metric=metric, sizes=sizes)
    units = agreed.get("units")

    without = tuple(m for m in members if not m.has_interval)
    withs = tuple(m for m in members if m.has_interval)
    low, high = _common_range(members)
    one_figure = None if not withs else low is not None
    separated = _separated(members)

    notes = []
    if missing:
        notes.append(
            f"{len(missing)} artefact(s) carry no {metric!r} metric and took no part: "
            f"{', '.join(missing)}. They are named rather than counted out silently.")
    if without:
        notes.append(
            f"{len(without)} member(s) carry no interval and are excluded from the verdict: "
            f"{', '.join(m.model for m in without)}. A member with no interval cannot be shown to "
            f"agree or disagree with the others, and dropping it quietly would make the family "
            f"look more consistent the less evidence there was for it.")
    bases = sorted({m.interval_basis for m in withs})
    if len(bases) > 1:
        notes.append(
            f"the intervals come from more than one source ({', '.join(bases)}). An interval the "
            f"producing tool computed may be a bootstrap or a paired interval, which is a "
            f"different object from a Wilson interval on a count, so the comparison between those "
            f"rows is weaker than within either group.")

    pooled = _pooled(members, units) if one_figure and not separated else None
    if one_figure and not separated and pooled is None:
        notes.append(
            "no pooled figure is offered. It needs proportions with a sample size on every "
            "member, and these do not have that, so adding them would weight the family by which "
            "artefacts happened to record an n.")

    return FamilyResult(
        metric=metric, estimator=agreed.get("estimator"), units=units,
        members=members, without_interval=without, one_figure_defensible=one_figure,
        common_low=low, common_high=high, separated=separated, pooled=pooled,
        trend=_trend(members) if want_trend else None, notes=tuple(notes))


def verdict_sentence(result) -> str:
    """The one line a reader takes away, in plain words."""
    if result.one_figure_defensible is None:
        return (f"Nothing can be said about {result.metric} across this family: no member carries "
                f"an interval, so there is no way to tell agreement from coincidence.")
    if not result.one_figure_defensible:
        n = len(result.separated)
        return (f"This family is not one thing on {result.metric}. No single value is consistent "
                f"with every member, and {n} pair(s) are separated outright, so a family figure "
                f"would describe none of them.")
    span = f"{result.common_low:.4g} to {result.common_high:.4g}"
    return (f"Nothing here distinguishes the members on {result.metric}: every interval contains "
            f"the range {span}, so one family figure is defensible. That is a statement about "
            f"what the evidence can resolve, not a claim that the models are identical.")


__all__ = ["COMPARABLE_FIELDS", "MIN_MEMBERS", "MIN_SEPARATED_PAIRS_FOR_TREND", "FamilyError",
           "FamilyResult", "Member", "aggregate", "collect", "verdict_sentence"]
