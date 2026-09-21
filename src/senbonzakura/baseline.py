# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The recorded measurement a later one is judged against, and the rules for judging.

WHY THIS EXISTS, and it is not "because CI should run more"

A number gets checked when somebody wonders about it. That is the difference between a tool used
occasionally and one used constantly, and it is the property this project most clearly lacks. A
gate closes it: a change that moves a measured property outside its interval fails a build, before
anything is promoted, without anyone having to wonder.

THE FAILURE THIS MODULE EXISTS TO PREVENT IS NOT A REGRESSION. It is a gate that compares two
numbers which were never comparable, and gives the result a green tick. This project has already
published a table where exactly that happened: three copies of the prompt renderer drifted, and a
configuration selected under one format was reported under another. Automating that comparison
would convert a subtle error into a passing build, which is worse than having no gate, because a
passing build is evidence and a missing gate is not.

So the refusal is the feature. A baseline records what it was measured under, and a comparison
against a baseline that does not match REFUSES rather than comparing anyway. The same discipline
already exists one layer down in `runrecord`, which stops a `--resume` carrying one run's trials
into another run's inputs, and the shape is deliberately the same: a small set of pinned fields,
a mismatch report, and a refusal that names the two things the reader can actually choose between.

THE GATE FIRES ON THE INTERVAL, NOT THE POINT ESTIMATE, and that is what keeps it switched on. A
gate that fails on noise is disabled within two weeks and never re-enabled, so "the mean moved" is
not a finding here; "the intervals no longer overlap" is. This is also why this work could not come
before the intervals existed.

Torch-free and import-light on purpose: the gate runs on whatever hardware a CI runner has.
"""
import json
from pathlib import Path

from .crashsafe import atomic_write

#: Bumped only for a change that makes an older file mean something different. Adding a field is
#: not that: readers use `.get`, and an absent field is reported as unknown rather than as a match.
SCHEMA = "senbonzakura-baseline/1"

#: What must agree for two measurements to be comparable, and what each one decides. This is the
#: whole safety argument of the module, so each entry says why it is here rather than only what it
#: is called.
PINNED = {
    "model": "the weights the measurement was taken on",
    "metric": "which property was measured, by identity rather than by display name",
    # NAMED FOR THE SLOT AND NOT FOR THE CORPUS, since 2026-09-21. It was `track_digest`, which
    # reads as "the digest of a track" and so only fits a measurement scored on a track.
    # `coherence.py` measures a fixed passage, invented `passage_digest` for the same slot, and
    # the two never met: `comparability` reports a pinned field absent on BOTH sides as a
    # mismatch, so a coherence figure and a baseline could never be compared and nothing said
    # why. One neutral name for "which input this number was taken on" is what makes them
    # comparable at all.
    "input_digest": "which input the measurement was taken on: the corpus for a scored run, the "
                    "passage for a probe. Neutral because the slot is the same question",
    "partition": "which rows of that corpus, so a measure-partition figure is never compared "
                 "against a search-partition one",
    "prompt_format": "how the prompt was rendered. Three copies of the renderer drifted once and "
                     "a table was published across the gap",
    "tool_version": "the code that produced it, because the edit and the scorer both live here",
}

#: A metric where a LARGER number is better (capability accuracy), against one where a SMALLER
#: number is better (refusal rate, KL divergence). Stored per baseline rather than inferred from
#: the name, because inferring it from a name is how "noncompliance" and "compliance" end up
#: sharing a direction.
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
DIRECTIONS = (HIGHER_IS_BETTER, LOWER_IS_BETTER)


class BaselineError(Exception):
    """A baseline that cannot be read, or one that cannot be compared against."""


def record(*, model, metric, direction, point, interval, input_digest, partition,
           prompt_format, tool_version, seeds, n, extra=None):
    """Build a baseline artefact. Every field is required except `extra`, deliberately.

    A baseline with a hole in it is the thing this module exists to refuse, so there is no way to
    build one by forgetting an argument: a missing field would be silently absent later, and
    absence is exactly what the comparability check has to treat as unknown.
    """
    if direction not in DIRECTIONS:
        raise BaselineError(
            f"direction must be one of {DIRECTIONS}, not {direction!r}. Whether a number moving up "
            f"is better or worse is not inferable from the metric's name, and guessing it wrong "
            f"turns a regression into a pass.")
    lo, hi = interval
    if lo > hi:
        raise BaselineError(f"interval {interval} is inverted: its low bound exceeds its high one.")
    if not (lo <= point <= hi):
        raise BaselineError(
            f"point estimate {point} lies outside its own interval {interval}. One of the two was "
            f"computed on different data from the other, and a gate built on it would compare a "
            f"number to an interval that never described it.")
    if n <= 0:
        raise BaselineError(f"a baseline measured on {n} observations is not a measurement.")
    return {
        "schema": SCHEMA,
        "model": str(model),
        "metric": str(metric),
        "direction": direction,
        "point": float(point),
        "interval": [float(lo), float(hi)],
        "n": int(n),
        "input_digest": str(input_digest),
        "partition": str(partition),
        "prompt_format": str(prompt_format),
        "tool_version": str(tool_version),
        # Sorted at the boundary so two runs that chose the same seeds in a different order
        # produce byte-identical baselines.
        "seeds": sorted(int(s) for s in seeds),
        "extra": dict(extra or {}),
    }


def write(path, baseline):
    """Write a baseline atomically. Never in place over an existing one.

    A new baseline is a NEW artefact and the old one stays, because "what did we compare against
    in March" has to remain answerable in June. Overwriting would make the history of a gate
    unauditable at exactly the moment somebody disputes a verdict.
    """
    path = Path(path)
    if path.exists():
        raise BaselineError(
            f"{path} already exists, and baselines are never overwritten in place. A new baseline "
            f"is a new file: name it for what changed, and keep the old one so the comparison it "
            f"was used for stays reproducible.")
    with atomic_write(path) as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read(path):
    """Load a baseline, or say which of the several different problems it is."""
    path = Path(path)
    if not path.is_file():
        raise BaselineError(f"no baseline at {path}.")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError) as e:
        raise BaselineError(f"{path} is not readable JSON ({e}).") from e
    if not isinstance(loaded, dict):
        raise BaselineError(f"{path} holds {type(loaded).__name__}, not a baseline object.")
    got = loaded.get("schema")
    if got != SCHEMA:
        raise BaselineError(
            f"{path} declares schema {got!r} and this build understands {SCHEMA!r}. A schema "
            f"change means an older file means something different, so it is refused rather than "
            f"read on the assumption that the fields still line up.")
    return loaded


def comparability(baseline, now):
    """Which pinned fields disagree, as (field, recorded, given, why) tuples. Empty means usable.

    A field that either side does not carry is reported as a MISMATCH rather than skipped, which is
    the opposite of what `runrecord.mismatches` does, and the difference is deliberate. There, an
    older record with a missing field still describes a run somebody is trying to finish, and
    refusing would strand them. Here, an unknown field means the gate cannot tell whether the two
    measurements are comparable, and "cannot tell" has to fail: the entire value of this module is
    that it does not compare things it cannot vouch for.
    """
    out = []
    for field, why in PINNED.items():
        was = (baseline or {}).get(field)
        current = (now or {}).get(field)
        if was is None or current is None:
            out.append((field, was, current,
                        f"not recorded on one side, so it cannot be shown to match: {why}"))
        elif str(was) != str(current):
            out.append((field, was, current, why))
    return out


def refuse_if_incomparable(baseline, now):
    """Stop a comparison that would put a green tick on two numbers that never matched.

    Raises BaselineError naming every disagreement at once, because a reader fixing them one
    error at a time is a reader who re-runs the gate six times.
    """
    bad = comparability(baseline, now)
    if not bad:
        return
    lines = [f"  {field}: baseline {was!r}, this run {current!r}  ({why})"
             for field, was, current, why in bad]
    raise BaselineError(
        "this measurement is not comparable to that baseline, so it was not compared:\n"
        + "\n".join(lines)
        + "\n\nEither measure under the conditions the baseline records, or record a new baseline "
          "and say in its filename what changed. Comparing across a mismatch is how a drifted "
          "prompt renderer got a published table and a passing build.")


def overlaps(a, b):
    """Do two closed intervals share any point? Touching at an endpoint counts as overlapping.

    Touching counts because the boundary case is noise, not evidence, and a gate that fires on
    `hi == lo` fires on rounding.
    """
    return a[0] <= b[1] and b[0] <= a[1]


def verdict(baseline, now_point, now_interval):
    """Compare, and return (ok, headline, detail). Refuses first; never compares blind.

    A regression requires BOTH that the intervals are disjoint and that the move is in the
    direction that is worse. Disjoint-and-better is a pass, and it still says so out loud, because
    a measurement that moved a long way is worth a reader's attention whichever way it went.
    """
    lo, hi = float(now_interval[0]), float(now_interval[1])
    if lo > hi:
        raise BaselineError(f"interval [{lo}, {hi}] is inverted.")
    base_iv = (float(baseline["interval"][0]), float(baseline["interval"][1]))
    metric, direction = baseline["metric"], baseline["direction"]
    moved = float(now_point) - float(baseline["point"])

    if overlaps(base_iv, (lo, hi)):
        # THE INTERVAL IS PRINTED ON A PASS, not only on a failure. A gate whose threshold is so
        # wide that nothing trips it reads as safety and is decoration, and the only way a reader
        # can tell the difference is by seeing how much room there was.
        return True, f"{metric}: within interval", (
            f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
            f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals overlap, so "
            f"there is no evidence the property moved. Moved {moved:+.4f} on the point estimate, "
            f"which on its own is not a finding.")

    worse = moved < 0 if direction == HIGHER_IS_BETTER else moved > 0
    where = "below" if moved < 0 else "above"
    if worse:
        return False, f"{metric}: REGRESSED", (
            f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
            f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals do not overlap "
            f"and the value moved {where} it by {abs(moved):.4f}, which is the worse direction for "
            f"a metric where {direction.replace('_', ' ')}.")
    return True, f"{metric}: improved beyond its interval", (
        f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
        f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals do not overlap and "
        f"the move is in the better direction, so this passes. It is still a change worth "
        f"recording: something altered the measurement by {abs(moved):.4f}.")


#: What a green verdict does NOT say, printed with every one of them. Somebody will eventually read a
#: green tick as "the model is safe", and the only defence is that the artefact says otherwise in
#: the same breath as the verdict.
VERDICT_CAVEAT = (
    "A pass means one measured property did not move outside its interval, on one track, under "
    "the conditions the baseline records. It is not a statement that the model is safe, that "
    "other properties held, or that the track represents anything beyond itself.")
