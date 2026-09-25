# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""How a measuring command fills the fields `baseline.PINNED` requires, in one place.

WHY THIS EXISTS. `baseline.PINNED` names eight fields that decide whether two measurements may be
compared, and `measurement.stamp` is where a writer records them. Between the two, every writer
had to know how to derive five of them, and four of the five writers derived none: on 2026-09-25 a
self-review pass measured `margin`, `drift`, `capability` and `score` and found the same five
absent from all four (`input_digest`, `partition`, `prompt_format`, `tool_version`, `precision`),
so no figure any of them produced could become a baseline. `coherence` carried all eight and was
refused on a different field entirely. Nothing this repository could produce was gateable, which
is the exact condition `baseline`'s own header says that module was written to end.

FIVE COPIES OF THE DERIVATION WOULD HAVE BEEN THE SAME BUG AGAIN. The reason the fields were
missing is not that anyone decided against them; it is that each writer would have had to invent
the same five answers independently, and four of them never got round to it. One place that knows
how to answer them is what stops the fifth writer from being written without them.

NOTHING HERE GUESSES. Each helper either knows the answer or says out loud that it does not, in a
value that will not compare equal to a known one. `comparability` treats a mismatch as a refusal to
compare, which is the safe direction: a run whose partition nobody can establish is refused rather
than silently compared against one whose partition is known.
"""

import hashlib

#: What `partition` says for a run that covered every row of its input.
#:
#: Not a hole and not `measure`: a run over the whole set includes the rows the surgery was fitted
#: on, so it is a different measurement from one over the held-out tail, and the two must never
#: compare equal. Naming it is what keeps them apart.
ALL_ROWS = "all-rows"

#: What `partition` says when the run skipped rows and nothing recorded where the boundary was.
#:
#: The number after it is the skip itself, so two runs that skipped the same rows of the same input
#: still compare equal, while a run whose boundary came from a manifest never compares equal to one
#: whose boundary was a flag nobody checked. That asymmetry is deliberate: a verified partition and
#: an unverified one are not the same claim, even when the row counts coincide.
UNVERIFIED_PREFIX = "rows-from-"

#: What `partition` says for a run over the held-out tail, as the track's own manifest recorded it.
MEASURE = "measure"


def input_digest_of(prompts) -> str:
    """A digest of the prompts a number was actually taken on, as sixteen hex characters.

    OVER WHAT WAS SCORED, not over the path it came from. A path is not an identity: `--eval` can
    name a Hub id whose contents changed, a directory somebody edited, or a file that was truncated
    between two runs, and all three read as the same input to a reader comparing two artefacts. The
    digest answers the question the field is pinned for, which is "were these the same prompts".

    Sixteen characters to match `coherence.passage_digest` and the track fingerprint, so the three
    digests a reader meets in this project's artefacts look like the same kind of thing.
    """
    h = hashlib.sha256()
    for p in prompts:
        # Length-prefixed, so two different lists of prompts cannot concatenate to the same bytes.
        # ["ab", "c"] and ["a", "bc"] are different inputs and a digest that cannot tell them apart
        # is a digest that reports two measurements as comparable when they are not.
        raw = str(p).encode("utf-8")
        h.update(str(len(raw)).encode("ascii"))
        h.update(b"\x00")
        h.update(raw)
    return h.hexdigest()[:16]


def model_precision(model, load_in_4bit) -> str:
    """What numerical precision this reading was taken at, as a short string.

    A PINNED FIELD SINCE 2026-09-21, added because a panel reviewer pointed out that the loader
    offers bfloat16 by default and nf4 double-quantised under `--load-in-4bit`, and that nothing
    recorded which one produced a number. A baseline taken on a rented card in bf16 and a candidate
    taken in 4-bit because that is the only way it fits on a 6 GB card are not the same
    measurement: the nf4 round-trip alone moves per-token likelihood by an amount comparable to
    what an ablation costs, and a gate that cannot see the difference attributes all of it to the
    edit.

    Lived in `coherence` until 2026-09-25, when the other four writers needed it too.

    `getattr` rather than attribute access because a fake or a wrapped model may not carry a dtype,
    and an absent dtype should read as unknown rather than crash a measurement already paid for.
    """
    if load_in_4bit:
        return "nf4"
    dtype = getattr(model, "dtype", None)
    return str(dtype).removeprefix("torch.") if dtype is not None else "unknown"


def prompt_format_of(tok) -> str:
    """Which prompt format produced a number: the chat template's name, or `raw`.

    `raw` is a claim rather than an absence. A model scored on bare prompts and one scored through
    its instruction format refuse at different rates, and three copies of this renderer drifted
    once and a table was published across the gap. The field says which of the two happened.
    """
    return str(getattr(tok, "senbon_chat_template", None) or "raw")


def partition_of(skip, recorded_skip=None, *, verified=None) -> str:
    """Which rows of the input a number was taken on, by name where that can be established.

    Three answers and they are not interchangeable:

    - `measure`, when the run skipped exactly the rows a track's manifest says the fitting and
      search partitions consumed. This is the only partition a published number may come from.
    - `all-rows`, when nothing was skipped. That set contains the rows the surgery was fitted on,
      so the figure is in-sample and must never compare equal to a held-out one.
    - `rows-from-N`, when rows were skipped and no manifest confirms where the boundary was. It
      carries the skip so two identical unverified runs still compare, and it never equals
      `measure`, so an unverified boundary cannot be compared against a verified one.

    `verified` is the caller stating outright whether a manifest confirmed the boundary, which is
    what `track.resolve_skip_for_arm` returns. Passing it is better than passing `recorded_skip`,
    because a caller that read the boundary FROM the manifest has nothing separate to compare it
    against and would otherwise look unverified.
    """
    skip = int(skip or 0)
    if skip == 0:
        return ALL_ROWS
    if verified if verified is not None else (
            recorded_skip is not None and int(recorded_skip) == skip):
        return MEASURE
    return f"{UNVERIFIED_PREFIX}{skip}"


def pinned(*, prompts, model=None, tok=None, load_in_4bit=False, skip=0, recorded_skip=None,
           verified=None, input_digest=None, partition=None, prompt_format=None, precision=None):
    """The five pinned fields a writer passes straight into `measurement.stamp`, plus the version.

    Every derived value can be overridden, because two of the five writers genuinely know better
    than the derivation does: `capability` has a digest of its exam items, and `coherence` reads a
    fixed passage rather than a corpus. An override is the writer saying what it knows; the
    derivation is for the writers that would otherwise say nothing.
    """
    from ._version import __version__
    return {
        "input_digest": input_digest if input_digest is not None else input_digest_of(prompts),
        "partition": partition if partition is not None else partition_of(
            skip, recorded_skip, verified=verified),
        "prompt_format": prompt_format if prompt_format is not None else prompt_format_of(tok),
        "precision": precision if precision is not None else model_precision(model, load_in_4bit),
        "tool_version": __version__,
    }
