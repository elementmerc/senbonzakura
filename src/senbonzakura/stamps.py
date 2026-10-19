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

from senbonzakura_check import measurement
from senbonzakura_check.measurement import normalise_seeds

#: The writer-facing name for "this method draws no samples", so a caller reads `stamps.NO_SEED`
#: beside `stamps.pinned` rather than reaching into the checker package.
NO_SEED = measurement.NO_SEED

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


def model_revision_of(model) -> str | None:
    """Which revision of a Hub model produced a number, or None when there is no such thing.

    WHY THIS WAS MISSING FOR THREE WEEKS. `evidence/README.md` has required "the model id and
    revision" of every committed result since the 2026-09-25 panel. Nothing ever recorded the
    revision, so the two artefacts in the tree were listed as exceptions in
    `tests/test_evidence_carries_its_provenance.py`, described as predating the rule. They did,
    and the tool had not caught up, so every NEW artefact landed in the same hole: the exemption
    list was absorbing a live gap rather than a historical one. Found 2026-10-11 when the guard
    refused a fresh artefact and the honest answer turned out not to be another exemption.

    AN ID IS NOT AN IDENTITY. `Qwen/Qwen3-1.7B` is a name on a repository somebody else can push
    to, so two runs a month apart can name the same model and have loaded different weights. This
    is the same argument `input_digest_of` makes about prompt paths and `track.revision_entry`
    makes about corpus names, and it has the same answer.

    Read off the config transformers already resolved, which is free and works offline: the loader
    sets `_commit_hash` from the cache when `HF_HUB_OFFLINE=1`, measured on the evidence run of
    2026-10-11 where it matched the snapshot directory exactly.

    NONE FOR A LOCAL DIRECTORY, deliberately, and the guard is then expected to refuse. A path on
    disk has no upstream revision, and the honest pin for one is a digest of its weights, which
    this project does not yet have and which is a design decision rather than a line of code: a
    hash of the index file alone would satisfy a presence check while recording nothing a reader
    could verify, which is exactly what `crashsafe.provenance` refuses to do for a corpus. So an
    artefact from a local checkpoint stays unpinned and says so by failing, rather than carrying a
    placeholder that reads as a receipt. `DEFERRED.md` carries the design question.
    """
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    # A local load leaves it unset or sets it to the path. Either way it is not a revision, and a
    # 40-character hex string is what one looks like.
    if not isinstance(revision, str) or len(revision) != 40:
        return None
    return revision if all(c in "0123456789abcdef" for c in revision.lower()) else None


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
    # The loader records how the weights came in; a model it loaded says so, and that outranks
    # the flag because a stored-quantised or GGUF model never saw `--load-in-4bit`.
    loaded = getattr(model, "senbon_load", None)
    if isinstance(loaded, dict) and loaded.get("precision"):
        return loaded["precision"]
    if load_in_4bit:
        return "nf4"
    dtype = getattr(model, "dtype", None)
    return str(dtype).removeprefix("torch.") if dtype is not None else "unknown"


def model_identity(model, given):
    """Which checkpoint a number came from: the string typed, the name it calls itself, and the pin.

    THREE FIELDS BECAUSE THEY ANSWER THREE QUESTIONS AND ONLY ONE OF THEM IS AN IDENTITY. `given`
    is the path or id as typed, which inside a container is a mount point and on a laptop is
    somebody else's disk. `id` is what the checkpoint calls itself, which is the part a stranger
    can look up. `revision` is the commit the Hub actually served, because a Hub id resolves to
    whatever is there on the day: a third party redoing this next year gets a different checkpoint
    under the same string, and without this field no artefact says so.

    ONE HOME, and it was nearly two. The abliteration record has derived these three inline since
    2026-09, correctly, and this is that derivation moved rather than copied: two mechanisms
    deriving one fact is the defect this project repeats most, and a second copy in a new command
    would have been the next instance of it.

    `getattr` throughout because a fake, a wrapped or a locally loaded model may carry none of
    these, and an absent revision must read as unknown rather than crash a measurement that has
    already been paid for.
    """
    config = getattr(model, "config", None)
    return {
        "given": given,
        "id": getattr(config, "_name_or_path", None),
        "revision": getattr(config, "_commit_hash", None),
    }


def prompt_format_of(tok) -> str:
    """Which prompt format produced a number: the chat template's name, or `raw`.

    `raw` is a claim rather than an absence. A model scored on bare prompts and one scored through
    its instruction format refuse at different rates, and three copies of this renderer drifted
    once and a table was published across the gap. The field says which of the two happened.

    THE DIGEST IS PART OF THE ANSWER, and leaving it out was the defect here. `ensure_chat_template`
    returns a mapping, `{"source": ..., "sha256": ..., "audit": ...}`, and this used to hand the
    whole mapping to `str()`. An artefact therefore recorded the literal text
    `{'source': 'tokenizer', 'sha256': '872be49dbb638044'}` as its prompt format, which is not a
    name, does not compare equal to anything a reader would write, and puts a Python repr in a
    published field.

    The source alone is not enough either, and that is why the digest is kept rather than dropped
    for tidiness: two tokenizers both answer "tokenizer" while rendering different templates, which
    is exactly the drift the paragraph above is about. `tokenizer:872be49dbb638044` distinguishes
    them; `tokenizer` does not.
    """
    raw = getattr(tok, "senbon_chat_template", None)
    if not raw:
        return "raw"
    if isinstance(raw, dict):
        source = raw.get("source") or "unknown"
        digest = raw.get("sha256")
        return f"{source}:{digest}" if digest else str(source)
    return str(raw)


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


#: What `partition` says for a run over the FIRST n rows of a corpus.
#:
#: A FOURTH NAME, ADDED 2026-10-05 BECAUSE THE OTHER THREE COULD NOT SAY IT. `partition_of` maps a
#: skip of zero to `all-rows`, which is true of a run that scored everything and false of one that
#: scored `--n 40` of a 313-row corpus: that run described the head of the corpus and the field
#: said it described the whole of it. The digests differ, so nothing would have compared the two
#: as equal; the field was simply wrong, which is worse on a provenance block than on a comment
#: because the whole point of it is being read later by somebody who cannot check.
HEAD_PREFIX = "first-"


def head_partition(n, total):
    """`all-rows` when the whole corpus was used, else a name for the head that was.

    Never collides with `ALL_ROWS`, `MEASURE` or an `UNVERIFIED_PREFIX` name, so a head slice can
    never compare equal to a held-out tail or to a verified measure partition.
    """
    n = int(n or 0)
    return ALL_ROWS if n in (0, int(total)) else f"{HEAD_PREFIX}{n}-rows"


#: What a document that INDEXES several measurements declares itself to be, rather than leaving a
#: reader to infer it from a shape.
#:
#: Q-116, taking option E of Q-54. Before this, the only way to tell a summary from a measurement
#: was "it has a stages dict and no metrics block", which is a guess about intent read off a shape:
#: a future artefact carrying both would be misread, and a reader had no field to check.
#:
#: HOISTED HERE FROM `measure.py` ON 2026-10-10, because the aggregate question has two askers.
#: Q-54 was raised precisely because `measure` and the head-to-head reporter reach it independently,
#: and the plan's own note says answering it twice differently is the defect. `measure` re-exports
#: these names, so nothing that already reads them had to move.
SUMMARY_KIND = "summary"

#: Why a summary carries no stamp of its own, in the words that go into the artefact.
#:
#: NAMED RATHER THAN OMITTED, which is the part of option E that does the work. An aggregate with
#: no stamp invites the reading "nobody stamped this"; one that says WHICH fields its measurements
#: disagree on is making a claim a reader can check against the files sitting beside it.
PINNED_MAY_DIFFER_DOC = (
    "the pinned fields on which this document's measurements legitimately differ. A summary has no "
    "stamp of its own BECAUSE of these: one value for a field several instruments measured "
    "differently would be a claim about the others that is not true. Each measurement's own file "
    "carries its own stamp, and the stamps recorded beside this hold them verbatim."
)


def pinned_blocks(doc, *, where):
    """One artefact's pinned fields, per metric slot, verbatim, or a marker saying why not.

    THE MARKER IS NOT A HOLE. An absent key and a key saying "this could not be read" are different
    claims and the second is the true one: dropping the entry would make a summary over four
    measurements describe three.
    """
    from . import baseline

    if not isinstance(doc, dict):
        return {"unstamped": f"{where} is not a document"}
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return {"unstamped": f"{where} carries no metrics block"}
    return {
        slot: {k: v for k, v in block.items() if k in baseline.PINNED}
        for slot, block in sorted(metrics.items()) if isinstance(block, dict)
    }


def read_pinned_blocks(path):
    """`pinned_blocks` for a file on disk, with an unreadable file reported rather than dropped.

    THE MARKER NAMES THE FILE AND NOT THE PATH. This goes into an artefact, and an artefact that
    carries an absolute path from whoever happened to run it is both a small disclosure and useless
    to the next reader, whose copy is somewhere else. The file sits beside the summary by
    construction, so its name is enough to find it.
    """
    import json
    from pathlib import Path

    target = Path(path)
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"unreadable": f"{type(e).__name__}: {e}"}
    return pinned_blocks(doc, where=target.name)


#: The two keys `pinned_blocks` uses to say it could not read a stamp, as opposed to a metric slot.
#: Named so a consumer of these tables does not have to spell them, and so adding a third cannot
#: leave one reader behind.
MARKER_KEYS = ("unreadable", "unstamped")


def pinned_fields_that_differ(stamps):
    """Which pinned fields take more than one value across a table of pinned blocks, sorted.

    Computed rather than typed, so a fifth measurement changes the answer without anybody having to
    remember a tuple. Sorted so two runs over the same inputs produce byte-identical output, which
    is the reproducibility rule: iteration order of a set must not reach a result file.
    """
    import json

    seen = {}
    for blocks in stamps.values():
        for slot, block in blocks.items():
            if slot in MARKER_KEYS:
                continue
            for field, value in block.items():
                seen.setdefault(field, set()).add(json.dumps(value, sort_keys=True))
    return sorted(field for field, values in seen.items() if len(values) > 1)


def pinned(*, prompts, seeds, model=None, tok=None, load_in_4bit=False, skip=0, recorded_skip=None,
           verified=None, input_digest=None, partition=None, prompt_format=None, precision=None):
    """The pinned fields a writer passes straight into `measurement.stamp`, plus the version.

    `seeds` is REQUIRED, with no default, and that is the point. The seed set an interval was drawn
    over is part of what the number is, and a derivation cannot know it: only the writer does. A
    default would let a writer that never thought about seeds stamp a figure the gate compares as if
    it were a figure from any other seed set. Pass the seeds the interval was drawn over, or
    `NO_SEED` when the method draws no samples (a Wilson interval, a deterministic probe).

    Every derived value can be overridden, because two of the five writers genuinely know better
    than the derivation does: `capability` has a digest of its exam items, and `coherence` reads a
    fixed passage rather than a corpus. An override is the writer saying what it knows; the
    derivation is for the writers that would otherwise say nothing.
    """
    from ._version import __version__
    return {
        "seeds": normalise_seeds(seeds),
        "input_digest": input_digest if input_digest is not None else input_digest_of(prompts),
        "partition": partition if partition is not None else partition_of(
            skip, recorded_skip, verified=verified),
        "prompt_format": prompt_format if prompt_format is not None else prompt_format_of(tok),
        "precision": precision if precision is not None else model_precision(model, load_in_4bit),
        "tool_version": __version__,
    }
