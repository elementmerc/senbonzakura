#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Build an evaluation track, and refuse to write one that cannot be trusted.

What a track is
---------------
Three datasets the rest of the tool reads by hardcoded name: `bad_ds` (harmful
prompts the refusal directions are fitted from), `good_ds` (harmless prompts, used
for fitting, for the KL reference and for the compass's harmless arm) and
`bad_eval_ds` (harmful prompts the search scores trials on and the compass measures).

Why it refuses rather than warns
--------------------------------
The corpus this project ran on for months had every one of its 200 harmful eval
prompts inside the 4,918-row fitting set, and 196 of 197 on the harmless side. Every
refusal rate measured through it scored a model on the exact prompts its direction was
fitted from, so those numbers described memorisation rather than generalisation, and
nothing anywhere said so. A checker that warns gets ignored on a tired evening; one
that will not write the file does not.

So the checks run BEFORE anything is written, and a failure produces no output at all.

The partition
-------------
Prompts are split by recorded index into three disjoint parts per side:

    fit      the directions are extracted from these
    search   the Optuna objective scores trials on these
    measure  the published number comes from these, and nothing else touches them

"Held out" then means a property of the file rather than an arithmetic convention
about how many rows to skip, and `track.json` records where the boundaries fell so a
reader a year later can check rather than trust.

Counts only
-----------
Nothing here prints a prompt. Every diagnostic is a number, because the inputs are
harmful text and the output of a build is meant to be safe to paste into a terminal,
a log, or an issue.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from itertools import zip_longest
from pathlib import Path

from . import dataset  # every accepted way of saying "the prompts are here"
from .crashsafe import atomic_write, provenance

# Rows shorter than this after normalisation are dropped as noise rather than prompts.
# Four is deliberate and low: "What is 2+2?" is a legitimate harmless prompt at twelve
# characters, and a threshold set by intuition rather than evidence throws away real data.
MIN_PROMPT_CHARS = 4

# Rows at or below this length are kept but reported, because a corpus with many very
# short prompts is usually a parsing accident upstream and the count is the tell.
SHORT_PROMPT_CHARS = 12

# The two sides may differ in size by at most this fraction. A lopsided contrast makes
# the difference-of-means partly a measure of which side had more rows.
MAX_BALANCE_SKEW = 0.10

# ...but not by fewer than this many rows, whatever the fraction says. Allocation deals
# whole requests, and a request can be several rows, so a difference smaller than a
# request group is granularity rather than imbalance. Without this floor a five-against-
# four split reads as "20% skew" and a correct track gets refused for being small.
BALANCE_MIN_ROWS = 8

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """The key two prompts are considered the same under.

    Unicode-normalised, case-folded, whitespace-collapsed, trailing punctuation removed.
    Deliberately aggressive: "How do I pick a lock?" and "how do i pick a lock" are the
    same prompt for the purpose of keeping an eval set disjoint from a fitting set, and
    treating them as different is how leakage survives a naive `sort -u`.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().strip()
    collapsed = _WHITESPACE.sub(" ", folded)
    return collapsed.rstrip(".?!,;: ")


def read_prompts(path) -> list[str]:
    """The prompts at `path`, in whatever shape they arrive in.

    A plain file is still read line by line and verbatim, because that is the format this
    builder documents and a corpus should not change under a user who did nothing. Anything
    else, a CSV with a `goal` column, a Hub id, a DatasetDict, goes through the shared resolver
    so `senbonzakura track` accepts exactly what every other command accepts.
    """
    p = Path(path)
    # A missing file that plainly IS meant to be a file keeps the direct error. Routing it to the
    # resolver would answer a typo in a filename with a paragraph about Hub dataset ids.
    if (not p.exists() and p.suffix and p.suffix.lower() not in dataset.TABLE_SUFFIXES
            and not dataset.looks_like_hub_id(str(path))):
        raise SystemExit(f"could not read {p}: no such file")
    if p.exists() and p.is_file() and p.suffix.lower() not in dataset.TABLE_SUFFIXES:
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise SystemExit(f"could not read {p}: {e}") from e
        return [line.strip() for line in raw.splitlines()]
    try:
        return dataset.resolve(str(path), what="prompt source")
    except dataset.DatasetError as e:
        raise SystemExit(str(e)) from e


def dedupe(rows: list[str]) -> tuple[list[str], dict[str, int]]:
    """Drop blanks, too-short rows and repeats. First occurrence wins, verbatim.

    Returns the kept rows and a count-only report. The original text is what gets
    written: normalisation decides *identity*, never content, because a corpus that has
    been silently lower-cased is a different corpus.
    """
    kept: list[str] = []
    seen: set[str] = set()
    stats = {"blank": 0, "too_short": 0, "duplicate": 0, "short_but_kept": 0}
    for row in rows:
        key = normalise(row)
        if not key:
            stats["blank"] += 1
            continue
        if len(key) < MIN_PROMPT_CHARS:
            stats["too_short"] += 1
            continue
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        if len(key) <= SHORT_PROMPT_CHARS:
            stats["short_but_kept"] += 1
        kept.append(row)
    return kept, stats


# A leading phrase has to be at least this many words, be shared by at least this many
# rows, and leave at least this many words behind, before it counts as a template rather
# than a coincidence of phrasing. Corpora built from templates are the normal case here:
# the one this project runs on crosses seven templates with every seed.
TEMPLATE_MIN_WORDS = 4
TEMPLATE_MAX_WORDS = 10
TEMPLATE_MIN_VARIANTS = 3
TEMPLATE_MIN_SUPPORT = 0.002
REQUEST_MIN_WORDS = 3


def discover_templates(rows: list[str]) -> list[str]:
    """Leading phrases shared by enough DIFFERENT requests to be a template, longest first.

    Derived from the corpus rather than hardcoded, so this works on anyone's data and
    cannot go stale against ours.

    Distinct remainders, not row counts, is what makes a template a template: a phrase
    followed by many different things is a frame, and one followed by the same thing
    repeatedly is just a common sentence. Counting rows missed a template used only four
    times, which let a leak through the check that exists to catch it.

    Over-stripping is close to harmless here and worth understanding: removing a shared
    prefix cannot merge two requests, because their remainders differ by construction.
    The only risk is a remainder short enough to collide with an unrelated prompt, which
    the minimum-remainder rule bounds.
    """
    variants: dict[str, set[str]] = {}
    floor = max(TEMPLATE_MIN_VARIANTS, int(len(rows) * TEMPLATE_MIN_SUPPORT))
    for row in rows:
        words = normalise(row).split(" ")
        for n in range(TEMPLATE_MIN_WORDS, min(TEMPLATE_MAX_WORDS, len(words) - REQUEST_MIN_WORDS) + 1):
            variants.setdefault(" ".join(words[:n]), set()).add(" ".join(words[n:]))
    return sorted((p for p, seen in variants.items() if len(seen) >= floor),
                  key=lambda p: (-len(p.split(" ")), p))


def request_key(text: str, templates: list[str]) -> str:
    """The underlying request, with any template prefix removed.

    This is the unit that must not be split across partitions. Seven templates share one
    seed, so "Explain in detail how to X" and "Write a step by step guide to X" are
    different prompts and the same request: on the project's own corpus, 120 of 200
    harmful eval rows had their request in the training set under another template, while
    a whole-prompt comparison reported zero overlap.
    """
    key = normalise(text)
    for tpl in templates:              # longest first, so the most specific one wins
        if key.startswith(tpl + " "):
            return key[len(tpl):].strip()
    return key


def group_by_request(rows: list[str]) -> dict[str, list[str]]:
    """Every row that expresses the same request, gathered under it."""
    templates = discover_templates(rows)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(request_key(row, templates), []).append(row)
    return groups


def read_labels(path):
    """A `label<TAB>prompt` file mapping prompts to strata. Returns {normalised prompt: label}.

    Optional, and deliberately external. The labels for this project's corpus are a
    taxonomy of harmful request categories, which belongs with the data rather than in a
    tool anyone can install; the mechanism here is generic and carries none of it.
    """
    out = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise SystemExit(f"could not read --labels {path}: {e}") from e
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise SystemExit(f"{path}:{lineno} is not `label<TAB>prompt`")
        label, prompt = line.split("\t", 1)
        out[normalise(prompt)] = label.strip()
    if not out:
        raise SystemExit(f"--labels {path} is empty")
    return out


UNLABELLED = "(unlabelled)"


def partition(rows: list[str], fit: int, search: int, labels=None) -> dict[str, list[str]]:
    """Split into fit / search / measure, whole requests at a time.

    Two properties, and the second was learned the hard way. Partitions are disjoint by
    construction; and every template variant of one request lands in the SAME partition,
    because splitting them puts the same question on both sides of a held-out boundary
    while a prompt-level check reports it as clean.

    Allocation is deterministic and spreads topics: requests are walked in sorted order
    and each goes to whichever partition has the largest shortfall **as a fraction of its
    own target**. The fraction matters. Comparing absolute shortfalls looks equivalent and
    is not: measure's target dwarfs the others, so it wins every comparison until it is
    nearly full, and fit and search end up drawn from the tail of the sorted order. On the
    real corpus that left four of nine harmful axes with no rows at all in the partition
    the search selects on. Proportional shortfall fills all three at the same rate, so each
    samples the whole corpus.

    Sizes are therefore approximate, since a request cannot be divided.
    """
    if fit < 0 or search < 0:
        raise SystemExit("fit and search sizes cannot be negative")
    if fit + search >= len(rows):
        raise SystemExit(
            f"fit {fit} + search {search} leaves nothing to measure on out of {len(rows)} rows. "
            f"The measure partition is the only one a published number may come from, so a "
            f"track without it is not worth writing.")

    groups = group_by_request(rows)
    measure = len(rows) - fit - search
    targets = {"fit": fit, "search": search, "measure": measure}
    out: dict[str, list[str]] = {"fit": [], "search": [], "measure": []}
    order = list(out)

    # With labels, allocate WITHIN each stratum, so every label that has enough requests
    # appears in all three partitions. Without them, the sorted request order is the only
    # stratification available: it spreads a partition across the corpus, but in a small
    # partition it cannot guarantee any particular category is present. On the real corpus
    # that difference decided whether the search selected against four of nine axes or all
    # of them, which is the whole reason labels are worth supplying.
    strata: dict[str, list[str]] = {}
    for key in sorted(groups):
        label = UNLABELLED
        if labels:
            found = {labels.get(normalise(r)) for r in groups[key]} - {None}
            if found:
                label = min(found)
        strata.setdefault(label, []).append(key)

    # Two phases, because interleaving alone cannot guarantee coverage. Phase one seeds
    # every partition with one request from every stratum, which is what makes "each
    # partition represents the corpus" a property rather than a hope. Phase two fills the
    # rest against the GLOBAL target, so the totals stay honest.
    #
    # The alternative, allocating each stratum against its own quota, was tried and
    # overshot: the rounding error compounds with the number of strata, and twelve of them
    # put the fit partition 15% over target, which the balance check then refused.
    def shortfall(p):
        return (targets[p] - len(out[p])) / max(1, targets[p])

    # Measure first, deliberately. A stratum with only one or two requests is exhausted by
    # seeding, and whichever partition comes last gets none of it: with fit first, two
    # strata ended up with ZERO rows in measure, so the published number covered neither
    # while every other check passed. The published arm gets first claim on every stratum.
    seed_order = ("measure", "fit", "search")
    seeded: set[str] = set()
    for label in sorted(strata):
        for name in seed_order:
            for key in strata[label]:
                if key in seeded:
                    continue
                if len(out[name]) >= targets[name]:
                    break                      # no room; the fill pass will handle it
                out[name].extend(groups[key])
                seeded.add(key)
                break

    # Interleaved, not stratum by stratum. Walking strata in order let whichever ones came
    # first absorb the small partitions' remaining room: the search partition came out with
    # 22 requests from one axis and 1 from another, so a configuration would have been
    # selected mostly against surveillance. Interleaving spreads the remainder too.
    remaining = [[k for k in strata[label] if k not in seeded] for label in sorted(strata)]
    for key in [k for tier in zip_longest(*remaining) for k in tier if k is not None]:
        # Largest shortfall as a FRACTION of target, ties broken by a fixed order so two
        # runs on the same input produce the same split.
        name = max(order, key=lambda p: (shortfall(p), -order.index(p)))
        out[name].extend(groups[key])

    return out


def check(harmful: dict[str, list[str]], harmless: dict[str, list[str]], labels=None) -> list[str]:
    """Every reason this track must not be written. Empty means it may be.

    Named separately from the build so a track can be audited without rebuilding it,
    and so each failure can be tested on its own.
    """
    failures: list[str] = []
    sides = {"harmful": harmful, "harmless": harmless}

    for name, side in sides.items():
        for part, rows in side.items():
            if not rows:
                failures.append(f"the {name} {part} partition is empty")

        # The whole point: what is measured must not be what was fitted or selected on.
        measure = {normalise(r) for r in side["measure"]}
        for other in ("fit", "search"):
            shared = measure & {normalise(r) for r in side[other]}
            if shared:
                failures.append(
                    f"{len(shared)} of {len(measure)} {name} measure prompts also appear in "
                    f"{name} {other}, so a number from them measures memorisation")

        # And the same question asked a different way. Templates are discovered across the
        # whole side, so a request is identified the same way wherever it sits. This check
        # exists because the prompt-level one above passed a corpus in which 120 of 200
        # harmful eval rows were template variants of training rows, reporting zero overlap.
        flat_rows = [r for part_rows in side.values() for r in part_rows]
        templates = discover_templates(flat_rows)
        requests = {part: {request_key(r, templates) for r in part_rows}
                    for part, part_rows in side.items()}
        for other in ("fit", "search"):
            shared = requests["measure"] & requests[other]
            if shared:
                failures.append(
                    f"{len(shared)} of {len(requests['measure'])} {name} measure REQUESTS also "
                    f"appear in {name} {other} under a different template, so the prompts differ "
                    f"and the question does not")

        flat = [normalise(r) for part in side.values() for r in part]
        if len(flat) != len(set(flat)):
            failures.append(f"the {name} side has {len(flat) - len(set(flat))} duplicate prompts across partitions")

        # A stratum the published arm never sees. The mirror image of leakage: not a claim
        # that is too good, but a claim narrower than it appears. Two strata were entirely
        # absent from measure while every other check passed.
        if labels:
            in_side, in_measure = set(), set()
            for part, part_rows in side.items():
                for r in part_rows:
                    label = labels.get(normalise(r))
                    if label:
                        in_side.add(label)
                        if part == "measure":
                            in_measure.add(label)
            missing = in_side - in_measure
            if missing:
                failures.append(
                    f"{len(missing)} {name} strata have no rows in measure, so a number from it "
                    f"does not cover them: {sorted(missing)}")

    harmful_keys = {normalise(r) for part in harmful.values() for r in part}
    harmless_keys = {normalise(r) for part in harmless.values() for r in part}
    both = harmful_keys & harmless_keys
    if both:
        failures.append(f"{len(both)} prompts are labelled both harmful and harmless")

    for part in ("fit", "search", "measure"):
        a, b = len(harmful[part]), len(harmless[part])
        if a and b:
            skew = abs(a - b) / max(a, b)
            if skew > MAX_BALANCE_SKEW and abs(a - b) > BALANCE_MIN_ROWS:
                failures.append(
                    f"the {part} partitions differ in size by {skew:.0%} ({a} harmful against "
                    f"{b} harmless), over the {MAX_BALANCE_SKEW:.0%} limit, so the contrast "
                    f"partly measures which side had more rows")
    return failures


def manifest(harmful: dict[str, list[str]], harmless: dict[str, list[str]], sources: dict[str, str]) -> dict:
    """What a reader needs to check the split rather than trust it.

    The skip values are recorded because they are what `senbonzakura.margin` must be
    given for its arms to be the measure partitions. Deriving them by hand is how a
    published number ends up describing the selection set.
    """
    return {
        "schema": "senbonzakura-track/1",
        "sources": sources,
        "counts": {
            "harmful": {k: len(v) for k, v in harmful.items()},
            "harmless": {k: len(v) for k, v in harmless.items()},
        },
        # bad_eval_ds is search then measure, so skipping the search part lands on measure.
        "skip_harmful": len(harmful["search"]),
        "skip_harmless": len(harmless["fit"]) + len(harmless["search"]),
        "n_harmful": len(harmful["measure"]),
        "n_harmless": len(harmless["measure"]),
    }


#: Manifest schema versions this build understands. A manifest with no `schema` key is
#: treated as the first version, because hand-written ones predate the field.
KNOWN_SCHEMAS = frozenset({"senbonzakura-track/1"})


def read_manifest(track_dir):
    """The recorded partition boundaries of a track, or None if it has none.

    None is not a failure: a hand-built track predates the builder and still has to run.
    It means the boundaries are unknown, which is exactly why nothing can be checked
    against them.

    A manifest whose `schema` is one this build does not know is a different matter and
    raises. Until 2026-08-03 the field was written and never read, which is worse than not
    having one: it looks like a compatibility guarantee and is not. When a second version
    exists, an older build reading it would take the fields it recognised, ignore whatever
    changed, and slice the datasets confidently at the wrong offsets. Every number
    downstream would come from the wrong rows and nothing would say so.

    `default` resolves to the bundled track, or the boundary check would silently switch off for
    the one corpus most users will run: the literal path "default/track.json" does not exist, and
    a missing manifest is a legal state meaning "boundaries unknown". A track that HAS recorded
    boundaries must never be read as though it had none.
    """
    if str(track_dir) == dataset.BUNDLED_ALIAS:
        from . import bundled
        track_dir = bundled.ensure()
    try:
        m = json.loads((Path(track_dir) / "track.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # A manifest is untrusted input: it is a file on disk that this process did not write,
    # and it decides which rows every published number is measured on. Anything that is not
    # the shape we understand is "no manifest", not "a manifest to interpret loosely".
    # `"counts" in m` alone let a null counts through, and it then reached a `.get` on None
    # deep inside the boundary check; a list-valued schema reached a frozenset membership
    # test and raised an unhashable-type TypeError instead of the refusal below.
    if not (isinstance(m, dict) and isinstance(m.get("counts"), dict)):
        return None
    schema = m.get("schema", "senbonzakura-track/1")
    if not isinstance(schema, str) or schema not in KNOWN_SCHEMAS:
        raise SystemExit(
            f"{track_dir}/track.json declares schema {schema!r}, which this build of "
            f"senbonzakura does not understand (it knows {sorted(KNOWN_SCHEMAS)}).\n"
            f"This track was written by a newer version. Upgrade senbonzakura rather than "
            f"running against it: the partition boundaries are read from this file, so "
            f"guessing at an unknown layout would slice the datasets at the wrong offsets "
            f"and every number would come from the wrong rows."
        )
    return m


def flag_violations(m, *, eval_refusal=0, eval_refusal_final=0, dir_prompts=0, eval_kl=0):
    """Ways a run's flags would reach past the boundaries the track records.

    The track makes "held out" a property of the files. That property survives only if the
    consumer respects it, and the consumer reads the HEAD of each dataset by count: the
    search takes the first `--eval-refusal-final` rows of `bad_eval_ds`, which is the
    search partition followed immediately by the measure partition. Ask for more rows than
    the search partition holds and the search starts selecting trials on the rows the
    published number comes from, with nothing to say so. That is the original defect
    reappearing through a flag rather than through a file.
    """
    counts = m.get("counts") or {}
    bad, good = counts.get("harmful") or {}, counts.get("harmless") or {}
    out = []
    search = bad.get("search")
    if search is not None:
        for flag, value in (("--eval-refusal", eval_refusal), ("--eval-refusal-final", eval_refusal_final)):
            if value > search:
                out.append(
                    f"{flag} {value} is more than the {search} harmful rows this track holds for "
                    f"selection, so the search would score trials on {value - search} of the rows "
                    f"the published number comes from. Lower it to {search} or rebuild the track "
                    f"with a larger --search")
    fit, gsearch = good.get("fit"), good.get("search")
    if None not in (fit, gsearch) and dir_prompts + eval_kl > fit + gsearch:
        out.append(
            f"--dir-prompts {dir_prompts} plus --eval-kl {eval_kl} reads "
            f"{dir_prompts + eval_kl} harmless rows, past the {fit + gsearch} this track keeps "
            f"aside, so the KL reference would be measured on rows the compass reports on")
    return out


def _save(rows: list[str], path: Path) -> None:
    from datasets import Dataset
    Dataset.from_dict({"text": rows}).save_to_disk(str(path))


def write_track(out: Path, harmful: dict[str, list[str]], harmless: dict[str, list[str]],
                sources: dict[str, str], log=print) -> dict:
    """Write the three datasets and the manifest, atomically, backing up once.

    Atomic because a corpus half-replaced by an interrupted run is worse than one not
    replaced at all: the next run would read it and say nothing. The backup is taken
    once and never overwritten, so re-running cannot destroy the pristine copy, which
    is the property that makes the whole thing safe to re-run.
    """
    staging = out.with_name(out.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # bad_ds is the fitting partition. bad_eval_ds is search then measure, in that order,
    # so the recorded skip_harmful lands exactly on the measure rows.
    _save(harmful["fit"], staging / "bad_ds")
    _save(harmful["search"] + harmful["measure"], staging / "bad_eval_ds")
    _save(harmless["fit"] + harmless["search"] + harmless["measure"], staging / "good_ds")
    m = manifest(harmful, harmless, sources)
    (staging / "track.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")

    if out.exists():
        backup = out.with_name(out.name + ".pre-build")
        if backup.exists():
            log(f"  keeping the existing backup at {backup} untouched")
        else:
            shutil.move(str(out), str(backup))
            log(f"  backed up the previous track to {backup}")
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return m


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura.track",
        description="Build an evaluation track with a fit / search / measure split that is "
                    "checked before it is written.")
    # Not required, because --audit reads an existing track and has no use for them; making
    # them mandatory forced anyone checking a track to invent two paths that are never read.
    ap.add_argument("--harmful", default="", help="text file of harmful prompts, one per line")
    ap.add_argument("--harmless", default="", help="text file of harmless prompts, one per line")
    ap.add_argument("--out", required=True, help="track directory to create")
    ap.add_argument("--fit", type=int, default=256,
                    help="prompts per side the directions are extracted from (default: the "
                         "auto presets' --dir-prompts)")
    ap.add_argument("--search", type=int, default=128,
                    help="prompts per side the search scores trials on (default: the largest "
                         "--eval-refusal-final any auto preset uses)")
    ap.add_argument("--labels", default="",
                    help="optional `label<TAB>prompt` file. With it, every label that has "
                         "enough distinct requests is guaranteed a share of all three "
                         "partitions; without it, stratification is by sorted request order, "
                         "which spreads a partition across the corpus but cannot guarantee a "
                         "small one contains any particular category")
    ap.add_argument("--audit", action="store_true",
                    help="run the checks against an existing track and write nothing")
    ap.add_argument("--contamination", default="",
                    help="text file of an external benchmark's prompts, one per line. Reports "
                         "how many of its requests this track has already fitted or searched "
                         "on, which is what decides whether a figure on that benchmark would "
                         "be in-sample. Reads the track and writes nothing")
    ap.add_argument("--contamination-name", default="",
                    help="what to call the external set in the report (default: the filename)")
    ap.add_argument("--contamination-report", default="",
                    help="also write the counts to this path as JSON, so a published claim "
                         "about the overlap traces to a file rather than to a terminal")
    ap.add_argument("--fail-on-contamination", action="store_true",
                    help="exit non-zero when any of the external set was fitted or searched "
                         "on, so the check can gate a run rather than only inform one")
    return ap


def load_partitions(track: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The three partitions per side, sliced back out of an existing track.

    The boundaries are RECORDED, not derivable: `bad_eval_ds` is search followed by measure
    in one file, and only `track.json` says where one ends. So a manifest that disagrees
    with the files beside it does not make the slicing approximate, it makes every row land
    in the wrong partition, and anything reading them answers a question about a track that
    does not exist. Refuse rather than report.

    Extracted so the audit and the contamination check cannot drift apart. Two copies of
    boundary arithmetic is the shape of this codebase's recurring defect: three copies of
    the prompt renderer disagreed and put the compass's read-out on the wrong token.
    """
    from datasets import load_from_disk
    try:
        m = json.loads((track / "track.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"{track} has no readable track.json, so its split cannot be checked: {e}") from e

    counts = m["counts"]
    bad_fit = [r["text"] for r in load_from_disk(str(track / "bad_ds"))]
    bad_eval = [r["text"] for r in load_from_disk(str(track / "bad_eval_ds"))]
    good = [r["text"] for r in load_from_disk(str(track / "good_ds"))]
    hs, gf, gs = counts["harmful"]["search"], counts["harmless"]["fit"], counts["harmless"]["search"]

    expected = {
        "bad_ds": (len(bad_fit), counts["harmful"]["fit"]),
        "bad_eval_ds": (len(bad_eval), counts["harmful"]["search"] + counts["harmful"]["measure"]),
        "good_ds": (len(good), gf + gs + counts["harmless"]["measure"]),
    }
    wrong = [f"{n}: {have} rows on disk, {want} in track.json"
             for n, (have, want) in expected.items() if have != want]
    if wrong:
        raise SystemExit(
            f"{track}/track.json does not describe the datasets beside it, so the recorded "
            f"partition boundaries cannot be trusted and neither could an audit using them: "
            + "; ".join(wrong))

    return (
        {"fit": bad_fit, "search": bad_eval[:hs], "measure": bad_eval[hs:]},
        {"fit": good[:gf], "search": good[gf:gf + gs], "measure": good[gf + gs:]},
    )


def contamination(external: list[str], partitions: dict[str, list[str]], name="external") -> dict:
    """How much of an external benchmark this track has already seen, and where.

    The question a published number depends on. Every abliteration tool quotes AdvBench, so
    a figure on it is what lets a reader place this project against the literature at all,
    and roughly 430 rows of this corpus's harmful side ARE AdvBench, reached through
    `mlabonne/harmful_behaviors`. If any of them sit in `fit` or `search`, the model was
    tuned on the benchmark it is being scored against and the figure is in-sample: exactly
    the defect this project is building a checker to find in other people's evaluations.

    **Matched by request, never by string.** Seven templates share one seed here, and on
    this project's own corpus a whole-prompt comparison once reported zero overlap while
    120 of 200 eval rows had their request in the training set wearing another template.
    Templates are discovered over the UNION of both sides, because a frame stripped from
    one and left on the other would make identical requests look different, which fails in
    the direction that hides a leak.

    Returns counts only. The inputs are harmful text and nothing here prints a row.
    """
    ext_rows = [r for r in external if r.strip()]
    track_rows = [r for part in partitions.values() for r in part]
    templates = discover_templates(ext_rows + track_rows)

    ext_keys = {request_key(r, templates) for r in ext_rows}
    where = {part: {request_key(r, templates) for r in rows} for part, rows in partitions.items()}

    seen = {part: len(ext_keys & keys) for part, keys in where.items()}
    fitting = ext_keys & (where.get("fit", set()) | where.get("search", set()))
    measure_only = (ext_keys & where.get("measure", set())) - fitting
    absent = ext_keys - set().union(*where.values()) if where else ext_keys

    return {
        "external": name,
        "external_rows": len(ext_rows),
        "external_requests": len(ext_keys),
        "templates_discovered": len(templates),
        "in_fit": seen.get("fit", 0),
        "in_search": seen.get("search", 0),
        "in_measure": seen.get("measure", 0),
        # The number the decision turns on. Anything here was fitted or searched on, so it
        # cannot appear in a published figure about this benchmark.
        "in_fitting_side": len(fitting),
        # What a clean figure could be computed over today, without changing anything.
        "publishable_requests": len(measure_only),
        # Rows of the benchmark this corpus has never held at all. They are already a
        # held-out external slice and cost nothing to start using.
        "absent_from_track": len(absent),
        "verdict": "contaminated" if fitting else "clean",
    }


def format_contamination(r: dict) -> list[str]:
    """The report, in the words a reader needs rather than a dump of the dict."""
    lines = [
        f"contamination: {r['external']} against this track",
        (f"  {r['external_rows']} rows, {r['external_requests']} distinct requests "
         f"(templates discovered: {r['templates_discovered']})"),
        (f"  in fit {r['in_fit']}, in search {r['in_search']}, in measure {r['in_measure']}, "
         f"absent {r['absent_from_track']}"),
    ]
    if r["verdict"] == "contaminated":
        lines += [
            f"  CONTAMINATED: {r['in_fitting_side']} of its requests were fitted or searched on.",
            ("  A figure on this benchmark would be in-sample. Either drop those requests and "
             "rebuild the split, or report only the clean part and say so in those words."),
        ]
    else:
        lines.append("  CLEAN: no request of this benchmark was fitted or searched on.")
    lines.append(
        f"  publishable today: {r['publishable_requests']} requests from the measure partition, "
        f"plus {r['absent_from_track']} this track has never held.")
    return lines


def audit(track: Path, labels=None) -> list[str]:
    """Re-run the checks on a track that already exists, using its recorded boundaries.

    `labels` matters more than it looks. The recorded counts are the ONLY thing that says
    where a partition boundary falls in an existing track, and the strata check is the one
    that catches the mirror image of leakage: an arm narrower than the number claims. Audit
    without labels cannot run it, so a hand-built track passes an audit that never asked
    the question.
    """
    harmful, harmless = load_partitions(track)
    return check(harmful, harmless, labels)


#: What a promotion stamp is called, inside the track it describes.
PROMOTED = "PROMOTED.json"


def dataset_digest(path):
    """A stable digest of a dataset directory's contents, for the promotion stamp.

    Over the FILE BYTES, sorted by name, not over the rows read back through `datasets`. Reading
    rows would make the digest depend on the library version that read them, and the question a
    stamp answers is "are these the same bytes I checked", which is a question about the disk.
    """
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def promote(track, *, labels=None, force=False, now=None, log=print):
    """Re-verify a built track and stamp it as the one measurements may come from.

    PROMOTION IS A GATE, NOT A COPY. Nothing is moved and nothing is duplicated; what it produces
    is a statement that on this date, this track passed the same checks the builder applies, with
    these row counts and these digests.

    That is the whole point. `senbon-track-35axis-clean` was the corpus this project measured on
    for months, and 189 of its 200 harmful eval rows sat inside its own fitting set. It had been
    built before the checks existed, so nothing ever re-asked the question, and every refusal rate
    taken through it described memorisation. A stamp that has to be earned is what makes "the
    promoted track" mean something other than "the track we happen to be using".

    It re-runs the checks rather than trusting the build, because the failure being guarded is a
    track that was fine when written and is not fine now: rebuilt in place, partially copied,
    or a manifest edited to make a flag violation go away.
    """
    track = Path(track)
    m = read_manifest(track)
    if m is None:
        raise SystemExit(
            f"{track} records no track.json, so where its partition boundaries fall is unknown "
            f"and nothing can be verified about it. A track built before the builder existed "
            f"cannot be promoted; rebuild it.")

    failures = audit(track, labels)
    if failures:
        log(f"PROMOTE_REFUSED {track}: this track is not one a published number may come from:")
        for f in failures:
            log(f"  {f}")
        if not force:
            raise SystemExit(1)
        # --force exists for a track whose only failure is one an operator has looked at and
        # accepted in writing. The stamp records that it was forced, so a reader of the artefact
        # can tell a clean promotion from an overridden one.
        log("  --force given: promoting anyway, and recording that it was forced")

    if labels is None:
        # Said out loud because an audit without labels cannot run the strata check, which is the
        # mirror image of leakage: an arm narrower than its count claims. Passing without having
        # asked is not the same as passing.
        log("  note: no --labels, so the per-stratum coverage check did not run")

    counts = (m.get("counts") or {})
    stamp = {
        "schema": "senbonzakura-track-promotion/1",
        "promoted_at": now or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "track": str(track.resolve()),
        "counts": counts,
        "skip_harmful": m.get("skip_harmful"),
        "skip_harmless": m.get("skip_harmless"),
        "forced": bool(failures) and force,
        "failures_accepted": failures if (failures and force) else [],
        "labels_checked": labels is not None,
        "digests": {name: dataset_digest(track / name)
                    for name in ("bad_ds", "bad_eval_ds", "good_ds")
                    if (track / name).is_dir()},
        "provenance": provenance(),
    }
    with atomic_write(track / PROMOTED) as f:
        json.dump(stamp, f, indent=2, sort_keys=True)
    log(f"TRACK_PROMOTED {track}")
    log(f"  harmful {counts.get('harmful')}  harmless {counts.get('harmless')}")
    log(f"  measure with: --skip-harmful {m.get('skip_harmful')} "
        f"--skip-harmless {m.get('skip_harmless')}")
    log(f"  stamp: {track / PROMOTED}")
    return stamp


def promotion_of(track):
    """The promotion stamp on a track, or None. None means unpromoted, which is a real state."""
    p = Path(track) / PROMOTED
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return doc if doc.get("schema", "").startswith("senbonzakura-track-promotion/") else None


def promotion_is_current(track):
    """Does the stamp still describe the bytes on disk? Returns (ok, reason).

    A stamp is a claim about a moment. Rebuilding a track in place leaves the stamp behind saying
    the old thing passed, which is worse than no stamp: it is a check that has stopped checking
    while still reading as green.
    """
    stamp = promotion_of(track)
    if stamp is None:
        return False, "not promoted"
    for name, recorded in (stamp.get("digests") or {}).items():
        d = Path(track) / name
        if not d.is_dir():
            return False, f"{name} is gone since promotion"
        if dataset_digest(d) != recorded:
            return False, (f"{name} has changed since it was promoted on "
                           f"{stamp.get('promoted_at', '?')}, so the stamp describes a track that "
                           f"is no longer here")
    return True, f"promoted {stamp.get('promoted_at', '?')}"


def main(argv=None):
    # Subcommands, added without disturbing the flat form. Every run spec and every README
    # example says `senbonzakura track --harmful X --out Y`, and a tool that renames its own
    # entry point breaks the record of what was already run.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("promote", "verify"):
        return _promote_main(argv)

    a = build_parser().parse_args(argv)
    out = Path(a.out)

    if a.contamination:
        ext_path = Path(a.contamination)
        harmful, _ = load_partitions(out)
        r = contamination(read_prompts(ext_path), harmful,
                          a.contamination_name or ext_path.stem)
        for line in format_contamination(r):
            print(line)
        if a.contamination_report:
            report = Path(a.contamination_report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
            print(f"  written to {report}")
        if a.fail_on_contamination and r["verdict"] == "contaminated":
            raise SystemExit(
                f"{r['in_fitting_side']} requests of {r['external']} were fitted or searched "
                f"on, so a figure on it from this track would be in-sample")
        return r

    if a.audit:
        failures = audit(out, read_labels(a.labels) if a.labels else None)
        if failures:
            print(f"TRACK_AUDIT_FAILED {out}", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            raise SystemExit(1)
        print(f"TRACK_AUDIT_OK {out}")
        return {}

    missing = [f for f, v in (("--harmful", a.harmful), ("--harmless", a.harmless)) if not v]
    if missing:
        raise SystemExit(f"building a track needs {' and '.join(missing)}")

    labels = read_labels(a.labels) if a.labels else None
    if labels:
        print(f"labels: {len(labels)} prompts across {len(set(labels.values()))} strata")
    sides = {}
    sources = {"labels": a.labels or None}
    for name, path in (("harmful", a.harmful), ("harmless", a.harmless)):
        rows, stats = dedupe(read_prompts(Path(path)))
        # Counts only, never content: these inputs are harmful text.
        print(f"{name}: {len(rows)} kept  (blank {stats['blank']}, too short "
              f"{stats['too_short']}, duplicate {stats['duplicate']}, "
              f"short but kept {stats['short_but_kept']})")
        sides[name] = partition(rows, a.fit, a.search, labels)
        sources[name] = str(path)

    failures = check(sides["harmful"], sides["harmless"], labels)
    if failures:
        print("TRACK_REFUSED: nothing was written, because this track would not be "
              "trustworthy:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)

    m = write_track(out, sides["harmful"], sides["harmless"], sources)
    print(f"TRACK_BUILT {out}  harmful {m['counts']['harmful']}  harmless {m['counts']['harmless']}")
    print(f"  measure with: --skip-harmful {m['skip_harmful']} --skip-harmless {m['skip_harmless']} "
          f"--n {min(m['n_harmful'], m['n_harmless'])}")
    return m


def _promote_main(argv):
    sub = argv[0]
    ap = argparse.ArgumentParser(
        prog=f"senbonzakura track {sub}",
        description=("Re-verify a track and stamp it as one measurements may come from."
                     if sub == "promote" else
                     "Re-verify a track and report, changing nothing."))
    ap.add_argument("track", help="the track directory")
    ap.add_argument("--labels", default="",
                    help="the axis-label TSV, so the per-stratum coverage check can run. Without "
                         "it that check is skipped and the output says so")
    if sub == "promote":
        ap.add_argument("--force", action="store_true",
                        help="promote a track that failed a check. The stamp records that it was "
                             "forced and which failures were accepted")
    a = ap.parse_args(argv[1:])
    labels = read_labels(a.labels) if a.labels else None

    if sub == "verify":
        _ok, why = promotion_is_current(a.track)
        failures = audit(Path(a.track), labels)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(f"promotion: {why}")
        if labels is None:
            print("  note: no --labels, so the per-stratum coverage check did not run")
        if failures:
            print(f"TRACK_VERIFY_FAILED {a.track}", file=sys.stderr)
            raise SystemExit(1)
        print(f"TRACK_VERIFY_OK {a.track}")
        return 0

    promote(a.track, labels=labels, force=a.force)
    return 0


if __name__ == "__main__":   # pragma: no cover
    main()
