#!/usr/bin/env python3
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
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

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


def read_prompts(path: Path) -> list[str]:
    """One prompt per line. Original text is preserved; only the key is normalised."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise SystemExit(f"could not read {path}: {e}") from e
    return [line.strip() for line in raw.splitlines()]


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

    for label in sorted(strata):
        filled = dict.fromkeys(order, 0)
        share = {p: targets[p] * sum(len(groups[k]) for k in strata[label]) / max(1, len(rows))
                 for p in order}
        for key in strata[label]:
            # Largest shortfall as a FRACTION of this stratum's share, ties broken by a
            # fixed order so two runs on the same input produce the same split.
            name = max(order, key=lambda p: ((share[p] - filled[p]) / max(1.0, share[p]),
                                             -order.index(p)))
            out[name].extend(groups[key])
            filled[name] += len(groups[key])
    return out


def check(harmful: dict[str, list[str]], harmless: dict[str, list[str]]) -> list[str]:
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
    ap.add_argument("--harmful", required=True, help="text file of harmful prompts, one per line")
    ap.add_argument("--harmless", required=True, help="text file of harmless prompts, one per line")
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
    return ap


def audit(track: Path) -> list[str]:
    """Re-run the checks on a track that already exists, using its recorded boundaries."""
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
    return check(
        {"fit": bad_fit, "search": bad_eval[:hs], "measure": bad_eval[hs:]},
        {"fit": good[:gf], "search": good[gf:gf + gs], "measure": good[gf + gs:]},
    )


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = Path(a.out)

    if a.audit:
        failures = audit(out)
        if failures:
            print(f"TRACK_AUDIT_FAILED {out}", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            raise SystemExit(1)
        print(f"TRACK_AUDIT_OK {out}")
        return {}

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

    failures = check(sides["harmful"], sides["harmless"])
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


if __name__ == "__main__":   # pragma: no cover
    main()
