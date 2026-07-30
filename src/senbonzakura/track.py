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


def partition(rows: list[str], fit: int, search: int) -> dict[str, list[str]]:
    """Split into fit / search / measure by index, disjoint by construction.

    Everything after fit + search is measure, so the published arm gets whatever is
    left rather than a fixed slice: a track that grows should widen the interval it can
    support, not leave the extra rows unused.
    """
    if fit < 0 or search < 0:
        raise SystemExit("fit and search sizes cannot be negative")
    if fit + search >= len(rows):
        raise SystemExit(
            f"fit {fit} + search {search} leaves nothing to measure on out of {len(rows)} rows. "
            f"The measure partition is the only one a published number may come from, so a "
            f"track without it is not worth writing.")
    return {
        "fit": rows[:fit],
        "search": rows[fit:fit + search],
        "measure": rows[fit + search:],
    }


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
            if skew > MAX_BALANCE_SKEW:
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

    sides = {}
    sources = {}
    for name, path in (("harmful", a.harmful), ("harmless", a.harmless)):
        rows, stats = dedupe(read_prompts(Path(path)))
        # Counts only, never content: these inputs are harmful text.
        print(f"{name}: {len(rows)} kept  (blank {stats['blank']}, too short "
              f"{stats['too_short']}, duplicate {stats['duplicate']}, "
              f"short but kept {stats['short_but_kept']})")
        sides[name] = partition(rows, a.fit, a.search)
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
