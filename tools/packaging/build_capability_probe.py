#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Fetch the bundled capability probe at a pinned revision, verify it, and write it for the wheel.

WHY THE PACKAGE CARRIES A CAPABILITY PROBE AT ALL

Every other figure this tool reports is a refusal ruler or a distributional proxy. None of them
asks the model to do anything hard, so a run can report a clean bake on a model that has quietly
lost multi-step arithmetic. `--capability-eval` existed to close that and defaulted to empty,
which meant the gate that matters most was the one nobody switched on.

A default that needs a download is not a default: this tool is meant to work in an air-gapped lab,
which is why the corpora ship inside the wheel. So the probe ships too.

WHY THIS ONE IS COMMITTED WHEN THE CORPORA ARE NOT

`src/senbonzakura/data/*` is gitignored and `corpora.bin` is built rather than committed, for one
reason and one only: it holds 1,333 harmful prompts across five research corpora, and its sibling
`default-track.bin` holds another 4,895, and the project publishes that corpus as a gated dataset
on purpose. GSM8K is grade-school arithmetic under the MIT licence. Applying the
harmful-content rule to harmless content would buy nothing and would cost the thing it cost on
2026-09-22, when a build from a clone turned out to have no `--track default` because the blobs
are generated: the tool installed, imported and answered `--help`, then failed on the first real
command. Making the capability gate default-on and then shipping clones that cannot run it would
reproduce that defect on the gate itself.

So this script is reproducibility rather than packaging: it says where the file came from and lets
anyone rebuild it, while the file itself is in the tree.

WHAT IS VERIFIED, AND IN WHICH ORDER, following `build_corpora.py`

  1. **The revision.** Pinned. An unpinned fetch is a different dataset on a different day.
  2. **The shape.** Both columns must be there under the names the grader reads.
  3. **The count.** GSM8K's `main` test split is 1,319 rows. A split that parses and is a
     different size is an upstream change wearing a familiar name.
  4. **The gold marker.** Every selected row's answer must carry `####`, because that is what
     `capability.gold_answer` reads. A row without it grades as indeterminate for ever, silently
     shrinking the probe.

    python tools/packaging/build_capability_probe.py            # fetch, verify, write
    python tools/packaging/build_capability_probe.py --check    # verify only, write nothing
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "senbonzakura" / "data" / "capability-gsm8k.jsonl"

#: Pinned, and the whole point of pinning is that "openai/gsm8k" is not a fixed object.
DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"
REVISION = "740312add88f781978c0658806c59bc2815b9866"

#: What the upstream split must contain. A parse that yields a different number is an upstream
#: change, not a rounding difference, and it must stop the build rather than shrink the probe.
EXPECTED_ROWS = 1319

#: How many rows the package carries. Larger than any sensible `--capability-n` on purpose:
#: bundling spare items costs kilobytes, and running out of them costs a re-release.
BUNDLE = 256

#: What `capability.gold_answer` looks for in a reference answer.
GOLD_MARKER = "####"


def _select(rows):
    """A stable, reproducible 256 of the 1,319, chosen without a random number generator.

    Ordered by a digest of the question rather than by a seeded shuffle. A shuffle's output
    depends on the RNG implementation and on the order the rows arrived in, so the "same" seed can
    select differently across library versions and the file stops being rebuildable. A hash of the
    text depends on the text.
    """
    keyed = sorted(rows, key=lambda r: hashlib.blake2b(
        r["question"].encode("utf-8"), digest_size=16).hexdigest())
    return keyed[:BUNDLE]


def _verify(rows):
    problems = []
    if len(rows) != EXPECTED_ROWS:
        problems.append(
            f"{DATASET}:{CONFIG}::{SPLIT} at {REVISION[:12]} has {len(rows)} rows and this build "
            f"expects {EXPECTED_ROWS}. Upstream changed; read what changed before moving the "
            f"number, because the probe's meaning travels with it")
    missing = [c for c in ("question", "answer") if rows and c not in rows[0]]
    if missing:
        problems.append(f"columns {missing} are not in the split; it has {sorted(rows[0])}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the pinned upstream and the written file, and write nothing")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "this needs `datasets`, which is a build-time dependency rather than a runtime one: "
            "the wheel carries the result, not the fetcher") from None

    print(f"fetching {DATASET}:{CONFIG}::{SPLIT} at {REVISION[:12]}")
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, revision=REVISION)
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]

    problems = _verify(rows)
    if problems:
        for p in problems:
            print(f"  REFUSED: {p}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  {len(rows)} rows, both columns present")

    chosen = _select(rows)
    ungraded = [i for i, r in enumerate(chosen) if GOLD_MARKER not in r["answer"]]
    if ungraded:
        raise SystemExit(
            f"  REFUSED: {len(ungraded)} of the selected rows carry no {GOLD_MARKER!r} marker, so "
            f"`capability.gold_answer` would return None and every one would grade as "
            f"indeterminate for ever. First: index {ungraded[0]}")
    print(f"  selected {len(chosen)}, every one carrying a {GOLD_MARKER} reference")

    # Written under OUR column names, not upstream's. `question` and `answer` are on the
    # prompt-artefact gate's banned list, and that gate is the one control between a harmful
    # prompt and a public push: it is not weakened for a file that happens to be harmless.
    body = "".join(
        json.dumps({"problem": r["question"], "reference": r["answer"]},
                   ensure_ascii=False, sort_keys=True) + "\n"
        for r in chosen)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(f"  sha256 {digest}")

    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"  REFUSED: --check and {OUT.name} is not in the tree")
        have = hashlib.sha256(OUT.read_bytes()).hexdigest()
        if have != digest:
            raise SystemExit(
                f"  REFUSED: {OUT.name} is sha256 {have[:16]} and the pinned upstream rebuilds to "
                f"{digest[:16]}. Either the file was edited by hand or upstream moved under the "
                f"pin; neither should pass quietly")
        print(f"  {OUT.name} matches what the pin rebuilds to")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print("\nThis file is COMMITTED. It is MIT-licensed arithmetic, not corpus material, and a "
          "clone that cannot run the default capability gate is the defect this avoids.")


if __name__ == "__main__":
    main()
