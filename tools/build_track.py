#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Assemble the harmful and harmless prompt pools this project measures on, from their sources.

WHY THIS EXISTS

Every prompt in this project's evaluation track came from somewhere else. Two of the three
upstream datasets declare no licence at all, and the probable root of the harmless side is
CC BY-NC 4.0. An absent licence is not a permissive one, so redistributing the rows is not
something this repository can grant anybody. What it can publish is the recipe.

So this fetches the named datasets at pinned revisions and writes the two plain text files
`senbonzakura track` expects. You fetch the rows yourself, under whatever licence position
attaches to you; nothing harmful is stored in this repository at any point.

It also refuses to run quietly past a licence that has changed under it. Each source records the
licence it declared when the card was written, and a mismatch at fetch time stops the build,
because the whole reason this file exists is that the licence position is load-bearing.

WHAT IT CANNOT DO

It cannot rebuild the exact track behind this project's published numbers. Harmless "top-ups"
were added to that corpus by hand and never recorded: not their source, not their count, not
their revision. A track built here is the same shape and is not the same rows, and the gap is
unquantified. See docs/evaluation-track-card.md.

Usage:
    python tools/build_track.py --out corpus
    senbonzakura track --harmful corpus/harmful.txt --harmless corpus/harmless.txt --out mytrack
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/datasets/{repo}"

#: Every upstream, with the licence it declared and the revision that declaration was read at.
#: `licence: None` records "this dataset declares nothing", which is a position rather than a gap
#: in this table: it is the reason the rows are not redistributed.
SOURCES = [
    {
        "repo": "Bahushruth/abliteration-harmful-enriched",
        "side": "harmful",
        "split": "train",
        "revision": "f29c0b778d3412d5cc7acfc9c6463ce13886212c",
        "licence": "apache-2.0",
        "note": "Synthetic harmful prompts across 33 categories, plus about 430 rows that trace "
                "to mlabonne/harmful_behaviors and from there to AdvBench.",
    },
    {
        "repo": "mlabonne/harmless_alpaca",
        "side": "harmless",
        "split": "train",
        "revision": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
        "licence": None,
        "note": "Declares no licence. Its name and provenance point at tatsu-lab/alpaca, which "
                "declares CC BY-NC 4.0: attribution required, non-commercial use only.",
    },
]

#: The column each upstream keeps its prompt text in. Both use `text` today; asserting it rather
#: than assuming it means a schema change upstream fails at the boundary instead of producing a
#: track full of empty strings that loads, splits and scores nothing.
TEXT_COLUMN = "text"

CITATIONS = """\
If you publish anything measured on a track built here, cite all of the following.

  Zou, Wang, Carlini, Nasr, Kolter and Fredrikson. Universal and Transferable Adversarial
  Attacks on Aligned Language Models (2023). The llm-attacks repository, MIT licence.
  AdvBench reaches this track through mlabonne/harmful_behaviors.

  Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang and Hashimoto. Stanford Alpaca: An
  Instruction-following LLaMA Model (2023). CC BY-NC 4.0. The probable root of the harmless
  side, reached through mlabonne/harmless_alpaca.

  Arditi, Obeso, et al. Refusal in Language Models Is Mediated by a Single Direction (2024).

  The HuggingFace datasets named in SOURCES above, at the revisions recorded there.

  This repository.
"""


def declared_licence(repo, timeout=60):
    """What the Hub says this dataset's licence is right now, or None if it says nothing.

    Read from `cardData.license` rather than the tag list: the tags are derived from the card and
    a dataset can carry a `license:` tag with no card field, so the card is the narrower and more
    honest source.
    """
    req = urllib.request.Request(          # noqa: S310 - fixed https host
        API.format(repo=repo), headers={"User-Agent": "senbonzakura"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - fixed https host
            doc = json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"build-track: could not read the licence HuggingFace publishes for {repo} ({e}). "
            f"Refusing to assemble a corpus without checking the position it is distributed "
            f"under. Retry, or pass --skip-licence-check with a reason you have recorded.") from e
    card = doc.get("cardData") or {}
    value = card.get("license")
    return value or None


def check_licences(sources, *, skip=False, log=print):
    """Stop if any upstream's declared licence has moved since the card was written.

    A relicensing upstream is not a detail: it is the single fact that decides whether a rebuilt
    track may be redistributed, and it changes silently. Checking it costs one HTTP request per
    source and catches the case where this file's table has quietly become fiction.
    """
    if skip:
        log("build-track: WARNING: licence check skipped. The recorded positions below are "
            "whatever this file last claimed, which may no longer be true upstream.")
        return
    drifted = []
    for src in sources:
        live = declared_licence(src["repo"])
        recorded = src["licence"]
        state = "declares nothing" if live is None else live
        log(f"  {src['repo']}: {state}")
        if live != recorded:
            drifted.append((src["repo"], recorded, live))
    if drifted:
        lines = [f"  {repo}: recorded {rec or 'nothing declared'}, now {live or 'nothing declared'}"
                 for repo, rec, live in drifted]
        raise SystemExit(
            "build-track: an upstream licence has changed since this recipe was written:\n"
            + "\n".join(lines)
            + "\nThat decides what a track built from it may be used for, so the build stops "
              "here. Update SOURCES and docs/evaluation-track-card.md together, then re-run.")


def fetch(src, log=print):
    """The prompt rows of one upstream, at its pinned revision, in upstream order.

    Order is upstream's own and is never shuffled here. The split into fit, search and measure
    happens downstream in `senbonzakura track`, which is seeded and records its boundaries; a
    shuffle at this layer would put an unrecorded permutation underneath a recorded one.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "build-track: the `datasets` package is required to fetch the upstream corpora. "
            "Install it with `pip install datasets`.") from e
    log(f"  fetching {src['repo']} split={src['split']} revision={src['revision'][:12]}")
    try:
        ds = load_dataset(src["repo"], split=src["split"], revision=src["revision"])
    except Exception as e:
        raise SystemExit(
            f"build-track: could not fetch {src['repo']} at revision {src['revision'][:12]} "
            f"({type(e).__name__}: {e}). A pinned revision that has been deleted upstream is the "
            f"likeliest cause; the recipe is then no longer reproducible and SOURCES needs "
            f"re-pinning against a revision that exists.") from e
    if TEXT_COLUMN not in ds.column_names:
        raise SystemExit(
            f"build-track: {src['repo']} has columns {sorted(ds.column_names)} and no "
            f"'{TEXT_COLUMN}' column. The schema has changed upstream; this recipe reads "
            f"'{TEXT_COLUMN}' and would otherwise assemble a corpus of empty strings.")
    rows = [str(t).strip() for t in ds[TEXT_COLUMN]]
    rows = [t for t in rows if t]
    if not rows:
        raise SystemExit(
            f"build-track: {src['repo']} yielded no non-empty prompts. Refusing to write an "
            f"empty side rather than producing a track that splits and scores nothing.")
    log(f"    {len(rows)} non-empty prompts")
    return rows


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, rows):
    """One prompt per line, written whole or not at all.

    A half-written pool that a later run picks up is a corpus nobody can account for, so the file
    appears at its final path only once every byte is on disk.
    """
    tmp = f"{path}.partial"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.replace("\n", " ").replace("\r", " ") + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def build(out, *, skip_licence_check=False, log=print):
    os.makedirs(out, exist_ok=True)
    log("build-track: checking what each upstream declares today")
    check_licences(SOURCES, skip=skip_licence_check, log=log)

    log("build-track: fetching")
    pools = {"harmful": [], "harmless": []}
    per_source = []
    for src in SOURCES:
        rows = fetch(src, log=log)
        pools[src["side"]].extend(rows)
        per_source.append({
            "repo": src["repo"], "side": src["side"], "split": src["split"],
            "revision": src["revision"], "declared_licence": src["licence"],
            "rows": len(rows),
        })

    missing = [side for side, rows in pools.items() if not rows]
    if missing:
        raise SystemExit(
            f"build-track: no rows for {', '.join(missing)}. A track needs both sides: a refusal "
            f"rate with no harmless arm cannot tell a working abliteration apart from a model too "
            f"damaged to refuse anything.")

    written = {}
    for side, rows in pools.items():
        path = os.path.join(out, f"{side}.txt")
        write_atomic(path, rows)
        written[side] = {"path": path, "rows": len(rows), "sha256": sha256_of(path)}
        log(f"  wrote {path}: {len(rows)} prompts, sha256 {written[side]['sha256'][:16]}")

    manifest = {
        "schema": "senbonzakura-corpus/1",
        "sources": per_source,
        "outputs": {k: {"rows": v["rows"], "sha256": v["sha256"]} for k, v in written.items()},
        "licence_check_skipped": bool(skip_licence_check),
        "reproduces_published_track": False,
    }
    mpath = os.path.join(out, "sources.json")
    tmp = f"{mpath}.partial"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, mpath)
    log(f"  wrote {mpath}")

    log("")
    log("This is NOT the corpus this project's published numbers were measured on. That one had "
        "harmless top-ups added by hand which were never recorded, so it cannot be rebuilt by "
        "anybody, including us. See docs/evaluation-track-card.md.")
    log("")
    log("Next, split it so you cannot mark your own homework:")
    log(f"  senbonzakura track --harmful {written['harmful']['path']} "
        f"--harmless {written['harmless']['path']} --out mytrack")
    log("")
    log(CITATIONS)
    return manifest


def build_parser():
    ap = argparse.ArgumentParser(
        prog="build_track.py",
        description="Fetch the upstream corpora this project measures on and write the two "
                    "prompt files `senbonzakura track` splits.")
    ap.add_argument("--out", required=True,
                    help="directory to write harmful.txt, harmless.txt and sources.json into")
    ap.add_argument("--skip-licence-check", action="store_true",
                    help="do not ask the Hub what each upstream declares today. Only for an "
                         "offline rebuild, and it means the recorded positions are unverified")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    build(args.out, skip_licence_check=args.skip_licence_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
