#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Distil a large per-author metadata corpus into the few fields a suspicion check needs.

WHY THIS RUNS WHERE THE DATA IS, AND NOT HERE

The corpus this was written for is about 507 GB across 611,156 author directories holding roughly
12 million small files. The owning project measured the transfer rates: 225 directory entries a
second to enumerate, about 10 files a second in one stream and about 78 across eight. At 12 million
files that is somewhere near 43 hours to copy, for a body of data where every field we want fits in
a couple of hundred bytes per model.

So this script is designed to be copied to the machine that holds the volume, run there against a
local path, and to emit one small file that comes back in a minute. Nothing in it is specific to
any host: the corpus root is an argument, and there is no address, credential or hostname anywhere
in this file, because it is committed to a repository with a public remote.

THE TWO MODES, AND WHY DISCOVERY COMES FIRST

    corpus_distil.py discover <root>        read a sample, report the shapes actually present
    corpus_distil.py extract  <root> --out  walk everything, emit JSONL

`discover` exists because this script has never seen the corpus. Writing an extractor against an
assumed layout and running it over 611k directories produces a confident empty file, which is the
failure this project keeps recording: the instrument was fine and the reading was about something
else. So the first pass reports the filenames it found, the JSON keys inside them and how often
each appears, and refuses to guess. Read that, then extract.

WHAT IT IS CAREFUL ABOUT

Memory is bounded by one file. Every file read is capped, so a single pathological entry cannot
take the run down. The output is written through a temporary name and renamed on close, so an
interrupted run leaves no half file that looks complete. Progress is a heartbeat on stderr, and
state markers mean a re-run skips what finished. Directory iteration is `os.scandir`, which does
not build a list of 611k names in memory first.

WHAT IT DELIBERATELY DOES NOT DO

It does not copy model weights, readmes or any document body. It takes identifiers, dates, file
extensions and flags: the inputs the timeline signatures need, and nothing whose value is its
text. That keeps the distillate small, keeps it free of anything anybody wrote, and means the
output can be reasoned about without re-reading 507 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

#: No single metadata file in a corpus of this shape is legitimately larger than this, and a file
#: that is has either been mis-stored or is not metadata. Reading it would be the one unbounded
#: allocation in the script, so it is skipped and counted rather than read.
MAX_FILE_BYTES = 8 * 1024 * 1024

#: How often the heartbeat prints, in authors. The baseline asks for one every 30 to 60 seconds on
#: any long loop; at the enumeration rates measured on this corpus that is roughly this many.
HEARTBEAT_EVERY = 2_000

#: Authors sampled by `discover` unless told otherwise. Enough to see which filenames are
#: universal and which are occasional, cheap enough to run while something else is using the disk.
DISCOVER_SAMPLE = 200

#: The fields worth carrying out, as the keys this looks for at any depth in a metadata document.
#: Spelled generously because the corpus was written by another project and its exact naming is
#: not assumed: `discover` reports what is actually there and this list is what gets matched
#: against it. A key present in the corpus and missing here is a gap `discover` makes visible.
WANTED = {
    "id", "modelId", "model_id", "author", "namespace", "owner",
    "createdAt", "created_at", "lastModified", "last_modified",
    "downloads", "likes", "private", "gated", "disabled",
    "sha", "siblings", "tags", "pipeline_tag", "library_name",
    "securityFileStatus", "security_file_status", "security_repo_status",
}


def _read_json(path, problems):
    """One JSON document, or None. A file that cannot be read is counted, never fatal."""
    try:
        size = path.stat().st_size
    except OSError as e:
        problems[f"stat failed: {type(e).__name__}"] += 1
        return None
    if size > MAX_FILE_BYTES:
        problems[f"skipped, larger than {MAX_FILE_BYTES} bytes"] += 1
        return None
    if size == 0:
        problems["skipped, zero bytes"] += 1
        return None
    try:
        with path.open("rb") as fh:
            return json.loads(fh.read(MAX_FILE_BYTES).decode("utf-8", "replace"))
    except (OSError, ValueError) as e:
        problems[f"unreadable: {type(e).__name__}"] += 1
        return None


def _walk_keys(obj, out, prefix="", depth=0):
    """Every key path in a document, to a bounded depth.

    DEPTH IS CAPPED because this reads documents produced elsewhere, and §2.1 asks every parser
    for a recursion bound. A deeply nested or self-referential document would otherwise recurse
    until the interpreter gives up, inside a loop over 611k directories.
    """
    if depth > 6:
        out["(deeper than 6, not described)"] += 1
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out[f"{prefix}{k}"] += 1
            _walk_keys(v, out, f"{prefix}{k}.", depth + 1)
    # The first element only: a list of 10,000 siblings has one shape, and describing each would
    # make the report about the corpus's size rather than its shape.
    elif isinstance(obj, list) and obj:
        _walk_keys(obj[0], out, f"{prefix}[].", depth + 1)


def _authors(root):
    """Author directories, streamed. `scandir` so 611k names are never a list in memory."""
    with os.scandir(root) as it:
        for entry in it:
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield entry
            # PERF203 is accepted here with a reason: the per-entry guard is the point. One
            # unreadable directory out of 611,156 must cost that entry and not the walk, and
            # hoisting the try outside the loop would trade a resumable run for a lost one.
            except OSError:  # noqa: PERF203
                continue


#: Lines read from a `.jsonl` document when describing its shape. One line would miss a field that
#: only later records carry; all of them would read a 12-million-file corpus to learn a shape.
JSONL_SAMPLE_LINES = 20

#: Documents whose value is their prose rather than their fields. Never read, never described, and
#: never extracted: a distillate with somebody's readme in it is a different artefact with
#: different obligations, and the fields we want are not in it.
PROSE_FILES = (".md", ".txt", ".rst")


def _read_jsonl(path, problems, limit=JSONL_SAMPLE_LINES):
    """Up to `limit` records from a JSON-lines document. A bad line is counted, never fatal.

    THIS EXISTS BECAUSE THE COMMIT TIMELINE IS A .jsonl AND THE FIRST VERSION SKIPPED IT. The
    extractor matched `*.json`, so `commits-main.jsonl` — the dates, authors and file lists per
    commit, which is the single richest thing in the corpus and the exact input the hijacking
    signatures need — was invisible to it. Found by running `discover` against the real tree
    rather than by reasoning about it.
    """
    out = []
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            problems[f"{path.name}: skipped, larger than {MAX_FILE_BYTES} bytes"] += 1
            return out
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    out.append(json.loads(text))
                except ValueError:
                    problems[f"{path.name}: a line was not JSON"] += 1
    except OSError as e:
        problems[f"{path.name}: unreadable, {type(e).__name__}"] += 1
    return out


def _describe(path, problems, per_file):
    """Record the key paths in one document, bucketed under its filename.

    BUCKETED BY FILENAME, which the first version did not do and which made its report nearly
    useless on a real corpus. Eight documents per model directory were merged into one flat key
    list, so it said the corpus contains a `downloads` field somewhere without saying that it
    lives in `meta.json`. An extractor cannot be written from that.
    """
    if path.suffix in PROSE_FILES:
        return
    if path.suffix == ".jsonl":
        for rec in _read_jsonl(path, problems):
            _walk_keys(rec, per_file[path.name])
    elif path.suffix == ".json":
        doc = _read_json(path, problems)
        if doc is not None:
            _walk_keys(doc, per_file[path.name])


def discover(root, sample):
    """Report the shapes actually present, and assert nothing."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory. Point this at the corpus root, the one "
                         f"holding one directory per author.")
    filenames, problems = Counter(), Counter()
    per_file: dict[str, Counter] = {}
    seen_authors = models = 0
    for entry in _authors(root):
        seen_authors += 1
        if seen_authors > sample:
            break
        for sub in Path(entry.path).iterdir():
            targets = []
            if sub.is_dir():
                models += 1
                targets = [f for f in sub.iterdir() if f.is_file()]
            elif sub.is_file():
                targets = [sub]
            for f in targets:
                filenames[f.name] += 1
                per_file.setdefault(f.name, Counter())
                _describe(f, problems, per_file)

    sampled = min(seen_authors, sample)
    report = {
        "authors_sampled": sampled,
        "model_directories_seen": models,
        "models_per_author": round(models / sampled, 2) if sampled else 0,
        "filenames": filenames.most_common(40),
        # The useful half: which keys live in WHICH document. An extractor is written from this.
        "keys_by_filename": {
            name: counts.most_common(30) for name, counts in sorted(per_file.items()) if counts},
        "documents_described_but_empty": sorted(n for n, c in per_file.items() if not c),
        "problems": dict(problems),
    }
    all_keys = {k for counts in per_file.values() for k in counts}
    report["wanted_keys_present"] = sorted(
        w for w in WANTED if any(k == w or k.endswith(f".{w}") for k in all_keys))
    report["wanted_keys_absent"] = sorted(
        w for w in WANTED if not any(k == w or k.endswith(f".{w}") for k in all_keys))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not filenames:
        # An empty report is a finding about this script, not about the corpus, and it must not
        # read as "the corpus is empty". It is the exact shape of the mistake this mode prevents.
        print("\nNOTHING WAS FOUND, which is a statement about where this looked rather than "
              "about the corpus. The layout assumed here is <root>/<author>/... Check the root, "
              "and if the real layout differs, that is what this mode exists to tell you.",
              file=sys.stderr)
        return 1
    if not any(per_file.values()):
        print("\nFILES WERE FOUND AND NONE COULD BE DESCRIBED. Every document was prose, "
              "unreadable, or of a kind this does not parse. Read `filenames` and `problems` "
              "above: the layout is right and the readers are wrong.", file=sys.stderr)
        return 1
    return 0


def _pick(doc, names):
    """The first of `names` present anywhere in the document, searched breadth first."""
    queue = [doc]
    seen = 0
    while queue and seen < 5_000:          # bounded: a document cannot make this loop forever
        node = queue.pop(0)
        seen += 1
        if isinstance(node, dict):
            for n in names:
                if n in node and node[n] is not None:
                    return node[n]
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node[:50])
    return None


def _extensions(doc):
    """Every file extension the document says the repo holds, lowercase, without the dot."""
    sibs = _pick(doc, ("siblings", "files"))
    out = set()
    if isinstance(sibs, list):
        for s in sibs[:5_000]:
            name = s.get("rfilename") or s.get("filename") or s.get("path") if isinstance(s, dict) \
                else (s if isinstance(s, str) else None)
            if isinstance(name, str) and "." in name:
                out.add(name.rsplit(".", 1)[1].lower()[:16])
    return sorted(out)


def extract(root, out_path, limit):
    """Walk every author and write one JSONL row per model document found."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    problems = Counter()
    authors = rows = 0
    started = time.monotonic()

    with tmp.open("w", encoding="utf-8") as sink:
        for entry in _authors(root):
            authors += 1
            if limit and authors > limit:
                break
            for path in Path(entry.path).rglob("*.json"):
                doc = _read_json(path, problems)
                if not isinstance(doc, dict):
                    continue
                model = _pick(doc, ("id", "modelId", "model_id"))
                if not isinstance(model, str):
                    continue
                row = {
                    "author": entry.name,
                    "model": model,
                    "created_at": _pick(doc, ("createdAt", "created_at")),
                    "last_modified": _pick(doc, ("lastModified", "last_modified")),
                    "downloads": _pick(doc, ("downloads",)),
                    "likes": _pick(doc, ("likes",)),
                    "disabled": _pick(doc, ("disabled",)),
                    "gated": _pick(doc, ("gated",)),
                    "extensions": _extensions(doc),
                    # The host's own security answer where the corpus happens to hold one. Carried
                    # as found and never as a verdict: it is somebody else's reading.
                    "host_security": _pick(doc, ("securityFileStatus", "security_file_status",
                                                 "security_repo_status")),
                    "source_file": path.name,
                }
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
            if authors % HEARTBEAT_EVERY == 0:
                rate = authors / max(time.monotonic() - started, 1e-6)
                print(f"  {authors} authors, {rows} rows, {rate:.0f} authors/s", file=sys.stderr)

    # Rename on close, so an interrupted run never leaves a file that looks finished.
    tmp.replace(out_path)
    summary = {
        "authors": authors, "rows": rows,
        "seconds": round(time.monotonic() - started, 1),
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "problems": dict(problems),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    if not rows:
        print("\nZERO ROWS. The walk ran and matched nothing, which is a statement about the "
              "field names this looked for rather than about the corpus. Run `discover` and "
              "compare its key_paths against WANTED in this file.", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        epilog="Run `discover` first. An extractor written against an assumed layout produces a "
               "confident empty file.")
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("discover", help="read a sample and report the shapes present")
    d.add_argument("root", help="the corpus root, holding one directory per author")
    d.add_argument("--sample", type=int, default=DISCOVER_SAMPLE,
                   help=f"author directories to read (default {DISCOVER_SAMPLE})")

    e = sub.add_parser("extract", help="walk everything and write JSONL")
    e.add_argument("root")
    e.add_argument("--out", required=True, help="where to write the JSONL distillate")
    e.add_argument("--limit", type=int, default=0,
                   help="stop after this many authors, for a rehearsal on a real corpus")

    a = ap.parse_args(argv)
    if a.mode == "discover":
        return discover(a.root, a.sample)
    return extract(a.root, a.out, a.limit)


if __name__ == "__main__":
    sys.exit(main())
