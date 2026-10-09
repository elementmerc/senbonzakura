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
import gzip
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

#: Commit records read per model. Measured: the sample averaged 2.75 commits per model, so this
#: is far above the common case and exists to stop one pathological history dominating a shard.
MAX_COMMITS = 500

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


#: The documents worth reading, measured rather than guessed: `discover` over 200 authors of the
#: real corpus found eight per model directory and these are the six that carry fields. The others
#: are `config.json` and `tokenizer_config.json` (model architecture, not provenance),
#: `discussions.json` (conversation bodies), `readme-main.md` (prose) and `gguf-metadata.bin`
#: (binary). Reading only these halves the file count, which on twelve million files is hours.
READ_FILES = ("meta.json", "commits-main.jsonl", "paths-info.json", "_deleted.json",
              "base-model.json", "ns-overview.json")


def _extensions_from(names):
    """Lowercase extensions, without the dot, from a list of filenames."""
    out = set()
    for n in names:
        if isinstance(n, str) and "." in n:
            out.add(n.rsplit(".", 1)[1].lower()[:16])
    return sorted(out)


def _model_row(author, model_dir, problems):
    """One model's distillate, or None when the directory carries nothing worth keeping.

    THE FIELD NAMES HERE ARE MEASURED, which is the whole reason `discover` exists. The first
    version of this file guessed `securityFileStatus` for the host's file verdict; the corpus
    spells it `status` inside `paths-info.json`, and a run against the guess would have written
    three million rows with that field empty and nothing saying why.
    """
    files = {p.name: p for p in model_dir.iterdir() if p.is_file() and p.name in READ_FILES}
    if not files:
        return None

    meta = _read_json(files["meta.json"], problems) if "meta.json" in files else None
    if not isinstance(meta, dict) or not meta.get("id"):
        # No identity means nothing downstream can refer to it. Counted so the total is honest.
        problems["model directory with no usable meta.json"] += 1
        return None

    siblings = meta.get("siblings")
    names = [s.get("rfilename") for s in siblings] if isinstance(siblings, list) else []
    row = {
        "author": author,
        "model": meta.get("id"),
        "created_at": meta.get("createdAt"),
        "last_modified": meta.get("lastModified"),
        "downloads": meta.get("downloads"),
        "likes": meta.get("likes"),
        "private": meta.get("private"),
        "gated": meta.get("gated"),
        "disabled": meta.get("disabled"),
        "pipeline_tag": meta.get("pipeline_tag"),
        "tags": meta.get("tags") if isinstance(meta.get("tags"), list) else [],
        "sha": meta.get("sha"),
        "extensions": _extensions_from(names),
        "file_count": len(names),
    }

    # THE TIMELINE, and the limitation is recorded in the row rather than left to a reader.
    # These records carry a date, a title, a message and the commit authors. They do NOT carry
    # the files each commit touched, so the four "first ever <format>" signatures cannot be
    # computed from this corpus at all, while dormancy, author change and burst can. A row that
    # did not say so would invite somebody to compute the other four and get a confident wrong
    # answer from a field that was never there.
    if "commits-main.jsonl" in files:
        commits = []
        for rec in _read_jsonl(files["commits-main.jsonl"], problems, limit=MAX_COMMITS):
            if not isinstance(rec, dict):
                continue
            who = rec.get("authors")
            user = None
            if isinstance(who, list) and who and isinstance(who[0], dict):
                user = who[0].get("user")
            commits.append([rec.get("date"), user])
        # SORTED OLDEST FIRST, AND THIS IS A FIX RATHER THAN A TIDY-UP. Measured on 4,776 real
        # rows: 3,362 arrive newest-first, 3 oldest-first and 4 in neither order. The ported
        # hijacking signatures take the LAST commit as the newest, so fed in corpus order every
        # one of them would have computed backwards: a three-year dormancy would have read as a
        # three-year-old burst, and the author change would have compared the wrong pair. Nothing
        # would have failed; the answers would just have been wrong.
        #
        # Sorting rather than reversing, because those 4 mixed rows mean the order is not a
        # property this corpus guarantees. A record with no date cannot be placed, so it goes last
        # and is counted instead of being given a position it did not earn.
        undated = sum(1 for c in commits if not c[0])
        commits.sort(key=lambda c: (c[0] is None, c[0] or ""))
        row["commits"] = commits
        row["commit_count"] = len(commits)
        # Stated rather than implied, for the same reason as the file-lists flag below: a
        # consumer that has to guess the order will guess wrong on 4 rows in 4,776.
        row["commits_oldest_first"] = True
        if undated:
            row["commits_without_a_date"] = undated
        row["commits_carry_file_lists"] = False
    else:
        row["commits"] = None
        row["commit_unavailable_because"] = "this model directory holds no commits-main.jsonl"

    # The host's own per-file verdict, carried as found and never as our own reading.
    if "paths-info.json" in files:
        pi = _read_json(files["paths-info.json"], problems)
        if isinstance(pi, list):
            statuses = sorted({e.get("status") for e in pi
                               if isinstance(e, dict) and e.get("status")})
            row["host_file_statuses"] = [s for s in statuses if s]

    # The claimed base. This is the provenance claim `senbonzakura diff` exists to check, so the
    # id is what matters; the rest of that document is a copy of the base's own metadata.
    if "base-model.json" in files:
        base = _read_json(files["base-model.json"], problems)
        if isinstance(base, dict):
            row["claimed_base"] = base.get("id")

    # Gone, and why. The field this corpus has that no live API answer does.
    if "_deleted.json" in files:
        dele = _read_json(files["_deleted.json"], problems)
        if isinstance(dele, dict):
            row["deleted_at"] = dele.get("deleted_at")
            row["deleted_reason"] = dele.get("reason")
    return row


def _author_row(author, model_dir, problems):
    """The namespace's own record, from any one of its models. None when absent.

    `ns-overview.json` describes the AUTHOR and is duplicated into each model directory, so it is
    read once per author rather than three million times. Account age and follower count are the
    signals here, and they are per-namespace by nature.
    """
    path = model_dir / "ns-overview.json"
    if not path.is_file():
        return None
    ns = _read_json(path, problems)
    if not isinstance(ns, dict):
        return None
    return {
        "author": author,
        "kind": ns.get("kind"),
        "fullname": ns.get("fullname"),
        "created_at": ns.get("created_at") or ns.get("created_at_decoded_from_id"),
        "num_models": ns.get("num_models"),
        "num_datasets": ns.get("num_datasets"),
        "num_followers": ns.get("num_followers"),
        "num_discussions": ns.get("num_discussions"),
        "is_pro": ns.get("is_pro"),
        "type_field": ns.get("type_field"),
        "orgs": [o.get("name") for o in ns.get("orgs") or []
                 if isinstance(o, dict) and o.get("name")],
    }


def _shard(root, out_dir, index, of, limit, queue=None):
    """One worker's slice: every author whose position modulo `of` is `index`.

    PARTITION, DISTRIBUTE, MERGE, which §12 asks for and which this needs rather than wants.
    Measured on the real corpus: `discover` read 200 authors in 12 seconds, and 611,154 authors at
    that rate is about ten hours in one process. Each shard writes its OWN pair of files, so there
    is no lock on a hot path and a crashed worker costs its slice rather than the run.

    Each shard writes a done-marker, so a re-run skips what finished. The markers are what make
    this safe to interrupt, which matters on a box somebody else is also using.
    """
    done = out_dir / f"shard-{index:03d}.done"
    if done.exists():
        return {"shard": index, "skipped": True}
    models_path = out_dir / f"models-{index:03d}.jsonl.gz"
    authors_path = out_dir / f"authors-{index:03d}.jsonl.gz"
    problems, n_models, n_authors, n_dirs = Counter(), 0, 0, 0
    started = time.monotonic()
    with gzip.open(models_path, "wt", encoding="utf-8") as mfh, \
            gzip.open(authors_path, "wt", encoding="utf-8") as afh:
        for position, entry in enumerate(_authors(root)):
            if position % of != index:
                continue
            n_dirs += 1
            if limit and n_dirs > limit:
                break
            author = entry.name
            seen_author = False
            try:
                subs = [d for d in Path(entry.path).iterdir() if d.is_dir()]
            except OSError as e:
                problems[f"author unreadable: {type(e).__name__}"] += 1
                continue
            for model_dir in subs:
                if not seen_author:
                    arow = _author_row(author, model_dir, problems)
                    if arow:
                        afh.write(json.dumps(arow, sort_keys=True) + "\n")
                        n_authors += 1
                        seen_author = True
                row = _model_row(author, model_dir, problems)
                if row:
                    mfh.write(json.dumps(row, sort_keys=True) + "\n")
                    n_models += 1
            if queue is not None and n_dirs % HEARTBEAT_EVERY == 0:
                queue.put((index, n_dirs, n_models))
    done.write_text(f"{n_dirs} authors, {n_models} models, {n_authors} namespaces\n")
    return {"shard": index, "authors": n_dirs, "models": n_models,
            "namespaces": n_authors, "seconds": round(time.monotonic() - started, 1),
            "problems": dict(problems)}


def extract(root, out_dir, limit, workers):
    """Walk the corpus and write a gzipped distillate per shard."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    if workers <= 1:
        results = [_shard(root, out_dir, 0, 1, limit)]
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            results = pool.starmap(
                _shard, [(root, out_dir, i, workers, limit) for i in range(workers)])

    models = sum(r.get("models", 0) for r in results)
    authors = sum(r.get("authors", 0) for r in results)
    namespaces = sum(r.get("namespaces", 0) for r in results)
    problems = Counter()
    for r in results:
        problems.update(r.get("problems") or {})
    size = sum(f.stat().st_size for f in out_dir.glob("*.jsonl.gz"))
    summary = {
        "authors": authors, "models": models, "namespaces": namespaces,
        "workers": workers,
        "seconds": round(time.monotonic() - started, 1),
        "compressed_bytes": size,
        "out_dir": str(out_dir),
        "problems": dict(problems),
        "shards": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    # A RE-RUN THAT SKIPPED EVERYTHING IS A SUCCESS, NOT AN EMPTY RESULT. Found by reading this
    # function's own output rather than its exit code: a second run over a finished extraction has
    # zero models because there was nothing left to do, and reporting that as the zero-result
    # refusal tells somebody who re-ran out of caution that their distillate is broken.
    if all(r.get("skipped") for r in results):
        print("\nEvery shard was already complete, so nothing was walked. The distillate in "
              f"{out_dir} is the earlier run's and is unchanged. Delete the .done markers to "
              f"force a re-extraction.", file=sys.stderr)
        return 0
    if not models:
        print("\nZERO MODELS. The walk ran and matched nothing, which is a statement about the "
              "field names this looked for rather than about the corpus. Run `discover` and "
              "compare its keys_by_filename against the readers in `_model_row`.", file=sys.stderr)
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

    e = sub.add_parser("extract", help="walk everything and write a gzipped distillate")
    e.add_argument("root")
    e.add_argument("--out", required=True,
                   help="a DIRECTORY for the distillate. One pair of gzipped JSONL files per "
                        "shard, plus a done-marker each so a re-run resumes")
    e.add_argument("--limit", type=int, default=0,
                   help="stop after this many authors PER SHARD, for a rehearsal on real data")
    e.add_argument("--workers", type=int, default=1,
                   help="parallel shards (default 1). The corpus this was built for needs about "
                        "ten hours in one process and an hour across four")

    a = ap.parse_args(argv)
    if a.mode == "discover":
        return discover(a.root, a.sample)
    if a.workers < 1:
        ap.error("--workers must be at least 1")
    return extract(a.root, a.out, a.limit, a.workers)


if __name__ == "__main__":
    sys.exit(main())
