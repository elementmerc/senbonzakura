# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Fetch the bundled corpora at their pinned commits, verify them, and pack them for the wheel.

WHY THE WHEEL CARRIES THEM

Once installed the tool runs with no network. A corpus fetched on demand fails in an air-gapped
lab, behind a proxy, on a rate limit, and on the day an upstream repository is renamed. All of
them together are about 380 KB of text.

WHAT IS VERIFIED, AND IN WHICH ORDER

  1. **The bytes.** Each file is fetched at a pinned commit and hashed. A recorded hash is checked;
     an unrecorded one is recorded, from what was actually received rather than transcribed.
  2. **The shape.** The prompt column must still be there. An upstream that renames a column
     produces a file that parses into nothing, and a corpus of zero prompts is not a failure any
     later step reports usefully.
  3. **The count.** Each corpus records how many prompts it should yield. This is the check that
     catches a filter quietly matching something different: the file parses, the column is there,
     and the corpus becomes a different size. HarmBench is 400 rows of which exactly 200 are
     standalone, so that number is load-bearing rather than decorative.

Only then is anything packed.

    senbonzakura corpora            # fetch, verify, pack
    senbonzakura corpora --check    # verify what is recorded, write nothing

WHY IT LIVES IN THE PACKAGE RATHER THAN IN `tools/`

It was `tools/packaging/build_corpora.py`, and `tools/` ships in no wheel. A wheel from a release
carries the packed corpora and never needs this; an install straight from the repository does not,
because the packed blob is generated rather than committed, and that is the install every current
instruction hands out while PyPI is still serving a withdrawn version. So the one command those
users have to run was the one command their install did not contain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import bundled, corpora

#: The installed package's own directory. Everything written below lands inside it, which is where
#: the loader reads from, so this works the same from a checkout and from an install.
PKG = Path(__file__).resolve().parent

#: Where the fetched CSVs and the recorded hashes live. The hashes are committed; the CSVs are not,
#: for the same reason the llama.cpp binaries are not: what ships is reviewable in a diff of one
#: small file rather than by unpacking a wheel.
PINS = PKG / "vendor" / "corpora-pins.json"

#: The repository this package was built from, when it was built from one: `src/senbonzakura`'s
#: grandparent. In an install that is whatever happens to sit two levels above site-packages, so
#: it is only ever used behind the check below.
ROOT = PKG.parents[1]


def in_a_checkout():
    """Is there a repository around this package to write the generated notices into?

    An install has no `pyproject.toml` above it, and writing `THIRD-PARTY-CORPORA.md` next to
    somebody's site-packages would put a file nobody reads in a directory nobody owns.
    """
    return (ROOT / "pyproject.toml").is_file() and (ROOT / "src" / "senbonzakura").is_dir()


def _relative(path):
    """A path the reader can act on: repo-relative in a checkout, absolute otherwise."""
    try:
        return path.relative_to(ROOT) if in_a_checkout() else path
    except ValueError:
        return path


#: Fetched CSVs are cached beside the build in a checkout, and under the package otherwise. They
#: are intermediate rather than shipped, so either location is fine as long as it is writable.
CACHE = (ROOT / "build" / "corpora") if in_a_checkout() else (PKG / "data" / "corpora-cache")


class BuildError(Exception):
    """A corpus that cannot be trusted, phrased for a person."""


def preflight():
    """Refuse before the first download rather than partway through it.

    `doctor` sends people here by name when the corpora are missing, so this script is something
    a stranger runs on a fresh machine on the strength of that instruction. Without `gh` it used
    to die on a raw `FileNotFoundError: 'gh'` from deep inside subprocess, which names the
    program that is missing and nothing about what to do next.
    """
    if shutil.which("gh") is None:
        raise BuildError(
            "the GitHub CLI (`gh`) is not installed. The corpus sources are fetched through it "
            "so that no credential is ever handled by this script. Install it from "
            "https://cli.github.com, run `gh auth login`, then run this again.")


def fetch(repo, commit, path, *, timeout=120):
    """Raw bytes of one file at one commit, through `gh` so the credential never enters here."""
    try:
        r = subprocess.run(
            ["gh", "api", f"repos/{repo}/contents/{path}?ref={commit}",
             "-H", "Accept: application/vnd.github.raw"],
            capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        # Belt and braces behind preflight(): anything importing this module and calling fetch()
        # directly gets the same sentence rather than the bare OSError.
        raise BuildError(
            "the GitHub CLI (`gh`) is not installed; see https://cli.github.com") from e
    if r.returncode != 0 or not r.stdout:
        raise BuildError(
            f"could not fetch {repo}@{commit[:12]}:{path}: "
            f"{r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout


def read_pins():
    try:
        return json.loads(PINS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": "senbonzakura-corpora-pins/1", "files": {}}
    except json.JSONDecodeError as e:
        raise BuildError(f"{PINS} is not valid JSON: {e}") from e


def source_key(c):
    """Two corpora can share one upstream file, so hashes are keyed by the FILE, not the corpus."""
    return f"{c.upstream}@{c.commit}:{c.path}"


def build(*, check_only=False, log=print):
    pins = read_pins()
    files = dict(pins.get("files") or {})
    CACHE.mkdir(parents=True, exist_ok=True)

    # One fetch per distinct upstream file. XSTest backs two corpora and HarmBench backs two more;
    # fetching per corpus would double the requests and could, in principle, retrieve two different
    # copies of one file and never notice.
    wanted = {}
    for c in corpora.CORPORA.values():
        wanted.setdefault(source_key(c), c)

    raws = {}
    for skey, c in sorted(wanted.items()):
        cached = CACHE / f"{c.commit[:12]}-{Path(c.path).name}"
        if cached.is_file():
            data = cached.read_bytes()
            log(f"{skey}\n  using the local copy at {_relative(cached)}")
        else:
            log(f"{skey}\n  fetching")
            data = fetch(c.upstream, c.commit, c.path)
            cached.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        recorded = files.get(skey)
        if recorded is None:
            log(f"  recording sha256 {digest} ({len(data):,} bytes, first fetch of this pin)")
            files[skey] = {"sha256": digest, "bytes": len(data)}
        elif recorded["sha256"] != digest:
            raise BuildError(
                f"{skey}: sha256 is {digest} and the manifest records {recorded['sha256']}. A "
                f"file at a PINNED COMMIT cannot legitimately change, so this is not an upstream "
                f"edit; investigate before touching the pin. Nothing has been packed.")
        else:
            log("  sha256 matches the recorded value")
        raws[skey] = data

    # Shape and count, per corpus, before anything is packed.
    payload = {}
    for key in sorted(corpora.CORPORA):
        c = corpora.CORPORA[key]
        prompts = corpora.extract(raws[source_key(c)], c)
        corpora.check_count(c, prompts)
        payload[key] = prompts
        log(f"  {key:<21} {len(prompts):>4} prompts  ({c.arm}, {c.licence})")

    if check_only:
        log("check only: nothing written")
        return payload

    pins.update({"schema": "senbonzakura-corpora-pins/1", "files": dict(sorted(files.items()))})
    PINS.parent.mkdir(parents=True, exist_ok=True)
    PINS.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    log(f"recorded {len(files)} file hash(es) in {_relative(PINS)}")

    # Packed through the same container as the evaluation track, and for the same stated reason:
    # a speed bump against a scraper, not protection. `bundled.py` says so in as many words.
    blob = PKG / "data" / corpora.CORPORA_BLOB
    blob.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "senbonzakura-corpora/1",
           "corpora": {k: payload[k] for k in sorted(payload)},
           "notices": corpora.notices()}
    blob.write_bytes(bundled.pack(json.dumps(doc, ensure_ascii=False).encode("utf-8")))
    log(f"packed {_relative(blob)} ({blob.stat().st_size:,} bytes)")

    # Read it straight back. Writing a pack is not the same claim as shipping a usable one, and
    # the round trip is the only thing that distinguishes them.
    for key in sorted(payload):
        got = corpora.load(key)
        if got != payload[key]:
            raise BuildError(f"{key}: the pack did not read back as what was written")
    log(f"  round-tripped all {len(payload)} corpora out of the pack")

    # Only in a checkout. The file is a committed artefact generated from the same table the
    # loader reads; an install already carries the notices inside the pack, and writing a
    # Markdown file two levels above site-packages would put it somewhere nobody looks.
    if not in_a_checkout():
        log("not a checkout, so THIRD-PARTY-CORPORA.md was not written. The notices travel "
            "inside the pack either way; `senbonzakura doctor` prints them.")
        return payload

    notices = ROOT / "THIRD-PARTY-CORPORA.md"
    notices.write_text(
        "# Bundled corpora\n\n"
        "These prompt sets ship inside the wheel so the tool runs with no network. Each is "
        "redistributed under its own licence, and MIT and CC-BY both require the notice to travel "
        "with the work. This file is generated from the same table the loader reads, so it cannot "
        "fall out of step with what actually ships.\n\n"
        "```\n" + corpora.notices() + "\n```\n", encoding="utf-8")
    log(f"wrote {_relative(notices)}")
    return payload


def build_parser():
    ap = argparse.ArgumentParser(prog="senbonzakura corpora",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the pins and the counts, write nothing")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        preflight()
        payload = build(check_only=a.check)
    except (BuildError, corpora.CorpusError) as e:
        print(f"corpora: {e}", file=sys.stderr)
        return 1
    total = sum(len(v) for v in payload.values())
    print(f"{len(payload)} corpora, {total:,} prompts")
    print(f"(bundling into the wheel is {Path(bundled.__file__).name}'s job at release time)")
    return 0


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry

    module_entry(main)
