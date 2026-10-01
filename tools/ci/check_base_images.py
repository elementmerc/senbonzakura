#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Are the container base images pinned by digest, and has any pin gone stale?

THE DEFECT THIS CLOSES

Baseline section 5 says container base images are pinned by digest, never by tag. Until
2026-10-01 both Dockerfiles this project PUBLISHES named their base by tag, while
`head-to-head/Dockerfile.tool` in the same tree carried a full digest. The house rule was
followed by the benchmark image and broken by the one a stranger pulls from GHCR.

TWO MODES, AND THE SPLIT IS THE SAME ONE `resolve_action_pins.py` USES

  --check   Offline. Reads the Dockerfiles and nothing else, and refuses a `FROM` that is not
            digest-pinned. No network, no credentials, so it runs on every push and in a clean
            room. This is the blocking half.
  --drift   Asks the registry what each tag resolves to TODAY and compares. Reports; never
            rewrites. A tag being repointed is the event digest pinning exists to survive, so a
            job that quietly re-pinned would erase the evidence of exactly what it is watching
            for and call it success.

WHY THE ARITHMETIC IS NOT IN THIS FILE

`senbonzakura.baseimages` holds the parsing and the verdicts as pure functions with tests, the
same way `senbonzakura.vendoring` does for the vendored llama.cpp pins. This file fetches and
prints. That is what lets every branch, including the cooldown boundary, be tested without a
network and without waiting a week.

FAILURE POSTURE

    exit 0   every FROM digest-pinned (--check), or the drift report produced (--drift)
    exit 1   a FROM is pinned by tag, a reference is malformed, or no Dockerfile was found
    exit 2   DID NOT RUN: the registry could not be consulted at all under --drift

Exit 2 rather than 1 for an unreachable registry, because "could not look" is a third outcome
and this repository has been bitten twice by a gate whose exit status could only express two.
A caller that wants an unreachable registry to be red can say so; what it must never do is read
it as a pass.

THE COUNT ASSERTION, WHICH IS THE POINT

`--check` refuses when it finds fewer `FROM` records than `--expect`. Verified by experiment
rather than assumed: a check that discovers its own inputs and finds none exits 0 having
verified nothing, and a renamed or moved Dockerfile would therefore turn this gate green by
emptying it. That is the same failure shape as the lockfile job's `-eq 12` and the workflow
lint's file count, and it is the one this project keeps rediscovering.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import baseimages  # noqa: E402 - after the sys.path insert, as the other tools do

#: The Dockerfiles in scope, relative to the repository root. `head-to-head/Dockerfile.tool` is
#: included deliberately even though it was already compliant: a gate scoped to the files that
#: were broken is a gate that stops noticing the day a compliant file regresses.
DOCKERFILES = ("Dockerfile", "Dockerfile.cuda", "head-to-head/Dockerfile.tool")

#: How many `FROM` records the tree is expected to hold. Two per published image plus one for the
#: benchmark image. Stated here AND derived from the files, so a truncation on either side shows.
EXPECTED_FROM_RECORDS = 5

#: Docker Hub's read-only tag endpoint. It answers without a credential and returns the
#: manifest-list digest, which is the digest a `FROM` line wants: it still selects the right
#: architecture, it just names an exact set of bytes while doing it.
HUB_TAG_URL = "https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"

TIMEOUT_S = 30


def _hub_repo(image):
    """Docker Hub's path for an image name, or None if it is not a Docker Hub image.

    `python` is `library/python` there, because an unqualified official image lives under the
    `library` namespace. Anything carrying a registry host (a dot or a port before the first
    slash) is somebody else's registry and is not guessed at.
    """
    name, _, _tag = image.partition(":")
    head = name.split("/", 1)[0]
    if "." in head or ":" in head:
        return None
    if "/" not in name:
        return f"library/{name}"
    return name


def resolve_tag(image, *, timeout=TIMEOUT_S, opener=None):
    """`(digest, published)` for an image reference's tag, or `(None, None)` if it cannot be read.

    `(None, None)` rather than an exception, because an unreachable registry must degrade to
    UNCHECKED rather than take a run down. The distinction is made here so no caller can treat a
    failure as "no newer digest", which would read as current and quietly stop being a check.
    """
    repo = _hub_repo(image)
    if repo is None:
        return None, None
    _name, _, tag = image.partition(":")
    url = HUB_TAG_URL.format(repo=repo, tag=tag or "latest")
    req = urllib.request.Request(url, headers={     # noqa: S310 - fixed https host
        "Accept": "application/json",
        "User-Agent": "senbonzakura-base-image-pin-check",
    })
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as response:
            doc = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError, KeyError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    digest = doc.get("digest")
    published = doc.get("last_updated")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None, None
    return digest, published if isinstance(published, str) else None


def collect(root=ROOT, files=DOCKERFILES):
    """Every `FROM` record across the Dockerfiles in scope, and the files that were missing.

    A missing file is returned rather than skipped. A gate that silently ignores a Dockerfile it
    could not find is a gate a rename switches off.
    """
    records, missing = [], []
    for relative in files:
        path = Path(root) / relative
        if not path.is_file():
            missing.append(relative)
            continue
        records.extend(baseimages.parse_from_lines(
            path.read_text(encoding="utf-8"), path=relative))
    return records, missing


def _summary(lines):
    """Append to the run summary when there is one. Running by hand must not need a runner."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def cmd_check(args):
    try:
        records, missing = collect(root=args.root or ROOT, files=args.files or DOCKERFILES)
    except baseimages.BaseImageError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    if missing:
        print(f"::error::DID NOT RUN: {', '.join(missing)} not found, so the base images in "
              f"those files were not checked. A gate a rename switches off is not a gate.",
              file=sys.stderr)
        return 1

    print(f"base image references found: {len(records)}")
    for record in records:
        state = "digest" if record["digest"] else "TAG ONLY"
        print(f"  {record['path']}:{record['line']}  {record['ref']}  [{state}]")

    if len(records) < args.expect:
        print(f"::error::found {len(records)} FROM records and expected at least {args.expect}. "
              f"A check over fewer base images than the repository has is not a check. If a "
              f"Dockerfile or a build stage was deliberately removed, lower --expect in the same "
              f"commit.", file=sys.stderr)
        return 1

    loose = baseimages.unpinned(records)
    if loose:
        for record in loose:
            print(f"::error file={record['path']},line={record['line']}::"
                  f"{record['ref']} is pinned by tag, not by digest. A tag is a name the registry "
                  f"can repoint at any time, so this build is neither reproducible nor auditable. "
                  f"Resolve the digest and write `FROM {record['ref']}@sha256:<digest>`.",
                  file=sys.stderr)
        _summary([
            f"### Base images: {len(loose)} of {len(records)} pinned by tag rather than digest",
            "",
            *(f"- `{r['path']}:{r['line']}` &mdash; `{r['ref']}`" for r in loose),
            "",
            "Baseline section 5: container base images are pinned by digest, never by tag.",
        ])
        return 1

    print(f"PASS: all {len(records)} base image references are digest-pinned.")
    _summary([f"### Base images: PASS ({len(records)} references, all digest-pinned)"])
    return 0


def cmd_drift(args):
    try:
        records, missing = collect(root=args.root or ROOT, files=args.files or DOCKERFILES)
    except baseimages.BaseImageError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    if missing:
        print(f"::error::DID NOT RUN: {', '.join(missing)} not found.", file=sys.stderr)
        return 1

    today = args.today or dt.datetime.now(dt.timezone.utc).date().isoformat()

    # One registry call per DISTINCT image reference. Both stages of a Dockerfile share a base by
    # design, so asking twice would double the request count for the same answer.
    seen, verdicts = {}, []
    for record in records:
        key = record["image"]
        if key not in seen:
            seen[key] = resolve_tag(key)
        digest, published = seen[key]
        verdicts.append({**baseimages.drift_verdict(record, digest, published, today),
                         "path": record["path"], "line": record["line"]})

    print(f"base image pins, checked {today} against a {args.cooldown}-day cooldown:")
    print(baseimages.format_report(verdicts))

    actions = [v["action"] for v in verdicts]
    if all(action == "unchecked" for action in actions):
        print("::warning::DID NOT RUN: not one base image could be resolved, so no pin was "
              "compared. This is a network result, not a pass.")
        _summary(["### Base images: drift check DID NOT RUN",
                  "",
                  "The registry answered for none of the pins, so nothing was compared."])
        return 2

    due = [v for v in verdicts if v["action"] == "refresh"]
    cooling = [v for v in verdicts if v["action"] == "cooling"]
    unchecked = [v for v in verdicts if v["action"] == "unchecked"]

    for verdict in due:
        print(f"::warning file={verdict['path']},line={verdict['line']}::{verdict['why']}")
    for verdict in unchecked:
        print(f"::warning::{verdict['why']}")

    _summary([
        (f"### Base images: {len(due)} refresh due, {len(cooling)} cooling, "
         f"{len(unchecked)} unchecked, "
         f"{len(verdicts) - len(due) - len(cooling) - len(unchecked)} current"),
        "",
        "```",
        baseimages.format_report(verdicts),
        "```",
        "",
        ("A refresh is proposed, never applied. A tag moving is not automatically an attack "
         "and a job that re-pinned by itself would erase the evidence of the one case where it "
         "is."),
    ])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="offline: every FROM must be digest-pinned (default)")
    mode.add_argument("--drift", action="store_true",
                      help="ask the registry whether each tag still resolves to its pinned digest")
    parser.add_argument("--expect", type=int, default=EXPECTED_FROM_RECORDS,
                        help="the floor on FROM records found, so an emptied scope cannot pass")
    parser.add_argument("--cooldown", type=int, default=baseimages.COOLDOWN_DAYS,
                        help="days a newer digest must have existed before a refresh is proposed")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Dockerfiles to read, relative to the repository root")
    # Exists so the gate's own guards can be driven against planted defects in a temporary tree.
    # Without it a test could only ever point the tool at the real Dockerfiles, so the refusal
    # paths, which are the half that matters, would have no way of being exercised at all.
    parser.add_argument("--root", default=None,
                        help="the tree to read the Dockerfiles from (default: this repository)")
    parser.add_argument("--today", default=None,
                        help="ISO date to evaluate against, for testing the cooldown boundary")
    args = parser.parse_args(argv)
    return cmd_drift(args) if args.drift else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
