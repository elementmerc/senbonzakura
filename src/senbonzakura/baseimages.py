# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Which container base images this project builds on, and whether their digest pins have rotted.

THE DEFECT THIS CLOSES

Baseline section 5 says container base images are pinned by digest, never by tag. Until
2026-10-01 the two Dockerfiles this project PUBLISHES read `FROM python:3.13-slim` and
`FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, both tags, while
`head-to-head/Dockerfile.tool` in the same tree carried a full digest. So the house rule was
being followed by the benchmark image and broken by the one a stranger pulls from GHCR.

WHY A TAG IS NOT AN IDENTITY

A tag is a name the registry can repoint at any time, and for `python:3.13-slim` it does, roughly
weekly, as Debian patches land. That is useful and it is also the reason a build from a tag is not
reproducible and not auditable: two builds a week apart contain different bytes and nothing in the
repository records which. A digest names the bytes, so a build either gets them or fails.

AND WHY PINNING ALONE IS HALF A CONTROL

A digest cannot move, which is the point, and that is also its cost: a pin freezes at whatever was
current the day somebody wrote it, and a frozen base stops receiving the patches the moving tag was
delivering. Baseline section 5's cooldown says what not to adopt (nothing younger than seven days,
because most compromised releases are caught within days); the operator's rule of 2026-08-17 adds
the other half, that every release refreshes every pin to the newest thing that HAS cleared the
cooldown. Adopt nothing young, adopt everything that has aged.

This module is the arithmetic for that, with no network in it, the same split
`senbonzakura.vendoring` already uses for the vendored llama.cpp binaries: pure functions here,
the registry calls in `tools/ci/check_base_images.py`, so every branch is testable without a
network and without waiting a week.

THE THREE QUESTIONS, KEPT SEPARATE ON PURPOSE

    is it pinned at all        offline, blocking, answered from the Dockerfile alone
    does the tag still resolve  needs the registry; a repointed tag is the event pinning
    to the recorded digest      exists to survive, so it is reported and never auto-applied
    has the newer digest aged   needs a publication date; proposes a refresh only past cooldown

Collapsing the second and third is the mistake that matters. A check that proposed every newer
digest the moment it appeared would be a cooldown violation wearing the costume of diligence,
and this project's own vendoring module carries the scar that taught it: llama.cpp publishes
several releases a day, two of them forty-two minutes apart.
"""
from __future__ import annotations

import re

from senbonzakura.vendoring import COOLDOWN_DAYS, VendorError, eligible

#: `FROM <image>[:<tag>][@sha256:<64 hex>] [AS <stage>]`, case-insensitive on the keywords because
#: Dockerfile instructions are. Deliberately strict about the digest: a 63-character hex string is
#: not a digest, and accepting one would make this gate report clean on a truncated pin.
FROM_LINE = re.compile(
    r"^\s*FROM\s+(?P<ref>\S+)(?:\s+(?i:AS)\s+(?P<stage>\S+))?\s*$",
    re.IGNORECASE,
)

#: An image reference split into its three parts. The digest group is what the gate is about.
REFERENCE = re.compile(
    r"^(?P<image>[^@:\s]+(?::[^@/\s]+)?)"      # name, with an optional tag
    r"(?:@(?P<algo>[a-z0-9]+):(?P<hex>[0-9a-f]+))?$",
)

#: Digest algorithms a registry actually serves. Anything else is refused rather than trusted:
#: an unknown algorithm with a plausible-looking hex string is the shape of a pin that pins
#: nothing.
DIGEST_WIDTHS = {"sha256": 64, "sha512": 128}

#: A `FROM` whose reference is one of these names something earlier in the same file (a named
#: build stage) or the empty parent, and neither is a base image to pin.
NOT_AN_IMAGE = {"scratch"}


class BaseImageError(VendorError):
    """A base image reference is unpinned, malformed, or not what the record says it is.

    Subclasses `VendorError` on purpose: a caller that already handles "a third-party pin is
    wrong" should not need a second except clause to handle the same category of problem one
    layer down.
    """


def parse_from_lines(text, *, path="<dockerfile>"):
    """Every `FROM` in a Dockerfile, as records, with build-stage references dropped.

    A multi-stage Dockerfile's later `FROM builder` names a stage declared above it, not an
    image, and treating it as an unpinned base image would make this gate cry wolf on every
    correctly-written file. Stages are tracked as they are declared rather than guessed at from
    whether the name happens to contain a slash.
    """
    stages, records = set(), []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if raw.lstrip().startswith("#"):
            continue
        match = FROM_LINE.match(raw)
        if match is None:
            continue
        ref, stage = match["ref"], match["stage"]
        if stage:
            stages.add(stage.lower())
        if ref.lower() in stages or ref.lower() in NOT_AN_IMAGE:
            continue
        records.append({
            "path": str(path),
            "line": lineno,
            "ref": ref,
            "stage": stage,
            **split_reference(ref, where=f"{path}:{lineno}"),
        })
    return records


def split_reference(ref, *, where="<reference>"):
    """`{"image", "digest"}` for one image reference. `digest` is None when it is pinned by tag.

    Raises rather than returning a half-understood reference. A reference this cannot parse is
    not evidence that it is unpinned, and reporting it as unpinned would be a guess presented as
    a finding.
    """
    match = REFERENCE.match(ref)
    if match is None:
        raise BaseImageError(
            f"{where}: cannot parse the image reference {ref!r}, so whether it is pinned by "
            f"digest is unknown. That is not the same as unpinned and is not reported as such.",
        )
    algo, hexdigest = match["algo"], match["hex"]
    if algo is None:
        return {"image": match["image"], "digest": None}
    expected = DIGEST_WIDTHS.get(algo)
    if expected is None:
        raise BaseImageError(
            f"{where}: {ref!r} carries a {algo!r} digest, which is not an algorithm a registry "
            f"serves. Known: {', '.join(sorted(DIGEST_WIDTHS))}.",
        )
    if len(hexdigest) != expected:
        raise BaseImageError(
            f"{where}: {ref!r} carries a {algo} digest {len(hexdigest)} characters long, and "
            f"{algo} digests are {expected}. A truncated digest matches nothing, so this would "
            f"fail at build time rather than pin anything.",
        )
    return {"image": match["image"], "digest": f"{algo}:{hexdigest}"}


def unpinned(records):
    """The records that name an image by tag alone, which is the thing baseline section 5 forbids."""
    return [r for r in records if r["digest"] is None]


def drift_verdict(record, live_digest, live_published, today, *, cooldown_days=COOLDOWN_DAYS):
    """Has this pin's tag moved, and if so may the new digest be adopted yet?

    Returns `{"action", "why", "live"}` with `action` in:

        current    the tag still resolves to the recorded digest. Nothing to do.
        cooling    the tag has moved and the new digest is younger than the cooldown. This is a
                   real state and NOT a finding: the correct response is to keep the pin.
        refresh    the tag has moved and the new digest has aged past the cooldown.
        unchecked  the registry could not be asked. Never reported as current, because not being
                   able to look is not the same as having looked.

    `cooling` exists as its own state rather than being folded into either neighbour, and that is
    the whole reason this function is not three lines. Folded into `current` it would hide a
    moving base; folded into `refresh` it would propose adopting bytes that are two days old,
    which is precisely what the cooldown is for.
    """
    image = record["image"]
    if live_digest is None:
        return {"action": "unchecked", "live": None,
                "why": (f"{image}: the registry did not answer, so whether the tag still "
                        f"resolves to the recorded digest is unknown.")}
    if record["digest"] is None:
        return {"action": "refresh", "live": live_digest,
                "why": (f"{image}: pinned by tag rather than by digest, so there is no recorded "
                        f"digest to compare. The tag resolves to {live_digest} today.")}
    if live_digest == record["digest"]:
        return {"action": "current", "live": live_digest,
                "why": f"{image}: the tag still resolves to the pinned digest."}
    if live_published is None:
        return {"action": "unchecked", "live": live_digest,
                "why": (f"{image}: the tag now resolves to {live_digest}, which is NOT the "
                        f"pinned digest, and the registry gave no publication date, so whether "
                        f"it has cleared the {cooldown_days}-day cooldown cannot be established. "
                        f"Keep the pin and look by hand.")}
    if not eligible(live_published, today, cooldown_days):
        return {"action": "cooling", "live": live_digest,
                "why": (f"{image}: the tag has moved to {live_digest}, published "
                        f"{live_published}, which has NOT cleared the {cooldown_days}-day "
                        f"cooldown. Keep the pin; this is the rule working, not a problem.")}
    return {"action": "refresh", "live": live_digest,
            "why": (f"{image}: the tag has moved to {live_digest}, published {live_published}, "
                    f"which has cleared the {cooldown_days}-day cooldown. Policy is to refresh "
                    f"every pin at every release, so update the FROM line and rebuild.")}


def format_report(verdicts):
    """The verdicts as lines somebody reads in a terminal or a run summary."""
    mark = {"current": "ok  ", "refresh": "DUE ", "cooling": "wait", "unchecked": "??  "}
    return "\n".join(f"  {mark.get(v['action'], '?')} {v['why']}" for v in verdicts)
