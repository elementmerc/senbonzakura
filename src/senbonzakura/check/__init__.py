# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Read somebody else's evaluation artefact and report how the number could be wrong.

WHAT THIS IS FOR

Everything else in this package makes THIS project's numbers believable. This subpackage points
the same questions at an artefact somebody else produced. The argument for it being the wedge is
`private/plans/strategy-2026-08-02.md` Part 5; the short version is that the failure modes are
documented one at a time across a dozen issue trackers and assembled nowhere, so nobody has to
be convinced the problem is real.

TWO HARD CONSTRAINTS, both of which shape every file in here.

**It must import with no torch, no CUDA, no model and no corpus.** Reading a config and a result
file needs none of them, and `roadmap.md` calls this the largest single adoption lever this
project has. Nothing under `senbonzakura/check/` may import from a module that pulls the
deep-learning stack. That boundary fails quietly and only on somebody else's machine, so it is
gated by a test rather than left as a convention.

**It never writes to the artefacts it reads.** It is pointed at other people's evidence and the
one unforgivable behaviour is modifying it.

WHERE THIS IS GOING

Decision Q-29 (2026-09-11) settled that the checker ships as a SECOND DISTRIBUTION rather than
as an extra on this one, so that `pip install senbonzakura` keeps meaning "I want to abliterate
a model" (Q-27) while the checker installs in seconds on any machine. This subpackage is laid
out to be lifted out whole when that happens: it depends on nothing above it, and its data lives
beside it rather than in the parent package's `data/`.
"""
from .adapters import UnknownArtefactError, detect, normalise
from .registry import (
    Check,
    CheckError,
    Finding,
    evaluate,
    load_checks,
    run_checks,
)


def check_document(doc, checks=None):
    """Normalise an artefact from any supported harness, then run the checks over it.

    THE CHECKS READ THE NORMALISED DOCUMENT, NOT THE RAW ONE. That is what normalisation is for:
    one set of checks over every harness, rather than a parallel set per harness, which is the
    maintenance treadmill the artefact-level rule exists to avoid. The original is reachable at
    `raw.*` for the handful of questions that genuinely need a harness's own spelling.

    Raises `UnknownArtefactError` on a file nothing recognises. That is deliberate and is the
    v0.8 plan's rule: a silent pass on a document nobody parsed is the worst possible output,
    because it is indistinguishable from a clean bill of health.
    """
    return run_checks(normalise(doc), checks)


__all__ = [
    "Check", "CheckError", "Finding", "UnknownArtefactError",
    "check_document", "detect", "evaluate", "load_checks", "normalise", "run_checks",
]
