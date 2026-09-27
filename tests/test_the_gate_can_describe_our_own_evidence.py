# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every artefact this project publishes gets a sentence that describes it, not a schema complaint.

WHAT PROMPTED IT, 2026-09-27

`baseline.read` learned to spot a result artefact handed to it by mistake, and said so usefully:
it looks for a `metrics` block and points at the command that records one. That fix shipped with a
known hole, stated in its own commit message: compass only started stamping `metrics` later, so
this project's own published evidence under `evidence/compass-2026-07-30/` still came back with
"declares schema None", a sentence about versioning for a file whose problem is that it is a
different KIND of thing.

A tool that cannot describe its author's own published figures is the exact failure this repository
criticises in other tools, which is why the test reads the committed evidence rather than a fixture
built to match. This project has been here before: a number that came from a fixture passed every
test and was wrong about the real file.

THE SHAPE OF THE RULE

Not "the gate accepts old evidence". It must not: nothing in those files records which metric the
AUC is, so accepting one would gate a later run on a property nobody chose, which is the failure
`only_metric` exists to prevent. The rule is that the refusal names the kind of file and a next
step that actually works, and in particular does NOT send the reader to `senbonzakura baseline`,
which reads the same missing block and would refuse a second time with a different sentence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from senbonzakura import baseline

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"


def _artefacts():
    if not EVIDENCE.is_dir():
        return []
    return sorted(p for p in EVIDENCE.rglob("*.json") if p.is_file())


def test_there_is_committed_evidence_to_read():
    """If this fails the suite below is vacuous, which is how a guard reports clean on nothing."""
    assert _artefacts(), f"no published evidence found under {EVIDENCE}, so nothing was checked"


@pytest.mark.parametrize("path", _artefacts(), ids=lambda p: p.name)
def test_every_published_artefact_gets_a_sentence_about_what_it_is(path):
    """Read every one. Either it is a baseline, or the refusal describes the file it actually got.

    The forbidden outcome is the schema sentence, because "declares schema None" tells a reader
    their file is the wrong VERSION when it is the wrong KIND, and sends them looking for a
    migration that does not exist.
    """
    try:
        baseline.read(path)
    except baseline.BaselineError as e:
        msg = str(e)
        assert "declares schema None" not in msg, (
            f"{path.name} is refused with a versioning complaint. Its problem is that it is not a "
            f"baseline at all, and the sentence has to say which kind of file it is:\n  {msg}")
        assert path.name in msg or str(path) in msg, (
            f"the refusal does not name the file it is about:\n  {msg}")


@pytest.mark.parametrize("path", _artefacts(), ids=lambda p: p.name)
def test_a_refusal_never_points_at_a_command_that_would_also_refuse(path):
    """The hint has to work. A correct diagnostic wired to a broken next step is worse than none.

    `senbonzakura baseline` reads the `metrics` block too, so recommending it for a file that has
    none buys the reader one more command and one more refusal.

    WHAT THIS ASSERTS, PRECISELY, because the first version of it was wrong

    It checks for the command applied to THIS FILE, not for the command being named at all. The
    pre-metrics refusal legitimately names it as the step after re-running compass, at which point
    the artefact does carry the block; a test that banned the string outright failed on a message
    that was correct, which is its own small lesson about checks that answer a broader question
    than the one being asked.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    try:
        baseline.read(path)
    except baseline.BaselineError as e:
        msg = str(e)
        pointed_at_this_file = f"senbonzakura baseline {path}" in msg
        if pointed_at_this_file:
            assert "metrics" in doc or "measurement" in doc, (
                f"{path.name} is sent to `senbonzakura baseline` ON ITSELF, and that command reads "
                f"a `metrics` block this file has not got, so it refuses too:\n  {msg}")


def test_the_pre_metrics_compass_evidence_is_recognised_by_name():
    """The specific file the hole was found on, asserted directly rather than only in the sweep."""
    path = EVIDENCE / "compass-2026-07-30" / "base-qwen3-0.6b.json"
    if not path.is_file():
        pytest.skip(f"{path} is not in this checkout")

    with pytest.raises(baseline.BaselineError) as e:
        baseline.read(path)
    msg = str(e.value)
    assert "compass" in msg, "the refusal has to say what kind of artefact this is"
    assert "re-measured" in msg or "re-run" in msg, (
        "a reader holding archived evidence needs to be told it cannot be converted")
    assert "declares schema" not in msg


def test_the_recogniser_needs_more_than_one_matching_key():
    """Three of four, so an artefact that merely carries an `auc` is not claimed as a compass run.

    A one-key test would have this function describing files it knows nothing about, which is the
    same class of error as the check it was written to fix.
    """
    assert not baseline._pre_metrics_compass({"auc": 0.9})
    assert not baseline._pre_metrics_compass({"auc": 0.9, "controls": {}})
    assert baseline._pre_metrics_compass({"auc": 0.9, "controls": {}, "n_harmful": 10})
