# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker, run unaided over measurement artefacts as they were while a wrong figure was live.

WHY THIS IS NOT THE SEEDED CORPUS

`test_seeded_incidents.py` runs the checker over reconstructions written against the checks, so a
pass there shows the checks work and says nothing about defects nobody built a file for. The files
in `fixtures/recovered/` are different. They are the committed blobs from git history, as they were
written on the day a wrong figure was published, with one home-directory path replaced by a
placeholder. Nothing below names a check to the checker: each file goes through `check_document`
exactly as a stranger's file would, and the test asserts only which findings come back.

THE FAILURE THIS EXISTS TO CATCH: the 2026-07-30 compass figures, AUC 0.9636 for Qwen3-1.7B, were
published with the verdict read at the wrong token position, so every prompt in both arms came out
HARMFUL. The first version of the 1.7B file carried that signature in its own fields, and before
the verdict-split check the checker reported "nothing found" on it. A check that fires on the
defective file and stays silent on the corrected one is the only result that means anything here.

WHAT IS NOT PROVEN BY THE 1.7B CATCH: the verdict-split check was written AFTER this file was read,
so its catch is post-hoc. The honest test of its coverage is an incident it was not written against,
which the cold-incident row of the v0.8 rung still owes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from senbonzakura_check import check_document, normalise

ROOT = Path(__file__).resolve().parent.parent
RECOVERED = ROOT / "tests" / "fixtures" / "recovered"

AS_FIRST_PUBLISHED_17B = RECOVERED / "2026-07-30-compass-qwen3-1.7b-as-first-published.json"
AS_FIRST_PUBLISHED_06B = RECOVERED / "2026-07-30-compass-qwen3-0.6b-as-first-published.json"
CORRECTED_17B = ROOT / "evidence" / "compass-2026-07-30" / "base-qwen3-1.7b.json"
CORRECTED_06B = ROOT / "evidence" / "compass-2026-07-30" / "base-qwen3-0.6b.json"
CORRECTED_1011 = ROOT / "evidence" / "compass-2026-10-11" / "base-qwen3-1.7b.json"

VERDICT_CONSTANT = "a-verdict-that-answers-the-same-to-every-item"
NULL_BEAT_INSTRUMENT = "a-figure-its-own-file-calls-invalid"

REQUIRED_PROVENANCE = ("incident", "documented_at", "kind", "inferred", "not_reproduced",
                       "recovered_from")
#: The keys a prompt-leak gate refuses anywhere in a committed artefact. A recovered file that
#: carried one would be a leak, and the scrub that makes these files safe to commit is not a licence.
FORBIDDEN_KEYS = {"prompt", "prompts", "generation", "generations"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finding_ids(path):
    findings, _skipped = check_document(_load(path))
    return {finding.check_id for finding in findings}


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_keys(value)


@pytest.mark.parametrize("path", [AS_FIRST_PUBLISHED_17B, AS_FIRST_PUBLISHED_06B],
                         ids=lambda p: p.stem)
def test_a_recovered_file_says_where_it_came_from_and_what_was_changed(path):
    """A recovered file with no provenance is an artefact nobody can tell from an invented one."""
    prov = _load(path).get("_provenance")
    assert isinstance(prov, dict), f"{path.name} carries no `_provenance` block"
    missing = [key for key in REQUIRED_PROVENANCE if not prov.get(key)]
    assert not missing, f"{path.name}: `_provenance` is missing {missing}"
    assert prov["kind"] == "reconstructed", (
        "a recovered file is reconstructed from the record, never invented; if this is "
        "not a recovered blob, it does not belong in this directory")


@pytest.mark.parametrize("path", [AS_FIRST_PUBLISHED_17B, AS_FIRST_PUBLISHED_06B],
                         ids=lambda p: p.stem)
def test_a_recovered_file_carries_no_home_path_and_no_prompt_field(path):
    """The repository is public. A scrub that missed a path, or a prompt-shaped key, is a leak."""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"/(home|Users)/", text), f"{path.name} names a home directory"
    leaked = sorted(FORBIDDEN_KEYS & set(_walk_keys(_load(path))))
    assert not leaked, f"{path.name} carries prompt-shaped keys {leaked}"


def test_unaided_run_raises_the_constant_verdict_on_the_1_7b_as_first_published():
    """The defect that was live in the first 1.7B file must come back as a finding, unaided.

    Before the verdict-split check existed this run returned "nothing found". The assertion is on
    the finding, not on the absence of an error, because a run that exits 0 and says nothing is the
    failure this suite exists to see.
    """
    assert VERDICT_CONSTANT in _finding_ids(AS_FIRST_PUBLISHED_17B)


def test_unaided_run_raises_the_constant_verdict_and_the_null_ruler_on_the_0_6b_as_first_published():
    """The 0.6B file was caught before this change, by a different rule, and must stay caught."""
    ids = _finding_ids(AS_FIRST_PUBLISHED_06B)
    assert {VERDICT_CONSTANT, NULL_BEAT_INSTRUMENT} <= ids


def test_the_corrected_1_7b_files_stay_silent():
    """The correct side of the pair. A check that fires on a corrected measurement is noise."""
    assert _finding_ids(CORRECTED_17B) == set()
    assert _finding_ids(CORRECTED_1011) == set()


def test_the_committed_0_6b_still_fires_because_its_verdict_is_still_constant():
    """The README already calls this file a null for exactly this reason, so the finding is right.

    It is pinned so that a later change which silences the check on this file has to say why, rather
    than passing quietly.
    """
    assert VERDICT_CONSTANT in _finding_ids(CORRECTED_06B)


def test_a_split_that_was_never_recorded_is_absent_not_zero():
    """`present` is how a check tells not-recorded from recorded, so the adapter must not invent one.

    A recorded zero is a real answer, a verdict that never fires, and must survive normalisation.
    """
    doc = _load(CORRECTED_17B)
    without = {k: v for k, v in doc.items()
               if k not in ("frac_harmful_positive", "frac_harmless_positive")}
    assert "harmful_arm_positive_fraction" not in normalise(without)
    assert "harmless_arm_positive_fraction" not in normalise(without)

    with_zero = dict(without, frac_harmful_positive=0.0, frac_harmless_positive=0.0)
    normalised = normalise(with_zero)
    assert normalised["harmful_arm_positive_fraction"] == 0.0
    assert normalised["harmless_arm_positive_fraction"] == 0.0


def test_a_split_stored_as_text_is_not_read_as_a_number():
    """Validate at the boundary: a string that looks like a fraction is not a recorded fraction."""
    doc = _load(CORRECTED_17B)
    doc["frac_harmful_positive"] = "1.0"
    doc["frac_harmless_positive"] = "1.0"
    normalised = normalise(doc)
    assert "harmful_arm_positive_fraction" not in normalised
    assert VERDICT_CONSTANT not in {f.check_id for f in check_document(doc)[0]}


#: The headline-against-control check. Its catch on the 1.7B is post-hoc in the same way as the verdict
#: check, because it was written after this file was read. It is pinned on the real published
#: head-to-head artefacts as well, which are the non-post-hoc half of its silence.
HEADLINE_CONTROL = "a-headline-that-its-own-control-does-not-reproduce"


def test_the_headline_control_check_fires_on_both_defective_files_and_not_on_the_corrected_ones():
    assert HEADLINE_CONTROL in _finding_ids(AS_FIRST_PUBLISHED_17B)
    assert HEADLINE_CONTROL in _finding_ids(AS_FIRST_PUBLISHED_06B)
    assert HEADLINE_CONTROL not in _finding_ids(CORRECTED_17B)
    assert HEADLINE_CONTROL not in _finding_ids(CORRECTED_1011)


def test_the_headline_control_check_is_silent_on_every_published_head_to_head_scored_artefact():
    """Ten real published figures, none written against this check. A check that fires on them
    would be firing on correct measurements, and that is the failure the silence half exists for.
    """
    scored = sorted((ROOT / "head-to-head" / "results" / "2026-09-10").glob("scored-*.json"))
    assert len(scored) == 10, f"expected the ten published scored arms, found {len(scored)}"
    fired = [p.name for p in scored if HEADLINE_CONTROL in _finding_ids(p)]
    assert not fired, f"the headline-control check fired on published correct artefacts: {fired}"
