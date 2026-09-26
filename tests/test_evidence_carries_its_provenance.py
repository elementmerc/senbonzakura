# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`evidence/README.md` sets a provenance rule, and until now nothing in `evidence/` met it.

THE DEFECT, found by the Reproducibility Referee on the 2026-09-25 panel. That README lists what
every committed result needs: the commit, the model id AND revision, the dataset AND its pinned
revision, the accelerator, the seeds, the device, both arm skips, and a pointer to the matching
`constraints/measured-*.md`. Measured against the tree, the compass artefacts carry no model
revision and no dataset identity at all, and the k-sweep summary carries seven of the ten not at
all. So a reader who obtained the gated dataset still could not re-take the headline figure,
because nothing records which revision of the dataset or of `Qwen/Qwen3-1.7B` was read, and both
can move upstream.

A rule that nothing meets is not a rule, it is an aspiration, and this project's own README says an
unsourced quoted number is a defect. So the rule is now enforced, with the artefacts that genuinely
predate it recorded as exceptions rather than silently tolerated: `EXEMPT` names each one, says what
it is missing, and says why it cannot be fixed rather than why nobody got round to it. Anything new
has to comply.

WHAT THIS FILE CANNOT DO is reconstruct the past. `constraints/measured-2026-07-27-compass-sweep.md`
is the record of what the July sweep cost, and it says the environment "cannot be reconstructed,
only re-run". This test exists so that sentence never has to be written again.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"

#: What `evidence/README.md` requires, as a field this test can look for.
#:
#: The names are ours; the requirement is the README's. Where an artefact may legitimately carry a
#: field under either of two spellings, both are listed, because the point is whether a reader can
#: find the fact and not whether the writer picked our favourite key.
REQUIRED = {
    "commit": ("provenance.senbonzakura.git.commit",),
    "model_id": ("model",),
    "model_revision": ("model_revision", "provenance.model.revision"),
    "dataset": ("track", "eval", "harmful", "dataset"),
    "dataset_revision": ("track_revision", "dataset_revision",
                         "provenance.track.revision"),
    "accelerator": ("provenance.accelerator", "provenance.device", "device"),
    "seeds": ("seed", "seeds"),
}

#: Artefacts that predate the rule, with what each is missing and why it stays missing.
#:
#: NOT A TOLERANCE LIST. Every entry is a measurement that cannot be re-annotated, because the run
#: that produced it did not capture the fact and inventing one would be worse than the gap. Each is
#: dated, and each is superseded by a re-measurement rather than by an edit.
EXEMPT = {
    "compass-2026-07-30/base-qwen3-0.6b.json": ("model_revision", "dataset", "dataset_revision"),
    "compass-2026-07-30/base-qwen3-1.7b.json": ("model_revision", "dataset", "dataset_revision"),
    # A hand-written summary of a run whose per-arm artefacts were never kept, not a tool output.
    # It carries what the run recorded and nothing else, which is why the panel found its
    # hard-refusal column unsourced until it was recovered from the run's own database.
    # `seeds` was in this list for about a minute and the narrowness test above removed it: the file
    # does carry them, under `seeds` rather than `seed`, and the script that first measured the gap
    # looked for the singular. That is the tolerance-creep this list is built to resist, caught on
    # its own author within a minute of being written.
    "k-sweep-2026-08-13/drift-per-seed.json": (
        "commit", "model_revision", "dataset", "dataset_revision", "accelerator"),
}


def _dig(doc, dotted):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _artefacts():
    return sorted(p for p in EVIDENCE.rglob("*.json"))


def _relative(path):
    return path.relative_to(EVIDENCE).as_posix()


def test_there_are_artefacts_to_check():
    """Without this, an empty or moved `evidence/` tree passes every test below."""
    found = _artefacts()
    assert found, "no artefacts under evidence/, so this whole file is reporting nothing"
    assert len(found) >= 3


@pytest.mark.parametrize("path", _artefacts(), ids=_relative)
def test_every_artefact_carries_what_the_readme_requires(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    exempt = set(EXEMPT.get(_relative(path), ()))
    missing = [name for name, keys in REQUIRED.items()
               if name not in exempt and all(_dig(doc, k) in (None, "", []) for k in keys)]
    assert not missing, (
        f"{_relative(path)} is missing {missing}, which `evidence/README.md` requires of every "
        f"committed result. Either the writer should record it, or, if this is an artefact from a "
        f"run that genuinely cannot be re-annotated, add it to EXEMPT with the reason. Do not "
        f"invent a value: a provenance field nobody measured is worse than an absent one, because "
        f"it reads as a receipt.")


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_every_exemption_still_names_a_real_artefact(name):
    """An exemption that outlives its file is an exemption covering something else."""
    assert (EVIDENCE / name).is_file(), (
        f"{name} is exempted from the provenance rule and does not exist. Remove the entry: a "
        f"stale exemption silently widens as new files arrive")


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_no_exemption_is_broader_than_it_needs_to_be(name):
    """The list shrinks as artefacts improve, and this is what makes it shrink.

    An exemption for a field the artefact actually carries is the beginning of a tolerance list.
    """
    doc = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    over = [field for field in EXEMPT[name]
            if any(_dig(doc, k) not in (None, "", []) for k in REQUIRED[field])]
    assert not over, (
        f"{name} is exempted from {over} and carries them. Narrow the exemption: every field left "
        f"in this list that does not need to be is a field nobody will notice going missing again")


def test_the_required_list_matches_the_readme():
    """The rule lives in the README and this file enforces it, so the two must not drift.

    Checked by wording rather than by structure, because the README is prose for a reader and
    turning it into a machine-readable list would move the rule out of the document that argues
    for it.
    """
    text = (EVIDENCE / "README.md").read_text(encoding="utf-8").lower()
    for phrase in ("commit", "revision", "accelerator", "seeds", "skips", "measured-"):
        assert phrase in text, (
            f"`evidence/README.md` no longer mentions {phrase!r}, so either the rule changed and "
            f"REQUIRED here is stale, or the document lost a requirement this test still imposes")
