# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A compass artefact pins the model revision and the corpus revision, not just their names.

`evidence/README.md` has required "the model id and revision" and "the dataset and its pinned
revision" of every committed result since the 2026-09-25 panel. Nothing recorded either.

The two artefacts in `evidence/compass-2026-07-30/` are listed in
`tests/test_evidence_carries_its_provenance.py` as exceptions for exactly these fields, described
as predating the rule. They did predate it. **The tool never caught up, so every new artefact
landed in the same hole**, and the exemption list was absorbing a live product gap rather than a
historical one. Found 2026-10-11 when the guard refused a fresh artefact and the honest answer
turned out not to be another exemption: both facts were obtainable, offline, from data the run
already had.

An id is not an identity. `Qwen/Qwen3-1.7B` names a repository somebody else can push to, and a
corpus can be rebuilt in place under an unchanged name, so two artefacts can agree on both names
and describe different measurements. That is the same argument `stamps.input_digest_of` makes about
prompt paths, and it has the same answer.

**The corpus half needed no new mechanism.** `crashsafe.provenance(corpus=)` has taken the entry
all along and `track.revision_entry` has built it, both docstrings arguing for precisely this
requirement. It was simply never passed from the writer. A mechanism wired to no caller is this
project's most repeated defect shape and this is another instance of it.
"""

import pytest

from senbonzakura import stamps
from senbonzakura.track import revision_entry


class _Cfg:
    def __init__(self, commit):
        self._commit_hash = commit


class _Model:
    def __init__(self, commit):
        self.config = _Cfg(commit)


# ── the model revision ───────────────────────────────────────────────────────────

def test_a_hub_model_yields_the_resolved_commit():
    """Read off the config transformers already resolved, which works offline.

    Measured on the evidence run of 2026-10-11: with `HF_HUB_OFFLINE=1` the loader still set
    `_commit_hash`, and it matched the snapshot directory name exactly.
    """
    sha = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert stamps.model_revision_of(_Model(sha)) == sha


@pytest.mark.parametrize("value", [
    "/home/someone/edited-model",   # a local load sets the path, not a revision
    "main",                         # a branch name is not a revision either
    "70d244cc",                     # an abbreviation cannot be resolved back
    "",
    None,
    object(),
])
def test_anything_that_is_not_a_full_revision_is_refused(value):
    """A placeholder that passes a presence check is worse than an absent field.

    `crashsafe.provenance` refuses a half-filled corpus entry for the same reason: it would
    "satisfy the gate's presence check while recording nothing a reader could fetch". A branch
    name is the sharpest case, because it looks like an answer and moves under you.
    """
    assert stamps.model_revision_of(_Model(value)) is None


def test_a_model_with_no_config_at_all_is_refused_rather_than_crashing():
    """The stamping path must not be the thing that fails a completed run."""
    assert stamps.model_revision_of(object()) is None


def test_a_forty_character_string_that_is_not_hex_is_refused():
    """Length alone is not the test, or any long branch name would pass."""
    assert stamps.model_revision_of(_Model("z" * 40)) is None


# ── the corpus revision ──────────────────────────────────────────────────────────

def test_the_bundled_track_pins_to_its_own_digest():
    """The corpus half, which needed only to be passed.

    The revision is the packed track's `sha256_of_tar`, so two runs naming `default/bad_eval_ds`
    can be told apart when the pack is rebuilt.
    """
    entry = revision_entry("default/bad_eval_ds")
    if entry is None:
        pytest.skip("no bundled track in this checkout, so there is nothing to pin")
    assert entry["id"] == "default/bad_eval_ds"
    assert entry["kind"] == "bundled-track"
    assert len(entry["revision"]) == 64, f"not a sha256: {entry['revision']!r}"


def test_the_writer_passes_the_corpus_entry_to_provenance():
    """Asserted against the source, because driving it needs a model and minutes.

    The point is the WIRING: `provenance(corpus=)` and `revision_entry` both existed and the
    writer called neither, which is why three weeks of artefacts went unpinned. A test that only
    checked the two helpers would have passed throughout that period.
    """
    import pathlib

    from senbonzakura import margin

    text = pathlib.Path(margin.__file__).read_text(encoding="utf-8")
    assert "corpus=revision_entry(" in text, (
        "margin no longer passes a corpus entry to provenance, so its artefacts name the corpus "
        "without pinning it and fail the rule in evidence/README.md")
    assert "stamps.model_revision_of(model)" in text, (
        "margin no longer records the model revision, so its artefacts name the model without "
        "pinning it")


def test_the_two_fields_are_the_ones_the_evidence_guard_looks_for():
    """The spellings have to match, or the fix passes here and the guard still refuses.

    This reads the requirement out of the guard rather than restating it, so a rename in either
    place is caught instead of quietly splitting the two files apart.
    """
    import importlib.util
    import pathlib

    guard_path = pathlib.Path(__file__).with_name("test_evidence_carries_its_provenance.py")
    spec = importlib.util.spec_from_file_location("_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    assert "model_revision" in guard.REQUIRED["model_revision"], (
        "the guard no longer accepts a top-level `model_revision`, which is where margin writes it")
    assert "provenance.corpus.revision" in guard.REQUIRED["dataset_revision"], (
        "the guard no longer accepts `provenance.corpus.revision`, which is where "
        "`crashsafe.provenance(corpus=)` puts it")
