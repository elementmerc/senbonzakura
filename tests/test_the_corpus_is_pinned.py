# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`evidence/README.md` has demanded a pinned dataset revision since the 2026-09-25 panel, and
nothing produced one.

THE DEFECT, measured on 2026-10-01. `tests/test_evidence_carries_its_provenance.py` enforces that
README's rule, and it was added with two artefact directories already exempted. Checking 41
candidate artefacts from two unrelated runs, including ones built by a properly stamped source
checkout, found `dataset_revision` on exactly none of them, because `crashsafe.provenance()`
recorded no such field and nothing downstream added it. The rule was right, the gate was right, and
the producer was the gap, so the only way to commit new evidence was to widen an exemption list
that the file it lives in forbids widening by name.

WHY THE NAME WAS NEVER ENOUGH. A corpus can be rebuilt in place under a name that does not change.
`senbon-track-35axis-clean` is this project's own instance of that: it was the corpus every number
came from for months, 189 of its 200 harmful eval rows sat inside its own fitting set, and nothing
re-asked the question because it had been built before the checks existed. A name identifies which
corpus somebody meant. A digest identifies which bytes they got.

WHAT THIS FILE PINS DOWN is that the pin is derived the same way the corpus is resolved, that an
unpinnable corpus produces no field rather than a null one, and that a malformed entry is refused
loudly instead of being written as a half-pin that satisfies a presence check.
"""
import hashlib
import json
import pathlib

import pytest

from senbonzakura import track
from senbonzakura.crashsafe import provenance

ROOT = pathlib.Path(__file__).resolve().parent.parent


def requires_the_packed_track():
    """Skip where there is no packed track to pin, which is a real and supported state.

    THE PACKED TRACK IS NOT IN GIT. `tools/packaging/pack_track.py` produces it at build time, so a
    tree obtained with `git archive` has every tracked file and no track, which is exactly what the
    "suite runs outside a git checkout" job tests and exactly what a tarball consumer gets.

    `revision_entry` returns None there, by design and by its own docstring: it declines to write a
    null revision, because a present-but-empty field satisfies a presence check while telling a
    reader nothing. So None is the correct answer and subscripting it is the test's error, not the
    code's. These three assertions did that and failed with `TypeError: 'NoneType' object is not
    subscriptable`, which blamed the pin for the absence of the thing being pinned.

    Skipping rather than asserting None, because "no track here" is not evidence either way about
    whether a track that exists is pinned, and a test that passes by confirming an absence is the
    narrower-question defect this project keeps finding.
    """
    from senbonzakura import bundled
    if not bundled.is_available():
        pytest.skip("no packed track in this tree, so there is no pin to check; "
                    "run tools/packaging/pack_track.py to make one")


def test_the_bundled_track_is_pinned_by_the_digest_its_own_manifest_records():
    requires_the_packed_track()
    entry = track.revision_entry("default")
    assert entry["kind"] == "bundled-track"
    from senbonzakura import bundled
    assert entry["revision"] == bundled.manifest()["sha256_of_tar"]


def test_naming_a_partition_pins_the_track_rather_than_the_partition():
    """`default/bad_eval_ds` and `default` pin the same thing, and that is deliberate.

    The partition boundaries live in the track's manifest, so a partition digested on its own
    cannot tell a reader whether the boundary moved underneath it, which is the one thing the
    boundary is there to prove.
    """
    requires_the_packed_track()
    whole = track.revision_entry("default")
    partition = track.revision_entry("default/bad_eval_ds")
    assert partition["revision"] == whole["revision"]
    assert partition["id"] == "default/bad_eval_ds", "the id still says which rows were read"


def test_a_bundled_corpus_is_pinned_by_the_blob_that_carries_it():
    from senbonzakura import corpora
    blob = pathlib.Path(__import__("senbonzakura.bundled", fromlist=["x"]).data_path()).parent \
        / corpora.CORPORA_BLOB
    if not blob.is_file():
        pytest.skip("no corpora blob in this tree, so there is no pin to check; it is built at "
                    "packaging time and a `git archive` export carries neither it nor the track")
    entry = track.revision_entry("advbench")
    assert entry["kind"] == "bundled-corpus"
    from senbonzakura import bundled, corpora
    blob = pathlib.Path(bundled.data_path()).parent / corpora.CORPORA_BLOB
    assert entry["revision"] == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_a_track_directory_is_pinned_by_its_contents(tmp_path):
    d = tmp_path / "mytrack"
    d.mkdir()
    (d / "track.json").write_text(json.dumps({"counts": {"bad_ds": 1}}))
    (d / "rows.txt").write_text("one\n")
    first = track.revision_entry(str(d))
    assert first["kind"] == "track-directory"
    assert first["revision"] == track.dataset_digest(d)

    # Rebuilt in place under the same name, which is the failure the pin exists to catch.
    (d / "rows.txt").write_text("two\n")
    assert track.revision_entry(str(d))["revision"] != first["revision"], (
        "a corpus whose bytes changed under an unchanged name must not keep its revision")


def test_a_partition_inside_a_track_directory_resolves_to_the_track(tmp_path):
    d = tmp_path / "mytrack"
    (d / "bad_eval_ds").mkdir(parents=True)
    (d / "track.json").write_text(json.dumps({"counts": {"bad_eval_ds": 1}}))
    (d / "bad_eval_ds" / "rows.txt").write_text("one\n")
    assert track.revision_entry(str(d / "bad_eval_ds"))["revision"] == track.dataset_digest(d)


@pytest.mark.parametrize("spec", ["org/some-hub-dataset", "/nonexistent/path/anywhere"])
def test_what_cannot_be_pinned_produces_no_field_rather_than_a_null_one(spec):
    """None, not `{"revision": None}`.

    A Hub dataset's revision is a fact only the Hub holds, so this cannot pin it offline. Writing
    the key with a null value would satisfy the provenance gate's presence check while telling a
    reader nothing they could fetch, which is worse than the absence: it reads as answered. The
    artefact then carries no corpus block at all, exactly as it did before this existed, so nothing
    regresses for the corpora this cannot reach.
    """
    assert track.revision_entry(spec) is None


def test_provenance_omits_the_block_when_there_is_nothing_to_pin():
    assert "corpus" not in provenance(device="cpu")
    assert "corpus" not in provenance(device="cpu", corpus=None)


def test_provenance_records_the_block_when_there_is():
    p = provenance(device="cpu", corpus={"id": "default", "revision": "abc", "kind": "x"})
    assert p["corpus"]["revision"] == "abc"


def test_the_old_name_is_gone_rather_than_silently_accepted():
    """Q-52: the key was `track` for one afternoon, and `track` means something else nearby.

    A margin manifest already carries a top-level `track` naming the directory its skip boundary
    came from, beside a `provenance` block. Renaming to `corpus` cost nothing because no artefact
    had been written with the old key. This asserts the rename went all the way: a caller still
    passing `track=` must fail loudly rather than have the pin silently swallowed by `**extra` or
    ignored, because a quietly-dropped pin is the exact failure the pin exists to prevent.
    """
    with pytest.raises(TypeError, match="track"):
        provenance(device="cpu", track={"id": "default", "revision": "abc"})
    assert "track" not in provenance(
        device="cpu", corpus={"id": "default", "revision": "abc"})


@pytest.mark.parametrize("bad", [
    {},
    {"id": "default"},
    {"revision": "abc"},
    {"id": "default", "revision": ""},
    {"id": "", "revision": "abc"},
    {"id": "default", "revision": None},
    "default",
])
def test_a_half_pin_is_refused_rather_than_written(bad):
    """Loud, per §2.1, because the quiet version of this is what the gate could not see.

    An entry with a name and no digest is the exact shape of the thing being fixed. If it were
    written anyway, `provenance.corpus` would exist on every artefact and the gate would pass,
    while no artefact recorded which bytes produced its number.
    """
    with pytest.raises(ValueError, match="corpus provenance entry"):
        provenance(device="cpu", corpus=bad)


def test_every_provenance_call_in_score_pins_the_corpus_it_measured():
    """The producer is the half that was missing, so the call sites are what this has to hold.

    A mutation pass over 21 fixes on this project found six that nothing would have noticed, five
    of them wired to call sites no test checked. `revision_entry` passing its own unit tests says
    the pin can be built; it does not say any artefact carries one. Deleting `corpus=` from
    `score.py` would leave every test above green.

    Read from the source rather than by running a scoring pass, because a pass needs a model and a
    card. That is a weaker check than observing a written artefact and it is the one that runs
    everywhere, so it is deliberately strict: every `provenance(` in this module must pass
    `corpus`.

    The keyword this matches moved from `track=` to `corpus=` on 2026-10-01 (Q-52). It matches the
    keyword rather than the value because the value is a call, and the parameter name is the thing
    that has to stay wired.
    """
    src = (ROOT / "src/senbonzakura/score.py").read_text()
    calls = []
    i = 0
    while (i := src.find("provenance(device=", i)) != -1:
        # To the closing bracket of the call, which may be several lines below it.
        depth, start = 0, src.index("(", i)
        end = start
        for end in range(start, len(src)):
            depth += (src[end] == "(") - (src[end] == ")")
            if depth == 0:
                break
        calls.append(src[i:end + 1])
        i = end
    assert calls, "no provenance call found in score.py; this test has stopped measuring"
    unpinned = [c for c in calls if "corpus=" not in c]
    assert not unpinned, (
        f"{len(unpinned)} of {len(calls)} provenance calls in score.py record no corpus pin, so "
        f"the artefacts they write cannot satisfy evidence/README.md: {unpinned}")


def test_the_resolution_order_still_matches_the_one_the_corpus_is_read_through():
    """A pin resolved in a different order from the read describes a different corpus.

    `dataset.paths` resolves the bundled alias and the bundled corpus names BEFORE touching the
    filesystem, so a directory called `default` in the working directory cannot shadow the packed
    track. If `revision_entry` checked the filesystem first, a number taken from the packed track
    would be pinned to whatever that directory happened to contain, and the artefact would be
    confidently wrong rather than merely unpinned.
    """
    src = (ROOT / "src/senbonzakura/track.py").read_text()
    body = src[src.index("def revision_entry("):]
    body = body[:body.index("\ndef ")]
    alias_at = body.index("BUNDLED_ALIAS")
    corpora_at = body.index("corpora.CORPORA")
    fs_at = body.index("path.is_dir()")
    assert alias_at < corpora_at < fs_at, (
        "revision_entry must resolve the bundled alias, then the bundled corpus names, then the "
        "filesystem, in that order, because that is the order dataset.paths reads them in")
