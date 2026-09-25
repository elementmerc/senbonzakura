# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`score` could not say which rows its number came from, so it could never be gated.

THE DEFECT, found on 2026-09-25 while wiring the pinned fields. `margin` takes `--track`, reads
the manifest and can therefore stamp `partition: measure`. `score` took only `--skip`, so the
honest value it could stamp was `rows-from-N`, which never compares equal to a verified `measure`.
A refusal rate genuinely taken on the held-out tail read as incomparable with a baseline taken the
same way, and the gate refused a comparison that was perfectly sound.

The safe direction, and not the useful one. This is the repair: the same flag and the same
reconciliation `margin` already has, so a boundary a manifest confirms is stamped as the partition
it is, and a boundary nobody checked still is not.
"""
import json

import pytest

from senbonzakura import stamps, track


def _track(tmp_path, **manifest):
    """A manifest the real reader accepts.

    `counts` is not decoration here: `read_manifest` returns None for any document without one,
    deliberately, because a manifest is untrusted input that decides which rows every published
    number is measured on. A fixture missing it reads as "no manifest at all", which is how the
    first version of this file tested the unverified path four times while believing otherwise.
    """
    d = tmp_path / "trk"
    d.mkdir()
    doc = {"schema": "senbonzakura-track/1",
           "counts": {"harmful": {"fit": 128, "search": 131, "measure": 4741},
                      "harmless": {"fit": 320, "search": 4316, "measure": 4982}},
           **manifest}
    (d / "track.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


def test_the_boundary_is_read_from_the_manifest_and_counts_as_verified(tmp_path):
    t = _track(tmp_path, skip_harmful=259, skip_harmless=4636)
    skip, verified = track.resolve_skip_for_arm(t, None, "harmful")
    assert (skip, verified) == (259, True)
    assert stamps.partition_of(skip, verified=verified) == stamps.MEASURE


def test_each_arm_reads_its_own_boundary(tmp_path):
    """The reason there is no default arm: the two numbers are different."""
    t = _track(tmp_path, skip_harmful=259, skip_harmless=4636)
    assert track.resolve_skip_for_arm(t, None, "harmful")[0] == 259
    assert track.resolve_skip_for_arm(t, None, "harmless")[0] == 4636


def test_a_flag_that_agrees_with_the_manifest_is_still_verified(tmp_path):
    t = _track(tmp_path, skip_harmful=259)
    assert track.resolve_skip_for_arm(t, 259, "harmful") == (259, True)


def test_a_flag_that_contradicts_the_manifest_is_refused(tmp_path):
    """Not preferred, either way round. One of the two is wrong about the corpus."""
    t = _track(tmp_path, skip_harmful=259)
    with pytest.raises(SystemExit, match="contradicts the track"):
        track.resolve_skip_for_arm(t, 100, "harmful")


def test_a_flag_without_a_track_is_unverified(tmp_path):
    skip, verified = track.resolve_skip_for_arm(None, 259, "harmful")
    assert (skip, verified) == (259, False)
    assert stamps.partition_of(skip, verified=verified) == f"{stamps.UNVERIFIED_PREFIX}259"


def test_a_track_that_records_nothing_leaves_the_flag_unverified(tmp_path):
    t = _track(tmp_path)
    log = []
    skip, verified = track.resolve_skip_for_arm(t, 259, "harmful", log=log.append)
    assert (skip, verified) == (259, False)
    assert any("unverified" in line for line in log), log


def test_nothing_given_anywhere_reads_as_every_row(tmp_path):
    """Not a guess at a boundary. A run over the whole set includes the fitted rows and says so."""
    skip, verified = track.resolve_skip_for_arm(None, None, "harmful")
    assert (skip, verified) == (0, False)
    assert stamps.partition_of(skip, verified=verified) == stamps.ALL_ROWS


def test_an_unknown_arm_is_refused_rather_than_guessed(tmp_path):
    with pytest.raises(SystemExit, match="not an arm this track records"):
        track.resolve_skip_for_arm(_track(tmp_path, skip_harmful=1), None, "harmfull")


def test_no_track_means_the_arm_is_not_needed():
    """Without a manifest there is no recorded boundary for an arm to name, so requiring one
    would refuse every caller that never had a track, for a value that changes nothing."""
    assert track.resolve_skip_for_arm(None, 5, None) == (5, False)
    assert track.resolve_skip_for_arm(None, None, None) == (0, False)


def test_an_unverified_boundary_never_reads_as_the_measure_partition():
    """The property the whole change rests on: the two must not compare equal."""
    verified = stamps.partition_of(259, verified=True)
    unverified = stamps.partition_of(259, verified=False)
    assert verified == stamps.MEASURE
    assert verified != unverified


def test_two_identical_unverified_runs_still_compare():
    """Unverified is not a refusal to record. Same rows, same string, so they are comparable."""
    assert stamps.partition_of(259, verified=False) == stamps.partition_of(259, verified=False)
    assert stamps.partition_of(259, verified=False) != stamps.partition_of(260, verified=False)


def test_score_refuses_a_track_without_an_arm():
    from senbonzakura import score
    argv = ["--model", "m", "--eval", "e", "--out", "o.json", "--track", "t"]
    with pytest.raises(SystemExit, match="needs --track-arm"):
        score.main(argv)


def test_score_refuses_an_arm_without_a_track():
    from senbonzakura import score
    argv = ["--model", "m", "--eval", "e", "--out", "o.json", "--track-arm", "harmful"]
    with pytest.raises(SystemExit, match=r"manifest to read a boundary from"):
        score.main(argv)


def test_skip_defaults_to_none_so_zero_stays_a_choice():
    """`--skip 0` against a track that records a boundary has to be refusable as a contradiction,
    which is impossible if not passing the flag also looks like zero.
    """
    from senbonzakura import score
    a = score.build_parser().parse_args(["--model", "m", "--eval", "e", "--out", "o.json"])
    assert a.skip is None
    b = score.build_parser().parse_args(
        ["--model", "m", "--eval", "e", "--out", "o.json", "--skip", "0"])
    assert b.skip == 0
