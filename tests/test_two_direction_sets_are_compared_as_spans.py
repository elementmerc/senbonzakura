# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The subspace parity instrument, and the four cases a per-vector comparison gets wrong.

THE POINT OF THE FIRST THREE TESTS. A sign flip, a rotation inside the span and a reordering are
all the SAME subspace, and all three make a per-vector cosine read as disagreement. Those three
are the reason this instrument exists rather than a cosine, so each one is pinned: if any of them
ever reports a difference, the instrument has become the thing it was built to replace.

The fourth is the forced-fail control. An instrument that only ever says yes has not been shown
to be able to say anything.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "research"))

import subspace_parity as sp


def _span(rng, positions=3, k=2, hidden=8):
    """A direction set whose rows are orthonormal, the shape extraction produces."""
    out = np.zeros((positions, k, hidden))
    for i in range(positions):
        q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
        out[i] = q.T
    return out


# ── the three transformations that are the same span ──────────────────────────────────

def test_a_sign_flip_is_the_same_span():
    """A singular vector's sign is arbitrary, and the cosine between a vector and its negation
    is -1, which reads as "completely different" and means "identical".
    """
    rng = np.random.default_rng(0)
    a = _span(rng)
    b = a.copy()
    b[:, 0, :] *= -1
    assert np.allclose([np.dot(a[0, 0], b[0, 0])], [-1.0]), "the premise: the cosine says -1"
    result = sp.compare(a, b)
    assert result["agrees"], (
        "a sign flip was read as a different subspace, which is the exact failure a per-vector "
        "cosine has and this instrument exists to avoid")
    assert result["worst_position"]["largest_angle_rad"] < 1e-9


def test_a_rotation_inside_the_span_is_the_same_span():
    """A near-degenerate subspace has an arbitrary rotation inside it, so two numerically
    identical implementations can return per-vector cosines near zero while spanning one space.
    """
    rng = np.random.default_rng(1)
    a = _span(rng, k=2)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = np.stack([rot @ a[i] for i in range(a.shape[0])])
    assert abs(np.dot(a[0, 0], b[0, 0])) < 0.9, "the premise: the per-vector cosine has moved"
    assert sp.compare(a, b)["agrees"]


def test_reordering_the_directions_is_the_same_span():
    rng = np.random.default_rng(2)
    a = _span(rng, k=3)
    b = a[:, ::-1, :].copy()
    assert sp.compare(a, b)["agrees"]


# ── the forced-fail control ───────────────────────────────────────────────────────────

def test_a_genuinely_different_span_is_reported_and_named():
    """The control. One position is replaced with an unrelated span and the result must say so,
    and must say WHICH position, because a failure that does not localise sends somebody back
    to bisect by hand.
    """
    rng = np.random.default_rng(3)
    a = _span(rng, positions=4, k=2, hidden=8)
    b = a.copy()
    other = _span(np.random.default_rng(99), positions=1, k=2, hidden=8)[0]
    b[2] = other
    result = sp.compare(a, b)
    assert not result["agrees"]
    assert result["positions_disagreeing"] == 1
    assert result["worst_position"]["position"] == 2
    assert result["per_position"][2]["agrees"] is False
    assert all(result["per_position"][i]["agrees"] for i in (0, 1, 3))


def test_a_tiny_tilt_is_caught_at_the_stated_tolerance():
    """The instrument's resolution, demonstrated rather than claimed. A rotation of one axis out
    of the span by a hundredth of a radian is ten times the tolerance and must be caught.
    """
    hidden = 8
    a = np.zeros((1, 2, hidden))
    a[0, 0, 0] = 1.0
    a[0, 1, 1] = 1.0
    b = a.copy()
    tilt = 1e-2
    b[0, 0] = np.array([np.cos(tilt), 0, np.sin(tilt)] + [0] * (hidden - 3))
    result = sp.compare(a, b)
    assert not result["agrees"]
    assert 5e-3 < result["worst_position"]["largest_angle_rad"] < 2e-2


def test_an_identical_set_agrees_at_every_position():
    rng = np.random.default_rng(4)
    a = _span(rng, positions=5)
    result = sp.compare(a, a.copy())
    assert result["agrees"]
    assert result["positions_disagreeing"] == 0
    assert result["positions_compared"] == 5


# ── the projector distance ────────────────────────────────────────────────────────────

def test_the_projector_distance_is_zero_for_the_same_span_and_one_for_orthogonal_spans():
    """The normalisation claim, checked at both ends. Normalised by `sqrt(2K)` so the figure means
    the same thing at every K, which is the property that makes it comparable across runs.
    """
    hidden = 6
    a = np.zeros((2, hidden))
    a[0, 0] = a[1, 1] = 1.0
    b = np.zeros((2, hidden))
    b[0, 2] = b[1, 3] = 1.0
    assert sp.projector_distance(a, a.copy()) == pytest.approx(0.0, abs=1e-12)
    assert sp.projector_distance(a, b) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_projector_distance_for_orthogonal_spans_does_not_depend_on_k(k):
    hidden = 12
    a = np.zeros((k, hidden))
    b = np.zeros((k, hidden))
    for i in range(k):
        a[i, i] = 1.0
        b[i, k + i] = 1.0
    assert sp.projector_distance(a, b) == pytest.approx(1.0, abs=1e-12)


# ── zero padding, which is not a direction ────────────────────────────────────────────

def test_a_padded_slot_is_not_counted_as_a_direction():
    """`dirs_multi` pads unused slots with zeros. Reading one as a basis vector would add an axis
    made of nothing, so the rank has to come from the populated rows alone.
    """
    hidden = 8
    a = np.zeros((1, 3, hidden))
    a[0, 0, 0] = 1.0
    result = sp.compare(a, a.copy())
    assert result["per_position"][0]["rank"] == 1


def test_two_all_zero_positions_agree_rather_than_raising():
    """A position where neither side ablates anything. Both spans are empty, which is agreement:
    they do the same nothing. Raising here would fail a comparison of two valid single-direction
    runs, because every unused slot is zero.
    """
    a = np.zeros((2, 2, 5))
    result = sp.compare(a, a.copy())
    assert result["agrees"]
    assert result["per_position"][0]["rank"] == 0


def test_a_rank_difference_is_the_finding_rather_than_an_error_to_swallow():
    hidden = 6
    a = np.zeros((1, 2, hidden))
    a[0, 0, 0] = a[0, 1, 1] = 1.0
    b = np.zeros((1, 2, hidden))
    b[0, 0, 0] = 1.0
    with pytest.raises(sp.SubspaceError) as e:
        sp.compare(a, b)
    msg = str(e.value)
    assert "2 and 1 dimensions" in msg
    assert "already the finding" in msg


def test_a_duplicated_direction_does_not_invent_an_axis():
    """Rank, not row count. Two copies of one vector span one dimension, and a QR that kept both
    columns would report two, making the comparison against a genuine single direction fail.
    """
    hidden = 6
    a = np.zeros((1, 2, hidden))
    a[0, 0, 0] = a[0, 1, 0] = 1.0
    b = np.zeros((1, 2, hidden))
    b[0, 0, 0] = 1.0
    assert sp.compare(a, b)["agrees"]


# ── the refusals ──────────────────────────────────────────────────────────────────────

def test_a_different_number_of_positions_is_refused():
    with pytest.raises(sp.SubspaceError) as e:
        sp.compare(np.zeros((3, 2, 4)), np.zeros((4, 2, 4)))
    assert "not two measurements of the same stack" in str(e.value)


def test_a_different_hidden_size_is_refused():
    a = np.zeros((1, 1, 4)); a[0, 0, 0] = 1.0
    b = np.zeros((1, 1, 6)); b[0, 0, 0] = 1.0
    with pytest.raises(sp.SubspaceError) as e:
        sp.compare(a, b)
    assert "hidden sizes differ" in str(e.value)


def test_a_wrong_shape_is_refused_with_both_shapes_named():
    with pytest.raises(sp.SubspaceError) as e:
        sp.compare(np.zeros((2, 4)), np.zeros((3, 2, 4)))
    assert "(2, 4)" in str(e.value)


def test_a_tolerance_of_zero_is_refused_rather_than_failing_everything():
    """A threshold that admits nothing makes every comparison fail, which looks like a finding."""
    with pytest.raises(sp.SubspaceError) as e:
        sp.compare(np.zeros((1, 1, 4)), np.zeros((1, 1, 4)), tolerance=0.0)
    assert "admits nothing" in str(e.value)


def test_the_tolerance_is_a_stated_constant_rather_than_read_off_a_run():
    """Pinned, because the failure mode is somebody widening it to make a run pass. 1e-3 rad sits
    between float32 reordering noise (near 1e-6) and any real disagreement, and that gap is what
    makes the exact value inside it unimportant.
    """
    assert sp.DEFAULT_TOLERANCE == 1e-3


# ── the command ───────────────────────────────────────────────────────────────────────

def test_the_command_exits_zero_on_agreement_and_one_on_disagreement(tmp_path, capsys):
    from senbonzakura import streambake

    rng = np.random.default_rng(5)
    a = _span(rng, positions=2, k=2, hidden=8).astype(np.float32)
    same = tmp_path / "a.safetensors"
    copy = tmp_path / "b.safetensors"
    other = tmp_path / "c.safetensors"
    streambake.save_directions(same, a, model="m", mode="per_layer")
    streambake.save_directions(copy, a.copy(), model="m", mode="per_layer")
    streambake.save_directions(other, _span(np.random.default_rng(6), positions=2, k=2,
                                            hidden=8).astype(np.float32),
                               model="m", mode="per_layer")

    assert sp.main([str(same), str(copy)]) == 0
    assert "SAME SPAN" in capsys.readouterr().err
    assert sp.main([str(same), str(other)]) == 1
    assert "DIFFERENT" in capsys.readouterr().err


def test_the_command_notes_when_the_two_files_name_different_models(tmp_path, capsys):
    from senbonzakura import streambake

    rng = np.random.default_rng(7)
    a = _span(rng, positions=1, k=1, hidden=8).astype(np.float32)
    one = tmp_path / "one.safetensors"
    two = tmp_path / "two.safetensors"
    streambake.save_directions(one, a, model="model-a", mode="per_layer")
    streambake.save_directions(two, a.copy(), model="model-b", mode="per_layer")
    sp.main([str(one), str(two)])
    err = capsys.readouterr().err
    assert "different models" in err
    assert "may be the models rather than the implementations" in err


def test_the_command_writes_the_full_result_as_json(tmp_path, capsys):
    import json

    from senbonzakura import streambake

    a = _span(np.random.default_rng(8), positions=2, k=1, hidden=8).astype(np.float32)
    path = tmp_path / "a.safetensors"
    streambake.save_directions(path, a, model="m", mode="per_layer")
    sp.main([str(path), str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["agrees"] is True
    assert len(payload["per_position"]) == 2


def test_the_command_turns_an_unreadable_file_into_a_sentence(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not safetensors")
    with pytest.raises(SystemExit) as e:
        sp.main([str(bad), str(bad)])
    assert "subspace parity:" in str(e.value)


def test_the_report_says_the_verdict_is_about_the_space_and_not_behaviour():
    """The caveat that stops this being quoted as a parity result for the edit. Two identical
    subspaces applied at different strengths give different models.
    """
    rng = np.random.default_rng(9)
    a = _span(rng, positions=1)
    msgs = []
    sp.report(sp.compare(a, a.copy()), log=msgs.append)
    assert "not about behaviour" in "\n".join(msgs)
