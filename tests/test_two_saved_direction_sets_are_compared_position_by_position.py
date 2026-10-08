# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`subspace.compare_sets`: the pair-of-files question, which is the one somebody actually has.

`subspace.compare` already answers it for ONE span, and its tolerance is derived from measured
noise rather than chosen. What was missing is the whole file: a direction set is one span per
residual-stream position, and the question is whether `stream-extract` finds what
`abliterate --save-directions` finds, or whether two machines agree. Asking a position at a time
and reading it by eye is how a disagreement at position 19 gets missed.

The three transformations that are the same span are pinned here too, not because `compare` is
doubted, but because the whole point of this layer is that it must not lose that property while
looping: a per-position loop that orthonormalised per FILE rather than per position would.
"""
import numpy as np
import pytest
import torch

from senbonzakura import streambake, subspace


def _spans(positions=3, k=2, hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((positions, k, hidden))
    for i in range(positions):
        q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
        out[i] = q.T
    return out


# ── the pair agrees, and keeps agreeing under the three arbitrary transformations ──────

def test_a_set_compared_with_itself_agrees_at_every_position():
    a = _spans(positions=5)
    same, per_position = subspace.compare_sets(a, a.copy())
    assert same
    assert len(per_position) == 5
    assert all(c.same for c in per_position)


def test_a_sign_flip_is_the_same_subspace():
    """A singular vector's sign is arbitrary and the cosine between a vector and its negation is
    -1, which reads as "completely different" and means "identical".
    """
    a = _spans(seed=1)
    b = a.copy()
    b[:, 0, :] *= -1
    assert np.isclose(np.dot(a[0, 0], b[0, 0]), -1.0), "the premise: the cosine says -1"
    assert subspace.compare_sets(a, b)[0]


def test_a_rotation_inside_each_span_is_the_same_subspace():
    """A near-degenerate span comes back arbitrarily rotated, so two numerically identical
    implementations can show per-vector cosines near zero while spanning one space.
    """
    a = _spans(k=2, seed=2)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = np.stack([rot @ a[i] for i in range(a.shape[0])])
    assert abs(np.dot(a[0, 0], b[0, 0])) < 0.9, "the premise: the per-vector cosine has moved"
    assert subspace.compare_sets(a, b)[0]


def test_reordering_the_directions_is_the_same_subspace():
    a = _spans(k=3, seed=3)
    assert subspace.compare_sets(a, a[:, ::-1, :].copy())[0]


def test_a_bfloat16_round_trip_still_agrees():
    """`dirs_multi` is stored in bfloat16, so this is the noise the tolerance was derived against.
    A set that has been through the stored dtype must still compare as the same subspace, or every
    comparison between a file and a live run fails for the dtype rather than for the directions.
    """
    a = _spans(positions=4, k=2, hidden=32, seed=4)
    b = torch.as_tensor(a).to(torch.bfloat16).to(torch.float64).numpy()
    assert subspace.compare_sets(a, b)[0]


# ── the control, and that it says WHERE ───────────────────────────────────────────────

def test_one_replaced_position_is_caught_and_named():
    """The forced-fail control. A failure that does not localise sends somebody back to bisect by
    hand, which is most of the cost of having asked.
    """
    a = _spans(positions=4, k=2, hidden=8, seed=5)
    b = a.copy()
    b[2] = _spans(positions=1, k=2, hidden=8, seed=99)[0]
    same, per_position = subspace.compare_sets(a, b)
    assert not same
    assert [i for i, c in enumerate(per_position) if not c.same] == [2]


def test_the_report_names_every_disagreeing_position():
    a = _spans(positions=4, k=2, hidden=8, seed=6)
    b = a.copy()
    b[1] = _spans(positions=1, k=2, hidden=8, seed=98)[0]
    b[3] = _spans(positions=1, k=2, hidden=8, seed=97)[0]
    msgs = []
    same, per_position = subspace.compare_sets(a, b)
    assert subspace.describe_sets(same, per_position, log=msgs.append) is False
    text = "\n".join(msgs)
    assert "DIFFERENT at 2 of 4 position(s)" in text
    assert "position 1:" in text
    assert "position 3:" in text
    assert "position 2:" not in text


def test_an_agreeing_report_says_it_is_about_the_space_and_not_behaviour():
    """The caveat that stops this being quoted as a parity result for the edit: the same subspace
    applied at a different strength is a different edit.
    """
    a = _spans(seed=7)
    msgs = []
    subspace.describe_sets(*subspace.compare_sets(a, a.copy()), log=msgs.append)
    text = "\n".join(msgs)
    assert "SAME SUBSPACE at every position" in text
    assert "not about behaviour" in text


def test_an_empty_comparison_is_not_reported_as_agreement():
    """`all([])` is True, so a set with no positions would otherwise print a pass. An instrument
    that checked nothing has not agreed about anything.
    """
    msgs = []
    subspace.describe_sets(True, [], log=msgs.append)
    text = "\n".join(msgs)
    assert "nothing was checked" in text
    assert "not agreement" in text


def test_a_dropped_direction_is_a_rank_difference_no_tolerance_covers():
    hidden = 8
    a = np.zeros((1, 2, hidden))
    a[0, 0, 0] = a[0, 1, 1] = 1.0
    b = a.copy()
    b[0, 1] = 0.0
    same, per_position = subspace.compare_sets(a, b)
    assert not same
    assert per_position[0].rank_a == 2
    assert per_position[0].rank_b == 1
    assert "rank difference" in per_position[0].describe()


def test_two_all_zero_positions_agree_rather_than_raising():
    """Every unused slot in `dirs_multi` is zero, so a single-direction run has padded rows at
    every position. Raising here would fail a comparison of two perfectly valid runs.
    """
    a = np.zeros((2, 3, 5))
    same, per_position = subspace.compare_sets(a, a.copy())
    assert same
    assert per_position[0].rank_a == 0


# ── the refusals ──────────────────────────────────────────────────────────────────────

def test_a_different_number_of_positions_is_refused():
    with pytest.raises(ValueError) as e:
        subspace.compare_sets(np.zeros((3, 2, 4)), np.zeros((4, 2, 4)))
    assert "not two measurements of the same stack" in str(e.value)


def test_a_different_hidden_size_is_refused():
    with pytest.raises(ValueError) as e:
        subspace.compare_sets(np.zeros((2, 2, 4)), np.zeros((2, 2, 6)))
    assert "not two measurements of the same model" in str(e.value)


def test_a_wrong_shape_is_refused_with_both_shapes_named():
    with pytest.raises(ValueError) as e:
        subspace.compare_sets(np.zeros((2, 4)), np.zeros((3, 2, 4)))
    assert "(2, 4)" in str(e.value)


# ── the command ───────────────────────────────────────────────────────────────────────

def _write(path, arr, model="m"):
    streambake.save_directions(path, np.asarray(arr, dtype=np.float32), model=model,
                               mode="per_layer")
    return path


def test_the_command_exits_zero_on_agreement_and_one_on_disagreement(tmp_path, capsys):
    a = _spans(positions=2, k=2, hidden=8, seed=8)
    same = _write(tmp_path / "a.safetensors", a)
    copy = _write(tmp_path / "b.safetensors", a.copy())
    other = _write(tmp_path / "c.safetensors", _spans(positions=2, k=2, hidden=8, seed=9))

    assert subspace.main(["--compare", str(same), str(copy)]) == 0
    assert "SAME SUBSPACE" in capsys.readouterr().out
    assert subspace.main(["--compare", str(same), str(other)]) == 1
    assert "DIFFERENT at" in capsys.readouterr().out


def test_the_command_notes_when_the_two_files_name_different_models(tmp_path, capsys):
    a = _spans(positions=1, k=1, hidden=8, seed=10)
    one = _write(tmp_path / "one.safetensors", a, model="model-a")
    two = _write(tmp_path / "two.safetensors", a.copy(), model="model-b")
    subspace.main(["--compare", str(one), str(two)])
    out = capsys.readouterr().out
    assert "different models" in out
    assert "may be the models rather than the implementations" in out


def test_asking_for_a_file_comparison_and_a_synthetic_control_at_once_is_refused(tmp_path):
    """Two different questions in one command. The control breaks a synthetic pair on purpose and
    --compare reads two real files, so honouring both would mean reporting on neither.
    """
    a = _write(tmp_path / "a.safetensors", _spans(positions=1, k=1, hidden=8, seed=11))
    with pytest.raises(SystemExit) as e:
        subspace.main(["--compare", str(a), str(a), "--control", "replace"])
    assert "two different questions" in str(e.value)


def test_the_command_turns_an_unreadable_file_into_a_sentence(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not safetensors")
    with pytest.raises(SystemExit) as e:
        subspace.main(["--compare", str(bad), str(bad)])
    assert "subspace parity:" in str(e.value)


def test_the_self_check_still_works_with_no_compare(capsys):
    """The pre-existing behaviour, asserted here because --compare was added to its parser."""
    assert subspace.main([]) == 0
    assert "same subspace" in capsys.readouterr().out
