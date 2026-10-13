# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The deflation order harness measures the two orders' disagreement and decides nothing.

The properties that make its report worth reading: the orders agree where the maths says they must
(one direction, no mask, no rounds), they disagree where the maths says they can, two runs on the
same seed are byte identical, and the report states what it did not decide.
"""
import importlib.util
import json
import pathlib

import pytest
import torch
from test_the_budget_refuses_before_the_run import build_checkpoint

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "research" / "deflation_order.py"
spec = importlib.util.spec_from_file_location("deflation_order", TOOL)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

NOISE = 1e-6


def stack():
    return harness.synthetic_weights((4, 64, 96), 0)[0][1]


@pytest.mark.parametrize(("k", "sparsity", "rounds"), [(1, 0.0, 0), (1, 0.3, 3), (3, 0.0, 0),
                                                       (3, 0.3, 0), (3, 0.0, 3)])
def test_the_orders_agree_wherever_the_maths_says_they_must(k, sparsity, rounds):
    out = harness.divergence(stack(), harness.basis(64, k, 0), sparsity=sparsity, rounds=rounds)
    assert out["relative_frobenius"] < NOISE


def test_the_orders_disagree_with_the_mask_and_the_rounds_both_on():
    out = harness.divergence(stack(), harness.basis(64, 3, 0), sparsity=0.3, rounds=3)
    assert out["relative_frobenius"] > 10 * NOISE
    assert out["elements_differing"] > 0


def test_the_input_weight_is_never_modified():
    weight = stack()
    before = weight.clone()
    harness.divergence(weight, harness.basis(64, 3, 0), sparsity=0.3, rounds=3)
    assert torch.equal(weight, before)


def test_a_two_dimensional_writer_is_accepted_as_a_stack_of_one():
    flat = stack()[0]
    out = harness.divergence(flat, harness.basis(64, 3, 0), sparsity=0.3, rounds=3)
    assert out["elements"] == flat.numel()


def test_the_summary_separates_the_cells_where_both_flags_are_on():
    rows = harness.sweep(harness.synthetic_weights((4, 64, 96), 0), directions=(1, 3),
                         sparsities=(0.0, 0.3), rounds=(0, 3), seed=0)
    summary = harness.summarise(rows)
    assert summary["cells"] == 8
    assert summary["worst_when_both_flags_on"] > summary["worst_everywhere_else"]


def test_a_grid_with_no_active_cell_has_no_active_worst_case():
    rows = harness.sweep(harness.synthetic_weights((2, 16, 16), 0), directions=(1,),
                         sparsities=(0.0,), rounds=(0,), seed=0)
    assert harness.summarise(rows)["worst_when_both_flags_on"] is None


def run(tmp_path, name, *extra):
    out = tmp_path / name
    code = harness.main(["--directions", "1", "3", "--sparsities", "0", "0.3", "--rounds", "0",
                         "3", "--out", str(out), *extra])
    assert code == 0
    return out.read_bytes()


def test_two_runs_on_the_same_seed_write_identical_bytes(tmp_path):
    assert run(tmp_path, "a.json") == run(tmp_path, "b.json")


def test_a_different_seed_changes_the_numbers_and_the_provenance(tmp_path):
    one, two = run(tmp_path, "a.json"), run(tmp_path, "b.json", "--seed", "9")
    assert one != two
    assert json.loads(two)["provenance"]["seed"] == 9


def test_the_report_states_that_it_decides_nothing_and_what_basis_it_used(tmp_path):
    prov = json.loads(run(tmp_path, "a.json"))["provenance"]
    assert "nothing" in prov["decides"]
    assert "not a measured refusal direction" in prov["basis"]


def test_stdout_is_used_when_no_out_is_given(capsys):
    assert harness.main(["--directions", "1", "--sparsities", "0", "--rounds", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["cells"] == 1


def test_a_real_checkpoint_is_read_through_the_streaming_index(tmp_path, capsys):
    root = build_checkpoint(tmp_path / "m", hidden=64, inter=96)
    code = harness.main(["--model", str(root), "--max-tensors", "2", "--directions", "2",
                         "--sparsities", "0", "--rounds", "0"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["provenance"]["tensors"]) == 2
    assert report["provenance"]["source"] == str(root)


def test_a_directory_that_is_not_a_checkpoint_is_refused_with_a_sentence(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 0}))
    assert harness.main(["--model", str(tmp_path)]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_a_checkpoint_with_no_writers_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(harness, "checkpoint_weights", lambda *a: [])
    assert harness.main(["--model", str(tmp_path)]) == 2
    assert "no residual writers" in capsys.readouterr().err


def test_a_non_positive_tensor_cap_is_refused(capsys):
    assert harness.main(["--max-tensors", "0"]) == 2
    assert "at least 1" in capsys.readouterr().err


def test_making_the_batched_side_follow_the_other_order_silences_the_active_cells(monkeypatch):
    """The control: if the divergence were a harness bug it would survive this."""
    def per_direction(block, rows, strength, sparsity, rounds, *, restore_norms):
        for e in range(block.shape[0]):
            harness.cli.orthogonalize_np_(block[e], rows, strength, sparsity=sparsity,
                                          rounds=rounds, restore_norms=restore_norms)

    monkeypatch.setattr(harness.cli, "_orthogonalize_np_3d_block_", per_direction)
    out = harness.divergence(stack(), harness.basis(64, 3, 0), sparsity=0.3, rounds=3)
    assert out["relative_frobenius"] == 0.0
