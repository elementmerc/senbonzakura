# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A config that declares an absurd depth is refused before any per-block list is built.

A config is untrusted input: a crafted or corrupt file can say it has a billion layers. Before the bound,
the map built a list with one entry per declared block, so the process ran out of memory before any caller
saw a count. Each test here is one way a declared number becomes a list, with the bound on either side of it.
"""
from __future__ import annotations

import pytest

from senbonzakura import modelmap


def test_a_billion_declared_layers_is_refused_with_the_reason_and_no_count():
    counts = modelmap.block_count({"model_type": "llama", "num_hidden_layers": 10**9})
    assert counts.total.known is False
    assert counts.declared.known is False
    assert "1,000,000,000 blocks" in counts.total.why
    assert f"at most {modelmap.MAX_DECLARED_BLOCKS:,}" in counts.total.why


def test_the_bound_is_inclusive_and_the_next_block_is_refused():
    at_bound = {"num_hidden_layers": modelmap.MAX_DECLARED_BLOCKS}
    past_bound = {"num_hidden_layers": modelmap.MAX_DECLARED_BLOCKS + 1}
    assert modelmap.block_count(at_bound).total.value == modelmap.MAX_DECLARED_BLOCKS
    assert modelmap.block_count(past_bound).total.known is False


def test_the_extra_prediction_blocks_count_against_the_bound():
    cfg = {"num_hidden_layers": modelmap.MAX_DECLARED_BLOCKS, "num_nextn_predict_layers": 1}
    total = modelmap.block_count(cfg).total
    assert total.known is False
    assert "declares" in total.why


def test_a_list_of_block_kinds_longer_than_the_bound_is_refused():
    cfg = {"layer_types": ["attention"] * (modelmap.MAX_DECLARED_BLOCKS + 1)}
    total = modelmap.block_count(cfg).total
    assert total.known is False
    assert "declares" in total.why


def test_a_list_at_the_bound_is_still_read():
    cfg = {"layer_types": ["attention"] * modelmap.MAX_DECLARED_BLOCKS}
    assert modelmap.block_count(cfg).total.value == modelmap.MAX_DECLARED_BLOCKS


def test_the_inverted_attention_list_does_not_build_a_per_layer_list_past_the_bound():
    cfg = {"num_hidden_layers": 10**9, "attn_layer_indices": [1, 2]}
    kinds = modelmap.mechanisms(cfg).kinds
    assert kinds.known is False
    assert "declares" in kinds.why


def test_a_dense_prefix_past_the_bound_is_refused_not_expanded():
    cfg = {"num_hidden_layers": 4, "first_k_dense_replace": 10**9, "n_routed_experts": 8}
    dense = modelmap.experts(cfg).dense_blocks
    assert dense.known is False
    assert "declares" in dense.why


def test_a_dense_prefix_within_the_bound_is_expanded_as_before():
    cfg = {"num_hidden_layers": 4, "first_k_dense_replace": 2, "n_routed_experts": 8}
    assert modelmap.experts(cfg).dense_blocks.value == [0, 1]


def test_a_billion_layers_describes_without_a_per_block_list():
    mapping = modelmap.describe(config={"model_type": "llama", "num_hidden_layers": 10**9,
                                        "hidden_size": 8})
    assert mapping.blocks.total.known is False
    assert "declares" in mapping.blocks.total.why


@pytest.mark.parametrize("bad", [10**6, 10**9])
def test_very_large_declared_depths_are_refused_for_every_spelling(bad):
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        assert modelmap.block_count({key: bad}).total.known is False
