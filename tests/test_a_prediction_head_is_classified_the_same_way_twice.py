# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Four answers were published to one question, so the classifier gets a test rather than a regex.

"How many checkpoints keep their prediction head inside the decoder stack" was answered 25, then
15, then 13, then 15 again over three days, each time by a hand-written query over the corpus and
each time reported as a fact about checkpoints. Two of those readings were wrong in ways a fixture
can hold still, and both are below.

The real corpus lives under `private/`, which a checkout does not have, so nothing here reads it.
These are hand-built records carrying the exact shapes that produced the wrong numbers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "count_prediction_heads",
    Path(__file__).resolve().parent.parent / "tools" / "research" / "count_prediction_heads.py")
heads = importlib.util.module_from_spec(_spec)
sys.modules["count_prediction_heads"] = heads
_spec.loader.exec_module(heads)


def record(config, counts, patterns=()):
    return {"status": "complete",
            "config": config,
            "tensors": {"layer_counts_by_stem": counts,
                        "pattern_layer_indices": dict.fromkeys(patterns, {})}}


def test_a_head_on_trailing_indices_is_in_stack():
    verdict, detail = heads.classify(record(
        {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}, {"model.layers": 62}))
    assert verdict == "in_stack"
    assert detail["extra"] == 1


def test_a_head_in_its_own_stack_is_not_in_stack():
    verdict, _ = heads.classify(record(
        {"num_hidden_layers": 48, "num_nextn_predict_layers": 1},
        {"model.layers": 48, "model.mtp.layers": 1}))
    assert verdict == "own_stack"


def test_the_declared_count_is_found_when_it_is_nested_under_text_config():
    """THIS IS THE ONE THAT MADE THE ANSWER 13.

    A multimodal checkpoint keeps the decoder's own layer count under `text_config`, so reading
    `config["num_hidden_layers"]` returns None. Two records fell out of the count that way and
    were reported as not-in-stack rather than as not-read.
    """
    verdict, detail = heads.classify(record(
        {"text_config": {"num_hidden_layers": 45}},
        {"model.language_model.layers": 46, "model.visual.blocks": 24},
        patterns=("model.language_model.layers.{i}.eh_proj.weight",)))
    assert verdict == "in_stack", "a nested layer count still describes a decoder stack"
    assert detail["declared"] == 45
    assert detail["declared_from"] == "text_config"
    assert detail["extra"] == 1


def test_a_vision_tower_is_never_mistaken_for_the_decoder_stack():
    """The longest stem in a multimodal record can be an image encoder.

    47 blocks of `vision_model.transformer.resblocks` says nothing about a prediction head, and
    comparing the BIGGEST stem against the declared count turns one into evidence about the other.
    """
    verdict, detail = heads.classify(record(
        {"text_config": {"num_hidden_layers": 92}},
        {"model.layers": 95, "vision_model.transformer.resblocks": 470},
        patterns=("model.layers.{i}.eh_proj.weight",)))
    assert verdict == "in_stack"
    assert detail["stem"] == "model.layers"
    assert detail["extra"] == 3, "95 against 92, not 470 against 92"


def test_a_head_with_no_config_key_is_still_found():
    """THIS IS THE ONE THAT MADE THE OWN-STACK ANSWER 11.

    Nine records carry the head's own tensors while declaring neither count field, so asking the
    config alone whether a head exists reports every one of them as having none.
    """
    verdict, detail = heads.classify(record(
        {"num_hidden_layers": 48},
        {"model.layers": 48, "model.mtp.layers": 1},
        patterns=("model.mtp.layers.{i}.eh_proj.weight", "model.mtp.layers.{i}.enorm.weight")))
    assert verdict == "own_stack", "found by its markers, with no config key to declare it"
    assert detail["markers"]
    assert not detail["config_keys"]


def test_a_checkpoint_with_no_head_at_all_is_not_counted_either_way():
    verdict, _ = heads.classify(record({"num_hidden_layers": 32}, {"model.layers": 32}))
    assert verdict == "no_head"


def test_a_record_that_cannot_be_read_says_so_rather_than_counting_as_absent():
    """The whole point of the fourth bucket. An unreadable record is not a finding."""
    verdict, detail = heads.classify(record(
        {"num_nextn_predict_layers": 1}, {"model.layers": 40}))
    assert verdict == "undetermined"
    assert "declared layer count" in detail["reason"]


def test_an_empty_corpus_is_an_error_and_not_a_count_of_zero():
    """A run that read nothing must not look like a run that found nothing.

    pytest exits 5 on an empty collection for the same reason: the two are indistinguishable in a
    tail of output, and this project has twice reported one as the other.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert heads.main([d]) == 2, "an empty corpus has to be loud"


def test_the_marker_set_and_the_nested_keys_are_not_silently_empty():
    # A classifier whose witness lists were emptied would call everything `no_head` and read as a
    # corpus with no prediction heads in it.
    assert heads.HEAD_MARKERS
    assert heads.NESTED_CONFIGS
    assert heads.NOT_THE_DECODER


@pytest.mark.parametrize("stem", ["vision_model.blocks", "model.visual.blocks",
                                  "vision_tower.resblocks", "model.ngram_embeddings.embedders"])
def test_every_non_decoder_stem_shape_is_excluded(stem):
    verdict, detail = heads.classify(record(
        {"num_hidden_layers": 40, "num_nextn_predict_layers": 1}, {"model.layers": 40, stem: 400}))
    assert verdict == "own_stack", f"{stem} was treated as the decoder stack"
    assert detail["stem"] == "model.layers"
