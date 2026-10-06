# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A checkpoint that declares NextN layers transformers never saves.

GLM-4.7-Flash declares `num_nextn_predict_layers: 1`. transformers does not save those layers,
and the converter counts them into `<arch>.block_count` anyway, so the GGUF promises a block its
tensors do not provide.

**It loads.** Measured 2026-10-04: the header declared 4 blocks, the tensors covered 3, and
`llama-imatrix` read the file and ran a chunk without complaint. Nothing downstream catches it,
so the first symptom is wrong output from a model that appeared to convert cleanly.

The vendored converter already carries `--no-nextn` / `--no-mtp` and was simply never given it.

A NOTE ON THE FIRST ATTEMPT AT THIS FIX, because the test exists to stop it coming back.
The first version rewrote `config.json`, subtracting the NextN count from `num_hidden_layers` on
the theory that the latter included the former. It does not: the fixture sets
`num_hidden_layers=3` for three decoder layers and declares the prediction layer on top. The
rewrite therefore deleted a real decoder layer, and the conversion check went GREEN, because its
assertion is that the header and the tensors agree and they now agreed about a smaller model.
A check can only compare what it was given.
"""

import json

import pytest

from senbonzakura.convert import MTP_LAYERS_KEY, mtp_layers


def test_a_config_declaring_none_asks_for_nothing():
    assert mtp_layers({"num_hidden_layers": 3}) == 0


def test_a_positive_declaration_is_counted():
    assert mtp_layers({MTP_LAYERS_KEY: 1}) == 1
    assert mtp_layers({MTP_LAYERS_KEY: 2}) == 2


@pytest.mark.parametrize("declared", [0, -1, "1", 1.5, None, [], {}])
def test_a_malformed_declaration_is_not_guessed_at(declared):
    assert mtp_layers({MTP_LAYERS_KEY: declared}) == 0, (
        "guessing at a malformed declaration would be a silent correction, which is the exact "
        "class of failure this fix exists to remove")


def test_true_is_not_one():
    # `isinstance(True, int)` is True in Python, so a config carrying a boolean here would
    # otherwise be read as "one prediction layer".
    assert mtp_layers({MTP_LAYERS_KEY: True}) == 0


def test_the_hidden_layer_count_is_left_alone(tmp_path):
    """The fix must not touch `num_hidden_layers`, which is what the first attempt got wrong."""
    from senbonzakura import convert

    src = tmp_path / "GLM-4.7-Flash"
    src.mkdir()
    cfg = {"architectures": ["Glm4MoeForCausalLM"], "num_hidden_layers": 3, MTP_LAYERS_KEY: 1}
    (src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    after = convert.read_config(src)
    assert after["num_hidden_layers"] == 3, (
        "num_hidden_layers counts DECODER layers and the prediction layer is declared on top "
        "of it; subtracting one here deletes a real layer and the conversion check still passes")
    assert mtp_layers(after) == 1

    # And nothing in the module should be rewriting a config any more.
    assert not hasattr(convert, "without_mtp_layer"), (
        "the config rewriting approach was wrong and must not return")
