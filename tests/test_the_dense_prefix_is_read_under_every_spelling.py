# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
""""The first N blocks are dense" has two spellings, and only one of them was read.

`num_dense_layers` was read and `first_k_dense_replace` was not, so for DeepSeek, GLM, Kimi,
Bailing, Mistral and Command-R configs the module reported that it could not tell which blocks
were dense. The config said so plainly, under a name the tuple did not list.

WHY THAT IS WORSE THAN A MISSING FEATURE. A reader drew each of those feed-forwards as "not
measured", and the whole point of that state is to mean "we looked and the config does not say".
Here the config did say. An unread key turned into a claim about the model, which is the failure
this module is built to avoid: it refuses to guess, so the honesty of every refusal depends on the
lookup actually being exhaustive.

The two keys mean the same thing, which is why they belong in one tuple rather than two branches.
The evidence is this repository's own vendored conversion code, which maps `first_k_dense_replace`
onto the leading-dense-block count for all six of those families.
"""
from senbonzakura import modelmap


def _moe(extra):
    """A minimal mixture-of-experts config, plus whichever dense spelling is under test."""
    cfg = {"num_hidden_layers": 8, "n_routed_experts": 64, "num_experts_per_tok": 6}
    cfg.update(extra)
    return cfg


def test_the_deepseek_spelling_is_read():
    got = modelmap.experts(_moe({"first_k_dense_replace": 3}))
    assert got.dense_blocks.known, (
        f"`first_k_dense_replace` is present and says the first three blocks are dense, but the "
        f"module reported it could not tell: {got.dense_blocks.why}")
    assert got.dense_blocks.value == [0, 1, 2]
    assert got.dense_blocks.source == "first_k_dense_replace", (
        "the record has to name the key it actually found, or a reader checking it against the "
        "config file looks for a name that is not in there")


def test_the_other_spelling_still_works_and_still_names_itself():
    got = modelmap.experts(_moe({"num_dense_layers": 1}))
    assert got.dense_blocks.value == [0]
    assert got.dense_blocks.source == "num_dense_layers"


def test_a_config_with_neither_still_refuses_and_names_both():
    """The refusal is the honest answer here, and it must list every name that was tried.

    A reason that names one key implies the other was not worth looking for, which is how the
    defect above survived: the message said `num_dense_layers` was absent, and it was.
    """
    got = modelmap.experts(_moe({}))
    assert not got.dense_blocks.known
    for key in modelmap.DENSE_PREFIX_KEYS:
        assert key in got.dense_blocks.why, (
            f"the reason does not mention {key!r}, so somebody reading it cannot tell whether "
            f"that spelling was tried or simply unknown to this tool")


def test_the_index_list_spelling_takes_precedence_over_neither():
    got = modelmap.experts(_moe({"mlp_only_layers": [0, 4]}))
    assert got.dense_blocks.value == [0, 4]
    assert got.dense_blocks.source == "mlp_only_layers"


def test_a_count_of_zero_is_a_measurement_and_not_an_absence():
    """Zero leading dense blocks means every block is mixture, which is a real answer.

    Written because `if prefix is not None` is the correct test and `if prefix` is the tempting
    one, and the second would report a wholly-mixture model as unreadable.
    """
    got = modelmap.experts(_moe({"first_k_dense_replace": 0}))
    assert got.dense_blocks.known, (
        "a declared zero means no leading dense blocks, which is a fact about the model, not a "
        "missing key")
    assert got.dense_blocks.value == []


def test_both_spellings_at_once_prefers_the_first_and_says_which():
    """Not expected in a real config, but it must be decided rather than arbitrary."""
    got = modelmap.experts(_moe({"num_dense_layers": 2, "first_k_dense_replace": 5}))
    assert got.dense_blocks.source in modelmap.DENSE_PREFIX_KEYS
    assert got.dense_blocks.source == modelmap.DENSE_PREFIX_KEYS[0]
    assert got.dense_blocks.value == [0, 1]
