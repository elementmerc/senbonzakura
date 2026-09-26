# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`score` decides which rows it will measure before it downloads a model to measure them with.

WHAT WAS WRONG

The eval set was resolved and its bounds checked AFTER `load_model_and_tokenizer`. So
`senbonzakura score --model <30B> --eval ... --skip 99999` downloaded and loaded tens of
gigabytes and then exited with "--skip 99999 leaves nothing: the set has 400 prompts". On a rented
card that is paid minutes for a fault decidable from the command line plus a dataset header, and
it is the same argument `preflight_snapshot_ram` already makes about this exact route.

`dataset.resolve` needs no model and neither does anything between it and the slice, so nothing
was bought by loading first.

READ AS TEXT RATHER THAN IMPORTED, deliberately. `score.py` imports torch at module scope, so a
test that imports it cannot run in the torch-free environments this project supports, and the
ordering is what is being asserted rather than any runtime behaviour. `test_every_flag_is
_documented.py` reads the package the same way and for the same reason.
"""
from __future__ import annotations

import pathlib

SOURCE = (pathlib.Path(__file__).resolve().parents[1]
          / "src" / "senbonzakura" / "score.py").read_text(encoding="utf-8")
MAIN = SOURCE[SOURCE.index("\ndef main("):]


def _at(needle):
    assert needle in MAIN, f"{needle!r} is no longer in score.main; this guard needs rewriting"
    return MAIN.index(needle)


def test_the_eval_set_is_resolved_before_the_model_loads():
    assert _at("dataset.resolve(") < _at("load_model_and_tokenizer(")


def test_the_slice_bounds_are_checked_before_the_model_loads():
    """The two refusals the defect was reported through, both of them free."""
    assert _at('raise SystemExit(f"--skip') < _at("load_model_and_tokenizer(")
    assert _at('raise SystemExit(f"--n') < _at("load_model_and_tokenizer(")


def test_the_recorded_boundary_is_still_resolved_before_the_slice():
    """`resolve_skip_for_arm` refuses a flag that contradicts the track manifest, and it has to
    happen before anything is sliced or the contradiction is acted on rather than refused.
    """
    assert _at("resolve_skip_for_arm(") < _at('raise SystemExit(f"--skip')
    assert _at("dataset.resolve(") < _at("resolve_skip_for_arm(")
    assert _at("resolve_skip_for_arm(") < _at("load_model_and_tokenizer(")


def test_nothing_between_the_parse_and_the_load_needs_a_model():
    """A guard against the ordering being restored by accident: the generation calls, which are
    the first thing that genuinely needs the model, all come after the load.
    """
    load = _at("load_model_and_tokenizer(")
    for needs_a_model in ("generate(model", "generate_prefixes(model"):
        assert _at(needs_a_model) > load
