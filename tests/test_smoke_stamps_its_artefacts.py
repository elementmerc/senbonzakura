# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Nothing the CI smoke writes can leave it looking like a measurement.

THE SHAPE OF THE PROBLEM. `tools/smoke_end_to_end.py` runs a genuine search on
`examples/toy-track`, which holds 8 harmful, 4 harmful-eval and 8 harmless rows, and whose
harmful rows are placeholders. Baseline refusal on it is 0.0, so refusal REMOVAL cannot be
demonstrated there at all. The artefacts it writes are shaped exactly like a real run's, carry a
real `provenance` block, and would be believed by anyone who found one.

This project has already published a p-value that came from its own synthetic fixture. A
reviewer caught it, not the tree. This is the thing in the tree.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "_smoke_under_test", ROOT / "tools" / "smoke_end_to_end.py")
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules["_smoke_under_test"] = smoke
_SPEC.loader.exec_module(smoke)


def _model_dir(tmp_path):
    """A saved-model directory as the abliterator leaves it: our artefacts beside theirs."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "abliteration.json").write_text(json.dumps({"post_bake_kl": 0.12}), encoding="utf-8")
    (d / "trials.json").write_text(json.dumps([{"number": 0}, {"number": 1}]), encoding="utf-8")
    # transformers' own files, which a loader reads and nothing of ours may edit.
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (d / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}), encoding="utf-8")
    (d / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
    return d


def test_our_result_artefacts_are_marked(tmp_path):
    _model_dir(tmp_path)
    (tmp_path / "score.json").write_text(json.dumps({"refusal": 0.0}), encoding="utf-8")

    stamped = smoke.stamp_smoke_artefacts(tmp_path)

    assert set(stamped) == {"abliteration.json", "trials.json", "score.json"}
    doc = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    assert "must not be quoted" in doc["smoke_artefact"]
    assert doc["refusal"] == 0.0, "the stamp is additive; it must not disturb what was measured"


@pytest.mark.parametrize("name", [
    "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
])
def test_the_models_own_files_are_never_touched(tmp_path, name):
    """THE DEFECT THE FIRST VERSION HAD. It swept `*.json` under the output directory and
    stamped all four of these, which is editing a model's configuration to leave a note in it.
    A saved model is the artefact a downstream user actually loads.
    """
    d = _model_dir(tmp_path)
    before = (d / name).read_text(encoding="utf-8")

    smoke.stamp_smoke_artefacts(tmp_path)

    assert (d / name).read_text(encoding="utf-8") == before, (
        f"{name} was modified, and transformers reads it")


def test_a_list_shaped_artefact_keeps_its_rows(tmp_path):
    """`trials.json` is a list and has nowhere to put a key, so it is wrapped. The rows have to
    survive the wrapping or the stamp has destroyed what it was annotating.
    """
    d = _model_dir(tmp_path)
    smoke.stamp_smoke_artefacts(tmp_path)
    doc = json.loads((d / "trials.json").read_text(encoding="utf-8"))
    assert doc["trials"] == [{"number": 0}, {"number": 1}]
    assert "smoke_artefact" in doc


def test_the_notice_says_the_numbers_are_meaningless_not_merely_provisional():
    """Wording matters here more than usual. "preliminary" or "toy" invites a reader to discount
    the number; the true statement is that there is no number to discount.
    """
    assert "meaningless" in smoke.SMOKE_NOTICE
    assert "must not be quoted" in smoke.SMOKE_NOTICE
    assert "8 harmful" in smoke.SMOKE_NOTICE, "it should say how small the track actually is"
