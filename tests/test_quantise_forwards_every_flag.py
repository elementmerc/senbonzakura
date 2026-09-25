# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`quantise <checkpoint>` must carry the user's flags, and must refuse before it converts.

The checkpoint path converts first and then quantises, and it rebuilt the inner command line by
hand. It carried `--type`, `--imatrix`, `--force` and `--threads`, and dropped
`--output-tensor-type`, `--token-embedding-type`, `--tensor-type` and `--allow-requantize`.

Both halves of that were bad. The run reported DONE having produced a file the user did not ask
for; and `--tensor-type` is validated by `parse_tensor_type` inside `_preflight_arguments`, which
this path skipped, so a malformed pair was accepted and ignored rather than refused.

The same skip deferred every other pre-flight refusal until after the conversion. `--imatrix
/nope` and `--threads 99999` are decidable from the command line, and were being reported after
tens of minutes and tens of gigabytes of writes.

The forwarding is now derived from the parser rather than from a second hand-written list,
because the hand-written list is the defect: a flag added later would be dropped again with
nothing to say so. These tests pin the derivation, not the list.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import quantise  # noqa: E402

#: Positionals are replaced by the intermediate, `--type` is passed explicitly, and the three
#: source-disposal flags belong to the OUTER run: the intermediate is scaffolding this path owns
#: and deletes, so pointing --keep-source or --prune-source at it would name the wrong file.
EXPECTED_NOT_FORWARDED = {"source", "out", "type", "keep_source", "prune_source", "help"}


def _parse(*argv):
    return quantise.build_parser().parse_args(["ckpt", "out.gguf", *argv])


def test_every_quantiser_flag_the_user_set_is_forwarded():
    a = _parse("--type", "Q4_K_M", "--output-tensor-type", "F16",
               "--token-embedding-type", "Q8_0", "--tensor-type", "attn_v=Q6_K",
               "--allow-requantize", "--threads", "8")
    got = dict(quantise._forwardable_quantiser_flags(a))
    for flag in ("--output-tensor-type", "--token-embedding-type", "--allow-requantize",
                 "--threads"):
        assert flag in got, f"{flag} was dropped on the checkpoint path"


def test_an_appended_flag_is_forwarded_once_per_occurrence():
    a = _parse("--tensor-type", "attn_v=Q6_K", "--tensor-type", "ffn_down=Q5_K")
    pairs = [v for f, v in quantise._forwardable_quantiser_flags(a) if f == "--tensor-type"]
    assert pairs == ["attn_v=Q6_K", "ffn_down=Q5_K"]


def test_unset_flags_are_not_forwarded():
    """An empty list or a zero must not become `--tensor-type []` on the inner command line."""
    flags = [f for f, _ in quantise._forwardable_quantiser_flags(_parse())]
    assert "--tensor-type" not in flags
    assert "--threads" not in flags
    assert "--allow-requantize" not in flags


def test_the_source_disposal_flags_stay_with_the_outer_run():
    a = _parse("--keep-source", "--prune-source")
    flags = [f for f, _ in quantise._forwardable_quantiser_flags(a)]
    assert "--keep-source" not in flags and "--prune-source" not in flags


def test_the_exclusion_list_matches_what_the_parser_has():
    """If a positional or a disposal flag is renamed, this says so rather than silently leaking."""
    assert quantise._NOT_FORWARDED == EXPECTED_NOT_FORWARDED
    dests = {act.dest for act in quantise.build_parser()._actions}
    missing = EXPECTED_NOT_FORWARDED - dests
    assert not missing, f"_NOT_FORWARDED names {missing}, which the parser no longer has"


@pytest.mark.parametrize(("argv", "needle"), [
    (["--threads", "-5"], "worker count"),
    (["--imatrix", "/nonexistent.senbonzakura.imatrix"], "imatrix"),
])
def test_the_checkpoint_path_refuses_before_it_converts(tmp_path, argv, needle, monkeypatch):
    """The refusal must arrive without the converter having been called at all."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}', encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"\x00")

    called = []
    monkeypatch.setattr(quantise, "_quantise_a_checkpoint",
                        lambda *a, **k: called.append(True))
    with pytest.raises(SystemExit) as e:
        quantise.run([str(ckpt), str(tmp_path / "o.gguf"), *argv], log=lambda _m: None)
    assert needle in str(e.value).lower()
    assert not called, "the conversion was reached despite a fault decidable from the argv"
