# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`measure` hands the loader flags to the stages it runs, so a flag that works on `score` works under it.

Before this, `measure` took only `--trust-remote-code` of the loading flags, so a 4-bit, 8-bit or GGUF
model could be scored on its own and not through the command that runs every instrument in turn.
`validate` is the exception, on purpose: it bakes the model, which needs full precision, and the stage
refuses the combination in a sentence before anything loads.
"""
import json

import pytest

from senbonzakura import measure

STAGES = ("score", "compass", "coherence", "capability", "drift")


def _parse(*extra, only=None):
    argv = ["m", "--track", "t", *extra]
    if only:
        argv += ["--only", only]
    return measure.build_parser().parse_args(argv)


def _with_track(tmp_path, args):
    d = tmp_path / "trk"
    d.mkdir(exist_ok=True)
    (d / "track.json").write_text(json.dumps({}), encoding="utf-8")
    args.track = str(d)
    args.baseline = args.baseline or "base"
    return args


@pytest.mark.parametrize("stage", STAGES)
def test_every_measuring_stage_gets_the_flags(tmp_path, stage):
    args = _with_track(tmp_path, _parse("--load-in-4bit", "--load-in-8bit", "--gguf-file", "m.gguf",
                                        "--trust-remote-code"))
    argv = measure.stage_argv(stage, args, tmp_path)
    assert "--load-in-4bit" in argv and "--load-in-8bit" in argv and "--trust-remote-code" in argv
    assert argv[argv.index("--gguf-file") + 1] == "m.gguf"


@pytest.mark.parametrize("stage", STAGES)
def test_no_flag_is_added_when_none_was_given(tmp_path, stage):
    argv = measure.stage_argv(stage, _with_track(tmp_path, _parse()), tmp_path)
    assert not {"--load-in-4bit", "--load-in-8bit", "--gguf-file", "--trust-remote-code"} & set(argv)


def test_the_forwarded_flags_are_ones_the_stage_parsers_accept(tmp_path):
    """The failure this guards is a flag the stage does not declare, which argparse rejects mid-run."""
    from senbonzakura import capability, coherence, drift, margin, score
    parsers = {"score": score, "compass": margin, "coherence": coherence, "capability": capability,
               "drift": drift}
    args = _with_track(tmp_path, _parse("--load-in-4bit", "--gguf-file", "m.gguf"))
    for stage, module in parsers.items():
        argv = measure.stage_argv(stage, args, tmp_path)
        module.build_parser().parse_known_args(argv)        # SystemExit on a flag it does not know
        ns, rest = module.build_parser().parse_known_args(argv)
        assert ns.load_in_4bit is True and ns.gguf_file == "m.gguf", stage
        assert "--load-in-4bit" not in rest


def test_validate_is_not_handed_flags_it_would_refuse(tmp_path):
    argv = measure.stage_argv("validate", _with_track(tmp_path, _parse("--load-in-4bit")), tmp_path)
    assert "--load-in-4bit" not in argv


@pytest.mark.parametrize("flags", [("--load-in-4bit",), ("--load-in-8bit",), ("--gguf-file", "m.gguf")])
def test_asking_for_validate_with_a_quantised_model_is_a_sentence(flags):
    with pytest.raises(SystemExit, match="rewrites the model's weights"):
        measure.wanted_stages(_parse(*flags, only="validate"))


def test_the_default_stages_with_a_flag_do_not_include_validate_and_do_not_refuse():
    assert "validate" not in measure.wanted_stages(_parse("--load-in-4bit"))


def test_a_namespace_without_the_flags_reads_as_plain():
    import types
    plain = types.SimpleNamespace(model="m", device="cpu", hf_token=None, trust_remote_code=False)
    assert "--load-in-4bit" not in measure.stage_argv("coherence", plain, __import__("pathlib").Path("."))
