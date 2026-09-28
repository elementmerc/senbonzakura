# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Two defects `measure` shipped with, both found by a self-review pass on 2026-09-25.

THE PARTITION. `stage_argv` built the `score` stage against `bad_eval_ds` with no `--skip`, and
that dataset is the SEARCH rows followed by the MEASURE rows. So `score` read from the head of the
file, which is the partition the search picked the configuration on, while the table printed the
figure under "measured on this track's held-out rows and on no others". `compass` in the same run
resolved the skip correctly through `margin.resolve_skips`, so one table was reading two
partitions and describing neither.

This is the project's own documented incident class, not a new one. `checker/.../
a-rate-with-no-partition-beside-it.json` exists because a 0.0% once travelled as a measured
refusal rate having been scored on the selection partition.

THE TOKEN. `--hf-token` was appended to the argv of all five stages, and three of them do not
declare it. argparse prints an unrecognised argument together with its VALUE, so a live
HuggingFace token went to stderr verbatim in three error messages, under a source comment on the
same line promising the value is never printed.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import measure  # noqa: E402

#: Not a credential, and deliberately under the secret gate's 16-character threshold: a longer
#: placeholder trips the pre-commit scanner, which is the scanner behaving correctly.
FAKE_TOKEN = "not-a-token"


def _args(track, **over):
    base = dict(model="m", device="cpu", hf_token=None, trust_remote_code=False, n=200,
                track=str(track), capability_n=40, capability_max_new=256, baseline="base-model")
    base.update(over)
    return types.SimpleNamespace(**base)


def _track(tmp_path, **manifest):
    d = tmp_path / "trk"
    d.mkdir()
    (d / "track.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


# ── the partition ────────────────────────────────────────────────────────────────────

def test_score_skips_the_search_rows_the_manifest_declares(tmp_path):
    argv = measure.stage_argv("score", _args(_track(tmp_path, skip_harmful=128)), tmp_path)
    # THE BOUNDARY IS HANDED OVER, NOT RESOLVED HERE, since 2026-09-25. This asserted
    # `--skip 128`, which `measure` computed with its own copy of the manifest reader. That
    # copy did not know about the bundled alias, so on the DEFAULT track it returned 0 and
    # the front-door command scored the selection rows under a caption promising held-out
    # ones. There is one resolver now, in `track`, and `score` is given what it needs to
    # call it: the flags, not the answer.
    assert "--skip" not in argv, (
        "measure must not resolve the boundary itself; a second manifest reader is what "
        "produced the in-sample refusal rate this test was written about")
    assert "--track" in argv and "--track-arm" in argv, (
        "score needs both to resolve the boundary and to stamp a verified partition")
    assert argv[argv.index("--track-arm") + 1] == "harmful", (
        "bad_eval_ds is the harmful arm; the wrong arm reads the wrong recorded boundary")


def test_a_track_that_declares_no_skip_gets_none(tmp_path):
    """Honest rather than invented: this cannot know a boundary the manifest does not state."""
    argv = measure.stage_argv("score", _args(_track(tmp_path)), tmp_path)
    assert "--skip" not in argv


@pytest.mark.parametrize("manifest", [{"skip_harmful": 0}, {"skip_harmful": -5},
                                      {"skip_harmful": "128"}, {"skip_harmful": None}])
def test_a_malformed_skip_is_ignored_rather_than_passed_through(tmp_path, manifest):
    argv = measure.stage_argv("score", _args(_track(tmp_path, **manifest)), tmp_path)
    assert "--skip" not in argv


def test_an_unreadable_manifest_does_not_crash_the_run(tmp_path):
    d = tmp_path / "trk"
    d.mkdir()
    (d / "track.json").write_text("{ not json", encoding="utf-8")
    assert "--skip" not in measure.stage_argv("score", _args(d), tmp_path)


# ── the token ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stage", ["compass", "coherence", "drift"])
def test_the_token_never_reaches_a_stage_that_cannot_parse_it(tmp_path, stage):
    argv = measure.stage_argv(stage, _args(_track(tmp_path), hf_token=FAKE_TOKEN), tmp_path)
    assert "--hf-token" not in argv, (
        f"{stage} does not declare --hf-token, and argparse prints an unrecognised argument "
        f"together with its value")
    assert FAKE_TOKEN not in argv


@pytest.mark.parametrize("stage", ["score", "capability"])
def test_the_token_still_reaches_the_stages_that_need_it(tmp_path, stage):
    argv = measure.stage_argv(stage, _args(_track(tmp_path), hf_token=FAKE_TOKEN), tmp_path)
    assert "--hf-token" in argv and FAKE_TOKEN in argv


def test_the_token_list_matches_the_parsers_that_actually_declare_the_flag():
    """The list is data, so this is what stops it drifting from the parsers it describes."""
    import importlib

    for stage, module in (("score", "score"), ("capability", "capability"),
                          ("compass", "margin"), ("coherence", "coherence"), ("drift", "drift")):
        src = Path(importlib.import_module(f"senbonzakura.{module}").__file__).read_text(
            encoding="utf-8")
        declares = "--hf-token" in src
        listed = stage in measure.TAKES_HF_TOKEN
        assert declares == listed, (
            f"{stage} {'declares' if declares else 'does not declare'} --hf-token but is "
            f"{'listed' if listed else 'not listed'} in TAKES_HF_TOKEN")


# ── the line that printed it anyway ───────────────────────────────────────────────────────────────

def test_run_stage_does_not_print_the_token_it_was_handed():
    """WHAT PROMPTED IT, 2026-09-28. The panel demonstrated this; the suite could not see it.

    `run_stage` logged the joined argv before executing it, so a real `measure --hf-token <live>`
    printed the token to stdout, twice, into the transcript people paste into bug reports and that
    CI jobs archive. `_shown` existed and was applied to the dry run and to `measure.json`, which
    are the two paths a test was watching.

    Every test that exercised `run()` monkeypatched `run_stage` away, and the three token tests
    above all assert on `stage_argv`'s output. So the one place the raw argv reached a stream was
    the one place nothing looked, under a comment in `stage_argv` promising the value is never
    printed. That promise had already been broken once, in argparse's stderr, in September.
    """
    module = types.ModuleType("senbonzakura.pretend_stage")
    module.main = lambda _argv: {"ok": True}
    monkey = {"score": ("pretend_stage", "x")}

    lines = []
    import senbonzakura.entry as entry_module
    real = entry_module.DELEGATED
    sys.modules["senbonzakura.pretend_stage"] = module
    entry_module.DELEGATED = monkey
    try:
        measure.run_stage("score", ["--model", "m", "--hf-token", FAKE_TOKEN, "--out", "x.json"],
                          log=lines.append)
    finally:
        entry_module.DELEGATED = real
        del sys.modules["senbonzakura.pretend_stage"]

    printed = "\n".join(lines)
    assert FAKE_TOKEN not in printed, (
        f"run_stage printed the token: {printed!r}")
    assert "--hf-token" in printed and measure.REDACTED in printed, (
        "the flag should still be visible so a reader can see the stage was given one; only the "
        f"value goes: {printed!r}")


@pytest.mark.parametrize(("argv", "expected"), [
    (["--hf-token", "s3cret"], ["--hf-token", "***"]),
    (["--hf-token=s3cret"], ["--hf-token=***"]),
    (["--model", "m", "--hf-token", "s3cret", "--device", "cpu"],
     ["--model", "m", "--hf-token", "***", "--device", "cpu"]),
    (["--model", "m"], ["--model", "m"]),
    (["--hf-token"], ["--hf-token"]),
])
def test_the_redaction_needs_no_token_to_do_its_job(argv, expected):
    """Keyed on the flag, so no call site has to be handed the secret in order to hide it.

    The `--hf-token=value` spelling is in here because argparse accepts it and a reader will type
    it, and a rule that only knows the two token form would have printed it whole.
    """
    assert measure.without_secrets(argv) == expected


def test_every_secret_flag_named_here_is_one_some_parser_declares():
    """Otherwise the list decays into a name nothing uses, and the redaction covers nothing."""
    import importlib

    sources = "".join(
        Path(importlib.import_module(f"senbonzakura.{m}").__file__).read_text(encoding="utf-8")
        for m in ("score", "capability", "measure", "parser"))
    for flag in measure.SECRET_FLAGS:
        assert flag in sources, f"{flag} is redacted and declared by no parser in this package"
