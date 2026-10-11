# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What `measure` hands each stage, the stage has to be able to read.

FOUND BY A JOURNEY PERSONA, 2026-10-11. `senbonzakura measure --only drift` failed two seconds in
with `FileNotFoundError: [Errno 2] No such file or directory: 'default/good_ds'`, so drift was the
one instrument of the five that could never run on a track, bundled or built.

`measure` passes every stage a dataset spec, and `_track_spec`'s own docstring says that is safe
because "`default/bad_eval_ds` is a real spec that `dataset.resolve` understands, so the bundled
track needs no special case anywhere downstream". True of the spec, and false of one reader of it:
`drift.read_prompts` was a bare `open()` and never went near `dataset.resolve`.

**Why the existing tests did not catch it, which is the part worth copying.**
`tests/test_the_table_says_what_it_measured.py` checks the argv `measure` BUILDS. That is a real
check and it passed, because the argv was right. Nobody checked that the stage on the other end
could read it. A contract asserted from one side only is half a contract, and this file is the
other half: it takes the argv `measure` really builds and drives the real reader with it.

Reading, not running. Loading a model is gigabytes and hours; resolving the inputs is where this
class of defect lives, and it costs milliseconds.
"""

import pathlib

import pytest

from senbonzakura import measure


class _Args:
    """The attributes `measure.stage_argv` reads, with the defaults its parser would give."""

    def __init__(self, **kw):
        self.model = "/some/model"
        self.baseline = "/some/base"
        self.track = "default"
        self.device = "cpu"
        self.out = "/tmp/journey2-measure"
        self.n = 60
        self.capability_n = 8
        self.trust_remote_code = False
        self.load_in_4bit = False
        self.chat_template = ""
        self.hf_token = ""
        self.__dict__.update(kw)


def _value_after(argv, flag):
    """The value `measure` put after one flag, or None when it passed no such flag."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def _stage_argv(name, **kw):
    """The argv `measure` really builds for one stage.

    Called by its real name and signature, deliberately. The first draft of this helper tried
    several plausible names and two argument orders and fell back to `pytest.skip`, which is the
    shape that lets a renamed function turn a guard into a silent pass. If `stage_argv` moves,
    this file should fail to import and say so.
    """
    return measure.stage_argv(name, _Args(**kw), pathlib.Path("/tmp/journey2-measure"))


def test_the_drift_stage_prompts_can_actually_be_read():
    """The defect itself, driven end to end through the two real functions.

    This is the whole finding in four lines: whatever `measure` puts after `--prompts`, hand it to
    the function that reads `--prompts`.
    """
    from senbonzakura import drift

    spec = _value_after(_stage_argv("drift"), "--prompts")
    assert spec is not None, "measure no longer passes --prompts to drift"
    prompts = drift.read_prompts(spec)
    assert len(prompts) > 100, (
        f"drift read {len(prompts)} prompts from the spec measure gave it ({spec!r}); the bundled "
        f"harmless set has thousands, so this is not the set measure meant")
    assert all(isinstance(p, str) and p.strip() for p in prompts[:50]), (
        "drift read rows that are not non-empty strings")


def test_the_drift_stage_prompts_resolve_for_a_built_track_too():
    """A directory track, not only the bundled one, because `_track_spec` serves both.

    The bundled track resolves through a special case for the word `default`. A track on disk
    takes the ordinary path, and that is the one a user who ran `senbonzakura track` has.
    """
    from senbonzakura import bundled, drift

    unpacked = pathlib.Path(bundled.cache_dir())
    if not (unpacked / "good_ds").exists():
        pytest.skip("the bundled track is not unpacked here, so there is no directory form to use")
    spec = _value_after(_stage_argv("drift", track=str(unpacked)), "--prompts")
    assert len(drift.read_prompts(spec)) > 100, f"drift could not read the directory spec {spec!r}"


@pytest.mark.parametrize(("stage", "flag"),
                         [("drift", "--prompts"), ("compass", "--harmful"),
                          ("compass", "--harmless"), ("score", "--eval")])
def test_every_dataset_spec_measure_builds_resolves(stage, flag):
    """The general form, so the next stage to grow a reader of its own is covered in advance.

    Each of these flags carries a dataset spec that `measure` derived from `--track`. Resolving
    them all through `dataset.resolve` proves the specs themselves are sound; the per-stage tests
    above prove the stage's own reader agrees, which is the half that was missing.
    """
    from senbonzakura import dataset

    argv = _stage_argv(stage)
    spec = _value_after(argv, flag)
    if spec is None:
        pytest.skip(f"measure passes no {flag} to {stage}")
    rows = dataset.resolve(spec, what=f"{stage} {flag}")
    assert rows, f"{stage} {flag} resolved to no rows from {spec!r}"
