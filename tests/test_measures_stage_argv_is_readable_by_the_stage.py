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


@pytest.fixture
def a_track_on_disk(tmp_path):
    """A track directory with the three splits `measure` derives specs for.

    THE VERSION THAT RUNS IN CI, and the reason it exists is the whole lesson of this file's second
    draft. The first draft pointed every assertion at `default/...`, the bundled track. That track
    is NOT in git: `tools/packaging/pack_track.py` builds it, because the blob holds harmful
    prompts and is deliberately kept out. So the tests passed on a development machine and failed
    on every CI platform at once, which `tests/test_doctor.py` had already written down as "this
    project's oldest recurring shape: my machine has an artefact CI does not".

    A synthetic track exercises the identical contract. `measure` builds `<track>/<split>` either
    way, `dataset.resolve` reads a directory split the same way, and nothing about the defect this
    file guards was specific to the bundled one. So the contract is tested everywhere and the
    bundled track became the bonus case below rather than the premise.
    """
    for split, rows in (("good_ds", 140), ("bad_eval_ds", 140), ("bad_ds", 40)):
        (tmp_path / split).write_text(
            "".join(f"a prompt numbered {i} for the {split} split\n" for i in range(rows)),
            encoding="utf-8")
    return tmp_path


def test_the_drift_stage_prompts_can_actually_be_read(a_track_on_disk):
    """The defect itself, driven end to end through the two real functions.

    This is the whole finding in four lines: whatever `measure` puts after `--prompts`, hand it to
    the function that reads `--prompts`. Before the fix this raised `FileNotFoundError`, because
    `drift.read_prompts` was a bare `open()` and the spec names a directory split.
    """
    from senbonzakura import drift

    spec = _value_after(_stage_argv("drift", track=str(a_track_on_disk)), "--prompts")
    assert spec is not None, "measure no longer passes --prompts to drift"
    prompts = drift.read_prompts(spec)
    assert len(prompts) == 140, (
        f"drift read {len(prompts)} prompts from the spec measure gave it ({spec!r}), not the 140 "
        f"the split holds")
    assert all(isinstance(p, str) and p.strip() for p in prompts), (
        "drift read rows that are not non-empty strings")


@pytest.mark.parametrize(("stage", "flag"),
                         [("drift", "--prompts"), ("compass", "--harmful"),
                          ("compass", "--harmless"), ("score", "--eval")])
def test_every_dataset_spec_measure_builds_resolves(stage, flag, a_track_on_disk):
    """The general form, so the next stage to grow a reader of its own is covered in advance.

    Each of these flags carries a dataset spec that `measure` derived from `--track`. Resolving
    them all through `dataset.resolve` proves the specs themselves are sound; the per-stage test
    above proves the stage's own reader agrees, which is the half that was missing.
    """
    from senbonzakura import dataset

    argv = _stage_argv(stage, track=str(a_track_on_disk))
    spec = _value_after(argv, flag)
    if spec is None:
        pytest.skip(f"measure passes no {flag} to {stage}")
    rows = dataset.resolve(spec, what=f"{stage} {flag}")
    assert rows, f"{stage} {flag} resolved to no rows from {spec!r}"


def test_the_bundled_track_resolves_the_same_way_where_one_is_installed():
    """The bonus case: the `default` alias, which takes a different path through `dataset`.

    Guarded rather than asserted, because a source checkout and a `git archive` tarball genuinely
    carry no packed track and that is a supported state. Skipping is honest here for the reason
    `test_the_corpus_is_pinned` gives about the same guard: "no track here" is not evidence either
    way about whether a track that exists resolves, and the contract itself is covered above on a
    track that always exists.
    """
    from senbonzakura import bundled, drift

    if not bundled.is_available():
        pytest.skip("no packed track in this tree; run tools/packaging/pack_track.py to make one")
    spec = _value_after(_stage_argv("drift"), "--prompts")
    assert spec == "default/good_ds", f"measure no longer derives the bundled spec: {spec!r}"
    assert len(drift.read_prompts(spec)) > 100, (
        "the bundled harmless split holds thousands of rows, so this is not the set measure meant")
