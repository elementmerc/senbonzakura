# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The first command of the published reproduction recipe has to run.

FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17, working from an installed wheel with no source
access. `head-to-head/CONTRACT.md` opens with "Run it yourself and tell us we are wrong.
Everything needed is in this directory", and the benchmark page gives a three step recipe. Step
one, against the only corpus that ships in the package:

    $ senbonzakura head-to-head stage --track default --out slices
    headtohead stage: no track at default.
      ...unlike the abliterator it does not resolve the bundled track by name, so
      `--track default` will not work here.

`--track default` is the documented path everywhere else in the tool; the top level help calls it
"the quickest way to a first run". The benchmark was the one command where it did not work, and
it is the command the benchmark's own invitation depends on. A reader who took us up on checking
our numbers could not take the first step without first building a corpus of harmful prompts they
have no way to obtain.

The refusal itself was well written and explained the limitation clearly. It was still a
limitation, and an invitation to refute us that cannot be accepted is not an invitation.
"""
import pytest

from senbonzakura import dataset, headtohead_stage

pytestmark = pytest.mark.skipif(
    not __import__("senbonzakura.bundled", fromlist=["x"]).is_available(),
    reason="no packed evaluation track in this checkout")


def test_the_alias_resolves_to_a_real_directory():
    resolved = headtohead_stage._resolve_track(dataset.BUNDLED_ALIAS)
    assert resolved.is_dir(), "`--track default` did not resolve to a directory that exists"
    assert (resolved / "bad_eval_ds").is_dir(), f"{resolved} carries no bad_eval_ds"


def test_an_ordinary_path_is_still_just_a_path(tmp_path):
    """The alias must not start swallowing directories that happen to be named oddly."""
    assert headtohead_stage._resolve_track(str(tmp_path)) == tmp_path


def test_the_first_command_of_the_recipe_runs(tmp_path):
    """The whole point. Not a unit of it, the command as the documentation prints it."""
    out = tmp_path / "slices"
    headtohead_stage.main(["--track", "default", "--out", str(out)])
    for slice_file in ("keyword_prompts.txt", "final_prompts.txt", "kl_prompts.txt",
                       "bestofn_kl_prompts.txt", "bad.txt", "good.txt"):
        written = out / slice_file
        assert written.is_file(), f"{slice_file} was not staged"
        assert written.read_text(encoding="utf-8").strip(), f"{slice_file} is empty"
    assert (out / "slices.json").is_file(), "the source corpus was not recorded"


def test_a_track_that_is_genuinely_absent_still_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        headtohead_stage.main(["--track", str(tmp_path / "nope"), "--out", str(tmp_path / "s")])
    assert "no track at" in str(e.value)


def test_the_refusal_no_longer_says_the_alias_will_not_work(tmp_path):
    """It said so truthfully for months. Leaving the sentence behind would send the next reader
    away from the thing that now works.
    """
    with pytest.raises(SystemExit) as e:
        headtohead_stage.main(["--track", str(tmp_path / "nope"), "--out", str(tmp_path / "s")])
    message = str(e.value)
    assert "will not work here" not in message
    assert "--track default" in message, "the refusal no longer offers the bundled track at all"
