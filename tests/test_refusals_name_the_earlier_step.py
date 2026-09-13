# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A refusal whose cause is a missing earlier step says which step.

THE CLASS THIS COVERS. Several commands sit downstream of another one: `head-to-head report`
reads what `head-to-head run` wrote, `head-to-head stage` needs a track that `track` builds,
`quantise` needs a GGUF that `convert` produces. Each of them refused correctly and stopped at
naming the problem, which leaves a reader who has not met the pipeline with a true sentence and
nothing to do about it.

The worked example was `head-to-head run` never saying `stage` must come first: the refusal named
every missing file, explained that nothing had run because the comparison would not be
trustworthy, and never said what to type. A peer driving the tool had to read our source to get
past it.

Under the standing assumption that nobody reads the source, "the thing you need does not exist"
is only half a message. The other half is the command that makes it exist.
"""
import re

import pytest

from senbonzakura import headtohead_report, headtohead_stage, quantise

#: (callable, argv, the command the refusal must name). One row per downstream command.
CASES = [
    pytest.param(headtohead_report.main, ["/definitely/not/here"],
                 "senbonzakura head-to-head run", id="report-needs-run"),
    pytest.param(headtohead_stage.main,
                 ["--track", "/definitely/not/here", "--out", "/tmp/unused-slices"],
                 "senbonzakura track", id="stage-needs-track"),
]


@pytest.mark.parametrize(("entry", "argv", "expected"), CASES)
def test_the_refusal_names_the_command_that_produces_what_is_missing(entry, argv, expected):
    with pytest.raises(SystemExit) as caught:
        entry(argv)
    message = str(caught.value)
    assert expected in message, (
        f"the refusal says what is missing and not how to make it:\n{message}")


def test_quantise_points_at_convert_rather_than_only_saying_the_file_is_absent(tmp_path):
    with pytest.raises(SystemExit) as caught:
        quantise.main([str(tmp_path / "nothing.gguf"), str(tmp_path / "out.gguf")])
    message = str(caught.value)
    assert "senbonzakura convert" in message, message


def test_stage_does_not_recommend_a_track_name_it_cannot_resolve():
    """THE MISTAKE THIS FILE ALMOST SHIPPED WITH.

    The first version of the stage refusal suggested `--track default`, copying the abliterator,
    where that name resolves to the bundled track. `headtohead_stage` calls no `bundled.ensure()`:
    it does `Path(a.track).is_dir()`, so `default` lands on this same refusal. Advice that loops
    the reader back to the error they are already reading is worse than no advice.
    """
    with pytest.raises(SystemExit) as caught:
        headtohead_stage.main(["--track", "default", "--out", "/tmp/unused-slices"])
    message = str(caught.value)
    assert not re.search(r"pass\s+--track default", message), (
        "the refusal recommends --track default, which this command cannot resolve")
    assert "does not resolve" in message, (
        "it should say outright that the bundled name does not work here")
