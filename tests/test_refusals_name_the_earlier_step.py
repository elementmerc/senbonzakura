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
from senbonzakura.bundled import BundledTrackError

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


def test_whatever_the_stage_refusal_recommends_actually_works(tmp_path):
    """THE MISTAKE THIS FILE ALMOST SHIPPED WITH, now checked as a property rather than a case.

    The first version of the stage refusal suggested `--track default`, copying the abliterator,
    where that name resolves to the bundled track. `headtohead_stage` resolved nothing: it did
    `Path(a.track).is_dir()`, so `default` landed straight back on the same refusal. Advice that
    loops the reader back to the error they are already reading is worse than no advice.

    REWRITTEN 2026-09-17, because the fix went the other way. A hostile outside reviewer pointed
    out that `--track default` failing here broke the FIRST command of the published
    reproduction recipe, under a CONTRACT.md that opens "Run it yourself and tell us we are
    wrong". So rather than keeping the recommendation out of the message, the command learned to
    resolve the name, and the refusal now offers it.

    The old assertion would have failed either way round, which is the tell that it was pinned to
    a case and not to the principle. The principle is: a refusal may only recommend something
    that works. So this drives the recommendation instead of reading it, and would catch both the
    original mistake and any future regression of the resolver.
    """
    with pytest.raises(SystemExit) as caught:
        headtohead_stage.main(["--track", str(tmp_path / "nope"), "--out", str(tmp_path / "s")])
    message = str(caught.value)

    recommended = re.search(r"--track\s+(\S+)", message)
    if not recommended:
        pytest.skip("the refusal recommends no track, so there is nothing to verify")
    name = recommended.group(1).strip("`'\"")
    if name in {"<your", "TRACK", "my-track"}:
        pytest.skip(f"{name!r} is a placeholder rather than a value to type")

    # The recommendation is followed, not read. If it cannot resolve, the reader would land back
    # on the message they are already looking at.
    #
    # GUARDED, and the reason is this project's oldest recurring failure. The first version of
    # this called the resolver flat and went red on every CI platform while passing here, because
    # a source checkout carries no packed evaluation track: the blob holds harmful prompts and is
    # kept out of git on purpose, so it exists on a developer's machine and on no runner that has
    # not fetched it. "My machine has an artefact CI does not" has now cost this project four
    # separate red builds, and I wrote that sentence into a handoff hours before doing it again.
    #
    # An install that cannot resolve the name because it never had the data is not the defect this
    # test is about. The defect is a refusal recommending a flag value the command REJECTS, and
    # that distinction is the whole point: BundledTrackError means "I understood you and the data
    # is missing", which is the tool working.
    try:
        resolved = headtohead_stage._resolve_track(name)
    except BundledTrackError:
        pytest.skip(f"`--track {name}` is understood here and this install carries no packed "
                    f"track, so whether it resolves cannot be answered on this machine")
    assert resolved.is_dir(), (
        f"the refusal recommends `--track {name}`, and that does not resolve to a directory. "
        f"Advice that returns the reader to the error they are reading is worse than none.")
