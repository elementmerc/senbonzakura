# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A refusal hands the reader a command, and never a Python exception.

WHAT PROMPTED IT, 2026-09-27

An output review ran the tool as a user and found raw interpreter strings reaching the screen:

    senbonzakura judge --judge bad.jsonl ...
      could not read the verdicts: Expecting property name enclosed in double quotes:
      line 1 column 3 (char 2)

    senbonzakura track --audit --fit 10
      track has no readable track.json, so its split cannot be checked:
      [Errno 2] No such file or directory: 'track/track.json'

Both are the project's own rule broken in its own words: error messages are plain language, not raw
error strings. The judge one is worse than it looks, because the decoder was called per line, so its
"line 1" was the line inside the line and not the line in the file: the coordinate a reader would
have used to go and look was wrong.

Two more refused without naming a next step at all: `score --track` with no `--track-arm`, and
`report` with no artefacts, which closed on a sentence of design justification and never showed the
flag shape.

WHAT THESE TESTS HOLD

Per refusal: it does not leak the exception's own text, it names the file or flag at fault, and it
carries a line a reader can run. The `senbonzakura ` prefix is the cheap proxy for that last part
and is deliberately cheap: a message with no command in it cannot pass, and grading the writing is
not a test's job.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: Strings that only ever arrive from an unwrapped exception.
LEAKS = ("Errno", "Traceback (most recent call last)", "(char ")


def _run(argv, cwd):
    return subprocess.run([sys.executable, "-m", "senbonzakura", *argv], cwd=str(cwd),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          check=False, timeout=300)


def _both(out):
    return out.stdout + out.stderr


class TestJudgeOnAMalformedFile:
    @pytest.fixture
    def files(self, tmp_path):
        (tmp_path / "bad.jsonl").write_text("{'verdict': 1}\n", encoding="utf-8")
        (tmp_path / "ok.jsonl").write_text(json.dumps({"verdict": "HARMFUL"}) + "\n",
                                           encoding="utf-8")
        return tmp_path

    def test_it_refuses(self, files):
        assert _run(["judge", "--judge", "bad.jsonl", "--reference", "ok.jsonl"], files) \
            .returncode != 0

    def test_it_leaks_no_interpreter_text(self, files):
        text = _both(_run(["judge", "--judge", "bad.jsonl", "--reference", "ok.jsonl"], files))
        assert not [s for s in LEAKS if s in text], text

    def test_it_names_the_file_and_the_line_in_that_file(self, files):
        """The coordinate has to be usable. The old message's line number was not."""
        (files / "bad.jsonl").write_text(
            json.dumps({"verdict": "HARMFUL"}) + "\n{'verdict': 1}\n", encoding="utf-8")
        text = _both(_run(["judge", "--judge", "bad.jsonl", "--reference", "ok.jsonl"], files))
        assert "bad.jsonl" in text
        assert "Line 2" in text, (
            "the bad row is the second line of the file, so that is the number a reader needs: "
            f"{text}")

    def test_it_shows_the_shape_the_file_should_have(self, files):
        text = _both(_run(["judge", "--judge", "bad.jsonl", "--reference", "ok.jsonl"], files))
        assert '"verdict"' in text

    def test_a_missing_file_is_its_own_message(self, files):
        text = _both(_run(["judge", "--judge", "gone.jsonl", "--reference", "ok.jsonl"], files))
        assert "gone.jsonl" in text
        assert not [s for s in LEAKS if s in text], text


class TestTrackAuditWithNoTrack:
    def test_a_missing_directory_says_how_to_get_one(self, tmp_path):
        out = _run(["track", "--audit"], tmp_path)
        text = _both(out)
        assert out.returncode != 0
        assert not [s for s in LEAKS if s in text], text
        assert "senbonzakura track build" in text, text

    def test_a_directory_with_no_manifest_is_a_different_message(self, tmp_path):
        (tmp_path / "track").mkdir()
        text = _both(_run(["track", "--audit"], tmp_path))
        assert "no track.json" in text, text
        assert not [s for s in LEAKS if s in text], text

    def test_a_corrupt_manifest_names_where_it_breaks(self, tmp_path):
        (tmp_path / "track").mkdir()
        (tmp_path / "track" / "track.json").write_text('{"counts" 7}', encoding="utf-8")
        text = _both(_run(["track", "--audit"], tmp_path))
        assert "not valid JSON" in text, text
        # The coordinates in words, not the decoder's own `(char 9)`, which counts bytes into a
        # file nobody is going to count bytes into.
        assert "line 1, column 11" in text, text
        assert not [s for s in LEAKS if s in text], text

    def test_the_three_messages_are_three_messages(self, tmp_path):
        """One message for three faults sent every reader down the same wrong path."""
        missing = _both(_run(["track", "--audit"], tmp_path))
        (tmp_path / "track").mkdir()
        empty = _both(_run(["track", "--audit"], tmp_path))
        (tmp_path / "track" / "track.json").write_text("{", encoding="utf-8")
        corrupt = _both(_run(["track", "--audit"], tmp_path))
        assert len({missing, empty, corrupt}) == 3


class TestRefusalsThatNamedNoNextStep:
    def test_score_names_the_flag_and_shows_the_command(self, tmp_path):
        out = _run(["score", "--model", "m", "--eval", "e.txt", "--out", "o.json",
                    "--track", "mytrack"], tmp_path)
        text = _both(out)
        assert out.returncode != 0
        assert "--track-arm harmful" in text, text
        assert "senbonzakura score" in text, (
            f"the refusal never shows a command that would work: {text}")

    def test_report_with_nothing_shows_the_flag_shape(self, tmp_path):
        out = _run(["report"], tmp_path)
        text = _both(out)
        assert out.returncode != 0
        for flag in ("--abliteration", "--capability", "--base-licence"):
            assert flag in text, f"{flag} is not named: {text}"
        assert "senbonzakura report" in text, text

    def test_report_no_longer_closes_on_a_design_argument(self, tmp_path):
        """The exact sentence the review quoted, which explained the refusal and showed nothing."""
        text = _both(_run(["report"], tmp_path))
        assert "would be a template" not in text, text
