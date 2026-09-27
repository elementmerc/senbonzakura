# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A refusal that names no next step leaves the reader exactly where they were.

WHAT PROMPTED IT, 2026-09-27

The last five findings of a 66-surface audit were all the same defect wearing different clothes:

- `senbonzakura baseline <a directory>` printed a bare errno, and a directory is the obvious thing
  to pass because every other command here takes one.
- `gate`'s missing-baseline refusal said "no baseline at X." and nothing about a baseline being a
  file this tool writes, or which command writes it.
- its wrong-schema refusal said "declares schema None", which reads as a version problem when the
  file was most likely never a baseline at all.
- `senbonzakura track` with no arguments gave one line naming two flags, where `report` answers the
  same mistake with a worked command.
- `--print-completion` on an install without the extra got argparse's "unrecognized arguments",
  which says the flag does not exist when it is one install away.

Each one is true and each one ends the conversation. The shared rule is that a refusal names the
thing to do next, and that the next step works: this project has already shipped a correct
diagnostic pointing at a command that refused a second time.
"""
from __future__ import annotations

import builtins
import json

import pytest

from senbonzakura import baseline, gate


class TestBaselineAnswersADirectory:
    """The commonest wrong argument, because every other command takes a directory."""

    def _run(self, path, capsys):
        code = baseline.main([str(path), "--seeds", "42", "--out", str(path) + ".out.json"])
        return code, " ".join(capsys.readouterr().err.split())

    def test_a_directory_is_named_as_a_directory_rather_than_as_an_errno(self, tmp_path, capsys):
        code, said = self._run(tmp_path, capsys)
        assert code != 0
        assert "is a directory" in said
        assert "Errno" not in said, f"a raw errno reached the reader: {said}"

    def test_it_says_what_to_pass_instead(self, tmp_path, capsys):
        _code, said = self._run(tmp_path, capsys)
        assert "result artefact" in said
        assert ".json" in said, "nothing tells the reader they are looking for a json file"

    def test_a_file_that_is_simply_absent_still_says_so(self, tmp_path, capsys):
        """The directory branch must not swallow the ordinary missing-file case."""
        _code, said = self._run(tmp_path / "nope.json", capsys)
        assert "is a directory" not in said
        assert "nope.json" in said


class TestGateSaysHowToGetABaseline:
    def test_a_missing_baseline_names_the_command_that_writes_one(self, tmp_path):
        said = []
        code = gate.run(["--baseline", str(tmp_path / "a.json"),
                         "--current", str(tmp_path / "b.json")], log=said.append)
        text = " ".join(" ".join(said).split())
        assert code == gate.REFUSED
        assert "senbonzakura baseline" in text, (
            f"the refusal does not name the command that produces the file it wants: {text}")
        assert "is not there" in text
        # AND THE EXPLANATION, not only the command. A mutation pass removed the sentence saying a
        # baseline is a file this tool writes and this test still passed, because the command line
        # below it survived. Somebody meeting this on a first run does not know a baseline is
        # something they produce rather than something they were supposed to have, and a bare command
        # with no sentence around it is a thing to copy rather than a thing to understand.
        assert "a file this tool writes" in text, (
            f"the refusal gives a command and never says what the file is: {text}")

    def test_a_directory_given_as_a_baseline_is_named_as_one(self, tmp_path):
        said = []
        gate.run(["--baseline", str(tmp_path), "--current", str(tmp_path)], log=said.append)
        assert "is a directory" in " ".join(said)

    def test_a_file_with_no_schema_is_not_described_as_a_version_problem(self, tmp_path):
        """"declares schema None" sends a reader looking for a migration that does not exist.

        The two branches above this one in `baseline.read` catch the shapes the tool recognises, a
        result artefact and an archived compass run. This is the case they do not match, and it had
        the defect those two were written to fix.
        """
        odd = tmp_path / "odd.json"
        odd.write_text(json.dumps({"something": "else"}), encoding="utf-8")
        said = []
        code = gate.run(["--baseline", str(odd), "--current", str(odd)], log=said.append)
        text = " ".join(" ".join(said).split())
        assert code == gate.REFUSED
        assert "declares schema None" not in text, f"the version sentence is back: {text}"
        assert "no `schema` field" in text
        assert "senbonzakura baseline" in text, "no next step is named"

    def test_a_genuinely_older_schema_is_still_a_version_problem(self, tmp_path):
        """The new branch must not swallow the case the old sentence was right about."""
        old = tmp_path / "old.json"
        old.write_text(json.dumps({"schema": "senbonzakura-baseline/0"}), encoding="utf-8")
        said = []
        gate.run(["--baseline", str(old), "--current", str(old)], log=said.append)
        text = " ".join(" ".join(said).split())
        assert "declares schema" in text and "understands" in text


class TestTrackAnswersLikeReportDoes:
    def test_no_arguments_gets_a_worked_command_rather_than_a_flag_list(self):
        from senbonzakura import track

        with pytest.raises(SystemExit) as e:
            track.main([])
        said = str(e.value)
        assert "senbonzakura track --harmful" in said, (
            f"no runnable command is shown, which is what `report` gives: {said}")
        assert "one prompt per line" in said, "nothing says what the two files contain"

    def test_it_distinguishes_building_a_track_from_using_the_bundled_one(self):
        """The likeliest reason somebody typed this is that they wanted the track that ships."""
        from senbonzakura import track

        with pytest.raises(SystemExit) as e:
            track.main([])
        assert "--track default" in str(e.value)

    def test_every_line_of_it_fits_a_terminal(self):
        from senbonzakura import say, track

        with pytest.raises(SystemExit) as e:
            track.main([])
        for line in str(e.value).splitlines():
            assert len(line) <= say.CEILING, f"{len(line)} columns: {line!r}"


class TestCompletionSaysItIsOneInstallAway:
    """An optional feature must not degrade into a denial that it exists."""

    def _parser_without_shtab(self, monkeypatch, full=False):
        real = builtins.__import__

        def refuse_shtab(name, *a, **k):
            if name == "shtab":
                raise ImportError("no shtab in this install")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", refuse_shtab)
        from senbonzakura import parser

        built = parser.build_parser(full=full)
        monkeypatch.setattr(builtins, "__import__", real)
        return built

    def test_the_flag_is_accepted_and_explains_itself(self, monkeypatch, capsys):
        built = self._parser_without_shtab(monkeypatch)
        with pytest.raises(SystemExit) as e:
            built.parse_args(["some/model", "--print-completion"])
        said = capsys.readouterr().err + capsys.readouterr().out
        assert e.value.code == 2
        assert "unrecognized" not in said, (
            f"the flag still reads as nonexistent rather than uninstalled: {said}")
        assert "senbonzakura[completion]" in said, "the install command is not named"

    def test_it_is_still_offered_in_the_full_help_so_the_flag_is_discoverable(self, monkeypatch):
        """`--help-all`, not `--help`: the short help is a curated handful and always has been.

        The point being held down is that the flag is listed SOMEWHERE on an install that cannot
        honour it, so a reader can find out it exists and what it needs.
        """
        built = self._parser_without_shtab(monkeypatch, full=True)
        assert "--print-completion" in built.format_help()
        assert "senbonzakura[completion]" in built.format_help()

    def test_the_audit_does_not_count_it_as_a_dead_flag(self):
        """It is handled at parse time, so nothing reads the dest, which is the audit's own signal.

        Asserted here as well as in `test_dead_flags.py` because the exemption it relies on had a
        hole: it was consulted only for a flag written with an explicit `dest=`, and this one lets
        argparse derive it.
        """
        import pathlib
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "research"))
        import audit_flags

        assert "print_completion" in audit_flags.SELF_HANDLED_DESTS
