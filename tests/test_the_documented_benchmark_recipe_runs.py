# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Four things found on 2026-09-26 by running `docs/guide/benchmark.md` as a user would.

HOW THESE WERE FOUND, because it is the reason they are worth a file. The operator asked for the
comparison to be set up from the tool's own `--help`, its documentation and its man page, with no
source reading at all. Twenty minutes of that turned up one defect that stops the documented recipe
dead and three places where the tool knows something and does not say it. None of them is visible
from the inside, which is the point: every one was invisible to a reader who already knew the
answer.

  1. THE ARMS WERE LAUNCHED AS BARE `python`. Fatal on the default path. `--isolate none` is the
     default, so the arms run on the host, and a virtualenv invoked by its full path
     (`~/venv/bin/python -m senbonzakura head-to-head run ...`) puts no `python` on PATH. The run
     died about thirty seconds in, after loading the model and cutting the baseline, with a ten
     frame traceback ending `could not start 'python'`. The same defect was fixed for the SCORER on
     2026-08-11, and the comment left beside that fix says the arms were thought not to need it
     because they "run inside a container where `python` exists". They do when asked to.
  2. THE REFUSAL FOR A MISSING `--harmful` DID NOT SAY WHAT TO PASS, while the refusal for missing
     slices one field along printed a runnable command. Both appeared in the same block.
  3. `--eval-refusal-final` READS AS A FILE LENGTH AND IS A REACH INTO THE CORPUS. Asked for 128 it
     writes 64, because the first 64 rows are the search-time slice and this file gets the rest. It
     took three runs at three settings to establish that, and a user who stops at one concludes the
     flag is broken. I did.
  4. `slices.json` RECORDED ONE KEY, the track path, while its own help says to re-cut the slices
     "whenever the track changes, or after an upgrade that adds a required slice". Neither question
     could be answered from the file.
"""
import json
import sys
from pathlib import Path

import pytest

from senbonzakura import headtohead

ROOT = Path(__file__).resolve().parents[1]


# ── 1. the interpreter the arms run under ────────────────────────────────────────────

class TestTheArmsRunUnderThisInterpreter:
    """THE FATAL ONE. A bare interpreter name is wrong twice over on a host."""

    def _argv_for(self, tmp_path, monkeypatch, isolate="none"):
        seen = {}

        def runner(argv, *, log=print):
            seen["argv"] = list(argv)
            return 0

        arm = tmp_path / "out"
        headtohead.run_arm(headtohead.ADAPTERS["senbon-k1"], seed=42, model="/models/q",
                           track=tmp_path / "track", out=arm, trials=3, runner=runner,
                           log=lambda _m: None, isolate=isolate,
                           slices=tmp_path / "slices", image="img")
        return seen.get("argv", [])

    def test_a_host_arm_is_launched_with_the_running_interpreter(self, tmp_path, monkeypatch):
        """Not `python`, which need not exist, and if it does need not be this one."""
        argv = self._argv_for(tmp_path, monkeypatch)
        assert argv, "the runner was never called"
        assert argv[0] == sys.executable, (
            f"the arm is launched as {argv[0]!r}. A host with no `python` cannot start it at all, "
            f"and a host with a different `python` runs the arm under another install, so "
            f"`-m senbonzakura` inside the arm need not be the senbonzakura being measured.")
        assert argv[0] != "python"

    def test_no_adapter_puts_a_bare_interpreter_name_on_a_host_command(self, tmp_path,
                                                                      monkeypatch):
        """Every adapter, not the one that was noticed. The fix is in one place for this reason:
        three adapters spelling the interpreter separately is how the scorer and the arms came to
        disagree for six weeks.
        """
        for name in sorted(headtohead.ADAPTERS):
            seen = {}

            def runner(argv, *, log=print, _s=seen):
                _s["argv"] = list(argv)
                return 0

            headtohead.run_arm(headtohead.ADAPTERS[name], seed=1, model="/models/q",
                               track=tmp_path / "track", out=tmp_path / name, trials=1,
                               runner=runner, log=lambda _m: None,
                               slices=tmp_path / "slices")
            assert seen["argv"][0] != "python", f"{name} still launches a bare `python` on a host"

    def test_a_container_arm_keeps_the_guest_interpreter(self, tmp_path, monkeypatch):
        """THE HALF THAT MUST NOT CHANGE. `sys.executable` is a path on THIS machine, and inside an
        image it is a path to nothing. The container's own `python` is the correct name there, so
        the repair is scoped to the host path rather than applied everywhere.
        """
        monkeypatch.setattr(headtohead, "find_run_isolated", lambda: None)
        argv = self._argv_for(tmp_path, monkeypatch, isolate="docker")
        assert argv[0] == "docker", "the docker wrapper is no longer the outer command"
        assert sys.executable not in argv, (
            "a host interpreter path was passed into a container, where it does not exist")
        assert "python" in argv, "the guest command lost its own interpreter"


# ── 2. a refusal that names what to pass ─────────────────────────────────────────────

def test_the_missing_scoring_input_refusal_names_the_track_directory(tmp_path):
    """A refusal that says a thing is absent and not what belongs there sends the reader to the
    docs for something the command line already knows.
    """
    track = tmp_path / "track"
    track.mkdir()
    problems = headtohead.preflight(
        tools=["senbon-k1", "senbon-k2"], track=track, out=tmp_path / "out",
        model="/models/q", isolate="none", images={}, slices=None,
        harmful="", harmless="", score=True)
    text = "\n".join(problems)
    assert "bad_eval_ds" in text, (
        "the --harmful refusal does not name the track's measured partition, so the reader has to "
        "find docs/guide/benchmark.md to learn that the answer is a subdirectory of the track they "
        "already passed")
    assert "good_ds" in text, "the --harmless refusal does not name the track's harmless side"
    assert "bad_ds" not in text.replace("bad_eval_ds", ""), (
        "the hint offers `bad_ds`, the rows the directions are FITTED on. Scoring on those is this "
        "release's headline critical, so the wrong one must not be suggested.")


# ── 3. the flag that is a reach, not a length ────────────────────────────────────────

def test_the_final_slice_flag_says_it_is_a_reach_and_not_a_file_length():
    """The help has to answer the question the file's line count raises, or three runs do."""
    parser_help = _stage_help()
    assert "--eval-refusal-final" in parser_help
    lowered = parser_help.lower()
    assert "not the size of the file" in lowered or "not the size" in lowered, (
        "the help still reads as the length of `final_prompts.txt`. Asked for 128 with "
        "--eval-refusal 64 it writes 64, and nothing said so.")


def _stage_help():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        headtohead.main(["stage", "--help"])
    return buf.getvalue()


# ── 4. what a slices directory can say about itself ──────────────────────────────────

class TestSliceProvenance:
    def test_the_track_field_keeps_its_exact_spelling(self, tmp_path):
        """THE ONLY LOAD-BEARING FIELD. `slices_match_track` reads it, so widening the record must
        not move it, and a file written before the record was widened must still be readable.
        """
        d = tmp_path / "slices"
        d.mkdir()
        track = tmp_path / "track"
        track.mkdir()
        headtohead.write_slice_provenance(d, track)
        doc = json.loads((d / headtohead.SLICE_PROVENANCE).read_text(encoding="utf-8"))
        assert Path(doc["track"]) == track.resolve()
        assert headtohead.slices_match_track(d, track) == []

    def test_a_record_written_before_the_counts_existed_is_still_accepted(self, tmp_path):
        """An older slices directory is stale, not damaged, and the distinction is one this command
        already draws elsewhere. A reader that required the new fields would refuse every set
        staged before today.
        """
        d = tmp_path / "slices"
        d.mkdir()
        track = tmp_path / "track"
        track.mkdir()
        (d / headtohead.SLICE_PROVENANCE).write_text(
            json.dumps({"track": str(track.resolve())}), encoding="utf-8")
        assert headtohead.slices_match_track(d, track) == []

    def test_the_counts_and_the_budgets_are_both_recorded(self, tmp_path):
        """BOTH, because they are different numbers and only one of them is the file's own length.
        Recording the request alone would record the more misleading of the pair.
        """
        d = tmp_path / "slices"
        d.mkdir()
        track = tmp_path / "track"
        track.mkdir()
        headtohead.write_slice_provenance(
            d, track, counts={"final": 64, "keyword": 64},
            budgets={"eval_refusal": 64, "eval_refusal_final": 128})
        doc = json.loads((d / headtohead.SLICE_PROVENANCE).read_text(encoding="utf-8"))
        assert doc["counts"]["final"] == 64
        assert doc["budgets"]["eval_refusal_final"] == 128, (
            "the budget and the count must both be present: they disagree by design and a reader "
            "cannot tell which is which from one of them")
        assert doc["counts"]["final"] != doc["budgets"]["eval_refusal_final"]

    def test_it_records_when_it_was_cut_and_by_what_version(self, tmp_path):
        """The two questions its own help raises: has the track changed since, and was this cut
        before the upgrade that added a required slice.
        """
        d = tmp_path / "slices"
        d.mkdir()
        track = tmp_path / "track"
        track.mkdir()
        headtohead.write_slice_provenance(d, track)
        doc = json.loads((d / headtohead.SLICE_PROVENANCE).read_text(encoding="utf-8"))
        assert doc["staged_at"].endswith("+00:00"), "the timestamp is not unambiguous"
        assert doc["tool_version"]

    def test_no_prompt_text_reaches_the_record(self, tmp_path):
        """Counts and budgets only. This file sits beside six files of prompts and must not become
        a seventh.
        """
        d = tmp_path / "slices"
        d.mkdir()
        track = tmp_path / "track"
        track.mkdir()
        headtohead.write_slice_provenance(d, track, counts={"final": 64},
                                          budgets={"eval_refusal": 64})
        doc = json.loads((d / headtohead.SLICE_PROVENANCE).read_text(encoding="utf-8"))
        assert set(doc) == {"track", "counts", "budgets", "staged_at", "tool_version"}
        for block in ("counts", "budgets"):
            assert all(isinstance(v, int) for v in doc[block].values()), (
                f"{block} carries something that is not a number")


# ── the recipe itself, which is what a reader follows ────────────────────────────────

def test_the_documented_recipe_passes_flags_this_command_still_has():
    """The recipe in the guide is the thing a stranger runs, so its flags have to exist.

    Cheap, and it is the check that would have caught the renamed harness the same page admits to:
    it told a reader to run `senbonzakura bench --help` for a month after the command became
    `head-to-head`.
    """
    page = (ROOT / "docs" / "guide" / "benchmark.md").read_text(encoding="utf-8")
    assert "head-to-head stage --track" in page, "the guide no longer shows the staging step first"
    run_help = _run_help()
    for flag in ("--tools", "--seeds", "--trials", "--model", "--track", "--eval-slices",
                 "--harmful", "--harmless", "--out"):
        assert flag in page, f"the documented recipe no longer passes {flag}"
        assert flag in run_help, f"the documented recipe passes {flag} and the command has no such flag"


def _run_help():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        headtohead.main(["run", "--help"])
    return buf.getvalue()
