# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A check that runs after the decision it should have informed is not a check.

Every pre-flight in this tool already existed, and every one of them ran inside `run_parsed`,
which the guided mode reaches only once somebody has answered "Run it? y". So the walk asked for
a commitment and then went to find out whether the run was possible.

These hold three things: the checks run before the confirm, the board collapses to one line when
it has nothing to say, and a failing check changes the question rather than disappearing.
"""
import json

import pytest

from senbonzakura import interactive


def _plan(tmp_path, **options):
    opts = {"--model": "Qwen/Qwen3-1.7B", "--track": "default",
            "--out": str(tmp_path / "abliterated"), "--trials": "200", "--device": "cpu"}
    opts.update(options)
    return {"command": "kageyoshi", "options": opts, "licence": None}


@pytest.fixture
def all_clear(monkeypatch):
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.PASS, "the device", "")])


def test_a_clean_board_is_one_line(tmp_path, all_clear):
    """NINE TICKS IS CEREMONY, and a reader learns to skip ceremony. The run this board protects
    is the one where the tenth line says something, so a clean board has to stay small enough
    that the day it is not clean is visible.
    """
    said = []
    assert interactive._board(_plan(tmp_path), log=said.append) is True
    body = [s for s in said if s.strip()]
    assert len(body) == 1, f"a clean pre-flight printed {len(body)} lines: {body}"
    assert "all clear" in body[0]
    assert "PRE-FLIGHT" not in body[0], "the board is drawn only when it has something to say"


def test_a_board_with_something_to_say_is_drawn_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.FAIL, "the device", "no CUDA device here")])
    said = []
    assert interactive._board(_plan(tmp_path), log=said.append) is False
    text = "\n".join(said)
    assert "PRE-FLIGHT" in text
    assert "no CUDA device here" in text
    assert "WHAT YOU CHOSE" in text and "YOUR MACHINE" in text and "WHAT WE CHECKED" in text
    assert "Qwen/Qwen3-1.7B" in text, "the expensive mistake is the model, so it is in the first rows"


def test_the_board_comes_before_the_question(tmp_path, all_clear):
    """THE WHOLE POINT. An answer that arrives after the person has committed is a receipt."""
    said = []
    asked = []

    def ask_fn(prompt):
        asked.append((len(said), prompt))
        return "n"

    interactive.present(_plan(tmp_path), ask_fn=ask_fn, log=said.append)
    board_at = next(i for i, s in enumerate(said) if "all clear" in s)
    confirm_at = asked[0][0]
    assert board_at < confirm_at, (
        "the confirm was asked before the pre-flight had reported, which is the ordering this "
        "whole change exists to invert")


def test_a_failing_check_changes_the_question(tmp_path, monkeypatch):
    """A SOFT GATE, in the words the rest of the tool uses. The person still decides, and the
    warning is repeated in the one word they are about to answer.
    """
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.FAIL, "the device", "no CUDA device here")])
    asked = []

    def ask_fn(prompt):
        asked.append(prompt)
        return "n"

    interactive.present(_plan(tmp_path), ask_fn=ask_fn, log=lambda *_a: None)
    assert any("anyway" in p for p in asked), asked


def test_a_refusal_from_a_real_preflight_becomes_a_row(tmp_path, monkeypatch):
    """The board CALLS the checks rather than reimplementing them, so this proves the wiring.

    A second copy of "is this device usable" that agreed with the first until one of them was
    edited is this project's most repeated defect.
    """
    from senbonzakura import cli

    def refuse(args, log=print):
        raise SystemExit("this device is not usable here")

    monkeypatch.setattr(cli, "_preflight_device", refuse)
    rows = interactive._checked_rows(_plan(tmp_path))
    assert rows is not None
    failing = [r for r in rows if r[0] == interactive.FAIL]
    assert failing, rows
    assert any("this device is not usable here" in r[2] for r in failing)


def test_a_check_that_crashes_is_a_finding_and_not_a_pass(tmp_path, monkeypatch):
    """Fail loud. A check that raises something other than a refusal used to take the run with it
    or, worse in a board, could have been counted as a tick.
    """
    from senbonzakura import cli

    def explode(args, log=print):
        raise RuntimeError("the check itself is broken")

    monkeypatch.setattr(cli, "_preflight_device", explode)
    rows = interactive._checked_rows(_plan(tmp_path))
    marks = {r[1]: r[0] for r in rows}
    assert marks.get("the device") == interactive.ADVISORY
    assert any("the check itself is broken" in r[2] for r in rows)


def test_a_base_install_says_so_rather_than_claiming_a_clean_board(tmp_path, monkeypatch):
    """The guided mode runs where torch does not. An unanswerable check must not read as a pass."""
    monkeypatch.setattr(interactive, "_checked_rows", lambda plan: None)
    said = []
    assert interactive._board(_plan(tmp_path), log=said.append) is True
    text = "\n".join(said)
    assert "not installed here" in text
    assert "all clear" not in text, "nothing was checked, so nothing may be declared clear"


# ── Scene 10, the screen that was not there ──────────────────────────────────────────────

def _finished_plan(tmp_path, **record):
    out = tmp_path / "abliterated"
    out.mkdir()
    if record:
        (out / "abliteration.json").write_text(json.dumps(record), encoding="utf-8")
    return {"command": "kageyoshi", "licence": None,
            "options": {"--model": "m", "--out": str(out), "--trials": "200"}}, out


def test_a_finished_run_says_what_you_now_have(tmp_path):
    """A successful guided run used to print NOTHING. Somebody waited twenty three minutes."""
    plan, out = _finished_plan(tmp_path, baseline_refusals=0.391, post_bake_refusals=0.0,
                               post_bake_kl=0.047)
    said = []
    interactive.finished(plan, seconds=1384, ask_fn=lambda _p: "1", log=said.append)
    text = "\n".join(said)
    assert str(out) in text, "where it is"
    assert "200 trials" in text and "23m 04s" in text, "what it cost"
    assert "39.1%" in text and "0.0%" in text, "what changed"
    assert "0.047" in text and "held-out prompts" in text, (
        "the drift line carries the caveat that says which of the two figures survives contact "
        "with anything else, and it is not decoration")


def test_the_numbers_are_read_back_out_of_the_artefact(tmp_path):
    """READ, NOT REMEMBERED. A figure carried out of the run in a variable is a second account of
    a measurement, and this project has withdrawn published numbers over exactly that.
    """
    plan, _out = _finished_plan(tmp_path, baseline_refusals=0.5, post_bake_refusals=0.25)
    said = []
    interactive.finished(plan, ask_fn=lambda _p: "1", log=said.append)
    assert "50.0%" in "\n".join(said) and "25.0%" in "\n".join(said)


def test_a_run_whose_artefact_is_unreadable_still_says_what_you_have(tmp_path):
    plan, out = _finished_plan(tmp_path)
    (out / "abliteration.json").write_text("{ this is not json", encoding="utf-8")
    said = []
    interactive.finished(plan, ask_fn=lambda _p: "1", log=said.append)
    assert str(out) in "\n".join(said), "a damaged artefact costs the figures, not the screen"


def test_pressing_enter_at_the_end_does_not_start_another_job(tmp_path):
    """AGAINST THE DESIGN, deliberately: it highlighted the conversion. Somebody who reaches this
    screen has what they came for, and a default keystroke that starts another job is the trap the
    confirm before the run was rewritten to avoid.
    """
    plan, _out = _finished_plan(tmp_path, baseline_refusals=0.4, post_bake_refusals=0.0)
    assert interactive.finished(plan, ask_fn=lambda _p: "", log=lambda *_a: None) is None


def test_the_follow_on_is_a_command_the_reader_can_see(tmp_path):
    """The rule this file keeps: anything it runs is printed first, and a follow-on is no
    exception. So every row names its command rather than only a verb.
    """
    plan, out = _finished_plan(tmp_path, baseline_refusals=0.4, post_bake_refusals=0.0)
    said = []
    step = interactive.finished(plan, ask_fn=lambda _p: "3", log=said.append)
    assert "senbonzakura score" in "\n".join(said) or "senbonzakura convert" in "\n".join(said)
    assert step is not None and step["command"] in ("convert", "score")


def test_a_refusal_keeps_its_own_layout_under_the_board(tmp_path, monkeypatch):
    """A pre-flight refusal is a formatted block with a "what to do" list and commands in it.

    Feeding the whole thing through the row renderer produced a hundred column line inside a
    board laid out for eighty, and `say` leaves indented lines alone deliberately, because they
    are commands somebody has to paste. So the row carries the first line and the block is
    printed below it, intact.
    """
    refusal = ("senbonzakura: --device is 'cuda' and this machine has no usable CUDA device.\n"
               "  What to do:\n"
               "    run on the processor instead:  --device cpu")
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.FAIL, "the device", refusal)])
    said = []
    interactive._board(_plan(tmp_path), log=said.append)
    row = next(s for s in said if "✗" in s)
    assert "What to do" not in row, "the whole refusal was folded into one row of the board"
    # ONCE, NOT TWICE. The row carries the first line and the block below carries all of it, so a
    # renderer that also folds the rest into the row says everything twice. Counting is what
    # catches that: the earlier version of this test asserted on the first row alone and passed
    # against exactly that renderer, because the duplicate landed on the line after it.
    assert "\n".join(said).count("What to do") == 1, (
        "the refusal's body is in the board and in the block below it, so the reader meets the "
        "same sentences twice, once with the board's padding through them")
    # VERBATIM, AS A BLOCK. Asserting the stripped text of some line somewhere is what the first
    # version of this did, and it passed against a renderer that folded the whole refusal into
    # the board and padded every continuation line to the label column. The refusal's own
    # indentation is the thing being protected, because that is what makes its commands
    # pasteable, so the block is matched exactly.
    block = "\n".join(f"    {line}" for line in refusal.splitlines())
    assert block in "\n".join(said), (
        "the refusal did not survive as its own block, so the commands in it lost the spacing "
        "that makes them pasteable")


def test_the_machine_does_not_report_the_card_twice(tmp_path, monkeypatch):
    """TWO MECHANISMS FOR ONE FACT, which is this project's most repeated defect.

    The first version of the board asked `device_available` itself and printed a card row beside
    the row that `_preflight_device` produces, so a machine with no card reported the same thing
    twice, in two wordings, and the counts line said two failures over one problem.
    """
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.FAIL, "the device", "no CUDA device here")])
    said = []
    interactive._board(_plan(tmp_path, **{"--device": "cuda"}), log=said.append)
    failing = [s for s in said if "✗" in s]
    assert len(failing) == 1, f"one absent card produced {len(failing)} failing rows: {failing}"
    assert "1 failing" in "\n".join(said)


def test_the_probe_budget_is_on_the_board(tmp_path, monkeypatch):
    """FOUND BY A REAL RUN, 2026-09-28, which is the only way this class of gap gets found.

    A guided walk on a CPU printed "14 checks, all clear", the run was confirmed, and thirty
    seconds later it refused because the capability probe would take about four hours on that
    machine. The refusal was right and it was decidable from the command line alone, so the board
    had said clear about a run the tool was already going to stop. A reader told twice believes
    the wrong one.
    """
    from senbonzakura import capability

    def refuse(*_a, **_k):
        raise SystemExit("this probe would take roughly 4.1 hours on this CPU")

    monkeypatch.setattr(capability, "refuse_a_slow_probe", refuse)
    rows = interactive._checked_rows(_plan(tmp_path))
    assert any(r[0] == interactive.FAIL and "4.1 hours" in r[2] for r in rows), (
        f"the probe budget is not among the checks the board runs: {[r[1] for r in rows]}")


def test_the_label_column_is_measured_rather_than_assumed(tmp_path, monkeypatch):
    """A fixed width was fine until a label outgrew it, and then that row's detail started four
    columns right of every other row's, which loses the single edge the board is laid out around.
    """
    monkeypatch.setattr(interactive, "_checked_rows",
                        lambda plan: [(interactive.FAIL, "a very long label indeed", "why")])
    said = []
    interactive._board(_plan(tmp_path), log=said.append)
    starts = {line.index(detail) for line in said
              for detail in ("why", "cpu") if detail in line and line.startswith("    ")}
    assert len(starts) == 1, f"the details begin at {sorted(starts)}, so the column has no edge"


#: Pre-flights `run_parsed` performs that the board deliberately does NOT, each with the reason.
#: A name here is a decision; a name missing from both this and the board is the defect below.
BOARD_LEAVES_TO_THE_RUN = {
    # Reads the checkpoint's safetensors headers off the Hub. It is a network call, and the run
    # makes it seconds later anyway, so the board would pay for it twice and would fail on a
    # machine that is merely offline rather than misconfigured.
    "preflight_snapshot_ram",
}


def test_every_check_the_run_makes_is_a_check_the_board_makes():
    """THE GUARD FOR THE NEXT ONE, rather than for the one that was found.

    The capability probe's budget was decidable from the command line and was missing from the
    board, so a walk printed "all clear" about a run the tool then refused. A second pre-flight
    added to `run_parsed` next month would do the same thing, silently, and the board would go on
    looking right. So the two lists are compared rather than kept in step by hand.
    """
    import inspect
    import re

    from senbonzakura import cli

    in_the_run = set(re.findall(r"\b(_preflight_\w+|refuse_without_a_track|refuse_a_slow_probe)\b",
                                inspect.getsource(cli.run_parsed)))
    in_the_run |= set(re.findall(r"\b(preflight_snapshot_ram)\b", inspect.getsource(cli.run_parsed)))
    on_the_board = set(re.findall(r"\b(_preflight_\w+|refuse_without_a_track|refuse_a_slow_probe)\b",
                                  inspect.getsource(interactive._checked_rows)
                                  + inspect.getsource(interactive._probe_budget)))
    missing = in_the_run - on_the_board - BOARD_LEAVES_TO_THE_RUN
    assert not missing, (
        f"{sorted(missing)} run inside the abliteration and not on the pre-flight board, so the "
        f"guided mode can print 'all clear' about a run that is then refused. Either add them to "
        f"`_checked_rows` or name them in BOARD_LEAVES_TO_THE_RUN with the reason.")


def test_a_cpu_walk_produces_a_command_its_own_preflight_accepts(tmp_path, monkeypatch):
    """THE WALK HANDED PEOPLE A COMMAND THAT COULD NOT RUN.

    The capability probe defaults to 200 items at 512 new tokens, minutes on a card and about
    four hours on a processor, and `refuse_a_slow_probe` stops the second before it starts. The
    walk never asked about either, so somebody on a laptop answered six questions and got a
    command the tool then refused, over flags they had no reason to know existed.
    """
    options = {"--device": "cpu"}
    note = interactive.size_the_probe_for("cpu", options)
    assert options["--capability-n"] == "40" and options["--capability-max-new"] == "256"
    assert note and "40 items rather than 200" in note, (
        "a smaller probe changes what the capability figure means, so it is never applied in "
        "silence")

    on_a_card = {"--device": "cuda"}
    assert interactive.size_the_probe_for("cuda", on_a_card) is None
    assert on_a_card == {"--device": "cuda"}, "a card needs no help and gets no extra flags"


def test_the_sizing_is_on_the_printed_command_rather_than_applied_invisibly(tmp_path):
    """The contract of this file: the line somebody copies is the line that ran."""
    plan = _plan(tmp_path, **{"--device": "cpu"})
    interactive.size_the_probe_for("cpu", plan["options"])
    line = interactive.render_command(plan["command"], plan["options"])
    assert "--capability-n 40" in line and "--capability-max-new 256" in line
