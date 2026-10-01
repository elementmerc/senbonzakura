# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The guided walk, driven the way somebody actually walks it.

EVERY DEFECT THESE ENCODE WAS FOUND BY DOING THIS BY HAND on 2026-09-28, and every one of them was
invisible to the unit tests, because each lived in the SEQUENCE rather than in any function: the
pre-flight board vouched for a run the tool then refused, the walk on a processor produced a
command its own pre-flight rejects, and a menu answer that was a valid index stopped being one when
the menu grew a row.

The answers below are positions on menus, which is exactly the thing that moved, so each journey
asserts on what came back rather than trusting the position: if a menu grows a row and a journey
starts answering a different question, the assertions are what notice.
"""
import json

import pytest

from .harness import INTERRUPT

#: A walk with nothing cached: the recipe, a model typed by hand, the bundled prompts, the
#: processor, an output directory and a trial budget.
WALK = ("1", "Qwen/Qwen2.5-0.5B-Instruct", "1", "2", "edited", "5")

#: No card, whatever the machine running this happens to have. A journey whose result depends on
#: the hardware under it is not a journey.
NO_CARD = {"CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.journey("command:-i", "state:no-tty")
def test_a_script_gets_the_flags_rather_than_a_menu(session, tmp_path):
    """A prompt in a non-interactive context is a hang, and a hung CI job is how this is found
    the expensive way. It is refused in one sentence that names what a script should use instead.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "senbonzakura", "-i"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        cwd=str(tmp_path), check=False)
    combined = done.stdout + done.stderr
    assert done.returncode == 2, f"exited {done.returncode}: {combined[-800:]}"
    assert "needs a terminal" in combined
    assert "--help" in combined, "a refusal that names no way forward is half a refusal"


@pytest.mark.journey("command:interactive", "state:fake-tty")
def test_a_fake_terminal_that_says_so_is_refused_rather_than_asked(session):
    """The case the journey above cannot reach, because this harness is a real terminal.

    `isatty` is an inference. The journey above pipes `/dev/null` in, which the inference gets
    right. This one runs through the harness's own pseudo-terminal with **no answers queued**,
    which is what `docker run -t`, `expect` and some CI agents give the command: something that
    satisfies `isatty` with nobody behind it. Without `--no-interactive` that walks into the menu
    and waits, and a waiting job is how this gets found the expensive way.

    No answers are passed on purpose, which makes the failure mode worth stating. If the flag ever
    stops being read, the walk stops at its first question with nothing left to answer it, and the
    harness now **proves** that rather than waiting it out: the kernel reports the child blocked
    reading its terminal, every answer is already sent, so nobody is ever going to type anything.
    Measured at 0.4s against the 120s it used to cost, and the message names the one cause instead
    of offering two.
    """
    result = session("interactive", "--no-interactive", env=NO_CARD)
    result.exited(2)
    result.says("--no-interactive")
    result.never_says("Choose 1 to")
    result.carried_no_traceback()


@pytest.mark.journey("command:-i", "state:nothing-cached", "state:cancelled-at-confirm")
def test_the_whole_walk_and_then_changing_your_mind(session):
    """The commonest session there is: answer everything, read the command, and decline.

    Nothing may be written, the tool may not claim anything ran, and the command has to be on the
    screen, because the whole contract of the guided mode is that what it prints is what it runs.
    """
    result = session("-i", answers=(*WALK, "n"), env=NO_CARD)
    result.exited(0).carried_no_traceback().fits_the_window(
        allowed=("senbonzakura kageyoshi",))
    result.says(
        "Nothing on this machine holds weights",     # nothing cached, so it asks rather than lists
        "senbonzakura kageyoshi --model Qwen/Qwen2.5-0.5B-Instruct",
        "--out edited",
        "--trials 5",
        "Nothing was run.")
    result.never_says("Qwen/Qwen3-1.7B")             # the nominated default that used to be here


@pytest.mark.journey("command:-i", "state:nothing-cached")
def test_the_preflight_reports_before_the_question_and_not_after(session):
    """THE ORDERING THIS WHOLE BOARD EXISTS TO INVERT. Every check already existed and every one
    ran inside the abliteration, which the walk reaches only once somebody has answered yes.
    """
    result = session("-i", answers=(*WALK, "n"), env=NO_CARD)
    board = result.output.index("Pre-flight")
    question = result.output.index("Run it?")
    assert board < question, (
        "the confirm was asked before the pre-flight had reported, so the answer arrived after "
        "the decision it should have informed")


@pytest.mark.journey("command:-i", "state:failing-preflight")
def test_choosing_a_card_this_machine_does_not_have(session):
    """The board draws in full, the reader gets the whole refusal, and the question changes.

    Found by driving it: the board used to report the absent card twice, in two wordings, from two
    mechanisms, and its counts line said two failures over one problem.
    """
    result = session("-i", answers=("1", "Qwen/Qwen2.5-0.5B-Instruct", "1", "1", "edited", "5",
                                    "n"), env=NO_CARD)
    result.exited(0).carried_no_traceback()
    result.says("PRE-FLIGHT", "WHAT WE CHECKED", "1 failing", "Run it anyway?")
    result.fits_the_window(allowed=("senbonzakura kageyoshi", "--device cpu", "--capability-n"))


@pytest.mark.journey("command:-i", "state:answered-wrong-then-right")
def test_a_wrong_answer_is_not_a_dead_end(session):
    """A menu that accepts nonsense, or that gives up on a typo, sends people back to the flags."""
    result = session("-i", answers=("99", "x", "1", "Qwen/Qwen2.5-0.5B-Instruct", "1", "2",
                                    "edited", "5", "n"), env=NO_CARD)
    result.exited(0).carried_no_traceback()
    result.says("is not one of", "senbonzakura kageyoshi")


@pytest.mark.journey("command:-i", "state:interrupted-mid-run")
def test_ctrl_c_partway_through_changes_nothing(session):
    """The walk's own first line promises that Ctrl+C stops at any point and changes nothing.

    It used to exit through optuna as a ninety line traceback, against that printed promise.
    """
    result = session("-i", answers=("1", INTERRUPT), env=NO_CARD)
    result.carried_no_traceback().says("Stopped. Nothing was changed.")
    result.exited(130)


@pytest.mark.journey("command:-i", "state:narrow-terminal")
def test_the_walk_in_a_half_width_window(session):
    """Hand wrapped at 79 columns reads correctly on a wide terminal and raggedly on a narrow one,
    and the guided mode is the screen most likely to be open beside something else.
    """
    result = session("-i", answers=(*WALK, "n"), columns=54, env=NO_CARD)
    result.exited(0).fits_the_window(allowed=("senbonzakura kageyoshi",))


@pytest.mark.journey("command:-i", "state:resumed")
def test_yesterday_s_run_is_offered_before_anything_is_asked(session, work):
    """THE CHEAPEST RUN IS THE ONE ALREADY HALF DONE, and the walk could create a paused run and
    could not find one, so somebody who paused yesterday was sent back to the flag list.
    """
    out = work / "edited"
    out.mkdir()
    (out / "senbon-study.db").write_bytes(b"not a real study, and this screen never opens it")
    (out / "best-config.json").write_text("{}", encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"model": "Qwen/Qwen2.5-0.5B-Instruct", "track": "default"}), encoding="utf-8")

    result = session("-i", answers=("1", "n"), env=NO_CARD)
    result.carried_no_traceback()
    result.says("run here you can pick up", "edited", "--resume")
