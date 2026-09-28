# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Tests for the guided mode (senbonzakura/interactive.py).

The load-bearing property is that the printed command is exactly the run that happens. A guided
mode that displayed one thing and passed another would be an excellent way to hide a mistake, so
that equivalence is tested directly rather than assumed from the code reading correctly.
"""
import sys

import pytest

from senbonzakura import interactive as it


class _Tty:
    def isatty(self):
        return True


class _NotTty:
    def isatty(self):
        return False


def _answers(*values):
    seq = iter(values)
    return lambda _prompt: next(seq)


def _row_number(side, key):
    """Where a corpus sits on its side's menu, 1 based. See the note in the track test."""
    return next(i for i, e in enumerate(it.by_side(side), 1) if e["key"] == key)


#: A MACHINE WITH NOTHING ON IT, and every walk below runs on one.
#:
#: Two screens in this walk read the machine rather than the answers: the model question offers
#: what is already in the Hub cache, and the pre-flight board runs the real checks. Left alone,
#: both make these tests depend on whatever is cached and writable on the box running them, which
#: is how `test_occupied_output.py` once passed on a build box holding a stray `abliterated/` and
#: would have failed on a clean runner. The screens themselves are covered by
#: `test_the_menus_describe_this_install.py` and `test_the_walk_checks_before_it_asks.py`, where
#: the machine is made explicitly rather than inherited.
@pytest.fixture(autouse=True)
def _a_machine_with_nothing_on_it(monkeypatch):
    monkeypatch.setattr(it, "models_on_this_machine", lambda root=".": [])
    monkeypatch.setattr(it, "_checked_rows", lambda plan: [])
    # AND THE BUNDLED TRACK IS PRESENT, which is the third thing these walks read off the machine.
    # A release wheel carries the packed track and a source checkout does not, so on CI the track
    # menu asks an extra question ("Ask for it anyway?") that it does not ask on a dev box, every
    # canned answer after it lines up against the wrong prompt, and the walk ends in StopIteration.
    # That is the same defect as the two above wearing different clothes: a test reading the
    # machine rather than saying what it means. The unbundled path has its own tests, which
    # monkeypatch this deliberately.
    monkeypatch.setattr(it.bundled, "is_available", lambda: True)


# ── quoting ──────────────────────────────────────────────────────────────────────
#
# BOTH SHELLS ARE TESTED ON EVERY MACHINE. The printed command exists to be pasted and run, so
# quoting it for the wrong shell makes it wrong rather than untidy, and the reader who meets that
# is on the platform the author is not. These simulate the platform instead of waiting for a CI
# job on another one to say so.
@pytest.fixture(params=["posix", "windows"])
def shell(request, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32" if request.param == "windows" else "linux")
    return request.param


@pytest.mark.parametrize("value", ["Qwen/Qwen3-1.7B", "out", "default", "train[:400]", "a=b"])
def test_ordinary_values_are_not_quoted(value, shell):
    assert it.quote(value) == value


def test_a_value_with_a_space_is_quoted(shell):
    assert it.quote("my track") == ('"my track"' if shell == "windows" else "'my track'")


def test_an_empty_value_is_quoted(shell):
    assert it.quote("") == ('""' if shell == "windows" else "''")


def test_a_value_with_a_quote_is_escaped(shell):
    if shell == "windows":
        # An apostrophe is an ordinary character to `cmd.exe`, so wrapping it changes nothing
        # about how it is read; it stays quoted because quoting conservatively is free here.
        assert it.quote("it's") == '"it\'s"'
        # A double quote is the one character `cmd.exe` cannot carry inside a quoted argument.
        # Doubling it is what PowerShell reads and is the nearest thing to a convention.
        assert it.quote('say "hi"') == '"say ""hi"""'
    else:
        assert it.quote("it's") == "'it'\\''s'"


def test_a_windows_path_prints_without_quotes():
    """The defect this branch exists for: every path on Windows contains backslashes.

    POSIX rules quoted all of them, and quoted them with single quotes, which `cmd.exe` passes
    through as part of the path. The resume command the guided mode prints was unusable for the
    only reader it was printed for.
    """
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sys, "platform", "win32")
        assert it.quote(r"C:\Users\dan\brain") == r"C:\Users\dan\brain"
        assert it.quote(r"C:\my runs\brain") == r'"C:\my runs\brain"'


def test_a_backslash_is_still_quoted_on_posix():
    """Where a backslash is an escape character rather than a separator, it has to stay quoted."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sys, "platform", "linux")
        assert it.quote("a\\b") == "'a\\b'"


# ── the printed command ──────────────────────────────────────────────────────────
def test_the_command_renders_in_the_order_asked():
    line = it.render_command("kageyoshi", {"--model": "M", "--track": "T", "--out": "O"})
    assert line == "senbonzakura kageyoshi --model M --track T --out O"


def test_a_true_option_is_a_bare_flag():
    assert it.render_command("x", {"--verbose": True}) == "senbonzakura x --verbose"


def test_none_and_false_options_are_dropped():
    line = it.render_command("x", {"--a": None, "--b": False, "--c": "keep"})
    assert line == "senbonzakura x --c keep"


def test_the_printed_command_is_exactly_what_would_run():
    """The whole design constraint, checked rather than trusted.

    `run` builds an argv from the same options dict it renders. If those ever drift, a user could
    be shown one command and given another, which is the failure this mode must not have.
    """
    plan = it.plan_abliteration(
        ask_fn=_answers("1", "Qwen/Qwen3-0.6B", "1", "2", "out dir", "5"), log=lambda *a: None)
    line = it.render_command(plan["command"], plan["options"])

    argv = [plan["command"]]
    for flag, value in plan["options"].items():
        if value is True:
            argv.append(flag)
        elif value not in (None, False):
            argv += [flag, str(value)]

    rebuilt = "senbonzakura " + " ".join(
        it.quote(a) if not a.startswith("--") and a != plan["command"] else a for a in argv)
    assert rebuilt == line


# ── asking ───────────────────────────────────────────────────────────────────────
def test_an_empty_answer_takes_the_default():
    assert it.ask("q", default="d", ask_fn=_answers(""), log=lambda *a: None) == "d"


def test_an_answer_beats_the_default():
    assert it.ask("q", default="d", ask_fn=_answers("mine"), log=lambda *a: None) == "mine"


def test_a_question_with_no_default_asks_again():
    said = []
    assert it.ask("q", ask_fn=_answers("", "  ", "finally"), log=said.append) == "finally"
    assert any("needs an answer" in s for s in said)


def test_ctrl_c_at_a_question_abandons():
    def boom(_p):
        raise KeyboardInterrupt

    with pytest.raises(it.AbandonedError):
        it.ask("q", ask_fn=boom, log=lambda *a: None)


def test_eof_at_a_question_abandons():
    def boom(_p):
        raise EOFError

    with pytest.raises(it.AbandonedError):
        it.ask("q", ask_fn=boom, log=lambda *a: None)


# ── choosing ─────────────────────────────────────────────────────────────────────
def test_choose_returns_the_index():
    opts = [("a", ""), ("b", ""), ("c", "")]
    assert it.choose("q", opts, ask_fn=_answers("3"), log=lambda *a: None) == 2


def test_choose_takes_the_default_on_enter():
    opts = [("a", ""), ("b", "")]
    assert it.choose("q", opts, default=1, ask_fn=_answers(""), log=lambda *a: None) == 1


def test_choose_rejects_out_of_range_and_asks_again():
    said = []
    opts = [("a", ""), ("b", "")]
    assert it.choose("q", opts, ask_fn=_answers("9", "0", "x", "2"), log=said.append) == 1
    assert sum("not one of" in s for s in said) == 3


def test_choose_marks_the_default():
    said = []
    it.choose("q", [("a", ""), ("b", "")], default=1, ask_fn=_answers(""), log=said.append)
    assert any(s.strip().startswith("* 2.") for s in said)


def test_choose_prints_each_description():
    said = []
    it.choose("q", [("a", "why a"), ("b", "why b")], ask_fn=_answers(""), log=said.append)
    joined = "\n".join(said)
    assert "why a" in joined and "why b" in joined


def test_ctrl_c_at_a_choice_abandons():
    def boom(_p):
        raise KeyboardInterrupt

    with pytest.raises(it.AbandonedError):
        it.choose("q", [("a", "")], ask_fn=boom, log=lambda *a: None)


# ── confirming ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("answer", "expected"), [("y", True), ("yes", True),
                                                  ("n", False), ("no", False)])
def test_confirm_reads_yes_and_no(answer, expected):
    assert it.confirm("q", ask_fn=_answers(answer), log=lambda *a: None) is expected


def test_confirm_takes_its_default_on_enter():
    assert it.confirm("q", default=False, ask_fn=_answers(""), log=lambda *a: None) is False


def test_confirm_asks_again_on_nonsense():
    said = []
    assert it.confirm("q", ask_fn=_answers("maybe", "y"), log=said.append) is True
    assert any("y or n" in s for s in said)


# ── the dataset menu ─────────────────────────────────────────────────────────────
def test_every_offered_dataset_names_a_licence():
    # A corpus whose position has not been checked does not go on a menu.
    for entry in it.KNOWN_DATASETS:
        assert entry["licence"], entry["key"]
        assert entry["note"], entry["key"]


def test_the_bundled_track_is_the_first_choice():
    assert it.KNOWN_DATASETS[0]["key"] == "default"


def test_every_corpus_is_offered_for_the_side_it_actually_holds():
    """THE DEFECT THIS ENCODES. There was one menu, and whatever it returned went to `--track`.

    `--track` takes a directory holding bad_ds, good_ds and bad_eval_ds. Three of the four
    corpora on that menu are a single Hub split holding ONE side of the contrast, so choosing any
    of them printed a command that dies in the pre-flight. Only the bundled entry ever worked.
    """
    assert {e["side"] for e in it.KNOWN_DATASETS} <= {"track", "harmful", "harmless"}
    by_key = {e["key"]: e["side"] for e in it.KNOWN_DATASETS}
    assert by_key["harmful-behaviors"] == "harmful"
    assert by_key["harmless-alpaca"] == "harmless"
    assert by_key["default"] == "track"
    # AND THE SAME PROPERTY OVER THE BUNDLED ROWS, which are generated rather than typed and so
    # could not be checked in the table above. A corpus offered on the wrong side prints a command
    # that dies in the pre-flight, whichever list it came from.
    for side in ("harmful", "harmless"):
        for row in it.by_side(side):
            assert row["side"] == side, row


def test_a_side_menu_shows_the_licence_before_the_choice_is_made():
    said = []
    it.pick_side("harmful", "which?", ask_fn=_answers("1"), log=said.append)
    assert "MIT" in "\n".join(said)
    assert "undeclared upstream" in "\n".join(said)


def test_picking_a_known_corpus_returns_its_spec():
    """The row's own spec, whatever position it sits in. Indices move when the menu grows."""
    harmful = it.by_side("harmful")
    where = next(i for i, e in enumerate(harmful, 1) if e["key"] == "harmful-behaviors")
    spec, licence = it.pick_side("harmful", "which?", ask_fn=_answers(str(where)),
                                 log=lambda *a: None)
    assert spec == "mlabonne/harmful_behaviors::train"
    assert licence == "undeclared upstream"


def test_the_default_harmful_choice_needs_no_account_at_all():
    """REPLACES A TEST THAT PROTECTED THE WRONG THING, 2026-09-28.

    It used to hold down that AdvBench is not the default BECAUSE IT IS GATED, and that its note
    tells the reader to run `hf auth login`. Both statements were about a Hub download this
    package does not need: AdvBench's 520 prompts are inside the wheel, and `--harmful advbench`
    reads them off the disk with the network unplugged. The old menu sent a newcomer to
    authenticate against a service for a file they had already installed.

    What is worth protecting is the property underneath: the row pressing Enter picks must work on
    a machine with no account and no network.
    """
    from senbonzakura import corpora

    first = it.by_side("harmful")[0]
    assert first["spec"] in corpora.CORPORA, (
        f"the default harmful choice is {first['spec']!r}, which is fetched. Taking the default "
        f"is what a newcomer does, and it must not be the one answer that needs an account.")
    for side in ("harmful", "harmless"):
        for row in it.by_side(side):
            assert "huggingface-cli login" not in row["note"], (
                "the superseded login command is back in the menu; see SUPERSEDED_COMMANDS in "
                "test_output_never_names_a_file_the_wheel_lacks.py")


def test_picking_your_own_asks_for_the_path():
    """"Something of my own" is always last, whichever side is being asked for."""
    n = len(it.by_side("harmless"))
    spec, licence = it.pick_side("harmless", "which?", ask_fn=_answers(str(n), "~/mine.txt"),
                                 log=lambda *a: None)
    assert spec == "~/mine.txt"
    assert licence == "yours"


def test_two_hub_corpora_become_a_track_build_step_rather_than_a_broken_track_flag():
    # LOCATED, NOT COUNTED. These were literal menu positions until the bundled corpora were
    # added above them, at which point "1" meant a different corpus and the test read as a
    # regression in the code rather than a move in the menu.
    harmful_at = _row_number("harmful", "harmful-behaviors")
    harmless_at = _row_number("harmless", "harmless-alpaca")
    spec, licence, build = it.pick_track(
        ask_fn=_answers("2", str(harmful_at), str(harmless_at), "mytrack"), log=lambda *a: None)
    assert spec == "mytrack"
    assert build == {"command": "track",
                     "options": {"--harmful": "mlabonne/harmful_behaviors::train",
                                 "--harmless": "mlabonne/harmless_alpaca::train",
                                 "--out": "mytrack"}}
    # Two corpora under two different licences, and the answer carries both rather than
    # reporting one of them as the licence of the pair.
    assert "(harmful)" in licence and "(harmless)" in licence


def test_a_track_directory_already_built_needs_no_build_step():
    spec, _licence, build = it.pick_track(ask_fn=_answers("3", "~/mytrack"),
                                          log=lambda *a: None)
    assert spec == "~/mytrack"
    assert build is None


def test_choosing_the_bundled_track_without_one_says_how_to_get_it(monkeypatch):
    """REWRITTEN 2026-09-28, and the property it pins is unchanged.

    The note used to end with "or choose another option" while `pick_track` returned the bundled
    track regardless, so the only way out of that screen was Ctrl+C at the final confirm. Taking it
    anyway is now a decision rather than the only outcome, which is the second answer here.
    """
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    monkeypatch.setattr(it.bundled, "running_from_a_checkout", lambda: True)
    said = []
    spec, _licence, _build = it.pick_track(ask_fn=_answers("1", "y"), log=said.append)
    assert spec == "default"
    assert any("pack_track.py" in s for s in said)


def test_scoring_asks_for_one_prompt_set_rather_than_a_three_way_split():
    """`score --eval` takes a prompt set. The walk used to append `/bad_eval_ds` to whatever the
    corpus menu returned, which made a real path out of the bundled alias and nonsense out of
    every other answer: `walledai/AdvBench::train/bad_eval_ds` exists nowhere.
    """
    spec, _licence = it.pick_eval(ask_fn=_answers("1"), log=lambda *a: None)
    assert spec == "default/bad_eval_ds"
    # `pick_eval` puts the held-out partition first and then the side menu, so every row below it
    # is one further down than `by_side` has it.
    spec, _licence = it.pick_eval(
        ask_fn=_answers(str(_row_number("harmful", "harmful-behaviors") + 1)), log=lambda *a: None)
    assert spec == "mlabonne/harmful_behaviors::train"
    spec, _licence = it.pick_eval(
        ask_fn=_answers(str(_row_number("harmful", "advbench") + 1)), log=lambda *a: None)
    assert spec == "advbench", "a bundled corpus is passed by name, not as a Hub id"


# ── the plan ─────────────────────────────────────────────────────────────────────
def test_the_plan_collects_the_flags_that_matter():
    plan = it.plan_abliteration(
        ask_fn=_answers("1", "M", "1", "1", "OUT", "50"), log=lambda *a: None)
    assert plan["command"] == "kageyoshi"
    assert plan["options"]["--model"] == "M"
    assert plan["options"]["--track"] == "default"
    assert plan["options"]["--out"] == "OUT"
    assert plan["options"]["--device"] == "cuda"
    assert plan["options"]["--trials"] == "50"


# ── presenting ───────────────────────────────────────────────────────────────────
def test_present_prints_the_command_and_the_licence():
    said = []
    plan = {"command": "kageyoshi", "options": {"--model": "M"}, "licence": "CC BY-NC 4.0"}
    it.present(plan, ask_fn=_answers("y"), log=said.append)
    joined = "\n".join(said)
    assert "senbonzakura kageyoshi --model M" in joined
    assert "CC BY-NC 4.0" in joined


def test_present_does_not_lecture_about_your_own_corpus():
    said = []
    plan = {"command": "kageyoshi", "options": {"--model": "M"}, "licence": "yours"}
    it.present(plan, ask_fn=_answers("y"), log=said.append)
    assert "Attribution is required" not in "\n".join(said)


def test_declining_returns_none_and_says_the_command_still_works():
    said = []
    plan = {"command": "kageyoshi", "options": {"--model": "M"}, "licence": "yours"}
    assert it.present(plan, ask_fn=_answers("n"), log=said.append) is None
    assert any("Nothing was run" in s for s in said)


# ── the entry point ──────────────────────────────────────────────────────────────
def test_a_pipe_is_refused_rather_than_hung():
    said = []
    assert it.run(stdin=_NotTty(), log=said.append) == 2
    joined = "\n".join(said)
    assert "needs a terminal" in joined
    assert "--help" in joined


def test_abandoning_exits_cleanly_without_running_anything():
    def boom(_p):
        raise KeyboardInterrupt

    said = []
    assert it.run(ask_fn=boom, log=said.append, stdin=_Tty()) == 130
    assert any("Nothing was changed" in s for s in said)


def test_declining_the_run_exits_zero():
    said = []
    code = it.run(ask_fn=_answers("1", "M", "3", "mytrack", "2", "OUT", "3", "n"),
                  log=said.append, stdin=_Tty())
    assert code == 0


def test_accepting_calls_the_cli_with_the_displayed_flags(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", fake_main)
    said = []
    # The last answer is the finished screen's "What next?", which a successful run now reaches.
    # Empty, because its highlighted default is "Nothing, I am done": see `interactive.finished`.
    code = it.run(ask_fn=_answers("1", "MODEL", "3", "mytrack", "2", "OUT", "7", "y", ""),
                  log=said.append, stdin=_Tty())
    assert code == 0
    argv = seen["argv"]
    assert argv[0] == "kageyoshi"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "MODEL"
    assert argv[argv.index("--track") + 1] == "mytrack"
    assert argv[argv.index("--device") + 1] == "cpu"
    assert argv[argv.index("--trials") + 1] == "7"
    # And every one of those appeared in what the user was shown.
    joined = "\n".join(said)
    assert "--model MODEL" in joined and "--trials 7" in joined


def test_interactive_is_a_registered_subcommand():
    from senbonzakura import cli
    assert "interactive" in cli.DELEGATED
    assert cli._delegate("interactive") is it.run


# ── the way back in ──────────────────────────────────────────────────────────────
def _paused_run(tmp_path, name="run1", record=None):
    d = tmp_path / name
    d.mkdir()
    (d / it.STUDY_DB).touch()
    if record is not None:
        from senbonzakura import runrecord
        runrecord.write(d, **record)
    return d


def test_the_resume_command_carries_the_model_and_the_track(tmp_path):
    """THE SCREEN THAT PRINTED A COMMAND NOBODY COULD RUN.

    It offered `senbonzakura kageyoshi --out <dir> --resume`. `--model` is required, so the one
    screen written to save somebody hours produced an argparse error instead. Adding `--model`
    alone would have been worse: `--track` has a default, so a resume that omits it carries on one
    corpus's trials while scoring new ones against another.
    """
    _paused_run(tmp_path, record={"model": "Qwen/Qwen3-1.7B", "track": "mytrack"})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["options"]["--model"] == "Qwen/Qwen3-1.7B"
    assert plan["options"]["--track"] == "mytrack"
    assert plan["options"]["--resume"] is True
    assert plan["options"]["--out"] == str(tmp_path / "run1")


def test_a_run_from_before_the_record_asks_rather_than_guessing(tmp_path):
    _paused_run(tmp_path)
    said = []
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1", "MODEL", "TRACK"),
                           log=said.append)
    assert plan["options"]["--model"] == "MODEL"
    assert plan["options"]["--track"] == "TRACK"
    assert any("does not say which model" in s for s in said)


def test_the_menu_row_says_which_model_the_paused_run_was_editing(tmp_path):
    _paused_run(tmp_path, record={"model": "Qwen/Qwen3-1.7B", "track": "default"})
    said = []
    it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=said.append)
    assert any("Qwen/Qwen3-1.7B" in s for s in said)


def test_the_resume_command_uses_the_mode_word_the_run_used(tmp_path):
    _paused_run(tmp_path, record={"model": "M", "track": "T", "bankai": False})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["command"] == "abliterate"


def test_starting_fresh_is_still_the_last_option(tmp_path):
    _paused_run(tmp_path, record={"model": "M", "track": "T"})
    assert it.offer_resume(root=tmp_path, ask_fn=_answers("2"), log=lambda *a: None) is None


# ── several commands, in order ───────────────────────────────────────────────────
def test_a_track_build_runs_before_the_abliteration_that_needs_it():
    plan = it.plan_abliteration(
        ask_fn=_answers("1", "M", "2", "1", "1", "mytrack", "2", "OUT", "5"),
        log=lambda *a: None)
    ordered = it.steps(plan)
    assert [s["command"] for s in ordered] == ["track", "kageyoshi"]
    assert ordered[1]["options"]["--track"] == "mytrack"


def test_every_step_is_shown_before_any_of_them_runs():
    plan = it.plan_abliteration(
        ask_fn=_answers("2", "M", "2", "1", "1", "mytrack", "2", "OUT", "5"),
        log=lambda *a: None)
    said = []
    it.present(plan, ask_fn=_answers("n"), log=said.append)
    joined = "\n".join(said)
    for step in it.steps(plan):
        assert it.render_command(step["command"], step["options"]) in joined
    assert "three commands" in joined


def test_a_failing_step_stops_the_ones_after_it(monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        return 1 if argv[0] == "track" else 0

    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", fake_main)
    said = []
    code = it.run(ask_fn=_answers("1", "M", "2", "1", "1", "mytrack", "2", "OUT", "5", "y"),
                  log=said.append, stdin=_Tty())
    assert code == 1
    assert calls == ["track"]


def test_a_failure_before_the_search_does_not_promise_completed_trials(monkeypatch):
    """Telling somebody whose track build failed that their trials are safe on disk would be a
    comforting sentence about a search that never started.
    """
    def fake_main(argv):
        return 1 if argv[0] == "track" else 0

    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", fake_main)
    said = []
    it.run(ask_fn=_answers("1", "M", "2", "1", "1", "mytrack", "2", "OUT", "5", "y"),
           log=said.append, stdin=_Tty())
    joined = "\n".join(said)
    assert "no partial run to recover" in joined
    assert "completed trials are not lost" not in joined


def test_a_nonzero_status_still_gets_a_recovery_block(monkeypatch, tmp_path):
    """The recovery line used to be reserved for exceptions, so a command that reported failure
    by returning a status printed its error and then nothing.

    THIS USED TO ASSERT "To pick up where it stopped" UNCONDITIONALLY, and that was the defect:
    the resume advice was printed whether or not anything resumable existed. A device pre-flight
    refusal writes nothing, and the user was still told their completed trials were safe and given
    a `--resume` command that would refuse identically. What the block must do is say something
    true about the disk, which is what is asserted now.
    """
    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", lambda argv: 1)
    monkeypatch.chdir(tmp_path)
    said = []
    code = it.run(ask_fn=_answers("1", "M", "1", "2", "OUT", "5", "y"),
                  log=said.append, stdin=_Tty())
    assert code == 1
    joined = "\n".join(said)
    assert "the run stopped" in joined, "a non-zero status produced no recovery block at all"
    assert "Nothing recoverable was written" in joined, (
        "nothing was written to OUT, so the block must say so rather than promising a resume")
    assert "To pick up where it stopped" not in joined, (
        "resume advice was offered for a directory holding nothing to resume")


def test_the_resume_advice_appears_once_there_is_something_to_resume(monkeypatch, tmp_path):
    """The other half: with a study on disk, the block must offer the resume it used to fake."""
    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", lambda argv: 1)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OUT").mkdir()
    (tmp_path / "OUT" / "senbon-study.db").write_text("x", encoding="utf-8")
    said = []
    code = it.run(ask_fn=_answers("1", "M", "1", "2", "OUT", "5", "y"),
                  log=said.append, stdin=_Tty())
    assert code == 1
    joined = "\n".join(said)
    assert "To pick up where it stopped" in joined, (
        "a persisted study is on disk and the block did not offer to resume it")
    assert "--resume" in joined


def test_a_refusal_does_not_claim_there_is_a_traceback(monkeypatch, tmp_path):
    """A SystemExit refusal prints no traceback, and the block used to point at one anyway."""
    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", lambda argv: 1)
    monkeypatch.chdir(tmp_path)
    said = []
    it.run(ask_fn=_answers("1", "M", "1", "2", "OUT", "5", "y"), log=said.append, stdin=_Tty())
    joined = "\n".join(said)
    assert "traceback above" not in joined, (
        "the block told the reader to look at a traceback that was never printed, which is what "
        "made a clean refusal read as a crash")
    assert "refusal, not a crash" in joined


def test_the_whole_walk_completes_on_an_install_with_no_deep_learning_stack():
    """It asked four questions and then died on an eleven-frame ImportError.

    `ask_output` reached through `cli` for `occupied_by`, and `cli` imports torch, optuna and
    transformers at module scope. On a base install (the dependency split put those behind the
    `[abliterate]` extra) the guided mode got as far as "Where should it run?" and then handed the
    person a traceback that named none of the things they needed to install.

    Run in a subprocess with those imports blocked, because the check is about what an import
    pulls in and this process has already imported everything.
    """
    import subprocess
    import sys
    probe = r"""
import sys
class Block:
    def find_spec(self, name, target=None, path=None):
        if name.split(".")[0] in ("torch", "optuna", "transformers", "accelerate"):
            raise ImportError("blocked for this probe: " + name)
        return None
sys.meta_path.insert(0, Block())
from senbonzakura import interactive as it
# A machine with nothing on it, for the reason the autouse fixture at the top of this file
# gives: otherwise this subprocess asks a different question depending on what happens to be
# in the Hub cache of whatever box is running the suite.
it.models_on_this_machine = lambda root=".": []
# AND A BUNDLED TRACK, for the same reason: without one the track menu asks an extra question
# and every answer after it lines up against the wrong prompt. A release wheel carries the
# packed track and a source checkout does not, so this differs between a dev box and CI.
it.bundled.is_available = lambda: True
answers = iter(["1", "M", "1", "2", "OUT", "5"])
plan = it.plan_abliteration(ask_fn=lambda _p: next(answers), log=lambda *a: None)
print(plan["options"]["--out"])
"""
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=False, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OUT"


def test_enter_at_the_final_prompt_does_not_start_a_run():
    """THE ONE PROMPT IN THE WALK THAT MUST NOT DEFAULT TO YES.

    Every question before it is safe to press Enter through, and each prints "press Enter to take
    the default" underneath, which trains the reflex. The last question spends GPU hours, and on a
    rented card, money. A peer session found this by sending bare newlines from a script whose
    own comments said it would not confirm the run, and starting an abliteration on a real card.

    Pinned because flipping the default back broke no test at all when it was changed.
    """
    asked = []

    def _ask(prompt):
        asked.append(prompt)
        return ""          # the bare Enter that used to mean yes

    assert it.confirm("Run it?", default=False, ask_fn=_ask, log=lambda *a: None) is False
    assert "[y/N]" in asked[0], f"the prompt should show the safe default: {asked[0]}"


def test_present_refuses_the_run_on_a_bare_enter(monkeypatch):
    """The same property through the real call site rather than through `confirm` directly."""
    plan = {"command": "kageyoshi", "options": {"--model": "m", "--out": "o"},
            "licence": None, "recipe": None}
    line = it.present(plan, ask_fn=lambda _p: "", log=lambda *a: None)
    assert line is None, "a bare Enter at 'Run it?' must not return a command to run"


class TestTheDeviceMenuKnowsThisMachine:
    """The guided mode does not propose a device this machine cannot use.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-16. `DEVICES` was a static list with cuda first and
    no detection, so on a GPU-less machine a newcomer taking every default was handed
    `--device cuda`, which cannot run there. This is the one surface built so newcomers do not
    have to know things, and it steered them into the failure the device pre-flight now refuses.
    """

    def test_the_default_is_a_device_that_works(self, monkeypatch):
        monkeypatch.setattr(it, "device_available", lambda name: name == "cpu")
        assert it.pick_device(ask_fn=lambda _p: "", log=lambda *a: None) == "cpu"

    def test_cuda_is_still_offered_and_marked(self, monkeypatch):
        """Hiding it would teach a wrong model of the tool: somebody may be composing a command
        to run on another machine.
        """
        shown = []
        monkeypatch.setattr(it, "device_available", lambda name: name == "cpu")
        it.pick_device(ask_fn=lambda _p: "", log=lambda *a: shown.append(" ".join(str(x) for x in a)))
        text = "\n".join(shown)
        assert "cuda" in text, "the option must still be listed"
        assert "not available on this machine" in text, "and marked as unusable here"

    def test_cuda_stays_the_default_when_it_works(self, monkeypatch):
        monkeypatch.setattr(it, "device_available", lambda _name: True)
        assert it.pick_device(ask_fn=lambda _p: "", log=lambda *a: None) == "cuda"

    def test_an_unaskable_device_is_not_reported_as_unavailable(self, monkeypatch):
        """On a base install torch may be absent. "cannot tell" must not read as "no"."""
        monkeypatch.setattr(it, "device_available", lambda _name: None)
        shown = []
        it.pick_device(ask_fn=lambda _p: "", log=lambda *a: shown.append(" ".join(str(x) for x in a)))
        assert "not available" not in "\n".join(shown)


# ── what a resume must not throw away (2026-09-28) ───────────────────────────────
#
# `run.json` records six things about a run and the resume screen read three, so the parser
# default won for the other three every time. Each of the tests below fails without the carry
# through, and each failure was reachable from the menu with no flags typed at all.

def _config_only_run(tmp_path, name="baked", record=None):
    d = tmp_path / name
    d.mkdir()
    (d / it.BAKEABLE).write_text("{}", encoding="utf-8")
    if record is not None:
        from senbonzakura import runrecord
        runrecord.write(d, **record)
    return d


def test_a_resume_carries_the_device_the_trials_and_the_search(tmp_path):
    """Three flags the record holds and the screen ignored.

    --device fell back to cuda, so a run resumed on a machine without a card died at the device
    pre-flight. --trials fell back to 60, so a study with 150 trials in it computed a remaining
    budget of zero, announced the budget spent and baked a model from a search the person asked to
    be 200. --search is pinned, so a scalar study resumed at the default pareto is refused over a
    flag this menu has never mentioned.
    """
    _paused_run(tmp_path, record={"model": "M", "track": "T", "device": "cpu",
                                  "trials": 200, "search": "scalar"})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["options"]["--device"] == "cpu"
    assert str(plan["options"]["--trials"]) == "200"
    assert plan["options"]["--search"] == "scalar"


def test_a_resume_says_what_it_took_off_the_record_before_the_confirm(tmp_path):
    """Carried silently, a recorded value is a flag on a line nobody reads twice."""
    _paused_run(tmp_path, record={"model": "M", "track": "T", "device": "cpu",
                                  "trials": 200, "search": "scalar"})
    said = []
    it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=said.append)
    text = "\n".join(said)
    assert "--device cpu" in text
    assert "--trials 200" in text
    assert "--search scalar" in text


def test_a_recorded_device_this_machine_cannot_use_is_said_out_loud(tmp_path, monkeypatch):
    """The defect the operator actually met: a run copied to a box with no card."""
    monkeypatch.setattr(it, "device_available", lambda name: name != "cuda")
    _paused_run(tmp_path, record={"model": "M", "track": "T", "device": "cuda"})
    said = []
    it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=said.append)
    assert "cannot use that device" in "\n".join(said)


def test_a_winning_config_with_no_study_re_bakes_instead_of_re_searching(tmp_path):
    """THE PROMISE THE FLAG COULD NOT KEEP.

    A directory holding only best-config.json is offered as "a winning config, so it re-bakes in
    minutes rather than re-searching", and the screen emitted --resume. With no senbon-study.db
    that finds no study, creates one, and starts a fresh search from trial zero under a log line
    saying "(resuming)": the whole search again, after a promise of minutes. --bake-config is the
    flag that does what the row says, and the guided mode had never emitted it.
    """
    d = _config_only_run(tmp_path, record={"model": "M", "track": "T", "trials": 200})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["options"]["--bake-config"] == str(d / it.BAKEABLE)
    assert "--resume" not in plan["options"]
    assert "--trials" not in plan["options"], "a direct bake runs no trials"


def test_a_study_on_disk_still_resumes_rather_than_re_baking(tmp_path):
    """The other half of the same decision, so the fix cannot swing too far."""
    _paused_run(tmp_path, record={"model": "M", "track": "T"})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["options"]["--resume"] is True
    assert "--bake-config" not in plan["options"]


def test_a_resume_does_not_claim_nothing_is_overwritten(tmp_path):
    """It said "Nothing there is overwritten", and a resume rewrites run.json, best-config.json
    and the saved weights in that directory. What survives is the completed trials.
    """
    _paused_run(tmp_path, record={"model": "M", "track": "T"})
    said = []
    it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=said.append)
    text = "\n".join(said)
    assert "Nothing there is overwritten" not in text
    assert "written over" in text


def test_resuming_a_bundled_track_run_still_shows_the_corpus_licence(tmp_path):
    """The resume path hard-coded "yours", so the CC BY-NC notice the fresh walk prints for the
    bundled corpus vanished for the same corpus reached the other way.
    """
    _paused_run(tmp_path, record={"model": "M", "track": "default"})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["licence"] == "CC BY-NC 4.0"
    said = []
    it.present(plan, ask_fn=_answers("n"), log=said.append)
    assert "CC BY-NC 4.0" in "\n".join(said)


def test_a_resume_of_a_track_of_your_own_claims_no_licence(tmp_path):
    _paused_run(tmp_path, name="two", record={"model": "M", "track": "/srv/mytrack"})
    plan = it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=lambda *a: None)
    assert plan["licence"] == "yours"


def test_a_resume_says_it_converts_nothing_and_names_the_second_command(tmp_path):
    """run.json records the abliteration's inputs and nothing about the recipe around it, so a
    person who chose "Build a local brain" and crashed gets the model re-baked and no GGUF. It
    cannot be added silently and there is no recorded answer to read, so the command is named.
    """
    _paused_run(tmp_path, record={"model": "M", "track": "T"})
    said = []
    it.offer_resume(root=tmp_path, ask_fn=_answers("1"), log=said.append)
    text = "\n".join(said)
    assert "converts nothing" in text
    assert "senbonzakura convert" in text


# ── the last question cannot invalidate the earlier ones ─────────────────────────

def _occupied(tmp_path, name="prev", **record):
    d = tmp_path / name
    d.mkdir()
    (d / "abliteration.json").write_text("{}", encoding="utf-8")
    from senbonzakura import runrecord
    runrecord.write(d, **record)
    return d


def test_continuing_a_run_shows_the_answers_it_contradicts(tmp_path):
    """"Continue that run" is the LAST question and it adds --resume, which pins the model, the
    track and the search against the run.json already in that directory. So four screens of
    answers could be contradicted by the last one, and the person found out from a refusal at the
    end, after a track may already have been rebuilt on disk.
    """
    d = _occupied(tmp_path, model="RECORDED", track="RTRACK", search="scalar")
    said = []
    out, resume = it.ask_output(ask_fn=_answers(str(d), "2", "1"), log=said.append,
                                chosen={"model": "CHOSEN", "track": "CTRACK"})
    assert (out, resume) == (str(d), True)
    text = "\n".join(said)
    assert "'RECORDED'" in text and "'CHOSEN'" in text
    assert "'RTRACK'" in text and "'CTRACK'" in text


def test_keeping_your_answers_sends_you_back_for_another_directory(tmp_path):
    d = _occupied(tmp_path, model="RECORDED", track="T")
    free = tmp_path / "fresh"
    out, resume = it.ask_output(ask_fn=_answers(str(d), "2", "2", str(free)),
                                log=lambda *a: None, chosen={"model": "CHOSEN", "track": "T"})
    assert (out, resume) == (str(free), False)


def test_a_resume_from_the_output_question_reaches_the_printed_command(tmp_path):
    """The reconciliation is worth nothing if it stops at the screen. The recorded answers have to
    end up on the line that runs, including --search, which this walk never asks about at all.
    """
    d = _occupied(tmp_path, model="RECORDED", track="RTRACK", search="scalar")
    plan = it.plan_abliteration(
        ask_fn=_answers("1", "CHOSEN", "1", "2", str(d), "2", "1", "7"), log=lambda *a: None)
    assert plan["options"]["--model"] == "RECORDED"
    assert plan["options"]["--track"] == "RTRACK"
    assert plan["options"]["--search"] == "scalar"
    assert plan["options"]["--resume"] is True


def test_a_track_build_is_dropped_when_the_study_pins_another_corpus(tmp_path):
    """Otherwise the walk builds a corpus on disk that the run it is building it for cannot use."""
    d = _occupied(tmp_path, model="M", track="RTRACK")
    said = []
    plan = it.plan_abliteration(
        ask_fn=_answers("1", "M", "2", "1", "1", "newtrack", "2", str(d), "2", "1", "7"),
        log=said.append)
    assert "first" not in plan, "the track build would produce a corpus this run cannot use"
    assert plan["options"]["--track"] == "RTRACK"
    assert "track build is dropped" in "\n".join(said)


# ── the recipe that could not finish ─────────────────────────────────────────────

def test_a_brain_checks_it_can_convert_before_anything_downloads(monkeypatch):
    """The convert step needs build-time artefacts a source checkout does not carry, and nothing
    looked for them until the abliteration had already finished. Every bit of that verdict is
    knowable before the model downloads, which is why the rest of the tool pre-flights at all.
    """
    monkeypatch.setattr(it, "missing_conversion_tools", lambda: ["llama-quantize"])
    said = []
    plan = it.plan_abliteration(ask_fn=_answers("2", "1", "M", "1", "2", "OUT", "5"),
                                log=said.append)
    assert "then" not in plan, "a step this install cannot run must not be in the plan"
    text = "\n".join(said)
    assert "cannot finish that recipe" in text
    assert "vendor_llama.py" in text, "a refusal without the remedy is just bad news"


def test_stopping_at_that_point_starts_nothing(monkeypatch):
    monkeypatch.setattr(it, "missing_conversion_tools", lambda: ["llama-quantize"])
    with pytest.raises(it.AbandonedError):
        it.plan_abliteration(ask_fn=_answers("2", "2"), log=lambda *a: None)


def test_a_brain_still_gets_its_convert_step_where_the_tools_exist(monkeypatch):
    monkeypatch.setattr(it, "missing_conversion_tools", list)
    plan = it.plan_abliteration(ask_fn=_answers("2", "M", "1", "2", "OUT", "5"),
                                log=lambda *a: None)
    assert plan["then"]["command"] == "convert"


def test_the_conversion_check_names_both_halves(monkeypatch):
    from senbonzakura import vendored

    def _refuse(*_a, **_k):
        raise vendored.VendorError("not here")

    monkeypatch.setattr(vendored, "find_script", _refuse)
    monkeypatch.setattr(vendored, "find_binary", _refuse)
    missing = it.missing_conversion_tools()
    assert len(missing) == 2
    assert any("llama-quantize" in m for m in missing)


# ── which step died decides the advice ───────────────────────────────────────────

def test_a_conversion_failure_does_not_write_off_the_saved_model():
    """The test was "this is not the main command", which is true of the track build AND the
    conversion, and those sit on opposite sides of the expensive part. The conversion runs after
    the abliteration has saved, so "there is no partial run to recover" wrote off a model that is
    on the person's disk and invited them to run the whole search again.
    """
    plan = {"command": "kageyoshi",
            "options": {"--model": "M", "--out": "brain"},
            "then": {"command": "convert", "options": {"brain": True, "--quantise": "Q4_K_M"}},
            "licence": None}
    said = []
    it.log_failure(plan, "boom", step=plan["then"], log=said.append)
    text = "\n".join(said)
    assert "no partial run to recover" not in text
    assert "the model was saved" in text
    assert "senbonzakura convert brain --quantise Q4_K_M" in text


def test_a_track_build_failure_still_says_there_is_nothing_to_recover():
    plan = {"command": "kageyoshi",
            "options": {"--model": "M", "--out": "OUT"},
            "first": {"command": "track", "options": {"--out": "T"}},
            "licence": None}
    said = []
    it.log_failure(plan, "boom", step=plan["first"], log=said.append)
    assert "no partial run to recover" in "\n".join(said)


# ── the questions that were not checked ──────────────────────────────────────────

def test_a_trial_budget_is_checked_at_the_prompt_rather_than_after_the_confirm():
    """It was free text handed straight to the plan, so "abc" and "0" were printed into the
    command, confirmed by the person, and only then refused by --trials.
    """
    answers = iter(["abc", "0", "-4", "12"])
    said = []
    assert it.ask_trials(ask_fn=lambda _p: next(answers), log=said.append) == "12"
    assert any("not a trial budget" in s for s in said)


def test_the_trials_prompt_does_not_invent_a_convention():
    """"200 is the usual" matched nothing: the flat default is 60 and the preset picks 100, 80 or
    64 by model size. A number invented for a prompt and described as the convention is how a
    reader ends up believing the tool has one it has never had.
    """
    asked = []

    def _ask(prompt):
        asked.append(prompt)
        return "60"

    it.ask_trials(ask_fn=_ask, log=lambda *a: None)
    assert "200 is the usual" not in asked[0]
    assert "60" in asked[0], "the number the tool actually defaults to belongs in the sentence"


def test_the_device_default_is_cpu_when_torch_cannot_be_asked(monkeypatch):
    """`device_available` answers None with torch absent, None is falsy, so nothing ever set the
    first usable index and the default fell back to entry zero, which is cuda, unmarked. A base
    install is exactly where torch is missing, so the screen written to stop a newcomer being
    handed `--device cuda` handed it to them by default.
    """
    monkeypatch.setattr(it, "device_available", lambda _name: None)
    said = []
    assert it.pick_device(ask_fn=lambda _p: "", log=said.append) == "cpu"
    assert any("PyTorch is not installed" in s for s in said)


def test_cuda_is_still_choosable_when_torch_cannot_be_asked(monkeypatch):
    """Somebody may be composing a command for a machine that does have a card."""
    monkeypatch.setattr(it, "device_available", lambda _name: None)
    assert it.pick_device(ask_fn=lambda _p: "1", log=lambda *a: None) == "cuda"


# ── the two dead ends ────────────────────────────────────────────────────────────

def test_an_installed_copy_is_not_told_to_run_a_tool_it_does_not_have(monkeypatch):
    """`pack_track.py` ships in no wheel. Naming it to somebody on an installed copy sends them
    looking for a file that was never shipped, and their actual problem is a packaging fault.
    """
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    monkeypatch.setattr(it.bundled, "running_from_a_checkout", lambda: False)
    said = []
    assert it.warn_if_unbundled(log=said.append) is True
    text = "\n".join(said)
    assert "pack_track.py" not in text
    assert "packaging fault" in text


def test_the_track_menu_offers_the_way_out_its_note_names(monkeypatch):
    """`pick_track` returned "default" the moment the note was printed, so the only way out of
    that screen was Ctrl+C at the final confirm.
    """
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    monkeypatch.setattr(it.bundled, "running_from_a_checkout", lambda: True)
    spec, _licence, build = it.pick_track(ask_fn=_answers("1", "", "3", "~/mytrack"),
                                          log=lambda *a: None)
    assert spec == "~/mytrack"
    assert build is None


def test_the_scoring_menu_offers_the_same_way_out(monkeypatch):
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    monkeypatch.setattr(it.bundled, "running_from_a_checkout", lambda: True)
    spec, _licence = it.pick_eval(
        ask_fn=_answers("1", "", str(_row_number("harmful", "harmful-behaviors") + 1)),
        log=lambda *a: None)
    assert spec == "mlabonne/harmful_behaviors::train"


def test_asking_for_the_flag_list_gets_every_flag_and_no_dead_end():
    """Two defects in one menu row. It promised "the flag list" and ran `senbonzakura --help`,
    which suppresses every non-core flag and never names `--help-all`. And its plan went to the
    same [y/N] confirm as an abliteration, so the bare Enter that is safe everywhere else in the
    walk answered "Nothing was run." to the one entry that is free to execute.
    """
    plan = it.plan_abliteration(ask_fn=_answers("4"), log=lambda *a: None)
    assert plan["command"] == "--help-all"

    asked = []

    def _enter(prompt):
        asked.append(prompt)
        return ""

    line = it.present(plan, ask_fn=_enter, log=lambda *a: None)
    assert line == "senbonzakura --help-all"
    assert "[Y/n]" in asked[-1], f"the free entry should not need a keystroke: {asked[-1]}"


def test_a_run_that_costs_gpu_hours_still_defaults_to_no():
    """The exception for the help page must not reach the plans that spend money."""
    plan = it.plan_abliteration(ask_fn=_answers("1", "M", "1", "2", "OUT", "5"),
                                log=lambda *a: None)
    asked = []
    assert it.present(plan, ask_fn=lambda p: (asked.append(p), "")[1],
                      log=lambda *a: None) is None
    assert "[y/N]" in asked[-1]
