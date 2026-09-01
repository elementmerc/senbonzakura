# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the guided mode (senbonzakura/interactive.py).

The load-bearing property is that the printed command is exactly the run that happens. A guided
mode that displayed one thing and passed another would be an excellent way to hide a mistake, so
that equivalence is tested directly rather than assumed from the code reading correctly.
"""
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


# ── quoting ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["Qwen/Qwen3-1.7B", "out", "default", "train[:400]", "a=b"])
def test_ordinary_values_are_not_quoted(value):
    assert it.quote(value) == value


def test_a_value_with_a_space_is_quoted():
    assert it.quote("my track") == "'my track'"


def test_a_value_with_a_quote_is_escaped():
    assert it.quote("it's") == "'it'\\''s'"


def test_an_empty_value_is_quoted():
    assert it.quote("") == "''"


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
        ask_fn=_answers("Qwen/Qwen3-0.6B", "1", "2", "out dir", "5"), log=lambda *a: None)
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


def test_the_menu_shows_the_licence_before_the_choice_is_made():
    said = []
    it.pick_dataset(ask_fn=_answers("2"), log=said.append)
    joined = "\n".join(said)
    assert "MIT" in joined
    assert "CC BY-NC 4.0" in joined


def test_picking_a_known_dataset_returns_its_spec():
    spec, licence = it.pick_dataset(ask_fn=_answers("2"), log=lambda *a: None)
    assert spec == "walledai/AdvBench::train"
    assert licence == "MIT"


def test_picking_your_own_asks_for_the_path():
    spec, licence = it.pick_dataset(ask_fn=_answers("5", "~/mytrack"), log=lambda *a: None)
    assert spec == "~/mytrack"
    assert licence == "yours"


def test_choosing_the_bundled_track_without_one_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    said = []
    it.pick_dataset(ask_fn=_answers("1"), log=said.append)
    assert any("pack_track.py" in s for s in said)


# ── the plan ─────────────────────────────────────────────────────────────────────
def test_the_plan_collects_the_flags_that_matter():
    plan = it.plan_abliteration(
        ask_fn=_answers("M", "1", "1", "OUT", "50"), log=lambda *a: None)
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
    code = it.run(ask_fn=_answers("M", "5", "mytrack", "2", "OUT", "3", "n"),
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
    code = it.run(ask_fn=_answers("MODEL", "5", "mytrack", "2", "OUT", "7", "y"),
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
