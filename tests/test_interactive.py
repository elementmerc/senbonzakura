# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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
    assert by_key["advbench"] == "harmful"
    assert by_key["harmless-alpaca"] == "harmless"
    assert by_key["default"] == "track"


def test_a_side_menu_shows_the_licence_before_the_choice_is_made():
    said = []
    it.pick_side("harmful", "which?", ask_fn=_answers("1"), log=said.append)
    assert "MIT" in "\n".join(said)
    assert "undeclared upstream" in "\n".join(said)


def test_picking_a_known_corpus_returns_its_spec():
    spec, licence = it.pick_side("harmful", "which?", ask_fn=_answers("2"),
                                 log=lambda *a: None)
    assert spec == "walledai/AdvBench::train"
    assert licence == "MIT"


def test_a_gated_corpus_is_not_the_default_and_says_it_is_gated():
    """AdvBench was the first harmful choice, so it was what pressing Enter picked, and it is
    gated: a newcomer with no Hugging Face account met an authentication error from a menu that
    had just called it the field's common yardstick.
    """
    harmful = it.by_side("harmful")
    assert harmful[0]["key"] != "advbench"
    advbench = next(e for e in harmful if e["key"] == "advbench")
    assert "GATED" in advbench["note"]
    assert "huggingface-cli login" in advbench["note"]


def test_picking_your_own_asks_for_the_path():
    """"Something of my own" is always last, whichever side is being asked for."""
    n = len(it.by_side("harmless"))
    spec, licence = it.pick_side("harmless", "which?", ask_fn=_answers(str(n), "~/mine.txt"),
                                 log=lambda *a: None)
    assert spec == "~/mine.txt"
    assert licence == "yours"


def test_two_hub_corpora_become_a_track_build_step_rather_than_a_broken_track_flag():
    spec, licence, build = it.pick_track(
        ask_fn=_answers("2", "1", "1", "mytrack"), log=lambda *a: None)
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
    monkeypatch.setattr(it.bundled, "is_available", lambda: False)
    said = []
    it.pick_track(ask_fn=_answers("1"), log=said.append)
    assert any("pack_track.py" in s for s in said)


def test_scoring_asks_for_one_prompt_set_rather_than_a_three_way_split():
    """`score --eval` takes a prompt set. The walk used to append `/bad_eval_ds` to whatever the
    corpus menu returned, which made a real path out of the bundled alias and nonsense out of
    every other answer: `walledai/AdvBench::train/bad_eval_ds` exists nowhere.
    """
    spec, _licence = it.pick_eval(ask_fn=_answers("1"), log=lambda *a: None)
    assert spec == "default/bad_eval_ds"
    spec, _licence = it.pick_eval(ask_fn=_answers("2"), log=lambda *a: None)
    assert spec == "mlabonne/harmful_behaviors::train"


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
    code = it.run(ask_fn=_answers("1", "MODEL", "3", "mytrack", "2", "OUT", "7", "y"),
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


def test_a_nonzero_status_gets_the_same_recovery_line_as_a_crash(monkeypatch):
    """The recovery line used to be reserved for exceptions, so a command that reported failure
    by returning a status printed its error and then nothing.
    """
    from senbonzakura import cli
    monkeypatch.setattr(cli, "main", lambda argv: 1)
    said = []
    code = it.run(ask_fn=_answers("1", "M", "1", "2", "OUT", "5", "y"),
                  log=said.append, stdin=_Tty())
    assert code == 1
    assert any("To pick up where it stopped" in s for s in said)


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
answers = iter(["1", "M", "1", "2", "OUT", "5"])
plan = it.plan_abliteration(ask_fn=lambda _p: next(answers), log=lambda *a: None)
print(plan["options"]["--out"])
"""
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=False, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OUT"
