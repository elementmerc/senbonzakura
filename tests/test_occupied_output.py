# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A run must not write into a directory another run already used.

WHY THIS FILE EXISTS

On 2026-08-12 three separate defects were traced to one mechanism: a previous run's output sitting
in the directory while something reported the work already done. The failure is quiet by
construction. A re-run replaces some files and leaves others, so the artefact ends up describing
two runs with no field that says so, and nothing in the output looks wrong.

The guided-mode critique of 2026-08-21 raised it again as finding 5, ranked second of thirteen by
what a person loses, and noted the tool had learned the lesson three times and still walked a
newcomer into it. Checking the code to build that fix found the wider version: the NON-interactive
path had no occupancy check either, so this was never only a guided-mode gap.

TWO DIFFERENT ANSWERS ON PURPOSE

The command line refuses, because a warning printed above an hour of GPU work is a warning nobody
reads. The guided mode offers the ways out instead, because refusing somebody who is being walked
through the tool is a dead end.
"""
import types

import pytest

from senbonzakura import cli, interactive


def _previous_run(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for name in names or ("abliteration.json",):
        (d / name).write_text("{}", encoding="utf-8")
    return d


# ── what counts as occupied ──────────────────────────────────────────────────────────

def test_an_empty_directory_is_free(tmp_path):
    assert cli.occupied_by(tmp_path) == []


def test_a_directory_that_does_not_exist_is_free(tmp_path):
    assert cli.occupied_by(tmp_path / "nope") == []


def test_unrelated_files_do_not_count_as_a_previous_run(tmp_path):
    """Only this tool's own artefacts mean "a run used this". A person's notes do not.

    Being strict here would make the check fire on a directory somebody deliberately prepared,
    which is how a safety check earns itself an override flag and then stops being read.
    """
    (tmp_path / "notes.md").write_text("mine", encoding="utf-8")
    (tmp_path / "README").write_text("mine", encoding="utf-8")
    assert cli.occupied_by(tmp_path) == []


@pytest.mark.parametrize("name", ["abliteration.json", "best-config.json", "trials.json",
                                  "config.json", "model.safetensors"])
def test_each_run_artefact_is_recognised(tmp_path, name):
    _previous_run(tmp_path, name)
    assert cli.occupied_by(tmp_path) == [name]


def test_sharded_weights_are_recognised_without_knowing_the_shard_count(tmp_path):
    """The filenames carry the shard count, so the exact names are not knowable in advance."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "model-00003-of-00007.safetensors").write_text("", encoding="utf-8")
    assert cli.occupied_by(tmp_path) == ["model-*.safetensors"]


def test_the_report_is_the_files_found_not_a_boolean(tmp_path):
    """A message a person can act on names what they are about to lose."""
    _previous_run(tmp_path, "abliteration.json", "trials.json")
    assert cli.occupied_by(tmp_path) == ["abliteration.json", "trials.json"]


# ── the command line refuses ─────────────────────────────────────────────────────────

def _args(out, **over):
    a = dict(out=str(out), resume=False, bake_config=None)
    a.update(over)
    return types.SimpleNamespace(**a)


def test_an_occupied_output_is_refused(tmp_path):
    _previous_run(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli._preflight_output(_args(tmp_path))
    msg = str(e.value)
    assert "already holds a previous run" in msg
    assert "abliteration.json" in msg, "the message must name what is there"


def test_the_refusal_offers_every_way_out(tmp_path):
    """A refusal that does not say what to do instead is an obstacle rather than a guard."""
    _previous_run(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli._preflight_output(_args(tmp_path))
    msg = str(e.value)
    assert "--out" in msg and "--resume" in msg and "delete" in msg


def test_resume_makes_an_occupied_directory_expected(tmp_path):
    """--resume is the one flag whose whole purpose is to continue what is already there."""
    _previous_run(tmp_path)
    cli._preflight_output(_args(tmp_path, resume=True))


def test_bake_config_passes_because_it_reads_what_the_previous_run_left(tmp_path):
    _previous_run(tmp_path, "best-config.json")
    cli._preflight_output(_args(tmp_path, bake_config="best-config.json"))


def test_a_free_directory_passes(tmp_path):
    cli._preflight_output(_args(tmp_path / "fresh"))


def test_the_check_runs_before_the_model_is_downloaded(tmp_path, monkeypatch):
    """WHERE it runs is the point. Save time is after the search, so finding it there costs the run.

    Makes the model constructor explode: if the check moved below it, this would see that
    explosion instead of the refusal.
    """
    _previous_run(tmp_path)

    def _never(*_a, **_k):
        raise AssertionError("the model was constructed before --out was checked")

    monkeypatch.setattr(cli, "Abliterator", _never)
    monkeypatch.setattr(cli, "_preflight_datasets", lambda _a: None)
    args = _args(tmp_path)
    args.load_in_4bit = False
    with pytest.raises(SystemExit, match="already holds a previous run"):
        cli.run_parsed(args, None, [])


# ── the guided mode offers a choice instead ──────────────────────────────────────────

def test_the_guided_mode_offers_the_ways_out_rather_than_refusing(tmp_path):
    """Finding 5's actual requirement: a choice, not a warning line.

    A warning is what somebody scrolls past on the way to the next question.
    """
    _previous_run(tmp_path)
    answers = iter([str(tmp_path), "2"])
    lines = []
    out, resume = interactive.ask_output(ask_fn=lambda _p: next(answers), log=lines.append)
    assert out == str(tmp_path)
    assert resume is True
    text = "\n".join(lines)
    assert "already holds a previous run" in text
    assert "abliteration.json" in text


def test_choosing_a_different_directory_loops_rather_than_giving_up(tmp_path):
    free = tmp_path / "fresh"
    _previous_run(tmp_path / "taken")
    answers = iter([str(tmp_path / "taken"), "1", str(free)])
    out, resume = interactive.ask_output(ask_fn=lambda _p: next(answers), log=lambda _m: None)
    assert out == str(free)
    assert resume is False


def test_the_guided_mode_never_offers_to_delete_somebody_elses_result(tmp_path):
    """Deleting a previous run on a person's behalf, inside a flow they are still learning, is not
    a choice this should offer. They can remove the directory themselves and come back.
    """
    _previous_run(tmp_path)
    answers = iter([str(tmp_path), "1", str(tmp_path / "fresh")])
    lines = []
    interactive.ask_output(ask_fn=lambda _p: next(answers), log=lines.append)
    text = "\n".join(lines).lower()
    assert "overwrite" not in text
    assert "delete" not in text


def test_the_resume_decision_reaches_the_printed_command():
    """The file's whole contract is that the command it prints is the command it runs, so a choice
    taken in the walkthrough has to appear on the line rather than in a hidden mode.

    The first version set the option to None, which `render_command` drops.
    """
    line = interactive.render_command("kageyoshi", {"--out": "x", "--resume": True})
    assert line == "senbonzakura kageyoshi --out x --resume"
    assert "--resume" not in interactive.render_command("kageyoshi", {"--out": "x",
                                                                     "--resume": False})


# ── the failure screen (critique finding 1) ──────────────────────────────────────────
# Ten drawn scenes and every one of them succeeds. The screen nobody drew is the one that loses
# hours: `cli.py` records that a traceback out of `_save_weights` has twice meant hours of card
# time producing nothing an operator could use.

def _plan(out="abliterated"):
    return {"command": "kageyoshi",
            "options": {"--model": "Qwen/Qwen3-1.7B", "--out": out, "--trials": "200"},
            "licence": "yours"}


def test_the_failure_screen_names_the_way_back_in():
    """THE POINT OF IT. Not a summary of the error, which the tool already printed better, but the
    sentence those messages do not carry: the search is on disk and one command resumes it.
    """
    lines = []
    interactive.log_failure(_plan(), "RuntimeError: CUDA out of memory", log=lines.append)
    text = "\n".join(lines)
    assert "RuntimeError: CUDA out of memory" in text
    assert "--resume" in text, "a failure screen without the recovery is just bad news"
    assert "abliterated" in text


def test_the_recovery_command_is_the_original_one_plus_resume():
    """A command a person can paste, not a description of one they should construct."""
    lines = []
    interactive.log_failure(_plan(), "boom", log=lines.append)
    line = next(ln.strip() for ln in lines if ln.strip().startswith("senbonzakura"))
    assert line == ("senbonzakura kageyoshi --model Qwen/Qwen3-1.7B --out abliterated "
                    "--trials 200 --resume")


def test_a_run_that_wrote_nothing_says_so_rather_than_offering_a_false_recovery():
    """Offering --resume when there is nothing to resume sends somebody to a second failure."""
    plan = _plan()
    plan["options"].pop("--out")
    lines = []
    interactive.log_failure(plan, "boom", log=lines.append)
    text = "\n".join(lines)
    assert "nothing to recover" in text
    assert "--resume" not in text


def test_the_guided_mode_adds_the_screen_and_still_lets_the_error_out(monkeypatch):
    """Not swallowed. The traceback is what a bug report needs and it still goes to stderr; this
    adds to it rather than replacing it.
    """
    lines = []
    monkeypatch.setattr(interactive, "is_tty", lambda _s=None: True)
    monkeypatch.setattr(interactive, "plan_abliteration", lambda **_k: _plan())
    monkeypatch.setattr(interactive, "present", lambda _p, **_k: "senbonzakura kageyoshi")

    def _boom(_argv):
        raise RuntimeError("the bake failed")

    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "main", _boom)
    with pytest.raises(RuntimeError, match="the bake failed"):
        interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None)
    text = "\n".join(lines)
    assert "the run stopped" in text
    assert "--resume" in text


def test_an_interrupt_gets_the_screen_too(monkeypatch):
    """Ctrl+C partway through a search is the most likely way a guided run ends early, and the
    trials completed before it are exactly what --resume exists to keep.
    """
    lines = []
    monkeypatch.setattr(interactive, "is_tty", lambda _s=None: True)
    monkeypatch.setattr(interactive, "plan_abliteration", lambda **_k: _plan())
    monkeypatch.setattr(interactive, "present", lambda _p, **_k: "senbonzakura kageyoshi")

    def _stop(_argv):
        raise KeyboardInterrupt

    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "main", _stop)
    assert interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None) == 130
    assert "--resume" in "\n".join(lines)


# ── the way back in (critique finding 3) ─────────────────────────────────────────────
# The guided mode could CREATE a paused run and could not FIND one: scene 9 writes the marker and
# says to resume with a flag, and scenes 1 and 2 never mention it, so the person who paused
# yesterday is sent back to the flag list. That is the single outcome the mode exists to avoid.

def test_a_directory_with_a_study_is_resumable(tmp_path):
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / interactive.STUDY_DB).write_text("", encoding="utf-8")
    found = interactive.resumable_runs(tmp_path)
    assert [p for p, _w in found] == [str(tmp_path / "brain")]
    assert "completed trials" in found[0][1]


def test_a_directory_with_only_a_winning_config_is_still_resumable(tmp_path):
    """The cheaper artefact and the more valuable one: it turns hours of re-searching into
    minutes of re-baking, and the design mentions it in none of its ten scenes.
    """
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / interactive.BAKEABLE).write_text("{}", encoding="utf-8")
    found = interactive.resumable_runs(tmp_path)
    assert "re-bakes in minutes" in found[0][1]


def test_a_directory_with_both_says_so(tmp_path):
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / interactive.STUDY_DB).write_text("", encoding="utf-8")
    (tmp_path / "brain" / interactive.BAKEABLE).write_text("{}", encoding="utf-8")
    assert "already picked" in interactive.resumable_runs(tmp_path)[0][1]


def test_an_ordinary_directory_is_not_offered(tmp_path):
    """A saved model with no study and no config cannot be carried on with, only overwritten."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "model.safetensors").write_text("", encoding="utf-8")
    assert interactive.resumable_runs(tmp_path) == []


def test_the_search_does_not_descend(tmp_path):
    """One level down on purpose: walking a home directory to fill a menu is slow and surprising,
    and would list runs from projects the person is not in.
    """
    deep = tmp_path / "a" / "b" / "brain"
    deep.mkdir(parents=True)
    (deep / interactive.STUDY_DB).write_text("", encoding="utf-8")
    assert interactive.resumable_runs(tmp_path) == []


def test_an_unreadable_root_is_not_a_crash(tmp_path):
    assert interactive.resumable_runs(tmp_path / "nope") == []


def test_nothing_is_offered_when_there_is_nothing_to_offer(tmp_path):
    """A menu row that is empty most of the time trains people to skip the first question, and
    the first question is the one that saves them hours.
    """
    lines = []
    assert interactive.offer_resume(tmp_path, ask_fn=lambda _p: "1", log=lines.append) is None
    assert lines == []


def test_choosing_a_found_run_produces_a_resume_command(tmp_path):
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / interactive.STUDY_DB).write_text("", encoding="utf-8")
    plan = interactive.offer_resume(tmp_path, ask_fn=lambda _p: "1", log=lambda _m: None)
    line = interactive.render_command(plan["command"], plan["options"])
    assert line == f"senbonzakura kageyoshi --out {tmp_path / 'brain'} --resume"


def test_starting_fresh_is_always_the_last_option(tmp_path):
    """Offering the found run must not trap somebody who wanted a new one."""
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / interactive.STUDY_DB).write_text("", encoding="utf-8")
    assert interactive.offer_resume(tmp_path, ask_fn=lambda _p: "2",
                                    log=lambda _m: None) is None


# ── every recipe has a path behind it (critique finding 2) ───────────────────────────
# Scene 2 offered five recipes and the document walked one. Two had nothing behind them at all,
# and a menu entry that leads nowhere is worse than no entry: the person has committed to the path
# before it stops making sense. "Measure a model I already have" was the sharp case, because the
# screens that follow ask where the edited model goes and there is no edited model.

def _walk(answers, log=None):
    it = iter(answers)
    return interactive.plan_abliteration(ask_fn=lambda _p: next(it),
                                         log=(log.append if log is not None else (lambda _m: None)))


def test_every_recipe_on_the_menu_produces_a_command():
    """The rule this list is kept short to satisfy."""
    for i, (key, _label, _note) in enumerate(interactive.RECIPES, 1):
        answers = [str(i), "m", "1", "1", "out", "200"]
        plan = _walk(answers)
        assert plan["command"], f"recipe {key} produced no command"
        assert plan.get("recipe") == key


def test_measuring_never_asks_where_the_edited_model_goes():
    """THE DEFECT FINDING 2 NAMES. There is no edited model, so the question is meaningless, and
    a person who answers it has been walked into a screen built for a different job.
    """
    lines = []
    plan = _walk(["3", "my-model", "1", "1", "scores.json"], log=lines)
    assert plan["command"] == "score"
    assert "--out" in plan["options"] and plan["options"]["--out"] == "scores.json"
    assert "--trials" not in plan["options"], "measuring does not search"
    text = "\n".join(lines).lower()
    assert "edited model go" not in text


def test_measuring_scores_the_held_out_arm_not_the_fitting_one():
    """bad_eval_ds, not bad_ds. Scoring on the rows a run fitted on is the error this project has
    withdrawn published numbers over.
    """
    plan = _walk(["3", "m", "1", "1", "s.json"])
    assert plan["options"]["--eval"].endswith("/bad_eval_ds")


def test_everything_by_hand_gets_out_of_the_way():
    """The person asked for the flags. Wrapping --help in a menu would be the guided mode
    insisting on itself.
    """
    plan = _walk(["4"])
    assert plan["command"] == "--help"
    assert plan["options"] == {}


def test_building_a_brain_is_two_commands_shown_as_two():
    """The convert step is a separate program with its own flags. Collapsing them into one line
    would print something nobody could type, which breaks the rule this file exists to keep.
    """
    plan = _walk(["2", "m", "1", "1", "brain", "200"])
    assert plan["command"] == "kageyoshi"
    assert plan["then"]["command"] == "convert"
    lines = []
    interactive.present(plan, ask_fn=lambda _p: "n", log=lines.append)
    shown = [ln.strip() for ln in lines if ln.strip().startswith("senbonzakura")]
    assert len(shown) == 2, f"both commands must be shown, got {shown}"
    assert shown[1].startswith("senbonzakura convert brain")


def test_the_printed_command_and_the_executed_one_cannot_drift():
    """One function builds both. A guided mode that ran something other than what it displayed
    would be a very good way to hide a mistake.
    """
    plan = _walk(["1", "m", "1", "1", "out", "200"])
    line = interactive.render_command(plan["command"], plan["options"])
    argv = interactive._argv_for(plan)
    assert line == "senbonzakura " + " ".join(
        interactive.quote(a) if not a.startswith("-") else a for a in argv)


def test_a_positional_with_a_space_is_quoted():
    """A path with a space in it is a real thing, and an unquoted one silently becomes two
    arguments. `render_command` appended bare keys without quoting.
    """
    line = interactive.render_command("convert", {"my model dir": True, "--quantise": "Q4_K_M"})
    assert line == "senbonzakura convert 'my model dir' --quantise Q4_K_M"


def test_the_second_command_of_a_brain_only_runs_if_the_first_succeeded(monkeypatch):
    """A convert that runs after a failed abliteration would package whatever was on disk."""
    calls = []
    lines = []
    plan = {"command": "kageyoshi", "options": {"--out": "brain"}, "licence": "yours",
            "then": {"command": "convert", "options": {"brain": True}}}
    monkeypatch.setattr(interactive, "is_tty", lambda _s=None: True)
    monkeypatch.setattr(interactive, "plan_abliteration", lambda **_k: plan)
    monkeypatch.setattr(interactive, "present", lambda _p, **_k: "line")

    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "main", lambda argv: (calls.append(argv), 3)[1])
    assert interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None) == 3
    assert len(calls) == 1, "the convert step ran after the abliteration failed"

    calls.clear()
    monkeypatch.setattr(_cli, "main", lambda argv: (calls.append(argv), 0)[1])
    assert interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None) == 0
    assert len(calls) == 2, "the convert step did not run after a successful abliteration"


def test_a_refusal_from_the_tool_still_gets_the_way_back_in(monkeypatch):
    """A SystemExit is a refusal the tool phrased itself; its message is better than anything this
    could add, so the screen appends the recovery and re-raises rather than replacing it.
    """
    lines = []
    plan = {"command": "kageyoshi", "options": {"--out": "brain"}, "licence": "yours"}
    monkeypatch.setattr(interactive, "is_tty", lambda _s=None: True)
    monkeypatch.setattr(interactive, "plan_abliteration", lambda **_k: plan)
    monkeypatch.setattr(interactive, "present", lambda _p, **_k: "line")

    def _refuse(_argv):
        raise SystemExit(2)

    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "main", _refuse)
    with pytest.raises(SystemExit):
        interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None)
    assert "--resume" in "\n".join(lines)


def test_a_clean_exit_does_not_get_a_failure_screen(monkeypatch):
    """SystemExit(0) is success. Printing "the run stopped" over it would be alarming nonsense."""
    lines = []
    plan = {"command": "kageyoshi", "options": {"--out": "brain"}, "licence": "yours"}
    monkeypatch.setattr(interactive, "is_tty", lambda _s=None: True)
    monkeypatch.setattr(interactive, "plan_abliteration", lambda **_k: plan)
    monkeypatch.setattr(interactive, "present", lambda _p, **_k: "line")

    def _clean(_argv):
        raise SystemExit(0)

    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "main", _clean)
    with pytest.raises(SystemExit):
        interactive.run(ask_fn=lambda _p: "", log=lines.append, stdin=None)
    assert "the run stopped" not in "\n".join(lines)
