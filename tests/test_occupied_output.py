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
