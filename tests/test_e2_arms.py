# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The E2 experiment script, checked by running it rather than by reading it.

WHY THESE ARE THE CHECKS

`tools/e2_arms.sh` is the shape an experiment takes in this project: a shipped file rather than a
command in a message, because a pasted heredoc once ate its own quoting and left a GPU idle for
hours. A file can also drift from the code it drives, and silently: a renamed flag turns a five
hour run into five hours of a script failing on its first arm.

So the script is exercised in `--dry-run`, which prints the exact commands it would run and
touches nothing, and the printed commands are checked against the real parsers. A test that read
the script with a regular expression would be confirming that our own reimplementation of the
script agreed with itself, which is the failure this project wrote a rule about.

THE HOLDOUT INVARIANT IS THE ONE THAT MATTERS

One arm puts the capability probe inside the search, and that probe reads the HEAD of the
benchmark. Scoring from the head would score that arm on the questions it was selected on. The
script skips past the probe's items for every arm, and this asserts the two numbers agree.
Getting that wrong produces a result that looks like a finding and is resampling.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from senbonzakura import methods

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "e2_arms.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("senbonzakura") is None,
    reason="the script drives the installed console script, which this environment lacks")


def _track(tmp_path):
    d = tmp_path / "track"
    d.mkdir()
    for part in ("bad_ds", "good_ds", "bad_eval_ds"):
        (d / part).write_text("", encoding="utf-8")
    return d


def _plan(tmp_path, *extra):
    """Every command the script would run, as lists of arguments."""
    out = subprocess.run(
        [str(SCRIPT), "--model", "Qwen/Qwen3-1.7B", "--track", str(_track(tmp_path)),
         "--out", str(tmp_path / "e2"), "--dry-run", *extra],
        capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr
    return [line.split() for line in out.stderr.splitlines()
            if line.strip().startswith("senbonzakura ")]


def _flags(argv):
    return {a for a in argv if a.startswith("--")}


def _value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_the_script_is_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "a script the other machine has to chmod is a script"


def test_every_method_it_names_exists(tmp_path):
    named = {_value(c, "--method") for c in _plan(tmp_path) if "--method" in c}
    assert named, "no arm passed --method, so the recipes are not being exercised at all"
    unknown = named - set(methods.CHOICES)
    assert not unknown, f"the script names methods that do not exist: {sorted(unknown)}"


def test_every_flag_it_passes_exists_on_the_real_parser(tmp_path):
    """The drift guard. A renamed flag fails here rather than four hours into a run."""
    from senbonzakura import capability as cap
    from senbonzakura import parser as psr

    abl_flags = set()
    for action in psr.build_parser().parse_args.__self__._actions:
        abl_flags.update(action.option_strings)
    cap_flags = set()
    for action in cap.build_parser()._actions:
        cap_flags.update(action.option_strings)

    for argv in _plan(tmp_path):
        known = cap_flags if argv[1] == "capability" else abl_flags
        missing = _flags(argv) - known
        assert not missing, f"`senbonzakura {argv[1]}` has no {sorted(missing)}"


def test_every_subcommand_it_calls_is_registered(tmp_path):
    """Flags being right is no help if the word before them was renamed."""
    from senbonzakura.entry import DELEGATED

    # `abliterate` is the default command and so is spelled in the parser rather than the table.
    known = set(DELEGATED) | {"abliterate"}
    called = {argv[1] for argv in _plan(tmp_path)}
    assert called, "the plan called nothing"
    assert not called - known, f"the script calls unregistered commands: {sorted(called - known)}"


def test_the_scored_items_are_held_out_from_the_in_search_probe(tmp_path):
    """No arm is scored on the questions the capgate arm's search chose against."""
    plan = _plan(tmp_path)
    probes = {int(_value(c, "--capability-n")) for c in plan if "--capability-n" in c}
    assert probes == {40}, f"expected one in-search probe size, got {probes}"
    skips = {int(_value(c, "--skip")) for c in plan if argv_is_scoring(c)}
    assert skips == probes, (
        f"the scoring runs skip {skips} items but the in-search probe reads the first {probes}. "
        f"Any arm scored below that boundary is scored on items it was selected on.")


def argv_is_scoring(argv):
    return len(argv) > 1 and argv[1] == "capability"


def test_the_probe_boundary_moves_together(tmp_path):
    """--probe-n has to move BOTH numbers, or the boundary is only honoured at its default."""
    plan = _plan(tmp_path, "--probe-n", "17")
    assert {int(_value(c, "--capability-n")) for c in plan if "--capability-n" in c} == {17}
    assert {int(_value(c, "--skip")) for c in plan if argv_is_scoring(c)} == {17}


def test_it_refuses_to_run_without_the_reference_arm(tmp_path):
    """Without stock there is nothing to compare against, and the run answers nothing."""
    out = subprocess.run(
        [str(SCRIPT), "--model", "m", "--track", str(_track(tmp_path)),
         "--out", str(tmp_path / "e2"), "--arms", "searched", "--dry-run"],
        capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode != 0
    assert "stock" in out.stderr


def test_it_refuses_an_arm_it_does_not_know(tmp_path):
    out = subprocess.run(
        [str(SCRIPT), "--model", "m", "--track", str(_track(tmp_path)),
         "--out", str(tmp_path / "e2"), "--arms", "stock,teleport", "--dry-run"],
        capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode != 0
    assert "teleport" in out.stderr


def test_the_stock_arm_is_scored_unedited_and_never_abliterated(tmp_path):
    """The reference has to be the model as it arrived, or every delta is measured against a
    model that was already edited.
    """
    plan = _plan(tmp_path)
    stock = [c for c in plan if "--label" in c and _value(c, "--label") == "stock"]
    assert len(stock) == 1
    assert _value(stock[0], "--model") == "Qwen/Qwen3-1.7B"
    assert not any(c[1] == "abliterate" and _value(c, "--out").endswith("/stock/model")
                   for c in plan)


def test_every_edited_arm_is_scored_and_labelled(tmp_path):
    """A run producing absolute accuracies rather than paired changes answers nothing.

    In a dry run stock/capability.json does not exist yet, so --compare-to cannot appear in the
    plan; what must hold is that every arm writes the artefact the comparison needs, and carries a
    label to be compared under.
    """
    plan = _plan(tmp_path)
    edited = [c for c in plan if argv_is_scoring(c) and _value(c, "--label") != "stock"]
    assert len(edited) == len({_value(c, "--out") for c in plan if c[1] == "abliterate"}), (
        "an arm was abliterated and never scored, or scored and never abliterated")
    for c in edited:
        assert _value(c, "--out").endswith("capability.json")
        assert _value(c, "--label")


def test_the_experiment_covers_every_recipe_the_tool_ships(tmp_path):
    """The drift guard in the other direction.

    A recipe added to `methods.py` and not to this script is a recipe that ships unmeasured, and
    nothing else would notice: the script would keep passing, on the arms it already had.
    """
    named = {_value(c, "--method") for c in _plan(tmp_path) if "--method" in c}
    missing = set(methods.CHOICES) - named
    assert not missing, (
        f"these recipes ship but no arm measures them: {sorted(missing)}. Add an arm, or record "
        f"in DEFERRED.md why the recipe is not worth a cell.")
