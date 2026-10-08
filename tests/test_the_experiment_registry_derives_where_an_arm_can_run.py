# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Where an arm can run is arithmetic, never a note somebody wrote down.

WHY THIS FILE EXISTS

Four experiments sat parked across two handoffs with "needs a card" beside them, and that note was
wrong about two of them: one needed a model nobody had looked for and one needed code rather than
hardware. Believing it would have rented a card to answer a question a laptop could answer.

So the registry stores the model size and the verdict is derived. These tests hold the one property
that matters: **no arm's `where` is a stored string.** Change a card size and every verdict moves;
change a verdict by hand and there is nowhere to put it.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "research"))
import experiments as ex


def test_every_arm_has_a_question_a_stake_and_its_prerequisites():
    """An arm with no stake cannot be ranked against the others, which is the whole job."""
    for arm in ex.ARMS:
        assert arm.question.endswith("?"), f"{arm.key}: the question is not a question"
        assert len(arm.stake) > 80, f"{arm.key}: the stake is too short to rank on"
        assert arm.needs, f"{arm.key}: nothing recorded about what it needs"
        assert all(isinstance(n, str) and len(n) > 20 for n in arm.needs), arm.key


def test_the_keys_are_unique():
    keys = [a.key for a in ex.ARMS]
    assert len(set(keys)) == len(keys)


def test_a_bigger_card_moves_arms_from_rent_to_here():
    """THE PROPERTY. If any verdict were a stored string this could not pass."""
    small = {r["key"]: r["where"] for r in ex.plan(gb=4)["arms"]}
    large = {r["key"]: r["where"] for r in ex.plan(gb=80)["arms"]}
    moved = [k for k in small if small[k].startswith("rent") and large[k].startswith("here")]
    assert moved, "no arm changed verdict between a 4 GB card and an 80 GB one"


def test_an_unbuilt_arm_is_blocked_whatever_the_card():
    """Needing code is a different kind of blocked from needing hardware, and conflating them is
    how an arm that costs an afternoon gets filed with one that costs a rental.
    """
    for row in ex.plan(gb=80)["arms"]:
        arm = next(a for a in ex.ARMS if a.key == row["key"])
        if not arm.built:
            assert row["where"].startswith("blocked"), row
            assert "rent" not in row["where"]


def test_a_machine_with_no_card_says_rent_rather_than_crashing():
    table = ex.plan(gb=0.0)
    assert table["runnable_here"] == [] or all(
        next(a for a in ex.ARMS if a.key == k).vram_gb() == 0
        for k in table["runnable_here"])


def test_the_vram_estimate_is_generous_rather_than_optimistic():
    """An arm called feasible that dies at 94% of a card has cost more than one called
    infeasible that was not. So the slack is above 1 and the test says so out loud.
    """
    assert ex.ACTIVATION_SLACK > 1.0
    arm = next(a for a in ex.ARMS if a.params_b)
    assert arm.vram_gb() > arm.params_b * ex.BYTES_PER_PARAM


def test_an_arm_that_loads_no_model_needs_no_card():
    made_up = ex.Arm(key="k", question="?", stake="x" * 90, params_b=0, hours=1,
                     needs=("a" * 30,), built=True, command=("true",))
    assert made_up.vram_gb() == 0.0
    assert made_up.where(0.0).startswith("here")


# ── the pod batch ───────────────────────────────────────────────────────────────────

def test_the_batch_holds_only_what_will_not_run_here():
    script = ex.pod_script(gb=80)
    for arm in ex.ARMS:
        if arm.built and arm.fits(80):
            assert f">>> {arm.key}" not in script, f"{arm.key} fits and is in the rented batch"


def test_the_batch_is_ordered_longest_first():
    """A pod bills from boot, so a short arm waiting on a long one is the cost this avoids."""
    script = ex.pod_script(gb=1)
    ordered = [line.split(">>> ")[1].strip("'") for line in script.splitlines()
               if ">>> " in line]
    hours = [next(a for a in ex.ARMS if a.key == k).hours for k in ordered]
    assert hours == sorted(hours, reverse=True), ordered


def test_the_batch_names_the_unbuilt_arms_without_putting_them_in_it():
    """They are the cheaper work. A batch that quietly omitted them would read as a full plan."""
    script = ex.pod_script(gb=1)
    for arm in ex.ARMS:
        if not arm.built:
            assert arm.key in script
            assert f">>> {arm.key}" not in script


def test_a_batch_with_nothing_to_rent_says_so_rather_than_printing_an_empty_script():
    script = ex.pod_script(gb=10_000)
    assert "Nothing needs renting" in script


def test_the_batch_states_the_estimated_total():
    assert "Estimated total" in ex.pod_script(gb=1)


# ── the command line ────────────────────────────────────────────────────────────────

def test_the_plan_runs_and_names_every_arm(capsys):
    assert ex.main(["--plan", "--card-gb", "6"]) == 0
    out = capsys.readouterr().out
    for arm in ex.ARMS:
        assert arm.key in out


def test_the_json_plan_is_machine_readable(capsys):
    import json
    assert ex.main(["--plan", "--json", "--card-gb", "6"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["card_gb"] == 6
    assert len(doc["arms"]) == len(ex.ARMS)


def test_running_an_unbuilt_arm_lists_what_it_still_needs():
    """Rather than a bare refusal: the needs ARE the next action."""
    unbuilt = next(a for a in ex.ARMS if not a.built)
    with pytest.raises(SystemExit) as caught:
        ex.main(["--run", unbuilt.key])
    assert unbuilt.needs[0][:30] in str(caught.value)


def test_running_an_unknown_arm_is_refused_by_name():
    with pytest.raises(SystemExit, match="no arm called"):
        ex.main(["--run", "not-an-arm"])


def test_a_built_arm_points_at_the_tool_rather_than_pretending_to_run_it():
    """Planning is all this file does. An arm runs through the tool's own command, and a
    planner that shelled out to it would be a second place the flags live.
    """
    built = next(a for a in ex.ARMS if a.built)
    with pytest.raises(SystemExit, match=" ".join(built.command)):
        ex.main(["--run", built.key])


def test_reading_the_card_never_raises_on_a_machine_without_one(monkeypatch):
    monkeypatch.setattr(ex.shutil, "which", lambda _n: None)
    assert ex.card_gb() == 0.0
