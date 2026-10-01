# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura check`: the thirty-second finding, and the three outcomes it must tell apart.

Item G of the shortlist, property 1 of the six. `strategy-2026-08-02.md` on why the 10x axis
scored "partly" and what would fix it: *"the axis is trust, and trust does not demo. Time to
find a real bug does."*

MOST OF WHAT IS TESTED HERE IS THE OUTPUT, not the arithmetic, and that is the right emphasis.
The checks are tested in `test_check_registry.py` and the adapters in `test_check_adapters.py`.
What this command adds is a report somebody has to be able to act on and an exit code a CI job
has to be able to branch on, and both of those are where a checker quietly becomes useless.
"""
import io
import json
import re
from typing import ClassVar

import pytest
from senbonzakura_check import cli

#: An artefact with nothing wrong with it, which is what every test below means by "clean".
#:
#: THE SAMPLE SIZE IS LOAD BEARING AND USED TO BE 10. A rate over 10 rows genuinely trips
#: `rate-reported-on-a-sample-too-small-to-carry-it`, whose floor is 30, so this fixture was never
#: clean; it only read as clean because the Inspect adapter reported no units and the check is gated
#: on them. Fixing that on 2026-09-27 turned the check on and this fixture started failing its own
#: description. Raised to 200 rather than the expectations being relaxed: the finding was correct.
GOOD = {"version": 2, "status": "success",
        "eval": {"task": "t", "model": "m"},
        "results": {"total_samples": 200, "completed_samples": 200,
                    "scores": [{"name": "s", "scorer": "choice", "scored_samples": 200,
                                "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}]}}

BAD = json.loads(json.dumps(GOOD))
BAD["results"]["scores"][0]["scorer"] = None


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _run(argv):
    out = io.StringIO()
    code = cli.main(argv, out=out)
    return code, out.getvalue()


# ── the three outcomes, which must never be confused ─────────────────────────────────────────

def test_a_clean_file_exits_zero(tmp_path):
    code, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert code == 0
    assert "nothing found" in text


def test_a_file_with_a_finding_exits_one(tmp_path):
    code, text = _run([str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    assert "metric-reported-without-its-estimator" in text


def test_a_file_that_cannot_be_read_exits_two(tmp_path):
    """A CI GATE THAT TREATS "could not check" THE SAME AS "checked, nothing found" IS WORSE
    THAN NO GATE, because it goes green on the day the format changes under it.
    """
    p = tmp_path / "strange.json"
    p.write_text(json.dumps({"unrelated": "json"}), encoding="utf-8")
    code, text = _run([str(p)])
    assert code == 2
    assert "UNCHECKED" in text


def test_unreadable_beats_a_finding_in_the_exit_code(tmp_path):
    """When both happen, the louder problem wins: a file nobody could parse is the one that
    invalidates the run, and 1 would let a CI job treat the report as complete.
    """
    code, _ = _run([
        str(_write(tmp_path, "bad.json", BAD)),
        str(_write(tmp_path, "strange.json", {"unrelated": 1})),
    ])
    assert code == 2


@pytest.mark.parametrize(("content", "why"), [
    ("{not json", "malformed JSON"),
    ("[1, 2, 3]", "valid JSON that is not an object"),
    ("null", "valid JSON that is nothing at all"),
])
def test_every_unreadable_shape_is_reported_rather_than_crashing(content, why, tmp_path):
    p = tmp_path / "x.json"
    p.write_text(content, encoding="utf-8")
    code, text = _run([str(p)])
    assert code == 2, why
    assert "UNCHECKED" in text


def test_a_missing_file_is_reported_rather_than_crashing(tmp_path):
    code, text = _run([str(tmp_path / "absent.json")])
    assert code == 2
    assert "could not read it" in text


# ── what the report has to say ───────────────────────────────────────────────────────────────

def test_a_finding_carries_everything_needed_to_judge_it(tmp_path):
    """THE WHOLE VALUE OF THE COMMAND IS IN THIS ASSERTION.

    A finding that says only what fired sends the reader to the source. Each of these four is a
    separate requirement from the v0.8 plan: the incident is the citation without which a finding
    is an opinion, and the false-positive sentence is what lets a reader decide whether to
    believe it.
    """
    code, text = _run([str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    # THE LABELS ARE HEADINGS NOW, capitalised and on their own line, with the body wrapped
    # underneath at a fixed indent. That was 2026-09-27: each section used to print as one
    # unwrapped line, and a single finding measured 5,435 characters across lines of up to 1,646
    # with no blank line anywhere. All four are still required, and this test is why a first attempt
    # to put `seen before` behind a flag was abandoned: the docstring above is right that the
    # incident is the citation without which a finding is an opinion.
    for label in ("What it is", "Seen before", "What to do",
                  "When this check is wrong"):
        assert label in text, f"a finding printed without {label!r}"
    assert "2026-08-05" in text, "the incident has to be quoted, not just referenced"


def test_the_summary_says_how_many_checks_applied(tmp_path):
    """SKIPPED IS NOT PASSED. "No findings" over a set of checks that mostly could not run is a
    different statement from "no findings", and the reader is entitled to tell them apart.

    THE WORDING TURNED ROUND ON 2026-09-27, and the property did not. It said "13 checks did not
    apply" beside a summary line saying "16 checks available", and `--min-applied` then printed
    "3 of 13 applied", reusing the count that did NOT apply as the denominator of those that did.
    Three numbers on one screen and one of them meaning two things. "3 of 16 checks applied" says
    the same thing and reconciles with the line below it.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert "of 16 checks applied" in text


def test_a_clean_report_refuses_to_read_as_a_certificate(tmp_path):
    """Loophole 7 of the v0.8 plan, and it is in the OUTPUT rather than the README because
    somebody will otherwise quote a clean report as a claim of correctness.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert "cannot tell you a number is right" in text


def test_quiet_prints_findings_and_nothing_else(tmp_path):
    _, text = _run(["--quiet", str(_write(tmp_path, "ok.json", GOOD)),
                    str(_write(tmp_path, "bad.json", BAD))])
    assert "bad.json" in text
    assert "ok.json" not in text
    assert "cannot tell you" not in text


def test_json_output_is_machine_readable_and_complete(tmp_path):
    """The shape a CI action consumes. Every field a human sees is in here too, because an
    action that has to re-derive the caveat from the check id will not bother.
    """
    code, text = _run(["--json", str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    got = json.loads(text)
    assert len(got) == 1
    entry = got[0]
    assert entry["unchecked"] is None
    assert entry["findings"][0]["check"] == "metric-reported-without-its-estimator"
    for key in ("title", "confidence", "detects", "incident", "remedy", "false_positive"):
        assert entry["findings"][0][key]


def test_json_output_records_an_unchecked_file_too(tmp_path):
    p = tmp_path / "strange.json"
    p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    code, text = _run(["--json", str(p)])
    assert code == 2
    assert json.loads(text)[0]["unchecked"]


# ── walking a directory ──────────────────────────────────────────────────────────────────────

def test_a_directory_is_searched_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    _write(tmp_path, "a.json", GOOD)
    _write(tmp_path / "nested", "b.json", BAD)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    code, text = _run([str(tmp_path)])
    assert code == 1
    assert "2 file(s)" in text, "the .txt must not be read as a result artefact"


def test_files_are_visited_in_a_stable_order(tmp_path):
    """Two runs over the same tree produce the same report, per baseline section 2.1. A report
    whose order depends on the filesystem cannot be diffed between runs.
    """
    for name in ("c.json", "a.json", "b.json"):
        _write(tmp_path, name, BAD)
    first = _run([str(tmp_path)])[1]
    second = _run([str(tmp_path)])[1]
    assert first == second
    assert first.index("a.json") < first.index("b.json") < first.index("c.json")


def test_an_empty_directory_is_not_an_error(tmp_path):
    code, text = _run([str(tmp_path)])
    assert code == 0
    assert "0 file(s)" in text


# ── the command is wired in, and stays cheap ─────────────────────────────────────────────────

def test_check_is_a_delegated_command():
    from senbonzakura import entry

    assert entry.DELEGATED["check"] == ("check", "main")
    assert entry.dispatch("check") is cli.main


def test_the_command_runs_from_the_entry_point(tmp_path, capsys):
    """Through `entry.main`, which is the path a user's shell actually takes."""
    from senbonzakura import entry

    code = entry.main(["check", str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    assert "metric-reported-without-its-estimator" in capsys.readouterr().out


def test_the_checker_does_not_write_anything_where_it_looked(tmp_path):
    """It is pointed at other people's evidence and the one unforgivable behaviour is changing
    it. Asserted on the directory as well as the file, because a report written beside the
    input would be just as wrong.
    """
    p = _write(tmp_path, "bad.json", BAD)
    before = (p.read_bytes(), p.stat().st_mtime_ns, sorted(q.name for q in tmp_path.iterdir()))
    _run([str(tmp_path)])
    assert (p.read_bytes(), p.stat().st_mtime_ns,
            sorted(q.name for q in tmp_path.iterdir())) == before


# ── naming a file is a claim about it; sweeping a directory is not ───────────────────────────

def test_an_unrecognised_file_named_explicitly_is_unchecked(tmp_path):
    """The user asserted this was a result, so failing to read it invalidates the run."""
    p = tmp_path / "claimed.json"
    p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    code, text = _run([str(p)])
    assert code == 2
    assert "UNCHECKED" in text


def test_an_unrecognised_file_swept_from_a_directory_costs_nothing(tmp_path):
    """FOUND BY DOGFOODING, and it is a usability defect rather than a nicety.

    `head-to-head/results/` holds thirty per-arm artefacts and one run summary. The summary is
    an index of which arms ran and is correctly not a measurement, so the whole directory exited
    2 and would have failed CI for a file that is exactly what it should be. Every real results
    directory has a config or a manifest sitting beside the results, and a checker that exits 2
    on all of them is a checker nobody can point at a directory.
    """
    _write(tmp_path, "result.json", GOOD)
    (tmp_path / "manifest.json").write_text(json.dumps({"ran": ["a", "b"]}), encoding="utf-8")

    code, text = _run([str(tmp_path)])
    assert code == 0, "a manifest beside the results must not fail the run"
    assert "not a result artefact" in text
    assert "1 not a result" in text
    assert "0 unchecked" in text


def test_the_distinction_survives_into_the_json_report(tmp_path):
    """A CI action branches on this, so the two states cannot share a field."""
    (tmp_path / "manifest.json").write_text(json.dumps({"ran": []}), encoding="utf-8")
    swept = json.loads(_run(["--json", str(tmp_path)])[1])[0]
    assert swept["unchecked"] is None
    assert swept["not_a_result"]

    named = json.loads(_run(["--json", str(tmp_path / "manifest.json")])[1])[0]
    assert named["unchecked"]
    assert named["not_a_result"] is None


def test_our_own_published_results_directory_passes_cleanly():
    """The dogfooding case itself, pinned so it cannot regress.

    CI runs exactly this command over the committed arms, and requires exit 0. A finding here
    means either a published artefact has the defect and the figure needs re-examining, or a
    check is wrong and would have fired on a stranger's file too.
    """
    from pathlib import Path

    results = Path(__file__).resolve().parents[1] / "head-to-head" / "results"
    if not results.is_dir():
        pytest.skip("no published results in this checkout")
    code, text = _run([str(results)])
    assert code == 0, text
    assert "0 finding(s)" in text
    assert "0 unchecked" in text


# ── --pair, for the defects that are not visible in one file ─────────────────────────────────

#: Two arms of one comparison, stamped so the instrument check can actually ask its question.
#: Written in the shape our own scorer writes rather than in the normalised vocabulary, so these
#: go through detection and adaptation the way a real file does.
def _arm(version, refusal=0.31):
    return {
        "model": "m", "label": "arm", "instrument": "senbonzakura.refusal_rate",
        # The rows the rate was scored on. Present so these fixtures do not trip the
        # single-document checks: what is under test here is the pair path, and an arm that fires
        # `a-rate-with-no-partition-beside-it` would make the exit code say nothing about it.
        "eval": "held-out",
        "metrics": {"refusal_rate": {
            "metric": "refusal_rate", "value": refusal, "estimator": "senbonzakura-ruler",
            "units": "proportion", "n": 200, "tool_version": version}},
    }


def test_pair_needs_exactly_two_named_files(tmp_path):
    """Which two artefacts are arms of one comparison is a claim only the caller can make, so
    the refusal is loud rather than a guess at what was meant.
    """
    one = str(_write(tmp_path, "a.json", _arm("0.7.1")))
    code, text = _run(["--pair", one])
    assert code == 2
    assert "exactly two" in text


def test_pair_refuses_a_directory_rather_than_pairing_everything_in_it(tmp_path):
    """`head-to-head/results/` holds thirty arms across several models and tools. Pairing them
    all would report findings about 435 comparisons, of which nobody ran more than a handful.
    """
    _write(tmp_path, "a.json", _arm("0.7.1"))
    _write(tmp_path, "b.json", _arm("0.6.0"))
    code, text = _run(["--pair", str(tmp_path)])
    assert code == 2
    assert "named on the command line" in text


def test_a_pair_check_fires_on_two_arms_measured_by_different_builds(tmp_path):
    a = str(_write(tmp_path, "a.json", _arm("0.6.0")))
    b = str(_write(tmp_path, "b.json", _arm("0.7.1")))
    code, text = _run(["--pair", a, b])
    assert code == 1
    assert "a-figure-compared-across-an-instrument-change" in text
    assert "1 pair" in text


def test_the_same_two_arms_are_quiet_when_one_build_measured_both(tmp_path):
    a = str(_write(tmp_path, "a.json", _arm("0.7.1")))
    b = str(_write(tmp_path, "b.json", _arm("0.7.1", refusal=0.12)))
    code, text = _run(["--pair", a, b])
    assert code == 0, text
    assert "a-figure-compared-across-an-instrument-change" not in text


def test_a_pair_check_is_not_reported_as_passing_on_a_single_file(tmp_path):
    """THE DISTINCTION THE WHOLE PAIR DESIGN TURNS ON. Without `--pair` the two pair checks did
    not examine anything, and the summary has to say so rather than folding them into a clean
    report.
    """
    one = str(_write(tmp_path, "a.json", _arm("0.7.1")))
    code, text = _run([one])
    assert code == 0
    assert "checks applied" in text
    entry = json.loads(_run(["--json", one])[1])[0]
    assert not entry["findings"]
    assert "a-figure-compared-across-an-instrument-change" in entry["skipped"]
    assert "arms-that-differ-in-more-than-the-named-variable" in entry["skipped"]


def test_a_pair_with_one_unreadable_arm_is_unchecked_rather_than_clean(tmp_path):
    a = str(_write(tmp_path, "a.json", _arm("0.7.1")))
    b = tmp_path / "b.json"
    b.write_text("{not json", encoding="utf-8")
    code, text = _run(["--pair", a, str(b)])
    assert code == 2
    assert "UNCHECKED" in text


def test_each_arm_is_still_checked_on_its_own_under_pair(tmp_path):
    """`--pair` ADDS the questions that need two artefacts; it does not replace the ones that
    need one. A defect visible in a single file is visible whether or not it is being compared.
    """
    a = str(_write(tmp_path, "a.json", BAD))
    b = str(_write(tmp_path, "b.json", GOOD))
    code, text = _run(["--pair", a, b])
    assert code == 1
    assert "metric-reported-without-its-estimator" in text


# ── the published action ─────────────────────────────────────────────────────────────────────

def _action():
    from pathlib import Path

    import yaml
    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "action.yml").read_text(encoding="utf-8"))


def test_the_action_installs_the_checker_distribution_and_not_the_big_one():
    """STAYING TORCH-FREE IS THE DESIGN, and the way it is achieved changed at Q-29.

    `pip install senbonzakura` brings torch, transformers, accelerate and optuna: most of a
    gigabyte on every CI run of a repository that only wants its result files read. An action
    costing five minutes of install per run is an action nobody keeps, which loses exactly the
    property it was added for. `test_torch_free.py` is the evidence that the checker works that
    way; this is the assertion that the action actually does it.

    WHAT THIS USED TO ASSERT, AND WHY THAT WAS NOT ENOUGH. It used to require `--no-deps` on every
    install line, and it passed on 2026-09-28 against an action that could not run at all.
    `--no-deps senbonzakura` installs the launcher WITHOUT the `senbonzakura_check` package it
    dispatches to, so the only command the action runs exited 1 saying so. The test was asking
    about a flag rather than about the package, and a flag is not the property: `senbonzakura-check`
    declares `dependencies = []`, so naming the right distribution is what keeps the stack out,
    and `--no-deps` on top of it would only hide the next break the same way.

    THE FIRST VERSION OF THIS TEST DID NOT CATCH IT EITHER. It asked whether the STEP mentioned
    `senbonzakura-check` anywhere, and the step also contains a comment and a `senbonzakura-check
    --help` line proving the install, so restoring the broken `pip install --no-deps senbonzakura`
    left the test green. Found by mutation rather than by reading. That is the same defect as the
    one recorded at the top of `test_the_gates_are_actually_wired_into_ci.py`, where a check was
    satisfied by a filename appearing somewhere in a file. So this reads the INSTALL LINE.

    THE PROPERTY IS NOW PER STEP RATHER THAN PER LINE, because the action grew a second command.
    The regression gate ships in the big distribution and is asked for by input, so an install of
    the big distribution is legitimate on the gate's step and nowhere else. The conditions that
    make it legitimate are asserted separately below: it is conditional, it resolves no
    dependencies, and it proves itself before anything trusts it. A blanket "no step installs the
    big one" would have to be deleted to add the gate, and a deleted assertion protects nothing.
    """
    steps = _action()["runs"]["steps"]
    gate_step_names = {"Install the regression gate"}
    lines = [(s.get("name"), ln.strip())
             for s in steps for ln in (s.get("run") or "").splitlines()
             if "pip install" in ln and not ln.lstrip().startswith("#")]
    assert lines, "the action does not install the package"
    checker_lines = [ln for name, ln in lines if name not in gate_step_names]
    assert checker_lines, (
        "no step outside the gate's installs anything, so the checker, which is what this action "
        "is for, is not installed at all")
    for line in checker_lines:
        assert re.search(r'"senbonzakura-check', line), (
            f"the install line does not name the torch-free checker distribution: {line!r}")
        assert not re.search(r'"senbonzakura(?!-check)', line), (
            f"this line installs the big distribution, which brings the deep-learning stack into "
            f"somebody else's CI, and under --no-deps brings a launcher that cannot run: {line!r}")
        assert "--no-deps" not in line, (
            f"the checker's install suppresses dependency resolution, which is the flag that hid "
            f"the 2026-09-28 break: {line!r}")


def test_the_gates_install_is_conditional_dependency_free_and_proves_itself():
    """Three conditions, and the gate's install is only defensible with all three.

    CONDITIONAL, because a repository using the checker alone must not pay for a distribution it
    never calls. DEPENDENCY-FREE, because the gate's whole argument is that it is cheap enough to
    run on every change, and resolving this distribution's dependencies means torch: most of a
    gigabyte on a step that is supposed to cost seconds. PROVES ITSELF, because the one thing the
    2026-09-28 break teaches is that "pip succeeded" is not the property worth asserting.

    `--no-deps` is the right flag here and was the wrong one there, and the difference is what
    this test pins rather than the flag. There, it was how the CHECKER was obtained, and it
    dropped `senbonzakura-check`, the package the `check` command dispatches into. Here the
    checker is installed by name as its own step, and the gate dispatches into nothing: its whole
    import closure is the standard library, which `tests/test_torch_free.py` measures by taking
    every declared dependency away and running the gate to all three of its verdicts.
    """
    steps = {s.get("name"): s for s in _action()["runs"]["steps"]}
    step = steps.get("Install the regression gate")
    assert step, (
        f"no step installs the regression gate. The steps are {sorted(k for k in steps if k)}, and "
        f"a gate nobody installs runs exactly as often as no gate.")

    condition = str(step.get("if") or "")
    assert "inputs.baseline" in condition, (
        f"the gate's install is not conditional on a baseline being asked for ({condition!r}), so "
        f"every consumer of the checker now installs the big distribution too")

    line = next((ln.strip() for ln in step["run"].splitlines()
                 if "pip install" in ln and not ln.lstrip().startswith("#")), None)
    assert line, "the gate's install step runs no pip install"
    assert "--no-deps" in line, (
        f"the gate's install resolves this distribution's dependencies, which means torch on a "
        f"step whose entire argument is that it is cheap: {line!r}")
    assert re.search(r'"senbonzakura(?!-check)', line), (
        f"the gate's install does not name the distribution the gate ships in: {line!r}")

    assert re.search(r"senbonzakura gate --help", step["run"]), (
        "the gate's install does not prove it can run the command. `pip succeeded` is what "
        "reported a clean run on a broken install once already.")


def test_the_action_runs_the_checkers_own_entry_point():
    """`senbonzakura check` needs BOTH distributions; `senbonzakura-check` needs only one.

    The action installs one of them, so it has to call the command that one provides. Calling the
    dispatching form is what broke it: the entry point resolved, the subcommand did not.
    """
    steps = _action()["runs"]["steps"]
    body = "\n".join(s.get("run") or "" for s in steps)
    assert "senbonzakura-check " in body, "the action never invokes the checker"
    assert not re.search(r"senbonzakura check\b", body), (
        "the action calls `senbonzakura check`, which is only available when the BIG distribution "
        "is installed alongside the checker. This action installs the checker alone.")


def test_an_unreadable_report_stops_the_action_rather_than_passing_it():
    """The report is the only evidence the command ran, because `|| true` hides its exit code.

    Findings are a normal non-zero exit, so the action cannot fail on the command's status and
    must read the report instead. When the command could not run at all, the report was empty,
    the count substitutions failed, the variables held empty strings, and every gate below is an
    integer test that ERRORS on an empty string rather than returning false. A skipped gate is a
    passed gate, so the step reached `exit 0` having checked nothing.
    """
    steps = _action()["runs"]["steps"]
    body = "\n".join(s.get("run") or "" for s in steps)
    assert re.search(r"if\s*!\s*counts=\$\(", body), (
        "nothing in the action treats an unreadable report as fatal, so a checker that could not "
        "run reports a clean sweep")
    assert "exit 1" in body.split("counts=$(", 1)[1][:800], (
        "the unreadable-report branch does not leave non-zero")


def test_the_action_can_be_told_not_to_block_on_findings_but_defaults_to_blocking():
    """Advisory mode is the sensible first week on an existing repository; blocking is the
    point of the thing, so it is what you get without asking.
    """
    inputs = _action()["inputs"]
    assert inputs["fail-on-findings"]["default"] == "true"
    assert inputs["fail-on-unchecked"]["default"] == "true"


def test_the_action_exposes_both_counts_separately():
    """A consumer has to be able to tell "we found something" from "we could not look"."""
    outputs = _action()["outputs"]
    assert {"findings", "unchecked", "report"} <= set(outputs)


def test_the_action_asks_for_a_pinned_version_in_its_own_help():
    """An unpinned checker changes what somebody's CI enforces without anything in their
    repository changing, which is the supply-chain rule this project applies to itself.
    """
    assert "Pin it" in _action()["inputs"]["version"]["description"]


# ── the pre-commit hook ──────────────────────────────────────────────────────────────────────

def test_skip_unknown_turns_a_named_unreadable_file_into_a_report(tmp_path):
    """A PATTERN CHOSE THESE PATHS, NOT A PERSON.

    pre-commit hands the hook whatever matched its `files` regex, so the paths arrive named on
    the command line while carrying none of the assertion that naming one usually carries.
    Without this flag the hook would block a commit over a config file that happened to live in
    `results/`, and a hook that blocks wrongly is a hook removed within the week.
    """
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"ran": ["a"]}), encoding="utf-8")

    assert _run([str(p)])[0] == 2, "naming a file is still a claim about it"

    code, text = _run(["--skip-unknown", str(p)])
    assert code == 0
    assert "not a result artefact" in text


def test_skip_unknown_does_not_hide_a_finding(tmp_path):
    """It changes how an UNRECOGNISED file is counted and nothing else. A flag that also
    softened findings would be a flag that quietly turns the hook off.
    """
    code, text = _run(["--skip-unknown", str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    assert "metric-reported-without-its-estimator" in text


def _hooks():
    from pathlib import Path

    import yaml
    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".pre-commit-hooks.yaml").read_text(
            encoding="utf-8"))


def test_the_pre_commit_hook_passes_skip_unknown():
    """The hook's paths come from a regex, so it must not treat them as claims."""
    hook = _hooks()[0]
    assert hook["id"] == "senbonzakura-check"
    assert "--skip-unknown" in hook["entry"]


def test_the_pre_commit_hook_only_fires_on_plausible_results():
    """A hook that runs on every commit regardless is a hook that gets skipped with
    `--no-verify`, and a skipped hook is worth nothing at all.
    """
    import re

    hook = _hooks()[0]
    assert hook["always_run"] is False
    pattern = re.compile(hook["files"])
    for path in ("results/run.json", "evals/log.json", "a/b/logs/x.json"):
        assert pattern.search(path), f"{path} should match"
    for path in ("package.json", "src/config.json", "docs/data.json"):
        assert not pattern.search(path), f"{path} should not match"


def test_the_pre_commit_hook_needs_no_extra_dependencies():
    """Nothing may be added to an environment the hook deliberately does not build.

    THIS TEST USED TO ASSERT `language == "python"`, and its docstring gave the reason as "a hook
    that pulls torch into it is a hook people remove after a week". The concern was exactly right
    and the assertion pinned the configuration that causes it: pre-commit's python language runs
    `pip install .` at this repository's root, which IS torch. The docstring reasoned about the
    checker's imports; pip reads the root distribution's dependencies. So the test described the
    risk and then required it. `test_the_pre_commit_hook_installs_nothing` now holds the language.

    What remains here is still worth asserting. `additional_dependencies` is the one way to put
    something back into a hook that installs nothing, and under `language: script` it would also be
    silently ineffective, so a declaration here is either a mistake or a misunderstanding.
    """
    hook = _hooks()[0]
    assert not hook.get("additional_dependencies"), (
        "the hook declares extra dependencies. It installs nothing by design, so these either do "
        "nothing or reintroduce the weight the script language exists to avoid.")


class TestNothingWasChecked:
    """Zero files is said out loud, and can be made to fail.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-16. A directory that EXISTS and holds no result
    artefacts produced `0 file(s), 0 finding(s), 0 unchecked` and exited 0. Arithmetically that is
    identical to a clean sweep, so a path that drifts, or artefacts that start landing somewhere
    else, reports green forever. A non-existent path was already rc=2; it is the existing-but-empty
    case that is silent.

    This is the checker's own version of the rule this project keeps relearning: silence is only
    evidence if a signal could have reached you. The whole adoption story for this command is CI,
    which is exactly where nobody reads a summary line.
    """

    def test_an_empty_directory_says_nothing_was_checked(self, tmp_path, capsys):
        (tmp_path / "notes.txt").write_text("not an artefact", encoding="utf-8")
        rc = cli.main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "NOTHING WAS CHECKED" in out, out
        assert "not the same as a clean result" in out
        assert rc == 0, "the default must stay usable on a tree with no artefacts yet"

    def test_fail_on_empty_turns_it_into_a_failure(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not an artefact", encoding="utf-8")
        assert cli.main([str(tmp_path), "--fail-on-empty"]) == 1

    def test_fail_on_empty_does_not_fire_when_something_was_checked(self, tmp_path):
        # THE FIXTURE WAS THE BUG IN THIS TEST, corrected 2026-09-21 rather than the assertion.
        # `{"refusal": 0.1}` is not recognised by any adapter: a bare `refusal` needs an
        # `instrument`, a `provenance` or a `label` and `model` pair beside it before the
        # senbonzakura adapter will claim it. So this file was swept, reported as not a result,
        # and never checked, and the test asserting that `--fail-on-empty` did not fire on it was
        # asserting the defect. It is a recognised artefact now, which is what the test says it is.
        _write(tmp_path, "r.json", GOOD)
        assert cli.main([str(tmp_path), "--fail-on-empty"]) in (0, 1, 2)
        assert cli.main([str(tmp_path), "--fail-on-empty"]) == cli.main([str(tmp_path)])


class TestTheMinimumAppliedFloor:
    """`--fail-on-empty` catches a run that examined nothing. This catches one that examined
    almost nothing, which is the likelier drift and looks identical from the outside.

    An artefact counts as checked when a SINGLE check applied to it. So a rename that stopped
    fourteen of fifteen checks recognising a file left the summary, the exit code and the
    dogfooding step in our own CI completely unchanged: a gate proving the checker still runs
    rather than that it still checks.
    """

    def test_an_artefact_with_too_few_checks_applied_is_a_failure(self, tmp_path, capsys):
        _write(tmp_path, "r.json", GOOD)
        rc = cli.main([str(tmp_path), "--min-applied", "99"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "FEWER THAN 99 CHECKS APPLIED" in out, out
        assert "of 15 applied" in out or "applied" in out

    def test_the_floor_names_the_artefact_and_the_count(self, tmp_path, capsys):
        """A failure saying only that something drifted leaves the reader to find which file.

        Asserted on the SHORTFALL LINE rather than anywhere in the output, because the path is
        printed by the ordinary sweep too. The first version of this test looked for the name
        anywhere and passed with the floor disabled, which mutation testing caught: a test for a
        gate that passes when the gate is off is not a test for the gate.
        """
        _write(tmp_path, "r.json", GOOD)
        cli.main([str(tmp_path), "--min-applied", "99"])
        out = capsys.readouterr().out
        shortfall = [ln for ln in out.splitlines() if "applied" in ln and "r.json" in ln]
        assert shortfall, out
        assert re.search(r"r\.json: \d+ of \d+ applied", shortfall[0]), shortfall

    def test_it_does_not_fire_when_the_floor_is_met(self, tmp_path):
        _write(tmp_path, "r.json", GOOD)
        assert cli.main([str(tmp_path), "--min-applied", "1"]) == cli.main([str(tmp_path)])

    def test_zero_means_off_and_is_the_default(self, tmp_path):
        """The default must stay usable: a sweep over a tree of mixed files is a normal thing to
        do, and a floor that fires by default would make the bare command unusable.
        """
        _write(tmp_path, "r.json", GOOD)
        assert cli.main([str(tmp_path), "--min-applied", "0"]) == cli.main([str(tmp_path)])

    def test_a_file_that_is_not_a_result_does_not_trip_the_floor(self, tmp_path):
        """It was never checked, so it has no applied count to be short of. Counting it would
        make the floor fire on any directory holding a README, which is every directory.
        """
        _write(tmp_path, "r.json", GOOD)
        (tmp_path / "notes.txt").write_text("not an artefact", encoding="utf-8")
        assert cli.main([str(tmp_path), "--min-applied", "1"]) == cli.main([str(tmp_path)])


class TestTheExitCodeTable:
    """EVERY OUTCOME AGAINST EVERY COMBINATION OF THE TWO FLAGS, asserted rather than reasoned
    about, because a CI pipeline branches on these numbers and a change to one of them is a
    change to somebody's build.

    THE DEFECT THIS TABLE WAS BUILT AROUND, found 2026-09-21: `--fail-on-empty --skip-unknown`
    exited 0 on a named non-result file. `--skip-unknown` stops it counting as unchecked, the
    file is still LISTED, and `--fail-on-empty` read the listing rather than what was examined.
    So the flag whose entire job is to catch "nothing happened" returned success having checked
    nothing, while its own help text promised the opposite.

    The rule the table encodes: a file is CHECKED when it was read, recognised, and at least one
    check applied to it. Listed is not checked.
    """

    def _outcome(self, tmp_path, kind):
        """One path, plus what it is, for every outcome this command can reach."""
        if kind == "findings":
            return str(_write(tmp_path, "bad.json", BAD))
        if kind == "clean":
            return str(_write(tmp_path, "ok.json", GOOD))
        if kind == "unrecognised":
            p = tmp_path / "manifest.json"
            p.write_text(json.dumps({"ran": ["a"]}), encoding="utf-8")
            return str(p)
        if kind == "missing":
            return str(tmp_path / "absent.json")
        if kind == "empty-directory":
            (tmp_path / "notes.txt").write_text("not an artefact", encoding="utf-8")
            return str(tmp_path)
        if kind == "directory-of-non-results":
            (tmp_path / "manifest.json").write_text(json.dumps({"ran": []}), encoding="utf-8")
            return str(tmp_path)
        raise AssertionError(kind)

    #: (outcome, flags) -> exit code. Read `--skip-unknown` as "these paths came from a pattern,
    #: not from a person", and `--fail-on-empty` as "a path that drifts must not report green".
    TABLE: ClassVar[dict] = {
        ("findings", ()): 1,
        ("findings", ("--fail-on-empty",)): 1,
        ("findings", ("--skip-unknown",)): 1,
        ("findings", ("--fail-on-empty", "--skip-unknown")): 1,

        ("clean", ()): 0,
        ("clean", ("--fail-on-empty",)): 0,
        ("clean", ("--skip-unknown",)): 0,
        ("clean", ("--fail-on-empty", "--skip-unknown")): 0,

        # Named and unreadable is 2 whatever else is set: it is the outcome that makes the rest
        # of the report meaningless. `--skip-unknown` downgrades it to "not a result", and
        # `--fail-on-empty` then catches that nothing was checked. That last cell is the one
        # that changed: it was 0.
        ("unrecognised", ()): 2,
        ("unrecognised", ("--fail-on-empty",)): 2,
        ("unrecognised", ("--skip-unknown",)): 0,
        ("unrecognised", ("--fail-on-empty", "--skip-unknown")): 1,

        ("missing", ()): 2,
        ("missing", ("--fail-on-empty",)): 2,
        ("missing", ("--skip-unknown",)): 0,
        ("missing", ("--fail-on-empty", "--skip-unknown")): 1,

        ("empty-directory", ()): 0,
        ("empty-directory", ("--fail-on-empty",)): 1,
        ("empty-directory", ("--skip-unknown",)): 0,
        ("empty-directory", ("--fail-on-empty", "--skip-unknown")): 1,

        # A directory holding only files that are not results. Swept, so nothing is UNCHECKED,
        # and nothing was checked either. Both `--fail-on-empty` cells changed from 0.
        ("directory-of-non-results", ()): 0,
        ("directory-of-non-results", ("--fail-on-empty",)): 1,
        ("directory-of-non-results", ("--skip-unknown",)): 0,
        ("directory-of-non-results", ("--fail-on-empty", "--skip-unknown")): 1,
    }

    @pytest.mark.parametrize(("case", "expected"), sorted(TABLE.items()),
                             ids=[f"{k}{'+'.join(f.lstrip('-') for f in fl) or '-'}"
                                  for k, fl in sorted(TABLE)])
    def test_the_exit_code(self, case, expected, tmp_path):
        kind, flags = case
        path = self._outcome(tmp_path, kind)
        out = io.StringIO()
        assert cli.main([*flags, path], out=out) == expected, out.getvalue()

    def test_nothing_was_checked_is_said_out_loud_whenever_it_is_true(self, tmp_path):
        """The sentence and the exit code have to agree, or a reader who sees one and a pipeline
        that branches on the other are reading two different reports.
        """
        for kind in ("unrecognised", "missing", "empty-directory",
                     "directory-of-non-results"):
            here = tmp_path / kind
            here.mkdir()
            out = io.StringIO()
            cli.main(["--skip-unknown", self._outcome(here, kind)], out=out)
            assert "NOTHING WAS CHECKED" in out.getvalue(), kind

    def test_a_recognised_file_with_findings_is_never_reported_as_unchecked(self, tmp_path):
        out = io.StringIO()
        cli.main([self._outcome(tmp_path, "findings")], out=out)
        assert "NOTHING WAS CHECKED" not in out.getvalue()


class TestTheEmptyFooterSaysWhichOutcomeThisIs:
    """"NOTHING WAS CHECKED" read identically whether it was about to exit 0 or fail.

    WHAT PROMPTED IT, 2026-09-27

    A surface audit found the same footer on both outcomes, so a reader could not tell from the
    output whether their CI step had just failed, and the one who exits 0 was not told that the
    behaviour they almost certainly want is one flag away. An empty sweep is legitimately fine from a
    shell and almost never fine in CI.

    Written after a mutation pass found the branch unguarded: the behaviour shipped with no test at
    all, which is the same gap as the footer itself had.
    """

    def test_an_empty_sweep_that_exits_zero_names_the_flag(self, tmp_path, capsys):
        assert cli.main([str(tmp_path)]) == 0
        said = capsys.readouterr().out
        assert "NOTHING WAS CHECKED" in said
        assert "--fail-on-empty" in said, (
            f"exit 0 and no mention of the flag that would have made this a failure: {said}")
        assert "This run exits 0" in said

    def test_an_empty_sweep_that_fails_says_that_is_why(self, tmp_path, capsys):
        assert cli.main([str(tmp_path), "--fail-on-empty"]) == 1
        said = capsys.readouterr().out
        assert "NOTHING WAS CHECKED" in said
        assert "Exiting non-zero because --fail-on-empty was given" in said
        assert "This run exits 0" not in said, "it claims to exit 0 while exiting 1"

    def test_the_two_outcomes_do_not_read_the_same(self, tmp_path, capsys):
        """The defect, stated as the property: the reader can tell which happened."""
        cli.main([str(tmp_path)])
        quiet = capsys.readouterr().out
        cli.main([str(tmp_path), "--fail-on-empty"])
        loud = capsys.readouterr().out
        assert quiet != loud, "the footer is identical on exit 0 and on failure"

    def test_a_sweep_that_checked_something_gets_no_footer(self, tmp_path):
        """The branch must not fire on a run that did work.

        `GOOD` rather than a hand-made object, for the reason recorded above
        `test_fail_on_empty_does_not_fire_when_something_was_checked`: a plausible-looking document
        that no adapter recognises is swept, reported as not a result and never checked, so a test
        built on one asserts the defect. My first version of this used `{"schema": "x", "metrics":
        {}}` and failed, correctly, on the same mistake that file already carries a comment about.
        """
        import io

        _write(tmp_path, "r.json", GOOD)
        out = io.StringIO()
        cli.main([str(tmp_path)], out=out)
        assert "NOTHING WAS CHECKED" not in out.getvalue()


class TestTheEmptyFooterNamesWhatItWouldHaveAccepted:
    """A swept file that no adapter knows is told "not a result artefact, skipped" and no more.

    WHAT PROMPTED IT, 2026-10-01

    A CLI surface sweep item: "`senbonzakura check` never says which files it accepts." Driven
    here, half of it held. A file NAMED on the command line is told, because
    `UnknownArtefactError` carries "Supported: lm-evaluation-harness, inspect, senbonzakura".
    A file SWEPT out of a directory is not: it gets five words, and the footer that follows said
    nothing either. So the one run where the reader has nothing else to go on was the run that
    named no kinds.

    The list is read off `adapters.ADAPTERS` rather than typed into the message, and the test
    below holds that coupling rather than the three names, because a fourth adapter must not be
    able to leave the sentence stale.
    """

    def _footer(self, path, capsys, *flags):
        cli.main([str(path), *flags])
        return capsys.readouterr().out

    def test_an_empty_sweep_names_the_kinds_it_recognises(self, tmp_path, capsys):
        from senbonzakura_check import adapters

        said = self._footer(tmp_path, capsys)
        assert "Recognised kinds" in said, said
        for adapter in adapters.ADAPTERS:
            assert adapter.name in said, f"{adapter.name} is registered and unnamed in: {said}"

    def test_a_directory_of_non_results_names_them_too(self, tmp_path, capsys):
        """The case the sweep item was actually about, rather than the empty directory."""
        _write(tmp_path, "notaresult.json", {"hello": 1})
        said = self._footer(tmp_path, capsys)
        assert "not a result artefact, skipped" in said
        assert "Recognised kinds" in said, said

    def test_the_kinds_are_not_named_on_a_run_that_checked_something(self, tmp_path, capsys):
        """Advice for a mistake nobody made is noise, and noise is what gets skimmed past."""
        _write(tmp_path, "r.json", GOOD)
        said = self._footer(tmp_path, capsys)
        assert "Recognised kinds" not in said, said

    @pytest.mark.parametrize("kind", ["empty", "non-results"])
    def test_the_footer_fits_an_eighty_column_terminal(self, tmp_path, capsys, kind):
        """Measured at 113 and 123 columns before this, which is what prompted the wrapping.

        The reported path is exempt and only the path: it is something a reader pastes, and
        breaking it would be the bigger defect. Every other line here is prose.
        """
        if kind == "non-results":
            _write(tmp_path, "notaresult.json", {"hello": 1})
        said = self._footer(tmp_path, capsys)
        for line in said.splitlines():
            if str(tmp_path) in line:
                continue
            assert len(line) <= 79, f"{len(line)} columns: {line!r}"


def test_the_pre_commit_hook_installs_nothing():
    """pre-commit's `python` language would install this repository's ROOT, which is the big one.

    Measured against pre-commit 4.6.2: `languages/python.py` runs `pip install .` in a clone of the
    hook repository, and this repository's root distribution depends on torch, transformers,
    accelerate, optuna, pyarrow and sentencepiece. There is no way to name a subdirectory. So a
    `language: python` hook here puts roughly a gigabyte into somebody's commit hook, whatever the
    manifest says about the checker importing nothing.

    `script` installs nothing at all, which is correct rather than merely cheaper: the checker
    declares `dependencies = []`, so there is no dependency for an isolated environment to hold.
    """
    hook = _hooks()[0]
    assert hook["language"] == "script", (
        f"the hook language is {hook['language']!r}. `python` installs this repository's root "
        "distribution, and that is the deep-learning stack.")


def test_the_pre_commit_entry_point_exists_and_can_be_executed():
    """`language: script` runs a FILE out of the clone, so the file has to be there and runnable.

    A path that does not resolve, or a file committed without its executable bit, fails at the
    stranger's machine and nowhere else. Git tracks the mode, so this is checkable here.
    """
    import os
    from pathlib import Path

    hook = _hooks()[0]
    entry = Path(hook["entry"].split()[0])
    target = Path(__file__).resolve().parents[1] / entry
    assert target.is_file(), (
        f"the hook entry {entry} does not exist, so the hook cannot run from a clone")
    assert os.access(target, os.X_OK), (
        f"{entry} is not executable. `language: script` runs it directly, so a missing mode bit "
        "breaks the hook for everyone who installs it and for nobody who develops it.")
    assert target.read_text(encoding="utf-8").startswith("#!"), (
        f"{entry} has no shebang. pre-commit parses it itself on Windows, so without one the hook "
        "is POSIX-only at best.")


def test_the_report_says_how_many_checks_actually_ran(tmp_path):
    """`applied` is what separates a file that was examined from one that was merely opened.

    It was computed and dropped, so every consumer had to infer "was this checked" from
    `unchecked` and `not_a_result`, and a file that parsed with every check skipped came out
    looking checked. The Action's `fail-on-empty` could not fire because of it.
    """
    out = io.StringIO()
    cli.main(["--json", str(_write(tmp_path, "good.json", GOOD))], out=out)
    entries = json.loads(out.getvalue())
    assert entries, "no report was produced"
    for entry in entries:
        assert "applied" in entry, (
            "the JSON report does not say how many checks ran, so a reader cannot reproduce the "
            "command's own n_checked and has to keep a second definition of it")
    assert entries[0]["applied"] >= 1, (
        "a recognised artefact reports no checks applied, so nothing actually ran on it")


def test_a_file_nobody_could_parse_reports_no_checks_applied(tmp_path):
    """The other half: `applied` has to be falsy exactly where the command counts nothing.

    Without this the field could be present, always non-zero, and the count built on it would be
    the same over-report in a new costume.
    """
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"ran": ["a"]}), encoding="utf-8")
    out = io.StringIO()
    cli.main(["--json", "--skip-unknown", str(p)], out=out)
    entry = json.loads(out.getvalue())[0]
    assert entry["not_a_result"], "the fixture was recognised as a result after all"
    assert not entry["applied"], (
        f"a file no adapter recognised reports applied={entry['applied']!r}, so counting it as "
        "checked is still possible")


class TestSweptIsNotAnExcuseForUnreadable:
    """The same bytes must not exit 2 when named and 0 when swept.

    FOUND BY RUNNING THE TOOL, 2026-09-28, not by reading it. `senbonzakura check dir/` exited 0
    over a directory whose every file was malformed JSON, while naming those same files exited 2.
    A CI step pointed at `results/` therefore went green on a directory of corrupt artefacts, and
    the GitHub Action, whose default path is a directory, inherited it: `fail-on-unchecked` is on
    by default and could never fire, because nothing was ever counted as unchecked.

    The sweep discount is still right for what it was written for. Sweeping a directory is not a
    claim that everything in it is a result, so a config file living in `results/` costs nothing.
    Being unable to PARSE a file is not that: it is true of the file however it was reached.
    """

    def _swept(self, tmp_path, name, text):
        (tmp_path / name).write_text(text, encoding="utf-8")
        out = io.StringIO()
        code = cli.main(["--json", str(tmp_path)], out=out)
        return code, json.loads(out.getvalue())[0]

    def test_malformed_json_swept_from_a_directory_is_unchecked(self, tmp_path):
        code, entry = self._swept(tmp_path, "broken.json", '{"results": {broken')
        assert entry["unchecked"], (
            "a file that could not be parsed was reported as merely not-a-result because it was "
            "swept rather than named, so nothing counted it and the run went green")
        assert not entry["not_a_result"]
        assert code == 2, f"a directory holding an unreadable artefact exited {code}, not 2"

    def test_a_file_that_is_simply_not_a_result_still_costs_nothing_when_swept(self, tmp_path):
        """The rule the discount exists for, asserted so the fix above cannot swallow it."""
        code, entry = self._swept(tmp_path, "config.json", json.dumps({"ran": ["a"]}))
        assert entry["not_a_result"], "a valid, unrecognised file stopped being discounted"
        assert not entry["unchecked"]
        assert code == 0, f"an unrecognised swept file now exits {code}; the discount is gone"

    def test_naming_it_and_sweeping_it_agree_for_an_unreadable_file(self, tmp_path):
        """The property in one line: the verdict is about the file, not about how it was reached."""
        p = tmp_path / "broken.json"
        p.write_text('{"results": {broken', encoding="utf-8")
        named = cli.main([str(p)], out=io.StringIO())
        swept = cli.main([str(tmp_path)], out=io.StringIO())
        assert named == swept == 2, (
            f"named exits {named} and swept exits {swept} for the same bytes")


def test_the_actions_description_fits_what_marketplace_accepts():
    """125 CHARACTERS, AND THE LISTING IS REFUSED ABOVE IT.

    This is not a style rule. The description that shipped with v0.4.0 was 185 characters, so
    the Marketplace publish form rejected it, and the action stayed unlisted while every other
    piece of its metadata (name, author, branding icon and colour) was correct and had been
    checked. Nothing in the repository measured the one field that was wrong, so "ready to
    publish" was asserted from the fields somebody thought to look at.

    The limit is on the rendered value, not the source lines, because the folded block scalar
    this field uses is wrapped for the file and joined before anybody counts it.
    """
    description = " ".join(_action()["description"].split())
    assert len(description) <= 125, (
        f"action.yml's description is {len(description)} characters and Marketplace takes at "
        f"most 125, so the listing would be refused with every other field correct:\n"
        f"  {description}\n"
        f"Shorten it. The longer explanation belongs in the comments above the field, or in the "
        f"action's own documentation, where no limit applies.")


class TestOurOwnEvidenceIsNotCalledForeign:
    """`senbonzakura check evidence/` said "not a result artefact" of this project's own file.

    WHAT PROMPTED IT, 2026-10-01

    A sweep item, driven. Pointed at the committed `evidence/` tree, the checker reports
    `k-sweep-2026-08-13/drift-per-seed.json` as "not a result artefact, skipped". That file
    declares `schema: senbonzakura-evidence/1` in its first field.

    **Skipping it is right.** It carries per-seed lists for two arms (`drift_kl.k1`,
    `drift_kl.k2`) rather than one measurement with an estimator, so there is no single figure
    any check can read, and `_OUR_METRICS` finds no top-level scalar because there is not one.
    Verified by reading the file rather than inferring it from the message.

    **The words were wrong**, and wrong in the direction that matters: they say a file carrying
    our own schema is foreign to us, so a reader checking our published evidence cannot tell a
    collection from a stranger's file. The verdict and the exit status are unchanged; only the
    sentence is.
    """

    COLLECTION: ClassVar[dict] = {"schema": "senbonzakura-evidence/1", "what": "per-seed drift",
                                  "drift_kl": {"k1": [0.04, 0.02], "k2": [0.06, 0.07]}}

    def _said(self, tmp_path, doc, name="thing.json"):
        import io

        _write(tmp_path, name, doc)
        out = io.StringIO()
        cli.main([str(tmp_path)], out=out)
        return out.getvalue()

    def test_an_evidence_collection_is_named_as_one(self, tmp_path):
        said = self._said(tmp_path, self.COLLECTION)
        assert "senbonzakura evidence collection" in said, said
        assert "not a result artefact" not in said, (
            "a file carrying our own schema is still described as foreign")

    def test_the_reason_it_cannot_be_checked_is_given(self, tmp_path):
        """Naming the file kind without saying why nothing read it leaves the reader where they were."""
        said = self._said(tmp_path, self.COLLECTION)
        assert "not a single measurement" in said, said

    def test_a_genuinely_foreign_file_keeps_the_general_wording(self, tmp_path):
        """The guard must not relabel everything it sweeps.

        A file with no schema, or somebody else's schema, is exactly what the original sentence
        was written for.
        """
        said = self._said(tmp_path, {"hello": 1})
        assert "not a result artefact" in said, said
        assert "senbonzakura evidence" not in said, said

    def test_another_tools_schema_is_not_claimed_as_ours(self, tmp_path):
        said = self._said(tmp_path, {"schema": "someone-elses-harness/3", "value": 1})
        assert "not a result artefact" in said, said

    def test_the_verdict_and_the_status_are_unchanged(self, tmp_path):
        """Only the sentence moved. A swept non-result still costs nothing and still exits 0.

        Asserted because a reworded disposition that also changed the count or the status would
        be a behaviour change dressed as a readability fix.
        """
        import io

        _write(tmp_path, "collection.json", self.COLLECTION)
        out = io.StringIO()
        assert cli.main([str(tmp_path)], out=out) == 0
        said = out.getvalue()
        assert "1 not a result" in said, said
        assert "0 unchecked" in said, said

    def test_naming_it_explicitly_is_still_unchecked_and_still_exits_two(self, tmp_path):
        """Naming a file is a claim about it, and that rule does not bend for our own schema.

        The sweep wording is about a file the user did not vouch for. Somebody who types the path
        asserted it was a result, and the answer to that is still UNCHECKED.
        """
        import io

        path = _write(tmp_path, "collection.json", self.COLLECTION)
        out = io.StringIO()
        assert cli.main([str(path)], out=out) == 2
        assert "UNCHECKED" in out.getvalue(), out.getvalue()

    def test_the_real_committed_artefact_is_the_one_this_is_about(self):
        """Read the actual file, so this cannot pass on a fixture that drifted from it.

        Skipped rather than failed if the evidence tree moves: its layout is not this test's
        subject, and a skip here is honest where an assertion would be about the wrong thing.
        """
        import json
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent
                / "evidence" / "k-sweep-2026-08-13" / "drift-per-seed.json")
        if not path.is_file():
            pytest.skip("the k-sweep evidence is no longer where this test expects it")
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert str(doc.get("schema", "")).startswith("senbonzakura-evidence/"), (
            "the committed artefact no longer declares our schema, so the wording this class "
            "guards would not fire on the file it was written for")
        assert not any(isinstance(v, (int, float)) and k != "prompts_per_score"
                       for k, v in doc.items()), (
            "the artefact now carries a top-level numeric metric, so an adapter may claim it and "
            "this whole class needs re-reading")
