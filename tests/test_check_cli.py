# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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

GOOD = {"version": 2, "status": "success",
        "eval": {"task": "t", "model": "m"},
        "results": {"total_samples": 10, "completed_samples": 10,
                    "scores": [{"name": "s", "scorer": "choice", "scored_samples": 10,
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
    for label in ("what it is:", "seen before:", "what to do:",
                  "when this check is wrong:"):
        assert label in text, f"a finding printed without {label!r}"
    assert "2026-08-05" in text, "the incident has to be quoted, not just referenced"


def test_the_summary_says_how_many_checks_did_not_apply(tmp_path):
    """SKIPPED IS NOT PASSED. "No findings" over a set of checks that mostly could not run is a
    different statement from "no findings", and the reader is entitled to tell them apart.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert "did not apply" in text


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
    assert "checks did not apply" in text
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


def test_the_action_installs_without_the_deep_learning_stack():
    """`--no-deps` IS THE WHOLE DESIGN, not an optimisation.

    `pip install senbonzakura` brings torch, transformers, accelerate and optuna: most of a
    gigabyte on every CI run of a repository that only wants its result files read. An action
    costing five minutes of install per run is an action nobody keeps, which loses exactly the
    property it was added for. `test_torch_free.py` is the evidence that the checker works that
    way; this is the assertion that the action actually does it.
    """
    steps = _action()["runs"]["steps"]
    install = [s for s in steps if "pip install" in (s.get("run") or "")]
    assert install, "the action does not install the package"
    assert all("--no-deps" in s["run"] for s in install), (
        "the action installs the deep-learning stack into somebody else's CI")


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
    """pre-commit builds an isolated environment from this repository, and a hook that pulls
    torch into it is a hook people remove after a week. The checker imports nothing beyond the
    standard library, which `test_torch_free.py` asserts by running it with the stack stripped.
    """
    hook = _hooks()[0]
    assert hook["language"] == "python"
    assert not hook.get("additional_dependencies"), (
        "the hook declares extra dependencies, which defeats the point of a torch-free checker")


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
