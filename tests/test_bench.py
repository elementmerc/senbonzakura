"""Tests for `senbonzakura bench head-to-head`, the operation that used to be a run spec.

The defects this module exists to have fixed are all from 2026-08-05 and 06, and every one of them
lived in shell embedded in orchestrator configuration rather than in either tool: a resume guard
that asked its artefact for a key the artefact had never carried, so it could never fire; a success
marker printed whether or not the work happened, so a rehearsal reported five jobs done having
measured nothing; a scorer invoked three wrong ways at once. Configuration cannot be tested. These
tests are the argument for moving the operation into code.

No test here starts a container, touches a GPU, or runs a model. The subprocess boundary is
injected, which is the whole reason the operation is worth having in Python.
"""
import json

import pytest

from senbonzakura import bench


def _host_out(argv):
    """Where an arm's artefacts land on THIS machine.

    An isolated arm is told `/work/out`, which is where it will see its output directory from
    inside the container; the host side of that mount is in the `--volume` argument. A stub that
    ignored the difference would write to a path that does not exist and would also hide the very
    mistake the mapping exists to prevent.
    """
    candidate = None
    for i, a in enumerate(argv):
        if a == "--volume" and argv[i + 1].endswith(":/work/out:rw"):
            return argv[i + 1].rsplit(":/work/out:rw", 1)[0]
        # run-isolated.sh takes the host output directory as its own `--out`, before the `--`
        # that separates its flags from the command it will run inside.
        if a == "--out" and i + 1 < len(argv) and candidate is None:
            candidate = argv[i + 1]
    return candidate


@pytest.fixture
def runner():
    """A fake arm runner: records what it was asked to run, and can be told to fail or to lie."""
    class Fake:
        def __init__(self):
            self.calls = []
            self.code = 0
            self.produce = True

        def __call__(self, argv, *, cwd=None, log=print):
            self.calls.append(list(argv))
            if self.produce:
                from pathlib import Path
                out = Path(_host_out(argv))
                joined = " ".join(argv)
                if "compass" in argv:
                    # The compass writes one results file, not a directory of artefacts.
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text("{}", encoding="utf-8")
                elif "best_of_n_heretic.py" in joined:
                    # ONLY THE SELECTION PASS WRITES best_of_n.json, because only it can.
                    #
                    # The fake used to write every artefact on every call, which made a Heretic
                    # arm look complete whether or not the pass that finishes it ran at all. That
                    # is how decision Q-5 dropped the pass without a single test noticing, and the
                    # real run then spent 53 minutes an arm producing no model. A fake that is not
                    # faithful about WHICH step produces WHAT cannot catch a missing step.
                    (out / "best_of_n.json").write_text("{}", encoding="utf-8")
                elif "run_heretic" in joined:
                    # Heretic's own search leaves its bookkeeping and no model: v1.4.0 ends at an
                    # interactive menu it cannot reach in a container.
                    for name in ("budget.json", "config.toml"):
                        (out / name).write_text("{}", encoding="utf-8")
                else:
                    for name in ("abliteration.json", "config.json"):
                        (out / name).write_text("{}", encoding="utf-8")
            return self.code
    return Fake()


SLICE_FILES = bench.SLICE_FILES


def _slices(tmp_path, track=None):
    """The staged prompt files every tool scores on. One set, or the arms are not comparable."""
    d = tmp_path / "slices"
    d.mkdir(exist_ok=True)
    for f in SLICE_FILES:
        (d / f).write_text("a prompt\n", encoding="utf-8")
    bench.write_slice_provenance(d, track if track is not None else tmp_path / "track")
    return d


def _args(tmp_path, **over):
    track = tmp_path / "track"
    track.mkdir(exist_ok=True)
    base = dict(model="/models/qwen", track=track, out=tmp_path / "out", trials=200,
                slices=_slices(tmp_path))
    base.update(over)
    return base


# ── the resume guard, which is the defect this file most exists for ───────────────────
def test_a_finished_arm_is_skipped_on_a_second_run(tmp_path, runner):
    a = _args(tmp_path)
    first = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    assert first.ran and first.ok
    second = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    assert not second.ran and second.ok
    assert len(runner.calls) == 1, "the finished arm was run again"


def test_an_arm_from_a_different_configuration_is_not_reused(tmp_path, runner):
    """"A file is here" is what a guard says when it has stopped guarding."""
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    a["trials"] = 60
    again = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    assert again.ran, "an arm searched for 200 trials was reused for a 60-trial run"


def test_a_different_model_is_not_reused_either(tmp_path, runner):
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    a["model"] = "/models/somethingelse"
    assert bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a).ran


def test_a_manifest_without_its_artefacts_does_not_count_as_done(tmp_path, runner):
    """The manifest is a claim; the artefacts are the evidence. Both, or it runs again."""
    a = _args(tmp_path)
    r = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    (r.arm / "abliteration.json").unlink()
    again = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    assert again.ran


def test_the_guard_always_says_why(tmp_path):
    """A guard that reports only true or false cannot be debugged when it is wrong."""
    arm = tmp_path / "senbon-seed42"
    arm.mkdir()
    expected = bench.arm_manifest(bench.ADAPTERS["senbon"], 42, "/m", 200)
    done, why = bench.arm_is_done(arm, expected, bench.ADAPTERS["senbon"])
    assert not done and "no manifest" in why


def test_force_re_runs_a_finished_arm(tmp_path, runner):
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, force=True, **a)
    assert len(runner.calls) == 2


# ── exit zero is not success ──────────────────────────────────────────────────────────
def test_an_arm_that_exits_zero_producing_nothing_is_a_failure(tmp_path, runner):
    """The 2026-08-05 rehearsal's arms exited 0 having written nothing at all."""
    runner.produce = False
    r = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **_args(tmp_path))
    assert not r.ok and "produced no" in r.reason


def test_a_failed_arm_writes_no_manifest(tmp_path, runner):
    """Otherwise the next run would skip it and the failure would become permanent and silent."""
    runner.produce = False
    r = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **_args(tmp_path))
    assert not (r.arm / bench.ARM_MANIFEST).exists()


def test_a_nonzero_exit_is_a_failure_and_names_the_code(tmp_path, runner):
    runner.code = 137
    r = bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **_args(tmp_path))
    assert not r.ok and "137" in r.reason


# ── the arms are given the same problem ───────────────────────────────────────────────
def test_every_arm_gets_the_same_trial_budget(tmp_path, runner):
    """An equal-budget claim is the whole comparison, and it used to live in a bash loop."""
    bench.head_to_head(tools=["senbon", "heretic"], seeds=[42, 43], runner=runner,
                       **_args(tmp_path))
    # The selection pass carries no trial budget: it re-scores candidates the search already
    # spent its budget producing, so it is filtered out rather than expected to declare one.
    budgets = [c[c.index("--trials") + 1] for c in runner.calls if "--trials" in c]
    assert budgets == ["200"] * 4


def test_every_arm_reads_prompts_traceable_to_one_corpus(tmp_path, runner):
    """The two tools no longer name the corpus the same way, so the invariant moved.

    senbonzakura is pointed at the track; Heretic is pointed at prompt files cut from it. Once
    that is true, "both tools read the same corpus" is no longer visible in either command line,
    and becomes an invariant something has to check rather than one a reader can see.
    """
    a = _args(tmp_path)
    bench.head_to_head(tools=["senbon", "heretic"], seeds=[42], runner=runner, **a)
    senbon, heretic = runner.calls[0], runner.calls[1]
    assert senbon[senbon.index("--track") + 1] == str(a["track"])
    assert str(a["slices"]) in " ".join(heretic)
    assert bench.slices_match_track(a["slices"], a["track"]) == []


def test_slices_cut_from_another_corpus_are_refused(tmp_path):
    """Every arm would finish, every artefact would be present, and the table would mean nothing."""
    other = tmp_path / "othercorpus"
    other.mkdir()
    d = _slices(tmp_path, track=other)
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path / "track", out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=d)
    assert any("different corpora" in x for x in p)


def test_slices_that_cannot_say_where_they_came_from_are_refused(tmp_path):
    d = _slices(tmp_path)
    (d / bench.SLICE_PROVENANCE).unlink()
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path / "track", out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=d)
    assert any("which corpus" in x for x in p)


def test_each_seed_reaches_its_arm(tmp_path, runner):
    bench.head_to_head(tools=["senbon"], seeds=[42, 43, 44], runner=runner, **_args(tmp_path))
    assert [c[c.index("--seed") + 1] for c in runner.calls] == ["42", "43", "44"]


def test_a_failing_arm_does_not_stop_the_others(tmp_path, runner):
    """One bad seed should cost that seed, not the night."""
    runner.code = 1
    results = bench.head_to_head(tools=["senbon"], seeds=[42, 43], runner=runner,
                                 **_args(tmp_path))
    assert len(results) == 2 and bench.summarise(results)["failed"] == 2


# ── isolation ─────────────────────────────────────────────────────────────────────────
def test_the_sealed_box_drops_the_network_and_the_capabilities():
    argv = bench.isolate_argv(["python", "run.py"], image="img",
                              mounts=[("/tmp", "/corpus", "ro")])
    for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL", "no-new-privileges"):
        assert flag in argv, f"{flag} missing, so the box is not sealed"


def test_inputs_are_mounted_read_only_and_output_is_not(tmp_path):
    argv = bench.isolate_argv(["x"], image="img",
                              mounts=[(tmp_path, "/corpus", "ro"), (tmp_path, "/work/out", "rw")])
    joined = " ".join(argv)
    assert "/corpus:ro" in joined and "/work/out:rw" in joined


def test_the_command_survives_wrapping_intact():
    argv = bench.isolate_argv(["python", "-m", "x", "--flag", "a b"], image="img", mounts=[])
    assert argv[-5:] == ["python", "-m", "x", "--flag", "a b"]


def test_an_arm_command_is_a_list_so_a_path_with_a_space_stays_one_argument(tmp_path, runner):
    a = _args(tmp_path, model="/models/my model")
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    assert "/models/my model" in runner.calls[0]


# ── preflight ─────────────────────────────────────────────────────────────────────────
def test_one_tool_is_not_a_head_to_head(tmp_path):
    p = bench.preflight(tools=["senbon"], track=tmp_path, out=tmp_path / "o", model="m",
                        isolate="none", images={})
    assert any("needs two" in x for x in p)


def test_an_unknown_tool_is_named_rather_than_ignored(tmp_path):
    p = bench.preflight(tools=["senbon", "nosuchtool"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="none", images={})
    assert any("nosuchtool" in x for x in p)


def test_a_missing_corpus_is_caught_before_the_first_gpu_second(tmp_path):
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path / "nowhere",
                        out=tmp_path / "o", model="m", isolate="none", images={})
    assert any("no corpus" in x for x in p)


def test_an_unwritable_output_directory_is_caught(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=blocked, model="m",
                        isolate="none", images={})
    assert any("cannot write" in x for x in p)


def test_asking_for_isolation_without_an_image_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "docker_available", lambda: True)
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="docker", images={"senbon": "img"})
    assert any("heretic" in x and "image" in x for x in p)


def test_asking_for_isolation_without_docker_says_what_the_alternative_costs(tmp_path,
                                                                             monkeypatch):
    monkeypatch.setattr(bench, "docker_available", lambda: False)
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="docker", images={})
    assert any("credentials" in x for x in p)


def test_a_clean_setup_has_no_complaints(tmp_path):
    assert bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                           model="m", isolate="none", images={},
                           slices=_slices(tmp_path, track=tmp_path)) == []


def test_a_tool_that_would_bring_its_own_prompts_is_refused_without_staged_slices(tmp_path):
    """A tool steered by its own scorer is solving a different problem from ours."""
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=None)
    assert any("eval-slices" in x for x in p)


def test_an_incomplete_slice_directory_names_what_is_missing(tmp_path):
    d = _slices(tmp_path, track=tmp_path)
    (d / "kl_prompts.txt").unlink()
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=d)
    assert any("kl_prompts.txt" in x for x in p)


def test_heretic_is_given_the_staged_prompts_rather_than_fetching_its_own(tmp_path, runner):
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["heretic"], seed=42, runner=runner, **a)
    argv = runner.calls[0]
    for flag in ("--good", "--bad", "--keyword-prompts", "--kl-prompts"):
        assert flag in argv, f"heretic was not told where {flag} is, so it would fetch its own"


def test_running_heretic_with_no_slices_fails_loudly_rather_than_silently(tmp_path, runner):
    a = _args(tmp_path, slices=None)
    with pytest.raises(bench.BenchError) as e:
        bench.run_arm(bench.ADAPTERS["heretic"], seed=42, runner=runner, **a)
    assert "eval-slices" in str(e.value)


# ── the command line ──────────────────────────────────────────────────────────────────
def test_a_refused_preflight_runs_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        bench.main(["head-to-head", "--tools", "senbon", "--model", "m",
                    "--track", str(tmp_path), "--out", str(tmp_path / "o")])
    assert "BENCH REFUSED" in capsys.readouterr().err


def test_running_unsealed_warns_rather_than_proceeding_quietly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "default_runner", lambda *a, **k: 0)
    (tmp_path / "track").mkdir()
    with pytest.raises(SystemExit):
        bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path / "track"),
                    "--out", str(tmp_path / "o"), "--seeds", "42", "--no-score",
                    "--eval-slices", str(_slices(tmp_path, tmp_path / "track"))])
    assert "run unsealed" in capsys.readouterr().err


def test_a_repeated_seed_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path),
                    "--out", str(tmp_path / "o"), "--seeds", "42,42"])
    assert "repeats a value" in str(e.value)


def test_a_seed_that_is_not_a_number_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path),
                    "--out", str(tmp_path / "o"), "--seeds", "42,forty-three"])
    assert "whole numbers" in str(e.value)


def test_a_malformed_image_argument_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path),
                    "--out", str(tmp_path / "o"), "--image", "justanimage"])
    assert "TOOL=IMAGE" in str(e.value)


def test_a_finished_run_writes_a_summary_that_traces_to_a_file(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(bench, "default_runner", runner)
    (tmp_path / "track").mkdir()
    out = tmp_path / "o"
    summary = bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path / "track"),
                          "--out", str(out), "--seeds", "42", "--no-score",
                          "--eval-slices", str(_slices(tmp_path, tmp_path / "track"))])
    assert json.loads((out / "bench-summary.json").read_text(encoding="utf-8")) == summary
    assert summary["failed"] == 0 and summary["ran"] == 2


def test_a_run_with_a_failed_arm_exits_non_zero(tmp_path, monkeypatch, runner):
    runner.code = 1
    monkeypatch.setattr(bench, "default_runner", runner)
    (tmp_path / "track").mkdir()
    with pytest.raises(SystemExit):
        bench.main(["head-to-head", "--model", "m", "--track", str(tmp_path / "track"),
                    "--out", str(tmp_path / "o"), "--seeds", "42", "--no-score",
                    "--eval-slices", str(_slices(tmp_path, tmp_path / "track"))])


# ── the adapters are data, and the two tools genuinely differ ─────────────────────────
def test_the_two_tools_disagree_about_where_the_model_lands():
    """Scoring one tool's arms against nothing at all cost a night on 2026-08-05."""
    assert bench.ADAPTERS["senbon"].model_subdir == ""
    assert bench.ADAPTERS["heretic"].model_subdir == "model"


def test_every_adapter_declares_what_proves_it_ran():
    for name, a in bench.ADAPTERS.items():
        assert a.produces, f"{name} declares no artefact, so nothing can check it"


def test_each_tool_reports_its_own_estimator_by_name(tmp_path):
    """Two tools' KL figures come from different estimators; one column would repeat a withdrawn claim."""
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "abliteration.json").write_text(
        json.dumps({"post_bake_refusals": 0.03, "post_bake_kl": 0.2}), encoding="utf-8")
    r = bench.ADAPTERS["senbon"].self_report(arm)
    assert "senbonzakura" in r["kl_estimator"]
    (arm / "best_of_n.json").write_text(
        json.dumps({"winner": {"refusals": 0.05, "kl": 0.004}}), encoding="utf-8")
    h = bench.ADAPTERS["heretic"].self_report(arm)
    assert "Heretic" in h["kl_estimator"]
    assert r["kl_estimator"] != h["kl_estimator"]


def test_an_unreadable_artefact_is_a_loud_error_not_an_empty_reading(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "abliteration.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(bench.BenchError):
        bench.ADAPTERS["senbon"].self_report(arm)


# ── scoring: the three wrong ways it was invoked, each now impossible ─────────────────
def test_the_compass_is_never_passed_a_track_it_does_not_have(tmp_path):
    argv = bench.score_argv(model=tmp_path, harmful="h", harmless="g", out="o.json",
                            label="x", skip_harmful=128, batch=16)
    assert "--track" not in argv, "the compass has no --track; passing one killed it outright"
    assert "--harmful" in argv and "--harmless" in argv


def test_skip_harmful_carries_its_count_and_never_travels_bare(tmp_path):
    """Passed bare, it swallowed the next argument and the run was measured on the wrong slice."""
    argv = bench.score_argv(model=tmp_path, harmful="h", harmless="g", out="o.json",
                            label="x", skip_harmful=128, batch=16)
    assert argv[argv.index("--skip-harmful") + 1] == "128"


def test_each_tools_model_is_looked_for_where_that_tool_puts_it(tmp_path):
    """Assuming one shape scored a whole tool's arms against nothing at all."""
    senbon = bench.ArmResult("senbon", 42, tmp_path / "senbon-seed42", True, True, "")
    heretic = bench.ArmResult("heretic", 42, tmp_path / "heretic-seed42", True, True, "")
    assert bench.arm_model_dir(senbon, bench.ADAPTERS["senbon"]) == tmp_path / "senbon-seed42"
    assert bench.arm_model_dir(heretic, bench.ADAPTERS["heretic"]) == \
        tmp_path / "heretic-seed42" / "model"


def test_an_arm_with_no_model_is_named_rather_than_skipped_quietly(tmp_path, runner):
    r = bench.ArmResult("senbon", 42, tmp_path / "senbon-seed42", True, True, "")
    (tmp_path / "senbon-seed42").mkdir()
    scored = bench.score_arms([r], harmful="h", harmless="g", out=tmp_path, runner=runner)
    assert scored[0]["ok"] is False and "no model" in scored[0]["reason"]
    assert not runner.calls, "the compass was run against a directory holding no model"


def test_scoring_that_exits_zero_without_writing_a_file_is_a_failure(tmp_path, runner):
    arm = tmp_path / "senbon-seed42"
    arm.mkdir()
    (arm / "config.json").write_text("{}", encoding="utf-8")
    runner.produce = False
    r = bench.ArmResult("senbon", 42, arm, True, True, "")
    scored = bench.score_arms([r], harmful="h", harmless="g", out=tmp_path, runner=runner)
    assert scored[0]["ok"] is False


def test_an_already_scored_arm_is_not_scored_again(tmp_path, runner):
    arm = tmp_path / "senbon-seed42"
    arm.mkdir()
    (arm / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scored-senbon-seed42.json").write_text("{}", encoding="utf-8")
    r = bench.ArmResult("senbon", 42, arm, True, True, "")
    bench.score_arms([r], harmful="h", harmless="g", out=tmp_path, runner=runner)
    assert not runner.calls


def test_every_arm_is_scored_at_the_same_batch_size(tmp_path, runner):
    """Two arms of one comparison measured under different conditions is the mistake the
    console is scheduled to remove; it should not be reachable here in the first place.
    """
    results = []
    for seed in (42, 43):
        arm = tmp_path / f"senbon-seed{seed}"
        arm.mkdir()
        (arm / "config.json").write_text("{}", encoding="utf-8")
        results.append(bench.ArmResult("senbon", seed, arm, True, True, ""))
    bench.score_arms(results, harmful="h", harmless="g", out=tmp_path, runner=runner, batch=8)
    batches = [c[c.index("--batch") + 1] for c in runner.calls]
    assert batches == ["8", "8"]


def test_scoring_inputs_are_checked_before_any_gpu_time_is_spent(tmp_path):
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=_slices(tmp_path, tmp_path),
                        score=True, harmful="", harmless="")
    assert any("--harmful" in x for x in p) and any("--harmless" in x for x in p)


def test_a_scoring_input_that_is_not_there_is_caught_at_preflight(tmp_path):
    p = bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                        model="m", isolate="none", images={}, slices=_slices(tmp_path, tmp_path),
                        score=True, harmful=tmp_path / "nowhere", harmless=tmp_path)
    assert any("not there" in x for x in p)


def test_no_score_leaves_the_scoring_inputs_unrequired(tmp_path):
    assert bench.preflight(tools=["senbon", "heretic"], track=tmp_path, out=tmp_path / "o",
                           model="m", isolate="none", images={},
                           slices=_slices(tmp_path, tmp_path), score=False) == []


# ── the report is reachable from the installed tool, not only from a checkout ─────────
def test_the_report_is_a_subcommand_rather_than_a_script_in_the_repository(tmp_path):
    """It lived under tools/, so it did not ship in the wheel.

    A reader's whole recourse against a table they doubt is being able to re-derive it, and the
    only thing that could read a published head-to-head was a checkout of this repository. Same
    defect as the compass shipping in no released artefact, and it matters more here.
    """
    with pytest.raises(SystemExit) as e:
        bench.main(["report", str(tmp_path)])
    assert "no scored arms" in str(e.value)


def test_the_report_refuses_a_directory_that_is_not_there(tmp_path):
    with pytest.raises(SystemExit) as e:
        bench.main(["report", str(tmp_path / "nowhere")])
    assert "no directory" in str(e.value)


# ── end to end: the whole operation, no GPU, no docker, no model ──────────────────────
def test_the_whole_operation_runs_from_one_command(tmp_path, monkeypatch, capsys):
    """Every unit above passes and the operation could still be wired wrong.

    That is the lesson of 2026-08-05: five jobs reported done, each one individually plausible,
    and nothing had been measured. So this drives `main` the way the spec drives it, through
    preflight, the arms, the manifests, the scoring and the report, with only the subprocess
    boundary replaced. Anything between those steps that does not line up fails here.
    """
    track = tmp_path / "track"
    track.mkdir()
    (track / "bad_eval_ds").mkdir()
    (track / "good_ds").mkdir()
    out = tmp_path / "out"

    def fake(argv, *, cwd=None, log=print):
        from pathlib import Path
        target = Path(argv[argv.index("--out") + 1])
        if "compass" in argv:
            label = argv[argv.index("--label") + 1]
            tool = "senbon" if label.startswith("senbon") else "heretic"
            # Vary by seed. Identical scores across every seed are what a seed that reaches
            # nothing looks like, and the report says so rather than naming a winner, so a stub
            # that returned one number would have tested the refusal instead of the verdict.
            seed = int(label.rsplit("seed", 1)[1])
            auc = (0.95 if tool == "senbon" else 0.60) + (seed - 43) * 0.01
            target.write_text(json.dumps({
                "label": label, "auc": auc, "auc_ci": [auc - 0.01, auc + 0.01],
                "controls": {"length_only_auc": 0.55}}), encoding="utf-8")
        else:
            # Each stub leaves its model where that tool really leaves it. senbonzakura writes
            # straight into the directory it was given; Heretic's comes out of the selection pass
            # in a subdirectory. A stub that ignored the difference would pass this test and hide
            # the mistake that scored a whole tool's arms against nothing on 2026-08-05.
            tool = "heretic" if "run_heretic" in " ".join(argv) else "senbon"
            model = target / bench.ADAPTERS[tool].model_subdir
            model.mkdir(parents=True, exist_ok=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (target / "abliteration.json").write_text(
                json.dumps({"post_bake_refusals": 0.02, "post_bake_kl": 0.21}), encoding="utf-8")
            (target / "best_of_n.json").write_text(
                json.dumps({"winner": {"refusals": 0.04, "kl": 0.005}}), encoding="utf-8")
        return 0

    monkeypatch.setattr(bench, "default_runner", fake)
    summary = bench.main([
        "head-to-head", "--tools", "senbon,heretic", "--seeds", "42,43,44",
        "--model", "/models/qwen", "--track", str(track), "--out", str(out),
        "--eval-slices", str(_slices(tmp_path, track)),
        "--harmful", str(track / "bad_eval_ds"), "--harmless", str(track / "good_ds"),
        "--trials", "6"])

    assert summary["failed"] == 0 and summary["ran"] == 6
    # Six arms, six scores, and a verdict that reads them.
    for tool in ("senbon", "heretic"):
        for seed in (42, 43, 44):
            assert (out / f"{tool}-seed{seed}" / bench.ARM_MANIFEST).is_file()
            assert (out / f"scored-{tool}-seed{seed}.json").is_file()
    printed = capsys.readouterr().out
    assert "senbon scores higher on harm recognition than heretic" in printed
    assert "NOT a comparison" in printed, "the two tools' own figures lost their warning"
    assert "length-only" in printed, "the null control did not reach the table"


def test_a_second_run_of_the_same_command_does_nothing_and_still_reports(tmp_path, monkeypatch,
                                                                        capsys):
    """Re-running an operation must be safe, and must not silently re-do ten hours of work."""
    track = tmp_path / "track"
    track.mkdir()
    (track / "bad_eval_ds").mkdir()
    (track / "good_ds").mkdir()
    out = tmp_path / "out"
    calls = []

    def fake(argv, *, cwd=None, log=print):
        from pathlib import Path
        calls.append(list(argv))
        target = Path(argv[argv.index("--out") + 1])
        if "compass" in argv:
            label = argv[argv.index("--label") + 1]
            target.write_text(json.dumps({
                "label": label, "auc": 0.9 + int(label.rsplit("seed", 1)[1]) * 0.001,
                "controls": {}}), encoding="utf-8")
        else:
            tool = "heretic" if "run_heretic" in " ".join(argv) else "senbon"
            model = target / bench.ADAPTERS[tool].model_subdir
            model.mkdir(parents=True, exist_ok=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            for n in ("abliteration.json", "best_of_n.json"):
                (target / n).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(bench, "default_runner", fake)
    argv = ["head-to-head", "--tools", "senbon,heretic", "--seeds", "42,43,44",
            "--model", "/models/qwen", "--track", str(track), "--out", str(out),
            "--eval-slices", str(_slices(tmp_path, track)),
            "--harmful", str(track / "bad_eval_ds"), "--harmless", str(track / "good_ds"),
            "--trials", "6"]
    bench.main(argv)
    first = len(calls)
    capsys.readouterr()

    second = bench.main(argv)
    assert len(calls) == first, "a completed run re-ran its arms"
    assert second["ran"] == 0 and second["skipped"] == 6
    assert "harm recognition" in capsys.readouterr().out, "the second run produced no report"


def test_a_run_where_every_seed_returned_the_same_score_is_not_a_verdict(tmp_path, monkeypatch,
                                                                         capsys):
    """A spread of zero across seeds is usually a seed that never reached the search.

    This is the half of the tie rule most easily lost: the gap can be enormous and the result
    still worthless. The reporter has always known it; this asserts the whole operation still
    surfaces it rather than printing a winner.
    """
    track = tmp_path / "track"
    track.mkdir()
    (track / "bad_eval_ds").mkdir()
    (track / "good_ds").mkdir()

    def fake(argv, *, cwd=None, log=print):
        from pathlib import Path
        target = Path(argv[argv.index("--out") + 1])
        if "compass" in argv:
            label = argv[argv.index("--label") + 1]
            auc = 0.95 if label.startswith("senbon") else 0.60   # identical for every seed
            target.write_text(json.dumps({"label": label, "auc": auc, "controls": {}}),
                              encoding="utf-8")
        else:
            tool = "heretic" if "run_heretic" in " ".join(argv) else "senbon"
            model = target / bench.ADAPTERS[tool].model_subdir
            model.mkdir(parents=True, exist_ok=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            for n in ("abliteration.json", "best_of_n.json"):
                (target / n).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(bench, "default_runner", fake)
    bench.main(["head-to-head", "--tools", "senbon,heretic", "--seeds", "42,43,44",
                "--model", "/m", "--track", str(track), "--out", str(tmp_path / "out"),
                "--eval-slices", str(_slices(tmp_path, track)),
                "--harmful", str(track / "bad_eval_ds"),
                "--harmless", str(track / "good_ds"), "--trials", "6"])
    printed = capsys.readouterr().out
    assert "spread is exactly" in printed and "never varied anything" in printed


# ── isolation reuses the proven wrapper rather than reimplementing its flags ──────────
def test_a_checkout_runs_arms_through_the_proven_wrapper(tmp_path, runner, monkeypatch):
    """`bench/run-isolated.sh` owns the isolation, and it needs seven mounts and a device.

    A WSL2 container needs `/dev/dxg` AND `/usr/lib/wsl/lib` AND `/usr/lib/wsl/drivers` to see a
    GPU; miss the driver store and libcuda loads, fails to initialise NVML and reports zero
    devices, which reads as "no GPU here" rather than "one bind mount short". `bench/selftest.py`
    verifies nine invariants about that set from inside the box. Reimplementing it in Python would
    be a second copy of a list this project has already been bitten by having three copies of.
    """
    script = tmp_path / "run-isolated.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("SENBON_RUN_ISOLATED", str(script))
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["heretic"], seed=42, runner=runner, isolate="docker",
                  image="senbon-bench:heretic", **a)
    argv = runner.calls[0]
    assert argv[0] == str(script), "the arm did not go through run-isolated.sh"
    assert "--tool" in argv and argv[argv.index("--tool") + 1] == "heretic"
    assert "--senbon-src" in argv, "the shared ruler would not be importable inside the box"
    assert "--eval" in argv, "the staged slices would not be mounted"
    assert "--" in argv, "the inner command was not separated from the wrapper's own flags"


def test_the_wrapper_is_given_the_image_for_that_tool(tmp_path, runner, monkeypatch):
    """Running an arm in the shared base image starts, loads torch, and dies on a missing
    dependency far from the cause.
    """
    script = tmp_path / "run-isolated.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("SENBON_RUN_ISOLATED", str(script))
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, isolate="docker",
                  image="senbon-bench:senbonzakura", **_args(tmp_path))
    argv = runner.calls[0]
    assert argv[argv.index("--image") + 1] == "senbon-bench:senbonzakura"


def test_without_the_wrapper_it_falls_back_to_a_plain_sealed_box(tmp_path, runner, monkeypatch):
    """An installed wheel has no `bench/` directory. The fallback is the minimum sealed box, so a
    stranger still gets no network and read-only inputs rather than nothing.
    """
    monkeypatch.setattr(bench, "find_run_isolated", lambda: None)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, isolate="docker",
                  image="img", **_args(tmp_path))
    argv = runner.calls[0]
    assert argv[0] == "docker"
    assert "--network" in argv and "none" in argv


def test_an_isolated_arm_is_given_the_paths_it_will_see(tmp_path, runner, monkeypatch):
    """A container mounts the model at /model. An arm told the host path dies on its first line.

    The 2026-08-10 rehearsal failed exactly this way: every arm exited 1 with "No module named
    senbonzakura", because the command was built against this machine's layout and then run
    somewhere with a different one. Nothing in the unit tests noticed, because they never crossed
    the boundary where the two layouts differ.
    """
    monkeypatch.setattr(bench, "find_run_isolated", lambda: None)
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, isolate="docker",
                  image="img", **a)
    argv = runner.calls[0]
    assert argv[argv.index("--model") + 1] == bench.GUEST_MODEL
    assert argv[argv.index("--track") + 1] == bench.GUEST_CORPUS
    assert argv[argv.index("--out") + 1] == bench.GUEST_OUT
    assert str(a["model"]) not in " ".join(argv[argv.index("img"):]), (
        "a host path reached the command that runs inside the box")


def test_an_unisolated_arm_still_gets_the_real_paths(tmp_path, runner):
    """The mapping applies only where there is a boundary to cross."""
    a = _args(tmp_path)
    bench.run_arm(bench.ADAPTERS["senbon"], seed=42, runner=runner, **a)
    argv = runner.calls[0]
    assert argv[argv.index("--model") + 1] == str(a["model"])


def test_heretics_staged_slices_are_addressed_inside_the_box(tmp_path, runner, monkeypatch):
    monkeypatch.setattr(bench, "find_run_isolated", lambda: None)
    bench.run_arm(bench.ADAPTERS["heretic"], seed=42, runner=runner, isolate="docker",
                  image="img", **_args(tmp_path))
    argv = runner.calls[0]
    assert argv[argv.index("--good") + 1] == f"{bench.GUEST_EVAL}/good.txt"


# ── the pass that finishes an arm the tool cannot finish itself ───────────────────────
def test_heretic_gets_the_selection_pass_that_writes_its_only_artefact(tmp_path, runner):
    """Heretic's search cannot save; the arm is not finished until the pass runs.

    v1.4.0 ends at an interactive menu it cannot reach in a container, so it leaves a complete
    study and no model. `bench/EQUAL-BUDGET.md` promises it the same best-of-N selection
    senbonzakura gives itself, and that pass is also what does the saving.

    Decision Q-5 moved the arms out of a run spec and into this command, and left the pass behind
    in the spec it replaced. Nothing failed: the adapter went on declaring `best_of_n.json` for
    five days while nothing could write it, and the first real run after that spent 53 minutes an
    arm producing no model at all. This test is the one that would have said so in three seconds.
    """
    bench.head_to_head(tools=["heretic"], seeds=[42], runner=runner, **_args(tmp_path))
    passes = [c for c in runner.calls if "best_of_n_heretic.py" in " ".join(c)]
    assert len(passes) == 1, "Heretic's arm ran without the selection pass that saves its model"
    argv = " ".join(passes[0])
    assert "--top-n 6" in argv, "the pass must consider the same six candidates ours does"
    assert "final_prompts.txt" in argv and "keyword_prompts.txt" in argv, \
        "both tools' winners are chosen on the same held-out slice, or it is not a comparison"


def test_our_own_arm_needs_no_second_pass(tmp_path, runner):
    """Senbonzakura saves its own winner, so a pass here would be a second spend nothing asked for."""
    bench.head_to_head(tools=["senbon"], seeds=[42], runner=runner, **_args(tmp_path))
    assert len(runner.calls) == 1


def test_an_arm_that_produced_nothing_cannot_pass_on_a_previous_run_s_output(tmp_path, runner):
    """Existence was the check, and existence is what a stale file has.

    On 2026-08-11 a rehearsal passed on a model five days old: the tool ran, saved nothing, and
    the previous run's output was still in the directory, so the artefact check found what it was
    looking for and the arm was written up as a success with a fresh manifest vouching for it. The
    gate that exists to prove the pipeline works was reading files from the pipeline it was meant
    to be testing.
    """
    a = _args(tmp_path)
    arm = a["out"] / "heretic-seed42"
    arm.mkdir(parents=True)
    stale = arm / "best_of_n.json"
    stale.write_text('{"winner": "from last week"}', encoding="utf-8")
    import os
    old = 1_600_000_000                      # comfortably before this arm starts
    os.utime(stale, (old, old))

    runner.produce = False                   # the tool runs and writes nothing, as Heretic does
    results = bench.head_to_head(tools=["heretic"], seeds=[42], runner=runner, **a)
    failed = [r for r in results if not r.ok]
    assert len(failed) == 1, "an arm that produced nothing was reported as a success"
    assert "earlier run" in failed[0].reason, failed[0].reason
    assert not (arm / bench.ARM_MANIFEST).exists(), \
        "a manifest was written vouching for a file this arm did not produce"


def test_the_scorer_runs_under_this_interpreter_not_a_bare_name():
    """The arms run in a container; the scorer runs on the host, and the two are not the same box.

    A literal "python" is fine inside the tool images and absent on the machine that dispatches
    them: the card has `python3` only. Scoring died there with "could not start 'python'" the
    first time it ran against an empty output directory, having been masked for days by finding a
    previous run's results and reporting "already scored".
    """
    import sys

    argv = bench.score_argv(model="m", harmful="h", harmless="g", out="o",
                            label="x", skip_harmful=0, batch=8)
    assert argv[0] == sys.executable, \
        "the compass must run under the interpreter the harness was started from"
    assert argv[0] != "python"
