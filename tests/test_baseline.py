# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The regression gate's core: what a baseline is, and when two numbers may be compared.

These tests are organised by the loophole hunt in `private/plans/06-v0.9-the-regression-gate.md`
rather than by function, because the failure this module exists to prevent is not a missed
regression. It is a gate that compares two numbers which were never comparable and puts a green
tick on the result, which is worse than no gate at all: a passing build is evidence, and a missing
gate is not.
"""
import json

import pytest

from senbonzakura import baseline as b

BASE = {
    "model": "Qwen/Qwen3-1.7B",
    "metric": "refusal_rate",
    "direction": b.LOWER_IS_BETTER,
    "point": 0.10,
    "interval": (0.05, 0.18),
    "input_digest": "sha256:aaa",
    "partition": "measure",
    "prompt_format": "renderer:v3",
    "tool_version": "0.4.0",
    "estimator": "senbonzakura-ruler",
    "precision": "bfloat16",
    "seeds": [42, 43, 44],
    "n": 200,
}

#: The conditions a later run reports, matching the baseline above. Tests mutate one field at a
#: time so a failure names exactly which pin did the refusing.
SAME_CONDITIONS = {k: BASE[k] for k in b.PINNED}


def a_baseline(**over):
    return b.record(**{**BASE, **over})


# ── loophole 1: comparing two numbers that were never comparable ─────────────────
class TestItRefusesRatherThanGuesses:
    """The published table this project already has is the argument for every test here.

    Three copies of the prompt renderer drifted, a configuration selected under one format was
    reported under another, and nothing noticed. A gate that compares across that gap does not
    merely miss the error, it certifies it.
    """

    def test_matching_conditions_compare_without_complaint(self):
        b.refuse_if_incomparable(a_baseline(), dict(SAME_CONDITIONS))

    @pytest.mark.parametrize("field", sorted(b.PINNED))
    def test_any_single_pinned_field_moving_refuses(self, field):
        """Every pin, one at a time, so none of them is decorative.

        Parameterised over `PINNED` itself rather than a hand-written list: adding a pin without
        a test is then impossible, and a pin quietly deleted fails here.
        """
        now = dict(SAME_CONDITIONS)
        now[field] = "something else entirely"
        with pytest.raises(b.BaselineError, match=field):
            b.refuse_if_incomparable(a_baseline(), now)

    def test_a_field_missing_on_either_side_refuses_rather_than_skips(self):
        """"Cannot tell" has to fail, and this is where it differs from `runrecord` on purpose.

        There, a record with a missing field still describes a run somebody is trying to finish,
        and refusing would strand them. Here, an unknown field means the gate cannot show the two
        measurements are comparable, and the whole value of the module is that it does not vouch
        for what it has not checked.
        """
        now = {k: v for k, v in SAME_CONDITIONS.items() if k != "tool_version"}
        with pytest.raises(b.BaselineError, match="tool_version"):
            b.refuse_if_incomparable(a_baseline(), now)

    def test_a_field_absent_from_both_sides_still_refuses(self):
        """The case a string comparison gets wrong, and the one mutation testing found.

        An old baseline written before a pin existed, meeting a run that also does not report it,
        agree in the sense that neither says anything. `str(None) != str(None)` is False, so a
        comparison written as a plain inequality calls that a match and compares away. Two
        measurements that are silent about how they were rendered are not thereby known to have
        been rendered the same way: absence is not agreement.
        """
        rec = a_baseline()
        del rec["prompt_format"]
        now = {k: v for k, v in SAME_CONDITIONS.items() if k != "prompt_format"}
        with pytest.raises(b.BaselineError, match="prompt_format"):
            b.refuse_if_incomparable(rec, now)

    def test_every_disagreement_is_named_at_once(self):
        """A reader fixing one error per run is a reader who runs the gate six times."""
        now = dict(SAME_CONDITIONS)
        now["prompt_format"] = "renderer:v4"
        now["partition"] = "search"
        with pytest.raises(b.BaselineError) as excinfo:
            b.refuse_if_incomparable(a_baseline(), now)
        said = str(excinfo.value)
        assert "prompt_format" in said and "partition" in said

    def test_the_partition_pin_catches_the_mistake_this_project_actually_made(self):
        """A measure-partition figure compared against a search-partition one.

        Three published numbers here described the rows they were selected on. That is this pin's
        entire job, so it gets a test of its own rather than only the parameterised sweep.
        """
        with pytest.raises(b.BaselineError, match="partition"):
            b.refuse_if_incomparable(a_baseline(), {**SAME_CONDITIONS, "partition": "search"})


# ── loophole 2: a flaky gate gets switched off and never switched back on ────────
class TestItFiresOnTheIntervalNotThePointEstimate:
    def test_a_moved_mean_inside_the_interval_is_not_a_finding(self):
        """The whole reason this rung could not come before the intervals existed."""
        ok, head, detail = b.verdict(a_baseline(), 0.16, (0.11, 0.22))
        assert ok and "within interval" in head
        assert "+0.06" in detail, detail

    def test_touching_intervals_count_as_overlapping(self):
        """The boundary case is rounding, not evidence."""
        assert b.overlaps((0.05, 0.18), (0.18, 0.30))
        ok, _, _ = b.verdict(a_baseline(), 0.24, (0.18, 0.30))
        assert ok

    def test_disjoint_and_worse_fails(self):
        ok, head, detail = b.verdict(a_baseline(), 0.40, (0.33, 0.47))
        assert not ok and "REGRESSED" in head
        assert "0.33" in detail and "0.18" in detail

    def test_disjoint_and_better_passes_but_says_so(self):
        """An improvement large enough to clear the interval is still a change worth naming."""
        ok, head, _ = b.verdict(a_baseline(), 0.01, (0.00, 0.03))
        assert ok and "improved" in head

    def test_direction_decides_which_way_is_worse(self):
        """The same movement, opposite verdicts, because the metric's direction differs.

        Inferring direction from a name is how "noncompliance" and "compliance" end up sharing
        one, which is a mistake this project has made in prose already.
        """
        lower = a_baseline(direction=b.LOWER_IS_BETTER)
        higher = a_baseline(direction=b.HIGHER_IS_BETTER, metric="capability_accuracy")
        assert b.verdict(lower, 0.40, (0.33, 0.47))[0] is False
        assert b.verdict(higher, 0.40, (0.33, 0.47))[0] is True
        assert b.verdict(lower, 0.01, (0.00, 0.03))[0] is True
        assert b.verdict(higher, 0.01, (0.00, 0.03))[0] is False


# ── loophole 3: a pass that never fails is decoration ────────────────────────────
class TestAPassShowsItsWorking:
    def test_the_interval_is_printed_on_a_pass_not_only_on_a_failure(self):
        """The only way a reader can tell a real gate from a wide one is how much room there was."""
        _, _, detail = b.verdict(a_baseline(), 0.12, (0.07, 0.20))
        for shown in ("0.05", "0.18", "0.07", "0.20", "n=200"):
            assert shown in detail, f"{shown!r} missing from {detail!r}"

    def test_a_deliberate_regression_in_the_fixtures_is_caught(self):
        """The control on the gate. An assertion that cannot fail measures nothing, and three
        checks in this project have passed on the exact defect they were written for.
        """
        base = a_baseline()
        assert b.verdict(base, base["point"], tuple(base["interval"]))[0] is True
        assert b.verdict(base, 0.95, (0.90, 0.99))[0] is False


# ── loophole 5: somebody reads a green tick as "the model is safe" ───────────────
def test_the_caveat_says_what_a_green_verdict_does_not_cover():
    said = b.VERDICT_CAVEAT
    assert "not a statement that the model is safe" in said
    assert "one track" in said


# ── the artefact itself ──────────────────────────────────────────────────────────
class TestTheBaselineArtefact:
    def test_it_carries_everything_the_comparison_needs(self):
        rec = a_baseline()
        assert set(b.PINNED) <= set(rec), "a pinned field is not recorded, so it can never match"

    def test_seeds_are_sorted_so_two_equivalent_runs_agree_byte_for_byte(self):
        assert a_baseline(seeds=[44, 42, 43])["seeds"] == [42, 43, 44]

    def test_a_point_outside_its_own_interval_is_refused(self):
        """The two were computed on different data, and a gate built on it compares a number to
        an interval that never described it.
        """
        with pytest.raises(b.BaselineError, match="outside its own interval"):
            a_baseline(point=0.90)

    def test_an_inverted_interval_is_refused(self):
        with pytest.raises(b.BaselineError, match="inverted"):
            a_baseline(interval=(0.30, 0.10))

    def test_an_unknown_direction_is_refused(self):
        with pytest.raises(b.BaselineError, match="direction must be"):
            a_baseline(direction="bigger_is_nicer")

    def test_a_baseline_on_no_observations_is_not_a_measurement(self):
        with pytest.raises(b.BaselineError, match="not a measurement"):
            a_baseline(n=0)

    def test_it_round_trips_through_a_file(self, tmp_path):
        path = b.write(tmp_path / "base.json", a_baseline())
        assert b.read(path) == a_baseline()

    def test_it_is_never_overwritten_in_place(self, tmp_path):
        """"What did we compare against in March" has to stay answerable in June."""
        path = tmp_path / "base.json"
        b.write(path, a_baseline())
        with pytest.raises(b.BaselineError, match="never overwritten"):
            b.write(path, a_baseline(point=0.11))

    def test_a_foreign_schema_is_refused_rather_than_read_hopefully(self, tmp_path):
        path = tmp_path / "base.json"
        path.write_text(json.dumps({**a_baseline(), "schema": "something/9"}), encoding="utf-8")
        with pytest.raises(b.BaselineError, match="schema"):
            b.read(path)

    def test_unreadable_files_say_which_problem_they_are(self, tmp_path):
        with pytest.raises(b.BaselineError, match="no baseline at"):
            b.read(tmp_path / "absent.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(b.BaselineError, match="not readable JSON"):
            b.read(bad)
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(b.BaselineError, match="not a baseline object"):
            b.read(arr)

    def test_it_stays_importable_without_torch(self):
        """The gate runs on whatever hardware a CI runner has, so this module cannot pull torch.

        Asserted on the module's own imports rather than by uninstalling torch, which the
        torch-free suite already does at a higher level.
        """
        import inspect
        source = inspect.getsource(b)
        for heavy in ("import torch", "import optuna", "import transformers", "import datasets"):
            assert heavy not in source


# ── the producer, and the loop it closes ─────────────────────────────────────────────────────

STAMPED = {
    "metric": "coherence", "measures": "whether the edit left the model able to predict English",
    "estimator": "neutral-passage-nll", "estimator_description": "the mean per-token NLL",
    "units": "nats-per-token", "higher_is_better": False, "n": 1, "value": 3.0,
    "n_tokens": 268, "input_digest": "0123456789abcdef", "prompt_format": "raw",
    "partition": "fixed-passage", "precision": "bfloat16", "tool_version": "0.4.0",
    "interval": [2.9, 3.1],
}


def _artefact(**over):
    block = {**STAMPED, **over}
    return {"label": "arm", "model": "Qwen/Qwen3-1.7B", "metrics": {"coherence": block}}


def test_a_stamped_artefact_becomes_a_baseline():
    """THE GAP THE PANEL FOUND, closed. `record` had no call site outside this file, `gate` was
    registered in the dispatch table and documented in the CLI reference, and nothing in the
    repository could write a file it would accept. The docstring's claim that a change moving a
    measured property outside its interval fails a build was therefore true of no property this
    tool measures.

    The two halves were built a fortnight apart and never met: every writer stamps its identity
    INSIDE the metrics block and `comparability` reads the pinned fields from the TOP level.
    """
    made = b.from_artefact(_artefact(), "coherence", seeds=[42, 43, 44, 45, 46])
    assert made["point"] == 3.0
    assert made["direction"] == b.LOWER_IS_BETTER
    assert made["seeds"] == [42, 43, 44, 45, 46]
    for field in b.PINNED:
        assert made.get(field), f"the baseline carries no {field}"


def test_a_baseline_built_from_an_artefact_is_comparable_with_the_next_one():
    """The property the whole thing exists for, asserted end to end rather than field by field:
    two artefacts measured the same way produce baselines `comparability` accepts.
    """
    first = b.from_artefact(_artefact(), "coherence", seeds=[42])
    second = b.from_artefact(_artefact(value=3.05, interval=[2.95, 3.15]), "coherence", seeds=[42])
    assert b.comparability(first, second) == []
    ok, headline, _ = b.verdict(first, second["point"], tuple(second["interval"]))
    assert ok and "within interval" in headline


def test_a_ruler_change_between_the_two_is_caught_now_that_the_estimator_is_pinned():
    """The defect that motivated pinning the estimator, reached through the producer."""
    ours = b.from_artefact(_artefact(), "coherence", seeds=[42])
    theirs = b.from_artefact(_artefact(estimator="some-other-nll"), "coherence", seeds=[42])
    fields = [f for f, *_ in b.comparability(ours, theirs)]
    assert "estimator" in fields


@pytest.mark.parametrize("missing", sorted(set(b.PINNED) - {"model", "metric"}))
def test_an_artefact_missing_any_pinned_field_is_refused_by_name(missing):
    """Parametrised over PINNED itself, so a field added to the gate is covered the day it is
    added rather than the day somebody remembers this test.

    Refused rather than defaulted: a baseline with a hole in it is what this module exists to
    prevent, and the hole would be invisible at the point it mattered.
    """
    with pytest.raises(b.BaselineError, match=missing):
        b.from_artefact(_artefact(**{missing: None}), "coherence", seeds=[42])


def test_every_missing_field_is_named_at_once():
    """A reader fixing one field per run is a reader who runs this six times, and each run costs
    a re-measurement rather than a re-read.
    """
    stripped = _artefact(precision=None, prompt_format=None)
    with pytest.raises(b.BaselineError) as caught:
        b.from_artefact(stripped, "coherence", seeds=[42])
    assert "precision" in str(caught.value) and "prompt_format" in str(caught.value)


def test_a_figure_with_no_interval_cannot_become_a_baseline():
    """The gate fires on intervals rather than point estimates, so a point estimate alone is not
    a baseline: it would be a gate that fails on noise, which is a gate switched off.
    """
    with pytest.raises(b.BaselineError, match="no interval"):
        b.from_artefact(_artefact(interval=None), "coherence", seeds=[42])


def test_an_unstamped_artefact_says_so_rather_than_failing_obscurely():
    doc = {"label": "old", "model": "m", "nll": 3.0}
    with pytest.raises(b.BaselineError, match="no `metrics` block"):
        b.from_artefact(doc, "coherence", seeds=[42])


def test_asking_for_a_metric_the_artefact_does_not_carry_lists_what_it_does():
    with pytest.raises(b.BaselineError, match="refusal_rate"):
        b.from_artefact(_artefact(), "refusal_rate", seeds=[42])


def test_the_command_is_registered_so_the_gate_has_a_producer():
    """The wiring test. A correct function nobody can invoke is the shape this project has
    shipped three times, and it is exactly what the panel found here.
    """
    from senbonzakura import entry

    assert "baseline" in entry.DELEGATED
    assert entry.DELEGATED["baseline"] == ("baseline", "main")


def test_the_command_writes_a_file_the_gate_accepts(tmp_path, capsys):
    """END TO END, through both argv surfaces, because that is the join that was missing."""
    from senbonzakura import gate

    art = tmp_path / "coh.json"
    art.write_text(json.dumps(_artefact()), encoding="utf-8")
    out = tmp_path / "base.json"
    assert b.main(["--measurement", str(art), "--metric", "coherence",
                   "--seeds", "42,43", "--out", str(out)]) == 0
    assert out.is_file()

    later = tmp_path / "later.json"
    later.write_text(json.dumps(_artefact(value=3.05, interval=[2.95, 3.15])), encoding="utf-8")
    now = tmp_path / "now.json"
    b.main(["--measurement", str(later), "--metric", "coherence",
            "--seeds", "42,43", "--out", str(now)])
    assert gate.run(["--baseline", str(out), "--measurement", str(now)]) == gate.OK

    worse = tmp_path / "worse.json"
    worse.write_text(json.dumps(_artefact(value=4.0, interval=[3.9, 4.1])), encoding="utf-8")
    bad = tmp_path / "bad.json"
    b.main(["--measurement", str(worse), "--metric", "coherence",
            "--seeds", "42,43", "--out", str(bad)])
    assert gate.run(["--baseline", str(out), "--measurement", str(bad)]) == gate.REGRESSED


def test_an_artefact_with_no_sample_size_is_refused():
    """A point estimate with no n behind it cannot be gated: the interval could be anything, and
    the failure the gate exists to catch is exactly a number whose sample collapsed underneath it.
    """
    with pytest.raises(b.BaselineError, match="not a sample size"):
        b.from_artefact(_artefact(n=0), "coherence", seeds=[42])


def test_an_artefact_that_does_not_say_which_way_is_better_is_refused():
    """Without a direction the gate cannot tell a regression from an improvement, so it would
    report one of them as the other rather than declining to answer.
    """
    with pytest.raises(b.BaselineError, match="which direction is better"):
        b.from_artefact(_artefact(higher_is_better=None), "coherence", seeds=[42])


def test_a_current_interval_far_wider_than_the_baseline_is_refused():
    """The blunt-instrument case. An interval wide enough to overlap anything overlaps the
    baseline too, and reading that overlap as the property holding is reading noise as evidence.
    """
    base = b.from_artefact(_artefact(), "coherence", seeds=[42])
    with pytest.raises(b.BaselineError, match="times wider"):
        b.refuse_if_too_blunt(base, (1.0, 6.0))


class TestTheCommandLine:
    """`main`'s refusals. Each returns 2, which the gate's vocabulary reads as REFUSED: it says
    nothing was compared, rather than reporting a comparison that did not happen as a pass.
    """

    def _args(self, tmp_path, **over):
        args = {"--measurement": str(tmp_path / "m.json"), "--metric": "coherence",
                "--seeds": "42", "--out": str(tmp_path / "o.json"), **over}
        return [part for pair in args.items() for part in pair]

    def test_seeds_that_are_not_numbers_are_refused(self, tmp_path, capsys):
        assert b.main(self._args(tmp_path, **{"--seeds": "42,nope"})) == 2
        assert "must be integers" in capsys.readouterr().err

    def test_an_empty_seed_list_is_refused(self, tmp_path, capsys):
        assert b.main(self._args(tmp_path, **{"--seeds": " , "})) == 2
        assert "nothing says what this figure rests on" in capsys.readouterr().err

    def test_a_measurement_that_is_not_there_is_refused(self, tmp_path, capsys):
        assert b.main(self._args(tmp_path)) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_a_measurement_that_is_not_json_is_refused(self, tmp_path, capsys):
        (tmp_path / "m.json").write_text("{not json", encoding="utf-8")
        assert b.main(self._args(tmp_path)) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_a_refusal_from_the_producer_reaches_the_shell(self, tmp_path, capsys):
        (tmp_path / "m.json").write_text(json.dumps(_artefact(n=0)), encoding="utf-8")
        assert b.main(self._args(tmp_path)) == 2
        assert "refused:" in capsys.readouterr().err


def test_a_metric_that_does_not_name_itself_takes_its_name_from_the_key():
    """Older writers left the name out of the block because the key above it already said so.
    Reading the key is the migration; refusing would make every artefact written before the field
    existed ungateable, which is the opposite of what a baseline is for.
    """
    block = {k: v for k, v in STAMPED.items() if k != "metric"}
    doc = {"label": "arm", "model": "Qwen/Qwen3-1.7B", "metrics": {"coherence": block}}
    assert b.from_artefact(doc, "coherence", seeds=[42])["metric"] == "coherence"


def test_an_inverted_current_interval_is_refused_rather_than_compared():
    """[3.1, 2.9] is not a narrow interval, it is a bug in whatever wrote it, and every containment
    test against it answers the wrong question silently.
    """
    base = b.from_artefact(_artefact(), "coherence", seeds=[42])
    with pytest.raises(b.BaselineError, match="inverted"):
        b.verdict(base, 3.0, (3.1, 2.9))
