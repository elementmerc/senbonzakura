# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The capability probe that ships in the package, and the gate that is now on by default.

WHY THIS FILE EXISTS

Until 2026-09-22 `--capability-eval` defaulted to empty. Every other figure a run reports is a
refusal ruler or a distributional proxy, and none of them asks the model to do anything hard, so a
run could report a clean bake on a model that had quietly lost multi-step arithmetic. The gate
that mattered most was the one nobody switched on.

Turning it on is only worth anything if the probe is actually THERE, in an install, offline. The
failure being guarded against is the one this project shipped as 0.3.0 and found again the same
morning: a build from a clone had no `--track default`, so the tool installed, imported, answered
`--help` normally, and failed on the first real command. Making a gate default-on and then
shipping installs that cannot run it would reproduce that on the gate whose absence is hardest to
notice, because a missing capability number looks exactly like a model that did not lose anything.

So these tests assert the unglamorous half: the file is in the tree, it is not ignored, it parses,
every row can actually be graded, and the defaults point at it.
"""
import json
import pathlib
import subprocess

import pytest

from senbonzakura import capability

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: What the builder selected. A round number that is checked rather than described, because the
#: probe shrinking silently is one of the ways this measurement could quietly stop meaning
#: anything.
BUNDLED = 256


def test_the_probe_ships_in_the_package():
    assert capability.probe_is_available(), (
        f"{capability.probe_path()} is not there. A capability gate that is on by default and "
        f"cannot find its probe is the `--track default` defect on a harder-to-notice gate")


def test_the_probe_is_tracked_by_git_and_not_ignored():
    """THE ONE THAT WOULD HAVE CAUGHT THE IGNORE RULE.

    `src/senbonzakura/data/*` is gitignored, and a `*.jsonl` rule further down the file undid the
    first negation written for this file. Git applies the LAST matching rule, so a negation in the
    obvious place was silently overridden and the probe would simply not have been committed.
    Nothing but asking git would have said so.
    """
    if not (ROOT / ".git").exists():
        # The build box runs the suite from a synced copy with no git metadata. Skipping there is
        # honest; skipping everywhere would not be, and CI checks the repository out, so this
        # assertion still runs on the machine whose answer decides what gets published.
        pytest.skip("not a git checkout, so git cannot be asked what it tracks")
    rel = "src/senbonzakura/data/capability-gsm8k.jsonl"
    listed = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert listed.returncode == 0, (
        f"{rel} is not tracked by git. Check the ignore rules with `git check-ignore -v {rel}`: "
        f"a later rule can undo an earlier negation")


def test_every_bundled_row_can_actually_be_graded():
    """A row whose answer carries no gold marker grades as indeterminate for ever.

    That does not fail, it shrinks the probe silently, and a probe that quietly got smaller
    reports a rate with a denominator nobody was told about.
    """
    pairs = capability.load_probe()
    assert len(pairs) == BUNDLED
    task = capability.get_task("numeric")
    ungradeable = [q for q, a in pairs if task.reference(a) is None]
    assert not ungradeable, (
        f"{len(ungradeable)} bundled rows have no extractable reference answer, so they would "
        f"score indeterminate whatever the model said")


def test_the_probe_carries_questions_and_reference_answers_and_nothing_else():
    """No stray columns, and the names are ours rather than upstream's.

    `tools/ci/check_prompt_artefacts.py` is the real gate on what reaches a public tree; this is
    the cheap one that runs with the suite, on a file that is deliberately committed.
    """
    for line in capability.probe_path().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert set(row) == {capability.PROBE_QUESTION_KEY, capability.PROBE_REFERENCE_KEY}, (
                f"unexpected columns: {sorted(row)}")


def test_load_probe_respects_a_limit():
    assert len(capability.load_probe(7)) == 7


def test_a_missing_probe_raises_rather_than_returning_nothing(monkeypatch, tmp_path):
    """An empty probe would score every candidate identically, and a run would report that as no
    capability cost. The absent measurement must not be readable as a good result.
    """
    monkeypatch.setattr(capability, "probe_path", lambda: tmp_path / "gone.jsonl")
    assert not capability.probe_is_available()
    with pytest.raises(FileNotFoundError, match="bundled capability probe"):
        capability.load_probe()


def test_the_notice_names_the_source_and_prints_once():
    """Somebody who installs a package and runs a default has not read a dataset card."""
    capability._probe_state["notified"] = False
    lines = []
    capability.probe_notice(log=lines.append)
    assert any("GSM8K" in line and "MIT" in line for line in lines), lines
    before = len(lines)
    capability.probe_notice(log=lines.append)
    assert len(lines) == before, "the notice printed twice in one process"


# ── the defaults ─────────────────────────────────────────────────────────────────────────────

def test_the_capability_gate_is_on_by_default():
    """THE POINT OF THE WHOLE CHANGE, asserted where a future edit would have to argue with it."""
    from senbonzakura import parser
    args = parser.build_parser().parse_args(["--model", "x"])
    assert args.capability_eval == "bundled"
    assert args.capability_n == 200


def test_the_default_sample_is_large_enough_to_carry_the_effect():
    """200 is a decision, not a round number, and it is recorded here so it cannot drift quietly.

    The drop this class of edit is reported to cause is several accuracy points. Before and after
    are scored on the SAME items, so the comparison is paired, but the sample still has to be big
    enough that the figure resolves the effect rather than merely having a decimal point. A much
    smaller default would produce an error bar wider than the thing being measured, which reads
    like a measurement and is not one.
    """
    from senbonzakura import parser
    args = parser.build_parser().parse_args(["--model", "x"])
    from senbonzakura.metrics import MIN_REPORTABLE_N
    assert args.capability_n >= 100, (
        "the default capability sample has been reduced below what resolves the effect it exists "
        "to detect; if that is deliberate, state the new resolution here")
    # And the harder floor underneath the soft one. `reportable_rate` WITHHOLDS a rate whose
    # sample is below MIN_REPORTABLE_N, so a small default would not produce a weak capability
    # figure, it would produce none at all, and the run would report "not gradeable" however well
    # the model did. Measured on 2026-09-22: a 24-item probe graded 23 and still reported n/a.
    # Some items always come back indeterminate, so the default needs headroom over the floor
    # rather than merely clearing it.
    assert args.capability_n >= 2 * MIN_REPORTABLE_N, (
        f"the default sample is {args.capability_n} and the reporting floor is {MIN_REPORTABLE_N}. "
        f"Without headroom for indeterminate answers the probe can clear the floor before grading "
        f"and fall under it afterwards, which reports nothing")
    assert args.capability_n <= len(capability.load_probe()), (
        "the default asks for more items than the bundled probe holds, so every default run would "
        "warn about a shortfall")


def test_the_probe_can_be_turned_off_explicitly():
    from senbonzakura import parser
    args = parser.build_parser().parse_args(["--model", "x", "--capability-eval", ""])
    assert not args.capability_eval
