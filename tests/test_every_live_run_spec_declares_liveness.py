# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A run spec that nothing can watch is a GPU job nobody can tell from a dead one.

WHAT PROMPTED IT, Q-81

Not one run spec in this project declared a liveness check, across every file in
`private/holst/`, so `holst plan` warned on every job, including ones with four and eight hour
ceilings. The warnings had been there long enough to become scenery.

That is not a theoretical gap, and the two incidents it covers are different from each other:

  a run died at 07:01 with every arm complete, because the orchestrator kept only the HEAD of a
  job's output and the tail, where the failure was, had scrolled away;

  an experiment sat for 46 minutes at 0% CPU, which is a job that is parked rather than slow.

The second is exactly what `liveness.blocked_secs` discriminates, and the reason it is the check
rather than output silence is worth stating once: silence cannot tell a stuck job from a busy
quiet one, because a model loading its weights prints nothing for minutes. Silence WITH a flat
CPU can.

WHY THIS IS A TEST AND NOT A CI JOB, WHICH IS A LIMITATION AND NOT A PREFERENCE

`private/` is excluded from this repository's remotes, so a CI clone has no `private/holst/` and
no `holst plan` in CI can ever see a spec. The same constraint already shapes the mutation
ledger, which has to be carried between runs in a cache for the same reason. So the enforcement
point is the suite on the machine where the specs actually live, and in CI this file SKIPS. A
skip is not a pass and pytest prints it as a skip, which is the honest signal; what it is not is
protection for anybody but the operator. If that ever needs to be stronger the answer is the
pre-push hook, not a CI job, because the push is the boundary the files are on the right side of.

THE EXEMPTIONS ARE DERIVED, NOT LISTED, WITH ONE EXCEPTION THAT EXPIRES

A spec whose own header retires it is exempt, read from the file rather than from a list here.
That matters: a hand-maintained list of exempt filenames rots the moment somebody adds a spec,
and the derived rule means a NEW spec needs liveness automatically and an UN-RETIRED spec needs
it the moment the header comes off.

The one hand-written allowlist is the July compass runs, which are finished runs against a
rented pod that no longer exists and which carry no retirement header. Each entry says why, and
this file refuses an entry whose spec has gone or whose spec has since been retrofitted, so the
allowance cannot outlive what it was written for.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECS = ROOT / "private" / "holst"

#: Long enough for a planner that only reads a file, short enough that a wedge is a failure.
TIMEOUT_S = 60

#: Both phrases have to be present for a header to count as a retirement. The word on its own is
#: not enough: a spec can mention that it SUPERSEDED something else, and reading that as its own
#: retirement would exempt a live spec on a sentence about its predecessor.
RETIRED_MARKERS = ("superseded",)
DO_NOT_RUN_MARKERS = ("do not run", "do not re-run")

#: How many lines of the header are read. A retirement notice goes at the top, where the next
#: person to open the file meets it; a sentence 300 lines down is not a notice.
HEADER_LINES = 12

#: The one hand-written allowance, with the reason per entry, which the test below holds to
#: account. These are the July 2026 compass runs: each names a dated run id
#: (`compass-2026-07-27d` and siblings) against `machine = "rented"`, a RunPod pod that filled
#: its disk and is gone. Re-running one would re-run a completed, published July sweep on
#: hardware that does not exist, so retrofitting liveness into them buys nothing.
#:
#: WHAT IS ACTUALLY OWED HERE, and it is the operator's call rather than this file's: these six
#: are records in everything but their headers, and this project's convention is that a spec
#: which should not be re-run SAYS SO at the top. Six that do not is the gap. The moment a
#: header is added the entry below becomes stale and this test says so.
COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER = {
    "compass-benign.toml": "finished 2026-07-27 benign arm, rented pod, no header",
    "compass-benign-topup.toml": "finished top-up for the passes the pod lost, rented pod",
    "compass-last.toml": "finished proof of the setsid launch fix, rented pod",
    "compass-margin.toml": "finished criterion-free arm, rented pod",
    "compass-rerun.toml": "finished 2026-07-27d sweep, rented pod",
    "compass-topup.toml": "finished top-up for the three unbanked abliterations, rented pod",
}


def _specs():
    if not SPECS.is_dir():
        return []
    return sorted(SPECS.glob("*.toml"))


def _header(path):
    lines = path.read_text(encoding="utf-8").splitlines()[:HEADER_LINES]
    return "\n".join(lines).lower()


def _is_retired(path):
    header = _header(path)
    return (any(m in header for m in RETIRED_MARKERS)
            and any(m in header for m in DO_NOT_RUN_MARKERS))


def plan_warnings(path, *, holst="holst"):
    """Every `warning:` line `holst plan` emits for this spec, as a list.

    Returns the lines rather than a count, because a gate that says "3 warnings" and makes
    somebody re-run the command by hand to see them is a gate they will route around.
    """
    proc = subprocess.run([holst, "plan", str(path)], capture_output=True, text=True,
                          timeout=TIMEOUT_S, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"`holst plan {path.name}` exited {proc.returncode}, so whether the spec declares a "
            f"liveness check is unknown. That is DID NOT RUN, not a pass.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    merged = proc.stdout + proc.stderr
    return [line for line in merged.splitlines() if line.startswith("warning:")]


def _live_specs():
    return [p for p in _specs()
            if not _is_retired(p) and p.name not in COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER]


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────

def test_there_are_specs_to_check_at_all():
    """Without this, every test below passes on an empty glob.

    The skip is the honest outcome in CI, where `private/` does not exist, and it is deliberately
    not a pass: a check with nothing to check reporting clean is the single most recurring defect
    shape in this project.
    """
    if not SPECS.is_dir():
        pytest.skip("no private/holst in this checkout, which is every clone and all of CI")
    assert _specs(), (
        f"{SPECS} exists and holds no .toml spec at all. Either the layout moved, in which case "
        f"this gate is pointed at nothing, or every spec was deleted.")


def test_at_least_one_spec_is_live_and_therefore_gated():
    """An exemption rule that happened to exempt everything would leave the gate guarding nothing."""
    if not SPECS.is_dir():
        pytest.skip("no private/holst in this checkout")
    live = _live_specs()
    assert live, (
        "every spec in private/holst is either retired by its own header or on the compass "
        "allowlist, so this gate currently checks nothing. That is a reason to look at the "
        "exemption rules, not a pass.")


@pytest.mark.parametrize("name", [p.name for p in _live_specs()] or ["<none>"])
def test_a_live_spec_plans_without_a_liveness_warning(name):
    """THE GATE. A spec nothing can watch is a GPU job nobody can tell from a dead one."""
    if not SPECS.is_dir():
        pytest.skip("no private/holst in this checkout")
    if shutil.which("holst") is None:
        pytest.skip("holst is not on PATH, so no spec could be planned")
    if name == "<none>":
        pytest.skip("no live specs were collected")

    warnings = plan_warnings(SPECS / name)
    assert not warnings, (
        f"{name} plans with {len(warnings)} liveness warning(s), so at least one of its jobs "
        f"cannot be told apart from a dead one until its ceiling expires:\n"
        + "\n".join(f"  {w}" for w in warnings)
        + "\n\nAdd a `[job.liveness]` block with `blocked_secs` to each job named above. A flat "
          "CPU is what distinguishes a parked job from a quiet busy one; output silence cannot. "
          "`private/holst/validate-e4-qwen3-1.7b.toml` is the shape to copy. If the spec should "
          "not be run again, say so in its header instead: a retirement notice at the top exempts "
          "it from this gate and tells the next reader the same thing.")


# ── the allowlist cannot outlive its reason ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER))
def test_each_compass_allowance_still_names_a_spec_that_needs_it(name):
    """Three ways an allowance expires, and each is its own failure.

    The spec is gone, so the entry is dead text that hides the next one. The spec has acquired a
    retirement header, so the derived rule already covers it and the entry is duplicate. Or the
    spec has been retrofitted and plans clean, so there is nothing left to allow.
    """
    if not SPECS.is_dir():
        pytest.skip("no private/holst in this checkout")
    path = SPECS / name
    reason = COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER[name]

    assert path.is_file(), (
        f"{name} is on the compass allowlist with the reason '{reason}' and no longer exists. "
        f"Remove the entry.")
    assert not _is_retired(path), (
        f"{name} now carries a retirement header, so the derived exemption already covers it and "
        f"this entry is a second answer to the same question. Remove the entry.")

    if shutil.which("holst") is None:
        pytest.skip("holst is not on PATH, so the spec could not be planned")
    assert plan_warnings(path), (
        f"{name} now plans with no liveness warnings, so it has been retrofitted and needs no "
        f"allowance. Remove the entry and let the gate cover it.")


@pytest.mark.parametrize("name", sorted(COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER))
def test_each_compass_allowance_carries_a_written_reason(name):
    """An allowance nobody has to justify is how a known gap becomes a permanent one."""
    reason = COMPASS_RUNS_WITHOUT_A_RETIREMENT_HEADER[name]
    assert len(reason.split()) >= 4, (
        f"the allowlist entry for {name} says only {reason!r}. An entry needs a reason a reader "
        f"can evaluate, not a label.")


# ── the gate's own negative control ──────────────────────────────────────────────────────────────

def test_the_gate_fails_on_a_spec_with_no_liveness_block(tmp_path):
    """Prove the gate FAILS on a planted defect before believing it passes on anything.

    This is the house rule that exists because a check which answers a narrower question than the
    one asked reports clean. The question here is "does `holst plan` warn", and the only way to
    know the predicate can answer it is to hand it a spec that should warn.
    """
    if shutil.which("holst") is None:
        pytest.skip("holst is not on PATH, so the negative control cannot be run")

    spec = tmp_path / "no-liveness.toml"
    spec.write_text(
        'run = "gate-negative-control"\n'
        "\n"
        "[[job]]\n"
        'id = "long"\n'
        'machine = "local"\n'
        "timeout_secs = 28800\n"
        'success = { type = "stdout-contains", needle = "OK" }\n'
        'command = """\n'
        "echo OK\n"
        '"""\n',
        encoding="utf-8",
    )
    warnings = plan_warnings(spec)
    assert warnings, (
        "an eight-hour job with no liveness block produced no warning from `holst plan`, so this "
        "gate's predicate cannot detect the thing it was written for. Either holst changed or the "
        "predicate is reading the wrong stream, and in both cases every green above is "
        "meaningless.")


def test_the_gate_passes_on_a_spec_that_declares_liveness(tmp_path):
    """The positive control, so the predicate is not simply "always warns"."""
    if shutil.which("holst") is None:
        pytest.skip("holst is not on PATH, so the positive control cannot be run")

    spec = tmp_path / "with-liveness.toml"
    spec.write_text(
        'run = "gate-positive-control"\n'
        "\n"
        "[[job]]\n"
        'id = "long"\n'
        'machine = "local"\n'
        "timeout_secs = 28800\n"
        'success = { type = "stdout-contains", needle = "OK" }\n'
        'command = """\n'
        "echo OK\n"
        '"""\n'
        "\n"
        "[job.liveness]\n"
        "blocked_secs = 1800\n"
        'abort_on = ["Traceback (most recent call last)"]\n'
        "\n"
        "[job.liveness.progress]\n"
        'marker = "OK"\n'
        "expect = 1\n",
        encoding="utf-8",
    )
    assert not plan_warnings(spec), (
        "a spec that declares blocked_secs still warns, so the gate would be red on every "
        "correctly written spec and would be switched off within the week.")


# ── the retirement rule itself ───────────────────────────────────────────────────────────────────

def test_a_header_mentioning_a_superseded_predecessor_is_not_read_as_a_retirement(tmp_path):
    """The exemption has to be narrow or it exempts live specs by accident.

    A live spec may well open by saying it SUPERSEDED something else. Reading that as its own
    retirement would be the gate quietly letting the most interesting file through.
    """
    spec = tmp_path / "live.toml"
    spec.write_text("# Supersedes the July sweep, whose artefacts are gone.\n"
                    'run = "x"\n', encoding="utf-8")
    assert not _is_retired(spec)


def test_a_real_retirement_header_is_recognised(tmp_path):
    """And the other side, in the exact words the retired specs use."""
    spec = tmp_path / "dead.toml"
    spec.write_text("# SUPERSEDED 2026-08-05. Do not re-run this spec to produce numbers.\n"
                    'run = "x"\n', encoding="utf-8")
    assert _is_retired(spec)


def test_a_retirement_notice_buried_below_the_header_does_not_count(tmp_path):
    """A notice the next reader will not meet is not a notice."""
    spec = tmp_path / "buried.toml"
    spec.write_text("\n".join(["# padding"] * 40)
                    + "\n# SUPERSEDED. Do not run this file.\n"
                    + 'run = "x"\n', encoding="utf-8")
    assert not _is_retired(spec)
