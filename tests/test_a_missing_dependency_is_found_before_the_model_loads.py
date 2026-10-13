# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A declared dependency behind a deferred import is refused before the weights load.

MEASURED 2026-10-11, running the README's own command on sixteen cores. `senbonzakura compass`
spent **3 minutes 45 seconds and 28 CPU-minutes** loading Qwen3-1.7B and scoring sixty prompts,
and then refused:

    senbonzakura: cannot run 'compass': it needs senbonzakura_check, which is not installed.

Every word of that is true and it arrived after all the work. The handler around `run` was doing
its job; nothing was asking the question early.

**Why nothing caught it.** `senbonzakura_check` owns the measurement stamp, so a command reaches
for it at the END of its run. Those imports are deferred into functions on purpose, to keep
`--help` fast, and `entry.dispatch` only sees imports that happen while a module is LOADING. So
the one declared, non-optional dependency that every measuring command needs was invisible to
both handlers until the moment it was wanted.

**Two designs were wrong before this one, and the reason is the interesting part.**

The first listed, in `entry`, which commands defer the import, and pre-flighted them before
dispatch. An AST walk written to verify that list showed the hand-written version was wrong in
both directions. Then it showed something better: *no* static walk gets this right. Direct imports
miss `leak`, which reaches the checker through `residualleak`. Transitive reachability sweeps in
`doctor`, which must run with nothing installed at all. Predicting which commands need it was the
wrong question.

The loader is the honest place, and it needs no prediction. Everything expensive is downstream of
it, every command that writes a stamped figure passes through it, and a command that never loads a
model never reaches it. That last part is asserted here too, because over-refusing would break the
one command whose whole job is to run on a broken install.
"""

import importlib.util
import time

import pytest

from senbonzakura import cli, doctor

DEP = cli.CHECKER


@pytest.fixture
def checker_missing(monkeypatch):
    """Make `find_spec` report the checker absent, without touching the real environment."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: None if n == DEP else real(n, *a, **k))


def test_the_loader_refuses_before_it_loads_anything(checker_missing):
    """The defect itself. The cost avoided is the whole load plus the scoring after it."""
    started = time.monotonic()
    with pytest.raises(SystemExit) as exit_info:
        cli.load_model_and_tokenizer("Qwen/Qwen3-1.7B", device="cpu")
    elapsed = time.monotonic() - started
    said = str(exit_info.value)
    assert DEP in said, f"the refusal does not name the missing package: {said}"
    assert "not installed" in said, f"the refusal does not say it is absent: {said}"
    assert "pip install senbonzakura" in said, f"the refusal does not say what to type: {said}"
    # Loose on purpose. What this pins is the difference between a `find_spec` call and fetching
    # or reading a multi-gigabyte checkpoint, which is orders of magnitude rather than per cent.
    assert elapsed < 2.0, (
        f"the refusal took {elapsed:.2f}s, so it is happening after work rather than before it")


def test_the_refusal_says_why_it_is_early(checker_missing):
    """The sentence a reader needs, which is not the same as the fact.

    "X is not installed" tells them what. The run they did not have to wait through is the thing
    that makes the message worth printing at all, and saying so is how the next person to move
    this check understands what it is for.
    """
    with pytest.raises(SystemExit) as exit_info:
        cli.load_model_and_tokenizer("Qwen/Qwen3-1.7B", device="cpu")
    said = str(exit_info.value)
    assert "before the model loads" in said or "Refusing now" in said, (
        f"the refusal does not say it came early, so a reader cannot tell it saved them "
        f"anything: {said}")


def test_doctor_still_runs_with_the_checker_absent(checker_missing):
    """The over-refusal this design had to avoid, and the reason a table was rejected.

    `doctor` exists to run on a broken install and say what is broken. A transitive-reachability
    pre-flight in `entry` swept it in, which would have made the one command for diagnosing a
    missing dependency the one command that refuses because of it. `doctor` never loads a model,
    so putting the check in the loader leaves it alone by construction rather than by exception.
    """
    checks = doctor.run_checks(log=lambda _m: None)
    status = doctor.report(checks, log=lambda _m: None, header=False)
    assert status in (doctor.OK, doctor.WARN, doctor.FAIL), status
    assert checks, "doctor produced no checks at all"


def test_the_present_case_costs_nothing_and_changes_nothing():
    """With the dependency there, the pre-flight is a no-op. It is a gate, not a step."""
    started = time.monotonic()
    cli._refuse_without_the_checker()
    assert time.monotonic() - started < 0.5, "the pre-flight is doing real work when it should not"


def test_the_checker_is_a_declared_dependency_and_not_an_extra():
    """The premise. Refusing on a missing OPTIONAL package would be the wrong call entirely.

    If this ever becomes an extra, the loader must stop refusing and start degrading, so the
    premise is asserted rather than assumed.

    Through `tomlread`, not `tomllib` directly. The first draft of this file imported `tomllib`,
    which is 3.11+, and `requires-python` is ">=3.10"; the suite would not have COLLECTED on the
    declared minimum interpreter. `test_no_test_imports_tomllib_directly` caught it, and its
    docstring records five earlier files making the same mistake, every one of them written on a
    machine where only new interpreters existed. This is the sixth, which is a good argument for
    the rule existing.
    """
    import pathlib

    from tomlread import tomllib

    root = pathlib.Path(cli.__file__).resolve().parents[2]
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = " ".join(meta["project"]["dependencies"])
    assert DEP.replace("_", "-") in declared, (
        f"{DEP} is not a hard dependency in pyproject.toml, so refusing a run over its absence is "
        f"no longer the right behaviour: it should degrade instead")
