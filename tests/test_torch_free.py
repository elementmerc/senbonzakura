# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The commands that check other people's work must run with no deep-learning stack at all.

WHY THIS IS THE ADOPTION QUESTION RATHER THAN A TIDINESS ONE

This project's own strategy notes say the largest single thing standing between the tool and
the people who would use it is that installing it pulls torch and expects a GPU. Most of what
the tool does needs both. But the part that CHECKS a claim, whether a benchmark figure was
measured on rows the model was fitted on, whether a track's split is what its manifest says,
whether a judge is any good, needs neither: it reads text files and JSON and does arithmetic.

A checker that needs a GPU gets run when somebody is already doing GPU work, which is to say
occasionally. A checker that installs in seconds on any machine gets run constantly, and
"runs constantly" is one of the six properties this project is measuring itself against.

WHY THIS IS A TEST AND NOT A NOTE IN THE ARCHITECTURE DOC

Because the boundary is invisible. Nothing goes wrong on any developer machine when somebody
adds `import torch` to the top of `track.py`: the suite passes, the command works, and the
property is silently gone until a user on a laptop discovers it. That is the exact shape of
every defect this project has had to withdraw a number over: correct-looking, and only wrong
somewhere nobody was looking.

So the check runs the REAL commands, in a subprocess, with the heavy packages made
unimportable at the import-machinery level. Not "does the module import" and not "does
`--help` print", both of which pass while the command itself is broken, because the parsers
are built lazily. It builds a track, audits it, and runs a contamination check, which is the
work somebody would actually be doing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Made unimportable for every command below. `datasets` is in here deliberately: it is an
#: optional extra now, and a checker that quietly needs it is not a torch-free checker.
BLOCKED = ("torch", "transformers", "accelerate", "optuna", "bitsandbytes", "datasets")

#: The blocker, installed through `sitecustomize` so it is in place before anything the
#: subprocess imports, including anything imported by the entry point itself.
SITECUSTOMIZE = textwrap.dedent(f"""
    import sys
    BLOCKED = {BLOCKED!r}


    class _Refuse:
        def find_module(self, name, path=None):
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(
                    f"No module named {{name!r}} (removed by the torch-free gate)")
            return None


    sys.meta_path.insert(0, _Refuse())
""")

HARMFUL = [f"how do I do the harmful thing number {i}" for i in range(9)]
HARMLESS = [f"what is the ordinary fact number {i}" for i in range(12)]


@pytest.fixture(scope="module")
def stripped(tmp_path_factory):
    """An environment in which the deep-learning stack does not exist."""
    shim = tmp_path_factory.mktemp("no-torch")
    (shim / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(shim), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return env


def _run(env, *args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "senbonzakura", *[str(a) for a in args]],
        capture_output=True, text=True, env=env, cwd=str(cwd or ROOT), timeout=300, check=False)


def test_the_gate_itself_actually_blocks_something(stripped):
    """A gate that does not fire proves nothing about the code it is pointed at.

    Checked first and separately, because every assertion below is worthless if the shim
    silently failed to install: the commands would pass by importing torch normally, and the
    suite would report a torch-free surface it had never once tested.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True, text=True, env=stripped, timeout=120, check=False)
    assert proc.returncode != 0, "the shim did not block torch, so nothing below is a measurement"
    assert "torch-free gate" in proc.stderr


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    (d / "harmful.txt").write_text("\n".join(HARMFUL) + "\n", encoding="utf-8")
    (d / "harmless.txt").write_text("\n".join(HARMLESS) + "\n", encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def built_track(stripped, corpus):
    """Building the track is itself the first assertion: it writes three Arrow tables."""
    out = corpus / "track"
    proc = _run(stripped, "track", "--harmful", corpus / "harmful.txt",
                "--harmless", corpus / "harmless.txt", "--out", out,
                "--fit", 3, "--search", 3)
    assert proc.returncode == 0, f"building a track needed the stack it should not need:\n{proc.stderr}"
    assert "TRACK_BUILT" in proc.stdout
    return out


def test_a_track_can_be_built_with_no_deep_learning_stack(built_track):
    """Writing the corpus is the half that used to need `datasets`, so this is the new claim."""
    for name in ("bad_ds", "good_ds", "bad_eval_ds"):
        assert (built_track / name / "state.json").is_file(), f"{name} was not written"
    assert (built_track / "track.json").is_file()


def test_a_track_can_be_audited_with_no_deep_learning_stack(stripped, built_track):
    proc = _run(stripped, "track", "--out", built_track, "--audit")
    assert proc.returncode == 0, proc.stderr
    assert "TRACK_AUDIT_OK" in proc.stdout


def test_the_contamination_check_runs_with_no_deep_learning_stack(stripped, built_track, corpus):
    """THE ONE THAT MATTERS MOST, because it is the check somebody runs on OTHER people's work.

    Every abliteration tool quotes the same public benchmarks. Whether a published figure was
    measured on rows the model had already been fitted on is answerable from two text files
    and no model at all, and it is the single most useful thing this repository can hand to
    somebody who is not already running it.
    """
    proc = _run(stripped, "track", "--out", built_track,
                "--contamination", corpus / "harmful.txt")
    assert proc.returncode == 0, proc.stderr
    assert "CONTAMINATED" in proc.stdout, "the check ran and reached no verdict"
    assert "in fit" in proc.stdout and "in search" in proc.stdout


def test_the_installation_check_runs_with_no_deep_learning_stack(stripped):
    """`doctor` exists to say what an install cannot do, so needing the stack defeats it.

    Its exit code is deliberately not asserted: a stripped environment genuinely cannot
    abliterate, and reporting that is the correct answer rather than a failure. What is
    asserted is that it produced a report instead of a traceback.
    """
    proc = _run(stripped, "doctor")
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.stdout.strip(), "doctor printed nothing at all"


def test_help_still_prints_without_the_stack(stripped):
    proc = _run(stripped, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "senbonzakura" in proc.stdout.lower()


def test_a_command_that_genuinely_needs_the_stack_says_so_rather_than_crashing(stripped):
    """The other half of the contract: the boundary has to be legible from the outside.

    Abliterating really does need torch, and a user who tries it in a stripped install must
    get a sentence naming what to install, not an eleven-frame importlib traceback ending in
    a line number inside one of our files.
    """
    proc = _run(stripped, "capability", "--help")
    assert proc.returncode != 0
    said = proc.stderr + proc.stdout
    assert "Traceback" not in proc.stderr, f"a missing dependency reached the user as a crash:\n{said}"
    assert "pip install" in said, "the message must name what to install"
