# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A stamp that is generated, gated on at release, and read by nothing.

WHAT PROMPTED IT, 2026-09-27

`setup.py` writes `src/senbonzakura/_build.py` at build time carrying the commit the wheel was cut
from. `tools/ci/check_wheel.py` refuses a release wheel that lacks it or carries a placeholder. And
`crashsafe.git_commit`, which every result artefact's provenance goes through, tried git, then a
tarball stamp file, then an environment variable, and never imported it.

So the one situation the stamp exists for, an installed wheel with no `.git` anywhere, recorded
`"git": null` on every artefact it produced, while the commit sat in a module beside it. The
function's own docstring names the failure it was written to prevent: "a night of GPU runs produced
artefacts with no commit at all".

WHY THE FIX WAITED, which is the more useful half of this file

A blast-radius pass found that fixing it arms a trap elsewhere. `tools/ci/artefact_ok.py` decides
whether a run is finished by matching keys in its artefact, and
`scripts/runpod/seed-sweep-bootstrap.sh` uses that to skip completed arms. A spec of the form
`provenance.senbonzakura.git.commit=null` would have passed for as long as the field was null and
become unsatisfiable for ever the day it was not, at which point a finished arm reads as stale and
spends its GPU hours again, on a pod, silently.

No such spec was ever written. The rule is now enforced rather than remembered, and the enforcement
is deliberately scoped: see `test_a_deliberate_permanent_absence_is_still_allowed`.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import types

import pytest

from senbonzakura import crashsafe


def _the_ci_guard():
    """`tools/ci` is not a package, so its directory goes on the path before the guard imports.

    Through a function rather than a bare `sys.path` line followed by an import, because an import
    that has to come after a statement is an import the sorter will keep trying to move. The guard
    lives in `tools/ci` because it is a CI tool, and it is tested from here because this is the one
    place the rule it enforces and the code that rule protects are both in view.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "ci"))
    import artefact_ok

    return artefact_ok


artefact_ok = _the_ci_guard()


# ── the wheel's own stamp ──────────────────────────────────────────────────────────────────────────

def _no_git(monkeypatch):
    """Make git unanswerable, which is the state of every installed wheel."""
    def _boom(*_a, **_k):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", _boom)
    # The tarball stamp and the environment variable are the two weaker sources. Cleared so this
    # asserts the build stamp specifically rather than whichever source happens to answer first.
    monkeypatch.setattr(crashsafe, "COMMIT_STAMP_DIRS", ())
    monkeypatch.delenv(crashsafe.COMMIT_ENV, raising=False)


def _with_stamp(monkeypatch, value):
    """Stand a `_build` module up as an install would have one, and take it down again properly.

    BOTH BINDINGS, and the second one is the interesting half. `from ._build import COMMIT` also
    sets `_build` as an attribute on the `senbonzakura` package, and monkeypatch knows nothing about
    an attribute the import machinery created behind its back. Patching only `sys.modules` therefore
    leaked a stale module between tests: the empty-stamp cases left the package pointing at a module
    whose COMMIT was None, and the end-to-end test that ran after them read that and saw no commit.
    It passed on its own and failed in the file, which is the signature of exactly this.
    """
    import senbonzakura

    module = types.ModuleType("senbonzakura._build")
    module.COMMIT = value
    monkeypatch.setitem(sys.modules, "senbonzakura._build", module)
    monkeypatch.setattr(senbonzakura, "_build", module, raising=False)


def test_a_wheel_with_no_git_reports_the_commit_it_was_built_from(monkeypatch, tmp_path):
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, "deadbeefcafe1234deadbeefcafe1234deadbeef")
    got = crashsafe.git_commit(repo_root=tmp_path)
    assert got is not None, "an installed wheel still cannot say which commit produced its results"
    assert got["commit"] == "deadbee", (
        "the stamp holds forty characters and every artefact in this repository holds seven, and "
        "this field is compared as a string")
    assert got["source"] == "build", (
        "the source has to name the stamp, so a reader can weigh a claim differently from a "
        "measurement of the tree")


def test_the_build_stamp_does_not_claim_the_tree_was_clean(monkeypatch, tmp_path):
    """`dirty` is None, not False. "Not checked" and "checked and clean" are different facts.

    The build box's tree may well have been clean. Nothing here measured it, and a provenance field
    that quietly asserts more than was checked is the kind of thing this module exists to refuse.
    """
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, "0123456789abcdef0123456789abcdef01234567")
    assert crashsafe.git_commit(repo_root=tmp_path)["dirty"] is None


def test_a_checkout_still_prefers_git_over_the_stamp(monkeypatch):
    """Git is a measurement of the tree in front of it; the stamp is a claim about another machine.

    A developer with both must get the one that can tell them their tree is dirty, or the fix
    silently downgrades every artefact produced in a checkout.
    """
    # GIT'S AVAILABILITY IS ESTABLISHED FIRST, before the stamp exists. The obvious way to write
    # this was to inject the stamp and skip when the answer came back None, and it was wrong: on a
    # build mirror the stamp answers, so the result is never None and the test fails on a machine
    # where git legitimately cannot speak. A precondition has to be measured without the thing
    # whose precedence is being tested.
    if (crashsafe.git_commit() or {}).get("source") != "git":
        pytest.skip("not a git checkout here, so precedence between git and the stamp is unobservable")

    _with_stamp(monkeypatch, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    got = crashsafe.git_commit()
    assert got["source"] == "git", f"the stamp displaced a real measurement: {got}"
    assert got["commit"] != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_a_declared_commit_outranks_a_stamp_nobody_checked(monkeypatch, tmp_path):
    """The ordering the panel caught, and the reason the stamp is last rather than second.

    `_no_git` deliberately clears the two weaker sources so the tests above measure the stamp alone,
    which left the ORDER between them asserted by nothing: the stamp could sit anywhere among the
    three and every test in this file would still have passed.

    It was second for a day, on the argument that it describes THIS package rather than a directory
    the package happens to sit in. That is true of an installed wheel and false of the shape the GPU
    runs take, which is an rsync of the source tree with no `.git` and a `_build.py` left behind by
    an editable install weeks earlier. Ahead of the other two, that stale stamp silently replaced
    the commit the operator had just declared, in the one situation where the declaration is the only
    provenance there is.
    """
    def _boom(*_a, **_k):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", _boom)
    _with_stamp(monkeypatch, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    (tmp_path / crashsafe.COMMIT_STAMP_FILE).write_text("fromfile\n", encoding="utf-8")
    got = crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: "fromenv"})
    assert got == {"commit": "fromfile", "dirty": None, "source": "stamp"}, (
        f"a stale build stamp displaced the tarball's own record of what it is: {got}")

    (tmp_path / crashsafe.COMMIT_STAMP_FILE).unlink()
    got = crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: "fromenv"})
    assert got == {"commit": "fromenv", "dirty": None, "source": "declared"}, (
        f"a stale build stamp displaced a commit the operator declared at launch: {got}")


def test_the_stamp_still_answers_when_nothing_else_can(monkeypatch, tmp_path):
    """Putting the stamp last must not cost the case it was added for.

    An installed wheel has no `CODE_VERSION` beside it and no variable exported, so it reaches the
    stamp anyway. If this ever stops being true the original defect is back: every result an
    installed wheel produces recording `"git": null` while the commit sits in a module next to it.
    """
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, "1111111111111111111111111111111111111111")
    assert crashsafe.git_commit(repo_root=tmp_path, env={}) == {
        "commit": "1111111", "dirty": None, "source": "build"}


def test_the_absence_of_a_stamp_is_something_a_caller_can_state(monkeypatch, tmp_path):
    """`commit_from_this_build` is a named seam so a test can say "pretend this install has none".

    It exists because the stamp used to be read through an import inside `git_commit`, which nothing
    could reach. The tests for the three later sources passed on a development box only because an
    editable install had written `COMMIT = None`, and failed on every runner at once. Patching the
    seam has to actually silence the stamp, or that fix is decorative.
    """
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, "cccccccccccccccccccccccccccccccccccccccc")
    assert crashsafe.git_commit(repo_root=tmp_path, env={})["source"] == "build", (
        "the precondition failed: the stamp is not being read at all, so silencing it proves nothing")

    monkeypatch.setattr(crashsafe, "commit_from_this_build", lambda: None)
    assert crashsafe.git_commit(repo_root=tmp_path, env={}) is None, (
        "patching the seam did not silence the stamp, so the fixture in test_crashsafe.py that "
        "relies on it is decoration and those eight tests are back to depending on the machine")


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_stamp_is_no_answer_rather_than_a_blank_commit(monkeypatch, tmp_path, value):
    """A stamp written but not filled in must not produce `commit: ""`, which reads as a commit.

    `check_wheel` refuses a placeholder at release time, so this is the belt to that braces: an
    artefact is better off saying it does not know than carrying an empty string a reader will try
    to look up.
    """
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, value)
    assert crashsafe.git_commit(repo_root=tmp_path) is None


def test_provenance_carries_the_commit_through_to_the_artefact(monkeypatch, tmp_path):
    """END TO END, because the gap was exactly that two correct halves never met.

    `git_commit` returning the right thing is worth nothing if `provenance`, which is what actually
    reaches a result file, does not carry it.
    """
    _no_git(monkeypatch)
    _with_stamp(monkeypatch, "1234abcd1234abcd1234abcd1234abcd1234abcd")
    monkeypatch.chdir(tmp_path)
    got = crashsafe.provenance()["senbonzakura"]["git"]
    assert got and got["commit"] == "1234abc", (
        f"the artefact records a spelling no other artefact in this repository uses: {got}")
    assert len(got["commit"]) == crashsafe.SHORT_COMMIT_CHARS


# ── the trap the fix would have armed ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", [
    "provenance.senbonzakura.git.commit=null",
    "provenance.senbonzakura.git=None",
    "provenance.device=",
    "provenance.accelerator=nil",
])
def test_a_spec_asserting_a_provenance_field_is_absent_is_refused(spec):
    """It passes until the gap is filled and is unsatisfiable afterwards, which re-runs the work."""
    with pytest.raises(ValueError, match="absent"):
        artefact_ok.parse_expectations([spec])


def test_the_refusal_says_what_it_costs_and_what_to_do_instead():
    with pytest.raises(ValueError) as e:
        artefact_ok.parse_expectations(["provenance.senbonzakura.git.commit=null"])
    said = " ".join(str(e.value).split())
    assert "read as stale" in said, f"the refusal does not say what goes wrong: {said}"
    assert "what the artefact contains" in said, f"the refusal names no alternative: {said}"


def test_a_deliberate_permanent_absence_is_still_allowed():
    """THE SCOPE, asserted, because the first version of this guard had none.

    It refused every absence assertion, and `scripts/runpod/seed-sweep-bootstrap.sh` already
    contains `directions_from=None`, which is legitimate: the baseline arm is DEFINED by having no
    directions file, so that field will never acquire a value and the assertion cannot rot.
    Refusing it would have made a finished arm re-run on every check, which is the exact cost the
    guard exists to prevent, delivered by the guard.
    """
    assert artefact_ok.parse_expectations(["directions_from=None"]) == [("directions_from", "None")]


def test_the_live_spec_in_the_pod_bootstrap_still_parses():
    """Read from the script rather than retyped, so the two cannot drift apart silently."""
    script = (pathlib.Path(__file__).resolve().parent.parent
              / "scripts" / "runpod" / "seed-sweep-bootstrap.sh")
    if not script.is_file():
        pytest.skip(f"{script} is not in this checkout")
    text = script.read_text(encoding="utf-8")
    specs = re.findall(r"^\s+(?:\"\$[A-Z_]+/[^\"]+\"\s+)?((?:[\w.]+=\S+\s*)+)\\?$", text, re.MULTILINE)
    pairs = [tok for block in specs for tok in block.split()
             if "=" in tok and not tok.startswith("$")]
    assert pairs, "no expectation pairs were found in the bootstrap, so this checked nothing"
    cleaned = [p.replace('"$MODEL_ID"', "X").replace('"$S"', "42") for p in pairs]
    artefact_ok.parse_expectations([p for p in cleaned if not p.endswith("=")])
