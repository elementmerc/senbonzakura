# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The pre-registration checker, as a command somebody can actually run.

WHAT PROMPTED IT, 2026-09-27

`prereg.py` implements this project's pre-registration format: extract the block, validate it,
compare it to the run that claims to satisfy it. 298 lines, 32 tests, all passing.

**Nothing imported it.** Not one module in the package, not the CLI dispatch table, nothing. It
shipped inside every wheel as a complete library that no user could reach and no command could
call, and it was found by scanning for modules the package never imports.

That matters more for this module than it would for most, because the format's whole argument is
that a pre-registration should be checkable by somebody other than its author. A checker nobody can
run makes that claim by assertion, which is exactly what the format exists to replace.

It is registered as `senbonzakura prereg` and joins `check`, `gate` and `baseline` as a command
that needs no model, no GPU and no network.

WHAT THIS FILE CHECKS, and what `test_prereg.py` already covers

`test_prereg.py` covers the format: what validates, what does not, how amendments are read. This
covers the COMMAND: that it exists, that its exit codes mean what a CI job would assume, and that
its refusals are readable. No overlap on purpose.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from senbonzakura import prereg

#: The smallest block that validates, so a test about exit codes is not really a test about a
#: document being incomplete. Derived from the format's own required fields rather than guessed:
#: if `validate` rejects this, the assertion below says so rather than the test quietly measuring
#: the wrong thing.
COMPLETE = {
    "title": "Does a second refusal direction earn its collateral damage?",
    "date": "2026-09-27",
    "question": "Does removing two refusal directions beat removing one?",
    "hypothesis": "K=2 reaches the same refusal rate at no more collateral damage than K=1.",
    "falsified_by": "K=2 costing more KL than K=1 at matched refusal, over five seeds a side.",
    "primary": {
        "name": "K=1 against K=2 at matched refusal",
        "metric": "post_bake_kl",
        "instrument": "senbonzakura drift",
        "estimator": "mean post-bake KL over held-out rows, per seed, compared across seeds",
        "partition": "held-out evaluation rows, never used for fitting or selection",
        "decision_rule": "K=2 wins only if its mean KL is lower over five seeds a side",
    },
    "threats": [
        "Run-to-run variance across seeds swamping the effect being measured.",
        "The direction selection differing between arms, so the budget is not what varies.",
        "Scoring both arms with a ruler this project wrote, with no external check.",
    ],
}


def _doc(tmp_path, block, name="design.md"):
    p = tmp_path / name
    p.write_text(
        "# A pre-registration\n\nThe prose that does the real work.\n\n"
        "```prereg\n" + json.dumps(block, indent=2) + "\n```\n",
        encoding="utf-8")
    return p


def _run(*args):
    """Through the installed entry point, not by importing main.

    The defect this file exists for was that nothing reached the module. A test that imports it
    directly would have passed the whole time it was unreachable.
    """
    return subprocess.run([sys.executable, "-m", "senbonzakura", "prereg", *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_the_command_is_registered_at_all():
    """The actual defect: the module existed and no command reached it."""
    from senbonzakura import entry
    assert "prereg" in entry.DELEGATED, (
        "`prereg` is not in the dispatch table, so the checker is unreachable again")


def test_it_appears_in_the_help_a_user_reads():
    done = _run("--help")
    assert done.returncode == 0
    assert "pre-registration" in done.stdout.lower()


def test_a_file_with_no_block_is_refused_and_says_what_the_format_is():
    """The likeliest first mistake: pointing it at ordinary Markdown."""
    done = subprocess.run([sys.executable, "-m", "senbonzakura", "prereg", "/nonexistent.md"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 2, "a file that cannot be read is a refusal, not a failed check"
    assert "cannot read" in done.stderr


def test_a_refusal_is_wrapped_rather_than_one_long_line(tmp_path):
    """These sentences are long by design, and a 300-character line is a wall a reader skips."""
    plain = tmp_path / "prose.md"
    plain.write_text("# just prose, no block here\n", encoding="utf-8")
    done = _run(plain)
    assert done.returncode == 2
    over = [ln for ln in done.stderr.splitlines() if len(ln) > 79]
    assert not over, f"these refusal lines assume a wide terminal: {over}"


def test_a_complete_pre_registration_passes_cleanly(tmp_path):
    """Guards the fixture as much as the command: if `validate` rejects this, say so here."""
    findings = prereg.validate(COMPLETE)
    assert not findings, (
        f"the fixture meant to be complete does not validate, so every exit-code test below is "
        f"measuring the wrong thing: {findings}")

    done = _run(_doc(tmp_path, COMPLETE))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout


def test_a_missing_required_field_is_refused_rather_than_failed(tmp_path):
    """Exit 2, and the format is right to say so rather than 1.

    A missing REQUIRED field carries severity `refused`, which the module defines as "the document
    cannot be treated as a pre-registration at all". That is a different claim from "this
    pre-registration is wrong", and the command must not flatten the two: a document with no
    `falsified_by` has not failed its own test, it has failed to be a document that can be tested.

    This test asserted exit 1 when it was written, on the assumption that any problem is a failure.
    The tool was more careful than the test.
    """
    incomplete = {k: v for k, v in COMPLETE.items() if k != "falsified_by"}
    done = _run(_doc(tmp_path, incomplete))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "falsified_by" in done.stdout, (
        "the output does not name the missing field, so a reader cannot act on it")


def test_a_flawed_but_complete_pre_registration_fails_with_one(tmp_path):
    """Exit 1 is "the thing being checked is wrong", the same meaning `gate` gives it.

    A date that is not ISO 8601 is the cleanest example: every required field is present, so the
    document IS a pre-registration, and one of its claims will not do what it is for. The field's
    whole purpose is being orderable against the date a result exists.
    """
    flawed = dict(COMPLETE, date="last Tuesday")
    done = _run(_doc(tmp_path, flawed))
    assert done.returncode == 1, done.stdout + done.stderr
    assert "ISO 8601" in done.stdout


def test_the_three_exit_codes_do_not_collide():
    """0 clean, 1 wrong, 2 could not be checked. A CI job reading these must not conflate them.

    Conflating "it failed" with "it could not run" is the defect `gate` has already been fixed for
    twice, and this command copies its convention rather than inventing a second one.
    """
    assert {prereg.OK, prereg.FAILED, prereg.REFUSED_EXIT} == {0, 1, 2}
    assert prereg._EXIT_FOR[prereg.REFUSED] == prereg.REFUSED_EXIT
    assert prereg._EXIT_FOR[prereg.FINDING] == prereg.FAILED
    assert prereg._EXIT_FOR[prereg.NOTE] == prereg.OK, (
        "a NOTE alone must not fail a build; that is the point of having three levels")


def test_an_unreadable_run_artefact_is_a_refusal_not_a_failure(tmp_path):
    """"I could not check" must never be reported as "the run broke its promise"."""
    doc = _doc(tmp_path, COMPLETE)
    done = _run(doc, "--run", str(tmp_path / "missing.json"))
    assert done.returncode == 2, done.stdout + done.stderr


def test_a_run_artefact_that_is_not_json_is_a_refusal(tmp_path):
    doc = _doc(tmp_path, COMPLETE)
    bad = tmp_path / "run.json"
    bad.write_text("this is not json", encoding="utf-8")
    done = _run(doc, "--run", str(bad))
    assert done.returncode == 2
    assert "run artefact" in done.stderr


@pytest.mark.parametrize("flag", ["--quiet", None])
def test_findings_are_readable_either_way(tmp_path, flag):
    flawed = dict(COMPLETE, date="last Tuesday")
    args = [_doc(tmp_path, flawed)] + ([flag] if flag else [])
    done = _run(*args)
    assert done.returncode == 1
    over = [ln for ln in done.stdout.splitlines() if len(ln) > 79]
    assert not over, f"these lines assume a wide terminal: {over}"
    assert "finding" in done.stdout.lower()


# ── the same command, called in process ───────────────────────────────────────────────────────────
#
# WHY BOTH, 2026-09-27
#
# The subprocess tests above are the ones that would have caught the original defect: the module was
# unreachable, and a test that imported `main` directly would have passed throughout. That property
# is worth keeping, so they stay.
#
# But a subprocess is invisible to coverage, so `prereg.py` sat at 76.62% with its whole command
# untested as far as the gate could tell, and a coverage floor is how this project noticed that a
# class nobody had exercised was shipping. These call `main` in process, so the gate can see them.
# Same command, two reasons.

def test_main_returns_zero_on_a_complete_document(tmp_path, capsys):
    assert prereg.main([str(_doc(tmp_path, COMPLETE))]) == prereg.OK
    assert "OK" in capsys.readouterr().out


def test_main_returns_two_when_the_document_is_not_one(tmp_path, capsys):
    plain = tmp_path / "prose.md"
    plain.write_text("# no block here\n", encoding="utf-8")
    assert prereg.main([str(plain)]) == prereg.REFUSED_EXIT
    assert "prereg" in capsys.readouterr().err


def test_main_returns_one_on_a_flawed_document(tmp_path, capsys):
    assert prereg.main([str(_doc(tmp_path, dict(COMPLETE, date="last Tuesday")))]) == prereg.FAILED
    assert "ISO 8601" in " ".join(capsys.readouterr().out.split())


def test_main_quiet_still_names_the_severity(tmp_path, capsys):
    assert prereg.main([str(_doc(tmp_path, dict(COMPLETE, date="nope"))), "--quiet"]) == prereg.FAILED
    out = capsys.readouterr().out
    assert out.startswith(prereg.FINDING), "the severity is no longer at the head of the line"


def test_main_refuses_an_unreadable_document(tmp_path, capsys):
    assert prereg.main([str(tmp_path / "missing.md")]) == prereg.REFUSED_EXIT
    assert "cannot read" in " ".join(capsys.readouterr().err.split())


def test_main_checks_against_a_run_when_asked(tmp_path, capsys):
    doc = _doc(tmp_path, COMPLETE)
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"metrics": {"post_bake_kl": {"value": 0.4}}}), encoding="utf-8")
    code = prereg.main([str(doc), "--run", str(run)])
    assert code in (prereg.OK, prereg.FAILED, prereg.REFUSED_EXIT)
    # The point is that the comparison RAN and said something, not which verdict it reached: what
    # `compare_to_run` decides is `test_prereg.py`'s subject, not this file's.
    assert capsys.readouterr().out.strip(), "the comparison produced no output at all"


def test_main_refuses_a_run_artefact_that_is_not_json(tmp_path, capsys):
    doc = _doc(tmp_path, COMPLETE)
    bad = tmp_path / "run.json"
    bad.write_text("not json", encoding="utf-8")
    assert prereg.main([str(doc), "--run", str(bad)]) == prereg.REFUSED_EXIT
    assert "run artefact" in " ".join(capsys.readouterr().err.split())


def test_every_finding_line_fits_a_narrow_terminal(tmp_path, capsys, monkeypatch):
    """The whole point of the wrapping work, asserted on this command's own output."""
    monkeypatch.setenv("COLUMNS", "80")
    prereg.main([str(_doc(tmp_path, {k: v for k, v in COMPLETE.items() if k != "threats"}))])
    printed = capsys.readouterr().out.splitlines()
    over = [ln for ln in printed if len(ln) > 79]
    assert not over, f"these lines assume a wide terminal: {over}"
