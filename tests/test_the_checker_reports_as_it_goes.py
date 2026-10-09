# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A sweep that dies at four thousand files has still told you about the first four thousand.

The checker used to build a list of every file's result and render after the loop, so a run that
was killed partway reported nothing at all. That is not a performance question. One order of
magnitude past today is a repository of a few thousand result artefacts, or a CI run over a
monorepo of them, and the thing a reader needs most from a long sweep is the part that finished.

TWO PROPERTIES THAT PULL AGAINST EACH OTHER, which is why both are pinned here.

The output streams, so a partial run is useful. And the exit status is still decided at the end,
because `--pair` runs after the singles and one unreadable file in the last artefact changes what
the whole run means. A streaming report whose status was decided at artefact one is worse than an
accumulating one, because the output looks complete. So a partial report carries no footer and no
status line at all, and that absence is what tells a reader it is partial.

THE HEARTBEAT IS ON STDERR, and that is asserted rather than trusted. `--json` on stdout is read
by `action.yml` and by the pre-commit entry point, and a progress line in that stream turns a
clean run into a parse error on somebody else's CI. This project also has the opposite scar: a
quiet mode once suppressed warnings because a recorded decision's premise about which stream
carried what was false.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from senbonzakura_check import cli

ROOT = Path(__file__).resolve().parent.parent

GOOD = {
    "version": 2, "status": "success",
    "eval": {"task": "t", "model": "m"},
    "results": {"total_samples": 200, "completed_samples": 200,
                "scores": [{"name": "s", "scorer": "choice", "scored_samples": 200,
                            "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}]},
}


@pytest.fixture
def many(tmp_path):
    """Twelve recognisable artefacts, which is enough for "partway" to mean something."""
    for i in range(12):
        (tmp_path / f"run-{i:02d}.json").write_text(json.dumps(GOOD), encoding="utf-8")
    return tmp_path


def _run(*args, clock=None):
    out = io.StringIO()
    status = cli.main([str(a) for a in args], out=out, clock=clock)
    return status, out.getvalue()


class TestTheOutputArrivesBeforeTheEnd:
    def test_an_interrupted_sweep_has_already_reported_what_it_finished(self, many, monkeypatch):
        """Simulated in-process so it is deterministic: the real thing is below.

        The assertion is the one that matters either way. When the sweep stops at the fourth
        file, the first three are already in the reader's hands.
        """
        real = cli.inspect_file
        seen = []

        def give_up_on_the_fourth(path, checks, **kw):
            # `**kw` rather than a pinned signature: `inspect_file` gained an optional
            # `row_select` when the leaderboard adapter landed, and a double that hardcodes the
            # exact parameter list fails with a TypeError the moment the real function grows one.
            # The TypeError then surfaces here as "the run was not interrupted", which points at
            # the streaming behaviour this test is about rather than at the stub.
            seen.append(path)
            if len(seen) == 4:
                raise KeyboardInterrupt
            return real(path, checks, **kw)

        monkeypatch.setattr(cli, "inspect_file", give_up_on_the_fourth)
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            cli.main([str(many), "--json"], out=out)
        written = out.getvalue()
        assert written.count('"artefact"') == 3, (
            f"three files were finished and {written.count(chr(34) + 'artefact' + chr(34))} were "
            f"reported:\n{written}")

    def test_an_interrupted_sweep_carries_no_footer(self, many, monkeypatch):
        """Loophole 10. The footer and the status are the run's conclusion, and a partial run has
        not reached one. An output that looks complete is worse than one that stops mid sentence.
        """
        real = cli.inspect_file
        seen = []

        def give_up_on_the_fourth(path, checks, **kw):
            # `**kw` rather than a pinned signature: `inspect_file` gained an optional
            # `row_select` when the leaderboard adapter landed, and a double that hardcodes the
            # exact parameter list fails with a TypeError the moment the real function grows one.
            # The TypeError then surfaces here as "the run was not interrupted", which points at
            # the streaming behaviour this test is about rather than at the stub.
            seen.append(path)
            if len(seen) == 4:
                raise KeyboardInterrupt
            return real(path, checks, **kw)

        monkeypatch.setattr(cli, "inspect_file", give_up_on_the_fourth)
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            cli.main([str(many)], out=out)
        written = out.getvalue()
        assert "file(s)," not in written, f"a partial run printed its summary line:\n{written}"
        assert "It cannot tell you a number is right" not in written, (
            "a partial run printed the closing caveat, which reads as the end of a finished run")

    def test_an_interrupted_json_report_is_not_parseable(self, many, monkeypatch):
        """Which is the correct outcome, not a shortcoming.

        `action.yml` reads this report and a file it cannot parse is reported as "nothing about
        this run is known", never as a clean sweep. The closing bracket is written last and
        nowhere else, so a report that has one is a report that finished.
        """
        real = cli.inspect_file
        seen = []

        def give_up_on_the_fourth(path, checks, **kw):
            # `**kw` rather than a pinned signature: `inspect_file` gained an optional
            # `row_select` when the leaderboard adapter landed, and a double that hardcodes the
            # exact parameter list fails with a TypeError the moment the real function grows one.
            # The TypeError then surfaces here as "the run was not interrupted", which points at
            # the streaming behaviour this test is about rather than at the stub.
            seen.append(path)
            if len(seen) == 4:
                raise KeyboardInterrupt
            return real(path, checks, **kw)

        monkeypatch.setattr(cli, "inspect_file", give_up_on_the_fourth)
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            cli.main([str(many), "--json"], out=out)
        with pytest.raises(ValueError):
            json.loads(out.getvalue())

    def test_a_real_kill_leaves_a_partial_report_rather_than_an_empty_one(self, tmp_path):
        """The same property through a kill, because the version above stops at a Python
        exception and a killed process does not get to run anything at all.

        REWRITTEN 2026-10-07, and the two defects it had are worth naming because both are the
        kind that make a suite lie rather than fail.

        IT WAS FLAKY. It wrote the report to a file, polled every 50ms until the file passed 400
        bytes, then killed. On a warm machine the whole sweep finished inside the first poll, the
        kill landed after the run, and the assertion that the report was incomplete failed. The
        fix is not a longer wait or a bigger margin: it is to make finishing IMPOSSIBLE rather
        than unlikely. The report is ~1,080 bytes per artefact and nothing drains the pipe, so at
        400 artefacts the child must block on a full pipe buffer long before it is done. That is
        an operating-system guarantee, not a timing hope.

        IT COULD NEVER HAVE PASSED ON WINDOWS. `signal.SIGKILL` does not exist there, so this
        raised `AttributeError` on every Windows run of a suite that CI runs on Windows. The
        environment was POSIX-only too, a bare `PATH=/usr/bin:/bin`. `Popen.kill()` is the
        portable spelling and the environment is now inherited with only PYTHONPATH forced.

        There is no sleep and no poll left. `read(PROOF)` blocks until those bytes exist, and
        they must exist, so the sequencing is decided by the pipe rather than by the clock.
        """
        artefacts = 400
        for i in range(artefacts):
            (tmp_path / f"run-{i:03d}.json").write_text(json.dumps(GOOD), encoding="utf-8")

        #: Enough to prove the report had started arriving, and far less than one artefact's
        #: worth of slack against the total, so reading it cannot let the child finish.
        proof = 2048

        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "checker" / "src")
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-m", "senbonzakura_check.cli", str(tmp_path), "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(ROOT), env=env)
        # BOUNDED, because `read` blocks until the bytes arrive and a checker that started and
        # then said nothing would otherwise hang the suite rather than fail it. The original had
        # a deadline for this reason and the rewrite has to keep one. Killing the child closes
        # the pipe, so the read below returns short and the assertion speaks instead of the
        # wall clock.
        watchdog = threading.Timer(120, proc.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            head = proc.stdout.read(proof)
            assert len(head) == proof, (
                f"the checker produced only {len(head)} byte(s) before ending, so it either "
                f"failed to start or buffered its whole report. Either way nothing was streamed, "
                f"which is the defect this test exists for.")
            proc.kill()
            proc.wait(timeout=60)
            written = (head + proc.stdout.read()).decode("utf-8", errors="replace")
        finally:
            watchdog.cancel()
            proc.stdout.close()
            if proc.poll() is None:                 # pragma: no cover - only on an assert above
                proc.kill()
                proc.wait(timeout=60)

        assert '"artefact"' in written, (
            "a killed sweep had reported nothing at all, which is the defect this streams to fix")

        # COMPLETENESS IS WHETHER IT PARSES, not whether it ends with a bracket.
        #
        # This was `not written.rstrip().endswith("]")` and it failed once in a full suite run on
        # 2026-10-09, reporting that the whole sweep had fitted in the pipe buffer and advising
        # that `artefacts` be raised above 400. Both halves of that were wrong. The report WAS
        # truncated, to about 171 KB of 423 KB, and the kill had landed exactly where it was
        # supposed to; the cut simply happened to fall just after a nested findings list, so the
        # last character was a `]` indented two spaces rather than the top-level close.
        #
        # Measured rather than reasoned about: of 423 truncation points across a real 400-artefact
        # report, exactly ONE ends with `]` after rstrip, and it is the one the failure landed on.
        # A one-in-423 flake is why this passed three times alone and six times under load.
        #
        # `json.loads` has no such coincidence mode: a truncated array cannot parse, and a complete
        # one always does. The guard now measures the property it names.
        try:
            json.loads(written)
        except ValueError:
            pass        # truncated, which is the point of the test
        else:
            raise AssertionError(
                f"the report parsed as complete JSON, so the whole sweep fitted inside the pipe "
                f"buffer and this test measured nothing. Raise `artefacts` above {artefacts} "
                f"until it cannot. ({len(written)} bytes read.)")



class TestTheCompletenessCheckItself:
    """A guard on the guard above, because the guard above was wrong for a month.

    The test that proves a killed sweep leaves a PARTIAL report has to be able to tell partial
    from complete. It used `endswith("]")`, which a truncated report can satisfy by landing just
    after a nested list, and on 2026-10-09 one did. These two tests pin the distinction without
    spawning anything, so a future rewrite cannot quietly reintroduce the coincidence.
    """

    def _report(self, tmp_path, n=40):
        for i in range(n):
            (tmp_path / f"run-{i:03d}.json").write_text(json.dumps(GOOD), encoding="utf-8")
        _status, written = _run(tmp_path, "--json")
        return written

    def test_a_truncation_that_ends_in_a_nested_bracket_is_not_complete(self, tmp_path):
        """The exact shape that misfired: cut the report just after a nested findings list, so the
        last character is a `]`, and assert the check still calls it truncated.
        """
        full = self._report(tmp_path)
        cuts = [i for i in range(len(full) - 1, 0, -1) if full[:i].rstrip().endswith("]")]
        coincidences = [c for c in cuts if c < len(full.rstrip())]
        assert coincidences, (
            "no truncation point in this report ends with a bracket, so the coincidence this "
            "guards against cannot occur here and the test is measuring nothing. The report's "
            "shape has changed; find the new coincidence or delete this test with a reason.")
        cut = full[:coincidences[0]]
        assert cut.rstrip().endswith("]"), "the premise: the old check would have passed this"
        with pytest.raises(ValueError):
            json.loads(cut)

    def test_a_complete_report_parses(self, tmp_path):
        """The other side, so the check cannot pass everything by refusing everything."""
        json.loads(self._report(tmp_path))


class TestTheStatusIsStillDecidedAtTheEnd:
    def test_an_unreadable_file_at_the_end_still_sets_the_status(self, many):
        """The specific shape of loophole 10: everything before it was fine."""
        (many / "zz-broken.json").write_text("{not json", encoding="utf-8")
        status, written = _run(many, "--json")
        assert status == 2, written

    def test_a_finding_anywhere_sets_the_status(self, many):
        bad = json.loads(json.dumps(GOOD))
        bad["results"]["scores"][0]["scorer"] = None
        (many / "aa-bad.json").write_text(json.dumps(bad), encoding="utf-8")
        status, _written = _run(many, "--json")
        assert status == 1

    def test_the_footer_counts_every_file_and_not_only_the_last(self, many):
        status, written = _run(many)
        assert status == 0, written
        assert "12 file(s)" in written, written


class TestTheHeartbeat:
    def test_it_fires_on_a_long_sweep(self, many):
        """A clock that jumps a minute per file, so the interval is reached without waiting."""
        ticks = iter([i * 60 for i in range(100)])
        errors = io.StringIO()
        out = io.StringIO()
        monkey = cli._Heartbeat

        def heartbeat(total, **kwargs):
            kwargs["stream"] = errors
            return monkey(total, **kwargs)

        cli._Heartbeat = heartbeat
        try:
            cli.main([str(many), "--json"], out=out, clock=lambda: next(ticks))
        finally:
            cli._Heartbeat = monkey
        assert errors.getvalue().count("checked, now at") >= 5, errors.getvalue()
        assert "of 12 checked" in errors.getvalue(), errors.getvalue()

    def test_stdout_stays_parseable_while_it_fires(self, many):
        """The whole reason it is on stderr. A progress line in the report turns a clean run into
        a parse error on somebody else's CI.
        """
        ticks = iter([i * 60 for i in range(100)])
        errors = io.StringIO()
        out = io.StringIO()
        monkey = cli._Heartbeat

        def heartbeat(total, **kwargs):
            kwargs["stream"] = errors
            return monkey(total, **kwargs)

        cli._Heartbeat = heartbeat
        try:
            cli.main([str(many), "--json"], out=out, clock=lambda: next(ticks))
        finally:
            cli._Heartbeat = monkey
        assert errors.getvalue(), "the heartbeat did not fire, so this proves nothing"
        report = json.loads(out.getvalue())
        assert len(report) == 12
        assert "checked, now at" not in out.getvalue()

    def test_it_stays_quiet_on_a_short_sweep(self, many):
        """A heartbeat on every file is not a heartbeat, it is noise, and noise is what gets
        redirected to /dev/null along with the thing somebody needed to read.
        """
        errors = io.StringIO()
        out = io.StringIO()
        monkey = cli._Heartbeat

        def heartbeat(total, **kwargs):
            kwargs["stream"] = errors
            return monkey(total, **kwargs)

        cli._Heartbeat = heartbeat
        try:
            cli.main([str(many), "--json"], out=out, clock=lambda: 0.0)
        finally:
            cli._Heartbeat = monkey
        assert errors.getvalue() == "", errors.getvalue()

    def test_the_interval_is_inside_the_range_the_baseline_asks_for(self):
        assert 30 <= cli.HEARTBEAT_SECONDS <= 60, cli.HEARTBEAT_SECONDS

    def test_stderr_is_the_default_and_not_only_what_these_tests_pass_in(self, capsys):
        """The tests above hand it a stream of their own so they can read it, which means they
        would pass over an implementation whose default was stdout. So the default is asserted
        directly: it is the only part of this a consumer's CI depends on.

        ASSERTED ON WHERE THE TEXT LANDS rather than on which object the heartbeat is holding.
        Comparing against `sys.stderr` by identity passes alone and fails inside a larger run,
        because pytest swaps the stream per test and the object captured at construction is no
        longer the one the assertion reads. A test that depends on how it was invoked is not a
        measurement of anything.
        """
        beat = cli._Heartbeat(1, clock=iter([0, 60]).__next__, every=30)
        assert beat.tick("somewhere") is True
        captured = capsys.readouterr()
        assert "checked, now at somewhere" in captured.err, captured
        assert "checked, now at" not in captured.out, captured.out

    def test_a_redirect_after_construction_still_reaches_the_new_stream(self):
        """FOUND BY A SURVIVING MUTANT, 2026-10-02, and the mutant's lesson was about the comment.

        `_Heartbeat.__init__` holds `stream` as given and resolves `sys.stderr` at write time
        instead of at construction. Reverting that to the pre-fix
        `stream if stream is not None else sys.stderr` left this whole file green, so the fix was
        unguarded, and the reason recorded beside it said construction-time resolution breaks
        under "every test harness". That reason is false, and its falseness is why nothing noticed:
        pytest installs its capture BEFORE a test body runs, so an object built inside the body
        captures the already-replaced stream and the text lands where the assertion looks.

        So the sequence has to be built on purpose. Construct first, redirect second, write third.
        Only write-time resolution sends the line to the stream that is current at the write;
        construction-time resolution sends it to the one that was current at the construction.
        Checked both ways round, because an assertion that the line arrived says nothing unless
        the stream it was NOT supposed to reach is also examined.
        """
        at_construction, after_redirect = io.StringIO(), io.StringIO()
        real = sys.stderr
        sys.stderr = at_construction
        try:
            beat = cli._Heartbeat(1, clock=iter([0, 60]).__next__, every=30)
            sys.stderr = after_redirect
            assert beat.tick("somewhere") is True
        finally:
            sys.stderr = real

        assert "checked, now at somewhere" in after_redirect.getvalue(), (
            "the heartbeat wrote to the stream that was current when it was built, not the one "
            "current when it wrote, so `sys.stderr` is being resolved at construction again")
        assert at_construction.getvalue() == "", (
            f"the heartbeat also wrote to the stream it was constructed under: "
            f"{at_construction.getvalue()!r}")

    def test_it_counts_towards_a_total_a_reader_can_use(self, tmp_path):
        """"still working" without a denominator does not tell anybody whether to wait."""
        errors = io.StringIO()
        beat = cli._Heartbeat(3, stream=errors, clock=iter([0, 60, 120, 180]).__next__, every=30)
        assert beat.tick("a") is True
        assert "1 of 3 checked" in errors.getvalue(), errors.getvalue()
