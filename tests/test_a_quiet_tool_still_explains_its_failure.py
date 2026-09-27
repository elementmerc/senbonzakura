# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Hiding a tool's output is only an improvement while a failure still explains itself.

WHAT PROMPTED IT, 2026-09-27

A surface audit found roughly 350 lines of vendored-converter chatter arriving ahead of this tool's
own five-line summary, so both `convert` and `quantise` now summarise it and `--verbose` restores it.

That trade has two ways of going wrong and they are what this file is about. A long job that prints
nothing is indistinguishable from a hung one, which is why a quiet run reports that it is still
working. And a failure whose diagnostic was swallowed is worse than the noise was, which is why the
tool's last lines are kept whatever the verbosity and quoted in the refusal.

Both were foreseen rather than found: an existing test asserted that every line of the quantiser's
output reaches the terminal, on the stated grounds that per-tensor progress is a long run's only
sign of life. It was right, and this file is the other half of the answer to it.
"""
from __future__ import annotations

import re

import pytest

from senbonzakura import vendored


def _stream(*lines):
    return iter([f"{line}\n" for line in lines])


class TestTheTailIsKeptWhateverTheVerbosity:
    def test_the_last_lines_come_back_when_the_output_was_suppressed(self, capsys):
        _matches, tail = vendored.relay(_stream("one", "two", "three"), verbose=False,
                                        log=lambda _m: None)
        assert tail == ["one", "two", "three"]
        assert capsys.readouterr().out == "", "a quiet run still wrote to the terminal"

    def test_the_last_lines_come_back_when_it_was_not(self, capsys):
        _matches, tail = vendored.relay(_stream("one", "two"), verbose=True, log=lambda _m: None)
        assert tail == ["one", "two"]
        assert "one" in capsys.readouterr().out, "a verbose run swallowed the output"

    def test_only_the_last_lines_are_kept_so_a_long_run_cannot_exhaust_memory(self):
        """A bound, because the whole point is that these tools are talkative."""
        _matches, tail = vendored.relay(_stream(*[str(i) for i in range(500)]),
                                        verbose=False, log=lambda _m: None, keep=5)
        assert tail == ["495", "496", "497", "498", "499"]

    def test_the_default_bound_is_enough_to_diagnose_something(self):
        assert vendored.KEEP_LINES >= 10, (
            "too few lines to be worth quoting in a refusal, which is what they are for")


class TestAQuietRunSaysItIsAlive:
    def test_a_long_run_reports_progress_against_an_injected_clock(self):
        """Injected rather than waited on: a test that sleeps for the interval is one nobody runs."""
        said = []
        clock = iter([0, 31, 62, 93]).__next__
        vendored.relay(_stream("[1/3] a", "[2/3] b", "[3/3] c"), verbose=False,
                       log=said.append, now=clock)
        assert said, "a long quiet run said nothing, which reads as a hang"
        assert all(re.search(r"still working, \d+ lines in", m) for m in said), said

    def test_a_short_run_says_nothing_at_all(self):
        """Below the interval there is nothing to reassure anybody about."""
        said = []
        clock = iter([0, 1, 2, 3]).__next__
        vendored.relay(_stream("a", "b", "c"), verbose=False, log=said.append, now=clock)
        assert said == [], f"a run that took no time announced itself: {said}"

    def test_the_heartbeat_quotes_the_tool_and_marks_it_if_it_had_to_cut_it(self):
        """A per-tensor line can be long, and a cut one must say so, per `say.shorten`."""
        said = []
        clock = iter([0, 99, 200]).__next__
        long_line = "[137/272] " + "blk.layers.self_attn.o_proj.weight" * 4
        vendored.relay(_stream(long_line, "next"), verbose=False, log=said.append, now=clock)
        assert said
        assert said[0].endswith("..."), f"a cut line reads as complete: {said[0]!r}"

    def test_the_heartbeat_never_sends_a_marker_shaped_line(self):
        """`say` passes a machine marker through verbatim, so a heartbeat must not look like one."""
        from senbonzakura import say

        said = []
        clock = iter([0, 99, 200]).__next__
        vendored.relay(_stream("MARGIN_DONE label auc=0.9", "x"), verbose=False,
                       log=said.append, now=clock)
        assert said
        assert not say.is_marker(said[0]), f"the heartbeat reads as a marker: {said[0]!r}"


class TestWatchedPatternsSurviveSuppression:
    """The fallback warning is the line that once contradicted this tool's own verdict."""

    def test_a_watched_pattern_is_found_in_a_quiet_run(self):
        pattern = re.compile(r"WARNING:\s*(\d+)\s+of\s+(\d+)")
        matches, _tail = vendored.relay(
            _stream("[1/272] x", "WARNING: 180 of 272 tensor(s) required fallback", "done"),
            verbose=False, log=lambda _m: None, watch=(pattern,))
        assert len(matches) == 1
        _pat, found = matches[0]
        assert (found.group(1), found.group(2)) == ("180", "272")

    def test_nothing_is_reported_when_nothing_matched(self):
        """None and an empty list are different claims; the caller turns one into "we did not see"."""
        matches, _tail = vendored.relay(_stream("all good"), verbose=False,
                                        log=lambda _m: None, watch=(re.compile("never"),))
        assert matches == []


class TestTheRefusalQuotesWhatTheToolSaid:
    """END TO END through `convert`, because the point is what a person reads when it breaks."""

    def test_a_failed_conversion_quotes_the_converters_last_words(self, tmp_path, monkeypatch):
        from senbonzakura import convert

        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **_k: ({"Qwen3ForCausalLM"}, [], None))

        class _Failed:
            returncode = 1
            stdout = _stream("loading model", "ERROR: unsupported tensor layout blk.0.attn")

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        monkeypatch.setattr(convert.subprocess, "Popen", lambda *_a, **_k: _Failed())
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}',
                                           encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"\0" * 64)

        with pytest.raises(SystemExit) as e:
            convert.run([str(model), str(tmp_path / "o.gguf")], log=lambda _m: None)
        said = str(e.value)
        assert "unsupported tensor layout" in said, (
            f"the converter's own diagnostic was swallowed by the summarising: {said}")
        assert "--verbose" in said, "nothing tells the reader how to see the rest"


class TestAWarningIsNeverSummarisedAway:
    """The half of the trade that the recorded decision got wrong about mechanism.

    The decision was "suppress by default, stderr always through", and that rests on a false premise:
    `convert_hf_to_gguf.py` calls `logging.basicConfig` with no stream, so its 350 lines ARE stderr,
    and llama-quantize's fallback warning is on stderr too and has to be scanned. Passing the stream
    through untouched would have left the defect in place and made the warning unreadable at the same
    time. So what reaches the reader is decided by what a line SAYS.
    """

    def test_a_warning_reaches_the_reader_in_a_quiet_run(self, capsys):
        vendored.relay(_stream("[1/272] blk.0", "WARNING: 180 of 272 required fallback",
                               "[2/272] blk.1"),
                       verbose=False, log=lambda _m: None)
        said = capsys.readouterr()
        assert "180 of 272" in said.err, "the warning was summarised away"
        assert "blk.0" not in said.err + said.out, "the chatter came through as well"

    @pytest.mark.parametrize("line", [
        "ERROR: unsupported tensor layout",
        "warning: tied embeddings ignored",
        "conversion failed at blk.3",
        "cannot open file",
        "llama_model_quantize_impl: WARNING: 4 of 272 tensor(s) required fallback",
    ])
    def test_every_shape_of_bad_news_gets_through(self, line, capsys):
        vendored.relay(_stream("[1/2] fine", line), verbose=False, log=lambda _m: None)
        assert line in capsys.readouterr().err, f"this never reached the reader: {line!r}"

    def test_an_ordinary_line_does_not(self):
        """Otherwise the exemption swallows the summarising it sits inside."""
        assert not any(p.search("[137/272] blk.4.attn_k.weight - converting to q4_K")
                       for p in vendored.LOUD_LINES)

    def test_it_goes_to_stderr_rather_than_stdout(self, capsys):
        """Where it came from, so a pipeline that separates the streams keeps getting it there."""
        vendored.relay(_stream("ERROR: boom"), verbose=False, log=lambda _m: None)
        said = capsys.readouterr()
        assert "ERROR: boom" in said.err and "ERROR: boom" not in said.out

    def test_it_is_still_kept_in_the_tail(self, capsys):
        """Passing a line through must not stop it counting toward the failure report."""
        _matches, tail = vendored.relay(_stream("ERROR: boom"), verbose=False, log=lambda _m: None)
        capsys.readouterr()
        assert tail == ["ERROR: boom"]

    def test_a_verbose_run_prints_it_once_and_not_twice(self, capsys):
        """The loud branch must not double up with the pass-everything branch."""
        vendored.relay(_stream("ERROR: boom"), verbose=True, log=lambda _m: None)
        said = capsys.readouterr()
        assert (said.out + said.err).count("ERROR: boom") == 1
