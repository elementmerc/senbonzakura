# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One of our own instruments, driven at arm's length, which is the half that would otherwise rot.

WHY THIS FILE IS NOT OPTIONAL

Row 6.19's done-when line asks for at least one DEFAULT plugin running out of process in CI, and
says in so many words that it is the part most likely to be dropped under time pressure. The
argument is that a path only other people take is a path nobody tests: if every instrument of ours
is in-process and only a stranger's plugin crosses a process boundary, the boundary breaks for
strangers first and we find out from them.

`gate` IS THE INSTRUMENT DRIVEN HERE, and the choice is deliberate rather than convenient. It
reads two JSON files and does arithmetic, with no torch, no model, no corpus and no network, which
its own module docstring pins for exactly this kind of reason: a gate's value is that it runs on
every change, so it has to be affordable on whatever a CI runner has. An out-of-process test over
`score` would need a card and would therefore be skipped in CI, which is the same as not having
one.

THE TEST THAT MATTERS MOST IS THE ONE COMPARING THE TWO TRANSPORTS. Everything else here is
boundary checking. The claim row 6.19 makes is that there is one contract and the transport is a
trust decision, and the only way that is a property of the code rather than a sentence in a plan is
if the same request through both paths produces the same result.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from senbonzakura import baseline as b
from senbonzakura import pluginhost, plugins

COMMON = {
    "model": "Qwen/Qwen3-1.7B",
    "metric": "refusal_rate",
    "direction": b.LOWER_IS_BETTER,
    "input_digest": "sha256:aaa",
    "partition": "measure",
    "prompt_format": "renderer:v3",
    "tool_version": "0.4.0",
    "estimator": "senbonzakura-ruler",
    "precision": "bfloat16",
    "seeds": [42],
    "n": 200,
}


@pytest.fixture
def measurements(tmp_path):
    """A baseline, a steady measurement and a regressed one, as `gate` reads them."""
    made = {}
    for name, over in (("base", {"point": 0.10, "interval": (0.05, 0.18)}),
                       ("steady", {"point": 0.12, "interval": (0.07, 0.20)}),
                       ("worse", {"point": 0.40, "interval": (0.33, 0.47)})):
        path = tmp_path / f"{name}.json"
        b.write(path, b.record(**{**COMMON, **over}))
        made[name] = path
    return made


def _inputs(measurements, which, out=None):
    argv = ["--baseline", str(measurements["base"]),
            "--measurement", str(measurements[which]), "--quiet"]
    if out is not None:
        argv += ["--store", str(out.parent)]
    return {"argv": argv}


# ── the same instrument, both transports, same answer ───────────────────────────────────────────


def test_a_steady_measurement_passes_the_gate_in_process(measurements):
    outcome = pluginhost.in_process("gate").run(
        plugins.call_for("gate", _inputs(measurements, "steady")))
    assert outcome.transport == "in-process"
    assert outcome.result == {"status": 0, "document": None}


def test_a_steady_measurement_passes_the_gate_out_of_process(measurements):
    outcome = pluginhost.out_of_process("gate").run(
        plugins.call_for("gate", _inputs(measurements, "steady"), deadline=120))
    assert outcome.transport == "subprocess"
    assert outcome.result == {"status": 0, "document": None}


@pytest.mark.parametrize("which", ["steady", "worse"])
def test_the_two_transports_agree_about_the_same_instrument_and_the_same_request(
        measurements, which):
    """The whole claim of row 6.19, as a measurement rather than as a sentence."""
    request = _inputs(measurements, which)
    here = pluginhost.in_process("gate").run(plugins.call_for("gate", request))
    there = pluginhost.out_of_process("gate").run(
        plugins.call_for("gate", request, deadline=120))
    assert here.result == there.result
    assert here.transport != there.transport


def test_a_regression_comes_back_as_a_non_zero_status_rather_than_as_a_failure(measurements):
    """A gate that says no has worked, so the reply is a success carrying a status of 1.

    The distinction is the one `measure.py` already draws and the one this host must not blur: an
    instrument answering "this regressed" has produced its number, and a reply marked ok=False
    would make a working gate look like a broken plugin.
    """
    outcome = pluginhost.in_process("gate").run(
        plugins.call_for("gate", _inputs(measurements, "worse")))
    assert outcome.result["status"] == 1


# ── the document an instrument writes comes back with its result ────────────────────────────────


#: A measurement with a stamped `metrics` block, which is what `baseline` reads. The same shape
#: `tests/test_baseline.py` uses, kept here rather than imported from it: a test file is not an
#: interface, and importing one into another makes a rename in either break both.
STAMPED = {
    "metric": "coherence", "measures": "whether the edit left the model able to predict English",
    "estimator": "neutral-passage-nll", "estimator_description": "the mean per-token NLL",
    "units": "nats-per-token", "higher_is_better": False, "n": 1, "value": 3.0,
    "n_tokens": 268, "input_digest": "0123456789abcdef", "prompt_format": "raw",
    "partition": "fixed-passage", "precision": "bfloat16", "tool_version": "0.4.0",
    "interval": [2.9, 3.1], "seeds": "no-seed",
}


def test_the_file_an_instrument_wrote_is_read_back_into_the_reply(tmp_path):
    """`baseline` writes a document, so it is the instrument that exercises the read-back path."""
    source = tmp_path / "coherence.json"
    source.write_text(json.dumps(
        {"label": "arm", "model": "Qwen/Qwen3-1.7B", "metrics": {"coherence": STAMPED}}),
        encoding="utf-8")
    out = tmp_path / "kept.json"
    inputs = {"argv": ["--measurement", str(source), "--seeds", "no-seed", "--out", str(out)],
              "out": str(out)}
    outcome = pluginhost.in_process("baseline").run(plugins.call_for("baseline", inputs))
    assert outcome.result["status"] == 0
    assert outcome.result["document"]["metric"] == "coherence"


def test_an_absent_document_is_reported_as_absent_rather_than_as_an_empty_result(
        tmp_path, measurements):
    inputs = _inputs(measurements, "steady")
    inputs["out"] = str(tmp_path / "never-written.json")
    outcome = pluginhost.in_process("gate").run(plugins.call_for("gate", inputs))
    assert outcome.result["document"] is None


def test_a_document_that_is_there_and_is_not_json_is_a_fault_rather_than_an_absence(
        tmp_path, measurements):
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    inputs = _inputs(measurements, "steady")
    inputs["out"] = str(broken)
    with pytest.raises(plugins.PluginFailedError, match="could not be read as JSON"):
        pluginhost.in_process("gate").run(plugins.call_for("gate", inputs))


def test_a_document_above_the_cap_is_refused_rather_than_read(tmp_path, measurements, monkeypatch):
    big = tmp_path / "big.json"
    big.write_text("[]" + " " * 64, encoding="utf-8")
    monkeypatch.setattr(pluginhost, "MAX_DOCUMENT_BYTES", 8)
    inputs = _inputs(measurements, "steady")
    inputs["out"] = str(big)
    with pytest.raises(plugins.PluginFailedError, match="above the 8"):
        pluginhost.in_process("gate").run(plugins.call_for("gate", inputs))


# ── the boundary ────────────────────────────────────────────────────────────────────────────────


def test_only_an_instrument_this_package_registers_can_be_served():
    with pytest.raises(pluginhost.HostError, match="not an instrument this package registers"):
        pluginhost.provider_for("os")


def test_every_servable_instrument_is_a_command_this_package_dispatches():
    from senbonzakura.entry import DELEGATED

    assert set(pluginhost.served()) == set(DELEGATED)


def test_every_title_names_an_instrument_that_exists():
    """A line for a command that has been renamed would describe a plugin nobody can reach."""
    unknown = sorted(set(pluginhost.TITLES) - set(pluginhost.served()))
    assert not unknown, f"TITLES names {unknown}, which this package does not dispatch"


def test_an_instrument_describes_itself_without_importing_its_module():
    """A describe call must stay affordable, which is why it reads no instrument's metrics."""
    manifest = pluginhost.in_process("gate").describe()
    assert manifest.name == "gate"
    assert manifest.kind == "instrument"
    assert manifest.produces == ()
    assert "interval" in manifest.title


def test_an_instrument_without_a_title_line_still_describes_itself_plainly(monkeypatch):
    monkeypatch.setattr(pluginhost, "TITLES", {})
    assert pluginhost.in_process("gate").describe().title == "senbonzakura gate"


@pytest.mark.parametrize(("argv", "fragment"), [
    ("--baseline x", "argv is a string"),
    (7, "cannot be read as a list"),
    ([1, 2], "argv[0] is int"),
])
def test_an_argv_that_is_not_a_list_of_strings_is_refused(argv, fragment):
    with pytest.raises(plugins.PluginFailedError, match=fragment.replace("[", "\\[")):
        pluginhost.in_process("gate").run(plugins.call_for("gate", {"argv": argv}))


def test_an_argv_above_the_count_cap_is_refused(monkeypatch):
    monkeypatch.setattr(pluginhost, "MAX_ARGV", 2)
    with pytest.raises(plugins.PluginFailedError, match="above the 2 this reads"):
        pluginhost.in_process("gate").run(
            plugins.call_for("gate", {"argv": ["a", "b", "c"]}))


def test_a_single_argument_above_the_length_cap_is_refused(monkeypatch):
    monkeypatch.setattr(pluginhost, "MAX_ARG", 4)
    with pytest.raises(plugins.PluginFailedError, match="above the 4 this reads"):
        pluginhost.in_process("gate").run(
            plugins.call_for("gate", {"argv": ["a-very-long-argument"]}))


def test_an_out_that_is_not_a_path_is_refused():
    with pytest.raises(plugins.PluginFailedError, match="out is int rather than a path"):
        pluginhost.in_process("gate").run(
            plugins.call_for("gate", {"argv": [], "out": 7}))


def test_inputs_that_are_not_a_mapping_are_refused_by_the_provider_itself():
    """`call_for` already refuses this, so the provider is driven directly to reach its own guard."""
    reply = pluginhost.provider_for("gate").run(["--baseline", "x"])
    assert reply["ok"] is False
    assert "rather than a mapping" in reply["reason"]


def test_an_instrument_refusing_its_input_comes_back_as_its_own_exit_status():
    """A refusal from argparse is `SystemExit(2)`, and 2 is what the reply carries.

    NOT a plugin failure. The instrument ran and answered, and its answer is "I will not compare
    these", which is a status a build system reads. Marking it ok=False would file a working
    refusal as a broken plugin, which is the distinction this host exists to keep.
    """
    outcome = pluginhost.in_process("gate").run(
        plugins.call_for("gate", {"argv": ["--nonsense"]}))
    assert outcome.result["status"] == 2


def test_an_instrument_that_exits_with_a_message_rather_than_a_status_is_a_failure(monkeypatch):
    """`SystemExit("...")` is a sentence, not a status, and reporting it as 0 would be a lie."""
    class _Chatty:
        @staticmethod
        def main(_argv):
            raise SystemExit("the track manifest is missing")

    monkeypatch.setattr(pluginhost.importlib, "import_module", lambda *_a, **_k: _Chatty)
    with pytest.raises(plugins.PluginFailedError, match="track manifest is missing"):
        pluginhost.in_process("gate").run(plugins.call_for("gate", {"argv": []}))


def test_an_instrument_returning_a_status_rather_than_none_keeps_it(monkeypatch):
    class _Returns:
        @staticmethod
        def main(_argv):
            return 7

    monkeypatch.setattr(pluginhost.importlib, "import_module", lambda *_a, **_k: _Returns)
    outcome = pluginhost.in_process("gate").run(plugins.call_for("gate", {"argv": []}))
    assert outcome.result["status"] == 7


def test_an_instruments_own_log_cannot_reach_the_reply_on_stdout(capsys, measurements):
    """The defect the first out-of-process test found, pinned where it was fixed.

    `gate` prints its verdict for a person. Out of process stdout carries the JSON reply, so an
    unredirected print lands in front of it and the caller refuses the exchange as not JSON.
    """
    pluginhost.in_process("gate").run(plugins.call_for("gate", _inputs(measurements, "worse")))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "REGRESSED" in captured.err


# ── the host as a command ───────────────────────────────────────────────────────────────────────


def test_the_host_refuses_to_start_without_exactly_one_instrument_named(capsys):
    assert pluginhost.main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_the_host_refuses_an_unknown_instrument_before_reading_a_request(capsys):
    assert pluginhost.main(["nonesuch"]) == 2
    assert "not an instrument this package registers" in capsys.readouterr().err


def test_the_host_refuses_a_name_that_could_not_be_logged_safely(capsys):
    assert pluginhost.main(["../../etc/passwd"]) == 2
    assert "refused here rather than stripped" in capsys.readouterr().err


def test_the_host_answers_a_describe_request_on_stdin():
    """Driven as a real process, because that is how a consumer meets it."""
    proc = subprocess.run(
        [sys.executable, "-m", "senbonzakura.pluginhost", "gate"],
        input=json.dumps({"protocol": plugins.PROTOCOL, "call": "describe"}),
        capture_output=True, text=True, timeout=180, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["name"] == "gate"
